from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional
from collections import Counter

import pandas as pd
import pymorphy2
import re
import spacy


# CONFIG

@dataclass
class AspectAnalysisConfig:
    input_sentence_file: str = (
        "topic_sentiment_2/topic_sentiment_results/"
        "sentence_topics_with_sentiment.csv"
    )

    base_output_dir: str = "aspect_analysis_3"
    results_subdir: str = "aspect_analysis_results"
    logs_subdir: str = "aspect_analysis_logs"

    log_filename: str = "aspect_analysis.log"

    extracted_filename: str = "aspects_extracted.csv"
    filtered_filename: str = "aspects_filtered.csv"
    summary_filename: str = "aspect_summary.csv"

    min_aspect_length: int = 3
    min_aspect_count: int = 10

    top_examples: int = 3


# Логирование

def setup_aspect_logging(config: AspectAnalysisConfig) -> logging.Logger:
    log_dir = os.path.join(config.base_output_dir, config.logs_subdir)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, config.log_filename)

    logger = logging.getLogger("aspect_analysis")
    logger.setLevel(logging.INFO)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%d.%m.%Y %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# Извлечение аспектов

class AspectExtractor:
    """
    Извлекает аспект-кандидаты из предложений.
    Используется подход:
    - токенизация
    - pymorphy2
    - существительные
    """

    def __init__(self):
    # Инициализация морфологического анализатора (pymorphy2)
    self.morph = pymorphy2.MorphAnalyzer()

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        # Список для хранения извлечённых аспектов
        rows = []

        # Проходим по строкам DataFrame
        for _, row in df.iterrows # _ — это имя для переменной, которая нам не нужна.
            # Берём текст предложения
            text = str(row["text"])

            # Токенизация: извлекаем слова на кириллице
            tokens = re.findall(r"[а-яА-ЯёЁ]+", text)

            for t in tokens:
                # Морфологический разбор токена
                parsed = self.morph.parse(t)[0]

                # Отбираем только существительные для кандидатов в аспекты
                if parsed.tag.POS == "NOUN":
                    rows.append({
                        "doc_id": row["doc_id"],
                        "sentence_id": row["sentence_id"],
                        "topic": row["topic"],
                        "aspect_raw": t.lower(),
                        "sentiment_label": row["sentiment_label"],
                        "sentiment_score": row["sentiment_score"],
                        "text": text
                    })

        # Возвращаем результат в виде DataFrame
        return pd.DataFrame(rows)


# Стоп-списки для аспектов

ABSTRACT_NOUNS = {
    "вещь", "штука", "момент", "случай", "раз", "день", "время",
    "человек", "люди", "народ", "клиент", "пользователь",
    "всё", "это", "ничто"
}

META_ENTITIES = {
    "оператор", "компания", "фирма", "салон", "офис",
    "приложение", "страница", "отзыв",
    "комментарий", "сообщение"
}

DISCOURSE_NOUNS = {
    "проблема", "ситуация", "история", "вопрос",
    "опыт", "ощущение", "мнение", "впечатление"
}

COMPANY_NAMES = {
    "билайн", "ростелеком", "мтс", "мегафон", "теле2"
}


# Фильтрация аспектов

class AspectFilter:
    """
    Фильтрация шумных аспектов:
    - слишком короткие
    - редкие
    - стоп-аспекты
    """

    STOP_ASPECTS = (
        ABSTRACT_NOUNS # Объединение
        | META_ENTITIES
        | DISCOURSE_NOUNS
        | COMPANY_NAMES
    )

    def __init__(self, min_length: int = 3, min_count: int = 10):
        self.min_length = min_length
        self.min_count = min_count
        self.morph = pymorphy2.MorphAnalyzer()

    def normalize(self, word: str) -> str:
        return self.morph.parse(word)[0].normal_form

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # нормализация
        df["aspect_norm"] = df["aspect_raw"].apply(self.normalize)

        # длина
        df = df[df["aspect_norm"].str.len() >= self.min_length]

        # стоп-аспекты
        df = df[~df["aspect_norm"].isin(self.STOP_ASPECTS)]

        # частотность
        counts = Counter(df["aspect_norm"])
        df["aspect_count"] = df["aspect_norm"].map(counts)

        df = df[df["aspect_count"] >= self.min_count]

        return df


# Aspect statistics (operator overloading)

@dataclass(frozen=True) # frozen=True делает объект неизменяемым (т.к. статистика накапливается через сложение → нет риска случайно изменить объект)
class AspectStats:
    # Общее число упоминаний аспекта
    mentions: int = 0

    # Количество позитивных упоминаний
    pos_mentions: int = 0

    # Количество негативных упоминаний
    neg_mentions: int = 0

    # Суммарная сила позитивного сентимента
    pos_strength: float = 0.0

    # Суммарная сила негативного сентимента
    neg_strength: float = 0.0

    def __add__(self, other: "AspectStats") -> "AspectStats":
        # Перегрузка оператора: позволяет суммировать статистику двух аспектов
        if not isinstance(other, AspectStats):
            return NotImplemented

        return AspectStats(
            mentions=self.mentions + other.mentions,
            pos_mentions=self.pos_mentions + other.pos_mentions,
            neg_mentions=self.neg_mentions + other.neg_mentions,
            pos_strength=self.pos_strength + other.pos_strength,
            neg_strength=self.neg_strength + other.neg_strength,
        )

    @property
    def net_sentiment(self) -> float:
        # Итоговый сентимент: разница между позитивом и негативом
        return self.pos_strength - self.neg_strength


# Агрегация аспектов

class AspectAggregator:
    """
    Агрегация аспектов:
    считает статистику и формирует
    интерпретируемый итоговый результат
    """

    def __init__(self, top_examples: int = 3):
        self.top_examples = top_examples

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        grouped = df.groupby(["topic", "aspect_norm"])

        for (topic, aspect), g in grouped:
            stats = AspectStats()

            for _, row in g.iterrows():
                if row["sentiment_label"] == "positive":
                    part = AspectStats(
                        mentions=1,
                        pos_mentions=1,
                        pos_strength=row["sentiment_score"]
                    )
                elif row["sentiment_label"] == "negative":
                    part = AspectStats(
                        mentions=1,
                        neg_mentions=1,
                        neg_strength=row["sentiment_score"]
                    )
                else:
                    part = AspectStats(mentions=1)

                stats = stats + part

            avg_score = g["sentiment_score"].mean()

            rows.append({
                "topic": topic,
                "aspect": aspect,
                "mentions": stats.mentions,
                "pos_mentions": stats.pos_mentions,
                "neg_mentions": stats.neg_mentions,
                "pos_strength": round(stats.pos_strength, 3),
                "neg_strength": round(stats.neg_strength, 3),
                "net_sentiment": round(stats.net_sentiment, 3),
                "avg_score": round(avg_score, 3),
            })

        return pd.DataFrame(rows)

# spaCy refinement

class SpacyAspectRefiner:
    """
    Уточнение негативных аспектов
    с помощью spaCy (на основе синтаксического анализа)
    """

    def __init__(self):
        # Загрузка русской модели spaCy
        self.nlp = spacy.load("ru_core_news_sm")

    def refine(self, df: pd.DataFrame) -> pd.DataFrame:
        # Список для хранения уточнённых аспектов
        rows = []

        # Проходим по строкам DataFrame
        for _, row in df.iterrows():
            # Синтаксический разбор текста предложения
            doc = self.nlp(row["text"])
            candidates = []

            for token in doc:
                # Шаблон 1: прилагательное + существительное
                if token.pos_ == "NOUN":
                    modifiers = [
                        child.text
                        for child in token.children
                        if child.dep_ in {"amod"} and child.pos_ == "ADJ"
                    ]
                    if modifiers:
                        candidates.append(" ".join(modifiers + [token.lemma_]))
                    else:
                        # Если модификаторов нет — берём только существительное
                        candidates.append(token.lemma_)

                # Шаблон 2: NP (родительный падеж)
                if token.dep_ == "nmod" and token.head.pos_ == "NOUN":
                    candidates.append(
                        f"{token.head.lemma_} {token.lemma_}"
                    )

            # Выбираем уточнённый аспект, который содержит исходный нормализованный аспект
            refined = None
            for c in candidates:
                if row["aspect_norm"] in c:
                    refined = c
                    break

            # Сохраняем результат (если не нашли уточнение — оставляем исходный аспект)
            rows.append({
                "topic": row["topic"],
                "aspect_original": row["aspect_norm"],
                "aspect_refined": refined or row["aspect_norm"],
                "sentiment_score": row["sentiment_score"],
                "text": row["text"],
            })

        # Возвращаем результат в виде DataFrame
        return pd.DataFrame(rows)


# Топ-причины негатива

class NegativeReasonAggregator:
    """
    Формирует ТОП-10 причин негатива по темам
    на основе spaCy-refined аспектов
    """

    def __init__(self, top_n: int = 5, top_examples: int = 3):
        self.top_n = top_n
        self.top_examples = top_examples

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        grouped = df.groupby(["topic", "aspect_refined"])

        for (topic, reason), g in grouped:
            mentions = len(g)
            neg_strength = g["sentiment_score"].sum()

            examples = (
                g.sort_values("sentiment_score", ascending=False)
                .head(self.top_examples)["text"]
                .tolist()
            )

            rows.append({
                "topic": topic,
                "reason": reason,
                "mentions": mentions,
                "neg_strength": round(neg_strength, 3),
                "examples": " | ".join(examples),
            })

        df_out = pd.DataFrame(rows)

        # ТОП-N причин внутри каждой темы
        df_out = (
            df_out
            .sort_values(["topic", "neg_strength"], ascending=[True, False])
            .groupby("topic")
            .head(self.top_n)
            .reset_index(drop=True)
        )

        return df_out



# Главный пайплайн

class AspectAnalysisPipeline:

    def __init__(self, config: Optional[AspectAnalysisConfig] = None):
        self.config = config or AspectAnalysisConfig()

        self.base_dir = self.config.base_output_dir
        self.results_dir = os.path.join(self.base_dir, self.config.results_subdir)
        self.logs_dir = os.path.join(self.base_dir, self.config.logs_subdir)

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger = setup_aspect_logging(self.config)

        self.logger.info("→ СТАРТ ШАГА 3: аспектный анализ")

        self.logger.info(
            f"Загрузка sentence-level данных: {self.config.input_sentence_file}"
        )
        df = pd.read_csv(self.config.input_sentence_file)

        # ШАГ 3.1
       
        self.logger.info("ШАГ 3.1: извлечение аспектов")
        extractor = AspectExtractor()
        df_aspects = extractor.extract(df)

        out_1 = os.path.join(
            self.results_dir,
            self.config.extracted_filename
        )
        df_aspects.to_csv(out_1, index=False, encoding="utf-8-sig")
        self.logger.info(f"Аспекты извлечены и сохранены: {out_1}")

        # ШАГ 3.2
       
        self.logger.info("ШАГ 3.2: очистка и фильтрация аспектов")
        aspect_filter = AspectFilter(
            min_length=self.config.min_aspect_length,
            min_count=self.config.min_aspect_count
        )
        df_filtered = aspect_filter.filter(df_aspects)

        out_2 = os.path.join(
            self.results_dir,
            self.config.filtered_filename
        )
        df_filtered.to_csv(out_2, index=False, encoding="utf-8-sig")
        self.logger.info(f"Отфильтрованные аспекты сохранены в файл: {out_2}")

        
        # ШАГ 3.3
       
        self.logger.info("ШАГ 3.3: агрегация аспектов и подсчет статистики")

        aggregator = AspectAggregator(
            top_examples=self.config.top_examples
        )

        df_summary = aggregator.aggregate(df_filtered)

        out_3 = os.path.join(
            self.results_dir,
            self.config.summary_filename
        )
        df_summary.to_csv(out_3, index=False, encoding="utf-8-sig")

        self.logger.info(
            f"Агрегированный аспектный анализ сохранён: {out_3}"
        )

        # ШАГ 3.4 

        self.logger.info(
            "ШАГ 3.4: отбор значимых негативных аспектов для углублённого анализа"
        )

        df_neg = df_filtered[
            df_filtered["sentiment_label"] == "negative"
        ].copy()

        self.logger.info(
            f"Найдено негативных упоминаний: {len(df_neg)}"
        )

        df_neg = df_neg.merge(
            df_summary[["aspect", "neg_strength"]],
            left_on="aspect_norm",
            right_on="aspect",
            how="left"
        )

        df_neg = df_neg.drop(columns=["aspect"])

        df_neg = df_neg.sort_values(
            "neg_strength", ascending=False
        ).head(500)

        self.logger.info(
            f"Отобрано {len(df_neg)} предложений "
            "с наибольшим вкладом в негатив"
        )

        # ШАГ 3.5     

        self.logger.info(
            "ШАГ 3.5: уточнение негативных аспектов с помощью spaCy"
        )

        refiner = SpacyAspectRefiner()
        df_refined = refiner.refine(df_neg)

        out_4 = os.path.join(
            self.results_dir,
            "aspect_negative_refined.csv"
        )

        df_refined.to_csv(
            out_4, index=False, encoding="utf-8-sig"
        )

        self.logger.info(
            f"spaCy-refined негативные аспекты сохранены: {out_4}"
        )
        
        # ШАГ 3.6
        
        self.logger.info("ШАГ 3.6: формирование ТОП-причин негатива по темам")

        df_refined = pd.read_csv(
            os.path.join(
                self.results_dir,
                "aspect_negative_refined.csv"
            )
        )

        reason_aggregator = NegativeReasonAggregator(
            top_n=5,
            top_examples=3
        )

        df_top_reasons = reason_aggregator.aggregate(df_refined)

        out_5 = os.path.join(
            self.results_dir,
            "top_negative_reasons_by_topic.csv"
        )

        df_top_reasons.to_csv(
            out_5, index=False, encoding="utf-8-sig"
        )

        self.logger.info(
            f"ТОП-причин негатива по темам сохранён: {out_5}"
        )

        self.logger.info("→ ШАГ 3 ЗАВЕРШЁН")
