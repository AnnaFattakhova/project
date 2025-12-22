from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, List
import re
from typing import Dict
import pandas as pd

from abc import ABC, abstractmethod

# ======================================================
# STEP 2.1 — SENTENCE SPLITTER
# ======================================================

class SentenceSplitter:
    def __init__(self, min_length: int = 2):
        self.min_length = min_length

    def split_to_dicts(self, documents: List[str]) -> List[dict]:
        import re

        results = []
        for doc_id, text in enumerate(documents):
            sentences = re.split(r"[.!?]+", str(text))
            sent_id = 0
            for s in sentences:
                s = s.strip()
                if len(s.split()) >= self.min_length:
                    results.append({
                        "doc_id": doc_id,
                        "sentence_id": sent_id,
                        "text": s
                    })
                    sent_id += 1
        return results


# ======================================================
# STEP 2.2 — SENTIMENT ANALYZER
# ======================================================

class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny-sentiment-balanced",
        device: Optional[str] = None
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.id2label = self.model.config.id2label

    # ← ВАЖНО: метод ВНЕ __init__
    def predict(self, sentences: List[dict], batch_size: int = 32) -> List[dict]:
        import torch
        from torch.nn.functional import softmax

        text_to_indices: Dict[str, List[int]] = {}
        for idx, s in enumerate(sentences):
            text = s["text"]
            text_to_indices.setdefault(text, []).append(idx)

        unique_texts = list(text_to_indices.keys())
        self.model.eval()

        cached_results: Dict[str, Dict] = {}

        with torch.no_grad():
            for i in range(0, len(unique_texts), batch_size):
                batch_texts = unique_texts[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=1)

                for text, prob in zip(batch_texts, probs):
                    label_id = int(torch.argmax(prob))
                    cached_results[text] = {
                        "sentiment_label": self.id2label[label_id],
                        "sentiment_score": float(prob[label_id])
                    }

        results = []
        for text, indices in text_to_indices.items():
            for idx in indices:
                results.append({
                    **sentences[idx],
                    **cached_results[text]
                })

        results.sort(key=lambda x: (x["doc_id"], x["sentence_id"]))
        return results

# ======================================================
# STEP 2.3 — TOPIC ↔ SENTENCE MAPPER
# ======================================================

class TopicSentenceMapper:
    def __init__(self, topic_model: "BerTopicModel"):
        self.topic_model = topic_model

    def map_sentences(self, sentences: List[dict]) -> List[dict]:
        texts = [s["text"] for s in sentences]
        topics, probs = self.topic_model.transform(texts)

        results = []
        for s, t, p in zip(sentences, topics, probs):
            # p может быть numpy.float64 или массивом вероятностей
            topic_prob = None
            if p is not None:
                try:
                    # если p — массив вероятностей
                    topic_prob = float(max(p))  # type: ignore[arg-type]
                except TypeError:
                    # если p — скаляр
                    topic_prob = float(p)

            results.append({
                **s,
                "topic": int(t),
                "topic_prob": topic_prob
            })

        return results

    
    
# ======================================================
# Защита от мата
# ======================================================

class BaseTextDetector(ABC):
    """
    Базовый класс для rule-based детекторов,
    работающих с текстом отзыва.
    """

    @abstractmethod
    def detect(self, text: str) -> Dict:
        """
        Анализирует текст и возвращает словарь с результатами.
        """
        pass


class ProfanityDetector(BaseTextDetector):
    """
    Детектор нецензурной лексики (rule-based).
    Используется для модерации отзывов.
    """

    def __init__(self):
        # базовый список корней (можно расширять)
        self.patterns = [
            r"\bбля[дт]?\b",
            r"\bхуй\w*\b",
            r"\bпизд\w*\b",
            r"\bеба\w*\b",
            r"\bсука\b",
            r"\bмуда\w*\b",
            r"\bговн\w*\b",
        ]

        self.regex = re.compile("|".join(self.patterns), flags=re.IGNORECASE)

    def detect(self, text: str) -> Dict:
        matches = self.regex.findall(text or "")
        profanity_count = len(matches)

        token_count = max(len((text or "").split()), 1)
        ratio = profanity_count / token_count

        if profanity_count == 0:
            action = "keep"
        elif ratio < 0.05:
            action = "flag"
        else:
            action = "hide"

        return {
            "has_profanity": profanity_count > 0,
            "profanity_count": profanity_count,
            "profanity_ratio": round(ratio, 3),
            "moderation_action": action,
        }


class GenderDetector(BaseTextDetector):
    """
    Эвристический детектор гендера только по тексту:
    ищем маркеры рода (купил/купила и т.п.).
    """

    def __init__(self):
        self.male_markers = [
            r"\bя\s+купил\b",
            r"\bя\s+заказал\b",
            r"\bя\s+получил\b",
            r"\bя\s+написал\b",
            r"\bя\s+остался\b",
            r"\bя\s+доволен\b",
            r"\bя\s+недоволен\b",
        ]

        self.female_markers = [
            r"\bя\s+купила\b",
            r"\bя\s+заказала\b",
            r"\bя\s+получила\b",
            r"\bя\s+написала\b",
            r"\bя\s+осталась\b",
            r"\bя\s+довольна\b",
            r"\bя\s+недовольна\b",
        ]

    # требование BaseTextDetector
    def detect(self, text: str) -> Dict:
        return self.predict_from_text(text)

    def predict_from_text(self, text: str) -> Dict:
        if not isinstance(text, str) or not text.strip():
            return {"gender": "unknown", "confidence": 0.0, "evidence": "пустой текст"}

        text_lc = text.lower()

        male_hits = [p for p in self.male_markers if re.search(p, text_lc)]
        female_hits = [p for p in self.female_markers if re.search(p, text_lc)]

        if male_hits and not female_hits:
            return {
                "gender": "male",
                "confidence": min(1.0, 0.6 + 0.1 * len(male_hits)),
                "evidence": "по тексту: мужские формы",
            }

        if female_hits and not male_hits:
            return {
                "gender": "female",
                "confidence": min(1.0, 0.6 + 0.1 * len(female_hits)),
                "evidence": "по тексту: женские формы",
            }

        if male_hits and female_hits:
            return {"gender": "unknown", "confidence": 0.2, "evidence": "по тексту: противоречивые маркеры"}

        return {"gender": "unknown", "confidence": 0.1, "evidence": "по тексту: маркеры не найдены"}


# ======================================================
# CONFIG
# ======================================================

@dataclass
class TopicSentimentConfig:
    corpus_path: str = "all_reviews.csv"
    max_reviews: 10000 | None = None

    base_output_dir: str = "topic_sentiment_2"
    results_subdir: str = "topic_sentiment_results"
    logs_subdir: str = "topic_sentiment_logs"
    models_subdir: str = "models"

    sentences_filename: str = "sentences.csv"
    sentence_topics_with_sentiment_filename: str = "sentence_topics_with_sentiment.csv"
    review_sentiment_with_rating_filename: str = "review_sentiment_with_rating.csv"

    bertopic_model_name: str = "bertopic_big"


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


@dataclass
class BerTopicConfig:
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    min_topic_size: int = 150
    n_gram_range: tuple = (1, 2)
    low_memory: bool = True
    verbose: bool = True


class BerTopicModel:
    def __init__(self, config: Optional[BerTopicConfig] = None):
        self.config = config or BerTopicConfig()
        self.model: Optional[BERTopic] = None

    def _ensure_model(self):
        if self.model is None:
            embedding_model = SentenceTransformer(self.config.embedding_model_name)
            self.model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=self.config.min_topic_size,
                n_gram_range=self.config.n_gram_range,
                low_memory=self.config.low_memory,
                verbose=self.config.verbose,
            )

    def fit_transform(self, documents: List[str]):
        self._ensure_model()
        topics, _ = self.model.fit_transform(documents)
        return topics

    def transform(self, documents: List[str]):
        self._ensure_model()
        topics, probs = self.model.transform(documents)
        return topics, probs

    def get_topic_info(self):
        if self.model is None:
            raise RuntimeError("BERTopic не обучена")
        return self.model.get_topic_info()

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("BERTopic не обучена")
        self.model.save(path)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        obj.model = BERTopic.load(path)
        return obj


class CSVReviewCorpus:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self):
        df = pd.read_csv(self.csv_path)

        if "full_text" not in df.columns:
            text_cols = [c for c in ["text", "review", "content"] if c in df.columns]
            if not text_cols:
                raise ValueError("В CSV нет текста отзыва")
            df["full_text"] = df[text_cols[0]].astype(str)

        return df


# ======================================================
# PIPELINE
# ======================================================

class TopicSentimentPipeline:
    """
    ШАГ 2:
    - тематическое моделирование (BERTopic)
    - sentence-level сентимент
    - сопоставление предложений с темами
    - агрегация сентимента по отзывам
    - (доп.) детекция нецензурной лексики по отзывам (variant A)
    """

    def __init__(self, config: Optional[TopicSentimentConfig] = None):
        self.config = config or TopicSentimentConfig()

        self.base_dir = self.config.base_output_dir
        self.results_dir = os.path.join(self.base_dir, self.config.results_subdir)
        self.logs_dir = os.path.join(self.base_dir, self.config.logs_subdir)
        self.models_dir = os.path.join(self.base_dir, self.config.models_subdir)

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        self.df_reviews: Optional[pd.DataFrame] = None
        self.topic_model: Optional[BerTopicModel] = None

    # --------------------------------------------------
    # LOAD CORPUS
    # --------------------------------------------------

    def load_corpus(self):
        self.logger.info(f"Загружаем корпуса отзывов из файла: {self.config.corpus_path}")

        corpus = CSVReviewCorpus(self.config.corpus_path)
        self.df_reviews = corpus.load()
        if self.config.max_reviews is not None:
            before = len(self.df_reviews)
            self.df_reviews = self.df_reviews.iloc[: self.config.max_reviews].reset_index(drop=True)
            self.logger.info(
                f"Ограничение корпуса включено: берём первые {len(self.df_reviews)} из {before} отзывов"
            )

    # --------------------------------------------------
    # GENDER DETECTION (review-level) — ONLY TEXT
    # --------------------------------------------------

    def gender_detection(self):
        self.logger.info("Детекция гендера: по роду глаголов")

        gender_detector = GenderDetector()
        results = self.df_reviews["full_text"].apply(gender_detector.predict_from_text)

        self.df_reviews["gender_pred"] = results.apply(lambda x: x["gender"])
        self.df_reviews["gender_confidence"] = results.apply(lambda x: x["confidence"])
        self.df_reviews["gender_evidence"] = results.apply(lambda x: x["evidence"])

        out_path = os.path.join(self.results_dir, "review_gender.csv")

        cols = ["full_text", "gender_pred", "gender_confidence", "gender_evidence"]
        if "rating" in self.df_reviews.columns:
            cols = ["rating"] + cols
        if "topic" in self.df_reviews.columns:
            cols = ["topic"] + cols

        df_out = self.df_reviews.reset_index().rename(columns={"index": "doc_id"})
        keep_cols = ["doc_id"] + [c for c in cols if c in df_out.columns]

        df_out[keep_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"Таблица гендеров сохранена: {out_path}")

    # --------------------------------------------------
    # TOPIC MODELING
    # --------------------------------------------------

    def build_topics(self):
        model_path = os.path.join(self.models_dir, self.config.bertopic_model_name)

        if os.path.exists(model_path):
            self.logger.info(
                f"Загружаем сохранённую BERTopic модель из: {model_path}"
            )
            self.topic_model = BerTopicModel.load(model_path)
        else:
            self.logger.info("Обучение модели BERTopic на текстах отзывов")

            self.topic_model = BerTopicModel(BerTopicConfig())
            documents = self.df_reviews["full_text"].tolist()

            topics = self.topic_model.fit_transform(documents)
            self.df_reviews["topic"] = topics

            reviews_path = os.path.join(
                self.results_dir, "reviews_with_topics.csv"
            )
            self.df_reviews.to_csv(
                reviews_path, index=False, encoding="utf-8-sig"
            )
            self.logger.info(
                f"Файл с темами по отзывам сохранён: {reviews_path}"
            )

            topic_info_path = os.path.join(
                self.results_dir, "topic_info.csv"
            )
            self.topic_model.get_topic_info().to_csv(
                topic_info_path, index=False, encoding="utf-8-sig"
            )
            self.logger.info(
                f"Информация о темах BERTopic сохранена: {topic_info_path}"
            )

            self.topic_model.save(model_path)
            self.logger.info(
                f"BERTopic модель сохранена: {model_path}"
            )

    # --------------------------------------------------
    # SENTENCE LEVEL
    # --------------------------------------------------

    def sentence_level_analysis(self):
        self.logger.info("Sentence-level анализ: разбиение отзывов на предложения")

        splitter = SentenceSplitter(min_length=2)
        sentences = splitter.split_to_dicts(
            self.df_reviews["full_text"].tolist()
        )

        self.logger.info(f"Получено предложений: {len(sentences)}")

        self.logger.info("Sentence-level анализ: сентимент-анализ предложений")
        analyzer = SentimentAnalyzer()
        sentiment = analyzer.predict(sentences)

        self.logger.info("Sentence-level анализ: сопоставление предложений с темами")
        mapper = TopicSentenceMapper(self.topic_model)
        mapped = mapper.map_sentences(sentiment)

        out_path = os.path.join(
            self.results_dir,
            self.config.sentence_topics_with_sentiment_filename
        )
        pd.DataFrame(mapped).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )

        self.logger.info(
            f"Результаты sentence-level анализа сохранены: {out_path}"
        )

    # --------------------------------------------------
    # REVIEW LEVEL
    # --------------------------------------------------

    def review_level_aggregation(self):
        self.logger.info(
            "Агрегация сентимента на уровне отзывов и сравнение с пользовательскими оценками"
        )

        in_path = os.path.join(
            self.results_dir,
            self.config.sentence_topics_with_sentiment_filename
        )
        self.logger.info(f"Загружаем данные sentence-level анализа: {in_path}")

        df = pd.read_csv(in_path)

        agg = (
            df.groupby("doc_id", as_index=False)
            .agg(model_score=("sentiment_score", "mean"))
        )

        def score_to_label(s):
            if s < 0.45:
                return "negative"
            if s > 0.55:
                return "positive"
            return "neutral"

        agg["model_sentiment"] = agg["model_score"].apply(score_to_label)

        if "rating" in self.df_reviews.columns:
            ratings = self.df_reviews[["rating"]].reset_index()
            ratings.rename(columns={"index": "doc_id"}, inplace=True)

            agg = agg.merge(ratings, on="doc_id", how="left")

            def rating_to_label(r):
                if pd.isna(r):
                    return None
                if r <= 2:
                    return "negative"
                if r == 3:
                    return "neutral"
                return "positive"

            agg["rating_sentiment"] = agg["rating"].apply(rating_to_label)
            agg["match"] = agg["model_sentiment"] == agg["rating_sentiment"]

        out_path = os.path.join(
            self.results_dir,
            self.config.review_sentiment_with_rating_filename
        )
        agg.to_csv(out_path, index=False, encoding="utf-8-sig")

        self.logger.info(
            f"Итоговая таблица по отзывам сохранена: {out_path}"
        )

        # --------------------------------------------------
        # DETECTION OF PROFANITY (review-level) — Variant A
        # --------------------------------------------------

        self.logger.info("Поиск нецензурной лексики в отзывах")

        # Собираем таблицу doc_id -> full_text (+ rating если есть, не мешает)
        df_reviews_local = self.df_reviews.reset_index().rename(columns={"index": "doc_id"})

        # Присоединяем full_text к агрегированной таблице, чтобы пройтись по тем же отзывам
        df_for_toxicity = agg.merge(
            df_reviews_local[["doc_id", "full_text"]],
            on="doc_id",
            how="left"
        )

        detector = ProfanityDetector()
        toxicity_rows = []

        for _, row in df_for_toxicity.iterrows():
            text = row.get("full_text", "")
            result = detector.detect(str(text) if text is not None else "")

            toxicity_rows.append({
                "doc_id": int(row["doc_id"]),
                **result
            })

        df_toxicity = pd.DataFrame(toxicity_rows)

        out_toxicity = os.path.join(
            self.results_dir,
            "review_toxicity.csv"
        )

        df_toxicity.to_csv(
            out_toxicity, index=False, encoding="utf-8-sig"
        )

        self.logger.info(
            f"Таблица токсичности отзывов сохранена: {out_toxicity}"
        )

    # --------------------------------------------------
    # RUN
    # --------------------------------------------------

    def run(self):
        self.logger.info(
            "→ СТАРТ ШАГА 2: тематическое моделирование и сентимент-анализ"
        )

        self.load_corpus()
        self.gender_detection()
        self.build_topics()
        self.sentence_level_analysis()
        self.review_level_aggregation()

        self.logger.info("→ ШАГ 2 ЗАВЕРШЁН")