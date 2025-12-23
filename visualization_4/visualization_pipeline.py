import logging
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Директории

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FIGURES_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "visualization_logs"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Логирование

LOG_FILE = LOG_DIR / "visualization.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VisualizationPipeline")


# Собственнное исключение

class VisualizationDataError(Exception):
    """
    Ошибка данных: отсутствующие или некорректные колонки.
    """
    pass


COLUMN_ALIASES = {
        "sentiment": [
        "model_score",          
        "sentiment_score",
        "polarity",
        "compound",
        "predicted_sentiment"
    ],

    "topic": [
        "topic",
        "topic_id"
    ],
    
        "negative_score": [
        "neg_mentions",    
        "neg_strength",
        "negative_score",
        "negative_count"
    ],
    
    "rating": [
        "rating",
        "stars",
        "score"
    ],
    
    "aspect": [
        "aspect",
        "aspect_term"
    ]
}


def resolve_column(df: pd.DataFrame, logical_name: str) -> str:
    """
    Возвращает реальное имя колонки по логическому имени.
    """
    candidates = COLUMN_ALIASES.get(logical_name, [])
    for col in candidates:
        if col in df.columns:
            return col

    raise VisualizationDataError(
        f"Не найдена колонка для '{logical_name}'. "
        f"Пробовали: {candidates}. "
        f"Доступные колонки: {df.columns.tolist()}"
    )


# Базовый визуализатор (Template Method)

class BaseVisualizer(ABC):

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    def run(self):
        self.logger.info("Загрузка данных")
        data = self.load_data()

        self.logger.info("Построение графика")
        fig = self.plot(data)

        self.logger.info("Сохранение результата")
        self.save(fig)

        plt.close(fig)

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot(self, data: pd.DataFrame):
        pass

    def save(self, fig):
        out_path = FIGURES_DIR / f"{self.name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"Фигура сохранена: {out_path}")


# 1. Сентимент vs рейтинг

class SentimentRatingVisualizer(BaseVisualizer):

    def __init__(self):
        super().__init__("sentiment_vs_rating")

    def load_data(self):
        path = (
            PROJECT_DIR
            / "topic_sentiment_2"
            / "topic_sentiment_results"
            / "review_sentiment_with_rating.csv"
        )
        return pd.read_csv(path)

    def plot(self, df):
        x_col = resolve_column(df, "rating")
        y_col = resolve_column(df, "sentiment")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            alpha=0.4,
            ax=ax
        )

        ax.set_title("Sentiment vs Rating")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        return fig


# 2. Средний сентимент по темам

class TopicSentimentVisualizer(BaseVisualizer):

    def __init__(self):
        super().__init__("avg_sentiment_by_topic")

    def load_data(self):
        path = (
            PROJECT_DIR
            / "topic_sentiment_2"
            / "topic_sentiment_results"
            / "sentence_topics_with_sentiment.csv"
        )
        return pd.read_csv(path)

    def plot(self, df):
        topic_col = resolve_column(df, "topic")
        sentiment_col = resolve_column(df, "sentiment")

        grouped = (
            df.groupby(topic_col)[sentiment_col]
            .mean()
            .sort_values()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=grouped,
            x=topic_col,
            y=sentiment_col,
            ax=ax
        )

        ax.set_title("Average Sentiment by Topic")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Average Sentiment")

        return fig

# 3. Топ негативный аспектов

class NegativeAspectVisualizer(BaseVisualizer):

    def __init__(self):
        super().__init__("top_negative_aspects")

    def load_data(self):
        path = (
            PROJECT_DIR
            / "aspect_analysis_3"
            / "aspect_analysis_results"
            / "aspect_summary.csv"
        )
        return pd.read_csv(path)

    def plot(self, df):
        aspect_col = resolve_column(df, "aspect")
        neg_col = resolve_column(df, "negative_score")

        df = df.sort_values(neg_col, ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=df,
            y=aspect_col,
            x=neg_col,
            ax=ax
        )

        ax.set_title("Top-10 Negative Aspects")
        ax.set_xlabel("Negative Score")
        ax.set_ylabel("Aspect")

        return fig

# Основной пайплайн

class VisualizationPipeline:

    def __init__(self):
        self.visualizers = [
            SentimentRatingVisualizer(),
            TopicSentimentVisualizer(),
            NegativeAspectVisualizer()
        ]

    def run(self):
        logger.info("→ СТАРТ ШАГА 4: ВИЗУАЛИЗАЦИЯ")

        for vis in self.visualizers:
            logger.info(vis.name)
            try:
                vis.run()
            except Exception:
                logger.exception(f"Ошибка в визуализаторе {vis.name}")

        logger.info("→ ШАГ 4 ЗАВЕРШЁН")
