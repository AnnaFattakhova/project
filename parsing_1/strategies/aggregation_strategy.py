# ШАБЛОН ПРОЕКТИРОВАНИЯ: STRATEGY
# Разные стратегии подсчёта статистики по JSON-файлам

from abc import ABC, abstractmethod
from datetime import datetime


class AggregationStrategy(ABC):
    """
    Абстрактная стратегия агрегации статистики по отзывам.
    """

    @abstractmethod
    def aggregate(self, reviews: list) -> dict:
        pass


# Базовая стратегия
class BasicAggregationStrategy(AggregationStrategy):
    """
    Базовая стратегия:
    - количество отзывов
    - средний рейтинг
    - количество слов
    """

    def aggregate(self, reviews: list) -> dict:
        num_reviews = len(reviews)
        word_count = sum(len(r.get("text", "").split()) for r in reviews)

        ratings = []
        for r in reviews:
            try:
                ratings.append(int(r.get("rating")))
            except:
                pass

        avg_rating = sum(ratings) / len(ratings) if ratings else None

        return {
            "num_reviews": num_reviews,
            "word_count": word_count,
            "avg_rating": avg_rating,
        }


# Расширенная стратегия 
class ExtendedAggregationStrategy(AggregationStrategy):
    """
    Расширенная стратегия:
    - базовая статистика
    - даты первого и последнего отзыва
    - средняя длина отзыва
    - распределение отзывов по годам и месяцам
    """

    def aggregate(self, reviews: list) -> dict:

        base = BasicAggregationStrategy().aggregate(reviews)

        # даты
        dates = []
        for r in reviews:
            d = r.get("date")
            try:
                dt = datetime.strptime(d, "%d.%m.%Y")
                dates.append(dt)
            except:
                pass

        first_date = min(dates).strftime("%d.%m.%Y") if dates else None
        last_date = max(dates).strftime("%d.%m.%Y") if dates else None

        # средняя длина
        avg_words = (
            base["word_count"] / base["num_reviews"]
            if base["num_reviews"] > 0 else None
        )

        # распределение по годам/месяцам
        per_year = {}
        per_month = {}

        for dt in dates:
            per_year[dt.year] = per_year.get(dt.year, 0) + 1
            key_month = f"{dt.year}-{dt.month:02d}"
            per_month[key_month] = per_month.get(key_month, 0) + 1

        base.update({
            "first_review_date": first_date,
            "last_review_date": last_date,
            "avg_words_per_review": avg_words,
            "reviews_per_year": per_year,
            "reviews_per_month": per_month,
        })

        return base
