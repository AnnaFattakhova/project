# =====================================================
# SUMMARY AGGREGATOR
# Собирает:
#   1) reviews_summary.csv  — агрегированная статистика по компаниям
#   2) all_reviews.csv      — корпус всех отзывов (для ML)
#   3) all_reviews.json     — то же в JSON
#
# Использует Strategy для агрегации
# =====================================================

import os
import json
from pathlib import Path
import pandas as pd


class SummaryAggregator:
    """
    Агрегатор результатов парсинга.

    folder   — папка с JSON-файлами парсинга (*_final_XXXXwords.json)
    strategy — объект стратегии агрегации (Strategy Pattern)
    """

    def __init__(self, folder: str, strategy):
        self.folder = Path(folder)
        self.strategy = strategy

    # =====================================================
    # Вспомогательное
    # =====================================================
    def _load_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _iter_result_files(self):
        for fname in self.folder.iterdir():
            if fname.suffix == ".json" and "final" in fname.name:
                yield fname

    # =====================================================
    # 1) Агрегация по компаниям (SUMMARY)
    # =====================================================
    def summarize(self) -> pd.DataFrame:
        """
        Возвращает DataFrame:
        1 строка = 1 компания
        """
        rows = []

        for path in self._iter_result_files():
            data = self._load_json(path)

            reviews = data.get("reviews", [])

            row = {
                "file": path.name,
                "company_name": data.get("product", "—"),
                "company_url": data.get("company_url", "—"),
                "category": data.get("category", "—"),
                "subcategory": data.get("subcategory", "—"),
                "total_reviews": len(reviews),
                "word_count": data.get("word_count"),
            }

            # Strategy
            if self.strategy is not None:
                stats = self.strategy.aggregate(reviews)
                row.update(stats)

            rows.append(row)

        return pd.DataFrame(rows)

    # =====================================================
    # 2) Корпус ВСЕХ отзывов (для ML)
    # =====================================================
    def collect_all_reviews(self) -> pd.DataFrame:
        """
        Возвращает DataFrame:
        1 строка = 1 отзыв
        """
        all_rows = []

        for path in self._iter_result_files():
            data = self._load_json(path)

            company = data.get("product", "—")
            company_url = data.get("company_url", "—")
            category = data.get("category", "—")
            subcategory = data.get("subcategory", "—")

            for r in data.get("reviews", []):
                all_rows.append({
                    "company_name": company,
                    "company_url": company_url,
                    "category": category,
                    "subcategory": subcategory,
                    "author": r.get("author"),
                    "date": r.get("date"),
                    "rating": r.get("rating"),
                    "title": r.get("title"),
                    "text": r.get("text"),
                    "advantages": r.get("advantages"),
                    "disadvantages": r.get("disadvantages"),
                    "source_file": path.name,
                })

        return pd.DataFrame(all_rows)

    # =====================================================
    # 3) Сохранение отдельных файлов
    # =====================================================
    def save_reviews_csv(self, out_path: str):
        df = self.collect_all_reviews()
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    def save_reviews_json(self, out_path: str):
        df = self.collect_all_reviews()
        df.to_json(out_path, orient="records", force_ascii=False, indent=2)
        return out_path

    # =====================================================
    # 4) КОМБАЙН-МЕТОД (исправляет твою ошибку)
    # =====================================================
    def save_all_outputs(self, out_dir: str):
        """
        Сохраняет ВСЕ итоговые файлы разом.

        Возвращает словарь с путями — удобно для логов.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # --- all_reviews ---
        paths["all_reviews_csv"] = str(out_dir / "all_reviews.csv")
        paths["all_reviews_json"] = str(out_dir / "all_reviews.json")

        self.save_reviews_csv(paths["all_reviews_csv"])
        self.save_reviews_json(paths["all_reviews_json"])

        # --- summary ---
        summary_df = self.summarize()
        paths["reviews_summary_csv"] = str(out_dir / "reviews_summary.csv")
        summary_df.to_csv(
            paths["reviews_summary_csv"],
            index=False,
            encoding="utf-8-sig"
        )

        return paths
