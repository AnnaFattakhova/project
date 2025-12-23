import os
import json
import time
import logging
import tracemalloc
import requests
from bs4 import BeautifulSoup
from dateparser import parse as parse_date
from functools import wraps
from abc import ABC, abstractmethod

# Основная директория
BASE_DIR = os.path.dirname(__file__)

# Декоратор логирования
def logged(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "logger", None):
            self.logger.info(f"Вызов функции: {func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper

# Базовый класс Template Method
class BaseParser(ABC):

    @logged
    def run(self):
        start_time = time.perf_counter()
        tracemalloc.start()

        try:
            self.setup_logging()
            self.open_page()
            self.parse_product_info()
            self.load_all_reviews()
            self.extract_reviews()
            self.save_json()

        except Exception:
            self.logger.exception("Ошибка при работе парсера")

        finally:
            try:
                current, peak = tracemalloc.get_traced_memory()
            except Exception:
                current, peak = 0, 0
            finally:
                tracemalloc.stop()

            elapsed = time.perf_counter() - start_time
            peak_mb = peak / 1024 / 1024

            self.logger.info(
                f"Затраченные ресурсы: время обработки {elapsed:.2f} сек, "
                f"пик памяти {peak_mb:.2f} МБ"
            )

            self.logger.info("→ Парсинг завершён")

    @abstractmethod
    def setup_logging(self): pass

    @abstractmethod
    def open_page(self): pass

    @abstractmethod
    def parse_product_info(self): pass

    @abstractmethod
    def load_all_reviews(self): pass

    @abstractmethod
    def extract_reviews(self): pass

    @abstractmethod
    def save_json(self, suffix=""): pass


# Otzyvru парсер (через Requests)
class OtzyvruParser(BaseParser):

    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; WOW64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/85.0.4183.102 YaBrowser/20.9.3.136 '
            'Yowser/2.5 Safari/537.36'
        )
    }

    def __init__(self, url: str, save_path: str, log_path: str):
        self.url = url
        self.save_path = save_path
        self.log_path = log_path

        self.all_reviews = []
        self.pages_html = []

        self.category = "—"
        self.subcategory = "—"
        self.product_title = "—"

        self.logger = None


    # Логирование
    def setup_logging(self):
        logger = logging.getLogger(f"parsing.worker.{os.getpid()}")
        logger.setLevel(logging.INFO)

        for h in logger.handlers[:]:
            logger.removeHandler(h)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            "%d.%m.%Y %H:%M:%S"
        )

        fh = logging.FileHandler(self.log_path, encoding="utf-8", mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.logger = logger


    
    # Скачать первую страницу
    
    def open_page(self):
        self.logger.info(f"Скачивается первая страница: {self.url}")
        r = requests.get(self.url, headers=self.HEADERS)
        r.raise_for_status()
        self.pages_html.append(r.text)
        self.soup_first = BeautifulSoup(r.text, "html.parser")


    
    # Извлечение информации о продукте
    
    def parse_product_info(self):
        try:
            breadcrumbs = self.soup_first.select("div.breadcrumb span[itemprop='name']")
            if len(breadcrumbs) > 1:
                self.category = breadcrumbs[1].text.strip()
            if len(breadcrumbs) > 2:
                self.subcategory = breadcrumbs[2].text.strip()

            title = self.soup_first.select_one("h1.element_name")
            if title:
                self.product_title = title.text.replace(" отзывы", "").strip()

            self.logger.info(f"Раздел: {self.category}")
            self.logger.info(f"Категория: {self.subcategory}")
            self.logger.info(f"Название компании: {self.product_title}")

        except Exception as e:
            self.logger.error(f"Ошибка в parse_product_info: {e}")

        try:
            count_span = self.soup_first.select_one("span.count")
            self.total_reviews_available = (
                int(count_span.text.strip()) if count_span else None
            )
            self.logger.info(
                f"Заявленное число отзывов на сайте: {self.total_reviews_available}"
            )
        except Exception as e:
            self.logger.error(f"Ошибка при получении количества отзывов: {e}")
            self.total_reviews_available = None


    
    # Пагинация
    
    def load_all_reviews(self):
        self.logger.info("Загружаются все страницы отзывов")

        page = 2
        while True:
            page_url = f"{self.url}?page={page}"
            self.logger.info(f"Скачивается страница {page}: {page_url}")

            r = requests.get(page_url, headers=self.HEADERS)
            if r.status_code != 200:
                self.logger.info("Страница не существует → процесс прерван")
                break

            soup = BeautifulSoup(r.text, "html.parser")
            reviews = soup.select(".commentbox")
            if not reviews:
                self.logger.info(f"На странице {page} отзывов нет → процесс прерван")
                break

            self.pages_html.append(r.text)
            page += 1
            time.sleep(0.2)

        self.logger.info(f"Всего страниц скачано: {len(self.pages_html)}")


    
    # Извлечение отзывов
    
    def extract_reviews(self):
        self.logger.info("Извлекаются отзывы")

        for html in self.pages_html:
            soup = BeautifulSoup(html, "html.parser")
            reviews = soup.select(".commentbox")

            for review in reviews:
                title = review.select_one("h2 span[itemprop='name']")
                title = title.text.strip() if title else "—"

                author = review.select_one(".reviewer")
                author = author.text.strip() if author else "—"

                date_raw = review.select_one(".dtreviewed")
                if date_raw:
                    parsed = parse_date(date_raw.text.strip(), languages=["ru"])
                    date = parsed.strftime("%d.%m.%Y") if parsed else date_raw.text.strip()
                else:
                    date = "—"

                full_text = review.select_one(".review-full-text")
                snippet = review.select_one(".review-snippet")
                if full_text and full_text.text.strip():
                    text = full_text.text.strip()
                elif snippet:
                    text = snippet.text.strip()
                else:
                    text = "—"

                rating = "—"
                star = review.select_one(".star_ring span")
                if star and "width" in star.get("style", ""):
                    px = int(star["style"].split("width:")[1].split("px")[0].strip())
                    rating = px // 13

                pluses = [
                    li.text.strip()
                    for li in review.select(".advantages ol li")
                    if li.text.strip()
                ]
                advantages = pluses if pluses else "—"

                minuses = [
                    li.text.strip()
                    for li in review.select(".disadvantages ol li")
                    if li.text.strip()
                ]
                disadvantages = minuses if minuses else "—"

                self.all_reviews.append({
                    "author": author,
                    "date": date,
                    "title": title,
                    "text": text,
                    "rating": rating,
                    "advantages": advantages,
                    "disadvantages": disadvantages
                })

        actual = len(self.all_reviews)
        expected = self.total_reviews_available

        if expected is not None:
            if actual == expected:
                self.logger.info(f"Спарсены ВСЕ отзывы ({actual} из {expected})")
            else:
                self.logger.warning(f"Спарсены НЕ ВСЕ отзывы ({actual} из {expected})")
        else:
            self.logger.warning(
                "Не удалось определить заявленное число отзывов → сравнение пропущено"
            )

    
    # Подсчёт слов
    
    def count_words(self):
        return sum(
            len(r["text"].split())
            for r in self.all_reviews
            if isinstance(r["text"], str)
        )


    # Сохранение результатов
    
    def save_json(self, suffix=""):
        word_count = self.count_words()

        data = {
            "section": self.category,
            "category": self.subcategory,
            "company": self.product_title,
            "company_url": self.url,
            "word_count": word_count,
            "reviews": self.all_reviews,
        }

        filename = self.save_path.replace(
            ".json", f"_final_{word_count}words.json"
        )

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"Информация об отзывах сохранена в файл {filename}"
        )
