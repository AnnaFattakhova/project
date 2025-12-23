import os
import logging
import time
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Константы селекторов (специальные шаблоны, которые используются для точного определения и извлечения нужных данных)
SLIDER_SELECTOR = "div.horizontal-scrolling.slider-ready"
COMPANY_LINK_SELECTOR = SLIDER_SELECTOR + " a[href]"
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = BASE_DIR 
os.makedirs(LOG_DIR, exist_ok=True)

# Логирование
def setup_collect_logging(log_path="collect_urls.log"):
    logger = logging.getLogger("parsing.collect")
    logger.setLevel(logging.INFO)
    log_path = os.path.join(LOG_DIR, log_path) 

    # очищаем хендлеры, если были (функции или методы, которые выполняют конкретные действия с данными после того, как парсер извлек их)
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


class OtzyvruCompanyCollector:
    """
    Сборщик URL компаний для одной категории сайта OtzyvRu.
    """

    def __init__(self, category_url: str, log_path="collect_urls.log", headless=True):
        self.category_url = category_url
        self.log_path = log_path
        self.headless = headless
        self.logger = setup_collect_logging(log_path)

    # Создание Selenium драйвера
    def _create_driver(self):
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-gpu")
        return webdriver.Chrome(options=opts)

    # Основной метод сбора ссылок
    def collect(self):
        """
        Возвращает отсортированный список URL компаний категории.
        """

        self.logger.info("→ Старт сбора компаний")
        self.logger.info(f"URL категории: {self.category_url}")

        driver = self._create_driver()

        try:
            driver.get(self.category_url)
            self.logger.info("Страница категории загружена, ожидается появление slider-блоков.")

            #  1. Ждём появления слайдеров 
            slider_blocks = WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, SLIDER_SELECTOR)
                )
            )
            self.logger.info(f"Найдено slider-блоков: {len(slider_blocks)}")

            #  2. Прокручиваем lazy-loaded блоки 
            for i, block in enumerate(slider_blocks, start=1):
                try:
                    self.logger.info(f"Прокрутка slider-блока #{i}...")
                    driver.execute_script(
                        "arguments[0].scrollLeft = arguments[0].scrollWidth", block
                    )
                    time.sleep(1.2)
                except Exception as e:
                    self.logger.warning(f"Прокрутка #{i} не удалась: {e}")

            #  3. Ждём появления ссылок 
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, COMPANY_LINK_SELECTOR)
                )
            )
            self.logger.info("Обнаружены ссылки внутри slider-блоков.")

            time.sleep(1)

            #  4. HTML → BS4 
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            #  5. Извлекаем ссылки 
            nodes = soup.select(COMPANY_LINK_SELECTOR)
            self.logger.info(f"Ссылок найдено(a[href]): {len(nodes)}")

            urls = set()
            for a in nodes:
                href = a["href"].strip()
                urls.add(href)

            self.logger.info(f"Из них уникальные: {len(urls)}")

            #  6. Нормализация URL 
            abs_urls = []
            for u in urls:
                if u.startswith("https://www.otzyvru.com"):
                    abs_urls.append(u)
                elif u.startswith("/"):
                    abs_urls.append("https://www.otzyvru.com" + u)

            abs_urls = sorted(abs_urls)

            for u in abs_urls:
                self.logger.info(f" → {u}")

            self.logger.info("→ Сбор компаний завершен")
            return abs_urls

        finally:
            self.logger.info("WebDriver закрыт")
            driver.quit()
