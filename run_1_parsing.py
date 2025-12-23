# run_parsing.py — главный пайплайн шага 1
# Использует:
#   - Factory Method (ParserFactory)
#   - Strategy (ExtendedAggregationStrategy)
#   - отдельный модуль collect_company_urls.py

import os
import time
import tracemalloc
import logging
from multiprocessing import Pool, cpu_count

#  Импорт фабрики парсеров (Factory Method) 
from parsing_1.factories.parser_factory import ParserFactory

#  Импорт стратегий агрегации (Strategy Pattern) 
from parsing_1.strategies.aggregation_strategy import ExtendedAggregationStrategy

#  Импорт агрегатора 
from parsing_1.aggregator.summary_aggregator import SummaryAggregator

#  Импорт Selenium-сборщика URL 
from parsing_1.collect_company_urls import OtzyvruCompanyCollector


# Логирование парсинга 
def setup_parsing_logging(log_path="parsing.log"):
    logger = logging.getLogger("parsing")
    logger.setLevel(logging.INFO)

    # очищаем старые хендлеры
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


# Одна задача (использует фабрику для создания парсера)

def run_one_url(args):
    url, save_json, save_log, source = args

    parser = ParserFactory.create_parser(
        source=source,
        url=url,
        save_path=save_json,
        log_path=save_log
    )

    parser.run()
    return url


# Итератор по партиям

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# Главный пайплайн

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(__file__)

    # Базовая папка шага 1
    PARSING_DIR = os.path.join(BASE_DIR, "parsing_1")

    # Логи парсинга
    LOG_DIR = os.path.join(PARSING_DIR, "parsing_logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Папка с результатами парсинга (по компаниям)
    RESULTS_DIR = os.path.join(PARSING_DIR, "parsing_results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Папка агрегатора (итоговые summary-файлы)
    AGGREGATOR_DIR = os.path.join(PARSING_DIR, "aggregator")
    os.makedirs(AGGREGATOR_DIR, exist_ok=True)

    logger = setup_parsing_logging(
        os.path.join(PARSING_DIR, "parsing.log")
    )


    # Старт измерения затраченных ресурсов
    total_start = time.perf_counter()
    tracemalloc.start()


    # URL категории
    CATEGORY = "https://www.otzyvru.com/companies/operatory-mobilnoy-svyazi/" # Страховые https://www.otzyvru.com/biznes-i-finansy/strahovye-kompanii/ # Грузоперевозки: https://www.otzyvru.com/companies/gruzoperevozki/ Операторы мобильной связи: https://www.otzyvru.com/companies/operatory-mobilnoy-svyazi/

    logger.info("→ Сбор ссылок на компании запущен")
    collector = OtzyvruCompanyCollector(
        CATEGORY,
        log_path="collect_urls.log",
        headless=True
    )

    urls = collector.collect()


    if not urls:
        logger.error("Не найдено ни одной компании → остановка.")
        exit(1)

    logger.info(f"Найдено компаний: {len(urls)}")
    for u in urls:
        logger.info(f" → {u}")

    # Формирование задач 
    tasks = [
        (
            url,
            os.path.join(RESULTS_DIR, f"{url.split('/')[-1]}.json"),
            os.path.join(PARSING_DIR, "parsing_logs", f"{url.split('/')[-1]}.log"),
            "otzyvru"
            )
        for url in urls
    ]


    cpu_cores = cpu_count()
    batch_size = cpu_cores

    logger.info(f"Используется ядер CPU: {cpu_cores}")
    logger.info(f"Количество задач: {len(tasks)}")

    #  Обработка по партиям 
    for batch_num, batch in enumerate(chunked(tasks, batch_size), start=1):

        logger.info(f"Запуск партии {batch_num} ({len(batch)} задач)")

        with Pool(len(batch)) as pool:
            for finished_url in pool.imap_unordered(run_one_url, batch):
                logger.info(f"Готово: {finished_url}")

        logger.info(f"Партия {batch_num} завершена")

    # Замер общих ресурсов
    try:
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    total_elapsed = time.perf_counter() - total_start
    peak_mb = peak / 1024 / 1024

    logger.info(
        f"Общее время работы: {total_elapsed:.2f} сек, "
        f"пик памяти: {peak_mb:.2f} МБ"
    )

   
   #Агрегация JSON-файлов (Strategy)
   
    logger.info("→ Сбор статистики запущен")

    aggregator = SummaryAggregator(
        folder=RESULTS_DIR,
        strategy=ExtendedAggregationStrategy()
    )

    # единый метод: summary + all_reviews + JSON
    outputs = aggregator.save_all_outputs(AGGREGATOR_DIR)

    logger.info("Агрегация завершена. Созданы файлы:")
    for name, path in outputs.items():
        logger.info(f"  {name}: {path}")

    logger.info("Все партии обработаны → процесс полностью завершен.")
