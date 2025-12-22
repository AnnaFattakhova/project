Структура проекта

# Общая логика
project/
│
├── parsing_1/
├── topic_sentiment_2/
├── aspect_analysis_3/
├── visualization_4/
│
├── run_1_parsing.py
├── run_2_topic_sentiment.py
├── run_3_aspect_analysis.py
├── run_4_visualization.py
│
├── unit_tests/
│   │
│   ├── tests/
│   │   │
│   │   ├── test_1.py
│   │   ├── test_2.py
│   │   ├── test_3.py
│   │   └── test_4.py
│   │
│   └── results/
│       │
│       ├── result_test_1.txt
│       ├── result_test_2.txt
│       ├── result_test_3.txt
│       └── result_test_4.txt
│
└── run

# Структура по шагам
## Шаг 1

project/
│
├── parsing_1/
│   │
│   ├── factories/
│   │   └── parser_factory.py
│   │
│   ├── aggregator/
│   │   │
│   │   ├── strategies/
│   │   │   └── aggregation_strategy.py
│   │   │
│   │   ├── all_reviews.csv
│   │   ├── all_reviews.json
│   │   └── reviews_summary.csv
│   │
│   ├── parsing_results/  
│   ├── parsing_logs/
│   │
│   ├── parsing.log              
│   ├── collect_urls.log         
│   │
│   ├── collect_company_urls.py
│   └── parser.py

ШАГ 1 — Парсинг отзывов и сбор корпуса

	→ отвечает за автоматизированный сбор отзывов с сайта OtzyvRu:

		* сбор URL компаний по категориям;
		* парсинг всех отзывов о каждой компании в рамках категории;
		* сохранение данных в JSON/CSV-формате;
		* агрегация результатов в единый корпус для последующего анализа.

	→ результат шага — корпус отзывов, готовый для тематического моделирования и сентимент-анализа.

	### Код шага 1

		→ collect_company_urls.py
			Цель: сбор списка URL компаний из заданной категории сайта OtzyvRu.
			Система классов:

			1) OtzyvruCompanyCollector
			  * класс-коллектор
			  * Использует Selenium для работы с динамическим контентом (lazy-loading).

		→ parser.py
			Назначение: парсинг страниц компаний и извлечение всех отзывов с метаданными (автор, дата и т.п.).
			Система классов:

			2) BaseParser (абстрактный класс, `ABC`)
			  * реализует шаблон проектирования Template Method;
			  * задаёт общий алгоритм парсинга:

				1. настройка логирования
				2. загрузка страницы
				3. извлечение информации о компании
				4. загрузка всех страниц отзывов
				5. извлечение отзывов и метаданных о них
				6. сохранение результата
			  → использует собственный декоратор @logged.

			3) OtzyvruParser(BaseParser)
			  * конкретная реализация парсера для сайта OtzyvRu;
			  * реализует все абстрактные методы базового класса;
			  * использует requests + BeautifulSoup;
			  * извлекает: название компании, категорию и подкатегорию, отзывы (текст, автор, дата, рейтинг, плюсы/минусы).

		→ factories/parser_factory.py
			Цель: фабрика создания парсеров.
			Система классов:

			4) ParserFactory
			  * реализует шаблон проектирования Factory Method;
			  * по имени источника sourct обращается к нужному парсеру;
        * позволяет добавлять новые источники без изменения основной логики.

		→ aggregator/summary_aggregator.py
			Цель: агрегация результатов парсинга и составление единого корпуса отзывов.
			Система классов: 

			5) SummaryAggregator
			  * наследует стратегию агрегации (Strategy);
			  * формирует summary-таблицу по компаниям,
				* собирает все отзывы в один DataFrame,
				* сохраняет корпус в CSV и JSON.

			→ aggregator/strategies/aggregation_strategy.py
			Цель: разные стратегии подсчёта статистики по отзывам.
			Система классов:

			6) AggregationStrategy (абстрактный класс)
			7) BasicAggregationStrategy

			  * число отзывов
			  * средний рейтинг
			  * количество слов

			8) ExtendedAggregationStrategy
			  * расширяет базовую стратегию
			  * добавляет временную аналитику и распределения

	
	### Логи шага 1

		* parsing.log — главный лог выполнения парсинга
		* collect_urls.log — лог сбора URL компаний
    * parsing_logs — папка для сбора логов по каждой обрабатываемой компании

	## Результаты шага 1

		 → _final_<N>words.json — результат парсинга каждой компании, содержащий метаданные компании, все отзывы и общее количество слов.
		 → all_reviews.json & all_reviews.csv — единый корпус отзывов (в двух форматах для удобства):
        JSON — хранение структурированных данных;
        CSV — удобство анализа в pandas и использования в следующих шагах;
		 →  reviews_summary.csv — агрегированная статистика по компаниям.

## Шаг 2

project/
│
├── topic_sentiment_2/
│   │
│   ├── models/                     
│   │   └── bertopic_big
│   │
│   ├── topic_sentiment_results/
│   │   │
│   │   ├── review_gender.csv
│   │   ├── review_sentiment_with_rating.csv
│   │   ├── review_toxicity.csv
│   │   ├── reviews_with_topics.csv
│   │   ├── sentence_topics_with_sentiment.csv
│   │   └── topic_info.csv
│   │
│   ├── topic_sentiment_logs/
│   │
│   └── topic_sentiment_pipeline.py


## Шаг 3

project/
│
├── aspect_analysis_3/
│   │
│   ├── aspect_analysis_results/
│   │   │
│   │   ├── aspects_extracted.csv
│   │   ├── aspects_filtered.csv
│   │   ├── aspect_summary.csv
│   │   ├── aspect_negative_refined.csv
│   │   └── top_negative_reasons_by_topic.csv
│   │
│   ├── aspect_analysis_logs/
│   │
│   └── aspect_analysis_pipeline.py

## Шаг 4

project/
│
├── visualization_4/
│   │
│   ├── figures/
│   │   │
│   │   ├── avg_sentiment_by_topic.png
│   │   ├── sentiment_vs_rating.png
│   │   └── top_negative_aspects.png
│   │
│   ├── visualization_logs/
│   │   │
│   │   └── visualization.log
│   │
└── └── visualization_pipeline.py


