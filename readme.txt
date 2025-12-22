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


