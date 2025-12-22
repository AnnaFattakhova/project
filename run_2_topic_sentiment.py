#!/usr/bin/env python
# coding: utf-8

import logging
from pathlib import Path

from topic_sentiment_2.topic_sentiment_pipeline import (
    TopicSentimentPipeline,
    TopicSentimentConfig
)

# ======================================================
# LOGGING
# ======================================================

LOG_DIR = Path("topic_sentiment_2/topic_sentiment_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "topic_sentiment.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":

    config = TopicSentimentConfig(
        corpus_path="parsing_1/aggregator/all_reviews.csv",
        max_reviews= None
    )


    pipeline = TopicSentimentPipeline(config)
    pipeline.run()
