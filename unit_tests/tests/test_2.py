#ТЕСТ 1 — проверяет корректность разбиения текста на предложения
#ТЕСТ 2 — проверяет сохранение порядка предложений при сопоставлении с темами.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


import pytest

from topic_sentiment_2.topic_sentiment_pipeline import (
    SentenceSplitter,
    TopicSentenceMapper,
)

# TEST 1. SentenceSplitter: фильтрация коротких предложений

def test_sentence_splitter_min_length():
    splitter = SentenceSplitter(min_length=2)

    documents = [
        "Ок. Это хорошая доставка! Быстро."
    ]

    result = splitter.split_to_dicts(documents)
    texts = [r["text"] for r in result]

    # "Ок" и "Быстро" — по 1 слову, должны быть отброшены
    assert "Ок" not in texts
    assert "Быстро" not in texts

    # это предложение должно остаться
    assert "Это хорошая доставка" in texts


# TEST 2. TopicSentenceMapper: порядок предложений сохраняется

class DummyTopicModel:
    """
    Заглушка вместо BERTopic:
    возвращает один и тот же топик для всех предложений.
    """
    def transform(self, texts):
        topics = [0 for _ in texts]
        probs = [0.9 for _ in texts]
        return topics, probs


def test_topic_sentence_mapper_preserves_order():
    mapper = TopicSentenceMapper(DummyTopicModel())

    sentences = [
        {"doc_id": 0, "sentence_id": 1, "text": "Второе предложение"},
        {"doc_id": 0, "sentence_id": 0, "text": "Первое предложение"},
    ]

    result = mapper.map_sentences(sentences)

    # мэппинг НЕ должен менять порядок
    assert result[0]["sentence_id"] == 1
    assert result[1]["sentence_id"] == 0
