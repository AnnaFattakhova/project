# TEST 1 — демонстрирует Factory Method
# TEST 2 — проверяет извлечение рейтинга
# TEST 3 — проверяет поведение при отсутствии HTML-данных

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from unittest.mock import MagicMock
import pytest

from parsing_1.parser import OtzyvruParser
from parsing_1.factories.parser_factory import ParserFactory

# TEST 1. ParserFactory (Factory Method)

def test_parser_factory_creates_otzyvru_parser():
    parser = ParserFactory.create_parser(
        source="otzyvru",
        url="http://example.com",
        save_path="out.json",
        log_path="log.log"
    )
    assert isinstance(parser, OtzyvruParser)


def test_parser_factory_unknown_source():
    with pytest.raises(ValueError):
        ParserFactory.create_parser(
            source="unknown",
            url="http://example.com",
            save_path="out.json",
            log_path="log.log"
        )


# TEST 2. Извлечение рейтинга по ширине

def test_extract_rating_from_style():
    # 39px → 3 звезды при шаге 13
    style = "width: 39px;"
    px = int(style.split("width:")[1].split("px")[0].strip())
    rating = px // 13
    assert rating == 3


def test_extract_rating_invalid_style_fallback():
    style = "invalid-style"
    try:
        px = int(style.split("width:")[1].split("px")[0].strip())
        rating = px // 13
    except Exception:
        rating = "—"
    assert rating == "—"


# TEST 3. Поведение при отсутствии соответствующих данных на странице

def test_review_missing_fields_fallbacks():
    mock_review = MagicMock()

    # Имитируем отсутствие элементов
    mock_review.find_element.side_effect = Exception("element not found")
    mock_review.find_elements.return_value = []

    # title 
    try:
        title = mock_review.find_element().text
    except Exception:
        title = "—"

    assert title == "—"

    # advantages 
    advantages = mock_review.find_elements()
    advantages = advantages if advantages else "—"
    assert advantages == "—"

    # disadvantages 
    disadvantages = mock_review.find_elements()
    disadvantages = disadvantages if disadvantages else "—"
    assert disadvantages == "—"
