#ТЕСТ 1 — проверяет, что учитываются альтернативные назания колонок
#ТЕСТ 2 — проверяет, что при отсутствии нужной колонки появляется наше собственное исключение, а не стандартная ошибка

import sys
from pathlib import Path

# ======================================================
# FIX PYTHON PATH
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import pytest

from visualization_4.visualization_pipeline import (
    resolve_column,
    VisualizationDataError,
)


# ======================================================
# TEST 1. resolve_column: корректный алиас
# ======================================================

def test_resolve_column_finds_alias():
    """
    logical_name = 'rating'
    допустимый алиас = 'stars'
    """
    df = pd.DataFrame({
        "stars": [5, 4, 3]
    })

    col = resolve_column(df, "rating")
    assert col == "stars"


# ======================================================
# TEST 2. resolve_column: кастомное исключение
# ======================================================

def test_resolve_column_raises_custom_error():
    """
    Если подходящей колонки нет —
    выбрасывается VisualizationDataError, а не KeyError.
    """
    df = pd.DataFrame({
        "foo": [1, 2, 3]
    })

    with pytest.raises(VisualizationDataError):
        resolve_column(df, "sentiment")
