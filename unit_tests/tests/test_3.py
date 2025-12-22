# TEST 1 — проверяет, что стоп-аспекты удаляются; слишком короткие аспекты удаляются
# TEST 2 — проверяет корректность перегрузки оператора

import sys
from pathlib import Path

# ======================================================
# FIX PYTHON PATH (как в test_1.py и test_2.py)
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from aspect_analysis_3.aspect_analysis_pipeline import (
    AspectFilter,
    AspectStats,
)


# ======================================================
# TEST 1. AspectFilter: стоп-аспекты и длина
# ======================================================

def test_aspect_filter_removes_stop_and_short_aspects():

    df = pd.DataFrame({
        "aspect_raw": ["билайн", "ок", "доставка"],
        "sentiment_label": ["negative", "negative", "negative"],
        "sentiment_score": [0.9, 0.8, 0.7],
    })

    flt = AspectFilter(min_length=3, min_count=1)
    out = flt.filter(df)

    aspects = out["aspect_norm"].tolist()

    assert "билайн" not in aspects      # стоп-аспект
    assert "ок" not in aspects          # слишком короткий
    assert "доставка" in aspects        # валидный аспект


# ======================================================
# TEST 2. AspectStats: перегрузка оператора +
# ======================================================

def test_aspect_stats_add_operator():

    a = AspectStats(
        mentions=1,
        pos_mentions=0,
        neg_mentions=1,
        pos_strength=0.0,
        neg_strength=0.6,
    )

    b = AspectStats(
        mentions=2,
        pos_mentions=2,
        neg_mentions=0,
        pos_strength=1.2,
        neg_strength=0.0,
    )

    c = a + b

    assert c.mentions == 3
    assert c.pos_mentions == 2
    assert c.neg_mentions == 1
    assert c.pos_strength == 1.2
    assert c.neg_strength == 0.6
