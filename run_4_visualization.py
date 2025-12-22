#!/usr/bin/env python
# coding: utf-8

import logging
from pathlib import Path

from visualization_4.visualization_pipeline import VisualizationPipeline


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "visualization_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "run_visualization.log"


# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RunVisualization")


# ======================================================
# RUN
# ======================================================

def main():
    logger.info("Запуск шага 4: визуализация результатов")

    pipeline = VisualizationPipeline()
    pipeline.run()

    logger.info("Шаг 4 успешно завершён")


if __name__ == "__main__":
    main()
