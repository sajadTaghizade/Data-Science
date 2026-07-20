"""End-to-end TRAINING pipeline (single command).

Runs the training half of Phase 3 Section 3 in order:

    import_to_db  ->  load_data  ->  preprocess  ->  train_model

i.e. Data Loading -> Data Preprocessing -> Feature Engineering & Modelling ->
save the trained model. Run this whenever the source data changes or you want to
re-select and retrain the recommender.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("train_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

TRAINING_STAGES = ["import_to_db.py", "load_data.py", "preprocess.py", "train_model.py"]


def run(script_name: str) -> None:
    logger.info("=== Running %s ===", script_name)
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    for stage in TRAINING_STAGES:
        run(stage)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
