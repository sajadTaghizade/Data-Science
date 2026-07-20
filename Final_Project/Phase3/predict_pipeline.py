"""End-to-end PREDICTION pipeline (single command).

Runs the prediction half of Phase 3 Section 3 in order:

    load_data  ->  preprocess  ->  make_predictions

i.e. Data Loading -> Data Preprocessing (same transforms as training) -> load the
saved model, retrieve top-K similar questions, and save the recommendations back
to the database. It never re-trains the model.

Run ``train_pipeline.py`` first: the trained artefact must already exist.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("predict_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

PREDICTION_STAGES = ["load_data.py", "preprocess.py", "make_predictions.py"]


def run(script_name: str) -> None:
    logger.info("=== Running %s ===", script_name)
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    for stage in PREDICTION_STAGES:
        run(stage)
    logger.info("Prediction pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
