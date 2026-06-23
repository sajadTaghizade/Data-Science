"""Main entry point for the reproducible Phase 2 data-science pipeline."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger("pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run(script_name: str) -> None:
    logger.info("=== Running %s ===", script_name)
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Import is included so a clean clone can rebuild the database before downstream stages.
    run("import_to_db.py")
    run("load_data.py")
    run("preprocess.py")
    run("feature_engineering.py")
    logger.info("Pipeline completed successfully.")
