"""Main entry point for the reproducible Phase 2 data-science pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run(script_name: str) -> None:
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    # Import is included so a clean clone can rebuild the database before downstream stages.
    run("import_to_db.py")
    run("load_data.py")
    run("preprocess.py")
    run("feature_engineering.py")
    print("\nPipeline completed successfully.")
