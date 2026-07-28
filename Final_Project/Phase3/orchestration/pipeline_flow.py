"""Prefect orchestration of the Phase 3 recommender pipeline (Section 3, bonus).

This is the *workflow-automation-tool* option from the Phase 3 brief (Apache
Airflow or Prefect). It orchestrates the SAME stage scripts that
``run_pipeline.py`` runs, but as a Prefect **DAG**, so the full train + predict
workflow becomes a single, observable orchestration with explicit task
dependencies:

    import_to_db -> load_data -> preprocess -> train_model -> make_predictions
    |------------------ training pipeline -----------------|  |-- prediction --|

Prefect only *orchestrates* — it does not change any model, data, or metric.
The results are byte-for-byte identical to ``run_pipeline.py`` (all stages are
seeded), it just adds retries, logging, and a UI/graph view on top.

Run it with::

    pip install prefect
    python orchestration/pipeline_flow.py

Optionally start the UI first to see the DAG and task states::

    prefect server start          # UI at http://127.0.0.1:4200
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run_stage(script_name: str) -> None:
    """Run one pipeline stage script exactly as train_pipeline.py / predict_pipeline.py do."""
    logger = get_run_logger()
    logger.info("=== Running %s ===", script_name)
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)],
        check=True,
        cwd=PROJECT_ROOT,
    )


@task(name="import_to_db")
def import_to_db() -> str:
    """Build the normalised SQLite database from the raw CSV."""
    _run_stage("import_to_db.py")
    return "db-ready"


@task(name="load_data")
def load_data(_upstream: str) -> str:
    """Query the database into a working CSV."""
    _run_stage("load_data.py")
    return "loaded"


@task(name="preprocess")
def preprocess(_upstream: str) -> str:
    """Clean the technical text and build the temporal split."""
    _run_stage("preprocess.py")
    return "preprocessed"


@task(name="train_model")
def train_model(_upstream: str) -> str:
    """Cross-validate the candidate models, tune BM25, select + save the recommender."""
    _run_stage("train_model.py")
    return "trained"


@task(name="make_predictions")
def make_predictions(_upstream: str) -> str:
    """Retrieve top-K similar questions and write the recommendations back to the DB."""
    _run_stage("make_predictions.py")
    return "predicted"


@flow(name="phase3-recommender-pipeline", log_prints=True)
def phase3_pipeline() -> None:
    """Training pipeline followed by the prediction pipeline, as one Prefect DAG.

    Tasks are called in sequence and each passes a token to the next, which both
    forces the correct order *and* records the dependency edges shown in the UI.
    """
    a = import_to_db()
    b = load_data(a)
    c = preprocess(b)
    d = train_model(c)
    make_predictions(d)


if __name__ == "__main__":
    phase3_pipeline()
