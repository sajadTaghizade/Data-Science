"""Full Phase 3 pipeline: run the training pipeline, then the prediction pipeline.

A single entry point that automates everything end to end, as required by
Phase 3 Section 3 ("the entire process, from data loading to predictions, can be
executed with a single command"):

    training pipeline  (import -> load -> preprocess -> train & select model)
        then
    prediction pipeline (load -> preprocess -> predict & save recommendations to DB)

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

import logging

try:
    from . import predict_pipeline, train_pipeline
except ImportError:  # pragma: no cover - direct script execution
    import predict_pipeline
    import train_pipeline

logger = logging.getLogger("run_pipeline")


def main() -> None:
    logger.info("########## TRAINING PIPELINE ##########")
    train_pipeline.main()
    logger.info("########## PREDICTION PIPELINE ##########")
    predict_pipeline.main()
    logger.info("Full Phase 3 pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
