"""PREDICTION pipeline stage: load the trained model, retrieve, and save to the DB.

This is the "prediction pipeline" required by Phase 3 Section 3. It:

1. loads the **already trained** recommender artefact (it never re-trains);
2. loads the **new/unseen** questions - by default the temporal test split that
   the training pipeline reserved, i.e. questions the model has never seen;
3. for each query, retrieves the top-K most similar historical questions from the
   candidate pool the model was fit on;
4. **saves the recommendations back into the database** (a new ``recommendations``
   table) so they are stored and queryable, and also writes a CSV copy.

Run it after ``train_model.py``. The trained model is used purely for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import joblib
import pandas as pd
from sqlalchemy import bindparam, text

try:  # Package import or direct ``python scripts/make_predictions.py``.
    from . import recommender as R
    from .database_connection import PROJECT_ROOT, get_engine
    from .load_data import load_question_dataframe
    from .preprocess import PROCESSED_PATH, preprocess_dataframe
    from .train_model import ARTIFACT_PATH, HOLDOUT_IDS_PATH
except ImportError:  # pragma: no cover - direct script execution
    import recommender as R
    from database_connection import PROJECT_ROOT, get_engine
    from load_data import load_question_dataframe
    from preprocess import PROCESSED_PATH, preprocess_dataframe
    from train_model import ARTIFACT_PATH, HOLDOUT_IDS_PATH

logger = logging.getLogger(__name__)

PREDICTIONS_CSV_PATH = PROJECT_ROOT / "data" / "reports" / "recommendations.csv"
DEFAULT_TOP_K = 10

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recommendations (
    query_question_id       INTEGER NOT NULL,
    rank                    INTEGER NOT NULL,
    recommended_question_id INTEGER NOT NULL,
    score                   REAL    NOT NULL,
    model_name              TEXT    NOT NULL,
    generated_at            TEXT    NOT NULL,
    PRIMARY KEY (query_question_id, rank),
    FOREIGN KEY (query_question_id)       REFERENCES questions(question_id),
    FOREIGN KEY (recommended_question_id) REFERENCES questions(question_id)
);
"""


def load_query_questions(holdout_only: bool = True) -> pd.DataFrame:
    """Load the cleaned questions that will be scored as new/unseen queries."""
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
    else:  # Clean straight from SQLite if the processed artefact is missing.
        df, _ = preprocess_dataframe(load_question_dataframe())
    df["question_id"] = df["question_id"].astype(int)

    if holdout_only and HOLDOUT_IDS_PATH.exists():
        holdout_ids = set(json.loads(HOLDOUT_IDS_PATH.read_text(encoding="utf-8"))["holdout_test_question_ids"])
        df = df[df["question_id"].isin(holdout_ids)].reset_index(drop=True)
        logger.info("Predicting for %d reserved holdout (unseen) questions.", len(df))
    else:
        logger.info("Predicting for all %d questions.", len(df))
    return df


def recommendations_to_frame(query_df: pd.DataFrame, recommendations: list[list[dict]],
                             model_name: str) -> pd.DataFrame:
    """Flatten the per-query recommendation lists into one tidy DataFrame."""
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    query_ids = query_df["question_id"].to_numpy()
    rows = []
    for position, recs in enumerate(recommendations):
        for rec in recs:
            rows.append({
                "query_question_id": int(query_ids[position]),
                "rank": rec["rank"],
                "recommended_question_id": rec["recommended_question_id"],
                "score": rec["score"],
                "model_name": model_name,
                "generated_at": generated_at,
                "recommended_title": rec.get("recommended_title", ""),
            })
    return pd.DataFrame(rows)


def save_to_database(predictions: pd.DataFrame) -> int:
    """Persist recommendations into the ``recommendations`` table (idempotent per query)."""
    engine = get_engine()
    db_columns = ["query_question_id", "rank", "recommended_question_id", "score", "model_name", "generated_at"]
    records = predictions[db_columns].to_dict(orient="records")
    query_ids = sorted(predictions["query_question_id"].unique().tolist())
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys = ON"))
            connection.execute(text(CREATE_TABLE_SQL))
            # Idempotent: clear any previous recommendations for these queries first.
            delete_stmt = text("DELETE FROM recommendations WHERE query_question_id IN :ids").bindparams(
                bindparam("ids", expanding=True))
            connection.execute(delete_stmt, {"ids": query_ids})
            connection.execute(
                text("""
                    INSERT INTO recommendations
                        (query_question_id, rank, recommended_question_id, score, model_name, generated_at)
                    VALUES
                        (:query_question_id, :rank, :recommended_question_id, :score, :model_name, :generated_at)
                """),
                records,
            )
            total = connection.execute(text("SELECT COUNT(*) FROM recommendations")).scalar_one()
    finally:
        engine.dispose()
    return int(total)


def main(top_k: int = DEFAULT_TOP_K, holdout_only: bool = True) -> None:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Trained artefact not found at {ARTIFACT_PATH}. Run train_model.py first.")

    artifact = joblib.load(ARTIFACT_PATH)
    logger.info("Loaded trained model '%s' (candidate pool = %d questions).",
                artifact.best_model_name, len(artifact.candidates))

    query_df = load_query_questions(holdout_only=holdout_only)
    if query_df.empty:
        logger.warning("No query questions to predict for; nothing written.")
        return

    recommendations = R.recommend(artifact, query_df, top_k=top_k)
    predictions = recommendations_to_frame(query_df, recommendations, artifact.best_model_name)

    PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_CSV_PATH, index=False)
    total_rows = save_to_database(predictions)

    logger.info("Wrote %d recommendations for %d queries (top-%d each) to the database.",
                len(predictions), predictions["query_question_id"].nunique(), top_k)
    logger.info("recommendations table now holds %d rows; CSV copy at %s", total_rows, PREDICTIONS_CSV_PATH)
    logger.info("Prediction pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate and store similar-question recommendations.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Recommendations per query (default 10).")
    parser.add_argument("--all", action="store_true",
                        help="Predict for every question instead of only the reserved holdout set.")
    args = parser.parse_args()
    main(top_k=args.top_k, holdout_only=not args.all)
