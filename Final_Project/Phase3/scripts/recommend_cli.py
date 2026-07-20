"""Interactive demo: get similar-question recommendations from the command line.

Loads the trained artefact and recommends similar questions for either an
existing question (by id) or a free-text query you type. Great for the
presentation/video demo.

Examples
--------
    # recommend for a free-text query, with diversity (MMR) on:
    python scripts/recommend_cli.py --text "how to std::move a vector into a thread" --diversity 0.7

    # recommend for an existing question already in the database:
    python scripts/recommend_cli.py --question-id 79907170 --top-k 5
"""

from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd

try:  # Package import or direct execution.
    from . import recommender as R
    from .preprocess import normalize_technical_text
    from .train_model import ARTIFACT_PATH
except ImportError:  # pragma: no cover - direct script execution
    import recommender as R
    from preprocess import normalize_technical_text
    from train_model import ARTIFACT_PATH

logger = logging.getLogger(__name__)


def query_from_text(text: str) -> pd.DataFrame:
    """Build a one-row query frame from free text, cleaned exactly like training data."""
    cleaned = normalize_technical_text(text)
    return pd.DataFrame([{"question_id": None, "title_clean": cleaned, "document_clean": cleaned}])


def query_from_question_id(artifact, question_id: int) -> pd.DataFrame:
    """Build a query frame from an existing question's cleaned text (from the processed file)."""
    from preprocess import PROCESSED_PATH  # local import keeps the module import graph simple
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("Preprocessed data not found; run the training pipeline first.")
    processed = pd.read_csv(PROCESSED_PATH)
    row = processed[processed["question_id"].astype(int) == question_id]
    if row.empty:
        raise SystemExit(f"question_id {question_id} not found in the processed data.")
    return row[["question_id", "title_clean", "document_clean"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend similar C++ questions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Free-text query (a question title/body).")
    group.add_argument("--question-id", type=int, help="An existing question id to find neighbours for.")
    parser.add_argument("--top-k", type=int, default=5, help="How many recommendations (default 5).")
    parser.add_argument("--diversity", type=float, default=1.0,
                        help="MMR lambda in (0,1]; <1 diversifies results (default 1.0 = pure relevance).")
    args = parser.parse_args()

    if not ARTIFACT_PATH.exists():
        raise SystemExit(f"Trained artefact not found at {ARTIFACT_PATH}. Run train_pipeline.py first.")
    artifact = joblib.load(ARTIFACT_PATH)

    if args.text is not None:
        query_df = query_from_text(args.text)
        header = f'Free-text query: "{args.text}"'
    else:
        query_df = query_from_question_id(artifact, args.question_id)
        title_lookup = artifact.candidates.set_index("question_id")["title"] \
            if "title" in artifact.candidates.columns else {}
        known = args.question_id in getattr(title_lookup, "index", [])
        header = f"Question #{args.question_id}" + (f': "{title_lookup[args.question_id]}"' if known else "")

    recommendations = R.recommend(artifact, query_df, top_k=args.top_k, diversity=args.diversity)[0]
    print("=" * 88)
    print(header)
    print(f"Model: {artifact.best_model_name}"
          + (f"  |  MMR diversity λ={args.diversity}" if args.diversity < 1.0 else ""))
    print("-" * 88)
    if not recommendations:
        print("No recommendations found.")
    for rec in recommendations:
        print(f"  {rec['rank']}. [{rec['score']:.3f}] {rec['recommended_title']}")
        if rec.get("recommended_url"):
            print(f"       {rec['recommended_url']}")
    print("=" * 88)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
