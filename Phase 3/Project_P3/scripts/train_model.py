"""TRAINING pipeline stage: model selection, tuning, evaluation, and artefact export.

This is the "training pipeline" required by Phase 3 Section 3. It:

1. loads the cleaned questions and makes a **temporal** train/validation/test split;
2. fits every candidate recommender model on the **training** corpus only;
3. **selects the best model** (Section 1) using cross-validated ranking quality on
   the validation queries - a real comparison of TF-IDF, char n-grams, LSA, BM25,
   and a tuned hybrid, not a formality;
4. **tunes the hybrid weights** on validation with the same cross-validation
   (hyperparameter tuning);
5. evaluates the selected model on the held-out **test** set exactly **once**, and
   against random/popularity baselines, to report honest generalisation (Section 2);
6. **refits** the chosen model on train+validation for deployment and saves the
   artefact, reserving the test questions as the "new/unseen" data the prediction
   pipeline will score;
7. logs the whole experiment to **MLflow** (Section 4 bonus) when it is installed.

The trained model is saved and is **not** re-trained by the prediction pipeline.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

try:  # Package import (tests, notebook) or direct ``python scripts/train_model.py``.
    from . import recommender as R
    from .database_connection import PROJECT_ROOT
    from .load_data import load_question_dataframe
    from .preprocess import PROCESSED_PATH, preprocess_dataframe, secondary_tags
except ImportError:  # pragma: no cover - direct script execution
    import recommender as R
    from database_connection import PROJECT_ROOT
    from load_data import load_question_dataframe
    from preprocess import PROCESSED_PATH, preprocess_dataframe, secondary_tags

logger = logging.getLogger(__name__)

# Output locations
ARTIFACT_PATH = PROJECT_ROOT / "data" / "models" / "best_recommender.joblib"
HOLDOUT_IDS_PATH = PROJECT_ROOT / "data" / "models" / "holdout_test_ids.json"
MODEL_SELECTION_PATH = PROJECT_ROOT / "data" / "reports" / "section1_model_selection.csv"
HYBRID_SEARCH_PATH = PROJECT_ROOT / "data" / "reports" / "section2_hybrid_weight_search.csv"
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "reports" / "section2_test_metrics.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "data" / "reports" / "section2_baseline_vs_model_test.csv"
TRAINING_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "phase3_training_report.json"

CV_FOLDS = 5
TOP_K_EVAL = 50


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_preprocessed() -> pd.DataFrame:
    """Reuse the preprocess.py artefact if present; otherwise clean straight from SQLite."""
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
    else:
        df, _ = preprocess_dataframe(load_question_dataframe())
    df["creation_at"] = pd.to_datetime(df["creation_at"], utc=True, errors="coerce")
    df["tag_list"] = df["tags"].fillna("").map(lambda value: [t for t in str(value).split("|") if t])
    return df


# --------------------------------------------------------------------------- #
# Cross-validated ranking quality on the validation queries
# --------------------------------------------------------------------------- #
def _fold_assignments(n_queries: int, n_folds: int, seed: int = R.RANDOM_STATE) -> list[np.ndarray]:
    """Shuffle query positions once and split them into ``n_folds`` index groups."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_queries)
    return [group for group in np.array_split(order, n_folds) if len(group)]


def cross_val_primary(score_matrix: np.ndarray, val_df: pd.DataFrame, candidate_secondary,
                      folds: list[np.ndarray], metric: str = R.PRIMARY_METRIC) -> tuple[float, float]:
    """Mean and std of ``metric`` over query folds, given a precomputed score matrix.

    Only the query set is folded; the candidate pool is fixed, so this measures
    how stable the ranking quality is across different sets of unseen queries.
    """
    tag_lists = val_df["tag_list"].tolist()
    fold_scores = []
    for fold in folds:
        rankings = R.rank_from_scores(score_matrix[fold], TOP_K_EVAL)
        metrics = R.evaluate_rankings(rankings, [tag_lists[i] for i in fold], candidate_secondary)
        if not np.isnan(metrics[metric]):
            fold_scores.append(metrics[metric])
    return (float(np.mean(fold_scores)), float(np.std(fold_scores))) if fold_scores else (float("nan"), float("nan"))


# --------------------------------------------------------------------------- #
# MLflow (optional bonus) - degrades gracefully when not installed
# --------------------------------------------------------------------------- #
def _mlflow_log(params: dict, metrics: dict, artifact_paths: list) -> str | None:
    """Log the run to a local MLflow file store. Tracking is a bonus, so any
    failure here is downgraded to a warning and never breaks the pipeline."""
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow not installed; skipping experiment tracking (Section 4 bonus).")
        return None
    try:
        mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
        mlflow.set_experiment("phase3_similar_question_recommender")
        # MLflow metric names may not contain '@', so 'Hit@10' -> 'Hit_at_10'.
        clean_metrics = {k.replace("@", "_at_"): v for k, v in metrics.items()
                         if v is not None and not np.isnan(v)}
        with mlflow.start_run() as run:
            mlflow.log_params({k: (json.dumps(v) if isinstance(v, dict | list) else v)
                               for k, v in params.items()})
            mlflow.log_metrics(clean_metrics)
            for path in artifact_paths:
                if path.exists():
                    mlflow.log_artifact(str(path))
            logger.info("Logged run %s to MLflow at %s", run.info.run_id, PROJECT_ROOT / "mlruns")
            return run.info.run_id
    except Exception as exc:  # pragma: no cover - tracking must never break training
        logger.warning("MLflow logging failed (%s); continuing without tracking.", exc)
        return None


# --------------------------------------------------------------------------- #
# Training pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    np.random.seed(R.RANDOM_STATE)
    questions = load_preprocessed()
    logger.info("Loaded %d preprocessed questions.", len(questions))

    # 1. Temporal split -----------------------------------------------------
    train_df, val_df, test_df = R.temporal_split(questions)
    val_df = R.sample_queries(val_df)
    test_df = R.sample_queries(test_df)
    train_secondary = [secondary_tags(tags) for tags in train_df["tag_list"]]
    logger.info("Split (temporal): train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # 2. Fit candidates on TRAIN only --------------------------------------
    models = R.fit_candidate_models(train_df)

    # Precompute each model's validation score matrix once (fast CV & tuning).
    val_scores = {model.name: model.score(val_df) for model in models}
    val_norm = {name: R.normalize_rows(scores) for name, scores in val_scores.items()}
    folds = _fold_assignments(len(val_df), CV_FOLDS)

    # 3+4. Model selection & hybrid tuning via cross-validated nDCG@10 ------
    selection_rows = []
    for model in models:
        cv_mean, cv_std = cross_val_primary(val_scores[model.name], val_df, train_secondary, folds)
        full = R.evaluate_rankings(R.rank_from_scores(val_scores[model.name], TOP_K_EVAL),
                                   val_df["tag_list"], train_secondary)
        selection_rows.append({"model": model.name, "type": "single",
                               "cv_nDCG@10_mean": cv_mean, "cv_nDCG@10_std": cv_std,
                               "val_Hit@10": full["Hit@10"], "val_MRR@10": full["MRR@10"],
                               "val_nDCG@10": full["nDCG@10"]})

    hybrid_rows = []
    for weights in R.hybrid_weight_grid():
        combined = sum(weights[name] * val_norm[name] for name in weights)
        cv_mean, cv_std = cross_val_primary(combined, val_df, train_secondary, folds)
        row = {"cv_nDCG@10_mean": cv_mean, "cv_nDCG@10_std": cv_std}
        row.update({name: round(weight, 4) for name, weight in weights.items()})
        hybrid_rows.append(row)
    hybrid_df = pd.DataFrame(hybrid_rows).sort_values("cv_nDCG@10_mean", ascending=False).reset_index(drop=True)
    best_weights = {name: float(hybrid_df.iloc[0][name]) for name in R.CANDIDATE_MODEL_NAMES}

    best_hybrid_combined = sum(best_weights[name] * val_norm[name] for name in best_weights)
    hybrid_cv_mean, hybrid_cv_std = cross_val_primary(best_hybrid_combined, val_df, train_secondary, folds)
    hybrid_full = R.evaluate_rankings(R.rank_from_scores(best_hybrid_combined, TOP_K_EVAL),
                                      val_df["tag_list"], train_secondary)
    selection_rows.append({"model": "hybrid", "type": "ensemble",
                           "cv_nDCG@10_mean": hybrid_cv_mean, "cv_nDCG@10_std": hybrid_cv_std,
                           "val_Hit@10": hybrid_full["Hit@10"], "val_MRR@10": hybrid_full["MRR@10"],
                           "val_nDCG@10": hybrid_full["nDCG@10"]})

    selection_df = pd.DataFrame(selection_rows).sort_values("cv_nDCG@10_mean", ascending=False).reset_index(drop=True)
    best_model_name = selection_df.iloc[0]["model"]
    logger.info("Model selection (CV %s):\n%s", R.PRIMARY_METRIC,
                selection_df[["model", "cv_nDCG@10_mean", "cv_nDCG@10_std", "val_nDCG@10"]].to_string(index=False))
    logger.info("Selected best model: %s", best_model_name)

    # 5. Evaluate the SELECTED model on TEST once + baselines ----------------
    if best_model_name == "hybrid":
        test_scores = R.combine_scores(models, test_df, best_weights)
    else:
        test_scores = next(m for m in models if m.name == best_model_name).score(test_df)
    test_metrics = R.evaluate_rankings(R.rank_from_scores(test_scores, TOP_K_EVAL),
                                       test_df["tag_list"], train_secondary)

    baseline_metrics = {
        "random": R.evaluate_rankings(R.rank_random(len(test_df), len(train_df), TOP_K_EVAL),
                                      test_df["tag_list"], train_secondary),
        "popularity": R.evaluate_rankings(R.rank_popularity(train_df, len(test_df), TOP_K_EVAL),
                                          test_df["tag_list"], train_secondary),
        f"selected_model[{best_model_name}]": test_metrics,
    }
    baseline_df = pd.DataFrame(baseline_metrics).T
    logger.info("TEST  %s: Hit@10=%.3f  MRR@10=%.3f  nDCG@10=%.3f",
                best_model_name, test_metrics["Hit@10"], test_metrics["MRR@10"], test_metrics["nDCG@10"])

    # 6. Refit the chosen model on TRAIN+VAL for deployment ------------------
    deploy_corpus = pd.concat([train_df, val_df], ignore_index=True)
    deploy_models = R.fit_candidate_models(deploy_corpus)
    candidate_meta = deploy_corpus[[c for c in R.CANDIDATE_META_COLUMNS if c in deploy_corpus.columns]].copy()
    artifact = R.RecommenderArtifact(
        models=deploy_models,
        candidates=candidate_meta.reset_index(drop=True),
        best_model_name=best_model_name,
        hybrid_weights=best_weights,
        text_fields=["title_clean", "document_clean"],
    )

    # 7. Persist everything -------------------------------------------------
    for path in (ARTIFACT_PATH, MODEL_SELECTION_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    holdout_test_ids = test_df["question_id"].astype(int).tolist()
    HOLDOUT_IDS_PATH.write_text(json.dumps({"holdout_test_question_ids": holdout_test_ids}, indent=2),
                                encoding="utf-8")

    selection_df.to_csv(MODEL_SELECTION_PATH, index=False)
    hybrid_df.to_csv(HYBRID_SEARCH_PATH, index=False)
    pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_PATH, index=False)
    baseline_df.to_csv(BASELINE_METRICS_PATH)

    report = {
        "task": "semantic_similar_question_recommendation",
        "phase": 3,
        "split": {"type": "temporal", "train_rows": len(train_df),
                  "validation_rows": len(val_df), "test_rows": len(test_df)},
        "candidate_models": list(R.CANDIDATE_MODEL_NAMES) + ["hybrid"],
        "model_selection_metric": f"cross_validated_{R.PRIMARY_METRIC}",
        "cv_folds": CV_FOLDS,
        "selected_best_model": best_model_name,
        "best_hybrid_weights": best_weights,
        "test_metrics": {k: (float(v) if pd.notna(v) else None) for k, v in test_metrics.items()},
        "baseline_test_metrics": {
            name: {k: (float(v) if pd.notna(v) else None) for k, v in metrics.items()}
            for name, metrics in baseline_metrics.items()
        },
        "deployment_candidate_pool": "train+validation",
        "holdout_test_questions_reserved_for_prediction": len(holdout_test_ids),
        "anti_overfit_controls": [
            "Temporal train/validation/test split (no future leaks into the past)",
            "All models fit on the training corpus only",
            "Best model & hybrid weights chosen by cross-validated validation nDCG@10",
            "Test set evaluated exactly once",
            "Generic c++ tag removed from the weak relevance labels",
        ],
    }
    TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 8. MLflow tracking (bonus) -------------------------------------------
    mlflow_params = {
        "candidate_models": list(R.CANDIDATE_MODEL_NAMES) + ["hybrid"],
        "selected_best_model": best_model_name,
        "best_hybrid_weights": best_weights,
        "cv_folds": CV_FOLDS, "top_k_eval": TOP_K_EVAL,
        "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
    }
    mlflow_metrics = {
        "test_Hit@10": test_metrics["Hit@10"], "test_MRR@10": test_metrics["MRR@10"],
        "test_nDCG@10": test_metrics["nDCG@10"], "test_MAP@10": test_metrics["MAP@10"],
        "test_Recall@10": test_metrics["Recall@10"],
        "cv_best_nDCG@10": float(selection_df.iloc[0]["cv_nDCG@10_mean"]),
    }
    _mlflow_log(mlflow_params, mlflow_metrics, [ARTIFACT_PATH, TRAINING_REPORT_PATH])

    logger.info("Saved trained artefact to %s", ARTIFACT_PATH)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
