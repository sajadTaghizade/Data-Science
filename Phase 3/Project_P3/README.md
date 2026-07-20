# Stack Overflow C++ Similar-Question Recommender — Phase 3

Phase 3 takes the exploratory recommender from Phase 2 and turns it into a
**production-shaped, end-to-end machine-learning system**: several candidate
models are compared and the best is selected, it is trained and evaluated with
proper ranking metrics, and it is wired into **two automated pipelines** — one
that **trains** the model and one that **predicts** and **saves recommendations
back into the database**. MLflow experiment tracking is included as the bonus.

## Task

The task is unchanged from Phase 2: **semantic similar-question recommendation**.
Given a C++ question, return the most similar previously-asked questions. The
"prediction" the model makes is a **ranked list of the nearest questions**; those
recommendations are what we save back to the database.

## Phase 3 coverage

| Phase 3 section | Delivered implementation |
| --- | --- |
| 1. Model development & task definition | Six candidates compared — `title_word_tfidf`, `document_word_tfidf`, `char_wb_tfidf`, `lsa_document`, **`bm25_document`** (new), and a **hybrid** — under a temporal train/validation/test split; best chosen by **cross-validated nDCG@10**. |
| 2. Model training & evaluation | Hybrid weights tuned on validation with 5-fold CV; the selected model is evaluated **once** on the held-out test set with Hit@K, Recall@K, MAP@K, MRR@K, nDCG@K, and against random/popularity baselines. |
| 3. Integration into the pipeline (automation) | Separate **training** (`train_pipeline.py`) and **prediction** (`predict_pipeline.py`) pipelines; `run_pipeline.py` runs both with one command. Predictions are written to a `recommendations` table in SQLite. |
| 4. MLflow model management (bonus) | Every training run logs params, metrics, and artefacts (model + report) to a local `mlruns/` store; browse with the MLflow UI. |
| 5. Final presentation video | Add the Google Drive link in `video_link.txt` (or here) at submission time. |

## Results (2,500-question dataset)

**Section 1 — model selection (cross-validated nDCG@10 on validation):**

| Model | CV nDCG@10 | | Model | CV nDCG@10 |
| --- | --- | --- | --- | --- |
| **hybrid** ✅ | **0.367 ± 0.026** | | bm25_document | 0.290 ± 0.029 |
| char_wb_tfidf | 0.364 ± 0.027 | | lsa_document | 0.286 ± 0.030 |
| document_word_tfidf | 0.335 ± 0.028 | | title_word_tfidf | 0.169 ± 0.019 |

The **hybrid** blend wins, narrowly ahead of the strong character n-gram model;
BM25 and LSA contribute complementary signal.

**Section 2 — selected model on the held-out test set vs. baselines:**

| | Hit@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- |
| **hybrid model** | **0.827** | **0.576** | **0.359** |
| popularity baseline | 0.133 | 0.048 | 0.021 |
| random baseline | 0.201 | 0.073 | 0.024 |

The recommender lands a relevant neighbour in the top-10 for ~**83%** of unseen
questions — about **4×** the random baseline.

## The new model: BM25

Phase 3 adds **Okapi BM25**, the classic probabilistic ranking function used by
search engines, implemented **dependency-free** on top of a `CountVectorizer`.
Unlike cosine TF-IDF it *saturates* term frequency and *penalises long documents*,
giving genuinely different (and competitive) rankings — so "test multiple models
and pick the best" is a real experiment. Per-document BM25 weights are precomputed
at fit time, so scoring a batch of queries is a single sparse matrix product, just
like the cosine models.

## Project structure

```text
Project_P3/
├── data/                              # Generated; excluded from Git
│   ├── stackoverflow_questions.db     #   SQLite DB (incl. the recommendations table)
│   ├── processed/preprocessed_questions.csv
│   ├── models/best_recommender.joblib #   deployment artefact
│   ├── models/holdout_test_ids.json   #   unseen questions reserved for prediction
│   └── reports/*.csv, *.json          #   metric tables + recommendations.csv
├── scripts/
│   ├── database_connection.py         # shared SQLite connection (from Phase 2)
│   ├── import_to_db.py                 # CSV → normalized SQLite (from Phase 2)
│   ├── load_data.py                   # DB → Pandas (from Phase 2)
│   ├── preprocess.py                  # technical text cleaning (from Phase 2)
│   ├── recommender.py                 # NEW: models (incl. BM25), metrics, hybrid, inference
│   ├── train_model.py                 # NEW: training pipeline stage (select, tune, eval, save, MLflow)
│   └── make_predictions.py            # NEW: prediction pipeline stage (retrieve, save to DB)
├── train_pipeline.py                   # training pipeline (single command)
├── predict_pipeline.py                 # prediction pipeline (single command)
├── run_pipeline.py                     # both pipelines end-to-end
├── tests/                              # pytest: metrics, BM25, models, inference
├── phase3_report.ipynb                 # executed report: selection, metrics, live demo
├── requirements.txt
├── pyproject.toml
└── docs/PHASE3_REPORT.md               # full walkthrough (Persian + English)
```

The reusable data/DB scripts are shared with Phase 2 so the whole project stays
consistent; the recommender, training, and prediction code is new in Phase 3.

## Installation

Python 3.11+, from this directory:

```bash
python -m pip install -r requirements.txt
```

The pipeline reads the source data from `../../Phase 1/stackoverflow_questions.csv`.

## Run it

**Everything, one command** (training then prediction):

```bash
python run_pipeline.py
```

**Or the two pipelines separately:**

```bash
python train_pipeline.py     # import → load → preprocess → train & select model (+ MLflow)
python predict_pipeline.py   # load → preprocess → retrieve top-K → save to the database
```

The prediction stage also accepts options when run directly:

```bash
python scripts/make_predictions.py --top-k 10        # recommendations per query (default 10)
python scripts/make_predictions.py --all             # predict for every question, not just the holdout
```

### The two pipelines

- **Training pipeline** — `import_to_db → load_data → preprocess → train_model`.
  Splits temporally, fits every candidate on the training corpus, selects the best
  by cross-validated nDCG@10, tunes the hybrid weights, evaluates once on test,
  refits the chosen model on train+validation, and saves `best_recommender.joblib`.
  The **test questions are reserved** as the unseen data for prediction.
- **Prediction pipeline** — `load_data → preprocess → make_predictions`.
  Loads the saved model (**never retrains**), retrieves the top-K similar questions
  for each reserved holdout question, and writes them to the database.

### Predictions in the database

Recommendations are stored in a `recommendations` table:

| column | meaning |
| --- | --- |
| `query_question_id` | the question being matched |
| `rank` | 1 = most similar |
| `recommended_question_id` | a similar historical question |
| `score` | similarity score |
| `model_name` | model that produced it |
| `generated_at` | UTC timestamp |

```sql
SELECT r.rank, q.title, r.score
FROM recommendations r JOIN questions q ON q.question_id = r.recommended_question_id
WHERE r.query_question_id = 79907170
ORDER BY r.rank;
```

A CSV copy is also written to `data/reports/recommendations.csv`.

## MLflow (bonus)

Training logs to a local file store under `mlruns/`. Browse and compare runs:

```bash
mlflow ui --backend-store-uri mlruns      # then open http://127.0.0.1:5000
```

If MLflow is not installed the pipeline still runs — tracking degrades to a log
message, so the core deliverable never depends on the bonus.

## Tests and linting

```bash
python -m pytest          # metrics, BM25 correctness, model interface, inference
python -m ruff check .    # style + common bugs
```

## Reports and evidence

`phase3_report.ipynb` is committed **with outputs saved**: the model-selection
table and chart, the test metrics, the baseline comparison, a **live
recommendation demo**, and the stored predictions read back from SQLite. It loads
the pipeline's outputs rather than retraining, so the report always matches the
code. `docs/PHASE3_REPORT.md` is a step-by-step walkthrough in Persian and English.
