# Stack Overflow C++ Similar-Question Recommender — Phase 2

This project processes a crawled dataset of Stack Overflow C++ questions and prepares it for a semantic similar-question recommendation system. It completes Sections 1–3 of the Phase 2 brief: relational storage, leakage-safe modeling preparation, and a reproducible data pipeline.

## Project decision

The selected task is **semantic similar-question recommendation**. Title and body text are converted to TF-IDF vectors; cosine similarity then retrieves related historical questions.

This choice is better supported by the data than the alternatives:

- Difficulty prediction has no genuine difficulty label.
- Auto-tagging has 1,691 unique tags for 2,500 questions, creating a sparse long-tailed multi-label problem.
- Recommendation directly uses the rich text available for every question.

## Sections 1–3 coverage

| Phase 2 section | Delivered implementation |
| --- | --- |
| 1. Database implementation and querying | SQLite database, normalized `users`, `questions`, `tags`, and `question_tags` tables; foreign keys, constraints, indexes, automated Pandas/SQLAlchemy import, and SQL query examples in `section1_database_report.ipynb`. |
| 2. EDA, preprocessing, and feature engineering | `section2_eda_report.ipynb`: EDA, technical text cleaning, **temporal anti-leakage split**, multiple TF-IDF representations, a tuned **hybrid ensemble**, and quantitative retrieval metrics (`Hit@K`, `Recall@K`, `MAP@K`, `MRR@K`, `nDCG@K`). |
| 3. AI pipeline | Modular scripts in `scripts/`, `pipeline.py`, declared dependencies, and reproducible generated artefacts. |

## Modeling methodology (Sections 2 & 3)

The pipeline treats the recommender as a real retrieval problem rather than just producing feature columns:

- **Temporal train/validation/test split (70/15/15)** ordered by question creation date. We recommend historical questions for newer ones, so a time-based split avoids leaking the future into the past.
- **Three text representations**, each fit on the training corpus only: word TF-IDF on the title, word TF-IDF on the full document, and a character n-gram TF-IDF (robust to C++ syntax, library and function names, and error strings).
- **Hybrid weighted ensemble** whose weights are selected by grid search **on validation only**.
- **Weak relevance labels** from secondary-tag overlap (the universal `c++` tag is removed so it cannot create artificial relevance). Tags are used as evaluation labels and optional metadata — **not** as scoring features — to avoid leakage.
- **The test set is evaluated exactly once** for the final report; the model is then refit on all data for deployment.
- **Baseline comparison**: the hybrid model is benchmarked against random-order and popularity baselines on the same test set to prove it adds real value (it beats both by a wide margin on every metric).

## Project structure

```text
Project_P2/
├── data/                              # Generated; excluded from Git
│   ├── stackoverflow_questions.db
│   ├── interim/questions_from_database.csv
│   ├── processed/preprocessed_questions.csv
│   ├── processed/question_features.csv
│   ├── models/best_hybrid_recommender.joblib
│   └── reports/*.json, *.csv
├── scripts/
│   ├── database_connection.py         # Shared SQLite connection
│   ├── import_to_db.py                 # CSV → normalized SQLite database
│   ├── load_data.py                   # Database → Pandas DataFrame
│   ├── preprocess.py                  # Technical text cleaning and preprocessing
│   └── feature_engineering.py         # Split, TF-IDF models, hybrid tuning, metrics, artefact
├── pipeline.py                         # Main pipeline entry point
├── tests/                              # pytest unit tests (metrics, cleaning, split)
├── section1_database_report.ipynb      # Section 1 report and SQL outputs
├── section2_eda_report.ipynb           # Section 2 EDA + modeling report
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.11+ and run the following from this directory:

```bash
python -m pip install -r requirements.txt
```

The pipeline reads the source data from `../../Phase 1/stackoverflow_questions.csv`. Do not move that file unless `SOURCE_CSV_PATH` in `scripts/database_connection.py` is updated too.

## Scaling the dataset (optional)

The dataset was crawled from the Stack Exchange API. To grow it (e.g. from 2.5k
to 20k–100k questions for a stronger model), run the crawler **locally** — CI
and cloud environments block outbound access to `api.stackexchange.com`:

```bash
pip install requests pandas
cd "../../Phase 1"
# No API key needed — one run collects up to ~30k questions (the keyless daily quota):
python crawl_stackoverflow.py --pages 300
# Run it again another day to accumulate more; duplicates are skipped automatically:
python crawl_stackoverflow.py --pages 300
# Behind a local proxy/VPN, as in the team's original scraper:
python crawl_stackoverflow.py --pages 300 --proxy http://127.0.0.1:10808
```

The crawler queries the `/search/advanced` endpoint (`q="c++"`, `sort=activity`,
`filter=withbody`) — the same configuration as the original scrape — and
**appends** to `Phase 1/stackoverflow_questions.csv` with the exact same column
layout, so nothing downstream needs to change — just re-run `python pipeline.py`.

It works **without an API key**: the keyless quota is ~300 requests/day
(100 questions each, so ~30k/day). Because the crawler is resumable — it skips
question IDs already saved and saves progress every few pages — you can run it
across multiple days to grow the dataset further, or add `--key YOUR_APP_KEY`
(free, raises the quota to 10,000 requests/day) to pull much more at once. The pipeline is size-agnostic; metric computation
caps the number of evaluation queries (`MAX_EVAL_QUERIES`) so it stays fast even
on large corpora, while the candidate corpus itself is never sub-sampled.

## Run the full pipeline

```bash
python pipeline.py
```

The pipeline runs these stages in order:

1. `import_to_db.py` validates and imports the source CSV into SQLite.
2. `load_data.py` queries normalized data into a Pandas CSV.
3. `preprocess.py` loads directly from SQLite, removes invalid/duplicate text records, cleans HTML while preserving technical tokens and code markers, normalizes tags and timestamps, and records its decisions in JSON.
4. `feature_engineering.py` performs the temporal split, fits the TF-IDF models on train, tunes the hybrid weights on validation, evaluates the test set once, refits on all data, and saves the recommender artefact, metric tables, and a report.

**Loading and stage chaining.** As required by the brief, both `preprocess.py` and `feature_engineering.py` can load directly from the database through `database_connection.py`. To avoid recomputing the cleaning step twice in one run, `feature_engineering.py` reuses the cleaned artefact written by `preprocess.py` when it exists, and falls back to a direct database load when run standalone. The `import_to_db.py` stage is included first so a clean clone can rebuild all local data automatically.

## Feature columns: model inputs vs. audit-only

`feature_engineering.py` records explicit lists in `data/reports/feature_engineering_report.json`:

- **`model_text_input` / `models`** — the actual modeling inputs: the hybrid TF-IDF representations compared with cosine similarity.
- **`tag_multihot_columns`** — multi-hot indicator columns for the most frequent informative tags, saved as **optional metadata only** (kept out of scoring because tags are also the evaluation labels, so using them would leak).
- **`audit_only_columns`** — raw skewed counts (`view_count`, `answer_count`, `score`) and the `log`/`signed-log` transforms, retained for traceability but not fed to the model, to avoid near-duplicate correlated features.

Owner profile fields and very sparse migration metadata are excluded from the modeling dataset because they do not describe question meaning and would add noise.

## Generated artefacts

| Artefact | Purpose |
| --- | --- |
| `data/stackoverflow_questions.db` | Local normalized SQLite database (or `stackoverflow_questions_active.db` if another app has the default file open). |
| `data/interim/questions_from_database.csv` | Direct database extract used for inspection. |
| `data/processed/preprocessed_questions.csv` | Cleaned question text and normalized metadata; consumed by `feature_engineering.py`. |
| `data/processed/question_features.csv` | Model-ready table: cleaned text, numeric metadata, and multi-hot tag columns. |
| `data/models/best_hybrid_recommender.joblib` | Deployment artefact: fitted TF-IDF models, chosen weights, and corpus metadata for inference. |
| `data/reports/preprocessing_report.json` | Row-removal, missing-data, and correlation decisions. |
| `data/reports/feature_engineering_report.json` | Split sizes, best weights, test metrics, and model-vs-audit feature metadata. |
| `data/reports/section2_single_model_validation_metrics.csv` | Per-representation validation metrics. |
| `data/reports/section2_hybrid_weight_search_validation.csv` | Hybrid weight grid-search results on validation. |
| `data/reports/section2_best_hybrid_test_metrics.csv` | Final hybrid metrics on the held-out test set. |
| `data/reports/section2_baseline_vs_model_test_metrics.csv` | Random and popularity baselines vs. the hybrid model on the test set. |

## Tests

Unit tests cover the retrieval metrics, the technical text cleaning, the tag
handling, and the temporal split (ordering and no leakage). Run them from this
directory:

```bash
python -m pytest
```

These tests are intended to run in CI (Section 4) so that every push verifies
correctness, not just that the pipeline executes.

## Reports and evidence

- `section1_database_report.ipynb` is committed **with its outputs saved**: database schema, import verification, integrity checks, and SQL query results. Its schema DDL and import logic are imported from `scripts/import_to_db.py` so the report cannot drift from the pipeline code.
- `section2_eda_report.ipynb` is committed **with its outputs saved**: EDA tables/plots, the cleaning demo, the validation/test retrieval metrics loaded from the pipeline outputs, and a live recommendation demo using the saved artefact. It loads the pipeline's results rather than retraining, so Section 2 and Section 3 stay in sync. Run `python pipeline.py` first if you want to regenerate everything.
- For submission screenshots, open the executed notebooks and capture the schema/query/EDA/metric outputs directly from Jupyter or VS Code.

If `data/stackoverflow_questions.db` is open in another application, the importer creates `stackoverflow_questions_active.db` and stores its name in `data/active_database.txt`; all pipeline scripts automatically use that same active database. Close DB Browser for SQLite or any SQLite extension before a later run if you want to return to the default filename.
