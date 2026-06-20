# Stack Overflow C++ Similar-Question Recommender — Phase 2

This project processes a crawled dataset of Stack Overflow C++ questions and prepares it for a semantic similar-question recommendation system. It completes Sections 1–3 of the Phase 2 brief: relational storage, modeling preparation, and a reproducible data pipeline.

## Project decision

The selected task is **semantic similar-question recommendation**. Title and body text are converted to TF-IDF vectors; a future model stage can compare them with cosine similarity to retrieve related historical questions.

This choice is better supported by the data than the alternatives:

- Difficulty prediction has no genuine difficulty label.
- Auto-tagging has 1,691 unique tags for 2,500 questions, creating a sparse long-tailed multi-label problem.
- Recommendation directly uses the rich text available for every question.

## Sections 1–3 coverage

| Phase 2 section | Delivered implementation |
| --- | --- |
| 1. Database implementation and querying | SQLite database, normalized `users`, `questions`, `tags`, and `question_tags` tables; foreign keys, constraints, indexes, automated Pandas/SQLAlchemy import, and SQL query examples in `section1_database_report.ipynb`. |
| 2. EDA, preprocessing, and feature engineering | `section2_eda_report.ipynb`, text cleaning, missing-value handling, duplicate checks, time/tag/text features, log transforms, numeric standardization, TF-IDF features, and quality reports. |
| 3. AI pipeline | Modular scripts in `scripts/`, `pipeline.py`, declared dependencies, and reproducible generated artefacts. |

## Project structure

```text
section1_database/
├── data/                              # Generated; excluded from Git
│   ├── stackoverflow_questions.db
│   ├── interim/questions_from_database.csv
│   ├── processed/preprocessed_questions.csv
│   ├── processed/question_features.csv
│   ├── models/tfidf_vectorizer.joblib
│   ├── models/question_tfidf_matrix.npz
│   └── reports/*.json
├── scripts/
│   ├── database_connection.py         # Shared SQLite connection
│   ├── import_data.py                 # CSV → normalized SQLite database
│   ├── load_data.py                   # Database → Pandas DataFrame
│   ├── preprocess.py                  # Cleaning and preprocessing
│   └── feature_engineering.py         # Features and TF-IDF artefacts
├── pipeline.py                         # Main pipeline entry point
├── section1_database_report.ipynb      # Section 1 report and SQL outputs
├── section2_eda_report.ipynb           # Section 2 EDA report
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.11+ and run the following from this directory:

```powershell
python -m pip install -r requirements.txt
```

The pipeline reads the source data from `../../Phase 1/stackoverflow_questions.csv`. Do not move that file unless `SOURCE_CSV_PATH` in `scripts/database_connection.py` is updated too.

## Run the full pipeline

```powershell
python pipeline.py
```

The pipeline deliberately runs these stages in order:

1. `import_data.py` validates and imports the source CSV into SQLite.
2. `load_data.py` queries normalized data into a Pandas CSV.
3. `preprocess.py` loads directly from SQLite, removes invalid/duplicate text records, cleans HTML, normalizes tags and timestamps, and records its decisions in JSON.
4. `feature_engineering.py` loads directly from SQLite, constructs text/time/tag/code/engagement features, standardizes selected numeric metadata, and saves TF-IDF artefacts.

Both preprocessing and feature engineering load directly from the database, as required by the brief. The import stage is included before them so a clean clone can rebuild all local data automatically.

## Data-preparation decisions

- **Section 1:** validates required source fields, primary-key uniqueness, and parsable tag lists; normalizes relational entities; converts timestamps to ISO-8601 UTC.
- **Section 2:** performs analysis and modeling-specific preparation: HTML/text cleaning, missing-data handling, duplicate checks, log transforms for skewed engagement variables, standardization, and feature creation.
- **Section 3:** turns those decisions into deterministic, modular scripts and writes reproducible artefacts.

Raw engagement counts are retained for auditing and possible tie-breaking but are not the semantic-similarity target. Owner profile fields and very sparse migration metadata are excluded from the modeling dataset because they do not describe question meaning and would add noise.

## Generated artefacts

| Artefact | Purpose |
| --- | --- |
| `data/stackoverflow_questions.db` | Local normalized SQLite database (or `stackoverflow_questions_active.db` if another app has the default file open). |
| `data/interim/questions_from_database.csv` | Direct database extract used for inspection. |
| `data/processed/preprocessed_questions.csv` | Cleaned question text and normalized metadata. |
| `data/processed/question_features.csv` | Text-length, tag, code, time, transformed, and standardized metadata features. |
| `data/models/tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer. |
| `data/models/question_tfidf_matrix.npz` | Sparse TF-IDF matrix for question text. |
| `data/reports/preprocessing_report.json` | Row-removal and missing-data decisions. |
| `data/reports/feature_engineering_report.json` | TF-IDF and scaled-feature metadata. |

## Reports and evidence

- Open `section1_database_report.ipynb` to see the database schema, import verification, query results, and interpretations below each query.
- Run `section2_eda_report.ipynb` after the pipeline to view EDA tables and plots.
- For submission screenshots, execute the notebooks and capture the schema/query/EDA outputs directly from Jupyter or VS Code.

If `data/stackoverflow_questions.db` is open in another application, the importer creates `stackoverflow_questions_active.db` and stores its name in `data/active_database.txt`; all pipeline scripts automatically use that same active database. Close DB Browser for SQLite or any SQLite extension before a later run if you want to return to the default filename.
