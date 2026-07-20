# Data Science Final Project — Stack Overflow C++ Similar-Question Recommender

An end-to-end data science project, from raw data collection to a deployed,
automated machine-learning pipeline. The goal is a **semantic similar-question
recommender**: given a C++ question, return the most similar previously-asked
Stack Overflow questions.

Everything lives in this one folder, split into three self-contained phases you
can run and test independently.

```text
Final_Project/
├── README.md                     ← you are here (project overview + quickstart)
├── Phase1/                       ← data collection (the crawler + the dataset)
│   ├── README.md
│   ├── crawl_stackoverflow.py
│   └── stackoverflow_questions.csv   ← shared dataset used by Phase 2 & 3
├── Phase2/                       ← database, EDA, preprocessing, pipeline, CI, Docker/K8s
│   ├── README.md
│   ├── pipeline.py  scripts/  tests/  k8s/  Dockerfile
│   └── section1_database_report.ipynb, section2_eda_report.ipynb
└── Phase3/                       ← model development, evaluation, train/predict pipelines, MLflow
    ├── README.md
    ├── docs/PHASE3_REPORT.md         ← full Persian + English walkthrough
    ├── run_pipeline.py  train_pipeline.py  predict_pipeline.py  scripts/  tests/
    └── phase3_report.ipynb
```

## The three phases at a glance

| Phase | What it does | Key deliverable |
| --- | --- | --- |
| **[Phase 1](Phase1/README.md)** | Collects ~2,500 C++ questions from the Stack Exchange API with a resumable crawler. | `stackoverflow_questions.csv` |
| **[Phase 2](Phase2/README.md)** | Loads the data into a normalized **SQLite** database, does EDA + leakage-safe text cleaning + feature engineering, wraps it in a reproducible **pipeline**, with **CI**, **Docker**, and **Kubernetes**. | Clean pipeline + hybrid recommender baseline |
| **[Phase 3](Phase3/README.md)** | Compares several recommender **models**, selects the best by cross-validation, evaluates it properly, and wires it into automated **training** and **prediction** pipelines that **save recommendations back to the database**. Adds **MLflow** tracking. | `best_recommender.joblib` + `recommendations` table |

## Quickstart — run any phase

Each phase has its own `requirements.txt` and its own README with details. In
short:

```bash
# ---- Phase 3 (the final model + pipelines) ----
cd Phase3
python -m pip install -r requirements.txt
python run_pipeline.py            # train + select model, then predict & save to the DB
python -m pytest                  # 30 unit tests
python scripts/recommend_cli.py --text "how to std::move a vector into a thread"

# ---- Phase 2 (database + data pipeline) ----
cd ../Phase2
python -m pip install -r requirements.txt
python pipeline.py                # build DB → preprocess → features
python -m pytest

# ---- Phase 1 (re-crawl more data, optional; run locally) ----
cd ../Phase1
python crawl_stackoverflow.py --pages 300
```

> **Shared dataset.** Phase 2 and Phase 3 read `Phase1/stackoverflow_questions.csv`
> automatically (resolved relative to each phase). Keep the three folders together
> and everything just works — no path configuration needed.

## Headline results (Phase 3, held-out test set)

The selected **hybrid** recommender lands a relevant neighbour in the top-10 for
about **84%** of unseen questions — roughly **4×** the random/popularity baselines.

| | Hit@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- |
| **hybrid model** | **0.839** | **0.613** | **0.385** |
| popularity baseline | 0.133 | 0.048 | 0.021 |
| random baseline | 0.201 | 0.073 | 0.024 |

## Reproducibility & quality

- **No heavy/opaque dependencies** — the models are TF-IDF / LSA / BM25 / PRF, so
  the whole project runs in CI and Docker and is fully reproducible.
- **Honest evaluation** — temporal train/validation/test split, training-only
  fitting, cross-validated model selection, and a single held-out test evaluation.
- **Tested & linted** — unit tests and `ruff` for both Phase 2 and Phase 3, run in
  GitHub Actions on every push.

## Final presentation (Phase 3, Section 5)

Add the Google Drive link to the ~15-minute presentation video and slides here (or
in a `video_link.txt`) at submission time.
