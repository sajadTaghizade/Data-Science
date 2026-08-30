# 📊 Data Science

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-Structured%20Streaming-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Welcome to my **Data Science** repository! It contains the six computer
assignments (`CA0`–`CA5`) and the team **final project** for the Data
Science course at the **University of Tehran (UT)**, spanning the field
end to end: exploratory data analysis, BI dashboards, big-data streaming,
classical ML, semi-supervised NLP, and a fully deployed recommender system.

The final project — [`Final_Project/`](./Final_Project) — was built with
teammates **Sadegh Samadi** and **Deniz Mostafazadegan**.

---

## 🛠️ Tech Stack & Tools

| Area | Tools |
|---|---|
| **Analysis & modeling** | Python, pandas, scikit-learn, CatBoost |
| **Visualization / BI** | Jupyter, Tableau, Power BI |
| **Big data / streaming** | Apache Spark (Structured Streaming), Apache Kafka, MongoDB |
| **NLP** | TF-IDF, LSA, BM25, semi-supervised learning |
| **MLOps** | Docker, Kubernetes, GitHub Actions CI, MLflow |
| **Database** | SQLite |

---

## 📂 Repository Layout

```
.
├── CA0/            Exploratory data analysis with pandas
├── CA1/            Data visualization & BI dashboards (Tableau, Power BI)
├── CA2/            Big-data streaming with Kafka + Spark Structured Streaming
├── CA3/            Gradient-boosted models (CatBoost) on tabular competitions
├── CA4/            ML foundations: regression, math foundations, knowledge tracing
├── CA5/            NLP: semi-supervised learning & sentiment classification
└── Final_Project/  Stack Overflow C++ similar-question recommender (team project)
```

---

## 📦 Assignments Overview

### [CA0 — Exploratory Data Analysis](./CA0)
First pass at pandas-based EDA: cleaning, summarizing, and visualizing two
real-world datasets (2016 US general election results, an FDA drug-safety
dataset).

### [CA1 — Data Visualization & BI](./CA1)
Three tasks moving from notebook plots to full BI tooling: geospatial join
and analysis of WHO TB-burden data against country coordinates alongside an
Iranian car-price dataset (Task 1), a Tableau workbook (Task 2), and a
Power BI dashboard built over a scraped Codal financial-news dataset with a
full write-up (Task 3).

### [CA2 — Big-Data Streaming](./CA2)
A Kafka + Spark Structured Streaming pipeline (with a MongoDB sink) built
around a restaurant-order dataset (`zomato.csv`, `historical_data.json`):
a producer/consumer pair streams and processes events in near real time.

### [CA3 — Gradient-Boosted Models](./CA3)
Kaggle-style tabular prediction tasks solved with **CatBoost**, including
full training/inference notebooks and submission files across three tasks
of increasing difficulty.

### [CA4 — Machine Learning Foundations](./CA4)
Regression on the California housing dataset (Task 1), the mathematical
foundations behind the models used — derivations and worked proofs (Task 2),
and an educational-data-mining task on the ASSISTments 2017 knowledge-tracing
dataset (Task 3).

### [CA5 — NLP & Semi-Supervised Learning](./CA5)
Semi-supervised classification over a partially labeled dataset (Task 1),
sentiment classification on a 40k-sentence dataset (Task 2), and a further
applied NLP task (Task 4).

### [Final Project — Similar-Question Recommender](./Final_Project)
A complete, end-to-end, team-built data product: given a C++ Stack Overflow
question, recommend the most semantically similar previously-asked
questions. Built in three independently runnable phases — data collection
(a resumable Stack Exchange crawler), a database + EDA + feature pipeline
with CI/Docker/Kubernetes, and model selection + evaluation + automated
train/predict pipelines tracked with MLflow. The selected hybrid model
reaches **Hit@10 = 0.893** on a held-out test set of ~26k questions — see
[`Final_Project/README.md`](./Final_Project/README.md) for the full
quickstart, architecture, and results.

---

## 🚀 Getting Started

Most `CAn/TaskN` folders are self-contained Jupyter notebooks — open the
`.ipynb` file and run it top to bottom; each PDF in the same folder is the
original assignment description.

The final project has its own multi-phase setup; see
[`Final_Project/README.md`](./Final_Project/README.md) for exact commands.

---

## 📄 License

This repository is shared for educational purposes. See individual
assignment/phase folders for any specific licensing notes.
