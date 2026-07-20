# syntax=docker/dockerfile:1
# Containerizes the Phase 2 data pipeline. Build from the Final_Project directory
# so the source CSV in "Phase1/" (which the pipeline reads at ../Phase1/...) is in
# the build context:
#
#   cd Final_Project
#   docker build -f Phase2/Dockerfile -t so-recommender .
#   docker run --rm so-recommender
#
FROM python:3.12-slim

WORKDIR /app

# 1) Install dependencies first so this layer is cached across code changes.
COPY ["Phase2/requirements.txt", "/app/Phase2/requirements.txt"]
RUN pip install --no-cache-dir -r "/app/Phase2/requirements.txt"

# 2) Source data — the pipeline resolves it at ../Phase1/stackoverflow_questions.csv
COPY ["Phase1/stackoverflow_questions.csv", "/app/Phase1/stackoverflow_questions.csv"]

# 3) Project code (scripts, pipeline.py, tests, config).
COPY ["Phase2", "/app/Phase2"]

WORKDIR "/app/Phase2"

# Running the container rebuilds the database and regenerates all artefacts.
CMD ["python", "pipeline.py"]
