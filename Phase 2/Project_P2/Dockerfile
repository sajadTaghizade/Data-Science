# syntax=docker/dockerfile:1
# Containerizes the Phase 2 data pipeline. Build from the REPOSITORY ROOT so the
# source CSV in "Phase 1/" (which the pipeline reads at ../../Phase 1/...) is in
# the build context:
#
#   docker build -f "Phase 2/Project_P2/Dockerfile" -t so-recommender .
#   docker run --rm so-recommender
#
FROM python:3.12-slim

WORKDIR /app

# 1) Install dependencies first so this layer is cached across code changes.
COPY ["Phase 2/Project_P2/requirements.txt", "/app/Phase 2/Project_P2/requirements.txt"]
RUN pip install --no-cache-dir -r "/app/Phase 2/Project_P2/requirements.txt"

# 2) Source data — the pipeline resolves it at ../../Phase 1/stackoverflow_questions.csv
COPY ["Phase 1/stackoverflow_questions.csv", "/app/Phase 1/stackoverflow_questions.csv"]

# 3) Project code (scripts, pipeline.py, tests, config).
COPY ["Phase 2/Project_P2", "/app/Phase 2/Project_P2"]

WORKDIR "/app/Phase 2/Project_P2"

# Running the container rebuilds the database and regenerates all artefacts.
CMD ["python", "pipeline.py"]
