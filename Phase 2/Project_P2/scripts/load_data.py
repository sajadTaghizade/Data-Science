"""Load the modeling-relevant relational data from SQLite into Pandas."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text

try:  # Supports both `python scripts/load_data.py` and `from scripts.load_data import ...`.
    from .database_connection import PROJECT_ROOT, get_engine
except ImportError:  # pragma: no cover - direct script execution
    from database_connection import PROJECT_ROOT, get_engine


logger = logging.getLogger(__name__)

INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "questions_from_database.csv"

QUESTION_QUERY = """
SELECT
    q.question_id,
    q.title,
    q.body_html,
    q.question_url,
    q.is_answered,
    q.view_count,
    q.answer_count,
    q.score,
    q.accepted_answer_id,
    q.creation_at,
    q.last_activity_at,
    q.last_edit_at,
    q.closed_at,
    q.closed_reason,
    GROUP_CONCAT(t.name, '|') AS tags,
    COUNT(t.tag_id) AS tag_count
FROM questions AS q
LEFT JOIN question_tags AS qt ON qt.question_id = q.question_id
LEFT JOIN tags AS t ON t.tag_id = qt.tag_id
GROUP BY q.question_id
ORDER BY q.question_id;
"""


def load_question_dataframe() -> pd.DataFrame:
    """Read core question text plus normalized tags directly from the database."""
    engine = get_engine()
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(text(QUESTION_QUERY), connection)
    finally:
        engine.dispose()
    return df


def main() -> None:
    df = load_question_dataframe()
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_PATH, index=False)
    logger.info("Loaded %d questions from SQLite into %s", len(df), INTERIM_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
