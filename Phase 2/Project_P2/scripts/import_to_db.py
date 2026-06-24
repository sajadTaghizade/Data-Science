"""Rebuild the normalized SQLite database from the crawled CSV source."""

from __future__ import annotations

import ast
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

try:  # Supports both `python scripts/import_to_db.py` and `from scripts.import_to_db import ...`.
    from .database_connection import DEFAULT_DATABASE_PATH, SOURCE_CSV_PATH, get_engine, set_active_database_path
except ImportError:  # pragma: no cover - direct script execution
    from database_connection import DEFAULT_DATABASE_PATH, SOURCE_CSV_PATH, get_engine, set_active_database_path


logger = logging.getLogger(__name__)


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    account_id INTEGER UNIQUE,
    reputation INTEGER,
    user_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    accept_rate REAL CHECK (accept_rate BETWEEN 0 AND 100),
    profile_image_url TEXT,
    profile_url TEXT
);

CREATE TABLE questions (
    question_id INTEGER PRIMARY KEY,
    owner_user_id INTEGER,
    title TEXT NOT NULL,
    body_html TEXT NOT NULL,
    question_url TEXT NOT NULL UNIQUE,
    content_license TEXT,
    is_answered INTEGER NOT NULL CHECK (is_answered IN (0, 1)),
    view_count INTEGER NOT NULL CHECK (view_count >= 0),
    answer_count INTEGER NOT NULL CHECK (answer_count >= 0),
    score INTEGER NOT NULL,
    accepted_answer_id INTEGER,
    creation_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    last_edit_at TEXT,
    closed_at TEXT,
    closed_reason TEXT,
    bounty_amount INTEGER CHECK (bounty_amount >= 0),
    bounty_closes_at TEXT,
    protected_at TEXT,
    community_owned_at TEXT,
    locked_at TEXT,
    FOREIGN KEY (owner_user_id) REFERENCES users(user_id)
        ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE question_tags (
    question_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (question_id, tag_id),
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX idx_questions_owner_user_id ON questions(owner_user_id);
CREATE INDEX idx_questions_creation_at ON questions(creation_at);
CREATE INDEX idx_questions_score ON questions(score DESC);
CREATE INDEX idx_question_tags_tag_id ON question_tags(tag_id);
"""


def none_if_missing(value: Any) -> Any:
    return None if pd.isna(value) else value


def nullable_int(value: Any) -> int | None:
    value = none_if_missing(value)
    return None if value is None else int(value)


def epoch_to_utc(value: Any) -> str | None:
    value = nullable_int(value)
    if value is None:
        return None
    return pd.to_datetime(value, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_tags(raw_tags: str) -> list[str]:
    try:
        tags = ast.literal_eval(raw_tags)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid tag list: {raw_tags!r}") from exc
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        raise ValueError(f"Tags must be a non-empty list of strings: {raw_tags!r}")
    return [tag.strip() for tag in tags]


def choose_database_path() -> Path:
    """Prefer the canonical path; use one stable fallback only if it is locked."""
    DEFAULT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_DATABASE_PATH.exists():
        return DEFAULT_DATABASE_PATH
    try:
        DEFAULT_DATABASE_PATH.unlink()
        return DEFAULT_DATABASE_PATH
    except PermissionError:
        fallback = DEFAULT_DATABASE_PATH.with_name("stackoverflow_questions_active.db")
        try:
            if fallback.exists():
                fallback.unlink()
        except PermissionError:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            fallback = fallback.with_name(f"stackoverflow_questions_active_{timestamp}.db")
        logger.warning(
            "%s is open in another application; building %s for this run.",
            DEFAULT_DATABASE_PATH.name, fallback.name,
        )
        return fallback


def build_database() -> tuple[Path, dict[str, int]]:
    """Validate the source and atomically populate normalized relational tables."""
    if not SOURCE_CSV_PATH.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV_PATH}")

    df = pd.read_csv(SOURCE_CSV_PATH)
    required_columns = {
        "question_id", "title", "body", "link", "tags", "is_answered",
        "view_count", "answer_count", "score", "creation_date", "last_activity_date",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    if df["question_id"].isna().any() or df["question_id"].duplicated().any():
        raise ValueError("question_id must be present and unique in the source CSV.")
    if df[["title", "body", "link", "tags"]].isna().any().any():
        raise ValueError("Required text fields contain missing values in the source CSV.")

    users: dict[int, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    tag_names: set[str] = set()
    question_tag_names: list[tuple[int, str]] = []
    optional_timestamps = {
        "last_edit_at": "last_edit_date",
        "closed_at": "closed_date",
        "bounty_closes_at": "bounty_closes_date",
        "protected_at": "protected_date",
        "community_owned_at": "community_owned_date",
        "locked_at": "locked_date",
    }

    for row in df.to_dict(orient="records"):
        owner_user_id = nullable_int(row["owner.user_id"])
        if owner_user_id is not None:
            users[owner_user_id] = {
                "user_id": owner_user_id,
                "account_id": nullable_int(row["owner.account_id"]),
                "reputation": nullable_int(row["owner.reputation"]),
                "user_type": str(row["owner.user_type"]),
                "display_name": str(row["owner.display_name"]),
                "accept_rate": none_if_missing(row["owner.accept_rate"]),
                "profile_image_url": none_if_missing(row["owner.profile_image"]),
                "profile_url": none_if_missing(row["owner.link"]),
            }

        question_id = int(row["question_id"])
        question = {
            "question_id": question_id,
            "owner_user_id": owner_user_id,
            "title": str(row["title"]),
            "body_html": str(row["body"]),
            "question_url": str(row["link"]),
            "content_license": none_if_missing(row["content_license"]),
            "is_answered": int(bool(row["is_answered"])),
            "view_count": int(row["view_count"]),
            "answer_count": int(row["answer_count"]),
            "score": int(row["score"]),
            "accepted_answer_id": nullable_int(row["accepted_answer_id"]),
            "creation_at": epoch_to_utc(row["creation_date"]),
            "last_activity_at": epoch_to_utc(row["last_activity_date"]),
            "last_edit_at": None,
            "closed_at": None,
            "closed_reason": none_if_missing(row["closed_reason"]),
            "bounty_amount": nullable_int(row["bounty_amount"]),
            "bounty_closes_at": None,
            "protected_at": None,
            "community_owned_at": None,
            "locked_at": None,
        }
        for target, source in optional_timestamps.items():
            question[target] = epoch_to_utc(row[source])
        questions.append(question)

        for tag in parse_tags(str(row["tags"])):
            tag_names.add(tag)
            question_tag_names.append((question_id, tag))

    database_path = choose_database_path()

    with sqlite3.connect(database_path) as connection:
        connection.executescript(SCHEMA_SQL)

    engine = get_engine(database_path)
    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys = ON"))
            connection.execute(
                text("""
                    INSERT INTO users (
                        user_id, account_id, reputation, user_type, display_name,
                        accept_rate, profile_image_url, profile_url
                    ) VALUES (
                        :user_id, :account_id, :reputation, :user_type, :display_name,
                        :accept_rate, :profile_image_url, :profile_url
                    )
                """),
                list(users.values()),
            )
            connection.execute(
                text("INSERT INTO tags (name) VALUES (:name)"),
                [{"name": tag} for tag in sorted(tag_names, key=str.casefold)],
            )
            connection.execute(
                text("""
                    INSERT INTO questions (
                        question_id, owner_user_id, title, body_html, question_url,
                        content_license, is_answered, view_count, answer_count, score,
                        accepted_answer_id, creation_at, last_activity_at, last_edit_at,
                        closed_at, closed_reason, bounty_amount, bounty_closes_at,
                        protected_at, community_owned_at, locked_at
                    ) VALUES (
                        :question_id, :owner_user_id, :title, :body_html, :question_url,
                        :content_license, :is_answered, :view_count, :answer_count, :score,
                        :accepted_answer_id, :creation_at, :last_activity_at, :last_edit_at,
                        :closed_at, :closed_reason, :bounty_amount, :bounty_closes_at,
                        :protected_at, :community_owned_at, :locked_at
                    )
                """),
                questions,
            )
            tag_id_by_name = {
                name: tag_id
                for tag_id, name in connection.execute(text("SELECT tag_id, name FROM tags")).all()
            }
            connection.execute(
                text("INSERT INTO question_tags (question_id, tag_id) VALUES (:question_id, :tag_id)"),
                [
                    {"question_id": question_id, "tag_id": tag_id_by_name[tag_name]}
                    for question_id, tag_name in question_tag_names
                ],
            )
            counts = {
                table: connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
                for table in ("users", "questions", "tags", "question_tags")
            }
            set_active_database_path(database_path)
            return database_path, counts
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    database_path, result = build_database()
    logger.info("Created database: %s", database_path)
    logger.info("Imported %s.", ", ".join(f"{count:,} {table}" for table, count in result.items()))
