"""Shared, reproducible SQLite connection settings for the Phase 2 pipeline."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATABASE_PATH = DATA_DIR / "stackoverflow_questions.db"
ACTIVE_DATABASE_POINTER = DATA_DIR / "active_database.txt"
SOURCE_CSV_PATH = PROJECT_ROOT.parent.parent / "Phase 1" / "stackoverflow_questions.csv"


def get_database_path() -> Path:
    """Return the current database, including a fallback built when the default is locked."""
    if ACTIVE_DATABASE_POINTER.exists():
        candidate = DATA_DIR / ACTIVE_DATABASE_POINTER.read_text(encoding="utf-8").strip()
        if candidate.exists():
            return candidate
    return DEFAULT_DATABASE_PATH


def set_active_database_path(database_path: Path) -> None:
    """Persist the filename that all pipeline scripts should use for this run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_DATABASE_POINTER.write_text(database_path.name, encoding="utf-8")


def get_engine(database_path: Path | None = None) -> Engine:
    """Return a SQLAlchemy engine for the active local SQLite database."""
    path = database_path or get_database_path()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path.as_posix()}", future=True)
