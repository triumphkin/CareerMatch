"""Shared engine and session factory; each caller owns its own session."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    hide_parameters=True,
    connect_args={"connect_timeout": 5},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Install vector before creating tables; propagate initialization failures."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    Base.metadata.create_all(bind=engine)
