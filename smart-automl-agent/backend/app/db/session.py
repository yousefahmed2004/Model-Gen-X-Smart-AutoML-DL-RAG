"""
SQLAlchemy engine + session factory.

We default to SQLite for the graduation demo. Switching to PostgreSQL
just requires changing DATABASE_URL — the rest of the code is engine-agnostic.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


# SQLite needs `check_same_thread=False` because FastAPI may use the
# session across threads (BackgroundTasks, dependency lifecycle, etc).
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""
    pass


def get_db():
    """FastAPI dependency that yields a database session and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. For the demo we use this instead of Alembic migrations."""
    # Import models so they register on Base.metadata before create_all.
    from app.models import user, project, dataset, ml_model, token_ledger, chat  # noqa: F401
    Base.metadata.create_all(bind=engine)
