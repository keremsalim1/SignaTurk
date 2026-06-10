import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL", "") or "sqlite:///local_dev.db"


def _engine_kwargs(url: str):
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def configure_database(url: str):
    """Rebind the SQLAlchemy engine/sessionmaker at runtime."""
    global DATABASE_URL, engine
    try:
        engine.dispose()
    except Exception:
        pass
    DATABASE_URL = url
    engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
    SessionLocal.configure(bind=engine)
    return engine


def get_db():
    """FastAPI dependency: yields a DB session, auto-closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
