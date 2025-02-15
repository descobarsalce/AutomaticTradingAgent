
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Generator

logger = logging.getLogger(__name__)

class DatabaseConfig:
    _instance = None
    _engine = None
    _SessionLocal = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._initialize_db()

    def _initialize_db(self) -> None:
        try:
            database_url = os.getenv('DATABASE_URL')
            engine_kwargs = {
                'pool_pre_ping': True,
                'pool_recycle': 300,
                'pool_size': 20,
                'max_overflow': 40,
                'pool_timeout': 30,
                'echo': False
            }
            
            if not database_url:
                logger.warning("No PostgreSQL connection found, falling back to SQLite")
                database_url = 'sqlite:///trading_data.db'
                engine_kwargs['connect_args'] = {'check_same_thread': False}
            else:
                # Use connection pooling for PostgreSQL
                database_url = database_url.replace('.us-east-2', '-pooler.us-east-2')
            
            self._engine = create_engine(database_url, **engine_kwargs)
            self._SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine
            )
            logger.info("Database configuration initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise

    @property
    def engine(self):
        return self._engine

    @property
    def SessionLocal(self):
        return self._SessionLocal

    def get_db(self) -> Generator[Session, None, None]:
        db = self._SessionLocal()
        try:
            yield db
        finally:
            db.close()

# Global instance
db_config = DatabaseConfig()

def get_db_session() -> Generator[Session, None, None]:
    return db_config.get_db()
