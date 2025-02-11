
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
            # Default to SQLite if no PostgreSQL connection is available
            database_url = 'sqlite:///trading_data.db'
            engine_kwargs = {
                'pool_pre_ping': True,
                'pool_recycle': 300
            }
            
            if os.getenv('DATABASE_URL'):
                try:
                    import psycopg2
                    database_url = os.getenv('DATABASE_URL')
                except ImportError:
                    logger.warning("PostgreSQL driver not found, using SQLite")
                    engine_kwargs['connect_args'] = {'check_same_thread': False}
            else:
                engine_kwargs['connect_args'] = {'check_same_thread': False}
            
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
