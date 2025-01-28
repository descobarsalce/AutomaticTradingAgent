"""
Centralized database configuration module.
Provides unified database connection handling and settings.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional

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
        """Initialize database connection"""
        try:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///trading_data.db')
            self._engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=300
            )
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
        """Get SQLAlchemy engine instance"""
        return self._engine

    @property
    def SessionLocal(self):
        """Get SQLAlchemy session factory"""
        return self._SessionLocal

    def get_session(self) -> Optional[Session]:
        """Create a new database session"""
        if self._SessionLocal is None:
            return None
        return self._SessionLocal()

# Global instance
db_config = DatabaseConfig()

def get_db_session() -> Session:
    """Utility function to get a database session"""
    return db_config.get_session()
