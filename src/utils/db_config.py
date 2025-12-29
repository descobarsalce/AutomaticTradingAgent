
import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, Generator

logger = logging.getLogger(__name__)

# Set SQL logging verbosity
SQL_LOGGING_ENABLED = os.getenv('SQL_LOGGING_ENABLED', 'false').lower() == 'true'
if not SQL_LOGGING_ENABLED:
    logging.getLogger('utils.db_config').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('data.data_SQL_interaction').setLevel(logging.WARNING)

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
        start_time = datetime.now()
        logger.info("🔄 Starting database initialization...")
        try:
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                logger.info("Found DATABASE_URL in environment")
                # Hide sensitive info while logging
                safe_url = database_url.split('@')[-1] if '@' in database_url else 'db-connection'
                logger.info(f"Attempting to connect to: {safe_url}")
            else:
                logger.warning("DATABASE_URL not set, using SQLite fallback")
                database_url = 'sqlite:///trading_data.db'
                
            logger.info("🔑 Configuring database connection parameters...")
            engine_kwargs = {
                'pool_pre_ping': True,
                'pool_recycle': 60,  # More frequent connection recycling
                'pool_size': 5,
                'max_overflow': 10,
                'pool_timeout': 30,
                'echo': False,  # Reduce logging noise
            }

            if not database_url:
                logger.warning("⚠️ No PostgreSQL connection found, falling back to SQLite")
                database_url = 'sqlite:///trading_data.db'

            if database_url.startswith('sqlite'):
                # SQLite only accepts limited connect_args
                engine_kwargs['connect_args'] = {'check_same_thread': False}
                logger.info("📁 Using SQLite database at: trading_data.db")
            else:
                # Use connection pooling for PostgreSQL and pass only relevant connect_args
                engine_kwargs['connect_args'] = {
                    'connect_timeout': 10,
                    'application_name': 'trading_app',
                    'keepalives': 1,
                    'keepalives_idle': 30,
                    'keepalives_interval': 10,
                    'keepalives_count': 5,
                    'sslmode': 'require'
                }
                database_url = database_url.replace('.us-east-2', '-pooler.us-east-2')
                logger.info("🐘 Configuring PostgreSQL connection with pooling")
                logger.info(f"📊 Pool configuration - Size: {engine_kwargs['pool_size']}, "
                          f"Max Overflow: {engine_kwargs['max_overflow']}, "
                          f"Timeout: {engine_kwargs['pool_timeout']}s")
            
            logger.info("🔌 Creating database engine...")
            self._engine = create_engine(database_url, **engine_kwargs)
            logger.info("✅ Database engine created successfully")
            
            logger.info("🔧 Configuring session factory...")
            self._SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine
            )
            logger.info("✅ Session factory configured successfully")
            
            # Test connection
            with self._engine.connect() as conn:
                logger.info("🔍 Testing database connection...")
                conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection test successful")
            
            end_time = datetime.now()
            init_time = (end_time - start_time).total_seconds()
            logger.info(f"✨ Database initialization completed in {init_time:.2f} seconds")
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
        retries = 3
        for attempt in range(retries):
            try:
                db = self._SessionLocal()
                # Test connection
                db.execute(text("SELECT 1"))
                yield db
                break
            except Exception as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {str(e)}")
                if db:
                    db.close()
                if attempt == retries - 1:
                    raise
            finally:
                if 'db' in locals():
                    db.close()

# Global instance
db_config = DatabaseConfig()

def get_db_session() -> Generator[Session, None, None]:
    return db_config.get_db()
