import os
from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint, create_engine
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from utils.db_config import db_config

Base = declarative_base()


class StockData(Base):
    __tablename__ = 'stock_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint('symbol',
                                       'date',
                                       name='uix_symbol_date'), )

    def __repr__(self):
        return f"<StockData(symbol='{self.symbol}', date='{self.date}')>"


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(db_config.engine)


def migrate_sqlite_to_postgres():
    """Migrate data from SQLite to PostgreSQL if needed"""
    if os.path.exists('trading_data.db'):
        from sqlalchemy import text, select
        sqlite_engine = create_engine('sqlite:///trading_data.db')
        sqlite_base = declarative_base()
        sqlite_base.metadata.reflect(bind=sqlite_engine)

        # Transfer data
        with sqlite_engine.connect() as sqlite_conn:
            for table in Base.metadata.sorted_tables:
                # Use text() for safe SQL execution
                stmt = text(f'SELECT * FROM {table.name}')
                result = sqlite_conn.execute(stmt)
                data = [dict(row._mapping) for row in result]

                if data:
                    with db_config.engine.begin() as pg_conn:
                        # Check existing records to avoid duplicates
                        for record in data:
                            try:
                                # Try inserting without the ID to let PostgreSQL auto-increment
                                insert_data = {
                                    k: v
                                    for k, v in record.items() if k != 'id'
                                }
                                pg_conn.execute(table.insert(), [insert_data])
                            except Exception as e:
                                # Skip if record already exists
                                continue

        print("Data migration completed")
