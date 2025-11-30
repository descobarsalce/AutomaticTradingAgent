
import os
from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint, create_engine
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from utils.db_config import db_config

Base = declarative_base()


class StockData(Base):
    __tablename__ = 'stock_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if value is None and key not in ['id', 'last_updated']:
                raise ValueError(f"Column {key} cannot be null")
            setattr(self, key, value)

    __table_args__ = (UniqueConstraint('symbol',
                                       'date',
                                       name='uix_symbol_date'), )

    def __repr__(self):
        return f"<StockData(symbol='{self.symbol}', date='{self.date}')>"


class OptionsData(Base):
    __tablename__ = 'options_data'

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    contract_symbol = Column(String(30), nullable=False, unique=True, index=True)
    option_type = Column(String(4), nullable=False, index=True)  # 'call' or 'put'
    expiration = Column(DateTime, nullable=False, index=True)
    strike = Column(Float, nullable=False)
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    change = Column(Float)
    percent_change = Column(Float)
    volume = Column(Float)
    open_interest = Column(Float)
    implied_volatility = Column(Float)
    in_the_money = Column(String(5))  # 'True' or 'False'
    contract_size = Column(String(10))
    currency = Column(String(5))
    last_trade_date = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<OptionsData(contract='{self.contract_symbol}', type='{self.option_type}', strike={self.strike})>"


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
