
from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint
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

    __table_args__ = (UniqueConstraint('symbol', 'date', name='uix_symbol_date'),)

    def __repr__(self):
        return f"<StockData(symbol='{self.symbol}', date='{self.date}')>"

def init_db():
    Base.metadata.create_all(db_config.engine)

init_db()
