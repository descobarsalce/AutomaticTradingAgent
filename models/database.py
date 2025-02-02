
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create Base class first
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
    last_updated = Column(DateTime, default=datetime.now(datetime.UTC))

    __table_args__ = (UniqueConstraint('symbol', 'date', name='uix_symbol_date'),)

# Create engine and session factory
engine = create_engine('sqlite:///trading_data.db', echo=True)
DBSession = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)

# Initialize database tables
init_db()
