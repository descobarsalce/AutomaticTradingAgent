from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

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
    
    __table_args__ = (
        # Composite unique constraint on symbol and date
        sqlalchemy.UniqueConstraint('symbol', 'date', name='uix_symbol_date'),
    )

# Database setup
engine = create_engine('sqlite:///trading_data.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
