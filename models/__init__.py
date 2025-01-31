"""
Database models and initialization
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from .database import Base, Session, StockData

# Initialize database
engine = create_engine('sqlite:///trading_data.db', echo=True)
Base.metadata.create_all(engine)