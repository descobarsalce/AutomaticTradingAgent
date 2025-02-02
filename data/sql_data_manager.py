from datetime import datetime, timedelta
from sqlalchemy import and_
import pandas as pd
from models.database import Session, StockData


class SQLDataManager:

    def __init__(self):
        self.session = Session()
