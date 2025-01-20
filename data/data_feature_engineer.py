import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def validate_numeric(value: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
    return isinstance(value, (int, float)) and (min_value is None or value >= min_value) and (max_value is None or value <= max_value)

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and all(col in df.columns for col in required_columns)

def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
    return isinstance(weights, dict) and all(isinstance(w, (int, float)) and 0 <= w <= 1 for w in weights.values()) and abs(sum(weights.values()) - 1.0) < 1e-6

def normalize_data(data: pd.Series) -> pd.Series:
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val) if max_val > min_val else pd.Series(0, index=data.index)

class FeatureEngineer:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_data(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data with technical indicators and correlations"""
        if not portfolio_data:
            raise ValueError("No data available. Please fetch data first.")
        
        if not isinstance(portfolio_data, dict):
            raise TypeError("Portfolio data must be a dictionary of DataFrames")
            
        prepared_data = {}
        for symbol, data in portfolio_data.items():
            try:
                if not validate_dataframe(data, ['Close']):
                    logger.error(f"Invalid data format for {symbol}")
                    continue
                
                prepared_df = data.copy()
                
                if len(prepared_df) >= 50:
                    # Calculate technical indicators with error handling
                    try:
                        prepared_df['SMA_20'] = prepared_df['Close'].rolling(window=20, min_periods=20).mean()
                        prepared_df['SMA_50'] = prepared_df['Close'].rolling(window=50, min_periods=50).mean()
                        prepared_df['RSI'] = self.calculate_rsi(prepared_df['Close'])
                        prepared_df['Volatility'] = prepared_df['Close'].pct_change().rolling(window=20, min_periods=20).std()
                        
                        # Calculate correlations with error handling
                        correlations = {}
                        for other_symbol, other_data in portfolio_data.items():
                            if other_symbol != symbol and validate_dataframe(other_data, ['Close']):
                                correlations[other_symbol] = prepared_df['Close'].corr(other_data['Close'])
                        prepared_df['Correlations'] = str(correlations)
                        
                        # Handle missing values
                        prepared_df = prepared_df.dropna()
                        
                        if not prepared_df.empty:
                            prepared_data[symbol] = prepared_df
                        else:
                            logger.warning(f"No valid data points remaining for {symbol} after calculations")
                    except Exception as e:
                        logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                        prepared_data[symbol] = data
                else:
                    logger.warning(f"Insufficient data points for {symbol} (minimum 50 required)")
                    prepared_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"Error preparing data for {symbol}: {str(e)}")
                prepared_data[symbol] = data
                
        return prepared_data
