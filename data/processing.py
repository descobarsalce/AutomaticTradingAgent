import pandas as pd
import numpy as np

class FeatureEngineer:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_data(self, portfolio_data: dict) -> dict:
        """Prepare data with technical indicators and correlations"""
        if not portfolio_data:
            raise ValueError("No data available. Please fetch data first.")
            
        prepared_data = {}
        for symbol, data in portfolio_data.items():
            try:
                # Create a copy of the data to avoid modifying the original
                prepared_df = data.copy()
                
                # Calculate technical indicators
                if len(prepared_df) >= 50:  # Ensure enough data points for all indicators
                    prepared_df['SMA_20'] = prepared_df['Close'].rolling(window=20, min_periods=20).mean()
                    prepared_df['SMA_50'] = prepared_df['Close'].rolling(window=50, min_periods=50).mean()
                    prepared_df['RSI'] = self.calculate_rsi(prepared_df['Close'])
                    prepared_df['Volatility'] = prepared_df['Close'].pct_change().rolling(window=20, min_periods=20).std()
                    
                    # Calculate correlation with other stocks
                    correlations = {}
                    for other_symbol, other_data in portfolio_data.items():
                        if other_symbol != symbol:
                            correlations[other_symbol] = prepared_df['Close'].corr(other_data['Close'])
                    prepared_df['Correlations'] = str(correlations)
                    
                    # Remove NaN values
                    prepared_df = prepared_df.dropna()
                    
                    if not prepared_df.empty:
                        prepared_data[symbol] = prepared_df
                    else:
                        print(f"Warning: No valid data points remaining for {symbol} after calculations")
                else:
                    print(f"Warning: Insufficient data points for {symbol} (minimum 50 required)")
                    prepared_data[symbol] = data  # Use original data if insufficient points
                    
            except Exception as e:
                print(f"Error preparing data for {symbol}: {str(e)}")
                prepared_data[symbol] = data  # Use original data if preparation fails
                
        return prepared_data
