import numpy as np
import logging
from typing import Optional, List, Dict, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(ch)

def calculate_returns(portfolio_history: List[float]) -> np.ndarray:
def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
    """
    Calculate return volatility (standard deviation) with optional annualization.
    
    Args:
        returns: numpy.ndarray of return values
        annualize: bool, whether to annualize the volatility
        
    Returns:
        float: Volatility value or 0.0 if calculation fails
    """
    if not isinstance(returns, np.ndarray):
        logger.warning("Invalid input type for volatility calculation")
        return 0.0
        
    if len(returns) <= 1:
        logger.debug("Insufficient data points for volatility calculation")
        return 0.0
        
    try:
        # Remove any non-finite values
        valid_returns = returns[np.isfinite(returns)]
        if len(valid_returns) <= 1:
            logger.debug("Insufficient valid return values for volatility calculation")
            return 0.0
            
        # Calculate standard deviation
        vol = np.std(valid_returns, ddof=1)
        
        # Annualize if requested (assuming daily data)
        if annualize:
            vol = vol * np.sqrt(252)
            
        return float(vol)
        
    except Exception as e:
        logger.exception("Error calculating volatility")
        return 0.0

    """
    Calculate returns from portfolio history with improved error handling
    and validation.
    
    Returns:
        numpy.ndarray: Array of returns or empty array if calculation fails
    """
    if len(portfolio_history) <= 1:
        logger.debug("Insufficient data points for returns calculation")
        return np.array([])
        
    try:
        # Validate portfolio values
        portfolio_array = np.array(portfolio_history)
        if not np.all(np.isfinite(portfolio_array)):
            logger.warning("Non-finite values found in portfolio history")
            portfolio_array = portfolio_array[np.isfinite(portfolio_array)]
        
        if len(portfolio_array) <= 1:
            logger.debug("Insufficient valid data points after filtering")
            return np.array([])
            
        # Calculate returns with validation
        denominator = portfolio_array[:-1]
        if np.any(denominator == 0):
            logger.warning("Zero values found in portfolio history")
            return np.array([])
            
        returns = np.diff(portfolio_array) / denominator
        
        # Filter out extreme values
        returns = returns[np.isfinite(returns)]
        if len(returns) > 0:
            # Remove extreme outliers (beyond 5 standard deviations)
            mean, std = np.mean(returns), np.std(returns)
            returns = returns[np.abs(returns - mean) <= 5 * std]
            logger.debug(f"Calculated returns with {len(returns)} valid data points")
            
        return returns
        
    except Exception as e:
        logger.exception("Error calculating returns")
        return np.array([])

def calculate_sharpe_ratio(returns: np.ndarray) -> float:
    """
    Calculate Sharpe ratio from returns with improved error handling
    and validation.
    
    Args:
        returns: numpy.ndarray of return values
        
    Returns:
        float: Annualized Sharpe ratio or 0.0 if calculation fails
    """
    if not isinstance(returns, np.ndarray):
        logger.warning("Invalid input type for returns calculation")
        return 0.0
        
    if len(returns) <= 252:  # Minimum one year of data for meaningful Sharpe ratio
        logger.debug(f"Insufficient data points for reliable Sharpe ratio: {len(returns)}")
        return 0.0
        
    try:
        # Remove any remaining non-finite values
        valid_returns = returns[np.isfinite(returns)]
        if len(valid_returns) <= 1:
            logger.debug("Insufficient valid return values for Sharpe ratio calculation")
            return 0.0
            
        # Calculate with improved precision
        avg_return = np.mean(valid_returns)
        std_return = np.std(valid_returns, ddof=1)  # Use unbiased estimator
        
        # Check for numerical stability
        if not np.isfinite(avg_return) or not np.isfinite(std_return):
            logger.warning("Non-finite values in Sharpe ratio calculation")
            return 0.0
            
        # Calculate annualized Sharpe ratio with validation
        if std_return > 1e-8:  # Avoid division by very small numbers
            annualization_factor = np.sqrt(252)  # Assuming daily returns
            sharpe = (avg_return / std_return) * annualization_factor
            sharpe_clipped = float(np.clip(sharpe, -100, 100))  # Limit extreme values
            logger.debug(f"Calculated Sharpe ratio: {sharpe_clipped}")
            return sharpe_clipped
        else:
            logger.warning("Standard deviation too small for reliable Sharpe ratio")
            return 0.0
            
    except Exception as e:
        logger.exception("Error calculating Sharpe ratio")
        return 0.0

def calculate_sortino_ratio(returns: np.ndarray) -> float:
    """
    Calculate Sortino ratio from returns with error handling and validation.
    Similar to Sharpe ratio but only penalizes downside volatility.
    
    Args:
        returns: numpy.ndarray of return values
        
    Returns:
        float: Annualized Sortino ratio or 0.0 if calculation fails
    """
    if not isinstance(returns, np.ndarray):
        logger.warning("Invalid input type for Sortino ratio calculation")
        return 0.0
        
    if len(returns) <= 252:  # Minimum one year of data for meaningful ratio
        logger.debug(f"Insufficient data points for reliable Sortino ratio: {len(returns)}")
        return 0.0
        
    try:
        # Remove any non-finite values
        valid_returns = returns[np.isfinite(returns)]
        if len(valid_returns) <= 1:
            logger.debug("Insufficient valid return values for Sortino ratio calculation")
            return 0.0
            
        # Calculate average return
        avg_return = np.mean(valid_returns)
        
        # Calculate downside deviation (only negative returns)
        negative_returns = valid_returns[valid_returns < 0]
        if len(negative_returns) == 0:
            logger.debug("No negative returns found for Sortino ratio calculation")
            return float('inf') if avg_return > 0 else 0.0
            
        downside_std = np.std(negative_returns, ddof=1)
        
        # Check for numerical stability
        if not np.isfinite(avg_return) or not np.isfinite(downside_std):
            logger.warning("Non-finite values in Sortino ratio calculation")
            return 0.0
            
        # Calculate annualized Sortino ratio with validation
        if downside_std > 1e-8:  # Avoid division by very small numbers
            annualization_factor = np.sqrt(252)  # Assuming daily returns
            sortino = (avg_return / downside_std) * annualization_factor
            sortino_clipped = float(np.clip(sortino, -100, 100))  # Limit extreme values
            logger.debug(f"Calculated Sortino ratio: {sortino_clipped}")
            return sortino_clipped
        else:
            logger.warning("Downside deviation too small for reliable Sortino ratio")
            return 0.0
            
    except Exception as e:
        logger.exception("Error calculating Sortino ratio")
        return 0.0

def calculate_information_ratio(returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> float:
    """
    Calculate Information ratio from returns with error handling and validation.
    Measures risk-adjusted excess returns relative to a benchmark.
    
    Args:
        returns: numpy.ndarray of return values
        benchmark_returns: Optional numpy.ndarray of benchmark return values
        
    Returns:
        float: Information ratio or 0.0 if calculation fails
    """
    if not isinstance(returns, np.ndarray):
        logger.warning("Invalid input type for Information ratio calculation")
        return 0.0
        
    if len(returns) <= 252:  # Minimum one year of data
        logger.debug(f"Insufficient data points for reliable Information ratio: {len(returns)}")
        return 0.0
        
    try:
        # If no benchmark provided, use risk-free rate of 0
        if benchmark_returns is None:
            benchmark_returns = np.zeros_like(returns)
        
        # Ensure arrays are the same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # Calculate excess returns
        excess_returns = returns - benchmark_returns
        
        # Remove any non-finite values
        valid_returns = excess_returns[np.isfinite(excess_returns)]
        if len(valid_returns) <= 1:
            logger.debug("Insufficient valid return values for Information ratio calculation")
            return 0.0
            
        # Calculate average excess return and tracking error
        avg_excess_return = np.mean(valid_returns)
        tracking_error = np.std(valid_returns, ddof=1)
        
        # Check for numerical stability
        if not np.isfinite(avg_excess_return) or not np.isfinite(tracking_error):
            logger.warning("Non-finite values in Information ratio calculation")
            return 0.0
            
        # Calculate Information ratio with validation
        if tracking_error > 1e-8:  # Avoid division by very small numbers
            information_ratio = avg_excess_return / tracking_error
            information_ratio_clipped = float(np.clip(information_ratio, -100, 100))
            logger.debug(f"Calculated Information ratio: {information_ratio_clipped}")
            return information_ratio_clipped
        else:
            logger.warning("Tracking error too small for reliable Information ratio")
            return 0.0
            
    except Exception as e:
        logger.exception("Error calculating Information ratio")
        return 0.0

def calculate_maximum_drawdown(portfolio_history: List[float]) -> float:
    """
    Calculate maximum drawdown from portfolio history with improved validation
    and error handling.
    
    Returns:
        float: Maximum drawdown value between 0.0 and 1.0
    """
    if len(portfolio_history) <= 1:
        logger.debug("Insufficient data points for drawdown calculation")
        return 0.0
    try:
        # Convert to numpy array for efficient calculation
        values = np.array([v for v in portfolio_history if isinstance(v, (int, float)) and v >= 0])
        if len(values) <= 1:
            logger.warning("No valid values for drawdown calculation after filtering")
            return 0.0
            
        # Calculate running maximum
        peak = np.maximum.accumulate(values)
        # Calculate drawdown for each point
        drawdowns = (peak - values) / peak
        # Get maximum drawdown
        max_dd = float(np.nanmax(drawdowns))
        
        logger.debug(f"Calculated maximum drawdown: {max_dd:.4f}")
        return max_dd
        
    except Exception as e:
        logger.exception("Error calculating maximum drawdown")
        return 0.0
