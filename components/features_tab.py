"""
Feature Selection Component
Allows users to configure which features are used for model training.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

logger = logging.getLogger(__name__)


# Feature descriptions with human-readable names and explanations
FEATURE_DESCRIPTIONS = {
    # Market Data Features - Price
    'price_change': {
        'name': 'Price Change ($)',
        'description': 'Absolute change in closing price from previous day',
        'category': 'Price',
        'example': '+2.50 or -1.75'
    },
    'returns': {
        'name': 'Daily Returns (%)',
        'description': 'Percentage change in price from previous day',
        'category': 'Price',
        'example': '0.025 (2.5%)'
    },
    'log_returns': {
        'name': 'Log Returns',
        'description': 'Natural log of price ratio (better for compounding)',
        'category': 'Price',
        'example': '0.0247'
    },
    'high_low_ratio': {
        'name': 'High/Low Ratio',
        'description': 'Ratio of daily high to low price (intraday volatility)',
        'category': 'Price',
        'example': '1.02 (2% intraday range)'
    },
    'close_open_ratio': {
        'name': 'Close/Open Ratio',
        'description': 'Ratio of close to open (daily direction)',
        'category': 'Price',
        'example': '1.01 (up day), 0.99 (down day)'
    },
    'typical_price': {
        'name': 'Typical Price',
        'description': 'Average of High, Low, Close prices',
        'category': 'Price',
        'example': '(H+L+C)/3 = 152.33'
    },
    'price_range': {
        'name': 'Daily Price Range ($)',
        'description': 'Difference between daily high and low',
        'category': 'Price',
        'example': '3.50 (High - Low)'
    },
    'gap': {
        'name': 'Overnight Gap ($)',
        'description': 'Difference between open and previous close',
        'category': 'Price',
        'example': '+1.20 (gap up), -0.80 (gap down)'
    },

    # Market Data Features - Volume
    'volume_change': {
        'name': 'Volume Change (%)',
        'description': 'Percentage change in volume from previous day',
        'category': 'Volume',
        'example': '0.50 (50% more volume)'
    },
    'volume_ma_ratio': {
        'name': 'Volume vs 20-day Avg',
        'description': 'Current volume relative to 20-day moving average',
        'category': 'Volume',
        'example': '1.5 (50% above average)'
    },

    # Market Data Features - Lagged
    'close_lag1': {
        'name': 'Price 1 Day Ago',
        'description': 'Closing price from 1 day ago',
        'category': 'Lagged Prices',
        'example': '150.25'
    },
    'close_lag2': {
        'name': 'Price 2 Days Ago',
        'description': 'Closing price from 2 days ago',
        'category': 'Lagged Prices',
        'example': '149.80'
    },
    'close_lag3': {
        'name': 'Price 3 Days Ago',
        'description': 'Closing price from 3 days ago',
        'category': 'Lagged Prices',
        'example': '148.50'
    },
    'close_lag4': {
        'name': 'Price 4 Days Ago',
        'description': 'Closing price from 4 days ago',
        'category': 'Lagged Prices',
        'example': '147.90'
    },
    'close_lag5': {
        'name': 'Price 5 Days Ago',
        'description': 'Closing price from 5 days ago',
        'category': 'Lagged Prices',
        'example': '146.75'
    },

    # Market Data Features - Rolling Statistics
    'rolling_mean_5': {
        'name': '5-Day Moving Avg',
        'description': 'Average closing price over last 5 days',
        'category': 'Moving Averages',
        'example': '151.20'
    },
    'rolling_mean_10': {
        'name': '10-Day Moving Avg',
        'description': 'Average closing price over last 10 days',
        'category': 'Moving Averages',
        'example': '149.85'
    },
    'rolling_mean_20': {
        'name': '20-Day Moving Avg',
        'description': 'Average closing price over last 20 days',
        'category': 'Moving Averages',
        'example': '148.50'
    },
    'rolling_std_5': {
        'name': '5-Day Std Dev',
        'description': 'Price standard deviation over last 5 days',
        'category': 'Rolling Stats',
        'example': '2.35'
    },
    'rolling_std_10': {
        'name': '10-Day Std Dev',
        'description': 'Price standard deviation over last 10 days',
        'category': 'Rolling Stats',
        'example': '3.12'
    },
    'rolling_std_20': {
        'name': '20-Day Std Dev',
        'description': 'Price standard deviation over last 20 days',
        'category': 'Rolling Stats',
        'example': '4.25'
    },
    'rolling_min_5': {
        'name': '5-Day Low',
        'description': 'Minimum closing price over last 5 days',
        'category': 'Rolling Stats',
        'example': '148.50'
    },
    'rolling_min_10': {
        'name': '10-Day Low',
        'description': 'Minimum closing price over last 10 days',
        'category': 'Rolling Stats',
        'example': '146.25'
    },
    'rolling_min_20': {
        'name': '20-Day Low',
        'description': 'Minimum closing price over last 20 days',
        'category': 'Rolling Stats',
        'example': '142.00'
    },
    'rolling_max_5': {
        'name': '5-Day High',
        'description': 'Maximum closing price over last 5 days',
        'category': 'Rolling Stats',
        'example': '153.75'
    },
    'rolling_max_10': {
        'name': '10-Day High',
        'description': 'Maximum closing price over last 10 days',
        'category': 'Rolling Stats',
        'example': '155.20'
    },
    'rolling_max_20': {
        'name': '20-Day High',
        'description': 'Maximum closing price over last 20 days',
        'category': 'Rolling Stats',
        'example': '158.00'
    },

    # Technical Indicators - Momentum
    'rsi': {
        'name': 'RSI (14-day)',
        'description': 'Relative Strength Index: momentum oscillator (0-100). >70 overbought, <30 oversold',
        'category': 'Momentum',
        'example': '65.5'
    },
    'macd': {
        'name': 'MACD Line',
        'description': 'Moving Average Convergence/Divergence (12-26 EMA difference)',
        'category': 'Momentum',
        'example': '2.35'
    },
    'macd_signal': {
        'name': 'MACD Signal Line',
        'description': '9-day EMA of MACD (for crossover signals)',
        'category': 'Momentum',
        'example': '1.85'
    },
    'macd_histogram': {
        'name': 'MACD Histogram',
        'description': 'Difference between MACD and Signal (momentum strength)',
        'category': 'Momentum',
        'example': '0.50'
    },
    'stoch_k': {
        'name': 'Stochastic %K',
        'description': 'Fast stochastic oscillator (0-100). Price position in range',
        'category': 'Momentum',
        'example': '78.5'
    },
    'stoch_d': {
        'name': 'Stochastic %D',
        'description': 'Slow stochastic (3-day SMA of %K). Smoother signal',
        'category': 'Momentum',
        'example': '72.3'
    },

    # Technical Indicators - Volatility
    'volatility': {
        'name': 'Volatility (20-day)',
        'description': 'Standard deviation of daily returns over 20 days',
        'category': 'Volatility',
        'example': '0.025 (2.5% daily)'
    },
    'bb_upper': {
        'name': 'Bollinger Upper Band',
        'description': '20-day SMA + 2 standard deviations',
        'category': 'Volatility',
        'example': '158.50'
    },
    'bb_lower': {
        'name': 'Bollinger Lower Band',
        'description': '20-day SMA - 2 standard deviations',
        'category': 'Volatility',
        'example': '142.30'
    },
    'bb_sma': {
        'name': 'Bollinger Middle (SMA)',
        'description': '20-day Simple Moving Average (BB center)',
        'category': 'Volatility',
        'example': '150.40'
    },
    'bb_width': {
        'name': 'Bollinger Band Width',
        'description': 'Width of bands relative to SMA (volatility measure)',
        'category': 'Volatility',
        'example': '0.108 (10.8%)'
    },
    'bb_percent': {
        'name': 'Bollinger %B',
        'description': 'Price position within bands (0=lower, 1=upper)',
        'category': 'Volatility',
        'example': '0.75 (75% up in band)'
    },

    # Technical Indicators - Volume
    'obv': {
        'name': 'On-Balance Volume',
        'description': 'Cumulative volume flow (+ on up days, - on down days)',
        'category': 'Volume Indicators',
        'example': '15,234,500'
    },
    'obv_ema': {
        'name': 'OBV 20-day EMA',
        'description': 'Smoothed On-Balance Volume trend',
        'category': 'Volume Indicators',
        'example': '14,850,200'
    },

    # Technical Indicators - Cycle
    'fft_1': {
        'name': 'FFT Component 1',
        'description': 'Primary frequency cycle amplitude from Fourier Transform',
        'category': 'Cycle Analysis',
        'example': '1250.5'
    },
    'fft_2': {
        'name': 'FFT Component 2',
        'description': 'Secondary frequency cycle amplitude',
        'category': 'Cycle Analysis',
        'example': '890.2'
    },
    'fft_3': {
        'name': 'FFT Component 3',
        'description': 'Tertiary frequency cycle amplitude',
        'category': 'Cycle Analysis',
        'example': '450.8'
    },
}

# Default feature configuration
DEFAULT_FEATURE_CONFIG = {
    'use_feature_engineering': True,
    'sources': {
        'MarketDataSource': {
            'enabled': True,
            'features': [
                'returns', 'log_returns', 'price_change',
                'high_low_ratio', 'volume_change'
            ]
        },
        'TechnicalSource': {
            'enabled': True,
            'features': [
                'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'volatility'
            ]
        }
    },
    'include_raw_prices': True,  # Include Open/Close prices in observation
    'include_positions': True,
    'include_balance': True,
    'normalize_features': True,
}


def get_feature_display_name(feature_key: str) -> str:
    """Get human-readable name for a feature."""
    if feature_key in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_key]['name']
    return feature_key.replace('_', ' ').title()


def get_feature_description(feature_key: str) -> str:
    """Get description for a feature."""
    if feature_key in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_key]['description']
    return ""


def get_feature_category(feature_key: str) -> str:
    """Get category for a feature."""
    if feature_key in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_key].get('category', 'Other')
    return 'Other'


def get_available_features() -> Dict[str, List[str]]:
    """Get all available features from registered sources."""
    try:
        from data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        features_by_source = {}
        for source_name in engineer.list_sources():
            features_by_source[source_name] = engineer.list_features(source_name)

        return features_by_source
    except Exception as e:
        logger.error(f"Error getting available features: {e}")
        return {
            'MarketDataSource': [
                'price_change', 'returns', 'log_returns', 'high_low_ratio',
                'close_open_ratio', 'typical_price', 'price_range', 'gap',
                'volume_change', 'volume_ma_ratio',
                'close_lag1', 'close_lag2', 'close_lag3', 'close_lag4', 'close_lag5',
                'rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5',
                'rolling_mean_10', 'rolling_std_10', 'rolling_min_10', 'rolling_max_10',
                'rolling_mean_20', 'rolling_std_20', 'rolling_min_20', 'rolling_max_20',
            ],
            'TechnicalSource': [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_sma', 'bb_width', 'bb_percent',
                'volatility', 'stoch_k', 'stoch_d', 'obv', 'obv_ema',
                'fft_1', 'fft_2', 'fft_3'
            ]
        }


def group_features_by_category(features: List[str]) -> Dict[str, List[str]]:
    """Group features by their category."""
    grouped = {}
    for feature in features:
        category = get_feature_category(feature)
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(feature)
    return grouped


def get_available_stocks() -> List[str]:
    """Get list of available stocks from the database.

    Returns:
        List of stock symbols available in the database
    """
    try:
        if 'data_handler' not in st.session_state:
            logger.warning("data_handler not in session state")
            return []

        from sqlalchemy import distinct
        from data.database import StockData

        symbols = [row[0] for row in st.session_state.data_handler.query(distinct(StockData.symbol)).all()]
        logger.info(f"Found {len(symbols)} stocks in database: {symbols}")
        return sorted(symbols)
    except Exception as e:
        logger.error(f"Error getting available stocks: {e}")
        return []


def compute_sample_features(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """Compute feature values for a specific symbol and date range.

    Args:
        symbol: Stock symbol
        start_date: Start of date range
        end_date: End of date range

    Returns:
        DataFrame with computed features for the date range
    """
    try:
        # Get data from data handler
        if 'data_handler' not in st.session_state:
            logger.error("data_handler not found in session state")
            return None

        # Fetch extra history for indicator computation (need lookback for indicators)
        fetch_start = start_date - timedelta(days=60)
        logger.info(f"Fetching data for {symbol} from {fetch_start} to {end_date}")

        # Use use_SQL=True to fetch from database
        data = st.session_state.data_handler.fetch_data([symbol], fetch_start, end_date, use_SQL=True)

        if data is None or data.empty:
            logger.error(f"No data returned for {symbol} in date range {fetch_start} to {end_date}")
            return None

        logger.info(f"Retrieved {len(data)} rows for {symbol}")
        logger.info(f"Data columns: {list(data.columns)}")
        logger.info(f"Data date range: {data.index.min()} to {data.index.max()}")

        # Compute features using the feature engineering system
        from data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        features_df = engineer.compute_features(
            data,
            symbols=[symbol],
            drop_na=False
        )

        if features_df is None or features_df.empty:
            logger.error(f"Feature computation returned empty DataFrame")
            return None

        logger.info(f"Computed {len(features_df.columns)} features")

        # Filter to requested date range
        if not features_df.empty:
            mask = (features_df.index >= start_date) & (features_df.index <= end_date)
            filtered_df = features_df.loc[mask]
            logger.info(f"Filtered to {len(filtered_df)} rows in requested date range")
            return filtered_df

        return None

    except Exception as e:
        logger.error(f"Error computing sample features: {e}", exc_info=True)
        return None


def build_feature_columns_for_symbol(
    sample_df: pd.DataFrame,
    symbol: str,
    selected_features: List[str],
    include_raw_prices: bool,
) -> List[str]:
    """Collect available feature columns for a symbol based on selection."""

    columns: List[str] = []

    if include_raw_prices:
        for base_col in ["Open", "High", "Low", "Close", "Volume"]:
            candidate = f"{base_col}_{symbol}"
            if candidate in sample_df.columns:
                columns.append(candidate)

    for feature in selected_features:
        candidate = f"{feature}_{symbol}"
        if candidate in sample_df.columns:
            columns.append(candidate)

    return columns


def generate_price_prediction_preview(
    sample_df: pd.DataFrame,
    symbol: str,
    feature_columns: List[str],
) -> Optional[Dict[str, Any]]:
    """Train a lightweight regression model to preview price predictions."""

    target_col = f"Close_{symbol}"
    if target_col not in sample_df.columns or not feature_columns:
        return None

    dataset = sample_df[[target_col, *feature_columns]].dropna().copy()
    dataset["target_next"] = dataset[target_col].shift(-1)
    dataset = dataset.dropna()

    if len(dataset) < 12:
        return None

    split_idx = int(len(dataset) * 0.8)
    X_train, X_test = dataset.iloc[:split_idx][feature_columns], dataset.iloc[split_idx:][feature_columns]
    y_train, y_test = dataset.iloc[:split_idx]["target_next"], dataset.iloc[split_idx:]["target_next"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    preview_history = pd.DataFrame(
        {
            "Date": dataset.index[split_idx:],
            "Actual Next Close": y_test.values,
            "Predicted Next Close": predictions,
        }
    ).set_index("Date")

    coef_frame = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Coefficient": model.coef_,
        }
    ).sort_values("Coefficient", key=lambda s: s.abs(), ascending=False)

    return {
        "mae": mae,
        "rmse": rmse,
        "history": preview_history,
    }


def initialize_feature_config():
    """Initialize feature configuration in session state."""
    if 'feature_config' not in st.session_state:
        st.session_state.feature_config = DEFAULT_FEATURE_CONFIG.copy()


def display_features_tab():
    """Renders the feature selection interface tab."""
    st.header("Feature Engineering Configuration")

    initialize_feature_config()
    config = st.session_state.feature_config

    # Main toggle
    st.subheader("Feature Engineering")
    use_fe = st.checkbox(
        "Enable Feature Engineering",
        value=config.get('use_feature_engineering', True),
        help="When enabled, computed features will be added to the observation space. "
             "When disabled, only raw OHLCV data is used."
    )
    config['use_feature_engineering'] = use_fe

    if not use_fe:
        st.info("Feature engineering is disabled. The model will use raw OHLCV data only.")
        st.session_state.feature_config = config
        return

    st.divider()

    st.markdown("### Observation Building Blocks")
    st.caption("Keep the core observation inputs together so you can quickly choose how the model sees the market.")

    choice_cols = st.columns(4)
    with choice_cols[0]:
        config['include_raw_prices'] = st.checkbox(
            "Include Raw Prices",
            value=config.get('include_raw_prices', True),
            help="Add Open/High/Low/Close/Volume so you can compare engineered signals against base prices."
        )
    with choice_cols[1]:
        config['include_positions'] = st.checkbox(
            "Include Current Positions",
            value=config.get('include_positions', True),
            help="Track current holdings alongside features."
        )
    with choice_cols[2]:
        config['include_balance'] = st.checkbox(
            "Include Cash Balance",
            value=config.get('include_balance', True),
            help="Expose available cash to downstream policies."
        )
    with choice_cols[3]:
        config['normalize_features'] = st.checkbox(
            "Normalize Features",
            value=config.get('normalize_features', True),
            help="Keep scales consistent by applying Z-score normalization."
        )

    st.divider()

    st.markdown("### Core Comparisons")
    st.caption("Pick the flavor of each common signal in one place so related knobs stay together.")

    comparison_groups = [
        {
            "title": "Price Change Lens",
            "source": "MarketDataSource",
            "options": [
                ("Absolute", "price_change"),
                ("% Returns", "returns"),
                ("Log Returns", "log_returns"),
            ],
            "help": "Choose how you want to express day-over-day movement.",
        },
        {
            "title": "Intraday Context",
            "source": "MarketDataSource",
            "options": [
                ("High vs Low", "high_low_ratio"),
                ("Close vs Open", "close_open_ratio"),
                ("Range ($)", "price_range"),
            ],
            "help": "Group the relative vs absolute intraday shape choices.",
        },
        {
            "title": "Volatility Wrapper",
            "source": "TechnicalSource",
            "options": [
                ("Bollinger %B", "bb_percent"),
                ("Band Width", "bb_width"),
                ("Std Dev (20d)", "volatility"),
            ],
            "help": "Keep the volatility selectors in a single row.",
        },
        {
            "title": "Momentum Flavor",
            "source": "TechnicalSource",
            "options": [
                ("RSI", "rsi"),
                ("MACD", "macd"),
                ("Stoch %K", "stoch_k"),
            ],
            "help": "Pick your primary momentum signal without digging through multiple sections.",
        },
    ]

    comparison_cols = st.columns(len(comparison_groups))
    for idx, group in enumerate(comparison_groups):
        if group["source"] not in config["sources"]:
            config["sources"][group["source"]] = {"enabled": True, "features": []}

        with comparison_cols[idx]:
            st.markdown(f"**{group['title']}**")
            existing = set(config["sources"][group["source"]].get("features", []))
            default_choice = next(
                (label for label, feature in group["options"] if feature in existing),
                group["options"][0][0],
            )
            chosen = st.radio(
                "",
                [label for label, _ in group["options"]],
                key=f"comparison_{group['title']}",
                help=group["help"],
                index=[label for label, _ in group["options"]].index(default_choice),
            )

            selected_option = next(
                feature for label, feature in group["options"] if label == chosen
            )

            for _, feature in group["options"]:
                if feature in existing and feature != selected_option:
                    existing.remove(feature)
            existing.add(selected_option)
            config["sources"][group["source"]]["features"] = list(existing)

    st.divider()

    # Get available features
    available_features = get_available_features()

    # Feature source selection with categories
    st.subheader("Feature Sources")

    # Quick summary row
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Sources Detected", len(available_features))
    with summary_cols[1]:
        st.metric(
            "Enabled Sources",
            sum(1 for src in config['sources'].values() if src.get('enabled', False))
        )
    with summary_cols[2]:
        total_selected = sum(len(src.get('features', [])) for src in config['sources'].values())
        st.metric("Selected Features", total_selected)

    for source_name, source_features in available_features.items():
        source_display = "Market Data Features" if source_name == "MarketDataSource" else "Technical Indicators"

        with st.expander(f"{source_display} ({len(source_features)} features)", expanded=True):
            # Source toggle
            source_enabled = st.checkbox(
                f"Enable {source_display}",
                value=config['sources'].get(source_name, {}).get('enabled', True),
                key=f"enable_{source_name}"
            )

            if source_name not in config['sources']:
                config['sources'][source_name] = {'enabled': True, 'features': []}

            config['sources'][source_name]['enabled'] = source_enabled

            if source_enabled:
                # Feature selection buttons
                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    if st.button("Select All", key=f"all_{source_name}"):
                        config['sources'][source_name]['features'] = source_features.copy()
                with col3:
                    if st.button("Clear All", key=f"clear_{source_name}"):
                        config['sources'][source_name]['features'] = []

                # Group features by category
                selected = set(config['sources'].get(source_name, {}).get('features', []))
                grouped = group_features_by_category(source_features)

                # Display features by category
                for category, cat_features in sorted(grouped.items()):
                    st.markdown(f"**{category}**")
                    cols = st.columns(2)

                    for idx, feature in enumerate(sorted(cat_features)):
                        display_name = get_feature_display_name(feature)
                        description = get_feature_description(feature)

                        with cols[idx % 2]:
                            is_selected = st.checkbox(
                                display_name,
                                value=feature in selected,
                                key=f"{source_name}_{feature}",
                                help=description
                            )
                            if is_selected:
                                selected.add(feature)
                            elif feature in selected:
                                selected.discard(feature)

                    st.write("")  # Spacing between categories

                config['sources'][source_name]['features'] = list(selected)
                st.caption(f"Selected: {len(selected)} / {len(source_features)} features")

    # Configuration Summary
    st.subheader("Configuration Summary")
    total_features = sum(
        len(src_config.get('features', []))
        for src_config in config['sources'].values()
        if src_config.get('enabled', False)
    )

    active_sources = [
        "Market Data" if name == "MarketDataSource" else "Technical"
        for name, src_config in config['sources'].items()
        if src_config.get('enabled', False)
    ]

    st.info(
        f"**Active Sources:** {', '.join(active_sources) if active_sources else 'None'}\n\n"
        f"**Total Features Selected:** {total_features}\n\n"
        f"**Include Raw Prices:** {'Yes' if config.get('include_raw_prices', True) else 'No'}\n\n"
        f"**Include Positions:** {'Yes' if config['include_positions'] else 'No'}\n\n"
        f"**Include Balance:** {'Yes' if config['include_balance'] else 'No'}\n\n"
        f"**Normalize:** {'Yes' if config['normalize_features'] else 'No'}"
    )

    # Save configuration
    st.session_state.feature_config = config

    st.divider()

    # Feature Preview Section
    st.subheader("Feature Preview")
    st.caption("See what feature values look like for a specific stock over a date range")

    # Get available stocks from database
    available_stocks = get_available_stocks()
    if not available_stocks:
        st.warning("No stocks found in database. Please add stock data in the Database Explorer tab first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        preview_symbol = st.selectbox(
            "Select Stock",
            options=available_stocks,
            key="preview_symbol"
        )
    with col2:
        preview_start = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now(),
            key="preview_start"
        )
    with col3:
        preview_end = st.date_input(
            "End Date",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now(),
            key="preview_end"
        )

    if st.button("Generate Feature Preview", type="primary"):
        if preview_start >= preview_end:
            st.error("Start date must be before end date")
        else:
            with st.spinner("Computing features..."):
                sample_df = compute_sample_features(
                    preview_symbol,
                    datetime.combine(preview_start, datetime.min.time()),
                    datetime.combine(preview_end, datetime.min.time())
                )

                if sample_df is not None and not sample_df.empty:
                    st.success(f"Feature values for {preview_symbol} from {preview_start} to {preview_end} ({len(sample_df)} rows)")

                    # Get selected features
                    selected_features = []
                    for src_config in config['sources'].values():
                        if src_config.get('enabled', False):
                            selected_features.extend(src_config.get('features', []))

                    # Build column mapping for display
                    column_mapping = {}
                    display_columns = ['Date']

                    # Add raw prices if configured
                    if config.get('include_raw_prices', True):
                        for col_type in ['Open', 'Close']:
                            col_name = f'{col_type}_{preview_symbol}'
                            if col_name in sample_df.columns:
                                column_mapping[col_name] = col_type
                                display_columns.append(col_type)

                    # Add selected features
                    for feature in selected_features:
                        col_name = f"{feature}_{preview_symbol}"
                        if col_name in sample_df.columns:
                            display_name = get_feature_display_name(feature)
                            column_mapping[col_name] = display_name
                            display_columns.append(display_name)

                    if column_mapping:
                        # Create display DataFrame
                        display_df = sample_df[list(column_mapping.keys())].copy()
                        display_df = display_df.rename(columns=column_mapping)
                        display_df.insert(0, 'Date', display_df.index.strftime('%Y-%m-%d'))
                        display_df = display_df.reset_index(drop=True)

                        # Show the full data table
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            height=min(400, 35 * len(display_df) + 38)
                        )

                        # Feature legend
                        with st.expander("Feature Descriptions"):
                            legend_data = []
                            for feature in selected_features:
                                col_name = f"{feature}_{preview_symbol}"
                                if col_name in sample_df.columns:
                                    legend_data.append({
                                        'Feature': get_feature_display_name(feature),
                                        'Category': get_feature_category(feature),
                                        'Description': get_feature_description(feature)
                                    })
                            if legend_data:
                                st.dataframe(
                                    pd.DataFrame(legend_data),
                                    use_container_width=True,
                                    hide_index=True
                                )

                        feature_columns = build_feature_columns_for_symbol(
                            sample_df,
                            preview_symbol,
                            selected_features,
                            config.get('include_raw_prices', True),
                        )

                        with st.expander("Machine Learning Price Preview", expanded=True):
                            if feature_columns:
                                preview = generate_price_prediction_preview(
                                    sample_df,
                                    preview_symbol,
                                    feature_columns,
                                )

                                if preview:
                                    metric_cols = st.columns(3)
                                    metric_cols = st.columns(2)
                                    with metric_cols[0]:
                                        st.metric("MAE (next close)", f"{preview['mae']:.4f}")
                                    with metric_cols[1]:
                                        st.metric("RMSE (next close)", f"{preview['rmse']:.4f}")
                                    with metric_cols[2]:
                                        st.metric("Features Used", len(feature_columns))

                                    st.caption("Quick regression preview using your selected inputs to predict the next closing price.")
                                    st.line_chart(
                                        preview["history"][
                                            ["Actual Next Close", "Predicted Next Close"]
                                        ]
                                    )

                                    st.markdown("**Which signals the model leaned on**")
                                    st.bar_chart(
                                        preview["coefficients"].set_index("Feature")
                                    )
                                else:
                                    st.info(
                                        "Need a few more rows or features to preview price predictions. Try a wider date range or enable raw prices."
                                    )
                            else:
                                st.info("Select at least one available column to preview machine learning behavior.")

                        # Download option
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name=f"{preview_symbol}_features_{preview_start}_{preview_end}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No matching features found in computed data")
                else:
                    st.error("Could not compute features. Make sure data is available for the selected date range.")


def get_selected_features() -> Dict[str, List[str]]:
    """Get the currently selected features configuration.

    Returns:
        Dictionary mapping source names to list of selected feature names.
    """
    initialize_feature_config()
    config = st.session_state.feature_config

    if not config.get('use_feature_engineering', False):
        return {}

    selected = {}
    for source_name, src_config in config.get('sources', {}).items():
        if src_config.get('enabled', False) and src_config.get('features'):
            selected[source_name] = src_config['features']

    return selected


def get_feature_config() -> Dict[str, Any]:
    """Get the full feature configuration.

    Returns:
        Complete feature configuration dictionary.
    """
    initialize_feature_config()
    return st.session_state.feature_config.copy()
