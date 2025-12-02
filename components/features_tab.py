"""
Feature Selection Component
Allows users to configure which features are used for model training.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# All features grouped by category - includes raw data, state, and engineered features
FEATURES_BY_CATEGORY = {
    'Raw Price Data': [
        ('open', 'Open', 'Opening price'),
        ('high', 'High', 'Highest price'),
        ('low', 'Low', 'Lowest price'),
        ('close', 'Close', 'Closing price'),
        ('volume', 'Volume', 'Trading volume'),
    ],
    'Portfolio State': [
        ('position_held', 'Position Held', 'Current shares held per asset'),
        ('position_value', 'Position Value', 'Current value of positions'),
        ('cash_balance', 'Cash Balance', 'Available cash'),
        ('total_portfolio_value', 'Total Value', 'Cash + positions'),
        ('unrealized_pnl', 'Unrealized P&L', 'Profit/loss on open positions'),
    ],
    'Price Returns': [
        ('price_change', 'Price Change ($)', 'Absolute change in closing price'),
        ('returns', 'Daily Returns (%)', 'Percentage change from previous day'),
        ('log_returns', 'Log Returns', 'Natural log of price ratio'),
    ],
    'Intraday': [
        ('high_low_ratio', 'High/Low Ratio', 'Intraday volatility measure'),
        ('close_open_ratio', 'Close/Open Ratio', 'Daily direction indicator'),
        ('typical_price', 'Typical Price', 'Average of H/L/C'),
        ('price_range', 'Price Range ($)', 'High minus Low'),
        ('gap', 'Overnight Gap ($)', 'Open vs previous Close'),
    ],
    'Volume Indicators': [
        ('volume_change', 'Volume Change (%)', 'Volume vs previous day'),
        ('volume_ma_ratio', 'Vol vs 20d Avg', 'Relative volume'),
        ('obv', 'On-Balance Volume', 'Cumulative volume flow'),
        ('obv_ema', 'OBV EMA', 'Smoothed OBV trend'),
    ],
    'Lagged Prices': [
        ('close_lag1', '1-Day Lag', 'Price 1 day ago'),
        ('close_lag2', '2-Day Lag', 'Price 2 days ago'),
        ('close_lag3', '3-Day Lag', 'Price 3 days ago'),
        ('close_lag4', '4-Day Lag', 'Price 4 days ago'),
        ('close_lag5', '5-Day Lag', 'Price 5 days ago'),
    ],
    'Moving Averages': [
        ('rolling_mean_5', '5-Day MA', 'Short-term trend'),
        ('rolling_mean_10', '10-Day MA', 'Medium-term trend'),
        ('rolling_mean_20', '20-Day MA', 'Longer-term trend'),
    ],
    'Rolling Stats': [
        ('rolling_std_5', '5-Day Std', 'Short volatility'),
        ('rolling_std_10', '10-Day Std', 'Medium volatility'),
        ('rolling_std_20', '20-Day Std', 'Long volatility'),
        ('rolling_min_5', '5-Day Low', 'Recent support'),
        ('rolling_max_5', '5-Day High', 'Recent resistance'),
        ('rolling_min_10', '10-Day Low', 'Support level'),
        ('rolling_max_10', '10-Day High', 'Resistance level'),
        ('rolling_min_20', '20-Day Low', 'Major support'),
        ('rolling_max_20', '20-Day High', 'Major resistance'),
    ],
    'Momentum': [
        ('rsi', 'RSI (14)', 'Relative Strength Index 0-100'),
        ('macd', 'MACD', '12-26 EMA difference'),
        ('macd_signal', 'MACD Signal', '9-day EMA of MACD'),
        ('macd_histogram', 'MACD Hist', 'MACD minus Signal'),
        ('stoch_k', 'Stoch %K', 'Fast stochastic'),
        ('stoch_d', 'Stoch %D', 'Slow stochastic'),
    ],
    'Volatility': [
        ('volatility', 'Volatility (20d)', 'Std dev of returns'),
        ('bb_upper', 'BB Upper', 'Upper Bollinger Band'),
        ('bb_lower', 'BB Lower', 'Lower Bollinger Band'),
        ('bb_sma', 'BB Middle', 'Bollinger SMA'),
        ('bb_width', 'BB Width', 'Band width ratio'),
        ('bb_percent', 'BB %B', 'Position in bands'),
    ],
    'Cycle': [
        ('fft_1', 'FFT 1', 'Primary cycle'),
        ('fft_2', 'FFT 2', 'Secondary cycle'),
        ('fft_3', 'FFT 3', 'Tertiary cycle'),
    ],
    'ML Predictions': [
        # Dynamically populated from enabled models
        # Format: (key, name, description)
    ],
}

# Build flat lookup for feature info
FEATURE_INFO = {}
for cat, features in FEATURES_BY_CATEGORY.items():
    for key, name, desc in features:
        FEATURE_INFO[key] = {'name': name, 'description': desc, 'category': cat}

ALL_FEATURE_KEYS = list(FEATURE_INFO.keys())

# Default selected features
DEFAULT_SELECTED = [
    # Raw data
    'close', 'volume',
    # Portfolio state
    'position_held', 'cash_balance',
    # Engineered
    'returns', 'log_returns', 'high_low_ratio', 'volume_change',
    'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'volatility'
]

DEFAULT_FEATURE_CONFIG = {
    'use_feature_engineering': True,
    'selected_features': DEFAULT_SELECTED.copy(),
    # Keep sources structure for backward compatibility with training code
    'sources': {
        'MarketDataSource': {'enabled': True, 'features': []},
        'TechnicalSource': {'enabled': True, 'features': []}
    }
}

# Features that map to MarketDataSource (for backward compat)
MARKET_DATA_FEATURES = {
    'open', 'high', 'low', 'close', 'volume',
    'price_change', 'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
    'typical_price', 'price_range', 'gap', 'volume_change', 'volume_ma_ratio',
    'close_lag1', 'close_lag2', 'close_lag3', 'close_lag4', 'close_lag5',
    'rolling_mean_5', 'rolling_mean_10', 'rolling_mean_20',
    'rolling_std_5', 'rolling_std_10', 'rolling_std_20',
    'rolling_min_5', 'rolling_min_10', 'rolling_min_20',
    'rolling_max_5', 'rolling_max_10', 'rolling_max_20',
}

# Portfolio state features (handled separately by environment)
PORTFOLIO_STATE_FEATURES = {
    'position_held', 'position_value', 'cash_balance', 'total_portfolio_value', 'unrealized_pnl'
}


def get_feature_display_name(feature_key: str) -> str:
    return FEATURE_INFO.get(feature_key, {}).get('name', feature_key.replace('_', ' ').title())


def get_ml_prediction_features() -> List[tuple]:
    """Get ML prediction features from enabled models."""
    try:
        from components.ml_models_tab import get_enabled_prediction_models, get_model_info
        enabled_models = get_enabled_prediction_models()

        features = []
        for model_id in enabled_models:
            info = get_model_info(model_id)
            if info and info.get('trained', False):
                model_type = info.get('type', 'Unknown')
                symbol = info.get('symbol', 'Unknown')

                # Add prediction features for this model
                for horizon in [1, 5, 21]:
                    key = f"ml_{model_id}_price_{horizon}d"
                    name = f"{model_type} {symbol} Price {horizon}d"
                    desc = f"{horizon}-day price prediction from {model_id}"
                    features.append((key, name, desc))

                    key = f"ml_{model_id}_direction_{horizon}d"
                    name = f"{model_type} {symbol} Dir {horizon}d"
                    desc = f"{horizon}-day direction prediction from {model_id}"
                    features.append((key, name, desc))

        return features
    except ImportError:
        return []
    except Exception as e:
        logger.debug(f"Could not get ML prediction features: {e}")
        return []


def update_ml_predictions_category():
    """Update the ML Predictions category with current enabled models."""
    ml_features = get_ml_prediction_features()
    FEATURES_BY_CATEGORY['ML Predictions'] = ml_features

    # Update FEATURE_INFO with ML features
    for key, name, desc in ml_features:
        FEATURE_INFO[key] = {'name': name, 'description': desc, 'category': 'ML Predictions'}


def initialize_feature_config():
    if 'feature_config' not in st.session_state:
        st.session_state.feature_config = DEFAULT_FEATURE_CONFIG.copy()
    if 'selected_features' not in st.session_state.feature_config:
        st.session_state.feature_config['selected_features'] = DEFAULT_SELECTED.copy()


def sync_sources_from_selected(config: Dict) -> None:
    """Sync the sources dict from selected_features for backward compatibility."""
    selected = set(config.get('selected_features', []))
    # Filter out portfolio state features - those are handled by environment
    market_features = [f for f in selected if f in MARKET_DATA_FEATURES]
    tech_features = [f for f in selected if f not in MARKET_DATA_FEATURES and f not in PORTFOLIO_STATE_FEATURES and not f.startswith('ml_')]

    # Extract ML prediction features
    ml_features = [f for f in selected if f.startswith('ml_')]

    config['sources'] = {
        'MarketDataSource': {'enabled': bool(market_features), 'features': market_features},
        'TechnicalSource': {'enabled': bool(tech_features), 'features': tech_features}
    }

    # Extract portfolio state config
    config['include_positions'] = any(f in selected for f in ['position_held', 'position_value', 'unrealized_pnl'])
    config['include_balance'] = any(f in selected for f in ['cash_balance', 'total_portfolio_value'])
    config['include_raw_prices'] = any(f in selected for f in ['open', 'high', 'low', 'close', 'volume'])

    # Store ML prediction config
    config['use_predictions'] = bool(ml_features)
    config['selected_ml_features'] = ml_features


def get_available_stocks() -> List[str]:
    try:
        if 'data_handler' not in st.session_state:
            return []
        from sqlalchemy import distinct
        from data.database import StockData
        symbols = [row[0] for row in st.session_state.data_handler.query(distinct(StockData.symbol)).all()]
        return sorted(symbols)
    except Exception as e:
        logger.error(f"Error getting available stocks: {e}")
        return []


def compute_sample_features(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    try:
        if 'data_handler' not in st.session_state:
            return None
        fetch_start = start_date - timedelta(days=60)
        data = st.session_state.data_handler.fetch_data([symbol], fetch_start, end_date, use_SQL=True)
        if data is None or data.empty:
            return None

        from data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()
        features_df = engineer.compute_features(data, symbols=[symbol], drop_na=False)

        if features_df is None or features_df.empty:
            return None

        mask = (features_df.index >= start_date) & (features_df.index <= end_date)
        return features_df.loc[mask]
    except Exception as e:
        logger.error(f"Error computing sample features: {e}", exc_info=True)
        return None


def display_features_tab():
    """Renders the compact feature selection interface."""
    st.header("Feature Selection")

    initialize_feature_config()

    # Update ML predictions category with currently enabled models
    update_ml_predictions_category()

    config = st.session_state.feature_config
    selected_features = set(config.get('selected_features', DEFAULT_SELECTED))

    # Quick actions row
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("Select All", key="fe_all"):
            selected_features = set(ALL_FEATURE_KEYS)
    with c2:
        if st.button("Clear All", key="fe_clear"):
            selected_features = set()

    # Display categories in a compact grid (2 columns of categories)
    categories = list(FEATURES_BY_CATEGORY.keys())
    left_cats = categories[:len(categories)//2 + len(categories) % 2]
    right_cats = categories[len(categories)//2 + len(categories) % 2:]

    col_left, col_right = st.columns(2)

    with col_left:
        for cat in left_cats:
            features = FEATURES_BY_CATEGORY[cat]
            options = [key for key, name, desc in features]
            defaults = [key for key in options if key in selected_features]

            selected_in_cat = st.multiselect(
                cat,
                options=options,
                default=defaults,
                format_func=lambda x: FEATURE_INFO[x]['name'],
                key=f"cat_{cat}",
                help=f"{len(options)} features"
            )
            for key in options:
                if key in selected_in_cat:
                    selected_features.add(key)
                else:
                    selected_features.discard(key)

    with col_right:
        for cat in right_cats:
            features = FEATURES_BY_CATEGORY[cat]
            options = [key for key, name, desc in features]
            defaults = [key for key in options if key in selected_features]

            selected_in_cat = st.multiselect(
                cat,
                options=options,
                default=defaults,
                format_func=lambda x: FEATURE_INFO[x]['name'],
                key=f"cat_{cat}",
                help=f"{len(options)} features"
            )
            for key in options:
                if key in selected_in_cat:
                    selected_features.add(key)
                else:
                    selected_features.discard(key)

    # Update config
    config['selected_features'] = list(selected_features)
    sync_sources_from_selected(config)
    st.session_state.feature_config = config

    # Summary by category
    raw_count = len([f for f in selected_features if f in {'open', 'high', 'low', 'close', 'volume'}])
    state_count = len([f for f in selected_features if f in PORTFOLIO_STATE_FEATURES])
    ml_count = len([f for f in selected_features if f.startswith('ml_')])
    eng_count = len(selected_features) - raw_count - state_count - ml_count

    summary = f"Selected: {len(selected_features)} total | Raw: {raw_count} | State: {state_count} | Engineered: {eng_count}"
    if ml_count > 0:
        summary += f" | ML Predictions: {ml_count}"
    st.caption(summary)

    # Feature Preview (collapsed by default)
    with st.expander("Feature Preview", expanded=False):
        available_stocks = get_available_stocks()
        if not available_stocks:
            st.warning("No stocks in database. Add data in Database Explorer first.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                preview_symbol = st.selectbox("Stock", options=available_stocks, key="preview_symbol")
            with c2:
                preview_start = st.date_input("Start", value=datetime.now() - timedelta(days=30), key="preview_start")
            with c3:
                preview_end = st.date_input("End", value=datetime.now() - timedelta(days=1), key="preview_end")

            if st.button("Generate Preview", key="gen_preview"):
                if preview_start >= preview_end:
                    st.error("Start must be before end")
                else:
                    with st.spinner("Computing..."):
                        sample_df = compute_sample_features(
                            preview_symbol,
                            datetime.combine(preview_start, datetime.min.time()),
                            datetime.combine(preview_end, datetime.min.time())
                        )
                        if sample_df is not None and not sample_df.empty:
                            cols_to_show = []
                            col_names = {}

                            # Show selected features that exist in the data
                            for feat in selected_features:
                                if feat in PORTFOLIO_STATE_FEATURES:
                                    continue  # Skip portfolio state - not in historical data
                                col_name = f"{feat}_{preview_symbol}"
                                # Also try capitalized version for OHLCV
                                alt_col_name = f"{feat.capitalize()}_{preview_symbol}"
                                if col_name in sample_df.columns:
                                    cols_to_show.append(col_name)
                                    col_names[col_name] = get_feature_display_name(feat)
                                elif alt_col_name in sample_df.columns:
                                    cols_to_show.append(alt_col_name)
                                    col_names[alt_col_name] = get_feature_display_name(feat)

                            if cols_to_show:
                                display_df = sample_df[cols_to_show].rename(columns=col_names)
                                display_df.insert(0, 'Date', display_df.index.strftime('%Y-%m-%d'))
                                st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True, height=300)

                                csv = display_df.to_csv(index=False)
                                st.download_button("Download CSV", csv, f"{preview_symbol}_features.csv", "text/csv")
                            else:
                                st.warning("No matching features in data")
                        else:
                            st.error("Could not compute features for selected range")


def get_selected_features() -> Dict[str, List[str]]:
    """Get selected features in source format for training."""
    initialize_feature_config()
    config = st.session_state.feature_config
    sync_sources_from_selected(config)

    selected = {}
    for source_name, src_config in config.get('sources', {}).items():
        if src_config.get('enabled', False) and src_config.get('features'):
            selected[source_name] = src_config['features']
    return selected


def get_feature_config() -> Dict[str, Any]:
    """Get the full feature configuration."""
    initialize_feature_config()
    config = st.session_state.feature_config
    sync_sources_from_selected(config)
    return config.copy()
