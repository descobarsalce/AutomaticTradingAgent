import optuna
import streamlit as st
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


"""
Technical Analysis Dashboard Component
Handles the visualization and analysis of stock data
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import os

from utils.stock_utils import parse_stock_list


def display_tech_analysis_tab():
    """
    Renders the technical analysis dashboard tab
    Args:
        model: The trading model instance
    """
    st.header("Technical Analysis Dashboard")

    # Use session's data handler
    data_handler = st.session_state.data_handler
    if not data_handler._sql_handler.session or not data_handler._sql_handler.session.is_active:
        data_handler._sql_handler.session  # Ensure database session is initialized

    # Visualization stock selection (separate from training)
    viz_stock_input = st.text_input("Stocks to Visualize (comma-separated)",
                                    value="AAPL, MSFT, GOOGL")
    viz_stocks = parse_stock_list(viz_stock_input)

    # Date selection for visualization
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        viz_start_date = datetime.combine(
            st.date_input("Analysis Start Date",
                          value=datetime.now() - timedelta(days=365),
                          min_value=datetime(2005, 1, 1)),
            datetime.min.time())
    with viz_col2:
        viz_end_date = datetime.combine(
            st.date_input("Analysis End Date", value=datetime.now()),
            datetime.min.time())

    # Plot controls
    st.subheader("Visualization Options")
    plot_col1, plot_col2, plot_col3 = st.columns(3)

    with plot_col1:
        show_rsi = st.checkbox("Show RSI", value=True, key="analysis_rsi")
        show_sma20 = st.checkbox("Show SMA 20",
                                 value=True,
                                 key="analysis_sma20")

    with plot_col2:
        show_sma50 = st.checkbox("Show SMA 50",
                                 value=True,
                                 key="analysis_sma50")
        rsi_period = st.slider("RSI Period",
                               min_value=7,
                               max_value=21,
                               value=14,
                               key="analysis_rsi_period") if show_rsi else 14

    with plot_col3:
        st.write("Layout Settings")
        num_columns = st.selectbox("Number of Columns",
                                   options=[1, 2, 3, 4],
                                   index=1,
                                   key="num_columns")
        # Add Burn-In Period control
        burn_in_days = st.number_input("Burn-In Period (days)",
                                       min_value=0,
                                       value=30,
                                       key="burn_in_days")
        # Layout Preview
        st.write("Layout Preview")
        preview_container = st.container()
        with preview_container:
            preview_cols = st.columns(num_columns)
            for i in range(num_columns):
                with preview_cols[i]:
                    st.markdown(f"""
                        <div style="
                            border: 2px dashed #666;
                            border-radius: 5px;
                            padding: 10px;
                            margin: 5px;
                            text-align: center;
                            background-color: rgba(100, 100, 100, 0.1);
                            min-height: 80px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        ">
                            <span style="color: #666;">Chart {i+1}</span>
                        </div>
                        """,
                                unsafe_allow_html=True)

    if st.button("Generate Analysis"):
        generate_analysis(viz_stocks, viz_start_date, viz_end_date, 
                          show_rsi, show_sma20, show_sma50, rsi_period,
                          num_columns, burn_in_days)


def generate_analysis(viz_stocks, viz_start_date, viz_end_date,
                      show_rsi, show_sma20, show_sma50, rsi_period,
                      num_columns, burn_in_days):
    viz_start_date = pd.Timestamp(viz_start_date).tz_localize('America/New_York')
    viz_end_date = pd.Timestamp(viz_end_date).tz_localize('America/New_York')

    with st.spinner('Generating analysis...'):
        analysis_container = st.container()
        with analysis_container:
            price_charts = {}
            volume_charts = {}
            rsi_charts = {}
            ma_charts = {}

            for stock in viz_stocks:
                try:
                    burn_in_start_date = viz_start_date - pd.Timedelta(days=burn_in_days)
                    data = st.session_state.data_handler.fetch_data(
                        [stock], burn_in_start_date, viz_end_date
                    )

                    if data.empty:
                        st.error(f"No data available for {stock}")
                        continue

                    # Price Chart
                    try:
                        price_fig = go.Figure()
                        price_fig.add_trace(go.Scatter(
                            x=data.index, 
                            y=data[f'Close_{stock}'],
                            name=f'{stock} Price'
                        ))
                        price_fig.update_layout(title=f'{stock} Price Chart')
                        price_charts[stock] = price_fig
                    except Exception as pe:
                        st.error(f"Price chart error for {stock}")
                        logger.error(f"Error generating price chart for {stock}: {str(pe)}")

                    # Volume Chart
                    try:
                        volume_fig = go.Figure()
                        volume_fig.add_trace(go.Bar(
                            x=data.index,
                            y=data[f'Volume_{stock}'],
                            name=f'{stock} Volume'
                        ))
                        volume_fig.update_layout(title=f'{stock} Volume')
                        volume_charts[stock] = volume_fig
                    except Exception as ve:
                        st.error(f"Volume chart error for {stock}")
                        logger.error(f"Error generating volume chart for {stock}: {str(ve)}")

                    # Moving Averages
                    if show_sma20 or show_sma50:
                        try:
                            ma_fig = go.Figure()
                            ma_fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data[f'Close_{stock}'],
                                name=f'{stock} Price'
                            ))

                            if show_sma20:
                                sma20 = data[f'Close_{stock}'].rolling(window=20).mean()
                                ma_fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=sma20,
                                    name=f'SMA 20'
                                ))

                            if show_sma50:
                                sma50 = data[f'Close_{stock}'].rolling(window=50).mean()
                                ma_fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=sma50,
                                    name=f'SMA 50'
                                ))

                            ma_fig.update_layout(title=f'{stock} Moving Averages')
                            ma_charts[stock] = ma_fig
                        except Exception as mae:
                            st.error(f"Moving Averages chart error for {stock}")
                            logger.error(f"Error generating moving averages chart for {stock}: {str(mae)}")

                except Exception as e:
                    error_details = {
                        'stock': stock,
                        'error_type': type(e).__name__,
                        'error_msg': str(e)
                    }
                    
                    # Only add data-related details if data exists and is valid
                    try:
                        if 'data' in locals() and isinstance(data, pd.DataFrame):
                            error_details.update({
                                'data_shape': data.shape,
                                'data_columns': data.columns.tolist(),
                                'date_range': f"{data.index.min()} to {data.index.max()}" if not data.empty else 'Empty DataFrame'
                            })
                    except Exception as data_error:
                        error_details.update({
                            'data_error': str(data_error)
                        })
                    
                    logger.error(f"Error processing {stock}:" + 
                               ''.join(f"\n{k}: {v}" for k, v in error_details.items()))
                    st.error(f"Error processing {stock}: {error_details['error_msg']}")
                    continue

            # Display charts
            if price_charts:
                st.subheader("Price Analysis")
                display_charts_grid(price_charts, "Price Charts", num_columns)

            if volume_charts:
                st.subheader("Volume Analysis")
                display_charts_grid(volume_charts, "Volume Charts", num_columns)

            if ma_charts:
                st.subheader("Moving Averages Analysis")
                display_charts_grid(ma_charts, "Moving Averages", num_columns)

        # Portfolio Analytics Section
        if hasattr(st.session_state, 'portfolio_history') and st.session_state.portfolio_history:
            st.header("Portfolio Analytics")

            # Portfolio Performance
            portfolio_df = pd.DataFrame(st.session_state.portfolio_history, columns=['Portfolio Value'])
            st.subheader("Portfolio Value Over Time")
            st.line_chart(portfolio_df)

            # Returns Analysis
            values = np.array(st.session_state.portfolio_history)
            returns = np.diff(values) / values[:-1]

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Returns Distribution
                fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=50)])
                fig.update_layout(title="Returns Distribution",
                                xaxis_title="Return",
                                yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)

                # Drawdown
                peak = np.maximum.accumulate(values)
                drawdowns = (peak - values) / peak
                st.subheader("Drawdown Over Time")
                st.area_chart(pd.DataFrame(drawdowns, columns=['Drawdown']))

            with chart_col2:
                # Cumulative Returns
                cum_returns = pd.DataFrame(np.cumprod(1 + returns) - 1, columns=['Returns'])
                st.subheader("Cumulative Returns")
                st.line_chart(cum_returns)

                # Rolling Volatility
                rolling_vol = pd.DataFrame(returns, columns=['Returns']).rolling(30).std() * np.sqrt(252)
                st.subheader("30-Day Rolling Volatility")
                st.line_chart(rolling_vol)


def display_charts_grid(charts: Dict[str, go.Figure], title: str,
                        num_columns: int) -> None:
    """
    Displays charts in a grid layout
    """
    if charts:
        st.subheader(title)
        stocks = list(charts.keys())
        for i in range(0, len(charts), num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(stocks):
                    with cols[j]:
                        st.plotly_chart(charts[stocks[i + j]],
                                        use_container_width=True)