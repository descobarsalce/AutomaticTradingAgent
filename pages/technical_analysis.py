"""
Technical Analysis Dashboard Module
Handles the technical analysis visualization functionality for the trading platform.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

def parse_stock_list(stock_string: str) -> List[str]:
    """Parse comma-separated stock symbols into a list."""
    if not stock_string or not isinstance(stock_string, str):
        return []
    
    stocks = [
        stock.strip().upper()
        for stock in stock_string.split(',')
        if stock.strip()
    ]
    
    seen = set()
    unique_stocks = [
        stock for stock in stocks if not (stock in seen or seen.add(stock))
    ]
    
    return unique_stocks

def render_technical_analysis(model):
    """Render the technical analysis dashboard."""
    st.header("Technical Analysis Dashboard")
    
    # Stock selection
    viz_stock_input = st.text_input(
        "Stocks to Visualize (comma-separated)", 
        value="AAPL, MSFT, GOOGL"
    )
    viz_stocks = parse_stock_list(viz_stock_input)
    
    # Date selection
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        viz_start_date = datetime.combine(
            st.date_input("Analysis Start Date",
                          value=datetime.now() - timedelta(days=365)),
            datetime.min.time()
        )
    with viz_col2:
        viz_end_date = datetime.combine(
            st.date_input("Analysis End Date", value=datetime.now()),
            datetime.min.time()
        )
    
    # Plot controls
    st.subheader("Visualization Options")
    plot_col1, plot_col2, plot_col3 = st.columns(3)
    
    with plot_col1:
        show_rsi = st.checkbox("Show RSI", value=True, key="analysis_rsi")
        show_sma20 = st.checkbox("Show SMA 20", value=True, key="analysis_sma20")
    
    with plot_col2:
        show_sma50 = st.checkbox("Show SMA 50", value=True, key="analysis_sma50")
        rsi_period = st.slider(
            "RSI Period",
            min_value=7,
            max_value=21,
            value=14,
            key="analysis_rsi_period"
        ) if show_rsi else 14
    
    with plot_col3:
        st.write("Layout Settings")
        num_columns = st.selectbox(
            "Number of Columns",
            options=[1, 2, 3, 4],
            index=1,
            key="num_columns"
        )
        
        # Layout Preview
        st.write("Layout Preview")
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
                    unsafe_allow_html=True
                )
    
    if st.button("Generate Analysis"):
        analysis_container = st.container()
        with analysis_container:
            # Create dictionaries to store different types of charts
            price_charts = {}
            volume_charts = {}
            rsi_charts = {}
            ma_charts = {}
            
            # First collect all data and create charts
            for stock in viz_stocks:
                portfolio_data = model.data_handler.fetch_data(
                    stock, viz_start_date, viz_end_date
                )
                
                if not portfolio_data:
                    st.error(f"No data available for {stock}")
                    continue
                
                portfolio_data = model.data_handler.prepare_data()
                
                if stock in portfolio_data:
                    data = portfolio_data[stock]
                    
                    # Create price chart
                    price_fig = go.Figure(data=[
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name=stock
                        )
                    ])
                    price_fig.update_layout(title=f'{stock} Price History')
                    price_charts[stock] = price_fig
                    
                    # Create volume chart
                    volume_fig = go.Figure()
                    volume_fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name=f'{stock} Volume'
                        )
                    )
                    volume_fig.update_layout(title=f'{stock} Trading Volume')
                    volume_charts[stock] = volume_fig
                    
                    # Create RSI chart if enabled
                    if show_rsi and 'RSI' in data.columns:
                        rsi_fig = go.Figure()
                        rsi_fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['RSI'] * 100,
                                name=f'{stock} RSI'
                            )
                        )
                        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                        rsi_fig.update_layout(title=f'{stock} RSI ({rsi_period} periods)')
                        rsi_charts[stock] = rsi_fig
                    
                    # Create Moving Averages chart if enabled
                    if show_sma20 or show_sma50:
                        ma_fig = go.Figure()
                        ma_fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                name=f'{stock} Price'
                            )
                        )
                        if show_sma20 and 'SMA_20' in data.columns:
                            ma_fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=data['SMA_20'],
                                    name=f'{stock} SMA 20'
                                )
                            )
                        if show_sma50 and 'SMA_50' in data.columns:
                            ma_fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=data['SMA_50'],
                                    name=f'{stock} SMA 50'
                                )
                            )
                        ma_fig.update_layout(title=f'{stock} Moving Averages')
                        ma_charts[stock] = ma_fig
            
            def display_charts_grid(charts: Dict, title: str) -> None:
                """Helper function to display charts in a grid layout."""
                if charts:
                    st.subheader(title)
                    stocks = list(charts.keys())
                    for i in range(0, len(charts), num_columns):
                        cols = st.columns(num_columns)
                        for j in range(num_columns):
                            if i + j < len(stocks):
                                with cols[j]:
                                    st.plotly_chart(
                                        charts[stocks[i + j]],
                                        use_container_width=True
                                    )
            
            # Display each chart type using the dynamic grid
            if price_charts:
                display_charts_grid(price_charts, "Price Analysis")
            
            if volume_charts:
                display_charts_grid(volume_charts, "Volume Analysis")
            
            if rsi_charts:
                display_charts_grid(rsi_charts, "RSI Analysis")
            
            if ma_charts:
                display_charts_grid(ma_charts, "Moving Averages Analysis")
