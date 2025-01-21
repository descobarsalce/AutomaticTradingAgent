import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Union
from utils import (
    format_date,
    format_money,
)
from metrics.metrics_calculator import MetricsCalculator

class TradingVisualizer:
    def __init__(self):
        self.figs = {}
        self.rsi_period = 14  # Default RSI period
        self.show_rsi = True
        self.show_sma20 = True
        self.show_sma50 = True
        
    def create_charts(self, portfolio_data, trades=None):
        """Create interactive trading charts for multiple stocks"""
        self.figs = {}
        
        try:
            for symbol, data in portfolio_data.items():
                fig = self.create_single_chart(symbol, data, trades.get(symbol) if trades else None)
                if fig:  # Only add if chart creation was successful
                    self.figs[symbol] = fig
        except Exception as e:
            print(f"Error creating charts: {str(e)}")
            
        return self.figs
        
    def create_single_chart(self, symbol, data, trades=None, info_history=None):
        """Create interactive trading chart with RSI indicator"""
        try:
            # Create a copy of data to avoid modifying original
            chart_data = data.copy()
            
            # Calculate RSI if enabled
            if self.show_rsi:
                chart_data['RSI'] = ta.momentum.RSIIndicator(chart_data['Close'], window=self.rsi_period).rsi()
            
            # Calculate SMAs if enabled
            if self.show_sma20:
                chart_data['SMA_20'] = ta.trend.sma_indicator(chart_data['Close'], window=20)
            
            if self.show_sma50:
                chart_data['SMA_50'] = ta.trend.sma_indicator(chart_data['Close'], window=50)
            
            # Determine number of rows based on enabled indicators
            num_rows = 2  # Price and Volume are always shown
            if self.show_rsi:
                num_rows += 1
            
            # Create subplot titles
            subplot_titles = [f'{symbol} Price', 'Volume']
            if self.show_rsi:
                subplot_titles.append('RSI')
            
            # Create figure with dynamic subplots
            fig = make_subplots(
                rows=num_rows + 1, cols=1, # Added a row for actions
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5] + [0.25] * (num_rows), #Adjusted row heights
                subplot_titles=subplot_titles + ['Agent Actions'] #Added subplot title
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name=symbol
                ),
                row=1, col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=chart_data.index,
                    y=chart_data['Volume'],
                    name='Volume'
                ),
                row=2, col=1
            )
            
            # Add moving averages if enabled
            if self.show_sma20 and 'SMA_20' in chart_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )
            
            if self.show_sma50 and 'SMA_50' in chart_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['SMA_50'],
                        name='SMA 50',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
            
            # Add RSI if enabled
            if self.show_rsi and 'RSI' in chart_data.columns:
                rsi_row = num_rows if self.show_rsi else None
                if rsi_row:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data.index,
                            y=chart_data['RSI'],
                            name='RSI',
                            line=dict(color='purple')
                        ),
                        row=rsi_row, col=1
                    )
                    
                    # Add RSI overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=rsi_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=rsi_row, col=1)
            
            # Add trade markers if available
            if trades is not None:
                try:
                    # Add buy markers
                    buys = trades[trades['action'] > 0]
                    if not buys.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=buys.index,
                                y=buys['price'],
                                mode='markers',
                                name='Buy',
                                marker=dict(
                                    symbol='triangle-up',
                                    size=8,
                                    color='green',
                                    line=dict(color='white', width=2)
                                )
                            ),
                            row=1, col=1
                        )
                    
                    # Add sell markers
                    sells = trades[trades['action'] < 0]
                    if not sells.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=sells.index,
                                y=sells['price'],
                                mode='markers',
                                name='Sell',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=8,
                                    color='red',
                                    line=dict(color='white', width=2)
                                )
                            ),
                            row=1, col=1
                        )
                except Exception as e:
                    print(f"Error adding trade markers: {str(e)}")
            
            #Added action visualization
            if info_history is not None:
                self.plot_actions_with_price(info_history, chart_data, fig)


            # Update layout with interactive features and date formatting
            fig.update_layout(
                height=1000, #Increased height for action plot
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                dragmode='zoom',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Configure x-axis for date display
            fig.update_xaxes(
                rangeslider=dict(visible=False),
                type='date',
                tickformat='%Y-%m-%d',
                tickmode='auto',
                nticks=10,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickangle=45
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating chart for {symbol}: {str(e)}")
            return None

    def plot_discrete_actions(self, info_history: List[Dict], fig: Optional[go.Figure] = None) -> go.Figure:
        """
        Creates a scatter plot of discrete actions over time using plotly.

        Args:
            info_history: List of info dictionaries from environment steps
            fig: Optional existing figure to add traces to
        """
        if fig is None:
            fig = go.Figure()

        # Extract actions and dates
        dates = [info['date'] for info in info_history]
        actions = [info['actions'][0] if isinstance(info['actions'], list) else info['actions'] for info in info_history]

        # Create a scatter plot for each action type
        action_colors = {0: 'blue', 1: 'green', 2: 'red'}
        action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

        for action_value in [0, 1, 2]:
            mask = [a == action_value for a in actions]
            if any(mask):
                fig.add_trace(go.Scatter(
                    x=[d for d, m in zip(dates, mask) if m],
                    y=[action_value] * sum(mask),
                    mode='markers',
                    name=action_names[action_value],
                    marker=dict(color=action_colors[action_value], size=8),
                ))

        fig.update_layout(
            title='Agent Actions Over Time',
            xaxis_title='Date',
            yaxis=dict(
                title='Action',
                ticktext=['Hold (0)', 'Buy (1)', 'Sell (2)'],
                tickvals=[0, 1, 2],
            ),
            showlegend=True
        )

        return fig

    def plot_actions_with_price(self, info_history: List[Dict], price_data: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Creates a combined plot of price and actions.

        Args:
            info_history: List of info dictionaries from environment steps
            price_data: DataFrame containing price data with 'Close' column
            fig: Existing figure to add traces to
        """

        # Add actions to the last subplot
        self.plot_discrete_actions(info_history, fig)
        fig.update_layout(height=1000, title='Price and Agent Actions')
        return fig