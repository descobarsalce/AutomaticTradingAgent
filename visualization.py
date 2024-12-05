import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class TradingVisualizer:
    def __init__(self):
        self.figs = {}
        
    def create_charts(self, portfolio_data, trades=None):
        """Create interactive trading charts for multiple stocks"""
        for symbol, data in portfolio_data.items():
            self.create_single_chart(symbol, data, trades.get(symbol) if trades else None)
        return self.figs
        
    def create_single_chart(self, symbol, data, trades=None):
        """Create interactive trading chart"""
        self.fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        self.fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume bars
        self.fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Add moving averages
        if 'SMA_20' in data.columns:
            self.fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
            
        if 'SMA_50' in data.columns:
            self.fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
        # Update layout
        self.fig.update_layout(
            title='Trading Chart',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark'
        )
        
        return self.fig
        
    def add_trades(self, trades):
        """Add trade markers to the chart"""
        if self.fig is None:
            raise ValueError("Create chart first before adding trades")
            
        # Add buy markers
        self.fig.add_trace(
            go.Scatter(
                x=trades[trades['action'] > 0].index,
                y=trades[trades['action'] > 0]['price'],
                mode='markers',
                name='Buy',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green'
                )
            ),
            row=1, col=1
        )
        
        # Add sell markers
        self.fig.add_trace(
            go.Scatter(
                x=trades[trades['action'] < 0].index,
                y=trades[trades['action'] < 0]['price'],
                mode='markers',
                name='Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red'
                )
            ),
            row=1, col=1
        )
