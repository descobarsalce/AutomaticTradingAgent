import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import ta

class TradingVisualizer:
    def __init__(self):
        self.figs = {}
        self.rsi_period = 14  # Default RSI period
        
    def create_charts(self, portfolio_data, trades=None):
        """Create interactive trading charts for multiple stocks"""
        for symbol, data in portfolio_data.items():
            self.create_single_chart(symbol, data, trades.get(symbol) if trades else None)
        return self.figs
        
    def create_single_chart(self, symbol, data, trades=None):
        """Create interactive trading chart with RSI indicator"""
        # Calculate RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=self.rsi_period).rsi()
        
        # Create figure with secondary y-axis
        self.fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f'{symbol} Price', 'Volume', 'RSI')
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
            
        # Add RSI
        self.fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        self.fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        self.fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout with interactive features
        self.fig.update_layout(
            title=dict(
                text='Trading Chart',
                x=0.5
            ),
            yaxis_title='Price',
            yaxis2_title='Volume',
            yaxis3_title='RSI',
            xaxis_rangeslider_visible=False,
            height=1000,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='x unified'
        )
        
        # Add range selector
        self.fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            row=1, col=1
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
