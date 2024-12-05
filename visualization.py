import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import ta

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
        
    def create_single_chart(self, symbol, data, trades=None):
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
                rows=num_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5] + [0.25] * (num_rows - 1),
                subplot_titles=subplot_titles
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
            
            # Update layout with interactive features
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                dragmode='zoom',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Add buttons for zoom levels
            fig.update_xaxes(rangeslider=dict(visible=False))
            
            return fig
            
        except Exception as e:
            print(f"Error creating chart for {symbol}: {str(e)}")
            return None
