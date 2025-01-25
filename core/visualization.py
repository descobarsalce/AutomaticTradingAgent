import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import ta

# =========================================================================
#  Existing Functions (Kept Intact)
# =========================================================================

# =========================================================================
#  Existing Class: TradingVisualizer (Kept Intact, but optionally extended)
# =========================================================================


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
                fig = self.create_single_chart(
                    symbol, data,
                    trades.get(symbol) if trades else None)
                if fig:  # Only add if chart creation was successful
                    self.figs[symbol] = fig
        except Exception as e:
            print(f"Error creating charts: {str(e)}")

        return self.figs

    def create_single_chart(self,
                            symbol,
                            data,
                            trades=None,
                            info_history=None):
        """Create interactive trading chart with RSI indicator"""
        try:
            # Create a copy of data to avoid modifying original
            chart_data = data.copy()

            # Calculate RSI if enabled
            if self.show_rsi:
                chart_data['RSI'] = ta.momentum.RSIIndicator(
                    chart_data['Close'], window=self.rsi_period).rsi()

            # Calculate SMAs if enabled
            if self.show_sma20:
                chart_data['SMA_20'] = ta.trend.sma_indicator(
                    chart_data['Close'], window=20)

            if self.show_sma50:
                chart_data['SMA_50'] = ta.trend.sma_indicator(
                    chart_data['Close'], window=50)

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
                rows=num_rows + 1,
                cols=1,  # Added a row for actions
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5] + [0.25] *
                (num_rows),  # Adjusted row heights
                subplot_titles=subplot_titles +
                ['Agent Actions']  # Added subplot title
            )

            # Add candlestick chart
            fig.add_trace(go.Candlestick(x=chart_data.index,
                                         open=chart_data['Open'],
                                         high=chart_data['High'],
                                         low=chart_data['Low'],
                                         close=chart_data['Close'],
                                         name=symbol),
                          row=1,
                          col=1)

            # Add volume bars
            fig.add_trace(go.Bar(x=chart_data.index,
                                 y=chart_data['Volume'],
                                 name='Volume'),
                          row=2,
                          col=1)

            # Add moving averages if enabled
            if self.show_sma20:
                chart_data['SMA_20'] = ta.trend.sma_indicator(chart_data['Close'], window=20)
                fig.add_trace(go.Scatter(x=chart_data.index,
                                         y=chart_data['SMA_20'],
                                         name='SMA 20',
                                         line=dict(color='orange')),
                              row=1,
                              col=1)

            if self.show_sma50:
                chart_data['SMA_50'] = ta.trend.sma_indicator(chart_data['Close'], window=50)
                fig.add_trace(go.Scatter(x=chart_data.index,
                                         y=chart_data['SMA_50'],
                                         name='SMA 50',
                                         line=dict(color='blue')),
                              row=1,
                              col=1)

            # Add RSI if enabled
            if self.show_rsi and 'RSI' in chart_data.columns:
                rsi_row = num_rows if self.show_rsi else None
                if rsi_row:
                    fig.add_trace(go.Scatter(x=chart_data.index,
                                             y=chart_data['RSI'],
                                             name='RSI',
                                             line=dict(color='purple')),
                                  row=rsi_row,
                                  col=1)

                    # Add RSI overbought/oversold lines
                    fig.add_hline(y=70,
                                  line_dash="dash",
                                  line_color="red",
                                  row=rsi_row,
                                  col=1)
                    fig.add_hline(y=30,
                                  line_dash="dash",
                                  line_color="green",
                                  row=rsi_row,
                                  col=1)

            # Add trade markers if available
            if trades is not None:
                try:
                    # Add buy markers
                    buys = trades[trades['action'] > 0]
                    if not buys.empty:
                        fig.add_trace(go.Scatter(x=buys.index,
                                                 y=buys['price'],
                                                 mode='markers',
                                                 name='Buy',
                                                 marker=dict(
                                                     symbol='triangle-up',
                                                     size=8,
                                                     color='green',
                                                     line=dict(color='white',
                                                               width=2))),
                                      row=1,
                                      col=1)

                    # Add sell markers
                    sells = trades[trades['action'] < 0]
                    if not sells.empty:
                        fig.add_trace(go.Scatter(x=sells.index,
                                                 y=sells['price'],
                                                 mode='markers',
                                                 name='Sell',
                                                 marker=dict(
                                                     symbol='triangle-down',
                                                     size=8,
                                                     color='red',
                                                     line=dict(color='white',
                                                               width=2))),
                                      row=1,
                                      col=1)
                except Exception as e:
                    print(f"Error adding trade markers: {str(e)}")

            # Added action visualization
            if info_history is not None:
                fig = plot_actions_with_price(info_history, chart_data)

            # Update layout with interactive features and date formatting
            fig.update_layout(
                height=1000,  # Increased height for action plot
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                dragmode='zoom',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=50, b=50))

            # Configure x-axis for date display
            fig.update_xaxes(rangeslider=dict(visible=False),
                             type='date',
                             tickformat='%Y-%m-%d',
                             tickmode='auto',
                             nticks=10,
                             showgrid=True,
                             gridcolor='rgba(128, 128, 128, 0.2)',
                             tickangle=45)

            return fig

        except Exception as e:
            print(f"Error creating chart for {symbol}: {str(e)}")
            return None

    def plot_correlation_heatmap(self, portfolio_data: Dict[str, pd.DataFrame],
                                 price_col: str = 'Close') -> go.Figure:
        """
        Creates a correlation heatmap for daily returns of multiple symbols in 'portfolio_data'.
        :param portfolio_data: dict[symbol -> DataFrame with OHLC data]
        :param price_col: 'Close' or another column to compute returns
        :return: Plotly Figure (heatmap)
        """
        # Build a DataFrame of daily returns for each symbol
        returns_dict = {}
        for symbol, df in portfolio_data.items():
            if price_col not in df.columns or df.empty:
                continue
            # Sort by index just in case
            sorted_df = df.sort_index()
            # Calculate daily returns
            rets = sorted_df[price_col].pct_change().dropna()
            returns_dict[symbol] = rets

        if not returns_dict:
            raise ValueError("No valid data to compute correlation heatmap.")

        returns_df = pd.DataFrame(returns_dict).dropna(how='all', axis=0)
        corr_matrix = returns_df.corr()

        # Use plotly express to create a heatmap
        fig = px.imshow(corr_matrix,
                        text_auto=True,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        title="Correlation Heatmap of Daily Returns")
        fig.update_layout(template="plotly_dark")
        return fig

    def plot_cumulative_returns(self, portfolio_data: Dict[str, pd.DataFrame],
                                price_col: str = 'Close') -> go.Figure:
        """
        Plots cumulative returns of each symbol in 'portfolio_data' on the same chart.
        :param portfolio_data: dict[symbol -> DataFrame with OHLC data]
        :param price_col: 'Close' or whichever price column you want to use
        :return: Plotly Figure (line chart)
        """
        fig = go.Figure()
        for symbol, df in portfolio_data.items():
            if price_col in df.columns and not df.empty:
                sorted_df = df.sort_index()
                daily_returns = sorted_df[price_col].pct_change().fillna(0)
                # Compute cumulative returns
                cum_returns = (1 + daily_returns).cumprod() - 1.0
                fig.add_trace(
                    go.Scatter(x=cum_returns.index,
                               y=cum_returns,
                               mode='lines',
                               name=symbol))
        fig.update_layout(title="Cumulative Returns Over Time",
                          xaxis_title="Date",
                          yaxis_title="Cumulative Return",
                          template="plotly_dark",
                          hovermode="x unified")
        return fig

    def plot_drawdown(self, portfolio_data: Dict[str, pd.DataFrame],
                      symbol: str,
                      price_col: str = 'Close') -> go.Figure:
        """
        Plots the drawdown for a single symbol (peak-to-trough declines over time).
        :param portfolio_data: dict[symbol -> DataFrame with OHLC data]
        :param symbol: symbol key to use from portfolio_data
        :param price_col: 'Close' or whichever price column you want
        :return: Plotly Figure with drawdown line
        """
        if symbol not in portfolio_data:
            raise ValueError(f"Symbol '{symbol}' not found in portfolio_data.")
        df = portfolio_data[symbol]
        if df.empty or price_col not in df.columns:
            raise ValueError(
                f"No valid data for symbol '{symbol}' to plot drawdown.")

        sorted_df = df.sort_index()
        # Compute daily returns
        daily_returns = sorted_df[price_col].pct_change().fillna(0)
        cum_returns = (1 + daily_returns).cumprod()

        # Rolling max of the cumulative returns
        rolling_max = cum_returns.cummax()
        # Drawdown is the percentage drop from the rolling max
        drawdown = (cum_returns - rolling_max) / rolling_max

        fig = make_subplots(rows=1, cols=1)
        # Plot cumulative returns for context
        fig.add_trace(go.Scatter(x=cum_returns.index,
                                 y=cum_returns,
                                 name="Cumulative Returns",
                                 mode="lines",
                                 line=dict(color='lightgreen')),
                      row=1,
                      col=1)

        # Plot drawdown on a secondary axis by using negative values or add second axis if you prefer
        fig.add_trace(go.Scatter(x=drawdown.index,
                                 y=drawdown,
                                 name="Drawdown",
                                 mode="lines",
                                 line=dict(color='red')),
                      row=1,
                      col=1)

        fig.update_layout(title=f"{symbol} Cumulative Returns & Drawdown",
                          template='plotly_dark',
                          hovermode='x unified')
        # We could add a second y-axis for drawdowns if you want them separate
        return fig

    def plot_performance_and_drawdown(self, portfolio_data: Dict[str, pd.DataFrame],
                                      symbol: str,
                                      price_col: str = 'Close') -> go.Figure:
        """
        Shows both the cumulative returns and drawdown on two subplots.
        """
        if symbol not in portfolio_data:
            raise ValueError(f"Symbol '{symbol}' not found in portfolio_data.")
        df = portfolio_data[symbol]
        if df.empty or price_col not in df.columns:
            raise ValueError(
                f"No valid data for symbol '{symbol}' to plot performance.")

        sorted_df = df.sort_index()
        daily_returns = sorted_df[price_col].pct_change().fillna(0)
        cum_returns = (1 + daily_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max

        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            subplot_titles=[
                                f"{symbol} Cumulative Returns",
                                f"{symbol} Drawdown"
                            ])

        # Plot cumulative returns (Row 1)
        fig.add_trace(go.Scatter(x=cum_returns.index,
                                 y=cum_returns,
                                 name="Cumulative Returns",
                                 mode="lines",
                                 line=dict(color='lightblue')),
                      row=1,
                      col=1)

        # Plot drawdown (Row 2)
        fig.add_trace(go.Scatter(x=drawdown.index,
                                 y=drawdown,
                                 name="Drawdown",
                                 mode="lines",
                                 line=dict(color='red')),
                      row=2,
                      col=1)

        fig.update_layout(title=f"{symbol} Performance & Drawdown",
                          template='plotly_dark',
                          hovermode='x unified',
                          height=800)
        return fig

    @staticmethod
    def plot_discrete_actions(info_history: List[Dict],
                              fig: Optional[go.Figure] = None) -> go.Figure:
        """Create scatter plot of discrete actions over time."""
        if fig is None:
            fig = go.Figure()

        # Extract actions and dates
        dates = [info['date'] for info in info_history]
        actions = [
            info['actions'][0]
            if isinstance(info['actions'], list) else info['actions']
            for info in info_history
        ]

        # Create scatter plot for each action type
        action_colors = {0: 'gray', 1: 'green', 2: 'red'}
        action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

        for action_value in [0, 1, 2]:
            mask = [a == action_value for a in actions]
            if any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=[d for d, m in zip(dates, mask) if m],
                        y=[action_value] * sum(mask),
                        mode='markers',
                        name=action_names[action_value],
                        marker=dict(color=action_colors[action_value], size=8),
                    ))

        fig.update_layout(title='Agent Actions Over Time',
                          xaxis_title='Date',
                          yaxis=dict(
                              title='Action',
                              ticktext=['Hold (0)', 'Buy (1)', 'Sell (2)'],
                              tickvals=[0, 1, 2],
                          ),
                          showlegend=True)

        return fig

    @staticmethod
    def plot_actions_with_price(info_history: List[Dict],
                                price_data: pd.DataFrame) -> go.Figure:
        """Create combined plot of price and actions."""
        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=[0.7, 0.3])

        # Add candlestick chart
        fig.add_trace(go.Candlestick(x=price_data.index,
                                     open=price_data['Open'],
                                     high=price_data['High'],
                                     low=price_data['Low'],
                                     close=price_data['Close'],
                                     name='Price'),
                      row=1,
                      col=1)

        # Add actions to bottom subplot
        action_fig = TradingVisualizer.plot_discrete_actions(info_history)
        for trace in action_fig.data:
            fig.add_trace(trace, row=2, col=1)

        fig.update_layout(height=800,
                          title='Price and Agent Actions',
                          showlegend=True)

        return fig
