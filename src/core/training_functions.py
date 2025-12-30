"""
Core training functionality module
Encapsulates training-related functions from the training tab
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import optuna
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.core.base_agent import UnifiedTradingAgent
from src.utils.callbacks import ProgressBarCallback
from src.utils.stock_utils import parse_stock_list
from src.core.visualization import TradingVisualizer
from src.metrics.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)

def initialize_training(stock_names: List[str], train_start_date: datetime,
                        train_end_date: datetime,
                        env_params: Dict[str, Any]) -> None:
    """
    Initialize training environment and model
    """
    if 'model' not in st.session_state:
        st.session_state.model = UnifiedTradingAgent()

    st.session_state.stock_names = stock_names
    st.session_state.train_start_date = train_start_date
    st.session_state.train_end_date = train_end_date
    st.session_state.env_params = env_params

    # Set logging level based on session state
    if st.session_state.get('enable_logging', False):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)


def execute_training(
        ppo_params: Dict[str, Any],
        progress_bar: Optional[st.progress] = None,
        status_placeholder: Optional[st.empty] = None) -> Dict[str, float]:
    """
    Execute model training with given parameters
    """
    progress_callback = None
    if progress_bar and status_placeholder:
        # Correct the timestep calculation: use end_date - start_date
        approx_data_points = max(1, (st.session_state.train_end_date - st.session_state.train_start_date).days)
        schedule_estimate = st.session_state.model._build_training_schedule(
            approx_data_points, st.session_state.get('schedule_config'))
        total_timesteps = schedule_estimate['total_timesteps']
        logger.info(f"Training for ~{total_timesteps} timesteps (scheduled)")
        progress_callback = ProgressBarCallback(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder)

    # Get feature configuration from session state
    feature_config = st.session_state.get('feature_config', None)
    if feature_config:
        logger.info(f"Using feature config: {feature_config.get('use_feature_engineering', False)}")

    return st.session_state.model.train(
        stock_names=st.session_state.stock_names,
        start_date=st.session_state.train_start_date,
        end_date=st.session_state.train_end_date,
        env_params=st.session_state.env_params,
        ppo_params=ppo_params,
        callback=progress_callback,
        feature_config=feature_config,
        validation_split=st.session_state.get('validation_split', 0.2),
        schedule_config=st.session_state.get('schedule_config'),
        eval_freq=st.session_state.get('eval_frequency', 500),
        eval_seeds=st.session_state.get('eval_seeds', [7, 21]))


def get_training_parameters(use_optuna_params: bool = False) -> Dict[str, Any]:
    """
    Get training parameters either from manual input or optuna optimization
    """
    if use_optuna_params:
        if st.session_state.ppo_params is not None:
            st.info("Using Optuna's optimized parameters")
            return st.session_state.ppo_params
        else:
            st.warning(
                "No Optuna parameters available. Please run hyperparameter tuning first."
            )
            return {}
    else:
        params = {}
        col3, col4 = st.columns(2)
        with col3:
            params['learning_rate'] = st.number_input("Learning Rate",
                                                      value=3e-4,
                                                      format="%.1e")
            params['n_steps'] = st.number_input("PPO Steps Per Update",
                                                value=512)
            params['batch_size'] = st.number_input("Batch Size", value=128)
            params['n_epochs'] = st.number_input("Number of Epochs", value=5)
        with col4:
            params['gamma'] = st.number_input("Gamma (Discount Factor)",
                                              value=0.99)
            params['clip_range'] = st.number_input("Clip Range", value=0.2)
            params['target_kl'] = st.number_input("Target KL Divergence",
                                                  value=0.05)
        return params


def display_training_metrics(metrics: Dict[str, float]) -> None:
    """Display training metrics from portfolio manager."""
    if not metrics:
        return

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.metric("Maximum Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    with metrics_col2:
        st.metric("Return", f"{metrics.get('total_return', 0):.2%}")
        st.metric("Total Value", f"${metrics.get('total_value', 0):,.2f}")
    with metrics_col3:
        st.metric("Total Trades", metrics.get('total_trades', 0))
        st.metric("Cash Balance", f"${metrics.get('cash_balance', 0):,.2f}")


def display_insample_performance(trade_history: List[Dict], metrics: Dict[str, float],
                                  portfolio_manager, stock_names: List[str]) -> None:
    """
    Display comprehensive in-sample performance statistics after training.
    Shows portfolio evolution, trade analysis, and comparison with buy & hold.
    """
    if not trade_history or len(trade_history) == 0:
        st.info("No trading history available for in-sample analysis")
        return

    st.markdown("---")
    st.subheader("In-Sample Performance Analysis")

    # Create tabs for organized display
    perf_tab, trades_tab, evolution_tab, benchmark_tab = st.tabs([
        "Performance Summary", "Trade Analysis", "Portfolio Evolution", "Benchmark Comparison"
    ])

    # Extract data from trade history
    dates = [info['date'] for info in trade_history]
    portfolio_values = [info['net_worth'] for info in trade_history]
    balances = [info['balance'] for info in trade_history]

    # Calculate returns
    returns = MetricsCalculator.calculate_returns(portfolio_values)

    with perf_tab:
        _display_performance_summary(metrics, returns, portfolio_values, portfolio_manager)

    with trades_tab:
        _display_trade_analysis(portfolio_manager, stock_names)

    with evolution_tab:
        _display_portfolio_evolution(dates, portfolio_values, balances, trade_history, stock_names)

    with benchmark_tab:
        _display_benchmark_comparison(trade_history, portfolio_values, stock_names)


def _display_performance_summary(metrics: Dict, returns: np.ndarray,
                                  portfolio_values: List[float], portfolio_manager) -> None:
    """Display comprehensive performance summary metrics."""
    st.markdown("### Key Performance Indicators")

    # Row 1: Core metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
    with col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with col3:
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")

    # Row 2: Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        volatility = MetricsCalculator.calculate_volatility(returns) if len(returns) > 0 else 0
        st.metric("Annualized Volatility", f"{volatility:.2%}")
    with col2:
        st.metric("Total Trades", metrics.get('total_trades', 0))
    with col3:
        st.metric("Final Portfolio Value", f"${metrics.get('total_value', 0):,.2f}")
    with col4:
        st.metric("Ending Cash", f"${metrics.get('cash_balance', 0):,.2f}")

    # Calculate and display additional statistics
    if len(returns) > 0:
        st.markdown("### Return Distribution")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Daily Return", f"{np.mean(returns):.4%}")
        with col2:
            st.metric("Std Daily Return", f"{np.std(returns):.4%}")
        with col3:
            positive_days = np.sum(returns > 0)
            total_days = len(returns)
            st.metric("Positive Days", f"{positive_days}/{total_days} ({positive_days/total_days:.1%})")
        with col4:
            best_day = np.max(returns) if len(returns) > 0 else 0
            worst_day = np.min(returns) if len(returns) > 0 else 0
            st.metric("Best/Worst Day", f"{best_day:.2%} / {worst_day:.2%}")

        # Return distribution histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns * 100, nbinsx=50, name="Daily Returns"))
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_vline(x=np.mean(returns) * 100, line_dash="solid", line_color="green",
                      annotation_text=f"Mean: {np.mean(returns):.2%}")
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key="returns_dist")


def _display_trade_analysis(portfolio_manager, stock_names: List[str]) -> None:
    """Display detailed trade analysis statistics."""
    trades_df = portfolio_manager.get_trade_history()

    if trades_df.empty:
        st.info("No trades executed during training")
        return

    st.markdown("### Trade Statistics")

    # Separate buys and sells
    buys = trades_df[trades_df['action'] == 'BUY']
    sells = trades_df[trades_df['action'] == 'SELL']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Buy Orders", len(buys))
    with col2:
        st.metric("Total Sell Orders", len(sells))
    with col3:
        total_volume = (trades_df['quantity'].abs() * trades_df['price']).sum()
        st.metric("Total Trading Volume", f"${total_volume:,.2f}")

    # Calculate win rate for sell trades (profit vs loss)
    if len(sells) > 0:
        st.markdown("### Sell Trade Analysis")

        # Calculate P&L for each sell
        sell_pnl = []
        for _, trade in sells.iterrows():
            symbol = trade['symbol']
            sell_price = trade['price']
            cost_basis = trade['cost_basis']
            quantity = abs(trade['quantity'])
            pnl = (sell_price - cost_basis) * quantity
            sell_pnl.append({
                'symbol': symbol,
                'sell_price': sell_price,
                'cost_basis': cost_basis,
                'quantity': quantity,
                'pnl': pnl,
                'pnl_pct': (sell_price - cost_basis) / cost_basis if cost_basis > 0 else 0
            })

        pnl_df = pd.DataFrame(sell_pnl)

        if len(pnl_df) > 0:
            winners = pnl_df[pnl_df['pnl'] > 0]
            losers = pnl_df[pnl_df['pnl'] < 0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                win_rate = len(winners) / len(pnl_df) if len(pnl_df) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1%}")
            with col2:
                avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
                st.metric("Avg Win", f"${avg_win:,.2f}")
            with col3:
                avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
                st.metric("Avg Loss", f"${avg_loss:,.2f}")
            with col4:
                total_wins = winners['pnl'].sum() if len(winners) > 0 else 0
                total_losses = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "N/A")

            # Profit/Loss by symbol
            if len(stock_names) > 1:
                st.markdown("### P&L by Symbol")
                symbol_pnl = pnl_df.groupby('symbol')['pnl'].agg(['sum', 'count', 'mean']).reset_index()
                symbol_pnl.columns = ['Symbol', 'Total P&L', 'Trades', 'Avg P&L']
                symbol_pnl['Total P&L'] = symbol_pnl['Total P&L'].apply(lambda x: f"${x:,.2f}")
                symbol_pnl['Avg P&L'] = symbol_pnl['Avg P&L'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(symbol_pnl, use_container_width=True)

    # Show trade history table
    with st.expander("Full Trade History", expanded=False):
        display_df = trades_df.copy()
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
        display_df['total_cost'] = display_df['total_cost'].apply(lambda x: f"${x:,.2f}")
        display_df['balance_after'] = display_df['balance_after'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True)


def _display_portfolio_evolution(dates: List, portfolio_values: List[float],
                                  balances: List[float], trade_history: List[Dict],
                                  stock_names: List[str]) -> None:
    """Display portfolio value evolution over time."""
    st.markdown("### Portfolio Value Over Time")

    # Create main portfolio value chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Portfolio Value", "Cash Balance", "Asset Allocation"]
    )

    # Portfolio value line
    fig.add_trace(
        go.Scatter(x=dates, y=portfolio_values, name="Portfolio Value",
                   line=dict(color='#00ff88', width=2), fill='tozeroy',
                   fillcolor='rgba(0, 255, 136, 0.1)'),
        row=1, col=1
    )

    # Add peak line for drawdown visualization
    peak_values = np.maximum.accumulate(portfolio_values)
    fig.add_trace(
        go.Scatter(x=dates, y=peak_values, name="Peak Value",
                   line=dict(color='yellow', width=1, dash='dot')),
        row=1, col=1
    )

    # Cash balance
    fig.add_trace(
        go.Scatter(x=dates, y=balances, name="Cash",
                   line=dict(color='#00bfff', width=2)),
        row=2, col=1
    )

    # Asset allocation stacked area
    position_values = {symbol: [] for symbol in stock_names}
    for info in trade_history:
        positions = info.get('positions', {})
        current_data = info.get('current_data', {})
        for symbol in stock_names:
            shares = positions.get(symbol, 0)
            price_key = f'Close_{symbol}'
            price = float(current_data.get(price_key, 0)) if current_data else 0
            position_values[symbol].append(shares * price)

    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181', '#aa96da']
    for idx, symbol in enumerate(stock_names):
        fig.add_trace(
            go.Scatter(x=dates, y=position_values[symbol], name=symbol,
                       stackgroup='positions', fillcolor=colors[idx % len(colors)],
                       line=dict(width=0.5, color=colors[idx % len(colors)])),
            row=3, col=1
        )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cash ($)", row=2, col=1)
    fig.update_yaxes(title_text="Position Value ($)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True, key="portfolio_evolution")

    # Drawdown chart
    st.markdown("### Drawdown Analysis")
    drawdown = [(peak - val) / peak if peak > 0 else 0
                for val, peak in zip(portfolio_values, peak_values)]

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(x=dates, y=[-d * 100 for d in drawdown], name="Drawdown",
                   fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
                   line=dict(color='red', width=1))
    )
    fig_dd.update_layout(
        title="Drawdown (%)",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=250,
        showlegend=False
    )
    st.plotly_chart(fig_dd, use_container_width=True, key="drawdown_chart")


def _display_benchmark_comparison(trade_history: List[Dict], portfolio_values: List[float],
                                  stock_names: List[str]) -> None:
    """Compare strategy performance against buy & hold benchmark."""
    st.markdown("### Buy & Hold Comparison")

    if len(trade_history) < 2:
        st.info("Not enough data for benchmark comparison")
        return

    # Get initial and final prices for each stock
    first_info = trade_history[0]
    last_info = trade_history[-1]

    initial_portfolio_value = portfolio_values[0]

    # Calculate buy & hold returns for each stock
    benchmark_data = []
    for symbol in stock_names:
        price_key = f'Close_{symbol}'
        initial_price = float(first_info.get('current_data', {}).get(price_key, 0))
        final_price = float(last_info.get('current_data', {}).get(price_key, 0))

        if initial_price > 0:
            stock_return = (final_price - initial_price) / initial_price
            benchmark_data.append({
                'symbol': symbol,
                'initial_price': initial_price,
                'final_price': final_price,
                'return': stock_return
            })

    if not benchmark_data:
        st.warning("Could not calculate benchmark returns")
        return

    # Calculate equal-weight buy & hold portfolio return
    equal_weight_return = np.mean([b['return'] for b in benchmark_data])
    strategy_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    spy_price_key = "Close_SPY"
    spy_available = spy_price_key in (first_info.get('current_data') or {})
    spy_return = None
    if spy_available:
        spy_initial = float(first_info.get('current_data', {}).get(spy_price_key, 0))
        spy_final = float(last_info.get('current_data', {}).get(spy_price_key, 0))
        if spy_initial > 0:
            spy_return = (spy_final - spy_initial) / spy_initial
        else:
            spy_available = False

    # Display comparison
    if spy_available:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1, col2, col3 = st.columns(3)
        col4 = None

    with col1:
        st.metric("Strategy Return", f"{strategy_return:.2%}")
    with col2:
        st.metric("Buy & Hold Return (Equal Weight)", f"{equal_weight_return:.2%}")
    with col3:
        excess_return = strategy_return - equal_weight_return
        st.metric("Excess Return (Alpha)", f"{excess_return:.2%}",
                  delta=f"{excess_return:.2%}", delta_color="normal")
    if spy_available and col4 is not None:
        with col4:
            st.metric("SPY Buy & Hold Return", f"{spy_return:.2%}")

    # Individual stock returns table
    st.markdown("### Individual Stock Returns")
    bench_df = pd.DataFrame(benchmark_data)
    bench_df['initial_price'] = bench_df['initial_price'].apply(lambda x: f"${x:.2f}")
    bench_df['final_price'] = bench_df['final_price'].apply(lambda x: f"${x:.2f}")
    bench_df['return'] = bench_df['return'].apply(lambda x: f"{x:.2%}")
    bench_df.columns = ['Symbol', 'Start Price', 'End Price', 'Return']
    st.dataframe(bench_df, use_container_width=True)

    # Create comparison chart
    dates = [info['date'] for info in trade_history]

    # Calculate buy & hold portfolio value over time
    buy_hold_values = []
    for info in trade_history:
        current_data = info.get('current_data', {})
        total_bh_value = 0
        for symbol in stock_names:
            price_key = f'Close_{symbol}'
            current_price = float(current_data.get(price_key, 0))
            initial_price = float(first_info.get('current_data', {}).get(price_key, 1))
            # Equal weight allocation
            allocation = initial_portfolio_value / len(stock_names)
            shares = allocation / initial_price if initial_price > 0 else 0
            total_bh_value += shares * current_price
        buy_hold_values.append(total_bh_value)

    spy_buy_hold_values = None
    if spy_available:
        spy_buy_hold_values = []
        spy_initial = float(first_info.get('current_data', {}).get(spy_price_key, 1))
        spy_shares = initial_portfolio_value / spy_initial if spy_initial > 0 else 0
        for info in trade_history:
            current_data = info.get('current_data', {})
            spy_price = float(current_data.get(spy_price_key, 0))
            spy_buy_hold_values.append(spy_shares * spy_price)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=portfolio_values, name="Strategy",
                   line=dict(color='#00ff88', width=2))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=buy_hold_values, name="Buy & Hold",
                   line=dict(color='#ff6b6b', width=2, dash='dash'))
    )
    if spy_buy_hold_values is not None:
        fig.add_trace(
            go.Scatter(x=dates, y=spy_buy_hold_values, name="SPY Buy & Hold",
                       line=dict(color='#5dade2', width=2, dash='dot'))
        )
    fig.update_layout(
        title="Strategy vs Buy & Hold",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, key="benchmark_comparison")


def run_training(ppo_params: Dict[str, Any]) -> None:
    """
    Executes the training process and displays results
    """
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    metrics = execute_training(ppo_params, progress_bar, status_placeholder)

    if metrics:
        st.subheader("Parameters Used for Training")
        col1, col2, col3 = st.columns(3)
        index_col = 0
        all_cols = [col1, col2, col3]
        for param, value in ppo_params.items():
            with all_cols[index_col % 3]:
                st.metric(param, value)
                index_col += 1

        display_training_metrics(metrics)

    # Display comprehensive in-sample performance analysis
    if hasattr(st.session_state.model.env, '_trade_history'):
        trade_history = st.session_state.model.env._trade_history
        stock_names = st.session_state.get('stock_names', [])
        portfolio_manager = st.session_state.model.env.portfolio_manager

        # Show quick trade history visualization first
        TradingVisualizer.display_trade_history(
            trade_history, "Training History", "training_trade")

        # Then show comprehensive in-sample analysis
        if metrics and trade_history:
            display_insample_performance(
                trade_history, metrics, portfolio_manager, stock_names)

    st.session_state.ppo_params = ppo_params
    st.success("Training completed and model saved!")
