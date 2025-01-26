
import optuna
import streamlit as st
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def hyperparameter_tuning(stock_name: str, train_start_date: datetime,
                         train_end_date: datetime, env_params: Dict[str, Any]) -> None:
    st.header("Hyperparameter Tuning Options")

    with st.expander("Tuning Configuration", expanded=True):
        trials_number = st.number_input("Number of Trials",
                                      min_value=1,
                                      value=20,
                                      step=1)
        pruning_enabled = st.checkbox("Enable Early Trial Pruning", value=True)

        st.subheader("Parameter Search Ranges")

        col1, col2 = st.columns(2)
        with col1:
            lr_min = st.number_input("Learning Rate Min",
                                   value=1e-5,
                                   format="%.1e")
            lr_max = st.number_input("Learning Rate Max",
                                   value=5e-4,
                                   format="%.1e")
            steps_min = st.number_input("Steps Min", value=512, step=64)
            steps_max = st.number_input("Steps Max", value=2048, step=64)
            batch_min = st.number_input("Batch Size Min", value=64, step=32)
            batch_max = st.number_input("Batch Size Max", value=512, step=32)

        with col2:
            epochs_min = st.number_input("Training Epochs Min",
                                       value=3,
                                       step=1)
            epochs_max = st.number_input("Training Epochs Max",
                                       value=10,
                                       step=1)
            gamma_min = st.number_input("Gamma Min",
                                      value=0.90,
                                      step=0.01,
                                      format="%.3f")
            gamma_max = st.number_input("Gamma Max",
                                      value=0.999,
                                      step=0.001,
                                      format="%.3f")
            gae_min = st.number_input("GAE Lambda Min",
                                    value=0.90,
                                    step=0.01,
                                    format="%.2f")
            gae_max = st.number_input("GAE Lambda Max",
                                    value=0.99,
                                    step=0.01,
                                    format="%.2f")

        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["sharpe_ratio", "sortino_ratio", "total_return"],
            help="Metric to optimize during hyperparameter search")

    if st.button("Start Hyperparameter Tuning"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner() if pruning_enabled else None)

        def objective(trial: optuna.Trial) -> float:
            try:
                ppo_params = {
                    'learning_rate':
                    trial.suggest_loguniform('learning_rate', lr_min, lr_max),
                    'n_steps':
                    trial.suggest_int('n_steps', steps_min, steps_max),
                    'batch_size':
                    trial.suggest_int('batch_size', batch_min, batch_max),
                    'n_epochs':
                    trial.suggest_int('n_epochs', epochs_min, epochs_max),
                    'gamma':
                    trial.suggest_uniform('gamma', gamma_min, gamma_max),
                    'gae_lambda':
                    trial.suggest_uniform('gae_lambda', gae_min, gae_max),
                }

                status_text.text(
                    f"Trial {trial.number + 1}/{trials_number}: Testing parameters {ppo_params}"
                )

                # Train with current parameters
                metrics = st.session_state.model.train(
                    stock_name=stock_name,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    env_params=env_params,
                    ppo_params=ppo_params)

                # Use Sharpe ratio as optimization metric
                trial_value = metrics.get('sharpe_ratio', float('-inf'))
                progress = (trial.number + 1) / trials_number
                progress_bar.progress(progress)

                return trial_value

            except Exception as e:
                st.error(f"Error in trial {trial.number}: {str(e)}")
                return float('-inf')

        try:
            study.optimize(objective, n_trials=trials_number)

            st.success("Hyperparameter tuning completed!")

            # Create detailed results dataframe
            trials_df = pd.DataFrame([{
                'Trial': t.number,
                'Value': t.value,
                **t.params
            } for t in study.trials if t.value is not None])

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs([
                "Best Parameters", "Optimization History",
                "Parameter Importance"
            ])

            with tab1:
                st.subheader("Best Configuration Found")
                for param, value in study.best_params.items():
                    if param == 'learning_rate':
                        st.metric(f"Best {param}", f"{value:.2e}")
                    elif param in ['gamma', 'gae_lambda']:
                        st.metric(f"Best {param}", f"{value:.4f}")
                    else:
                        st.metric(f"Best {param}", f"{int(value)}")
                st.metric(f"Best {optimization_metric}",
                          f"{study.best_value:.6f}")

            with tab2:
                st.subheader("Trial History")
                history_fig = go.Figure()
                history_fig.add_trace(
                    go.Scatter(x=trials_df.index,
                               y=trials_df['Value'],
                               mode='lines+markers',
                               name='Trial Value'))
                history_fig.update_layout(
                    title='Optimization History',
                    xaxis_title='Trial Number',
                    yaxis_title=optimization_metric.replace('_', ' ').title())
                st.plotly_chart(history_fig)

            with tab3:
                st.subheader("Parameter Importance")
                importance_dict = optuna.importance.get_param_importances(
                    study)
                importance_df = pd.DataFrame({
                    'Parameter':
                    list(importance_dict.keys()),
                    'Importance':
                    list(importance_dict.values())
                }).sort_values('Importance', ascending=True)

                importance_fig = go.Figure()
                importance_fig.add_trace(
                    go.Bar(x=importance_df['Importance'],
                           y=importance_df['Parameter'],
                           orientation='h'))
                importance_fig.update_layout(
                    title='Parameter Importance Analysis',
                    xaxis_title='Relative Importance',
                    yaxis_title='Parameter',
                    height=400)
                st.plotly_chart(importance_fig)

            # Save full results to CSV
            trials_df.to_csv('hyperparameter_tuning_results.csv', index=False)

            # Download button for results
            st.download_button("Download Complete Results CSV",
                               trials_df.to_csv(index=False),
                               "hyperparameter_tuning_results.csv", "text/csv")

            # Save best parameters
            st.session_state.ppo_params = study.best_params

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            logger.exception("Hyperparameter optimization error")


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


def display_analysis_tab(model):
    """
    Renders the technical analysis dashboard tab
    Args:
        model: The trading model instance
    """
    st.header("Technical Analysis Dashboard")

    # Visualization stock selection (separate from training)
    viz_stock_input = st.text_input("Stocks to Visualize (comma-separated)",
                                    value="AAPL, MSFT, GOOGL")
    viz_stocks = parse_stock_list(viz_stock_input)

    # Date selection for visualization
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        viz_start_date = datetime.combine(
            st.date_input("Analysis Start Date",
                          value=datetime.now() - timedelta(days=365)),
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
        generate_analysis(viz_stocks, viz_start_date, viz_end_date, model,
                          show_rsi, show_sma20, show_sma50, rsi_period,
                          num_columns)


def generate_analysis(viz_stocks, viz_start_date, viz_end_date, model,
                      show_rsi, show_sma20, show_sma50, rsi_period,
                      num_columns):
    """
    Generates and displays technical analysis charts
    """
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
                stock, viz_start_date, viz_end_date)

            if not portfolio_data:
                st.error(f"No data available for {stock}")
                continue

            portfolio_data = model.data_handler.prepare_data()

            if stock in portfolio_data:
                data = portfolio_data[stock]

                # Create price chart
                price_fig = go.Figure(data=[
                    go.Candlestick(x=data.index,
                                   open=data['Open'],
                                   high=data['High'],
                                   low=data['Low'],
                                   close=data['Close'],
                                   name=stock)
                ])
                price_fig.update_layout(title=f'{stock} Price History')
                price_charts[stock] = price_fig

                # Create volume chart
                volume_fig = go.Figure()
                volume_fig.add_trace(
                    go.Bar(x=data.index,
                           y=data['Volume'],
                           name=f'{stock} Volume'))
                volume_fig.update_layout(title=f'{stock} Trading Volume')
                volume_charts[stock] = volume_fig

                # Create RSI chart if enabled
                if show_rsi and 'RSI' in data.columns:
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(
                        go.Scatter(x=data.index,
                                   y=data['RSI'] * 100,
                                   name=f'{stock} RSI'))
                    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                    rsi_fig.add_hline(y=30,
                                      line_dash="dash",
                                      line_color="green")
                    rsi_fig.update_layout(
                        title=f'{stock} RSI ({rsi_period} periods)')
                    rsi_charts[stock] = rsi_fig

                # Create Moving Averages chart if enabled
                if show_sma20 or show_sma50:
                    ma_fig = go.Figure()
                    ma_fig.add_trace(
                        go.Scatter(x=data.index,
                                   y=data['Close'],
                                   name=f'{stock} Price'))
                    if show_sma20 and 'SMA_20' in data.columns:
                        ma_fig.add_trace(
                            go.Scatter(x=data.index,
                                       y=data['SMA_20'],
                                       name=f'{stock} SMA 20'))
                    if show_sma50 and 'SMA_50' in data.columns:
                        ma_fig.add_trace(
                            go.Scatter(x=data.index,
                                       y=data['SMA_50'],
                                       name=f'{stock} SMA 50'))
                    ma_fig.update_layout(title=f'{stock} Moving Averages')
                    ma_charts[stock] = ma_fig

        # Display charts using the dynamic grid system
        display_charts_grid(price_charts, "Price Analysis", num_columns)
        display_charts_grid(volume_charts, "Volume Analysis", num_columns)
        display_charts_grid(rsi_charts, "RSI Analysis", num_columns)
        display_charts_grid(ma_charts, "Moving Averages Analysis", num_columns)

        # Advanced Analytics Section
        if model and hasattr(model, 'portfolio_history') and model.portfolio_history:
            st.header("Portfolio Analytics")
            
            # Portfolio Performance
            portfolio_df = pd.DataFrame(model.portfolio_history, columns=['Portfolio Value'])
            st.subheader("Portfolio Value Over Time")
            st.line_chart(portfolio_df)
            
            # Returns Analysis
            if hasattr(model, 'evaluation_metrics') and model.evaluation_metrics.get('returns'):
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    fig = go.Figure(data=[
                        go.Histogram(x=model.evaluation_metrics['returns'],
                                    nbinsx=50)
                    ])
                    fig.update_layout(title="Returns Distribution",
                                    xaxis_title="Return",
                                    yaxis_title="Frequency")
                    st.plotly_chart(fig, use_container_width=True)

                    # Drawdown
                    values = np.array(model.portfolio_history)
                    peak = np.maximum.accumulate(values)
                    drawdowns = (peak - values) / peak
                    st.subheader("Drawdown Over Time")
                    st.area_chart(pd.DataFrame(drawdowns, columns=['Drawdown']))

                with chart_col2:
                    # Cumulative Returns
                    returns = np.diff(values) / values[:-1]
                    cum_returns = pd.DataFrame(
                        np.cumprod(1 + returns) - 1,
                        columns=['Returns'])
                    st.subheader("Cumulative Returns")
                    st.line_chart(cum_returns)

                    # Rolling Volatility
                    rolling_vol = pd.DataFrame(
                        returns, columns=['Returns']).rolling(30).std() * np.sqrt(252)
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
