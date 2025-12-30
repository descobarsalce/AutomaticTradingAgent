from datetime import datetime
import gymnasium as gym
from src.metrics.metrics_calculator import MetricsCalculator
from src.environment.rewards_calculator import RewardsCalculator
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from gymnasium import spaces
from src.utils.common import (MAX_POSITION_SIZE, MIN_POSITION_SIZE, MIN_TRADE_SIZE,
                          POSITION_PRECISION)
from src.core.portfolio_manager import PortfolioManager
from src.data.data_handler import TradingDataManager
from src.data.providers import DataProvider
from src.metrics.metric_sink import MetricsSink

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_verbosity = "Low"  # "High" or "Low"

# Try to import feature processor (optional dependency)
try:
    from src.data.feature_engineering.feature_processor import FeatureProcessor
    HAS_FEATURE_PROCESSOR = True
except ImportError:
    HAS_FEATURE_PROCESSOR = False
    FeatureProcessor = None


def fetch_trading_data(stock_names: List[str], start_date: datetime,
                       end_date: datetime,
                       provider: DataProvider) -> pd.DataFrame:
    """Fetch preprocessed trading data using the data manager."""
    manager = TradingDataManager(provider)
    return manager.fetch(stock_names, start_date, end_date)


class TradingEnv(gym.Env):

    def __init__(self,
                 stock_names: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.0,
                 transaction_cost_bps: float = 0.0,
                 slippage_bps: float = 0.0,
                 max_daily_loss_pct: Optional[float] = None,
                 max_pct_position_by_asset: float = 0.2,
                 use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False,
                 training_mode: bool = True,
                 observation_days: int = 2,
                 burn_in_days: int = 20,
                 feature_config: Optional[Dict[str, Any]] = None,
                 feature_pipeline: Optional[Any] = None,
                 provider: DataProvider = None,
                 risk_budget_params: Optional[Dict[str, float]] = None,
                 metrics_sink: Optional[MetricsSink] = None,
                 risk_window: int = 20):  # Feature engineering config
        # Fetch trading data
        if provider is None:
            raise ValueError("TradingEnv requires an explicit data provider")

        self.provider = provider
        data = fetch_trading_data(stock_names, start_date, end_date, self.provider)
        # Data arrives fully preprocessed from the TradingDataManager.
        self._validate_init_params(data, initial_balance, transaction_cost,
                                   transaction_cost_bps, slippage_bps,
                                   max_daily_loss_pct,
                                   max_pct_position_by_asset)
        self._full_data = data.copy()
        self.data = data
        self.max_pct_position_by_asset = max_pct_position_by_asset
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_daily_loss_pct = max_daily_loss_pct
        self.stock_names = stock_names
        self.risk_budget_params = risk_budget_params or {}
        self.metrics_sink = metrics_sink or MetricsSink()
        self.risk_window = risk_window
        self._leakage_warnings: set[str] = set()

        # Set reward shaping flags
        # Initialize rewards calculator
        self.rewards_calculator = RewardsCalculator(
            use_position_profit=use_position_profit,
            use_holding_bonus=use_holding_bonus,
            use_trading_penalty=use_trading_penalty)
        self.training_mode = training_mode

        # Initialize portfolio manager before constructing observation space.
        self.portfolio_manager = PortfolioManager(
            initial_balance,
            transaction_cost,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            risk_budget_params=self.risk_budget_params,
            metrics_sink=self.metrics_sink)

        # initialize state & observation space.
        self.observation_days = observation_days  # Store the number of days to keep in the observation
        self.burn_in_days = burn_in_days

        # Initialize feature processor if configured
        self.feature_config = feature_config
        self.feature_processor = None
        self._processed_data = None

        if feature_config:
            logger.warning(
                "feature_config-based initialization is deprecated; provide a prebuilt feature_pipeline instead.")

        if feature_pipeline is not None:
            self._processed_data = self._prepare_features(feature_pipeline, data)

        self._initialize_state()
        self.observation_space = self._create_observation_space(
        )  # Ensure observation space is created before use

        # Set the action space to Box with shape (n_stocks + gate,) where the
        # final component controls the blending gate in [0, 1].
        lows = np.concatenate([np.full(len(stock_names), -1.0), np.array([0.0])])
        highs = np.concatenate([np.full(len(stock_names), 1.0), np.array([1.0])])
        self.action_space = spaces.Box(low=lows, high=highs, dtype=np.float32)

    def _validate_init_params(self, data: Union[pd.DataFrame,
                                                Dict[str, pd.DataFrame]],
                              initial_balance: float, transaction_cost: float,
                              transaction_cost_bps: float, slippage_bps: float,
                              max_daily_loss_pct: Optional[float],
                              max_pct_position_by_asset: float) -> None:
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if transaction_cost_bps < 0:
            raise ValueError("Transaction cost bps cannot be negative")
        if slippage_bps < 0:
            raise ValueError("Slippage bps cannot be negative")
        if max_daily_loss_pct is not None and max_daily_loss_pct < 0:
            raise ValueError("Max daily loss pct cannot be negative")
        if not 0 < max_pct_position_by_asset <= 1:
            raise ValueError("Position size must be between 0 and 1")
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError("Empty data provided")

    def _initialize_state(self) -> None:
        self.current_step = 0
        self.episode_trades = {symbol: 0 for symbol in self.stock_names}
        self.observation_history: List[np.ndarray] = [
        ]  # New observation history
        self.gate_history: List[float] = []

    # --- Auxiliary functions to reduce repetition ---
    def _get_current_price(self, symbol: str, open_price=True) -> float:
        """Retrieve the current closing price for a symbol using _full_data."""
        if open_price:
            return float(self._full_data.iloc[self.current_step][f'Open_{symbol}'])
        else:
            return float(self._full_data.iloc[self.current_step][f'Close_{symbol}'])
            
    # def _get_yesterday_high(self, symbol: str) -> float:
    #     """Retrieve the current high price for a symbol using _full_data."""
    #     return float(
    #         self._full_data.iloc[self.current_step][f'High_{symbol}'])
    # def _get_yesterday_low(self, symbol: str) -> float:
    #     """Retrieve the current low price for a symbol using _full_data."""
    #     return float(
    #         self._full_data.iloc[self.current_step][f'Low_{symbol}'])
    # def _get_yesterday_volume(self, symbol: str) -> float:
    #     """Retrieve the current volume for a symbol using _full_data."""
    #     return float(
    #         self._full_data.iloc[self.current_step][f'Volume_{symbol}'])

    def _get_current_data(self) -> pd.Series:
        """Retrieve the current data row."""
        return self.data.iloc[self.current_step]

    def _compute_risk_controls(self) -> Dict[str, float]:
        """Derive forecast volatility, correlation, and drawdown inputs."""

        forecast_vol = 0.0
        corr_proxy = 0.0
        close_cols = [f'Close_{symbol}' for symbol in self.stock_names]
        start_idx = max(0, self.current_step - self.risk_window + 1)

        if self.current_step > 0 and all(col in self._full_data.columns for col in close_cols):
            window_frame = self._full_data.iloc[start_idx:self.current_step + 1][close_cols]
            returns = window_frame.pct_change().dropna()
            if not returns.empty:
                forecast_vol = float(returns.stack().std(ddof=0))
                corr_proxy = float(returns.corr().abs().mean().mean()) if returns.shape[0] > 1 else 0.0

        drawdown = float(self.portfolio_manager.max_drawdown or 0.0)
        return {
            'forecast_volatility': forecast_vol,
            'correlation_proxy': corr_proxy,
            'drawdown': drawdown,
        }

    # --- End Auxiliary functions ---

    def _create_observation_space(self) -> spaces.Box:
        # Define a fixed observation space shape based on observation_days.
        if self.feature_processor is not None:
            # Use feature processor's observation size
            n_features = self.feature_processor.get_observation_size()
        elif self._processed_data is not None:
            n_features = len(self._processed_data.columns) + len(self.stock_names) + 1
        else:
            # Default: prices + positions + balance
            n_features = len(self.stock_names) * 2 + 1

        obs_dim = self.observation_days * n_features
        # Use wider range for feature-engineered observations
        low = -np.inf if self.feature_processor else -1
        high = np.inf if self.feature_processor else 1
        return spaces.Box(low=low, high=high, shape=(obs_dim, ), dtype=np.float32)

    def _construct_observation(self) -> np.ndarray:
        """Collect current/past prices, positions, and balance. This is the information available             to the agent in a given day."""
        # Construct current observation
        if self.feature_processor is not None and self._processed_data is not None:
            # Use feature processor for rich observations
            masked_data = self._processed_data
            masked_row = self._mask_unavailable_features(
                self._processed_data.iloc[self.current_step],
                self.data.index[self.current_step],
            )
            if not masked_row.equals(self._processed_data.iloc[self.current_step]):
                masked_data = self._processed_data.copy()
                masked_data.iloc[self.current_step] = masked_row
            current_obs = self.feature_processor.get_observation_vector(
                data=masked_data,
                current_index=self.current_step,
                positions=self.portfolio_manager.positions,
                balance=self.portfolio_manager.current_balance,
            )
        elif self._processed_data is not None:
            feature_row = self._processed_data.iloc[self.current_step]
            feature_row = self._mask_unavailable_features(feature_row, self.data.index[self.current_step])
            feature_array = feature_row.to_numpy(dtype=np.float32)
            positions = np.array(
                [self.portfolio_manager.positions.get(symbol, 0.0) for symbol in self.stock_names],
                dtype=np.float32)
            balance = np.array([self.portfolio_manager.current_balance], dtype=np.float32)
            current_obs = np.concatenate([feature_array, positions, balance])
        else:
            # Default observation: prices + positions + balance
            current_obs = []
            for symbol in self.stock_names:
                current_price = self._get_current_price(symbol)
                position = self.portfolio_manager.positions.get(symbol, 0.0)
                current_obs.extend([current_price, position])
            current_obs.append(self.portfolio_manager.current_balance)
            current_obs = np.array(current_obs, dtype=np.float32)

        if log_verbosity == "High":
            logger.info(f"Current observation: {current_obs}")

        # Add to history
        self.observation_history.append(current_obs)

        if len(self.observation_history) < self.observation_days:
            missing = self.observation_days - len(self.observation_history)
            pad = [self.observation_history[0]
                   ] * missing if self.observation_history else [
                       np.zeros(self.observation_space.shape[0] //
                                self.observation_days,
                                dtype=np.float32)
                   ] * missing
            history = pad + self.observation_history
        else:
            history = self.observation_history[
                -self.
                observation_days:]  # returns a fixed amnount of past days as input (history)
        combined_obs = np.concatenate(history, axis=0)

        # New: Check and replace NaNs in the observation.
        if np.isnan(combined_obs).any():
            logger.warning(
                "NaN values detected in observation; replacing with zeros.")
            combined_obs = np.nan_to_num(combined_obs)

        if log_verbosity == "High":
            # Log full observation with history and current observation separately
            logger.info(
                f"Full observation (including history):\nShape: {combined_obs.shape}\nData: {combined_obs}"
            )
            logger.info(
                f"Current observation (this date):\nShape: {current_obs.shape}\nData: {current_obs}"
            )

        return combined_obs

    def use_rewards_calculator(self, trades_executed: Dict[str, bool]) -> float:
        """Compute reward using the specialized RewardsCalculator."""
        try:
            for symbol in self.stock_names:
                close_price = self._get_current_price(symbol, open_price=False)
                self.portfolio_manager._update_metrics(close_price, symbol)
            history = self.portfolio_manager.portfolio_value_history
            turnover = self.portfolio_manager.calculate_turnover()
            portfolio_value = self.portfolio_manager.get_total_value()
            reward = self.rewards_calculator.compute_reward(
                portfolio_history=history,
                trades_executed=trades_executed,
                transaction_cost=self.transaction_cost,
                transaction_cost_bps=self.transaction_cost_bps,
                slippage_bps=self.slippage_bps,
                turnover=turnover,
                portfolio_value=portfolio_value)
            logger.debug(f"Computed reward: {reward}")
            return reward
        except Exception as e:
            logger.error(f"Error computing reward: {str(e)}")
            return 0.0

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        try:
            timestamp = self.data.index[self.current_step]
            actions_arr = np.array(action, dtype=np.float32).flatten()
            risk_controls = self._compute_risk_controls()

            if actions_arr.shape[0] == len(self.stock_names) + 1:
                gate_value = float(np.clip(actions_arr[-1], 0.0, 1.0))
                weight_actions = actions_arr[:-1]
            else:
                gate_value = 1.0
                weight_actions = actions_arr[:len(self.stock_names)]

            abs_sum = np.sum(np.abs(weight_actions))
            if abs_sum > 0:
                w_proposed = weight_actions / abs_sum
            else:
                w_proposed = np.zeros_like(weight_actions)

            prices = {
                symbol: self._get_current_price(symbol, open_price=True)
                for symbol in self.stock_names
            }
            w_prev = self.portfolio_manager.get_current_weights(prices)
            w_final = (1 - gate_value) * w_prev + gate_value * w_proposed

            max_weight = self.max_pct_position_by_asset
            w_final = np.clip(w_final, -max_weight, max_weight)

            # Ensure we stay within aggregate exposure limits
            weight_norm = np.sum(np.abs(w_final))
            if weight_norm > 0 and weight_norm > max_weight * len(self.stock_names):
                w_final = w_final * (max_weight * len(self.stock_names) / weight_norm)

            trades_executed = self.portfolio_manager.rebalance_to_weights(
                self.stock_names, w_final, prices, timestamp, risk_controls)
            self.gate_history.append(gate_value)

            # Calculate reward and update environment state.
            reward = self.use_rewards_calculator(trades_executed)
            self.current_step += 1
            obs = self._construct_observation()
            done = self.current_step >= len(self.data) - 1
            hit_daily_loss_cap = False
            if self.max_daily_loss_pct is not None:
                history = self.portfolio_manager.portfolio_value_history
                if len(history) > 1:
                    prev_value = history[-2]
                    current_value = history[-1]
                    if prev_value > 0:
                        daily_return = (current_value - prev_value) / prev_value
                        if daily_return <= -self.max_daily_loss_pct:
                            hit_daily_loss_cap = True
                            done = True

            if log_verbosity == "High":
                # Log one structured table summarizing the step.
                log_table = (
                    "+-------------------------+----------------------------------------------+\n"
                    f"| {'Step':<23} | {self.current_step!s:<44} |\n"
                    f"| {'Timestamp':<23} | {timestamp!s:<44} |\n"
                    f"| {'Balance':<23} | {self.portfolio_manager.current_balance!s:<44} |\n"
                    f"| {'Positions':<23} | {str(self.portfolio_manager.positions):<44} |\n"
                    f"| {'Actions':<23} | {str(action):<44} |\n"
                    f"| {'Trades Exec':<23} | {str(trades_executed):<44} |\n"
                    f"| {'Reward':<23} | {reward!s:<44} |\n"
                    f"| {'Total Value':<23} | {self.portfolio_manager.get_total_value()!s:<44} |\n"
                    "+-------------------------+----------------------------------------------+"
                )
                logger.info(log_table)

            info = {
                'net_worth': self.portfolio_manager.get_total_value(),
                'balance': self.portfolio_manager.current_balance,
                'positions': self.portfolio_manager.positions.copy(),
                'trades_executed': trades_executed,
                'episode_trades': self.episode_trades.copy(),
                'actions': action,
                'gate_value': gate_value,
                'target_weights': w_final.tolist(),
                'date': timestamp,
                'current_data': self._get_current_data(),
                'portfolio_value': self.portfolio_manager.get_total_value(),
                'step': self.current_step,
                'forecast_volatility': risk_controls['forecast_volatility'],
                'correlation_proxy': risk_controls['correlation_proxy'],
                'risk_scale': self.portfolio_manager.risk_ledger[-1]['scale']
                if self.portfolio_manager.risk_ledger else 1.0,
                'hit_daily_loss_cap': hit_daily_loss_cap,
            }
            return obs, reward, done, False, info

        except Exception as e:
            logger.error(f"Error in step method: {str(e)}")
            return self._construct_observation(), 0.0, True, False, {
                'error':
                str(e),
                'net_worth':
                self.portfolio_manager.get_total_value(),
                'balance':
                self.portfolio_manager.current_balance,
                'positions':
                self.portfolio_manager.positions.copy(),
                'trades_executed': {},
                'episode_trades':
                self.episode_trades.copy(),
                'actions':
                action,
                'date':
                self.data.index[self.current_step - 1]
                if self.current_step > 0 else self.data.index[0],
                'current_data':
                self._get_current_data(),
                'portfolio_value':
                self.portfolio_manager.get_total_value(),
                'step':
                self.current_step
            }

    def _verify_observation_consistency(self) -> None:
        """Verify observation history consistency"""
        if len(self.observation_history) == 0:
            raise ValueError("Observation history is empty")

        # Verify shape consistency
        first_obs_shape = self.observation_history[0].shape
        for idx, obs in enumerate(self.observation_history):
            if obs.shape != first_obs_shape:
                raise ValueError(
                    f"Inconsistent observation shape at index {idx}")

        # Verify no NaN values
        if any(np.isnan(obs).any() for obs in self.observation_history):
            raise ValueError("NaN values detected in observation history")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        self._initialize_state()
        self.portfolio_manager = PortfolioManager(
            self.initial_balance,
            self.transaction_cost,
            transaction_cost_bps=self.transaction_cost_bps,
            slippage_bps=self.slippage_bps,
            risk_budget_params=self.risk_budget_params,
            metrics_sink=self.metrics_sink)
        self.portfolio_manager.portfolio_value_history = [self.initial_balance]
        self.current_step = 0
        self.observation_history.clear()

        # Burn-in period: automatically take no-op action (assumed to be action "0")
        burn_in_counter = 0
        while burn_in_counter < self.burn_in_days and not self._burn_in_done():
            obs, reward, done, _, info = self.step(
                np.zeros(len(self.stock_names)))
            burn_in_counter += 1
            if done:
                break

        combined_obs = self._construct_observation()
        info = {
            'net_worth': self.initial_balance,
            'balance': self.portfolio_manager.current_balance,
            'positions': self.portfolio_manager.positions.copy(),
            'trades_executed': {
                symbol: False
                for symbol in self.stock_names
            },
            'episode_trades': self.episode_trades,
            'actions': {},
            'date': self.data.index[self.current_step],
            'current_data': self._get_current_data()
        }

        logger.debug(f"Reset info: {info}")
        return combined_obs, info

    def _burn_in_done(self) -> bool:
        return len(self.observation_history) >= self.observation_days

    def _prepare_features(self, feature_pipeline: Any, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalize feature inputs from the agent into processed data for observation construction."""

        if HAS_FEATURE_PROCESSOR and isinstance(feature_pipeline, FeatureProcessor):
            self.feature_processor = feature_pipeline
            try:
                if hasattr(self.feature_processor, "initialize"):
                    self.feature_processor.initialize(data)
                if hasattr(self.feature_processor, "compute_features"):
                    return self.feature_processor.compute_features(data)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Feature processor failed during initialization: {exc}. Using raw data instead.")
                self.feature_processor = None
            return None

        if isinstance(feature_pipeline, pd.DataFrame):
            if not feature_pipeline.index.equals(data.index):
                logger.warning(
                    "Provided feature data index does not match trading data; aligning via reindex.")
            aligned = feature_pipeline.reindex(data.index).copy()
            filtered = self._filter_leaky_features(aligned)
            return self._apply_availability_mask(filtered)

        if isinstance(feature_pipeline, dict):
            try:
                feature_frame = pd.DataFrame(feature_pipeline)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Unable to convert feature mapping to DataFrame: {exc}")
                return None
            if not feature_frame.index.equals(data.index):
                feature_frame = feature_frame.reindex(data.index)
            filtered = self._filter_leaky_features(feature_frame)
            return self._apply_availability_mask(filtered)

        if isinstance(feature_pipeline, list):
            missing = [col for col in feature_pipeline if col not in data.columns]
            if missing:
                logger.warning(
                    f"Requested feature columns missing from trading data: {missing}. Using available columns only.")
            available = [col for col in feature_pipeline if col in data.columns]
            if available:
                filtered = self._filter_leaky_features(data[available].copy())
                return self._apply_availability_mask(filtered)
        return None

    def _apply_availability_mask(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Mask feature values that are not released at the corresponding timestamps."""
        release_times = getattr(self._full_data, "attrs", {}).get("release_times", {})
        if not release_times:
            return feature_frame

        masked = feature_frame.copy()
        for column in masked.columns:
            release_series = release_times.get(column)
            if release_series is None:
                continue
            if not release_series.index.equals(masked.index):
                release_series = release_series.reindex(masked.index)
            mask = release_series > masked.index
            if mask.any():
                if column not in self._leakage_warnings:
                    logger.warning(
                        "Masking unavailable feature column %s for %d rows",
                        column,
                        int(mask.sum()),
                    )
                    self._leakage_warnings.add(column)
                masked.loc[mask, column] = 0.0
        return masked

    def _mask_unavailable_features(self, feature_row: pd.Series,
                                   timestamp: pd.Timestamp) -> pd.Series:
        """Mask feature values that should not be available at the current timestamp."""
        release_times = getattr(self._full_data, "attrs", {}).get("release_times", {})
        if not release_times:
            return feature_row

        masked = feature_row.copy()
        for column in feature_row.index:
            release_series = release_times.get(column)
            if release_series is None:
                continue
            if timestamp not in release_series.index:
                continue
            release_time = release_series.loc[timestamp]
            if pd.notna(release_time) and release_time > timestamp:
                if column not in self._leakage_warnings:
                    logger.warning(
                        "Masking feature %s with release time %s after timestamp %s",
                        column,
                        release_time,
                        timestamp,
                    )
                    self._leakage_warnings.add(column)
                masked[column] = 0.0
        return masked

    def _filter_leaky_features(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Remove obvious look-ahead feature columns to reduce leakage risk."""
        leak_keywords = ("future", "target", "label", "next", "lead")
        leaky_cols = [
            col for col in feature_frame.columns
            if any(keyword in col.lower() for keyword in leak_keywords)
        ]
        if leaky_cols:
            logger.warning(
                "Dropping potential look-ahead feature columns: %s",
                leaky_cols,
            )
            return feature_frame.drop(columns=leaky_cols)
        return feature_frame

    def verify_env_state(self) -> Dict[str, Any]:
        """Verify environment state consistency"""
        state = {
            'current_step':
            self.current_step,
            'observation_shapes':
            [obs.shape for obs in self.observation_history],
            'has_nans':
            any(np.isnan(obs).any() for obs in self.observation_history),
            'portfolio_balance':
            self.portfolio_manager.current_balance,
            'data_remaining':
            len(self._full_data) - self.current_step
        }
        return state

    def get_portfolio_history(self) -> List[float]:
        return self.portfolio_manager.portfolio_value_history

    def get_gate_history(self) -> List[float]:
        return self.gate_history

    def _prepare_test_results(self,
                              info_history: List[Dict]) -> Dict[str, Any]:
        portfolio_history = self.portfolio_manager.portfolio_value_history
        returns = MetricsCalculator.calculate_returns(portfolio_history)
        # ...existing code...
        result = {
            # ...existing code...
            'metrics': {
                'sharpe_ratio':
                float(MetricsCalculator.calculate_sharpe_ratio(returns)),
                'sortino_ratio':
                float(MetricsCalculator.calculate_sortino_ratio(returns)),
                'information_ratio':
                float(MetricsCalculator.calculate_information_ratio(returns)),
                'max_drawdown':
                float(
                    MetricsCalculator.calculate_maximum_drawdown(
                        portfolio_history)),
                'volatility':
                float(MetricsCalculator.calculate_volatility(returns))
            }
        }
        return result
