import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from gymnasium import spaces

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TradingEnv(gym.Env):
    """
    Consolidated trading environment that combines features from all implementations:
    - Multi-asset support from MarketEnvironment
    - Advanced reward shaping from TradingEnvironment
    - Comprehensive logging and tracking from TradingEnv
    """

    def __init__(self,
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.0,
                 position_size: float = 0.2,
                 use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False,
                 training_mode: bool = False,
                 log_frequency: int = 30,
                 stock_names: Optional[List[str]] = None):
        super().__init__()

        self._portfolio_history = []
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty
        self.training_mode = training_mode
        self.log_frequency = log_frequency
        self.position_size = position_size

        self.data = data
        # Extract symbols from column names (e.g., Close_AAPL -> AAPL)
        try:
            self.symbols = sorted(list({col.split('_')[1] for col in self.data.columns if '_' in col and len(col.split('_')) == 2}))
            if not self.symbols:
                raise ValueError("No valid symbols found in data columns")
            logger.info(f"Extracted symbols: {self.symbols}")
        except Exception as e:
            logger.error(f"Error extracting symbols from columns: {self.data.columns}")
            raise ValueError(f"Failed to extract valid symbols from data columns: {e}")

        # Action space: 0=hold, 1=buy, 2=sell for each asset
        self.action_space = self.create_action_space(self.symbols,
                                                     num_actions=3)

        # Observation space includes OHLCV + positions + balance for each asset
        obs_dim = (len(self.symbols) *
                   6) + 1  # OHLCV + position for each asset + balance
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(obs_dim, ),
                                            dtype=np.float32)

        # Initialize trading state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.cost_bases = {symbol: 0.0 for symbol in self.symbols}
        self.holding_periods = {symbol: 0 for symbol in self.symbols}
        self.net_worth = initial_balance
        self.current_step = 0
        self.last_logged_step = -1
        self.episode_count = 0
        self.total_steps = 0
        self.episode_trades = {symbol: 0 for symbol in self.symbols}
        self.transaction_cost = transaction_cost

    @staticmethod
    def create_action_space(symbols, num_actions: int = 3) -> gym.Space:
        """
        Create a Gym action space for a trading environment.

        Args:
            num_symbols (int): Number of symbols (assets) being traded.
            num_actions (int): Number of discrete actions per asset. 
                               For example, 3 could represent [Hold, Buy, Sell].

        Returns:
            gym.Space: A Gym action space. 
                       If multiple symbols, uses MultiDiscrete, otherwise Discrete.
        """
        num_symbols = len(symbols)
        if num_symbols > 1:
            return spaces.MultiDiscrete([num_actions] * num_symbols)
        return spaces.Discrete(num_actions)

    def _get_observation(self) -> np.ndarray:
        """Get current observation of market and account state."""
        obs = []
        for symbol in self.symbols:
            try:
                data_step = self.data.iloc[self.current_step]
                obs.extend([
                    float(data_step[f'Open_{symbol}']),
                    float(data_step[f'High_{symbol}']),
                    float(data_step[f'Low_{symbol}']),
                    float(data_step[f'Close_{symbol}']),
                    float(data_step[f'Volume_{symbol}']),
                    float(self.positions[symbol])
                ])
            except Exception as e:
                logger.error(f"Error getting observation for {symbol}: {e}")
                raise
        obs.append(float(self.balance))
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, prev_net_worth: float, current_net_worth: float,
                        actions: Union[int, np.ndarray],
                        trades_executed: Dict[str, bool]) -> float:
        """Calculate reward based on trading performance and behavior."""
        reward = 0.0

        # Penalize attempting trades without sufficient balance
        for symbol, executed in trades_executed.items():
            if not executed and isinstance(actions, (list, np.ndarray)):
                idx = self.symbols.index(symbol)
                if actions[idx] == 1:  # Attempted buy
                    reward -= 0.01  # Small penalty for failed trades

        # Portfolio return reward
        if prev_net_worth > 0:
            portfolio_return = (current_net_worth -
                                prev_net_worth) / prev_net_worth
            reward += portfolio_return

        if self.use_holding_bonus:
            # Add holding bonus for profitable positions
            for symbol in self.symbols:
                current_price = self.data[symbol].iloc[
                    self.current_step]['Close']
                if self.positions[
                        symbol] > 0 and current_price > self.cost_bases[symbol]:
                    reward += 0.001 * self.holding_periods[symbol]

        if self.use_trading_penalty:
            # Penalize excessive trading
            trade_penalty = sum(trades_executed.values()) * 0.0001
            reward -= trade_penalty

        return reward

    def reset(self, seed=None, options=None):
        """Reset the portfolio and balance to initial state."""
        self._portfolio_history = []
        self._trade_history = []  # Reset trade history
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.cost_bases = {symbol: 0.0 for symbol in self.symbols}
        self.holding_periods = {symbol: 0 for symbol in self.symbols}
        self.net_worth = self.initial_balance
        self.last_logged_step = -1
        self.episode_trades = {symbol: 0 for symbol in self.symbols}
        self.episode_count += 1

        # if seed is not None:
        #     super().reset(seed=seed)
        # Prepare info dict
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'trades_executed': {symbol: False for symbol in self.symbols},
            'episode_trades': {symbol: 0 for symbol in self.symbols},
            'actions': {},  # Store actions in info
            'date': self.data.index[self.current_step],
            'current_data': self.data.iloc[self.current_step]
        }
        return self._get_observation(), info

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.total_steps += 1

        # Convert single action to list for unified processing
        actions = [action] if isinstance(action, (int, np.integer)) else action
        if len(actions) != len(self.symbols):
            raise ValueError(
                f"Expected {len(self.symbols)} actions, got {len(actions)}")

        # Store previous state
        prev_net_worth = self.net_worth
        trades_executed = {symbol: False for symbol in self.symbols}

        # Process actions for each asset
        for idx, symbol in enumerate(self.symbols):
            # logger.info(f"PROCESSING SYMBOL {symbol:5}")
            current_price = float(
                self.data.iloc[self.current_step][f'Close_{symbol}'])
            action = int(actions[idx])

            if action == 1:  # Buy
                # Calculate maximum affordable shares considering transaction cost

                max_trade_amount = max(0, self.balance - self.transaction_cost)
                trade_amount = min(max_trade_amount * self.position_size,
                                   max_trade_amount)
                shares_to_buy = trade_amount / current_price if current_price > 0 else 0
                total_cost = (shares_to_buy *
                              current_price) + self.transaction_cost

                logger.info(f"PROCESSING SYMBOL {symbol:5} AND CHECKING BALANCE")
                # Only execute trade if we can buy at least 0.01 shares
                if total_cost <= self.balance and shares_to_buy >= 0.01:
                    # logger.info(
                    #     f"BUY  | {symbol:5} | Price: ${current_price:.2f} | Shares: {shares_to_buy:.4f} | Cost: ${total_cost:.2f}"
                    # )
                    self.balance -= total_cost
                    if self.positions[symbol] > 0:
                        old_cost = self.cost_bases[symbol] * self.positions[
                            symbol]
                        new_cost = current_price * shares_to_buy
                        self.cost_bases[symbol] = (old_cost + new_cost) / (
                            self.positions[symbol] + shares_to_buy)
                    else:
                        self.cost_bases[symbol] = current_price
                    self.positions[symbol] += shares_to_buy
                    self.holding_periods[symbol] = 0
                    self.episode_trades[symbol] += 1
                    trades_executed[symbol] = True
                    logger.info(f"PROCESSING SYMBOL {symbol:5} AND BUYING!! FOR A TOTAL OF {shares_to_buy:.4f} SHARES and a cost of ${total_cost:.2f}")

            elif action == 2:  # Sell
                if self.positions[symbol] > 0:
                    shares_to_sell = self.positions[symbol]
                    sell_amount = shares_to_sell * current_price
                    net_sell_amount = sell_amount - self.transaction_cost

                    # logger.info(
                    #     f"SELL | {symbol:5} | Price: ${current_price:.2f} | Shares: {shares_to_sell:.4f} | Amount: ${net_sell_amount:.2f}"
                    # )

                    self.balance += net_sell_amount
                    self.positions[symbol] = 0
                    self.holding_periods[symbol] = 0
                    self.episode_trades[symbol] += 1
                    trades_executed[symbol] = True
                    logger.info(f"PROCESSING SYMBOL {symbol:5} AND SELLING!! FOR A TOTAL OF {shares_to_sell:.4f} SHARES and a cost of ${net_sell_amount:.2f}")

            # Update holding period for non-zero positions
            if self.positions[symbol] > 0:
                self.holding_periods[symbol] += 1

        # Update portfolio value
        self.net_worth = self.balance + sum(
            self.positions[symbol] *
            self.data.iloc[self.current_step][f'Close_{symbol}']
            for symbol in self.symbols)
        self._portfolio_history.append(self.net_worth)

        # Calculate reward
        reward = self._compute_reward(prev_net_worth, self.net_worth, actions,
                                      trades_executed)

        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False

        # Get next observation
        observation = self._get_observation()

        # Prepare info dict
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'trades_executed': trades_executed,
            'episode_trades': self.episode_trades.copy(),
            'actions': actions,  # Store actions in info
            'date': self.data.index[self.current_step],
            'current_data': self.data.iloc[self.current_step]
        }
        self.last_info = info  # Store last info for callbacks

        # Store trade history
        if not hasattr(self, '_trade_history'):
            self._trade_history = []
        self._trade_history.append(info)

        return observation, reward, done, truncated, info

    def get_portfolio_history(self) -> List[float]:
        """Return the history of portfolio values."""
        return self._portfolio_history