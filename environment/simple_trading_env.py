import gymnasium as gym
import numpy as np

class SimpleTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super().__init__()
        # Store market data and initial balance
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) if data is not None else 100
        
        # Define action space as continuous values between -1 and 1
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space for OHLCV + position + balance
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # OHLCV + position + balance
            dtype=np.float32
        )
        
    def _get_observation(self):
        """Get current observation of market and account state."""
        obs = np.array([
            self.data.iloc[self.current_step]['Open'],
            self.data.iloc[self.current_step]['High'],
            self.data.iloc[self.current_step]['Low'],
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['Volume'],
            self.shares_held,
            self.balance
        ], dtype=np.float32)
        return obs
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        try:
            super().reset(seed=seed)
            self.current_step = 0
            
            self.balance = self.initial_balance
            self.shares_held = 0
            self.net_worth = self.initial_balance
            self.cost_basis = 0
            
            observation = self._get_observation()
            info = {
                'initial_balance': self.initial_balance,
                'net_worth': self.net_worth,
                'shares_held': self.shares_held
            }
            
            return observation, info
            
        except Exception as e:
            print(f"Error in reset method: {str(e)}")
            raise
        
    def step(self, action):
        """Execute one step in the environment."""
        try:
            # Get current price from market data
            current_price = self.data.iloc[self.current_step]['Close']
            
            # Validate and process action
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            action = float(action[0])  # Extract single action value
            action = np.clip(action, -1.0, 1.0)  # Ensure action is in [-1, 1]
            
            # Calculate position sizing based on continuous action
            amount = self.balance * abs(action)
            
            if action > 0:  # Buy
                shares_bought = amount / current_price
                self.balance -= amount
                self.shares_held += shares_bought
                self.cost_basis = current_price
            elif action < 0:  # Sell
                shares_sold = self.shares_held * abs(action)
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold
            
            # Calculate reward based on portfolio performance
            self.net_worth = self.balance + self.shares_held * current_price
            reward = (self.net_worth - self.initial_balance) / self.initial_balance
            
            # Update state
            self.current_step += 1
            terminated = self.current_step >= len(self.data) - 1
            truncated = False
            
            next_state = self._get_observation()
            
            info = {
                'net_worth': self.net_worth,
                'balance': self.balance,
                'shares_held': self.shares_held,
                'current_price': current_price
            }
            
            return next_state, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step method: {str(e)}")
            raise
