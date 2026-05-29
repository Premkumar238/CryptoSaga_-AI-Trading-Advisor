import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION CONSTANTS ---
# Initial capital for the simulation
INITIAL_ACCOUNT_BALANCE = 10000 
# Transaction fee (e.g., 0.1% for Binance)
TRADING_FEE = 0.001 
# How many previous candles the agent sees at each step
LOOKBACK_WINDOW = 10 

def _calculate_technical_indicators(df):
    """
    Calculates essential technical indicators (SMA, RSI, and MACD) and adds them
    to the DataFrame. This is the 'State' the agent observes.
    """
    # Simple Moving Average (SMA) - Trend Indicator
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI) - Momentum Indicator
    window = 14
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # --- NEW FEATURE: MACD (Moving Average Convergence Divergence) ---
    # MACD is calculated using Exponential Moving Averages (EMA)
    
    # Fast EMA (12 periods)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    # Slow EMA (26 periods)
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    # MACD Line
    df['MACD'] = exp1 - exp2
    # Signal Line (9-period EMA of the MACD Line)
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Drop the first rows containing NaN values due to the indicators
    df.dropna(inplace=True)
    return df

class CryptoTradingEnv(gym.Env):
    """A custom crypto trading environment using the standardized Gymnasium API."""
    
    # Required metadata for Gymnasium
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, df):
        super().__init__()
        
        # 1. DATA AND INDICATORS
        # Ensure the data is preprocessed with indicators
        self.df = _calculate_technical_indicators(df).reset_index(drop=True)
        self.initial_balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.crypto_held = 0 # Amount of cryptocurrency held (initially 0)
        self.current_step = LOOKBACK_WINDOW # Start index after the lookback period
        self.history = [] # To track trades and net worth

        # 2. ACTION SPACE (What the agent can do)
        # Discrete(3) means the agent chooses an action: 0, 1, or 2
        # 0: SELL, 1: HOLD, 2: BUY
        self.action_space = spaces.Discrete(3) 

        # 3. OBSERVATION SPACE (What the agent sees)
        # Features now include: Open, High, Low, Close, Volume, SMA_50, RSI_14, MACD, Signal_Line (9 features)
        self.n_features = self.df.shape[1]
        
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(LOOKBACK_WINDOW, self.n_features), 
            dtype=np.float32
        )

    def _get_observation(self):
        """Returns the current observation window for the agent."""
        # FIX: Explicitly cast indices to standard Python integers (int) 
        start_index = int(self.current_step - LOOKBACK_WINDOW)
        # The end index is the current step (non-inclusive slice end)
        end_index = int(self.current_step) 
        
        # Use the safer ILOC (Integer Location) indexing method
        obs = self.df.iloc[start_index:end_index].values
        
        # Ensure the final output is a float32 NumPy array for the neural network
        return obs.astype(np.float32)

    def _calculate_reward(self, current_price, action):
        """
        Calculates the reward based on the action and the resulting portfolio change.
        """
        # FIX: Explicitly cast the index to int and use ILOC
        prev_step_index = int(self.current_step - 1)
        
        # Use iloc for the previous close price
        previous_close = self.df.iloc[prev_step_index]['Close'] 
        
        # Calculate net worth BEFORE the trade for reward comparison
        previous_net_worth = self.net_worth + (self.crypto_held * previous_close)
        
        # Calculate net worth AFTER the trade
        current_net_worth = self.net_worth + (self.crypto_held * current_price)
        
        # Reward is the change in portfolio value (percentage gain/loss)
        if previous_net_worth > 0:
            reward = (current_net_worth - previous_net_worth) / previous_net_worth
        else:
            reward = 0
            
        return reward
    
    def step(self, action):
        """
        Executes one timestep of the environment's dynamics.
        
        Returns: observation, reward, terminated, truncated, info
        """
        
        # 1. Check for Truncation (End of Data) BEFORE taking the next step
        # We check against len(df) - 1 because we index up to len(df) - 1 in iloc.
        truncated = self.current_step >= int(len(self.df) - 1)
        terminated = False 

        # If we are truncated, we stop, calculate final stats, and don't try to access data
        if truncated:
            # Get the price for the very last possible step
            current_price = self.df.iloc[int(self.current_step)]['Close']
            reward = self._calculate_reward(current_price, action) # Reward for the final move
            observation = self._get_observation()
            
            # Final portfolio value calculation
            current_portfolio_value = self.net_worth + (self.crypto_held * current_price)
            info = {
                'net_worth': current_portfolio_value,
                'crypto_held': self.crypto_held,
                'cash_balance': self.net_worth,
                'trade': "DONE" 
            }
            self.history.append(info)
            
            return observation, reward, terminated, truncated, info
        
        # 2. If not truncated, we proceed to the NEXT step
        self.current_step = int(self.current_step) + 1 
        current_step_index = int(self.current_step)
        
        # Get the closing price for the CURRENT candle (the result of the action)
        current_price = self.df.iloc[current_step_index]['Close']

        # --- 3. Execute the Trade ---
        if action == 2: # BUY
            funds_to_use = self.net_worth * 0.99
            crypto_to_buy = (funds_to_use / current_price) * (1 - TRADING_FEE)
            
            self.crypto_held += crypto_to_buy
            self.net_worth -= funds_to_use
            
            trade_type = "BUY"

        elif action == 0: # SELL
            crypto_to_sell = self.crypto_held
            cash_received = (crypto_to_sell * current_price) * (1 - TRADING_FEE)
            
            self.net_worth += cash_received
            self.crypto_held = 0
            
            trade_type = "SELL"
            
        else: # HOLD (action == 1)
            trade_type = "HOLD"

        # --- 4. Calculate Reward ---
        reward = self._calculate_reward(current_price, action)

        # --- 5. Get Next Observation ---
        observation = self._get_observation()
        
        # --- 6. Update History and Info ---
        current_portfolio_value = self.net_worth + (self.crypto_held * current_price)
        info = {
            'net_worth': current_portfolio_value,
            'crypto_held': self.crypto_held,
            'cash_balance': self.net_worth,
            'trade': trade_type
        }
        self.history.append(info)
        
        # If the portfolio is wiped out, the episode is terminated
        if current_portfolio_value <= 0:
            terminated = True
            reward = -10.0 # Heavy penalty for termination
        
        # NOTE: DummyVecEnv expects 5 outputs, so we return the correct tuple structure
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        # Reset financial state
        self.net_worth = self.initial_balance
        self.crypto_held = 0
        self.history = []
        
        # Set current step to a random point to enhance robustness
        # max_start_index ensures we have at least LOOKBACK_WINDOW steps available + 1 step for the lookahead
        max_start_index = int(len(self.df) - LOOKBACK_WINDOW - 2) 
        self.current_step = int(self.np_random.integers(LOOKBACK_WINDOW, max_start_index)) 
        
        # Get initial observation
        observation = self._get_observation()
        info = {'msg': 'Environment reset successfully'}
        
        return observation, info

    def render(self, mode='human'):
        """Prints the current state of the environment (for debugging)."""
        if self.current_step > 0:
            latest = self.history[-1]
            print(f"Step: {self.current_step}, Action: {latest['trade']}")
            print(f"  Net Worth: ${latest['net_worth']:.2f} | Crypto: {latest['crypto_held']:.4f} | Cash: ${latest['cash_balance']:.2f}")

    def close(self):
        """Clean up the environment (not strictly needed here)."""
        pass

# --- DEMONSTRATION OF USAGE ---

def test_environment(csv_filename):
    """
    Function to test the environment with random actions.
    """
    try:
        # Load the data generated in Task 2
        df = pd.read_csv(csv_filename, index_col='Open Time', parse_dates=True)
        
        print(f"\n--- Testing Environment with {len(df)} data points ---")
        
        # Initialize the environment
        env = CryptoTradingEnv(df)
        
        # Reset the environment to start a new episode
        observation, info = env.reset()
        
        # Run 50 steps with random actions (a simple dummy agent)
        steps_to_run = 50
        total_reward = 0
        
        for i in range(steps_to_run):
            # Agent chooses a random action (0=SELL, 1=HOLD, 2=BUY)
            action = env.action_space.sample() 
            
            # Take a step in the environment (DummyVecEnv returns 5 items)
            # The last element (info) is a list of dictionaries, one for each env.
            observation, reward, terminated, truncated, info = env.step([action])
            total_reward += reward[0] # Reward is an array, so extract first element
            
            # Print status every 10 steps
            if i % 10 == 0:
                 env.render()
            
            # Check if episode is over
            if terminated[0] or truncated[0]:
                print("\n--- Episode Finished ---")
                break
        
        # Final info is the info from the last step
        final_info = info[0]
        
        print(f"\nTotal reward after {i+1} steps: {total_reward:.4f}")
        print(f"Final Net Worth: ${final_info['net_worth']:.2f}")

    except FileNotFoundError:
        print(f"\nERROR: The file '{csv_filename}' was not found.")
        print("Please ensure you run 'data_collector.py' first and use the correct filename.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")

if __name__ == "__main__":
    # --- CHANGE THIS FILENAME TO MATCH YOUR SAVED CSV ---
    TEST_CSV_FILE = 'BTCUSDT_20251129.csv' 
    test_environment(TEST_CSV_FILE)