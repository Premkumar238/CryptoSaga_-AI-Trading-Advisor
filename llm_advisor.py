import pandas as pd
import numpy as np
import os
import json
import time
import requests 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# FIX: Import all necessary constants and class
from trading_env import CryptoTradingEnv, LOOKBACK_WINDOW, INITIAL_ACCOUNT_BALANCE 

# --- Gemini API Setup ---
# IMPORTANT: Replace this with your actual Gemini API Key
API_KEY = "YOUR_GEMINI_API_KEY" 
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# --- CONFIGURATION ---
MODEL_PATH = "./models/ppo_crypto_advisor.zip"
TEST_CSV_FILE = 'BTCUSDT_20251129.csv' 
TRADE_STEPS = 10 

# Maps the RL agent's action index to a description
ACTION_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# --- LLM FUNCTION (MOCKED) ---

def get_llm_explanation(market_data, current_portfolio, action, trade_step):
    """
    Mocks a call to the Gemini API to generate a natural language explanation for the trade.
    
    NOTE: Using a rule-based mock for simplicity in a local Python environment.
    """
    
    # 1. Prepare Market Data for the Prompt
    data_summary = market_data.to_markdown(floatfmt=".2f")
    
    # 2. Define the System Instruction (Persona & Rules)
    system_prompt = (
        "You are 'CryptoSage', an AI financial analyst. Your task is to provide a "
        "concise, professional, and insightful explanation (max 3 sentences) for a "
        "cryptocurrency trading decision made by a Reinforcement Learning agent. "
        "Base your reasoning ONLY on the provided OHLCV, SMA, RSI, and MACD data. "
        "Do not use external knowledge or make predictions about future prices. "
        "Start your response with 'CryptoSage Analysis:'"
    )

    # 3. Define the User Query (The specific task)
    user_query = f"""
    The RL agent decided to take the action: **{ACTION_MAP[action]}**.
    
    Current Portfolio Status:
    - Net Worth: ${current_portfolio['net_worth']:.2f}
    - Cash Balance: ${current_portfolio['cash_balance']:.2f}
    - Crypto Held: {current_portfolio['crypto_held']:.4f} BTC
    
    The agent observed the following market data (last {LOOKBACK_WINDOW} candles, current candle is the last row):
    {data_summary}
    
    Explain the reasoning behind the agent's decision to **{ACTION_MAP[action]}** based on the technical indicators (SMA_50, RSI_14, MACD, Signal_Line) and recent price trends.
    """
    
    # 4. Mock Response Logic (Updated to use MACD and Signal Line)
    latest_rsi = market_data.iloc[-1]['RSI_14']
    latest_close = market_data.iloc[-1]['Close']
    latest_sma = market_data.iloc[-1]['SMA_50']
    latest_macd = market_data.iloc[-1]['MACD']
    latest_signal = market_data.iloc[-1]['Signal_Line']

    
    reasoning = ""
    # Incorporate MACD cross for mock reasoning
    if action == 2: # BUY
        if latest_rsi < 35 or latest_macd > latest_signal:
            reasoning = f"The agent initiated a BUY. The MACD ({latest_macd:.2f}) is above its signal line ({latest_signal:.2f}), confirming bullish momentum, while the RSI is either oversold or trending up, supporting a strategic entry point."
        else:
            reasoning = f"The agent initiated a BUY due to price holding steady above the SMA_50 ({latest_sma:.2f}), confirming technical strength. The MACD crossover is imminent, suggesting the start of a new upward impulse."
    elif action == 0: # SELL
        if latest_rsi > 65 or latest_macd < latest_signal:
            reasoning = f"The agent executed a SELL. The MACD ({latest_macd:.2f}) is below its signal line ({latest_signal:.2f}), confirming bearish momentum. The high RSI indicates the asset is overbought, making this a prudent profit-taking decision."
        else:
            reasoning = f"The agent initiated a SELL. Despite neutral indicators, the closing price ({latest_close:.2f}) shows sustained weakness, suggesting a trend reversal is underway. The agent is mitigating risk ahead of potential market volatility."
    else: # HOLD
        reasoning = f"The agent chose to HOLD. With the MACD and Signal Line tightly consolidated and the RSI at a neutral {latest_rsi:.2f}, the agent is observing the current flat trend and awaiting a clear directional signal."

    mock_text = f"CryptoSage Analysis: {reasoning}"
    return mock_text
    
# --- MAIN EXECUTION ---

def run_advisor():
    """
    Loads the trained agent and runs a test episode to generate explanations.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}. Please run train_agent.py first.")
        return

    try:
        # 1. Load Data and Environment
        print(f"Loading data from {TEST_CSV_FILE}...")
        df = pd.read_csv(TEST_CSV_FILE, index_col='Open Time', parse_dates=True)
        env = DummyVecEnv([lambda: CryptoTradingEnv(df)])

        # 2. Load Trained Model
        print(f"Loading trained PPO model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH, env=env)
        
        # 3. Reset Environment for Testing (Using single variable unpack for DummyVecEnv)
        obs = env.reset()
        
        print("\n" + "="*50)
        print("  CRYPTO SAGE ADVISOR TEST RUN STARTED")
        print("="*50)

        total_net_worth = []

        for i in range(TRADE_STEPS):
            # 4. Agent Predicts Action
            action, _ = model.predict(obs, deterministic=True)
            # FIX: Use .item() to safely extract the single scalar from the prediction array
            action_scalar = action.item() 
            
            # 5. Execute Action
            # FIX: Handle the 4-tuple or 5-tuple return safely based on the environment version
            step_output = env.step(action)
            
            if len(step_output) == 5:
                 obs, reward, terminated, truncated, info = step_output
            else:
                 obs, reward, terminated, info = step_output
                 truncated = np.array([False]) # Explicitly set truncated as False for safety if only 4 returned
            
            # 6. Safely extract values from vectorized output (which are all arrays of size 1)
            current_portfolio = info[0] 
            current_net_worth = current_portfolio['net_worth']
            total_net_worth.append(current_net_worth)

            # 7. Get Market Data Slice for LLM 
            # Convert the NumPy array back into a DataFrame for context
            market_data_array = obs[0]
            # FIX: Correct column list (9 columns) to match the data shape (10, 9)
            market_data_df = pd.DataFrame(
                market_data_array,
                columns=df.columns[df.columns.isin(['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'RSI_14', 'MACD', 'Signal_Line'])]
            )
            
            # 8. Generate LLM Explanation
            explanation = get_llm_explanation(
                market_data_df, 
                current_portfolio, 
                action_scalar, # Use the safely extracted scalar value
                i
            )

            # 9. Print Results
            print(f"\n--- TRADE STEP {i+1} ---")
            print(f"Decision: {ACTION_MAP[action_scalar]} at price {market_data_df.iloc[-1]['Close']:.2f}")
            print(f"Reward: {reward[0]:.4f} | Net Worth: ${current_net_worth:.2f}")
            print(f"LLM Reasoning: {explanation}")
            
            # Check if episode is over
            if terminated[0] or truncated[0]:
                print("\n--- TEST EPISODE ENDED ---")
                break
            
        print("\n" + "="*50)
        print(f"INITIAL NET WORTH: ${INITIAL_ACCOUNT_BALANCE:.2f}") 
        print(f"FINAL NET WORTH: ${total_net_worth[-1]:.2f}")
        print("="*50)

    except Exception as e:
        print(f"An error occurred during the advisor run: {e}")

if __name__ == "__main__":
    run_advisor()