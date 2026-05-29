import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from trading_env import CryptoTradingEnv # Import the environment we just created

# --- CONFIGURATION ---
# IMPORTANT: Change this to match your saved CSV file name
TEST_CSV_FILE = 'BTCUSDT_20251129.csv' 
LOG_DIR = "./logs/ppo_crypto_v1"
MODEL_SAVE_PATH = "./models/ppo_crypto_advisor.zip"

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def train_ppo_agent():
    """
    Sets up the environment, trains the PPO agent, and saves the model.
    """
    try:
        # 1. Load the data
        print(f"Loading data from {TEST_CSV_FILE}...")
        df = pd.read_csv(TEST_CSV_FILE, index_col='Open Time', parse_dates=True)
        
        # 2. Create the Environment
        # Stable-Baselines3 requires the environment to be wrapped in a Vectorized Environment
        env = DummyVecEnv([lambda: CryptoTradingEnv(df)])

        # 3. Define the PPO Model
        # policy='MlpPolicy' is a Multi-Layer Perceptron (standard neural network)
        # verbose=1 shows training information
        # tensorboard_log=LOG_DIR enables tracking the training process
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            learning_rate=0.0003, # Standard starting learning rate
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=LOG_DIR
        )

        # 4. Define Callbacks (To save the model during training)
        # Save a checkpoint every 10,000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path='./checkpoints/', 
            name_prefix='ppo_checkpoint'
        )

        # 5. Train the Agent
        TIMESTEPS = 100000 # The agent will learn over 100,000 market steps (actions)
        print(f"\n--- Starting PPO Training for {TIMESTEPS} steps ---")
        
        model.learn(
            total_timesteps=TIMESTEPS, 
            callback=checkpoint_callback
        )
        
        # 6. Save the final trained model
        model.save(MODEL_SAVE_PATH)
        print(f"\n--- Training Complete ---")
        print(f"Model saved to {MODEL_SAVE_PATH}")

    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    train_ppo_agent()