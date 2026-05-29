import pandas as pd
from binance.client import Client
import os
from datetime import datetime

# --- CONFIGURATION ---
# IMPORTANT: Replace these with your actual Binance API Key and Secret.
# For this project, you can use placeholder keys initially, but for real usage
# (or paper trading later), you need valid keys.
# Safety Tip: NEVER hardcode your real API keys in shared code.
# We use os.environ.get() as a best practice, but for a simple test,
# you can temporarily set them here.
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

# Define the asset and time frame we want to download
SYMBOL = 'BTCUSDT' # Bitcoin/Tether trading pair
INTERVAL = Client.KLINE_INTERVAL_1HOUR # 1 hour candlestick data
START_DATE = "1 Jan, 2023" # Start downloading from this date

# --- FUNCTION DEFINITION ---

def fetch_historical_data(symbol, interval, start_date):
    """
    Fetches historical candlestick data from Binance.

    Args:
        symbol (str): The trading pair (e.g., 'BTCUSDT').
        interval (str): The candlestick interval (e.g., '1h', '1d').
        start_date (str): The start date for the data fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the market data.
    """
    print(f"Connecting to Binance API to fetch data for {symbol}...")
    try:
        # Initialize the client with the (placeholder) API credentials
        client = Client(API_KEY, API_SECRET)

        # Get historical kline data
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_date
        )

        # Structure the data into a Pandas DataFrame
        df = pd.DataFrame(klines, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close Time', 'Quote Asset Volume', 'Number of Trades', 
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ])

        # Convert 'Open Time' to datetime object for easier handling
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        
        # Set 'Open Time' as the index
        df.set_index('Open Time', inplace=True)

        # Convert price and volume columns to numeric types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # Select and return only the columns we need for trading
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"Successfully fetched {len(df)} data points.")
        return df

    except Exception as e:
        print(f"An error occurred during data fetching: {e}")
        return pd.DataFrame()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Fetch the data
    btc_data = fetch_historical_data(SYMBOL, INTERVAL, START_DATE)

    # 2. Check if data was successfully fetched
    if not btc_data.empty:
        # 3. Save the data to a CSV file for later use (RL training)
        file_name = f"{SYMBOL}_{datetime.now().strftime('%Y%m%d')}.csv"
        btc_data.to_csv(file_name)
        print(f"\nData preview:\n{btc_data.head()}")
        print(f"\nData successfully saved to {file_name}")
    else:
        print("\nData fetch failed. Please check your API keys or connection.")