# CryptoSaga: Explainable AI Trading Advisor

A reinforcement-learning crypto trading advisor with an interactive Streamlit dashboard. A **PPO** (Proximal Policy Optimization) agent learns to trade BTC/USDT from historical Binance data, and each decision is paired with a natural-language explanation based on technical indicators (SMA, RSI, MACD).

---

## Overview — What This Project Does

CryptoSaga is an end-to-end system that **teaches an AI agent to trade Bitcoin and then explains its decisions in plain English**. Instead of just predicting prices, it learns a *trading policy* — when to buy, sell, or hold — and shows the reasoning behind each move.

The project works in four stages:

1. **Collect data** — It downloads years of hourly BTC/USDT price history (Open, High, Low, Close, Volume) from the Binance exchange and saves it as a CSV. A sample dataset is already included.

2. **Learn to trade** — The price data is turned into a simulated trading "game" (a Gymnasium environment). A PPO reinforcement-learning agent plays this game thousands of times, starting with $10,000 in virtual cash. At each hour it looks at the last 10 candles plus technical indicators (SMA, RSI, MACD) and chooses to **BUY**, **SELL**, or **HOLD**. It is rewarded when its portfolio grows and penalized when it shrinks, so over time it learns a profitable strategy.

3. **Explain every decision** — For each trade, the system generates a short, human-readable explanation of *why* the agent acted — for example, "MACD crossed above its signal line, confirming bullish momentum." This makes the AI's behavior transparent rather than a black box.

4. **Visualize everything** — An interactive web dashboard runs the trained agent over the data and displays the results: portfolio value over time, profit/loss, a full trade log with explanations, and five exploratory charts of the Bitcoin market (return distribution, monthly averages, volume vs. volatility, and time-of-day patterns).

**In short:** it answers two questions at once — *"Can an AI learn to trade crypto?"* and *"Can it tell us why it made each trade?"*

> ⚠️ This is an **educational / academic project**. It uses historical data in a simulation and is **not** financial advice or a live trading bot.

---

## Features

- **Custom trading environment** built on the Gymnasium API (`trading_env.py`).
- **PPO agent** trained with Stable-Baselines3 (`train_agent.py`).
- **Explainable decisions** — every BUY / SELL / HOLD comes with reasoning derived from RSI, MACD, and Signal Line (`llm_advisor.py`, mocked rule-based for now).
- **Interactive dashboard** with portfolio tracking, a trade log, and 5 EDA charts (`streamlit_dashboard.py`).
- **Data collector** to pull fresh historical candles from Binance (`data_collector.py`).

---

## Project Structure

| File | Description |
|------|-------------|
| `data_collector.py`     | Downloads historical BTCUSDT hourly data from Binance into a CSV. |
| `trading_env.py`        | Custom Gymnasium environment + technical indicators (SMA, RSI, MACD). |
| `train_agent.py`        | Trains the PPO agent and saves it to `./models/ppo_crypto_advisor.zip`. |
| `llm_advisor.py`        | Terminal-based advisor that prints trade decisions + explanations. |
| `streamlit_dashboard.py`| The main interactive web dashboard. |
| `requirements.txt`      | Python dependencies. |
| `BTCUSDT_20251129.csv`  | Sample historical dataset (already included). |

---

## Setup

Requires **Python 3.11**.

```powershell
# 1. Move into the project folder
cd "c:\Prem Kumar\Projects\Crypto Saga\Semester Project"

# 2. (Optional) create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the agent (one-time)

This creates the model file the dashboard needs (`./models/ppo_crypto_advisor.zip`):

```powershell
python train_agent.py
```

Training runs 100,000 steps (~2–3 minutes on CPU).

### 2. Launch the dashboard

```powershell
streamlit run streamlit_dashboard.py
```

Then open **http://localhost:8501** in your browser. Press `Ctrl+C` in the terminal to stop.

> **Note:** Always use `streamlit run`, not the IDE "Run" button. Running with `python streamlit_dashboard.py` triggers harmless "missing ScriptRunContext" warnings because there is no Streamlit server.

### Optional scripts

```powershell
# Re-download fresh market data from Binance (creates a new CSV)
python data_collector.py

# Run the advisor in the terminal (no UI)
python llm_advisor.py

# Test the trading environment with random actions
python trading_env.py
```

---

## How It Works

1. **State** — at each step the agent observes the last 10 candles, each with 9 features: Open, High, Low, Close, Volume, SMA_50, RSI_14, MACD, Signal_Line.
2. **Actions** — `0 = SELL`, `1 = HOLD`, `2 = BUY`.
3. **Reward** — the percentage change in portfolio net worth after each action.
4. **Explanation** — based on the indicator values driving the decision (e.g. MACD crossing its signal line, RSI overbought/oversold).

Default simulation settings: `$10,000` starting capital, `0.1%` trading fee.

---

## Configuration

Key constants you can tweak:

- `trading_env.py` — `INITIAL_ACCOUNT_BALANCE`, `TRADING_FEE`, `LOOKBACK_WINDOW`.
- `train_agent.py` — `TIMESTEPS` and PPO hyperparameters.
- `streamlit_dashboard.py` — `TRADE_STEPS`, `MODEL_PATH`, `TEST_CSV_FILE`.

---

## Security Note

`data_collector.py` and `llm_advisor.py` contain placeholders for API keys. **Do not commit real Binance or Gemini API keys** to source control — use environment variables instead, and rotate any keys that have been exposed.
