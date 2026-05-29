import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import CryptoTradingEnv, LOOKBACK_WINDOW, INITIAL_ACCOUNT_BALANCE

# --- CONFIGURATION ---
MODEL_PATH = "./models/ppo_crypto_advisor.zip"
TEST_CSV_FILE = 'BTCUSDT_20251129.csv'
TRADE_STEPS = 15
ACTION_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# ─────────────────────────────────────────────
#  VISUALIZATION HELPERS
# ─────────────────────────────────────────────

@st.cache_data
def load_full_csv():
    """Load the full BTCUSDT CSV for EDA visualizations."""
    df = pd.read_csv(TEST_CSV_FILE, parse_dates=['Open Time'])
    df['Return']     = df['Close'].pct_change() * 100
    df['PriceRange'] = df['High'] - df['Low']
    df['Hour']       = df['Open Time'].dt.hour
    df['DayOfWeek']  = df['Open Time'].dt.day_name()
    df['Year']       = df['Open Time'].dt.year
    df['Month']      = df['Open Time'].dt.month
    df['Direction']  = df['Return'].apply(
        lambda x: 'Bullish' if x > 0 else ('Bearish' if x < 0 else 'Flat')
    )
    return df


def chart_histogram(df):
    """1. Histogram — Hourly Return Distribution."""
    clipped = df['Return'].dropna().clip(-5, 5)
    fig = px.histogram(
        clipped,
        nbins=80,
        color_discrete_sequence=['#58a6ff'],
        title='Hourly Return Distribution (clipped ±5%)',
        labels={'value': 'Hourly Return (%)', 'count': 'Frequency'},
    )
    # Colour positive/negative bins differently
    fig.update_traces(marker_color='#58a6ff', opacity=0.85)
    fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)
    fig.update_layout(
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font_color='#c9d1d9', showlegend=False,
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
        title_font_size=16,
    )
    return fig


def chart_bar(df):
    """2. Bar Chart — Average Monthly Close Price."""
    month_avg = (
        df.groupby(['Year', 'Month'])['Close']
        .mean()
        .reset_index()
    )
    month_avg['Label'] = month_avg.apply(
        lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1
    )
    month_avg = month_avg.sort_values(['Year', 'Month'])
    overall_avg = df['Close'].mean()
    month_avg['Color'] = month_avg['Close'].apply(
        lambda v: '#f7931a' if v >= overall_avg else '#58a6ff'
    )

    fig = go.Figure()
    fig.add_bar(
        x=month_avg['Label'],
        y=month_avg['Close'],
        marker_color=month_avg['Color'],
        opacity=0.88,
        name='Avg Close',
    )
    fig.add_hline(
        y=overall_avg, line_dash='dash', line_color='white', opacity=0.6,
        annotation_text=f"Overall Avg: ${overall_avg:,.0f}",
        annotation_font_color='#c9d1d9',
    )
    fig.update_layout(
        title='Average Monthly Close Price (2023–2025)',
        xaxis_title='Month', yaxis_title='Avg Close Price (USD)',
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font_color='#c9d1d9', showlegend=False,
        xaxis=dict(gridcolor='#21262d', tickangle=-45),
        yaxis=dict(gridcolor='#21262d', tickprefix='$'),
        title_font_size=16,
    )
    return fig


def chart_scatter(df):
    """3. Scatter Plot — Volume vs Price Range (coloured by Close)."""
    sample = df.dropna(subset=['PriceRange', 'Volume', 'Close']).sample(
        min(4000, len(df)), random_state=42
    )
    fig = px.scatter(
        sample, x='PriceRange', y='Volume', color='Close',
        color_continuous_scale='plasma',
        opacity=0.4, size_max=4,
        title='Volume vs Hourly Price Range',
        labels={
            'PriceRange': 'Price Range (High − Low, USD)',
            'Volume': 'Volume (BTC)',
            'Close': 'Close Price',
        },
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font_color='#c9d1d9',
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
        coloraxis_colorbar=dict(title='Close (USD)', tickfont_color='#8b949e'),
        title_font_size=16,
    )
    return fig


def chart_heatmap(df):
    """4. Heatmap — Avg Return by Day of Week & Hour."""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday', 'Sunday']
    pivot = (
        df.groupby(['DayOfWeek', 'Hour'])['Return']
        .mean()
        .unstack()
        .reindex(day_order)
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(h) for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title='Avg Return (%)', tickfont_color='#8b949e'),
        )
    )
    fig.update_layout(
        title='Avg Hourly Return by Day & Hour (UTC)',
        xaxis_title='Hour of Day (UTC)', yaxis_title='',
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font_color='#c9d1d9',
        xaxis=dict(gridcolor='#21262d'), yaxis=dict(gridcolor='#21262d'),
        title_font_size=16,
    )
    return fig


def chart_countplot(df):
    """5. Count Plot — Candle Direction by Hour."""
    counts = (
        df.groupby(['Hour', 'Direction'])
        .size()
        .reset_index(name='Count')
    )
    color_map = {'Bullish': '#3fb950', 'Bearish': '#f85149', 'Flat': '#555555'}
    fig = px.bar(
        counts, x='Hour', y='Count', color='Direction',
        barmode='group',
        color_discrete_map=color_map,
        title='Candle Direction Count by Hour of Day',
        labels={'Hour': 'Hour of Day (UTC)', 'Count': 'Number of Candles'},
    )
    fig.update_layout(
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font_color='#c9d1d9',
        xaxis=dict(gridcolor='#21262d', dtick=1),
        yaxis=dict(gridcolor='#21262d'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
        title_font_size=16,
    )
    return fig


# --- LLM FUNCTION (MOCKED) ---

@st.cache_data
def get_llm_explanation(market_data, current_portfolio, action, trade_step):
    latest_rsi    = market_data.iloc[-1]['RSI_14']
    latest_close  = market_data.iloc[-1]['Close']
    latest_sma    = market_data.iloc[-1]['SMA_50']
    latest_macd   = market_data.iloc[-1]['MACD']
    latest_signal = market_data.iloc[-1]['Signal_Line']

    reasoning = ""
    if action == 2:
        if latest_rsi < 35 or latest_macd > latest_signal:
            reasoning = (f"The agent initiated a BUY. The MACD ({latest_macd:.2f}) is above its signal line "
                         f"({latest_signal:.2f}), confirming bullish momentum, while the RSI is either oversold "
                         f"({latest_rsi:.2f}) or trending up.")
        else:
            reasoning = (f"The agent initiated a BUY due to price holding steady above the SMA_50 "
                         f"({latest_sma:.2f}), confirming technical strength.")
    elif action == 0:
        if latest_rsi > 65 or latest_macd < latest_signal:
            reasoning = (f"The agent executed a SELL. The MACD ({latest_macd:.2f}) is below its signal line "
                         f"({latest_signal:.2f}), confirming bearish momentum. High RSI ({latest_rsi:.2f}).")
        else:
            reasoning = (f"The agent initiated a SELL. Closing price ({latest_close:.2f}) shows weakness, "
                         f"suggesting a trend reversal.")
    else:
        reasoning = (f"The agent chose to HOLD. MACD and Signal Line are consolidated and RSI is neutral "
                     f"at {latest_rsi:.2f}.")

    return f"CryptoSaga Analysis: {reasoning}"


# --- MAIN ADVISOR FUNCTION ---

@st.cache_resource
def load_data_and_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"ERROR: Model not found at {MODEL_PATH}. Please run train_agent.py first.")
        return None, None, None
    try:
        df = pd.read_csv(TEST_CSV_FILE, index_col='Open Time', parse_dates=True)
        env = DummyVecEnv([lambda: CryptoTradingEnv(df)])
        model = PPO.load(MODEL_PATH, env=env)
        return df, env, model
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None


def run_advisor_simulation(df, env, model):
    st.info(f"Running CryptoSaga Advisor for {TRADE_STEPS} steps...")
    obs = env.reset()
    results = []
    market_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                      'SMA_50', 'RSI_14', 'MACD', 'Signal_Line']

    for i in range(TRADE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        action_scalar = action.item()
        step_output = env.step(action)

        if len(step_output) == 5:
            obs, reward, terminated, truncated, info = step_output
        else:
            obs, reward, terminated, info = step_output
            truncated = np.array([False])

        current_portfolio  = info[0]
        current_net_worth  = current_portfolio['net_worth']
        market_data_array  = obs[0]
        market_data_df     = pd.DataFrame(market_data_array, columns=market_columns)

        explanation = get_llm_explanation(
            market_data_df, current_portfolio, action_scalar, i
        )

        results.append({
            'Step': i + 1,
            'Decision': ACTION_MAP[action_scalar],
            'Price': market_data_df.iloc[-1]['Close'],
            'Net_Worth': current_net_worth,
            'Reward': reward[0],
            'RSI_14': market_data_df.iloc[-1]['RSI_14'],
            'MACD': market_data_df.iloc[-1]['MACD'],
            'Signal_Line': market_data_df.iloc[-1]['Signal_Line'],
            'Explanation': explanation,
        })

        if terminated[0] or truncated[0]:
            st.warning(f"Episode terminated/truncated at step {i+1}.")
            break

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────

def main():
    st.set_page_config(layout="wide", page_title="CryptoSaga AI Advisor")

    df_model, env, model = load_data_and_model()

    st.title("💰 CryptoSaga: Explainable AI Trading Advisor")
    st.subheader("Reinforcement Learning Agent with LLM Reasoning")
    st.markdown("---")

    if df_model is None:
        return

    # Run Simulation
    if 'results_df' not in st.session_state:
        st.session_state.results_df = run_advisor_simulation(df_model, env, model)
        st.session_state.initial_net_worth = INITIAL_ACCOUNT_BALANCE

    # Sidebar
    with st.sidebar:
        st.header("Simulation Settings")
        st.metric("Initial Capital", f"${st.session_state.initial_net_worth:.2f}")
        st.metric("Total Trading Steps", TRADE_STEPS)
        st.metric("Data Points Available", f"{len(df_model)} candles")
        if st.button("Rerun Simulation", width='stretch'):
            st.session_state.results_df = run_advisor_simulation(df_model, env, model)
            st.cache_data.clear()
            st.rerun()

    results_df = st.session_state.results_df

    # Performance metrics
    final_net_worth = results_df['Net_Worth'].iloc[-1]
    net_profit      = final_net_worth - st.session_state.initial_net_worth

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Net Worth", f"${final_net_worth:.2f}")
    with col2:
        st.metric("Net Profit / Loss", f"${net_profit:.2f}",
                  delta=f"{net_profit/st.session_state.initial_net_worth * 100:.2f}%")
    with col3:
        st.metric("Total Trades Made", len(results_df[results_df['Decision'] != 'HOLD']))
    with col4:
        st.metric("RL Agent Model", "PPO (MACD/RSI)",
                  help="Proximal Policy Optimization trained on custom features.")

    st.markdown("---")

    # Portfolio chart
    st.header("Portfolio Value Over Time")
    plot_data = results_df[['Step', 'Net_Worth']].set_index('Step')
    st.line_chart(plot_data)

    st.markdown("---")

    # Trade log
    st.header("Trade Decisions & CryptoSaga Reasoning")
    st.dataframe(
        results_df[['Step', 'Decision', 'Price', 'Net_Worth',
                    'RSI_14', 'MACD', 'Signal_Line', 'Explanation']],
        column_config={
            "Price":        st.column_config.NumberColumn("Price (USDT)", format="%.2f"),
            "Net_Worth":    st.column_config.NumberColumn("Net Worth",     format="$%.2f"),
            "RSI_14":       st.column_config.NumberColumn("RSI",           format="%.2f"),
            "MACD":         st.column_config.NumberColumn("MACD",          format="%.2f"),
            "Signal_Line":  st.column_config.NumberColumn("Signal Line",   format="%.2f"),
            "Explanation":  st.column_config.TextColumn("CryptoSaga Explanation"),
        },
        hide_index=True,
        width='stretch',
    )

    st.markdown("---")

    # ─────────────────────────────────────────
    #  EDA VISUALIZATIONS
    # ─────────────────────────────────────────
    st.header("📊 Market Data Analysis")
    st.markdown(
        "Interactive charts built from the full **BTCUSDT hourly dataset (2023–2025)**."
    )

    eda_df = load_full_csv()

    # Row 1: Histogram + Bar
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1 · Return Distribution")
        st.plotly_chart(chart_histogram(eda_df), width='stretch')
    with col_b:
        st.subheader("2 · Monthly Avg Close Price")
        st.plotly_chart(chart_bar(eda_df), width='stretch')

    # Row 2: Scatter (full width)
    st.subheader("3 · Volume vs Price Range")
    st.plotly_chart(chart_scatter(eda_df), width='stretch')

    # Row 3: Heatmap + Count Plot
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("4 · Return Heatmap (Day × Hour)")
        st.plotly_chart(chart_heatmap(eda_df), width='stretch')
    with col_d:
        st.subheader("5 · Candle Direction by Hour")
        st.plotly_chart(chart_countplot(eda_df), width='stretch')


if __name__ == "__main__":
    if not os.path.exists(TEST_CSV_FILE):
        st.error(f"Missing data file: {TEST_CSV_FILE}. Please run data_collector.py.")
    elif not os.path.exists(MODEL_PATH):
        st.error(f"Missing model file: {MODEL_PATH}. Please run train_agent.py.")
    else:
        main()
