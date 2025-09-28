import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Auto-refresh every 1 minute
# -----------------------------
st_autorefresh(interval=60*1000, key="data_refresh")

st.title("AI Stock Signal Agent (5-min interval)")

# -----------------------------
# Sidebar: select stock
# -----------------------------
tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
st.sidebar.header("Select a stock")
selected_ticker = st.sidebar.radio("Tickers", tickers)

# -----------------------------
# Fetch stock data (5-min interval)
# -----------------------------
@st.cache_data(ttl=60)
def fetch_data(ticker):
    df = yf.download(ticker, period="5d", interval="5m")
    if df.empty:
        return df
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1]=='' else f"{col[0]}_{col[1]}" for col in df.columns]
    # Ensure numeric
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

df = fetch_data(selected_ticker)

if df.empty:
    st.warning(f"No data found for {selected_ticker}")
    st.stop()

# -----------------------------
# Compute indicators with pandas-ta
# -----------------------------
df["EMA_10"] = ta.ema(df["Close"], length=10)
df["EMA_20"] = ta.ema(df["Close"], length=20)
df["RSI_14"] = ta.rsi(df["Close"], length=14)
df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

# -----------------------------
# Generate buy/sell signals
# -----------------------------
def generate_signals(df):
    signals = []
    for i in range(1, len(df)):
        ema10 = df["EMA_10"].iloc[i]
        ema20 = df["EMA_20"].iloc[i]
        prev_ema10 = df["EMA_10"].iloc[i-1]
        prev_ema20 = df["EMA_20"].iloc[i-1]
        rsi = df["RSI_14"].iloc[i]

        if ema10 > ema20 and prev_ema10 <= prev_ema20 and rsi < 70:
            signals.append("BUY")
        elif ema10 < ema20 and prev_ema10 >= prev_ema20 and rsi > 30:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    signals.insert(0, "HOLD")
    df["Signal"] = signals
    return df

df = generate_signals(df)

# -----------------------------
# Display latest info
# -----------------------------
st.subheader(f"{selected_ticker} Latest Close: {df['Close'].iloc[-1]:.2f}")
st.dataframe(df.tail(10))

# -----------------------------
# Plot chart with signals
# -----------------------------
fig = go.Figure()

# Close and EMAs
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_10"], mode='lines', name='EMA10'))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], mode='lines', name='EMA20'))

# Buy/Sell markers
buy_signals = df[df["Signal"] == "BUY"]
sell_signals = df[df["Signal"] == "SELL"]

fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals["Close"],
    mode='markers',
    name='BUY',
    marker=dict(color='green', size=10, symbol='triangle-up')
))
fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals["Close"],
    mode='markers',
    name='SELL',
    marker=dict(color='red', size=10, symbol='triangle-down')
))

fig.update_layout(
    title=f"{selected_ticker} Price & Signals (5-min interval)",
    xaxis_title="Datetime",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
