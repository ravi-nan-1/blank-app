import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# Auto-refresh every 1 min
# -----------------------------
st_autorefresh(interval=60*1000, key="data_refresh")

st.title("AI Stock Signal Agent")

# -----------------------------
# Tickers selection
# -----------------------------
tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
st.sidebar.header("Select a stock")
selected_ticker = st.sidebar.radio("Tickers", tickers)

# -----------------------------
# Load pretrained models (optional)
# -----------------------------
try:
    rf_model = joblib.load("rf_model.pkl")
except:
    rf_model = None

try:
    lstm_model = load_model("lstm_model.h5")
except:
    lstm_model = None

# -----------------------------
# Fetch stock data
# -----------------------------
@st.cache_data(ttl=60)
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="1h")
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

df = fetch_data(selected_ticker)

if df.empty:
    st.warning(f"No data found for {selected_ticker}")
    st.stop()

# -----------------------------
# Compute indicators with pandas-ta
# -----------------------------
df.ta.ema(length=10, append=True)  # EMA_10
df.ta.ema(length=20, append=True)  # EMA_20
df.ta.rsi(length=14, append=True)  # RSI_14
df.ta.atr(length=14, append=True)  # ATR_14

# -----------------------------
# Generate basic signals
# -----------------------------
def generate_signals(df):
    signals = []
    for i in range(1, len(df)):
        ema10 = df[f'EMA_10'].iloc[i]
        ema20 = df[f'EMA_20'].iloc[i]
        prev_ema10 = df[f'EMA_10'].iloc[i-1]
        prev_ema20 = df[f'EMA_20'].iloc[i-1]
        rsi = df[f'RSI_14'].iloc[i]
        
        if ema10 > ema20 and prev_ema10 <= prev_ema20 and rsi < 70:
            signals.append("BUY")
        elif ema10 < ema20 and prev_ema10 >= prev_ema20 and rsi > 30:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    signals.insert(0, "HOLD")
    df['Signal'] = signals
    return df

df = generate_signals(df)

# -----------------------------
# Validate signals using AI (placeholder)
# -----------------------------
if rf_model is not None and lstm_model is not None:
    df['Validated_Signal'] = df['Signal']  # replace with AI predictions
else:
    df['Validated_Signal'] = df['Signal']

# -----------------------------
# Display latest info
# -----------------------------
st.subheader(f"{selected_ticker} Latest Close: {df['Close'].iloc[-1]:.2f}")
st.dataframe(df.tail(10))

# -----------------------------
# Plot chart
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], mode='lines', name='EMA10'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA20'))

# Highlight buy/sell signals
buy_signals = df[df['Validated_Signal'] == 'BUY']
sell_signals = df[df['Validated_Signal'] == 'SELL']

fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                         mode='markers', name='BUY',
                         marker=dict(color='green', size=10, symbol='triangle-up')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                         mode='markers', name='SELL',
                         marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title=f"{selected_ticker} Price & Signals", xaxis_title="Datetime", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
