import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# Auto-refresh every 60 seconds
# -----------------------------
st_autorefresh(interval=60*1000, key="data_refresh")

st.title("AI Stock Signal Agent")

# -----------------------------
# Example tickers
# -----------------------------
tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]

st.sidebar.header("Select a stock")
selected_ticker = st.sidebar.radio("Tickers", tickers)

# -----------------------------
# Load pretrained models
# -----------------------------
# Placeholder paths - replace with your own trained models
# RandomForest model for initial signal
try:
    rf_model = joblib.load("rf_model.pkl")
except:
    rf_model = None

# LSTM model to validate signals
try:
    lstm_model = load_model("lstm_model.h5")
except:
    lstm_model = None

# -----------------------------
# Functions to calculate indicators manually
# -----------------------------
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# -----------------------------
# Fetch stock data
# -----------------------------
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="1h")
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()  # return empty if error

df = fetch_data(selected_ticker)

if df.empty:
    st.warning(f"No data found for {selected_ticker}")
    st.stop()

# -----------------------------
# Calculate indicators
# -----------------------------
df['EMA10'] = EMA(df['Close'], 10)
df['EMA20'] = EMA(df['Close'], 20)
df['RSI'] = RSI(df['Close'], 14)
df['ATR'] = ATR(df, 14)

# -----------------------------
# Generate basic signals
# -----------------------------
def generate_signals(df):
    signals = []
    for i in range(1, len(df)):
        # Buy signal: EMA10 crosses above EMA20 & RSI<70
        if df['EMA10'].iloc[i] > df['EMA20'].iloc[i] and df['EMA10'].iloc[i-1] <= df['EMA20'].iloc[i-1] and df['RSI'].iloc[i] < 70:
            signals.append("BUY")
        # Sell signal: EMA10 crosses below EMA20 & RSI>30
        elif df['EMA10'].iloc[i] < df['EMA20'].iloc[i] and df['EMA10'].iloc[i-1] >= df['EMA20'].iloc[i-1] and df['RSI'].iloc[i] > 30:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    signals.insert(0, "HOLD")  # first row
    df['Signal'] = signals
    return df

df = generate_signals(df)

# -----------------------------
# Validate signals using AI models
# -----------------------------
# Placeholder: just pass through current signal if models not available
if rf_model is not None and lstm_model is not None:
    # Example: you would implement feature extraction here
    # and run rf_model.predict() and lstm_model.predict()
    df['Validated_Signal'] = df['Signal']  # replace with AI prediction
else:
    df['Validated_Signal'] = df['Signal']

# -----------------------------
# Display latest info
# -----------------------------
st.subheader(f"{selected_ticker} Latest Close: {df['Close'].iloc[-1]:.2f}")
st.dataframe(df.tail(10))

# -----------------------------
# Plot chart with signals
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA10'], mode='lines', name='EMA10'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20'))

# Highlight buy/sell signals
buy_signals = df[df['Validated_Signal'] == 'BUY']
sell_signals = df[df['Validated_Signal'] == 'SELL']
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='BUY', marker=dict(color='green', size=10, symbol='triangle-up')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='SELL', marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title=f"{selected_ticker} Price & Signals", xaxis_title="Datetime", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
