# stock_agent_lstm_live.py
"""
Streamlit Stock AI Agent:
- Rule-based / RandomForest agent
- LSTM validator for confirming signals
- Auto-refresh every 1 minute
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import plotly.graph_objects as go
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
import joblib

# -----------------------
# Paths for models
# -----------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
RF_MODEL_PATH = MODEL_DIR / "rf_model.joblib"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_validator.h5"

# -----------------------
# Parameters
# -----------------------
SEQ_LEN = 30  # bars for LSTM sequence
EPOCHS = 10
BATCH_SIZE = 32

# -----------------------
# Functions
# -----------------------
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df[["Open","High","Low","Close","Volume"]]
    df = add_features(df)
    df = df.dropna()
    return df

def add_features(df):
    df["EMA5"] = ta.ema(df["Close"], length=5)
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    # Returns for RF
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ema5_ema20_diff"] = (df["EMA5"] - df["EMA20"]) / (df["EMA20"].replace(0,1))
    df = df.fillna(0)
    return df

def create_labels(df, lookahead=3, up_pct=0.01, down_pct=-0.01):
    future_max = df["Close"].shift(-1).rolling(lookahead, min_periods=1).max()
    future_min = df["Close"].shift(-1).rolling(lookahead, min_periods=1).min()
    df["future_max_ret"] = future_max / df["Close"] - 1
    df["future_min_ret"] = future_min / df["Close"] - 1
    df["label"] = np.where(df["future_max_ret"]>=up_pct,1,
                           np.where(df["future_min_ret"]<=down_pct,-1,0))
    df = df.drop(columns=["future_max_ret","future_min_ret"])
    return df

def build_features(df):
    features = ["EMA5","EMA20","RSI14","ATR14","ret_1","ret_3","ema5_ema20_diff","Volume"]
    return df[features].fillna(0).astype(float)

def train_rf_model(X, y, n_estimators=100):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test,y_pred,output_dict=True,zero_division=0)
    acc = accuracy_score(y_test,y_pred)
    return clf, report, acc

def make_sequences(df, features, seq_len=SEQ_LEN, label_col="label"):
    X, y = [], []
    data = df[features].values
    labels = df[label_col].values
    for i in range(seq_len, len(df)):
        X.append(data[i-seq_len:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

def build_lstm(input_shape, num_classes=3):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def prepare_last_sequence(df, features, seq_len=SEQ_LEN):
    seq = df[features].iloc[-seq_len:].values.reshape(1,seq_len,len(features))
    return seq

def map_lstm_output(pred):
    # 0->SELL, 1->HOLD, 2->BUY
    return np.argmax(pred)-1

def plot_chart(df, confirmed_signal=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    if confirmed_signal is not None:
        last_idx = df.index[-1]
        if confirmed_signal == 1:
            fig.add_trace(go.Scatter(x=[last_idx], y=[df["Close"].iloc[-1]],
                                     mode="markers", marker=dict(symbol="triangle-up", color="green", size=12), name="Confirmed BUY"))
        elif confirmed_signal == -1:
            fig.add_trace(go.Scatter(x=[last_idx], y=[df["Close"].iloc[-1]],
                                     mode="markers", marker=dict(symbol="triangle-down", color="red", size=12), name="Confirmed SELL"))
    fig.update_layout(title="Price Chart with Confirmed Signal", xaxis_rangeslider_visible=False, height=600)
    return fig

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(layout="wide", page_title="Stock Agent + LSTM Validator")
st.title("📈 Stock AI Agent + LSTM Validator (Live)")

# Sidebar
tickers_text = st.sidebar.text_area("Tickers (comma separated)", "AAPL,MSFT,GOOG")
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
ticker = st.sidebar.selectbox("Select ticker", tickers)

st.sidebar.header("Auto-refresh")
auto_refresh = st.sidebar.checkbox("Enable auto-refresh (every 1 min)", value=True)
if auto_refresh:
    st_autorefresh(interval=60*1000, key=f"autorefresh_{ticker}")

st.sidebar.header("Training parameters")
lookahead = st.sidebar.slider("Lookahead bars for labeling", 1, 10, 3)
up_pct = st.sidebar.number_input("UP threshold (%)", 0.001, 0.05, 0.01, step=0.001)
down_pct = st.sidebar.number_input("DOWN threshold (%)", 0.001, 0.05, 0.01, step=0.001)
down_pct = -abs(down_pct)
n_estimators = st.sidebar.number_input("RF n_estimators", 50, 500, 100, step=10)
retrain_btn = st.sidebar.button("Retrain RandomForest + LSTM Validator")
load_models_btn = st.sidebar.button("Load existing models")

if not ticker:
    st.warning("Add tickers and select one")
    st.stop()

# -----------------------
# Fetch data
# -----------------------
with st.spinner("Fetching data..."):
    df = fetch_data(ticker, period="6mo", interval="1d")
df = create_labels(df, lookahead=lookahead, up_pct=up_pct, down_pct=down_pct)
X = build_features(df)
y = df["label"]

features_lstm = ["EMA5","EMA20","RSI14","ATR14","ret_1","ret_3","ema5_ema20_diff","Volume"]

# -----------------------
# Train or Load models
# -----------------------
rf_model = None
lstm_model = None
if load_models_btn and RF_MODEL_PATH.exists() and LSTM_MODEL_PATH.exists():
    rf_model = joblib.load(RF_MODEL_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    st.success("Models loaded from disk ✅")

if retrain_btn:
    with st.spinner("Training RandomForest..."):
        rf_model, report, acc = train_rf_model(X, y, n_estimators=n_estimators)
        joblib.dump(rf_model, RF_MODEL_PATH)
        st.success(f"RF trained. Accuracy: {acc:.2f}")
        st.json(report)

    with st.spinner("Training LSTM Validator..."):
        # Scale features
        scaler = StandardScaler()
        df[features_lstm] = scaler.fit_transform(df[features_lstm])
        X_seq, y_seq = make_sequences(df, features_lstm, seq_len=SEQ_LEN, label_col="label")
        y_cat = to_categorical(y_seq+1, num_classes=3)
        X_train,X_test,y_train,y_test = train_test_split(X_seq,y_cat,test_size=0.2,shuffle=False)

        lstm_model = build_lstm((SEQ_LEN,len(features_lstm)))
        lstm_model.fit(X_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,
                       validation_data=(X_test,y_test),verbose=0)
        lstm_model.save(LSTM_MODEL_PATH)
        st.success("LSTM Validator trained and saved ✅")

# -----------------------
# Generate candidate signal
# -----------------------
ml_pred = None
if rf_model is not None:
    ml_pred = rf_model.predict(X.iloc[[-1]])[0]

# -----------------------
# Validate with LSTM
# -----------------------
confirmed_signal = None
if lstm_model is not None and ml_pred != 0:
    # prepare sequence
    last_seq = prepare_last_sequence(df, features_lstm)
    lstm_pred = lstm_model.predict(last_seq)[0]
    pred_class = map_lstm_output(lstm_pred)
    # Confirm if LSTM agrees
    confirmed_signal = ml_pred if pred_class == ml_pred else 0
else:
    confirmed_signal = 0 if ml_pred==0 else ml_pred

# -----------------------
# Display
# -----------------------
st.subheader(f"{ticker} Latest Close: {df['Close'].iloc[-1]:.2f}")

signal_map = {1:"BUY", -1:"SELL", 0:"HOLD"}
colors = {1:"green", -1:"red", 0:"gray"}
st.metric("Confirmed Signal", signal_map[confirmed_signal], delta_color=colors[confirmed_signal])

# Chart
fig = plot_chart(df, confirmed_signal=confirmed_signal)
st.plotly_chart(fig, use_container_width=True)
st.caption("✅ Confirmed signals only if LSTM validator agrees. Auto-refresh every 1 minute.")
