import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import requests

# ===== Lấy dữ liệu nến từ Binance =====
def fetch_klines(symbol="BTCUSDT", interval="5m", limit=1000):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100/(1+rs))
    return rsi

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ===== Chuẩn bị dữ liệu =====
df = fetch_klines("BTCUSDT", "5m", 1000)
df["RSI"] = calc_rsi(df["close"], 14)
df["EMA9"] = calc_ema(df["close"], 9)
df["EMA21"] = calc_ema(df["close"], 21)
df["ATR"] = calc_atr(df, 14)

future_return = df["close"].shift(-3) / df["close"] - 1
df["label"] = 0
df.loc[future_return > 0.003, "label"] = 1   # BUY
df.loc[future_return < -0.003, "label"] = -1 # SELL

df = df.dropna()
X = df[["RSI","EMA9","EMA21","ATR","volume"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ===== Train model =====
model = XGBClassifier(
    max_depth=4,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

print("Train acc:", model.score(X_train, y_train))
print("Test acc:", model.score(X_test, y_test))

# ===== Save model =====
joblib.dump(model, "ai_model.pkl")
print("✅ AI model đã lưu vào ai_model.pkl")
