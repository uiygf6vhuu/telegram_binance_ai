# train_ai.py
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_klines(symbol="BTCUSDT", interval="5m", limit=1000):
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if not isinstance(data, list) or not data:
            logging.error(f"⚠️ Lỗi lấy dữ liệu nến cho {symbol}: Dữ liệu trống hoặc không hợp lệ.")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df
    except Exception as e:
        logging.error(f"⚠️ Lỗi kết nối khi lấy dữ liệu nến cho {symbol}: {e}")
        return pd.DataFrame()

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd - signal
    return macd, signal, macd_hist

def train_from_binance(symbol="BTCUSDT"):
    try:
        logging.info(f"⏳ Bắt đầu huấn luyện mô hình AI cho {symbol}...")
        df = fetch_klines(symbol, "5m", 1000)
        if df.empty or len(df) < 50:
            logging.warning(f"Không đủ dữ liệu để huấn luyện cho {symbol}. Bỏ qua.")
            return
        
        # Thêm các chỉ báo kỹ thuật nâng cao
        df["RSI"] = calc_rsi(df["close"], 14)
        df["EMA9"] = calc_ema(df["close"], 9)
        df["EMA21"] = calc_ema(df["close"], 21)
        df["ATR"] = calc_atr(df, 14)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calc_macd(df["close"])

        # Tạo nhãn: 1=BUY, -1=SELL, 0=NEUTRAL
        future_return = df["close"].shift(-3) / df["close"] - 1
        df["label"] = 0
        df.loc[future_return > 0.003, "label"] = 1
        df.loc[future_return < -0.003, "label"] = -1
        df = df.dropna()
        
        # Các đặc trưng đầu vào cho mô hình
        features = df[["RSI", "EMA9", "EMA21", "ATR", "MACD", "MACD_Signal", "MACD_Hist", "volume"]]
        labels = df["label"]
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Sử dụng Pipeline để tự động chuẩn hóa dữ liệu và huấn luyện mô hình
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        pipeline.fit(X_train, y_train)
        
        train_accuracy = pipeline.score(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)
        
        logging.info(f"✅ Huấn luyện thành công cho {symbol}.")
        logging.info(f"   Độ chính xác trên tập Train: {train_accuracy:.2f}")
        logging.info(f"   Độ chính xác trên tập Test: {test_accuracy:.2f}")

        os.makedirs("ai_models", exist_ok=True)
        joblib.dump(pipeline, f"ai_models/ai_{symbol}.pkl")
        logging.info(f"   Đã lưu mô hình AI mới cho {symbol} vào file: ai_models/ai_{symbol}.pkl")
        return True

    except Exception as e:
        logging.error(f"⚠️ Lỗi trong quá trình huấn luyện AI cho {symbol}: {str(e)}")
        return False

if __name__ == "__main__":
    train_from_binance("BTCUSDT")
