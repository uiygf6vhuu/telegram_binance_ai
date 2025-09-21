import json
import hmac
import hashlib
import time
import threading
import urllib.request
import urllib.parse
import numpy as np
import websocket
import logging
import requests
import os
import math
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import joblib
from scipy import stats
from sklearn.cluster import KMeans
from train_ai import train_from_binance

# Detailed logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_errors.log')
    ]
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get configuration from environment variables
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
# Get bot configuration from environment variables (JSON format)
bot_config_json = os.getenv('BOT_CONFIGS', '[]')
try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    logging.error(f"Error parsing BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

API_KEY = BINANCE_API_KEY
API_SECRET = BINANCE_SECRET_KEY

# ========== TELEGRAM FUNCTIONS ==========
def send_telegram(message, chat_id=None, reply_markup=None):
    """Sends a message via Telegram with detailed error handling."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram Bot Token is not configured.")
        return
    
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not chat_id:
        logger.warning("Telegram Chat ID is not configured.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"Telegram send error ({response.status_code}): {error_msg}")
    except Exception as e:
        logger.error(f"Telegram connection error: {str(e)}")

def create_menu_keyboard():
    """Creates a 3-button menu for Telegram."""
    return {
        "keyboard": [
            [{"text": "📊 Danh sách Bot"}],
            [{"text": "➕ Thêm Bot"}, {"text": "⛔ Dừng Bot"}],
            [{"text": "💰 Số dư tài khoản"}, {"text": "📈 Vị thế đang mở"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }

def create_cancel_keyboard():
    """Creates a cancel keyboard."""
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard():
    """Creates a keyboard for selecting coin pairs."""
    popular_symbols = ["SUIUSDT", "DOGEUSDT", "1000PEPEUSDT", "TRUMPUSDT", "XRPUSDT", "ADAUSDT"]
    keyboard = []
    row = []
    for symbol in popular_symbols:
        row.append({"text": symbol})
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_leverage_keyboard():
    """Creates a keyboard for selecting leverage."""
    leverages = ["3", "8", "10", "20", "30", "50", "75", "100"]
    keyboard = []
    row = []
    for lev in leverages:
        row.append({"text": f" {lev}x"})
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== BINANCE API HELPER FUNCTIONS ==========
def sign(query):
    try:
        return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Error creating signature: {str(e)}")
        send_telegram(f"⚠️ <b>SIGN ERROR:</b> {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    """General function for Binance API requests with detailed error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if method.upper() == 'GET':
                if params:
                    query = urllib.parse.urlencode(params)
                    url = f"{url}?{query}"
                req = urllib.request.Request(url, headers=headers or {})
            else:
                data = urllib.parse.urlencode(params).encode() if params else None
                req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    logger.error(f"API Error ({response.status}): {response.read().decode()}")
                    if response.status == 429:  # Rate limit
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error ({e.code}): {e.reason}")
            if e.code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"API connection error: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Failed to make API request after {max_retries} attempts")
    return None

def get_step_size(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        data = binance_api_request(url)
        if not data:
            return 0.001
            
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        logger.error(f"Error getting step size: {str(e)}")
        send_telegram(f"⚠️ <b>STEP SIZE ERROR:</b> {symbol} - {str(e)}")
    return 0.001

def set_leverage(symbol, lev):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "leverage": lev,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        response = binance_api_request(url, method='POST', headers=headers)
        if response and 'leverage' in response:
            return True
    except Exception as e:
        logger.error(f"Error setting leverage: {str(e)}")
        send_telegram(f"⚠️ <b>LEVERAGE ERROR:</b> {symbol} - {str(e)}")
    return False

def get_balance():
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            return 0
            
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
    except Exception as e:
        logger.error(f"Error getting balance: {str(e)}")
        send_telegram(f"⚠️ <b>BALANCE ERROR:</b> {str(e)}")
    return 0

def place_order(symbol, side, qty):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        return binance_api_request(url, method='POST', headers=headers)
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        send_telegram(f"⚠️ <b>ORDER ERROR:</b> {symbol} - {str(e)}")
    return None

def cancel_all_orders(symbol):
    try:
        ts = int(time.time() * 1000)
        params = {"symbol": symbol.upper(), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        binance_api_request(url, method='DELETE', headers=headers)
        return True
    except Exception as e:
        logger.error(f"Error canceling orders: {str(e)}")
        send_telegram(f"⚠️ <b>CANCEL ORDER ERROR:</b> {symbol} - {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Error getting price: {str(e)}")
        send_telegram(f"⚠️ <b>PRICE ERROR:</b> {symbol} - {str(e)}")
    return 0

def get_positions(symbol=None):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        if symbol:
            params["symbol"] = symbol.upper()
            
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}
        
        positions = binance_api_request(url, headers=headers)
        if not positions:
            return []
            
        if symbol:
            for pos in positions:
                if pos['symbol'] == symbol.upper():
                    return [pos]
            
        return positions
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        send_telegram(f"⚠️ <b>POSITIONS ERROR:</b> {symbol if symbol else ''} - {str(e)}")
    return []

# ========== TECHNICAL INDICATORS ==========
def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series([None])

def calc_ema(series, period):
    try:
        return series.ewm(span=period, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return pd.Series([None])

def calc_atr(df, period=14):
    try:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return pd.Series([None])

def calc_macd(series, fast_period=12, slow_period=26, signal_period=9):
    try:
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - signal
        return macd, signal, macd_hist
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return pd.Series([None]), pd.Series([None]), pd.Series([None])

# ========== WEBSOCKET MANAGER ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def add_symbol(self, symbol, callback):
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self.connections:
                self._create_connection(symbol, callback)
                
    def _create_connection(self, symbol, callback):
        if self._stop_event.is_set():
            return
            
        stream = f"{symbol.lower()}@trade"
        url = f"wss://fstream.binance.com/ws/{stream}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'p' in data:
                    price = float(data['p'])
                    self.executor.submit(callback, price)
            except Exception as e:
                logger.error(f"WebSocket message processing error {symbol}: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"WebSocket error {symbol}: {str(error)}")
            if not self._stop_event.is_set():
                time.sleep(300)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(300)
                self._reconnect(symbol, callback)
                
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        
        self.connections[symbol] = {
            'ws': ws,
            'thread': thread,
            'callback': callback
        }
        logger.info(f"WebSocket started for {symbol}")
        
    def _reconnect(self, symbol, callback):
        logger.info(f"Reconnecting WebSocket for {symbol}")
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket {symbol}: {str(e)}")
                del self.connections[symbol]
                logger.info(f"WebSocket removed for {symbol}")
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)
            
# ========== PROBABILITY BOT CLASS ==========
class ProbabilityBot:
    def __init__(self, symbol):
        self.symbol = symbol
        self.historical_data = None
        self.support_levels = []
        self.resistance_levels = []
    
    def load_historical_data(self):
        try:
            df = pd.DataFrame(binance_api_request(f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval=4h&limit=500"), columns=["open_time","open","high","low","close","volume","close_time","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"])
            if df.empty or len(df) < 50:
                return False
            self.historical_data = df.astype({"high": float, "low": float, "close": float, "volume": float})
            self.calculate_support_resistance()
            return True
        except Exception as e:
            logging.error(f"Error fetching probability data for {self.symbol}: {e}")
            return False
        
    def calculate_support_resistance(self):
        high_prices = self.historical_data['high'].values.reshape(-1, 1)
        low_prices = self.historical_data['low'].values.reshape(-1, 1)
        if len(high_prices) < 5 or len(low_prices) < 5:
            return
        kmeans_high = KMeans(n_clusters=5, random_state=42, n_init=10).fit(high_prices)
        self.resistance_levels = sorted([float(center[0]) for center in kmeans_high.cluster_centers_])
        kmeans_low = KMeans(n_clusters=5, random_state=42, n_init=10).fit(low_prices)
        self.support_levels = sorted([float(center[0]) for center in kmeans_low.cluster_centers_])
    
    def calculate_probability_features(self, current_price):
        if not self.support_levels or not self.resistance_levels:
            return None
        support_distances = [abs(current_price - level) for level in self.support_levels]
        resistance_distances = [abs(current_price - level) for level in self.resistance_levels]
        nearest_support = min(support_distances)
        nearest_resistance = min(resistance_distances)
        total_range = nearest_support + nearest_resistance
        prob_bounce = nearest_resistance / total_range if total_range > 0 else 0.5
        prob_breakout = nearest_support / total_range if total_range > 0 else 0.5
        current_volume = self.historical_data['volume'].iloc[-1]
        avg_volume = self.historical_data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        price_trend = self.calculate_price_trend()
        return { 'prob_bounce': prob_bounce, 'prob_breakout': prob_breakout, 'volume_ratio': volume_ratio, 'price_trend': price_trend }
    
    def calculate_price_trend(self):
        if len(self.historical_data['close']) < 50:
            return 0
        prices = self.historical_data['close'].tail(50)
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices)
        return slope
    
    def get_probability_signal(self, current_price):
        if not self.load_historical_data():
            return "HOLD", 0.5
        features = self.calculate_probability_features(current_price)
        if features is None:
            return "HOLD", 0.5
        buy_conditions = (features['prob_bounce'] > 0.7 and features['price_trend'] > 0 and features['volume_ratio'] > 1.2)
        sell_conditions = (features['prob_breakout'] < 0.3 and features['price_trend'] < 0 and features['volume_ratio'] > 1.2)
        if buy_conditions:
            confidence = min(features['prob_bounce'] * 1.2, 0.95)
            return "BUY", confidence
        elif sell_conditions:
            confidence = min((1 - features['prob_breakout']) * 1.2, 0.95)
            return "SELL", confidence
        else:
            return "HOLD", 0.5

# ========== AI MODEL RETRAINER ==========
class Retrainer(threading.Thread):
    def __init__(self, interval_hours, symbols):
        super().__init__()
        self.interval_hours = interval_hours
        self.symbols = symbols
        self.running = True
    def run(self):
        while self.running:
            try:
                logging.info(f"⏳ Starting AI model retraining process...")
                for symbol in self.symbols:
                    train_from_binance(symbol)
                logging.info(f"✅ Retraining completed for symbols: {self.symbols}")
            except Exception as e:
                logging.error(f"⚠️ Error during retraining: {str(e)}")
            time.sleep(self.interval_hours * 3600)
    def stop(self):
        self.running = False

# ========== MAIN BOT CLASS ==========
class IndicatorBot:
    def __init__(self, symbol, lev, percent, tp, sl, indicator, ws_manager):
        self.symbol = symbol.upper()
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.indicator = indicator
        self.ws_manager = ws_manager
        
        self.ai_model_path = f"ai_models/ai_{self.symbol}.pkl"
        os.makedirs("ai_models", exist_ok=True)
        self.ai_model = None
        self.load_ai_model()
        
        self.prob_bot = ProbabilityBot(self.symbol)
        
        self.check_position_status()
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        
        self._stop = False
        self.position_open = False
        self.last_trade_time = 0
        self.position_check_interval = 60
        self.last_position_check = 0
        self.last_error_log_time = 0
        self.last_close_time = 0
        self.cooldown_period = 60
        self.max_position_attempts = 3
        self.position_attempt_count = 0
        
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log(f"🟢 Bot started for {self.symbol}")

    def load_ai_model(self):
        if os.path.exists(self.ai_model_path):
            self.ai_model = joblib.load(self.ai_model_path)
            logging.info(f"✅ AI model for {self.symbol} loaded from file.")
        else:
            logging.info(f"⚡ No AI model for {self.symbol}, training...")
            if train_from_binance(self.symbol):
                self.ai_model = joblib.load(self.ai_model_path)
                logging.info(f"✅ AI model for {self.symbol} created and loaded.")
            else:
                logging.error(f"❌ Could not create AI model for {self.symbol}.")
                
    def log(self, message):
        logger.info(f"[{self.symbol}] {message}")
        send_telegram(f"<b>{self.symbol}</b>: {message}")
    
    def _handle_price_update(self, price):
        if self._stop:
            return
        self.prices.append(price)
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def get_signal(self):
        try:
            # Get signal from the AI model
            df_ai = pd.DataFrame(binance_api_request(f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval=5m&limit=100"), columns=["open_time","open","high","low","close","volume","close_time","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]).astype({"close": float, "high": float, "low": float, "volume": float})
            if df_ai.empty or len(df_ai) < 50:
                return None
            
            rsi = calc_rsi(df_ai["close"], 14)
            ema_fast = calc_ema(df_ai["close"], 9)
            ema_slow = calc_ema(df_ai["close"], 21)
            atr = calc_atr(df_ai, 14)
            macd, macd_signal, macd_hist = calc_macd(df_ai["close"])

            # Drop any NaN or None values
            if any(pd.isna([rsi.iloc[-1], ema_fast.iloc[-1], ema_slow.iloc[-1], atr.iloc[-1], macd.iloc[-1], macd_signal.iloc[-1], macd_hist.iloc[-1], df_ai["volume"].iloc[-1]])):
                return None
            
            features = pd.DataFrame([[rsi.iloc[-1], ema_fast.iloc[-1], ema_slow.iloc[-1], atr.iloc[-1], macd.iloc[-1], macd_signal.iloc[-1], macd_hist.iloc[-1], df_ai["volume"].iloc[-1]]], columns=["RSI", "EMA9", "EMA21", "ATR", "MACD", "MACD_Signal", "MACD_Hist", "volume"])
            
            ai_prediction = None
            if self.ai_model:
                ai_prediction = self.ai_model.predict(features)[0]
            
            # Get signal from the probability model
            current_price = df_ai['close'].iloc[-1]
            prob_signal, prob_confidence = self.prob_bot.get_probability_signal(current_price)
            
            # Combine and filter signals
            if (ai_prediction == 1 or prob_signal == "BUY") and prob_confidence > 0.5:
                return "BUY"
            if (ai_prediction == -1 or prob_signal == "SELL") and prob_confidence > 0.5:
                return "SELL"
            
            return None
        except Exception as e:
            self.log(f"get_signal error: {str(e)}")
            return None

    def _run(self):
        while not self._stop:
            try:
                current_time = time.time()
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                signal = self.get_signal()
                if not self.position_open and self.status == "waiting":
                    if current_time - self.last_close_time < self.cooldown_period:
                        time.sleep(1)
                        continue
                    if signal and current_time - self.last_trade_time > 60:
                        self.open_position(signal)
                        self.last_trade_time = current_time
                if self.position_open and self.status == "open":
                    self.check_tp_sl()
                    if signal:
                        if (self.side == "BUY" and signal == "SELL") or (self.side == "SELL" and signal == "BUY"):
                            current_price = self.prices[-1] if self.prices else get_current_price(self.symbol)
                            if self.entry > 0 and current_price > 0:
                                profit = (current_price - self.entry) * self.qty if self.side == "BUY" else (self.entry - current_price) * abs(self.qty)
                                invested = self.entry * abs(self.qty) / self.lev
                                roi = (profit / invested) * 100 if invested != 0 else 0
                                if roi >= 20:
                                    self.close_position(f"🔄 ROI {roi:.2f}% exceeded threshold, reversing to {signal}")
                time.sleep(1)
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"System error: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(1)

    def stop(self):
        self._stop = True
        self.ws_manager.remove_symbol(self.symbol)
        try:
            cancel_all_orders(self.symbol)
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Order cancellation error: {str(e)}")
                self.last_error_log_time = time.time()
        self.log(f"🔴 Bot stopped for {self.symbol}")

    def check_position_status(self):
        try:
            positions = get_positions(self.symbol)
            if not positions or len(positions) == 0:
                self.position_open = False
                self.status = "waiting"
                self.side = ""
                self.qty = 0
                self.entry = 0
                return
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    position_amt = float(pos.get('positionAmt', 0))
                    if abs(position_amt) > 0:
                        self.position_open = True
                        self.status = "open"
                        self.side = "BUY" if position_amt > 0 else "SELL"
                        self.qty = position_amt
                        self.entry = float(pos.get('entryPrice', 0))
                        return
            self.position_open = False
            self.status = "waiting"
            self.side = ""
            self.qty = 0
            self.entry = 0
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Position check error: {str(e)}")
                self.last_error_log_time = time.time()

    def check_tp_sl(self):
        if not self.position_open or not self.entry or not self.qty:
            return
        try:
            current_price = self.prices[-1] if self.prices else get_current_price(self.symbol)
            if current_price <= 0:
                return
            profit = (current_price - self.entry) * self.qty if self.side == "BUY" else (self.entry - current_price) * abs(self.qty)
            invested = self.entry * abs(self.qty) / self.lev
            if invested <= 0:
                return
            roi = (profit / invested) * 100
            if roi >= self.tp:
                self.close_position(f"✅ TP hit at {self.tp}% (ROI: {roi:.2f}%)")
            elif self.sl is not None and roi <= -self.sl:
                self.close_position(f"❌ SL hit at {self.sl}% (ROI: {roi:.2f}%)")
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"TP/SL check error: {str(e)}")
                self.last_error_log_time = time.time()

    def open_position(self, side):
        self.check_position_status()
        try:
            cancel_all_orders(self.symbol)
            if not set_leverage(self.symbol, self.lev):
                self.log(f"Could not set leverage to {self.lev}")
                return
            balance = get_balance()
            if balance <= 0:
                self.log(f"Insufficient USDT balance")
                return
            if self.percent > 100:
                self.percent = 100
            elif self.percent < 1:
                self.percent = 1
            usdt_amount = balance * (self.percent / 100)
            price = get_current_price(self.symbol)
            if price <= 0:
                self.log(f"Error getting price")
                return
            step = get_step_size(self.symbol)
            if step <= 0:
                step = 0.001
            qty = (usdt_amount * self.lev) / price
            if step > 0:
                steps = qty / step
                qty = round(steps) * step
            qty = max(qty, 0)
            qty = round(qty, 8)
            min_qty = step
            if qty < min_qty:
                self.log(f"⚠️ Quantity is too small ({qty}), not placing order")
                return
            self.position_attempt_count += 1
            if self.position_attempt_count > self.max_position_attempts:
                self.log(f"⚠️ Reached max position attempt limit ({self.max_position_attempts})")
                self.position_attempt_count = 0
                return
            res = place_order(self.symbol, side, qty)
            if not res:
                self.log(f"Error placing order")
                return
            executed_qty = float(res.get('executedQty', 0))
            if executed_qty < 0:
                self.log(f"Order not filled, executed quantity: {executed_qty}")
                return
            self.entry = float(res.get('avgPrice', price))
            self.side = side
            self.qty = executed_qty if side == "BUY" else -executed_qty
            self.status = "open"
            self.position_open = True
            self.position_attempt_count = 0
            message = (f"✅ <b>POSITION OPENED {self.symbol}</b>\n" f"📌 Direction: {side}\n" f"🏷️ Entry Price: {self.entry:.4f}\n" f"📊 Quantity: {executed_qty}\n" f"💵 Value: {executed_qty * self.entry:.2f} USDT\n" f" Leverage: {self.lev}x\n" f"🎯 TP: {self.tp}% | 🛡️ SL: {self.sl}%")
            self.log(message)
        except Exception as e:
            self.position_open = False
            self.log(f"❌ Error entering position: {str(e)}")

    def close_position(self, reason=""):
        try:
            cancel_all_orders(self.symbol)
            if abs(self.qty) > 0:
                close_side = "SELL" if self.side == "BUY" else "BUY"
                close_qty = abs(self.qty)
                step = get_step_size(self.symbol)
                if step > 0:
                    steps = close_qty / step
                    close_qty = round(steps) * step
                close_qty = max(close_qty, 0)
                close_qty = round(close_qty, 8)
                res = place_order(self.symbol, close_side, close_qty)
                if res:
                    price = float(res.get('avgPrice', 0))
                    message = (f"⛔ <b>POSITION CLOSED {self.symbol}</b>\n" f"📌 Reason: {reason}\n" f"🏷️ Exit Price: {price:.4f}\n" f"📊 Quantity: {close_qty}\n" f"💵 Value: {close_qty * price:.2f} USDT")
                    self.log(message)
                    self.status = "waiting"
                    self.side = ""
                    self.qty = 0
                    self.entry = 0
                    self.position_open = False
                    self.last_trade_time = time.time()
                    self.last_close_time = time.time()
                else:
                    self.log(f"Error closing position")
        except Exception as e:
            self.log(f"❌ Error closing position: {str(e)}")

# ========== BOT MANAGER ==========
class BotManager:
    def __init__(self):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        self.admin_chat_id = TELEGRAM_CHAT_ID
        self.log("🟢 BOT SYSTEM STARTED")
        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()
        self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
        self.telegram_thread.start()
        if self.admin_chat_id:
            self.send_main_menu(self.admin_chat_id)

    def log(self, message):
        logger.info(f"[SYSTEM] {message}")
        send_telegram(f"<b>SYSTEM</b>: {message}")

    def send_main_menu(self, chat_id):
        welcome = ("🤖 <b>BINANCE FUTURES TRADING BOT</b>\n\nChoose an option below:")
        send_telegram(welcome, chat_id, create_menu_keyboard())

    def add_bot(self, symbol, lev, percent, tp, sl, indicator):
        if sl == 0:
            sl = None
        symbol = symbol.upper()
        if symbol in self.bots:
            self.log(f"⚠️ Bot already exists for {symbol}")
            return False
        if not API_KEY or not API_SECRET:
            self.log("❌ API Key and Secret Key not configured!")
            return False
        try:
            price = get_current_price(symbol)
            if price <= 0:
                self.log(f"❌ Cannot get price for {symbol}")
                return False
            positions = get_positions(symbol)
            if positions and any(float(pos.get('positionAmt', 0)) != 0 for pos in positions):
                self.log(f"⚠️ Open position found for {symbol}")
            bot = IndicatorBot(symbol, lev, percent, tp, sl, indicator, self.ws_manager)
            self.bots[symbol] = bot
            self.log(f"✅ Bot added: {symbol} | Lev: {lev}x | %: {percent} | TP/SL: {tp}%/{sl}%")
            return True
        except Exception as e:
            self.log(f"❌ Error creating bot {symbol}: {str(e)}")
            return False

    def stop_bot(self, symbol):
        symbol = symbol.upper()
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()
            if bot.status == "open":
                bot.close_position("⛔ Manual bot stop")
            self.log(f"⛔ Bot stopped for {symbol}")
            del self.bots[symbol]
            return True
        return False

    def stop_all(self):
        self.log("⛔ Stopping all bots...")
        for symbol in list(self.bots.keys()):
            self.stop_bot(symbol)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 System stopped")

    def _status_monitor(self):
        while self.running:
            try:
                uptime = time.time() - self.start_time
                hours, rem = divmod(uptime, 3600)
                minutes, seconds = divmod(rem, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                active_bots = [s for s, b in self.bots.items() if not b._stop]
                balance = get_balance()
                status_msg = (f"📊 <b>SYSTEM STATUS</b>\n" f"⏱ Uptime: {uptime_str}\n" f"🤖 Active Bots: {len(active_bots)}\n" f"📈 Active Pairs: {', '.join(active_bots) if active_bots else 'None'}\n" f"💰 Available Balance: {balance:.2f} USDT")
                send_telegram(status_msg)
                for symbol, bot in self.bots.items():
                    if bot.status == "open":
                        status_msg = (f"🔹 <b>{symbol}</b>\n" f"📌 Direction: {bot.side}\n" f"🏷️ Entry Price: {bot.entry:.4f}\n" f"📊 Quantity: {abs(bot.qty)}\n" f" Leverage: {bot.lev}x\n" f"🎯 TP: {bot.tp}% | 🛡️ SL: {bot.sl}%")
                        send_telegram(status_msg)
            except Exception as e:
                logger.error(f"Status report error: {str(e)}")
            time.sleep(6 * 3600)

    def _telegram_listener(self):
        last_update_id = 0
        while self.running:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={last_update_id+1}&timeout=30"
                response = requests.get(url, timeout=35)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        for update in data['result']:
                            update_id = update['update_id']
                            message = update.get('message', {})
                            chat_id = str(message.get('chat', {}).get('id'))
                            text = message.get('text', '').strip()
                            if chat_id != self.admin_chat_id:
                                continue
                            if update_id > last_update_id:
                                last_update_id = update_id
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    logger.error("Conflict error: Only one bot instance can listen to Telegram")
                    break
            except Exception as e:
                logger.error(f"Telegram listener error: {str(e)}")
                time.sleep(5)

    def _handle_telegram_message(self, chat_id, text):
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        if current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                symbol = text.upper()
                self.user_states[chat_id] = {'step': 'waiting_leverage', 'symbol': symbol}
                send_telegram(f"Choose leverage for {symbol}:", chat_id, create_leverage_keyboard())
        elif current_step == 'waiting_leverage':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            elif 'x' in text:
                leverage = int(text.replace('', '').replace('x', '').strip())
                user_state['leverage'] = leverage
                user_state['step'] = 'waiting_percent'
                send_telegram(f"📌 Pair: {user_state['symbol']}\n Leverage: {leverage}x\n\nEnter % of balance to use (1-100):", chat_id, create_cancel_keyboard())
        elif current_step == 'waiting_percent':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                try:
                    percent = float(text)
                    if 1 <= percent <= 100:
                        user_state['percent'] = percent
                        user_state['step'] = 'waiting_tp'
                        send_telegram(f"📌 Pair: {user_state['symbol']}\n Lev: {user_state['leverage']}x\n📊 %: {percent}%\n\nEnter % Take Profit (e.g., 10):", chat_id, create_cancel_keyboard())
                    else:
                        send_telegram("⚠️ Please enter a % from 1-100", chat_id)
                except Exception:
                    send_telegram("⚠️ Invalid value, please enter a number", chat_id)
        elif current_step == 'waiting_tp':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                try:
                    tp = float(text)
                    if tp > 0:
                        user_state['tp'] = tp
                        user_state['step'] = 'waiting_sl'
                        send_telegram(f"📌 Pair: {user_state['symbol']}\n Lev: {user_state['leverage']}x\n📊 %: {user_state['percent']}%\n🎯 TP: {tp}%\n\nEnter % Stop Loss (e.g., 5):", chat_id, create_cancel_keyboard())
                    else:
                        send_telegram("⚠️ TP must be greater than 0", chat_id)
                except Exception:
                    send_telegram("⚠️ Invalid value, please enter a number", chat_id)
        elif current_step == 'waiting_sl':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                try:
                    sl = float(text)
                    if sl >= 0:
                        symbol = user_state['symbol']
                        leverage = user_state['leverage']
                        percent = user_state['percent']
                        tp = user_state['tp']
                        if self.add_bot(symbol, leverage, percent, tp, sl, "AI"):
                            send_telegram(f"✅ <b>BOT ADDED SUCCESSFULLY</b>\n\n" f"📌 Pair: {symbol}\n" f" Leverage: {leverage}x\n" f"📊 % Balance: {percent}%\n" f"🎯 TP: {tp}%\n" f"🛡️ SL: {sl}%", chat_id, create_menu_keyboard())
                        else:
                            send_telegram("❌ Could not add bot, please check logs", chat_id, create_menu_keyboard())
                        self.user_states[chat_id] = {}
                    else:
                        send_telegram("⚠️ SL must be greater than or equal to 0", chat_id)
                except Exception:
                    send_telegram("⚠️ Invalid value, please enter a number", chat_id)
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 No bots are currently running", chat_id)
            else:
                message = "🤖 <b>LIST OF RUNNING BOTS</b>\n\n"
                for symbol, bot in self.bots.items():
                    status = "🟢 Open" if bot.status == "open" else "🟡 Waiting"
                    message += f"🔹 {symbol} | {status} | {bot.side}\n"
                send_telegram(message, chat_id)
        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_symbol'}
            send_telegram("Choose a coin pair:", chat_id, create_symbols_keyboard())
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 No bots are currently running", chat_id)
            else:
                message = "⛔ <b>CHOOSE BOT TO STOP</b>\n\n"
                keyboard = []
                row = []
                for i, symbol in enumerate(self.bots.keys()):
                    message += f"🔹 {symbol}\n"
                    row.append({"text": f"⛔ {symbol}"})
                    if len(row) == 2 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                send_telegram(message, chat_id, {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True})
        elif text.startswith("⛔ "):
            symbol = text.replace("⛔ ", "").strip().upper()
            if symbol in self.bots:
                self.stop_bot(symbol)
                send_telegram(f"⛔ Stop command sent for bot {symbol}", chat_id, create_menu_keyboard())
            else:
                send_telegram(f"⚠️ Bot not found {symbol}", chat_id, create_menu_keyboard())
        elif text == "💰 Số dư tài khoản":
            try:
                balance = get_balance()
                send_telegram(f"💰 <b>AVAILABLE BALANCE</b>: {balance:.2f} USDT", chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Error getting balance: {str(e)}", chat_id)
        elif text == "📈 Vị thế đang mở":
            try:
                positions = get_positions()
                if not positions:
                    send_telegram("📭 No open positions", chat_id)
                    return
                message = "📈 <b>OPEN POSITIONS</b>\n\n"
                for pos in positions:
                    position_amt = float(pos.get('positionAmt', 0))
                    if position_amt != 0:
                        symbol = pos.get('symbol', 'UNKNOWN')
                        entry = float(pos.get('entryPrice', 0))
                        side = "LONG" if position_amt > 0 else "SHORT"
                        pnl = float(pos.get('unRealizedProfit', 0))
                        message += (f"🔹 {symbol} | {side}\n" f"📊 Quantity: {abs(position_amt):.4f}\n" f"🏷️ Entry Price: {entry:.4f}\n" f"💰 PnL: {pnl:.2f} USDT\n\n")
                send_telegram(message, chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Error getting positions: {str(e)}", chat_id)
        elif text:
            self.send_main_menu(chat_id)

# ========== MAIN FUNCTION ==========
def main():
    manager = BotManager()
    symbols_to_retrain = [config[0] for config in BOT_CONFIGS]
    if symbols_to_retrain:
        retrainer_thread = Retrainer(interval_hours=24, symbols=symbols_to_retrain)
        retrainer_thread.start()
        logging.info(f"🤖 Retraining process started, once every 24 hours.")
    if BOT_CONFIGS:
        for config in BOT_CONFIGS:
            manager.add_bot(*config)
    else:
        manager.log("⚠️ No bot configurations found!")
    try:
        balance = get_balance()
        manager.log(f"💰 INITIAL BALANCE: {balance:.2f} USDT")
    except Exception as e:
        manager.log(f"⚠️ Error getting initial balance: {str(e)}")
    try:
        while manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.log("👋 Received user stop signal...")
    except Exception as e:
        manager.log(f"⚠️ SEVERE SYSTEM ERROR: {str(e)}")
    finally:
        manager.stop_all()
        if 'retrainer_thread' in locals() and retrainer_thread.is_alive():
            retrainer_thread.stop()
            retrainer_thread.join()

if __name__ == "__main__":
    main()


