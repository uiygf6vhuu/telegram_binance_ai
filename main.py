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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import SGDClassifier
import joblib
from train_ai import train_from_binance
# Cấu hình logging chi tiết
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

# Lấy cấu hình từ biến môi trường
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
# Cấu hình bot từ biến môi trường (dạng JSON)
bot_config_json = os.getenv('BOT_CONFIGS', '[]')
try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    logging.error(f"Lỗi phân tích cấu hình BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

API_KEY = BINANCE_API_KEY
API_SECRET = BINANCE_SECRET_KEY

# ========== HÀM GỬI TELEGRAM VÀ XỬ LÝ LỖI ==========
def send_telegram(message, chat_id=None, reply_markup=None):
    """Gửi thông báo qua Telegram với xử lý lỗi chi tiết"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Cấu hình Telegram Bot Token chưa được thiết lập")
        return
    
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not chat_id:
        logger.warning("Cấu hình Telegram Chat ID chưa được thiết lập")
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
            logger.error(f"Lỗi gửi Telegram ({response.status_code}): {error_msg}")
    except Exception as e:
        logger.error(f"Lỗi kết nối Telegram: {str(e)}")

# ========== HÀM TẠO MENU TELEGRAM ==========
def create_menu_keyboard():
    """Tạo menu 3 nút cho Telegram"""
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
    """Tạo bàn phím hủy"""
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard():
    """Tạo bàn phím chọn cặp coin"""
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
    """Tạo bàn phím chọn đòn bẩy"""
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

# ========== HÀM HỖ TRỢ API BINANCE VỚI XỬ LÝ LỖI CHI TIẾT ==========
def sign(query):
    try:
        return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI SIGN:</b> {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    """Hàm tổng quát cho các yêu cầu API Binance với xử lý lỗi chi tiết"""
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
                    logger.error(f"Lỗi API ({response.status}): {response.read().decode()}")
                    if response.status == 429:  # Rate limit
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            if e.code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Lỗi kết nối API: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
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
        logger.error(f"Lỗi lấy step size: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI STEP SIZE:</b> {symbol} - {str(e)}")
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
        logger.error(f"Lỗi thiết lập đòn bẩy: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI ĐÒN BẨY:</b> {symbol} - {str(e)}")
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
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI SỐ DƯ:</b> {str(e)}")
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
        logger.error(f"Lỗi đặt lệnh: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI ĐẶT LỆNH:</b> {symbol} - {str(e)}")
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
        logger.error(f"Lỗi hủy lệnh: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI HỦY LỆNH:</b> {symbol} - {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI GIÁ:</b> {symbol} - {str(e)}")
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
        logger.error(f"Lỗi lấy vị thế: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI VỊ THẾ:</b> {symbol if symbol else ''} - {str(e)}")
    return []

# ========== TÍNH CHỈ BÁO KỸ THUẬT VỚI KIỂM TRA DỮ LIỆU ==========
def calc_rsi(prices, period=14):
    try:
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))
    except Exception as e:
        logger.error(f"Lỗi tính RSI: {str(e)}")
        return None

def calc_ema(prices, period):
    prices = np.array(prices)
    if len(prices) < period:
        return None
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(prices, weights, mode='valid')
    return float(ema[-1])  # lấy giá trị EMA cuối cùng dạng float




# ========== QUẢN LÝ WEBSOCKET HIỆU QUẢ VỚI KIỂM SOÁT LỖI ==========
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
                logger.error(f"Lỗi xử lý tin nhắn WebSocket {symbol}: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"Lỗi WebSocket {symbol}: {str(error)}")
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket đóng {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(5)
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
        logger.info(f"WebSocket bắt đầu cho {symbol}")
        
    def _reconnect(self, symbol, callback):
        logger.info(f"Kết nối lại WebSocket cho {symbol}")
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception as e:
                    logger.error(f"Lỗi đóng WebSocket {symbol}: {str(e)}")
                del self.connections[symbol]
                logger.info(f"WebSocket đã xóa cho {symbol}")
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)
            
class Candle:
    def __init__(self, timestamp, open_price, high_price, low_price, close_price, volume):
        self.timestamp = int(timestamp)
        self.open = float(open_price)
        self.high = float(high_price)
        self.low = float(low_price)
        self.close = float(close_price)
        self.volume = float(volume)

    @classmethod
    def from_binance(cls, kline):
        """
        Tạo Candle từ 1 cây nến của Binance (list 12 phần tử).
        Cấu trúc chuẩn của Binance:
        [
            1499040000000,      # 0: Open time
            "0.01634790",       # 1: Open
            "0.80000000",       # 2: High
            "0.01575800",       # 3: Low
            "0.01577100",       # 4: Close
            "148976.11427815",  # 5: Volume
            1499644799999,      # 6: Close time
            "2434.19055334",    # 7: Quote asset volume
            308,                # 8: Number of trades
            "1756.87402397",    # 9: Taker buy base asset volume
            "28.46694368",      # 10: Taker buy quote asset volume
            "17928899.62484339" # 11: Ignore
        ]
        """
        if not isinstance(kline, list) or len(kline) < 6:
            raise ValueError(f"❌ Dữ liệu nến không hợp lệ: {kline}")

        try:
            return cls(
                timestamp=kline[0],  # Open time
                open_price=kline[1],
                high_price=kline[2],
                low_price=kline[3],
                close_price=kline[4],
                volume=kline[5]
            )
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"❌ Lỗi khi tạo Candle từ dữ liệu: {kline} → {str(e)}")

    def body_size(self):
        return abs(self.close - self.open)

    def candle_range(self):
        return self.high - self.low

    def direction(self):
        if self.close > self.open:
            return "BUY"
        elif self.close < self.open:
            return "SELL"
        return "DOJI"

    def average_price(self):
        return (self.open + self.close) / 2
    
    def upper_wick(self):
        return self.high - max(self.open, self.close)

    def lower_wick(self):
        return min(self.open, self.close) - self.low
    
    def wick_direction(self):
        """Xác định hướng chân nến: 'UP', 'DOWN', 'BALANCED'"""
        upper = self.upper_wick()
        lower = self.lower_wick()
        body =  self.body_size()

        if upper > lower and upper >= body:
            return "UP"
        if lower > upper and lower >= body:
            return "DOWN"
        else:
            return "BALANCED"
    
    def __str__(self):
        return f"[{self.timestamp}] O:{self.open} H:{self.high} L:{self.low} C:{self.close} V:{self.volume}"


# ========== BOT CHÍNH VỚI ĐÓNG LỆNH CHÍNH XÁC ==========
class IndicatorBot:
    def __init__(self, symbol, lev, percent, tp, sl, indicator, ws_manager):
        self.symbol = symbol.upper()
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.indicator = indicator
        self.ws_manager = ws_manager

        # ==== AI online learning ====
        self.classes = np.array([-1, 0, 1])  # SELL, NEUTRAL, BUY
        model_path = f"models/ai_{self.symbol}.pkl"
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(model_path):
            # Load model đã có
            self.model = joblib.load(model_path)
        else:
            # Nếu chưa có thì train mới
            print(f"⚡ Chưa có model cho {self.symbol}, đang train...")
            train_from_binance(self.symbol)  # train_ai sẽ tạo ai_model.pkl
        
            # Đổi tên file vừa train thành model riêng cho symbol
            if os.path.exists("ai_model.pkl"):
                os.rename("ai_model.pkl", model_path)
        
            # Load model vừa tạo
            self.model = joblib.load(model_path)
            print(f"✅ Model cho {self.symbol} đã được tạo và load")

        # Phần khởi tạo khác giữ nguyên
        self.check_position_status()
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        self.rsi_history = []

        self._stop = False
        self.position_open = False
        self.last_trade_time = 0
        self.last_rsi = 50
        self.position_check_interval = 60
        self.last_position_check = 0
        self.last_error_log_time = 0
        self.last_close_time = 0
        self.cooldown_period = 60  # Thời gian chờ sau khi đóng lệnh
        self.max_position_attempts = 3  # Số lần thử tối đa
        self.position_attempt_count = 0
        
        # Đăng ký với WebSocket Manager
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        
        # Bắt đầu thread chính
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log(f"🟢 Bot khởi động cho {self.symbol}")

    def log(self, message):
        """Ghi log và gửi qua Telegram"""
        logger.info(f"[{self.symbol}] {message}")
        send_telegram(f"<b>{self.symbol}</b>: {message}")

    def _handle_price_update(self, price):
        if self._stop: 
            return
            
        self.prices.append(price)
        # Giới hạn số lượng giá lưu trữ
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]
        rsi = calc_rsi(np.array(self.prices))
        if rsi is not None:
            self.rsi_history.append(rsi)
            if len(self.rsi_history) > 15:
                self.rsi_history = self.rsi_history[-15:]


        # ====== THÊM MỚI: tiện ích lấy klines nhanh ======
    def _fetch_klines(self, interval="5m", limit=50):
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval={interval}&limit={limit}"
        data = binance_api_request(url)
        if not data or len(data) < 20:
            return None
        return data  # danh sách klines gốc của Binance

    def _calc_rsi_series(self, closes, period=14):
        if len(closes) < period + 1:
            return [None] * len(closes)

        deltas = np.diff(closes)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(closes, dtype=float)
        rsi[:period] = 100. - 100. / (1. + rs)

        upval, downval = up, down
        for i in range(period, len(closes)):
            delta = deltas[i - 1]
            upval = (upval * (period - 1) + (delta if delta > 0 else 0)) / period
            downval = (downval * (period - 1) + (-delta if delta < 0 else 0)) / period
            rs = upval / downval if downval != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi


    # ====== THÊM MỚI: EMA cuối cùng (nhanh & gọn) ======
    def _ema_last(self, values, period):
        if len(values) < period:
            return None
        k = 2 / (period + 1)
        ema_val = float(values[0])
        for x in values[1:]:
            ema_val = float(x) * k + ema_val * (1 - k)
        return ema_val

    # ====== THÊM MỚI: ATR (dùng True Range) ======
    def _atr(self, highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return None
        trs = []
        for i in range(1, len(closes)):
            h = float(highs[i]); l = float(lows[i]); pc = float(closes[i-1])
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        if len(trs) < period:
            return None
        return sum(trs[-period:]) / period  # SMA ATR

    # ====== PHÂN LOẠI NẾN: chỉ còn TĂNG / GIẢM / MẠNH / YẾU / QUÁ ======
    def _candle_full(self, o, h, l, c, rsi, atr, ema_fast, ema_slow):
        body = abs(c - o)
        candle_range = h - l
        signal = "NEUTRAL"

        # Xác định nến xanh / đỏ
        if c > o:  # Nến tăng
            if rsi > 85:
                signal = "UP_OVERBOUGHT"
            elif rsi > 65:
                signal = "UP_STRONG"
            else:
                signal = "UP_WEAK"
        elif c < o:  # Nến giảm
            if rsi < 15:
                signal = "DOWN_OVERSOLD"
            elif rsi < 35:
                signal = "DOWN_STRONG"
            else:
                signal = "DOWN_WEAK"

        # Điều chỉnh mạnh/yếu bằng ATR + thân nến
        if atr:
            if candle_range >= 1.4 * atr and "WEAK" in signal:
                signal = signal.replace("WEAK", "STRONG")
            if body >= 0.6 * atr and "WEAK" in signal:
                signal = signal.replace("WEAK", "STRONG")

        # Lọc EMA trend: chỉ giữ tín hiệu thuận xu hướng
        if ema_fast and ema_slow:
            if "UP" in signal and ema_fast < ema_slow:
                signal = "NEUTRAL"  # bỏ BUY khi trend đang DOWN
            if "DOWN" in signal and ema_fast > ema_slow:
                signal = "NEUTRAL"  # bỏ SELL khi trend đang UP

        return signal

    # ====== LOGIC ĐỆ QUY (gọn) ======
    
    # ====== GET SIGNAL ======
    def get_signal(self):
        """
        Dùng AI online learning để dự đoán tín hiệu
        """
        try:
            data = self._fetch_klines(interval="5m", limit=100)
            if not data or len(data) < 50:
                return None
    
            closes = [float(k[4]) for k in data]
            highs  = [float(k[2]) for k in data]
            lows   = [float(k[3]) for k in data]
            volumes = [float(k[5]) for k in data]
    
            rsi = self._calc_rsi_series(closes, period=14)[-1]
            ema_fast = self._ema_last(closes, 9)
            ema_slow = self._ema_last(closes, 21)
            atr = self._atr(highs, lows, closes)
    
            if None in [rsi, ema_fast, ema_slow, atr]:
                return None
    
            features = np.array([rsi, ema_fast, ema_slow, atr, volumes[-1]]).reshape(1, -1)
    
            # AI dự đoán
            signal = self.ai_model.predict(features)[0]  # -1 SELL, 0 NEUTRAL, 1 BUY
    
            if signal == 1 and ema_fast > ema_slow:
                return "BUY"
            if signal == -1 and ema_fast < ema_slow:
                return "SELL"
            return None
    
        except Exception as e:
            self.log(f"Lỗi get_signal AI online: {str(e)}")
            return None


    def get_ema_crossover_signal(self, prices, short_period=9, long_period=21):
        if len(prices) < long_period:
            return None
    
        def ema(values, period):
            k = 2 / (period + 1)
            ema_val = float(values[0])
            for price in values[1:]:
                ema_val = float(price) * k + ema_val * (1 - k)
            return float(ema_val)
    
        short_ema = ema(prices[-long_period:], short_period)
        long_ema = ema(prices[-long_period:], long_period)
    
        if short_ema > long_ema:
            return "BUY"
        elif short_ema < long_ema:
            return "SELL"
        else:
            return None

    def update_model(self, data):
        """
        Học thêm từ dữ liệu nến mới:
        - Tính RSI, EMA, ATR, Volume
        - Tạo nhãn theo biến động giá trong 3 nến tới
        - Cập nhật model bằng partial_fit
        - Lưu lại model vào file
        """
        try:
            closes = [float(k[4]) for k in data]
            highs  = [float(k[2]) for k in data]
            lows   = [float(k[3]) for k in data]
            volumes = [float(k[5]) for k in data]
    
            rsi = self._calc_rsi_series(closes, 14)[-1]
            ema_fast = self._ema_last(closes, 9)
            ema_slow = self._ema_last(closes, 21)
            atr = self._atr(highs, lows, closes)
    
            if None in [rsi, ema_fast, ema_slow, atr]:
                return
    
            features = np.array([rsi, ema_fast, ema_slow, atr, volumes[-1]]).reshape(1, -1)
    
            # Nhãn dựa trên biến động giá 3 nến tới
            future_return = closes[-1] / closes[-4] - 1
            label = 0
            if future_return > 0.003:
                label = 1
            elif future_return < -0.003:
                label = -1
    
            self.ai_model.partial_fit(features, [label])
    
            # Lưu model lại
            joblib.dump(self.ai_model, f"models/ai_{self.symbol}.pkl")
            self.log(f"🤖 AI model đã học thêm (label={label})")
    
        except Exception as e:
            self.log(f"Lỗi update_model: {str(e)}")
    

    def _run(self):
        """Luồng chính quản lý bot với kiểm soát lỗi chặt chẽ"""
        while not self._stop:
            try:
                current_time = time.time()
                
                # Kiểm tra trạng thái vị thế định kỳ
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                signal = self.get_signal()
                
                # Xử lý logic giao dịch
                if not self.position_open and self.status == "waiting":
                    # Kiểm tra thời gian chờ sau khi đóng lệnh
                    if current_time - self.last_close_time < self.cooldown_period:
                        time.sleep(1)
                        continue

                    if signal and current_time - self.last_trade_time > 60:
                        self.open_position(signal)
                        self.last_trade_time = current_time
                # Kiểm tra TP/SL cho vị thế đang mở
                if self.position_open and self.status == "open":
                    self.check_tp_sl()
                    
                    if signal:
                        if (self.side == "BUY" and signal == "SELL") or (self.side == "SELL" and signal == "BUY"):
                            # Tính ROI hiện tại
                            current_price = self.prices[-1] if self.prices else get_current_price(self.symbol)
                            if self.entry > 0 and current_price > 0:
                                profit = (current_price - self.entry) * self.qty if self.side == "BUY" else (self.entry - current_price) * abs(self.qty)
                                invested = self.entry * abs(self.qty) / self.lev
                                roi = (profit / invested) * 100 if invested != 0 else 0
                    
                                if roi >= 20:
                                    self.close_position(f"🔄 ROI {roi:.2f}% vượt ngưỡng, đảo chiều sang {signal}")
                    
                time.sleep(1)
                # Kiểm tra tín hiệu ngược chiều để đóng vị thế
                
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"Lỗi hệ thống: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(1)

    def stop(self):
        self._stop = True
        self.ws_manager.remove_symbol(self.symbol)
        try:
            cancel_all_orders(self.symbol)
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Lỗi hủy lệnh: {str(e)}")
                self.last_error_log_time = time.time()
        self.log(f"🔴 Bot dừng cho {self.symbol}")

    def check_position_status(self):
        """Kiểm tra trạng thái vị thế từ API Binance với kiểm soát lỗi"""
        try:
            positions = get_positions(self.symbol)
            
            if not positions or len(positions) == 0:
                '''self.position_open = False
                self.status = "waiting"
                self.side = ""
                self.qty = 0
                self.entry = 0'''
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
                self.log(f"Lỗi kiểm tra vị thế: {str(e)}")
                self.last_error_log_time = time.time()

    def check_tp_sl(self):
        """Tự động kiểm tra và đóng lệnh khi đạt TP/SL với kiểm soát rủi ro"""
        if not self.position_open or not self.entry or not self.qty:
            return
            
        try:
            if len(self.prices) > 0:
                current_price = self.prices[-1]
            else:
                current_price = get_current_price(self.symbol)
                
            if current_price <= 0:
                return
                
            # Tính ROI
            if self.side == "BUY":
                profit = (current_price - self.entry) * self.qty
            else:
                profit = (self.entry - current_price) * abs(self.qty)
                
            # Tính % ROI dựa trên vốn ban đầu
            invested = self.entry * abs(self.qty) / self.lev
            if invested <= 0:
                return
                
            roi = (profit / invested) * 100
            
            # Kiểm tra TP/SL
            if roi >= self.tp:
                self.close_position(f"✅ Đạt TP {self.tp}% (ROI: {roi:.2f}%)")
            elif self.sl is not None and roi <= -self.sl:
                self.close_position(f"❌ Đạt SL {self.sl}% (ROI: {roi:.2f}%)")
                
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Lỗi kiểm tra TP/SL: {str(e)}")
                self.last_error_log_time = time.time()

    def open_position(self, side):
        # Kiểm tra lại trạng thái trước khi vào lệnh
        self.check_position_status()    
        try:
            # Hủy lệnh tồn đọng
            cancel_all_orders(self.symbol)
            
            # Đặt đòn bẩy
            if not set_leverage(self.symbol, self.lev):
                self.log(f"Không thể đặt đòn bẩy {self.lev}")
                return
            
            # Tính toán khối lượng
            balance = get_balance()
            if balance <= 0:
                self.log(f"Không đủ số dư USDT")
                return
            
            # Giới hạn % số dư sử dụng
            if self.percent > 100:
                self.percent = 100
            elif self.percent < 1:
                self.percent = 1
                
            usdt_amount = balance * (self.percent / 100)
            price = get_current_price(self.symbol)
            if price <= 0:
                self.log(f"Lỗi lấy giá")
                return
                
            step = get_step_size(self.symbol)
            if step <= 0:
                step = 0.001
            
            # Tính số lượng với đòn bẩy
            qty = (usdt_amount * self.lev) / price
            
            # Làm tròn số lượng theo step size
            if step > 0:
                steps = qty / step
                qty = round(steps) * step
            
            qty = max(qty, 0)
            qty = round(qty, 8)
            
            min_qty = step
            
            if qty < min_qty:
                self.log(f"⚠️ Số lượng quá nhỏ ({qty}), không đặt lệnh")
                return
                
            # Giới hạn số lần thử
            self.position_attempt_count += 1
            if self.position_attempt_count > self.max_position_attempts:
                self.log(f"⚠️ Đã đạt giới hạn số lần thử mở lệnh ({self.max_position_attempts})")
                self.position_attempt_count = 0
                return
                
            # Đặt lệnh
            res = place_order(self.symbol, side, qty)
            if not res:
                self.log(f"Lỗi khi đặt lệnh")
                return
                
            executed_qty = float(res.get('executedQty', 0))
            if executed_qty < 0:
                self.log(f"Lệnh không khớp, số lượng thực thi: {executed_qty}")
                return

            # Cập nhật trạng thái
            self.entry = float(res.get('avgPrice', price))
            self.side = side
            self.qty = executed_qty if side == "BUY" else -executed_qty
            self.status = "open"
            self.position_open = True
            self.position_attempt_count = 0  # Reset số lần thử
            
            # Thông báo qua Telegram
            message = (
                f"✅ <b>ĐÃ MỞ VỊ THẾ {self.symbol}</b>\n"
                f"📌 Hướng: {side}\n"
                f"🏷️ Giá vào: {self.entry:.4f}\n"
                f"📊 Khối lượng: {executed_qty}\n"
                f"💵 Giá trị: {executed_qty * self.entry:.2f} USDT\n"
                f" Đòn bẩy: {self.lev}x\n"
                f"🎯 TP: {self.tp}% | 🛡️ SL: {self.sl}%"
            )
            self.log(message)

        except Exception as e:
            self.position_open = False
            self.log(f"❌ Lỗi khi vào lệnh: {str(e)}")

    def close_position(self, reason=""):
        """Đóng vị thế với số lượng chính xác, không kiểm tra lại trạng thái"""
        try:
            # Hủy lệnh tồn đọng
            cancel_all_orders(self.symbol)
            
            if abs(self.qty) > 0:
                close_side = "SELL" if self.side == "BUY" else "BUY"
                close_qty = abs(self.qty)
                
                # Làm tròn số lượng CHÍNH XÁC
                step = get_step_size(self.symbol)
                if step > 0:
                    # Tính toán chính xác số bước
                    steps = close_qty / step
                    # Làm tròn đến số nguyên gần nhất
                    close_qty = round(steps) * step
                
                close_qty = max(close_qty, 0)
                close_qty = round(close_qty, 8)
                
                res = place_order(self.symbol, close_side, close_qty)
                if res:
                    price = float(res.get('avgPrice', 0))
                    # Thông báo qua Telegram
                    message = (
                        f"⛔ <b>ĐÃ ĐÓNG VỊ THẾ {self.symbol}</b>\n"
                        f"📌 Lý do: {reason}\n"
                        f"🏷️ Giá ra: {price:.4f}\n"
                        f"📊 Khối lượng: {close_qty}\n"
                        f"💵 Giá trị: {close_qty * price:.2f} USDT"
                    )
                    self.log(message)
                    
                    # Cập nhật trạng thái NGAY LẬP TỨC
                    self.status = "waiting"
                    self.side = ""
                    self.qty = 0
                    self.entry = 0
                    self.position_open = False
                    self.last_trade_time = time.time()
                    self.last_close_time = time.time()  # Ghi nhận thời điểm đóng lệnh
                else:
                    self.log(f"Lỗi khi đóng lệnh")
        except Exception as e:
            self.log(f"❌ Lỗi khi đóng lệnh: {str(e)}")

# ========== QUẢN LÝ BOT CHẠY NỀN VÀ TƯƠNG TÁC TELEGRAM ==========
class BotManager:
    def __init__(self):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}  # Lưu trạng thái người dùng
        self.admin_chat_id = TELEGRAM_CHAT_ID
        
        self.log("🟢 HỆ THỐNG BOT ĐÃ KHỞI ĐỘNG")
        
        # Bắt đầu thread kiểm tra trạng thái
        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()
        
        # Bắt đầu thread lắng nghe Telegram
        self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
        self.telegram_thread.start()
        
        # Gửi menu chính khi khởi động
        if self.admin_chat_id:
            self.send_main_menu(self.admin_chat_id)

    def log(self, message):
        """Ghi log hệ thống và gửi Telegram"""
        logger.info(f"[SYSTEM] {message}")
        send_telegram(f"<b>SYSTEM</b>: {message}")

    def send_main_menu(self, chat_id):
        """Gửi menu chính cho người dùng"""
        welcome = (
            "🤖 <b>BOT GIAO DỊCH FUTURES BINANCE</b>\n\n"
            "Chọn một trong các tùy chọn bên dưới:"
        )
        send_telegram(welcome, chat_id, create_menu_keyboard())

    def add_bot(self, symbol, lev, percent, tp, sl, indicator):
        if sl == 0:
            sl = None
        symbol = symbol.upper()
        if symbol in self.bots:
            self.log(f"⚠️ Đã có bot cho {symbol}")
            return False
            
        # Kiểm tra API key
        if not API_KEY or not API_SECRET:
            self.log("❌ Chưa cấu hình API Key và Secret Key!")
            return False
            
        try:
            # Kiểm tra kết nối API
            price = get_current_price(symbol)
            if price <= 0:
                self.log(f"❌ Không thể lấy giá cho {symbol}")
                return False
            
            # Kiểm tra vị thế hiện tại
            positions = get_positions(symbol)
            if positions and any(float(pos.get('positionAmt', 0)) != 0 for pos in positions):
                self.log(f"⚠️ Có vị thế mở cho {symbol}")
            
            # Tạo bot mới
            bot = IndicatorBot(
                symbol, lev, percent, tp, sl, 
                indicator, self.ws_manager
            )
            self.bots[symbol] = bot
            self.log(f"✅ Đã thêm bot: {symbol} | ĐB: {lev}x | %: {percent} | TP/SL: {tp}%/{sl}%")
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def stop_bot(self, symbol):
        symbol = symbol.upper()
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()
            if bot.status == "open":
                bot.close_position("⛔ Dừng bot thủ công")
            self.log(f"⛔ Đã dừng bot cho {symbol}")
            del self.bots[symbol]
            return True
        return False

    def stop_all(self):
        self.log("⛔ Đang dừng tất cả bot...")
        for symbol in list(self.bots.keys()):
            self.stop_bot(symbol)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

    def _status_monitor(self):
        """Kiểm tra và báo cáo trạng thái định kỳ"""
        while self.running:
            try:
                # Tính thời gian hoạt động
                uptime = time.time() - self.start_time
                hours, rem = divmod(uptime, 3600)
                minutes, seconds = divmod(rem, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                # Báo cáo số bot đang chạy
                active_bots = [s for s, b in self.bots.items() if not b._stop]
                
                # Báo cáo số dư tài khoản
                balance = get_balance()
                
                # Tạo báo cáo
                status_msg = (
                    f"📊 <b>BÁO CÁO HỆ THỐNG</b>\n"
                    f"⏱ Thời gian hoạt động: {uptime_str}\n"
                    f"🤖 Số bot đang chạy: {len(active_bots)}\n"
                    f"📈 Bot hoạt động: {', '.join(active_bots) if active_bots else 'Không có'}\n"
                    f"💰 Số dư khả dụng: {balance:.2f} USDT"
                )
                send_telegram(status_msg)
                
                # Log chi tiết
                for symbol, bot in self.bots.items():
                    if bot.status == "open":
                        status_msg = (
                            f"🔹 <b>{symbol}</b>\n"
                            f"📌 Hướng: {bot.side}\n"
                            f"🏷️ Giá vào: {bot.entry:.4f}\n"
                            f"📊 Khối lượng: {abs(bot.qty)}\n"
                            f" Đòn bẩy: {bot.lev}x\n"
                            f"🎯 TP: {bot.tp}% | 🛡️ SL: {bot.sl}%"
                        )
                        send_telegram(status_msg)
                
            except Exception as e:
                logger.error(f"Lỗi báo cáo trạng thái: {str(e)}")
            
            # Kiểm tra mỗi 6 giờ
            time.sleep(6 * 3600)

    def _telegram_listener(self):
        """Lắng nghe và xử lý tin nhắn từ Telegram"""
        last_update_id = 0
        
        while self.running:
            try:
                # Lấy tin nhắn mới
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
                            
                            # Chỉ xử lý tin nhắn từ admin
                            if chat_id != self.admin_chat_id:
                                continue
                            
                            # Cập nhật ID tin nhắn cuối
                            if update_id > last_update_id:
                                last_update_id = update_id
                            
                            # Xử lý tin nhắn
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    # Xử lý xung đột - chỉ có một instance của bot có thể lắng nghe
                    logger.error("Lỗi xung đột: Chỉ một instance bot có thể lắng nghe Telegram")
                    break
                
            except Exception as e:
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(5)

    def _handle_telegram_message(self, chat_id, text):
        """Xử lý tin nhắn từ người dùng"""
        # Lưu trạng thái người dùng
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        # Xử lý theo bước hiện tại
        if current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                symbol = text.upper()
                self.user_states[chat_id] = {
                    'step': 'waiting_leverage',
                    'symbol': symbol
                }
                send_telegram(f"Chọn đòn bẩy cho {symbol}:", chat_id, create_leverage_keyboard())
        
        elif current_step == 'waiting_leverage':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            elif 'x' in text:
                leverage = int(text.replace('', '').replace('x', '').strip())
                user_state['leverage'] = leverage
                user_state['step'] = 'waiting_percent'
                send_telegram(
                    f"📌 Cặp: {user_state['symbol']}\n Đòn bẩy: {leverage}x\n\nNhập % số dư muốn sử dụng (1-100):",
                    chat_id,
                    create_cancel_keyboard()
                )
        
        elif current_step == 'waiting_percent':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                try:
                    percent = float(text)
                    if 1 <= percent <= 100:
                        user_state['percent'] = percent
                        user_state['step'] = 'waiting_tp'
                        send_telegram(
                            f"📌 Cặp: {user_state['symbol']}\n ĐB: {user_state['leverage']}x\n📊 %: {percent}%\n\nNhập % Take Profit (ví dụ: 10):",
                            chat_id,
                            create_cancel_keyboard()
                        )
                    else:
                        send_telegram("⚠️ Vui lòng nhập % từ 1-100", chat_id)
                except:
                    send_telegram("⚠️ Giá trị không hợp lệ, vui lòng nhập số", chat_id)
        
        elif current_step == 'waiting_tp':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                try:
                    tp = float(text)
                    if tp > 0:
                        user_state['tp'] = tp
                        user_state['step'] = 'waiting_sl'
                        send_telegram(
                            f"📌 Cặp: {user_state['symbol']}\n ĐB: {user_state['leverage']}x\n📊 %: {user_state['percent']}%\n🎯 TP: {tp}%\n\nNhập % Stop Loss (ví dụ: 5):",
                            chat_id,
                            create_cancel_keyboard()
                        )
                    else:
                        send_telegram("⚠️ TP phải lớn hơn 0", chat_id)
                except:
                    send_telegram("⚠️ Giá trị không hợp lệ, vui lòng nhập số", chat_id)
        
        elif current_step == 'waiting_sl':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                try:
                    sl = float(text)
                    if sl >= 0:
                        # Thêm bot
                        symbol = user_state['symbol']
                        leverage = user_state['leverage']
                        percent = user_state['percent']
                        tp = user_state['tp']
                        
                        if self.add_bot(symbol, leverage, percent, tp, sl, "RSI"):
                            send_telegram(
                                f"✅ <b>ĐÃ THÊM BOT THÀNH CÔNG</b>\n\n"
                                f"📌 Cặp: {symbol}\n"
                                f" Đòn bẩy: {leverage}x\n"
                                f"📊 % Số dư: {percent}%\n"
                                f"🎯 TP: {tp}%\n"
                                f"🛡️ SL: {sl}%",
                                chat_id,
                                create_menu_keyboard()
                            )
                        else:
                            send_telegram("❌ Không thể thêm bot, vui lòng kiểm tra log", chat_id, create_menu_keyboard())
                        
                        # Reset trạng thái
                        self.user_states[chat_id] = {}
                    else:
                        send_telegram("⚠️ SL phải lớn hơn 0", chat_id)
                except:
                    send_telegram("⚠️ Giá trị không hợp lệ, vui lòng nhập số", chat_id)
        
        # Xử lý các lệnh chính
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
                for symbol, bot in self.bots.items():
                    status = "🟢 Mở" if bot.status == "open" else "🟡 Chờ"
                    message += f"🔹 {symbol} | {status} | {bot.side}\n"
                send_telegram(message, chat_id)
        
        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_symbol'}
            send_telegram("Chọn cặp coin:", chat_id, create_symbols_keyboard())
        
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id)
            else:
                message = "⛔ <b>CHỌN BOT ĐỂ DỪNG</b>\n\n"
                keyboard = []
                row = []
                
                for i, symbol in enumerate(self.bots.keys()):
                    message += f"🔹 {symbol}\n"
                    row.append({"text": f"⛔ {symbol}"})
                    if len(row) == 2 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                
                send_telegram(
                    message, 
                    chat_id, 
                    {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True}
                )
        
        elif text.startswith("⛔ "):
            symbol = text.replace("⛔ ", "").strip().upper()
            if symbol in self.bots:
                self.stop_bot(symbol)
                send_telegram(f"⛔ Đã gửi lệnh dừng bot {symbol}", chat_id, create_menu_keyboard())
            else:
                send_telegram(f"⚠️ Không tìm thấy bot {symbol}", chat_id, create_menu_keyboard())
        
        elif text == "💰 Số dư tài khoản":
            try:
                balance = get_balance()
                send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy số dư: {str(e)}", chat_id)
        
        elif text == "📈 Vị thế đang mở":
            try:
                positions = get_positions()
                if not positions:
                    send_telegram("📭 Không có vị thế nào đang mở", chat_id)
                    return
                
                message = "📈 <b>VỊ THẾ ĐANG MỞ</b>\n\n"
                for pos in positions:
                    position_amt = float(pos.get('positionAmt', 0))
                    if position_amt != 0:
                        symbol = pos.get('symbol', 'UNKNOWN')
                        entry = float(pos.get('entryPrice', 0))
                        side = "LONG" if position_amt > 0 else "SHORT"
                        pnl = float(pos.get('unRealizedProfit', 0))
                        
                        message += (
                            f"🔹 {symbol} | {side}\n"
                            f"📊 Khối lượng: {abs(position_amt):.4f}\n"
                            f"🏷️ Giá vào: {entry:.4f}\n"
                            f"💰 PnL: {pnl:.2f} USDT\n\n"
                        )
                
                send_telegram(message, chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy vị thế: {str(e)}", chat_id)
        
        # Gửi lại menu nếu không có lệnh phù hợp
        elif text:
            self.send_main_menu(chat_id)

# ========== HÀM KHỞI CHẠY CHÍNH ==========
def main():
    # Khởi tạo hệ thống
    manager = BotManager()
    
    # Thêm các bot từ cấu hình
    if BOT_CONFIGS:
        for config in BOT_CONFIGS:
            manager.add_bot(*config)
    else:
        manager.log("⚠️ Không có cấu hình bot nào được tìm thấy!")
    
    # Thông báo số dư ban đầu
    try:
        balance = get_balance()
        manager.log(f"💰 SỐ DƯ BAN ĐẦU: {balance:.2f} USDT")
    except Exception as e:
        manager.log(f"⚠️ Lỗi lấy số dư ban đầu: {str(e)}")
    
    try:
        # Giữ chương trình chạy
        while manager.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        manager.log("👋 Nhận tín hiệu dừng từ người dùng...")
    except Exception as e:
        manager.log(f"⚠️ LỖI HỆ THỐNG NGHIÊM TRỌNG: {str(e)}")
    finally:
        manager.stop_all()

if __name__ == "__main__":
    main()











