import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.signal import argrelextrema
import time
import requests
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantumShield Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PRESETS
# ══════════════════════════════════════════════════════════════════════════════
ACCIONES = {
    "NVIDIA (NVDA)":      "NVDA",
    "Apple (AAPL)":       "AAPL",
    "Microsoft (MSFT)":   "MSFT",
    "Tesla (TSLA)":       "TSLA",
    "Amazon (AMZN)":      "AMZN",
    "Meta (META)":        "META",
    "Alphabet (GOOGL)":   "GOOGL",
    "JPMorgan (JPM)":     "JPM",
    "S&P 500 ETF (SPY)":  "SPY",
    "IPSA Chile (^IPSA)": "^IPSA",
}

CRYPTOS = {
    "Bitcoin (BTC)":      "BTC-USD",
    "Ethereum (ETH)":     "ETH-USD",
    "Solana (SOL)":       "SOL-USD",
    "BNB (BNB)":          "BNB-USD",
    "XRP (XRP)":          "XRP-USD",
    "Cardano (ADA)":      "ADA-USD",
    "Avalanche (AVAX)":   "AVAX-USD",
    "Dogecoin (DOGE)":    "DOGE-USD",
    "Chainlink (LINK)":   "LINK-USD",
    "Polkadot (DOT)":     "DOT-USD",
}

# Mapa ticker app -> símbolo Binance (USDT)
BINANCE_SYMBOLS = {
    "BTC-USD":  "BTCUSDT",
    "ETH-USD":  "ETHUSDT",
    "SOL-USD":  "SOLUSDT",
    "BNB-USD":  "BNBUSDT",
    "XRP-USD":  "XRPUSDT",
    "ADA-USD":  "ADAUSDT",
    "AVAX-USD": "AVAXUSDT",
    "DOGE-USD": "DOGEUSDT",
    "LINK-USD": "LINKUSDT",
    "DOT-USD":  "DOTUSDT",
}

BENCHMARK_ACCIONES = "SPY"
BENCHMARK_CRYPTO   = "BTC-USD"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — Paper Trading & Notas
# ══════════════════════════════════════════════════════════════════════════════
if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []
if "notas" not in st.session_state:
    st.session_state.notas = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "analisis_cache" not in st.session_state:
    st.session_state.analisis_cache = {}
if "noticias_cache" not in st.session_state:
    st.session_state.noticias_cache = {}
if "signal_history" not in st.session_state:
    st.session_state.signal_history = {}   # {ticker: [{fecha, precio, senal, puntos}]}
if "finviz_cache" not in st.session_state:
    st.session_state.finviz_cache = {}     # {ticker: {news:[], fundamentals:{}}}

# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rsi(serie: pd.Series, periodo: int = 14) -> pd.Series:
    delta    = serie.diff()
    ganancia = delta.clip(lower=0).rolling(periodo).mean()
    perdida  = (-delta.clip(upper=0)).rolling(periodo).mean()
    rs       = ganancia / perdida.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calcular_macd(serie: pd.Series, rapido=12, lento=26, senal=9):
    ema_r       = serie.ewm(span=rapido, adjust=False).mean()
    ema_l       = serie.ewm(span=lento,  adjust=False).mean()
    macd_line   = ema_r - ema_l
    signal_line = macd_line.ewm(span=senal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calcular_bollinger(serie: pd.Series, periodo=20, desv=2):
    media = serie.rolling(periodo).mean()
    std   = serie.rolling(periodo).std()
    return media + desv * std, media, media - desv * std

def calcular_atr(df: pd.DataFrame, periodo=14) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([(h - l), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(periodo).mean()

def calcular_adx(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask]  = 0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_s    = tr.rolling(periodo).mean()
    plus_di  = 100 * (plus_dm.rolling(periodo).mean()  / atr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(periodo).mean() / atr_s.replace(0, np.nan))
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(periodo).mean()

def detectar_soportes_resistencias(df: pd.DataFrame, orden: int = 10):
    close    = df['Close'].values
    idx_max  = argrelextrema(close, np.greater, order=orden)[0]
    idx_min  = argrelextrema(close, np.less,    order=orden)[0]
    resistencias = [(df.index[i], close[i]) for i in idx_max]
    soportes     = [(df.index[i], close[i]) for i in idx_min]
    return resistencias[-5:], soportes[-5:]

def volumen_anomalo(df: pd.DataFrame, factor: float = 2.0) -> pd.Series:
    if 'Volume' not in df.columns:
        return pd.Series(False, index=df.index)
    vol_media = df['Volume'].rolling(20).mean()
    return df['Volume'] > (factor * vol_media)

def generar_senal(df: pd.DataFrame) -> dict:
    ultima  = df.iloc[-1]
    puntos  = 0
    razones = []

    if float(ultima['Close']) > float(ultima['EMA50']):
        puntos += 1; razones.append("✅ Precio sobre EMA50 (alcista)")
    else:
        puntos -= 1; razones.append("❌ Precio bajo EMA50 (bajista)")

    rsi_val = float(ultima['RSI'])
    if rsi_val < 30:
        puntos += 2; razones.append(f"✅ RSI sobrevendido ({rsi_val:.1f}) — posible rebote")
    elif rsi_val > 70:
        puntos -= 2; razones.append(f"❌ RSI sobrecomprado ({rsi_val:.1f}) — posible corrección")
    elif 40 <= rsi_val <= 60:
        puntos += 1; razones.append(f"✅ RSI en zona neutral ({rsi_val:.1f})")

    if float(ultima['MACD']) > float(ultima['MACD_SIGNAL']):
        puntos += 1; razones.append("✅ MACD sobre señal (impulso alcista)")
    else:
        puntos -= 1; razones.append("❌ MACD bajo señal (impulso bajista)")

    if float(ultima['Close']) < float(ultima['BB_LOWER']):
        puntos += 1; razones.append("✅ Precio bajo BB inferior (sobreventa)")
    elif float(ultima['Close']) > float(ultima['BB_UPPER']):
        puntos -= 1; razones.append("❌ Precio sobre BB superior (sobrecompra)")

    if   puntos >= 3:  senal, color = "🟢 COMPRA FUERTE", "#00ff88"
    elif puntos >= 1:  senal, color = "🟡 COMPRA DÉBIL",  "#ffd700"
    elif puntos <= -3: senal, color = "🔴 VENTA FUERTE",  "#ff4444"
    elif puntos <= -1: senal, color = "🟠 VENTA DÉBIL",   "#ff8c00"
    else:              senal, color = "⚪ NEUTRAL",        "#aaaaaa"

    atr_val = float(ultima['ATR']) if not pd.isna(ultima['ATR']) else 0
    adx_val = float(ultima['ADX']) if 'ADX' in ultima.index and not pd.isna(ultima['ADX']) else 0
    precio  = float(ultima['Close'])

    if adx_val >= 25:
        razones.append(f"✅ ADX {adx_val:.1f} — tendencia fuerte (señal más confiable)")
    elif adx_val >= 15:
        razones.append(f"⚠️ ADX {adx_val:.1f} — tendencia moderada")
    else:
        puntos = max(-1, min(1, puntos))
        razones.append(f"⚠️ ADX {adx_val:.1f} — sin tendencia clara (señal poco confiable)")

    return {
        "senal": senal, "color": color, "puntos": puntos,
        "razones": razones,
        "sl": precio - 2 * atr_val,
        "tp": precio + 3 * atr_val,
        "atr": atr_val, "rsi": rsi_val, "adx": adx_val,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS — Twelve Data (principal) + yfinance (fallback)
# ══════════════════════════════════════════════════════════════════════════════

PERIOD_DAYS    = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 732}
PERIOD_OUTPUT  = {"1mo": 35, "3mo": 95, "6mo": 190, "1y": 370, "2y": 740}
TD_API_KEY     = {"key": ""}   # se rellena desde el sidebar
TD_BASE_URL    = "https://api.twelvedata.com/time_series"
BINANCE_BASE   = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVALS = {"1mo": ("1d", 35), "3mo": ("1d", 95), "6mo": ("1d", 190), "1y": ("1d", 370), "2y": ("1d", 740)}
# Para MTF: intervalos fijos independientes del período principal
BINANCE_MTF = {
    "4H":  ("4h",  180),   # ~30 días de velas 4H
    "1D":  ("1d",  200),   # 200 días
    "1W":  ("1w",  104),   # 2 años de velas semanales
}
YF_MTF = {
    "4H":  ("4h",  "60d"),
    "1D":  ("1d",  "1y"),
    "1W":  ("1wk", "2y"),
}

def _binance_download(ticker: str, period: str) -> pd.DataFrame:
    """Descarga OHLCV desde Binance (gratuito, sin key, datos al segundo)."""
    sym = BINANCE_SYMBOLS.get(ticker)
    if not sym:
        return pd.DataFrame()
    interval, limit = BINANCE_INTERVALS.get(period, ("1d", 190))
    try:
        resp = requests.get(BINANCE_BASE, params={
            "symbol": sym, "interval": interval,
            "limit": limit,
        }, timeout=10)
        if not resp.ok:
            return pd.DataFrame()
        raw = resp.json()
        if not isinstance(raw, list) or not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=[
            "open_time","Open","High","Low","Close","Volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        df["Open"]  = pd.to_numeric(df["Open"],  errors="coerce")
        df["High"]  = pd.to_numeric(df["High"],  errors="coerce")
        df["Low"]   = pd.to_numeric(df["Low"],   errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"]= pd.to_numeric(df["Volume"],errors="coerce")
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        df.index.name = None
        return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

def _ticker_td(ticker: str) -> str:
    """Convierte tickers al formato de Twelve Data (BTC-USD -> BTC/USD, ^IPSA -> IPSA)."""
    t = ticker.replace("-USD", "/USD").replace("-", "/")
    if t.startswith("^"):
        t = t[1:]
    return t

def _td_download(ticker: str, period: str) -> pd.DataFrame:
    """Descarga OHLCV desde Twelve Data. Retorna DataFrame vacio si falla o sin key."""
    key = TD_API_KEY.get("key", "").strip()
    if not key:
        return pd.DataFrame()
    sym      = _ticker_td(ticker)
    outsize  = PERIOD_OUTPUT.get(period, 190)
    params   = {
        "symbol":     sym,
        "interval":   "1day",
        "outputsize": outsize,
        "apikey":     key,
        "format":     "JSON",
        "order":      "ASC",
    }
    try:
        resp = requests.get(TD_BASE_URL, params=params, timeout=15)
        if not resp.ok:
            return pd.DataFrame()
        data = resp.json()
        if data.get("status") == "error" or "values" not in data:
            return pd.DataFrame()
        values = data["values"]
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.index.name = None
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.rename(columns={"open":"Open","high":"High","low":"Low",
                            "close":"Close","volume":"Volume"}, inplace=True)
        return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

def _yf_fallback(ticker: str, period: str) -> pd.DataFrame:
    """Fallback a yfinance cuando no hay key de Twelve Data o falla."""
    from datetime import datetime, timedelta
    dias  = PERIOD_DAYS.get(period, 183)
    today = datetime.today()
    start = (today - timedelta(days=dias)).strftime("%Y-%m-%d")
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    for intento in range(3):
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True, timeout=15)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    nivel0 = df.columns.get_level_values(0).tolist()
                    nivel1 = df.columns.get_level_values(1).tolist()
                    campos = {"Open","High","Low","Close","Volume"}
                    df.columns = nivel0 if set(nivel0) & campos else nivel1
                return df
            df2 = yf.Ticker(ticker).history(start=start, end=end)
            if not df2.empty:
                return df2
            return pd.DataFrame()
        except Exception as e:
            nombre_error = type(e).__name__
            if "RateLimit" in nombre_error or "TooManyRequests" in nombre_error:
                if intento < 2:
                    time.sleep(2 ** (intento + 1))
                    continue
                raise
            raise
    return pd.DataFrame()

def _descargar_raw(ticker: str, period: str) -> pd.DataFrame:
    """
    Orden de prioridad:
      Crypto  → Binance (gratis, tiempo real) → Twelve Data → yfinance
      Acciones→ Twelve Data → yfinance
    """
    if ticker in BINANCE_SYMBOLS:
        df = _binance_download(ticker, period)
        if not df.empty:
            return df
    df = _td_download(ticker, period)
    if df.empty:
        df = _yf_fallback(ticker, period)
    return df

@st.cache_data(ttl=300)
def get_data(ticker: str, period: str) -> pd.DataFrame:
    df = _descargar_raw(ticker, period)
    if df.empty:
        return pd.DataFrame()
    df = df[[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]].copy()
    df.dropna(subset=["Close"], inplace=True)

    close = df["Close"]
    df["EMA20"]                                    = close.ewm(span=20, adjust=False).mean()
    df["EMA50"]                                    = close.ewm(span=50, adjust=False).mean()
    df["RSI"]                                      = calcular_rsi(close)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = calcular_macd(close)
    df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"]   = calcular_bollinger(close)
    df["ATR"]                                      = calcular_atr(df)
    df["ADX"]                                      = calcular_adx(df)
    if "Volume" in df.columns:
        df["VOL_ANOMALO"] = volumen_anomalo(df)
    return df

@st.cache_data(ttl=300)
def get_close_only(ticker: str, period: str) -> pd.Series:
    try:
        df = _descargar_raw(ticker, period)
    except Exception:
        return pd.Series(dtype=float, name=ticker)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    return df["Close"].rename(ticker)


@st.cache_data(ttl=120)
def get_data_mtf(ticker: str, tf: str) -> pd.DataFrame:
    """Descarga datos para un timeframe específico (4H, 1D, 1W)."""
    df = pd.DataFrame()

    # Binance para crypto
    if ticker in BINANCE_SYMBOLS:
        sym      = BINANCE_SYMBOLS[ticker]
        interval, limit = BINANCE_MTF[tf]
        try:
            resp = requests.get(BINANCE_BASE, params={
                "symbol": sym, "interval": interval, "limit": limit,
            }, timeout=10)
            if resp.ok:
                raw = resp.json()
                if isinstance(raw, list) and raw:
                    df = pd.DataFrame(raw, columns=[
                        "open_time","Open","High","Low","Close","Volume",
                        "close_time","qav","trades","tbbav","tbqav","ignore"
                    ])
                    for col in ["Open","High","Low","Close","Volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.index = pd.to_datetime(df["open_time"], unit="ms")
                    df.index.name = None
                    df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
        except Exception:
            pass

    # yfinance como fallback
    if df.empty:
        yf_interval, yf_period = YF_MTF[tf]
        try:
            raw = yf.download(ticker, period=yf_period, interval=yf_interval,
                              progress=False, auto_adjust=True)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    nivel0 = raw.columns.get_level_values(0).tolist()
                    nivel1 = raw.columns.get_level_values(1).tolist()
                    campos = {"Open","High","Low","Close","Volume"}
                    raw.columns = nivel0 if set(nivel0) & campos else nivel1
                df = raw[[c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]].dropna(subset=["Close"])
        except Exception:
            pass

    if df.empty:
        return pd.DataFrame()

    # Calcular indicadores
    close = df["Close"]
    df["EMA20"]                                    = close.ewm(span=20, adjust=False).mean()
    df["EMA50"]                                    = close.ewm(span=50, adjust=False).mean()
    df["RSI"]                                      = calcular_rsi(close)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = calcular_macd(close)
    df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"]   = calcular_bollinger(close)
    df["ATR"]                                      = calcular_atr(df)
    df["ADX"]                                      = calcular_adx(df)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# IA — Groq API (gratuito · llama-3.3-70b)
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_HEADERS = {"Content-Type": "application/json"}

def _ia_activa() -> bool:
    return bool(GROQ_HEADERS.get("Authorization", "").replace("Bearer ", "").strip())

def _groq(system: str, user: str, max_tokens: int = 1000) -> str:
    if not _ia_activa():
        return "❌ API key no configurada."
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=payload, timeout=30)
        if not resp.ok:
            try:
                detalle = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                detalle = resp.text[:300]
            if resp.status_code == 401:
                return "❌ API key inválida. Verifica en console.groq.com"
            if resp.status_code == 429:
                return "⚠️ Rate limit alcanzado. Espera unos segundos e intenta de nuevo."
            return f"❌ Error HTTP {resp.status_code}: {detalle}"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error de conexión: {e}"

def _groq_stream(system: str, messages: list, max_tokens: int = 1000):
    if not _ia_activa():
        yield "❌ API key no configurada."
        return
    mensajes_limpios = [
        {"role": m["role"], "content": str(m["content"])}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not mensajes_limpios:
        yield "❌ Sin mensajes para enviar."
        return
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + mensajes_limpios,
        "stream": True,
    }
    try:
        with requests.post(GROQ_API_URL, headers=GROQ_HEADERS,
                           json=payload, timeout=60, stream=True) as resp:
            if not resp.ok:
                try:
                    detalle = resp.json().get("error", {}).get("message", resp.text[:300])
                except Exception:
                    detalle = resp.text[:300]
                yield f"\n❌ Error {resp.status_code}: {detalle}"
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        yield delta.get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except Exception as e:
        yield f"\n❌ Error de conexión: {e}"

def _resumen_tecnico(ticker: str, df: pd.DataFrame, info: dict) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg  = ((float(last["Close"]) / float(prev["Close"])) - 1) * 100
    return f"""
Ticker: {ticker}
Precio: ${float(last["Close"]):,.4f} ({chg:+.2f}% hoy)
EMA20: {float(last["EMA20"]):,.4f} | EMA50: {float(last["EMA50"]):,.4f}
RSI(14): {float(last["RSI"]):.1f}
MACD: {float(last["MACD"]):.4f} | Señal MACD: {float(last["MACD_SIGNAL"]):.4f}
BB Superior: {float(last["BB_UPPER"]):,.4f} | BB Inferior: {float(last["BB_LOWER"]):,.4f}
ATR(14): {float(last["ATR"]):,.4f}
Stop Loss sugerido: ${info["sl"]:,.4f} | Take Profit sugerido: ${info["tp"]:,.4f}
Señal de confluencia: {info["senal"]} (puntuación: {info["puntos"]:+d})
Razones: {"; ".join(info["razones"])}
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════

def backtest_estrategia(df: pd.DataFrame, umbral_compra: int = 1, umbral_venta: int = -1, comision_pct: float = 0.1) -> dict:
    """
    comision_pct: % cobrado por lado (entrada + salida = 2x).
    Ejemplo: Binance 0.1% → 0.2% por operación completa.
    """
    resultados = []
    en_posicion = False
    precio_entrada = 0.0
    fecha_entrada  = None
    senal_entrada  = ""

    puntos_serie = []
    for i in range(len(df)):
        if i < 50:
            puntos_serie.append(0)
            continue
        fila = df.iloc[i]
        p = 0
        try:
            if float(fila['Close']) > float(fila['EMA50']): p += 1
            else: p -= 1
            rsi = float(fila['RSI'])
            if rsi < 30:   p += 2
            elif rsi > 70: p -= 2
            elif 40 <= rsi <= 60: p += 1
            if float(fila['MACD']) > float(fila['MACD_SIGNAL']): p += 1
            else: p -= 1
            if float(fila['Close']) < float(fila['BB_LOWER']):   p += 1
            elif float(fila['Close']) > float(fila['BB_UPPER']): p -= 1
        except Exception:
            pass
        puntos_serie.append(p)

    df = df.copy()
    df['_puntos'] = puntos_serie

    for i in range(51, len(df) - 1):
        pts_hoy    = df['_puntos'].iloc[i]
        precio_sig = float(df['Close'].iloc[i + 1])
        fecha_sig  = df.index[i + 1]

        if not en_posicion:
            if pts_hoy >= umbral_compra:
                en_posicion    = True
                precio_entrada = precio_sig
                fecha_entrada  = fecha_sig
                senal_entrada  = f"+{pts_hoy}"
        else:
            if pts_hoy <= umbral_venta or i == len(df) - 2:
                retorno_bruto = (precio_sig - precio_entrada) / precio_entrada * 100
                retorno  = retorno_bruto - (comision_pct * 2)   # entrada + salida
                duracion = (fecha_sig - fecha_entrada).days if hasattr(fecha_sig - fecha_entrada, 'days') else 1
                resultados.append({
                    "Entrada":        fecha_entrada.strftime("%d/%m/%Y") if hasattr(fecha_entrada, 'strftime') else str(fecha_entrada),
                    "Salida":         fecha_sig.strftime("%d/%m/%Y")     if hasattr(fecha_sig,      'strftime') else str(fecha_sig),
                    "Precio entrada": round(precio_entrada, 4),
                    "Precio salida":  round(precio_sig, 4),
                    "Retorno bruto %": round(retorno_bruto, 2),
                    "Retorno %":      round(retorno, 2),
                    "Días":           duracion,
                    "Señal entrada":  senal_entrada,
                    "Resultado":      "✅ Ganada" if retorno > 0 else "❌ Perdida",
                })
                en_posicion = False

    if not resultados:
        return {"ops": pd.DataFrame(), "win_rate": 0, "retorno_total": 0,
                "retorno_bh": 0, "n_ops": 0, "promedio_op": 0, "max_ganancia": 0, "max_perdida": 0,
                "sharpe": 0, "max_drawdown": 0}

    df_ops   = pd.DataFrame(resultados)
    n_ops    = len(df_ops)
    ganadas  = (df_ops['Retorno %'] > 0).sum()
    win_rate = round(ganadas / n_ops * 100, 1) if n_ops > 0 else 0
    ret_bh   = round((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[50]) - 1) * 100, 2)

    # Sharpe ratio (asume tasa libre de riesgo = 0, retornos por operación)
    retornos = df_ops['Retorno %']
    sharpe = round(retornos.mean() / retornos.std() * np.sqrt(n_ops), 2) if retornos.std() > 0 else 0

    # Max Drawdown sobre la curva de equity acumulada
    equity   = retornos.cumsum()
    peak     = equity.cummax()
    drawdown = equity - peak
    max_dd   = round(float(drawdown.min()), 2)

    return {
        "ops": df_ops,
        "win_rate": win_rate,
        "retorno_total": round(retornos.sum(), 2),
        "retorno_bh": ret_bh,
        "n_ops": n_ops,
        "promedio_op": round(retornos.mean(), 2),
        "max_ganancia": round(retornos.max(), 2),
        "max_perdida":  round(retornos.min(), 2),
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
    }

# ══════════════════════════════════════════════════════════════════════════════
# FINVIZ — Noticias + Fundamentales (scraping gratuito)
# ══════════════════════════════════════════════════════════════════════════════

FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

@st.cache_data(ttl=600)
def finviz_scrape(ticker: str) -> dict:
    """
    Extrae noticias recientes y datos fundamentales de Finviz.
    Funciona solo para acciones USA (no crypto).
    Retorna {"news": [...], "fundamentals": {...}, "ok": bool}
    """
    result = {"news": [], "fundamentals": {}, "ok": False}
    if not BS4_OK:
        return result
    # Finviz solo tiene datos para acciones USA, no crypto ni índices
    if "-USD" in ticker or ticker.startswith("^"):
        return result
    try:
        url  = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
        resp = requests.get(url, headers=FINVIZ_HEADERS, timeout=10)
        if not resp.ok:
            return result
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Fundamentales ──────────────────────────────────────
        fundamentals = {}
        tabla = soup.find("table", class_="snapshot-table2")
        if not tabla:
            tabla = soup.find("table", {"class": lambda c: c and "snapshot" in c})
        if tabla:
            celdas = tabla.find_all("td")
            for i in range(0, len(celdas) - 1, 2):
                key = celdas[i].get_text(strip=True)
                val = celdas[i+1].get_text(strip=True)
                if key and val:
                    fundamentals[key] = val

        # ── Noticias ───────────────────────────────────────────
        noticias = []
        tabla_news = soup.find("table", id="news-table")
        if tabla_news:
            filas = tabla_news.find_all("tr")
            fecha_actual = ""
            for fila in filas[:20]:
                celdas = fila.find_all("td")
                if len(celdas) < 2:
                    continue
                fecha_td = celdas[0].get_text(strip=True)
                if any(c.isalpha() for c in fecha_td):
                    partes    = fecha_td.split()
                    fecha_actual = partes[0] if len(partes) > 1 else fecha_actual
                    hora      = partes[-1] if len(partes) > 1 else fecha_td
                else:
                    hora = fecha_td
                enlace = celdas[1].find("a")
                fuente = celdas[1].find("span")
                if enlace:
                    noticias.append({
                        "fecha":  f"{fecha_actual} {hora}".strip(),
                        "titulo": enlace.get_text(strip=True),
                        "url":    enlace.get("href", "#"),
                        "fuente": fuente.get_text(strip=True) if fuente else "",
                    })

        result["news"]         = noticias
        result["fundamentals"] = fundamentals
        result["ok"]           = bool(fundamentals or noticias)
        return result
    except Exception:
        return result

# ══════════════════════════════════════════════════════════════════════════════
# FIBONACCI — Niveles de retroceso automáticos
# ══════════════════════════════════════════════════════════════════════════════

def calcular_fibonacci(df: pd.DataFrame) -> dict:
    """Calcula niveles de retroceso Fibonacci del período completo."""
    maximo = float(df['High'].max())
    minimo = float(df['Low'].min())
    rango  = maximo - minimo
    niveles = {
        "0%":    maximo,
        "23.6%": maximo - 0.236 * rango,
        "38.2%": maximo - 0.382 * rango,
        "50%":   maximo - 0.500 * rango,
        "61.8%": maximo - 0.618 * rango,
        "78.6%": maximo - 0.786 * rango,
        "100%":  minimo,
    }
    return niveles

FIBONACCI_COLORES = {
    "0%":    "rgba(255,255,255,0.5)",
    "23.6%": "rgba(255,215,0,0.6)",
    "38.2%": "rgba(0,207,255,0.6)",
    "50%":   "rgba(0,255,136,0.7)",
    "61.8%": "rgba(0,207,255,0.6)",
    "78.6%": "rgba(255,140,0,0.6)",
    "100%":  "rgba(255,255,255,0.5)",
}

# ══════════════════════════════════════════════════════════════════════════════
# DIVERGENCIAS RSI
# ══════════════════════════════════════════════════════════════════════════════

def detectar_divergencias(df: pd.DataFrame, ventana: int = 5) -> list:
    """
    Detecta divergencias alcistas y bajistas entre precio y RSI.
    Retorna lista de {tipo, fecha1, fecha2, precio1, precio2, rsi1, rsi2}
    """
    if 'RSI' not in df.columns or len(df) < ventana * 3:
        return []

    close = df['Close'].values
    rsi   = df['RSI'].values
    idx   = df.index
    divs  = []

    idx_min_precio = argrelextrema(close, np.less,    order=ventana)[0]
    idx_max_precio = argrelextrema(close, np.greater, order=ventana)[0]

    # Divergencia alcista: precio hace mínimo más bajo, RSI hace mínimo más alto
    for i in range(1, len(idx_min_precio)):
        i1, i2 = idx_min_precio[i-1], idx_min_precio[i]
        if close[i2] < close[i1] and rsi[i2] > rsi[i1] and not np.isnan(rsi[i1]) and not np.isnan(rsi[i2]):
            divs.append({
                "tipo":   "alcista",
                "fecha1": idx[i1], "fecha2": idx[i2],
                "precio1": close[i1], "precio2": close[i2],
                "rsi1":    rsi[i1],   "rsi2":    rsi[i2],
            })

    # Divergencia bajista: precio hace máximo más alto, RSI hace máximo más bajo
    for i in range(1, len(idx_max_precio)):
        i1, i2 = idx_max_precio[i-1], idx_max_precio[i]
        if close[i2] > close[i1] and rsi[i2] < rsi[i1] and not np.isnan(rsi[i1]) and not np.isnan(rsi[i2]):
            divs.append({
                "tipo":   "bajista",
                "fecha1": idx[i1], "fecha2": idx[i2],
                "precio1": close[i1], "precio2": close[i2],
                "rsi1":    rsi[i1],   "rsi2":    rsi[i2],
            })

    # Solo las 3 más recientes de cada tipo
    alcistas = [d for d in divs if d["tipo"] == "alcista"][-3:]
    bajistas = [d for d in divs if d["tipo"] == "bajista"][-3:]
    return alcistas + bajistas

# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL DE SEÑALES — Registro automático en sesión
# ══════════════════════════════════════════════════════════════════════════════

def registrar_senal(ticker: str, info: dict, precio: float):
    """Guarda la señal actual en el historial si es diferente a la última."""
    hist = st.session_state.signal_history.setdefault(ticker, [])
    nueva_senal = info["senal"]
    ultima_senal = hist[-1]["Señal"] if hist else None
    if ultima_senal != nueva_senal:
        hist.append({
            "Fecha":  datetime.now().strftime("%d/%m/%Y %H:%M"),
            "Ticker": ticker,
            "Señal":  nueva_senal,
            "Puntos": info["puntos"],
            "Precio": round(precio, 4),
            "RSI":    round(info["rsi"], 1),
            "ADX":    round(info["adx"], 1),
        })
        st.session_state.signal_history[ticker] = hist[-50:]

# ══════════════════════════════════════════════════════════════════════════════
# FINVIZ — Patrones chartistas (scraping homepage)
# ══════════════════════════════════════════════════════════════════════════════

# Descripciones de cada patrón en español
PATTERN_INFO = {
    "Trendline Supp.":  {
        "emoji": "📈", "bias": "alcista",
        "desc": "El precio rebotó al alza desde una línea de tendencia ascendente. Indica soporte dinámico y continuación de la tendencia alcista.",
        "accion": "Buscar entradas LONG cerca de la línea, con SL por debajo de ella.",
    },
    "Trendline Resist.": {
        "emoji": "📉", "bias": "bajista",
        "desc": "El precio chocó contra una línea de tendencia descendente. Indica resistencia dinámica y probable continuación bajista.",
        "accion": "Posible entrada SHORT en el rechazo, o esperar ruptura confirmada para LONG.",
    },
    "Horizontal S/R": {
        "emoji": "↔️", "bias": "neutro",
        "desc": "El precio está en un nivel horizontal clave que actuó como soporte y resistencia en el pasado.",
        "accion": "Esperar reacción clara en el nivel. Ruptura con volumen confirma la dirección.",
    },
    "Wedge Up": {
        "emoji": "🔺", "bias": "bajista",
        "desc": "Cuña ascendente — máximos y mínimos suben pero el rango se estrecha. Patrón de agotamiento alcista, suele romper a la baja.",
        "accion": "Esperar ruptura de la línea inferior. Objetivo = altura de la cuña.",
    },
    "Wedge": {
        "emoji": "🔷", "bias": "neutro",
        "desc": "Cuña simétrica — precio se comprime entre dos líneas convergentes. La ruptura puede ser en cualquier dirección.",
        "accion": "Operar la ruptura en la dirección que ocurra, con volumen de confirmación.",
    },
    "Wedge Down": {
        "emoji": "🔻", "bias": "alcista",
        "desc": "Cuña descendente — máximos y mínimos bajan pero el rango se estrecha. Patrón de agotamiento bajista, suele romper al alza.",
        "accion": "Esperar ruptura de la línea superior. Objetivo = altura de la cuña.",
    },
    "Triangle Asc.": {
        "emoji": "△", "bias": "alcista",
        "desc": "Triángulo ascendente — resistencia horizontal + mínimos crecientes. Compradores acumulando presión. Suele romper al alza.",
        "accion": "Entrada en ruptura del techo horizontal con volumen. SL bajo el último mínimo.",
    },
    "Triangle Desc.": {
        "emoji": "▽", "bias": "bajista",
        "desc": "Triángulo descendente — soporte horizontal + máximos decrecientes. Vendedores dominando. Suele romper a la baja.",
        "accion": "Entrada SHORT en ruptura del soporte con volumen. SL sobre el último máximo.",
    },
    "Channel Up": {
        "emoji": "📊", "bias": "alcista",
        "desc": "Canal ascendente — precio sube entre dos líneas paralelas. Tendencia alcista establecida.",
        "accion": "Comprar en rebotes en la línea inferior del canal. Toma de ganancias en línea superior.",
    },
    "Channel": {
        "emoji": "📊", "bias": "neutro",
        "desc": "Canal lateral — precio oscila entre soporte y resistencia horizontales. Mercado en rango.",
        "accion": "Comprar cerca del soporte, vender cerca de la resistencia. Operar la ruptura cuando ocurra.",
    },
    "Channel Down": {
        "emoji": "📉", "bias": "bajista",
        "desc": "Canal descendente — precio baja entre dos líneas paralelas. Tendencia bajista establecida.",
        "accion": "Evitar compras. SHORT en rebotes a la línea superior. Esperar ruptura alcista para cambio de tendencia.",
    },
    "Double Top": {
        "emoji": "🔔", "bias": "bajista",
        "desc": "Doble techo — el precio alcanzó dos máximos similares sin poder superarlos. Señal de inversión bajista muy fiable.",
        "accion": "Entrada SHORT en ruptura del cuello (mínimo entre los dos techos). Objetivo = distancia techo-cuello.",
    },
    "Multiple Top": {
        "emoji": "🔔🔔", "bias": "bajista",
        "desc": "Múltiples techos — tres o más intentos fallidos de superar un nivel. Resistencia muy fuerte, señal bajista.",
        "accion": "SHORT en la zona de resistencia con SL justo por encima. Mayor confiabilidad que el doble techo.",
    },
    "Double Bottom": {
        "emoji": "🏔️", "bias": "alcista",
        "desc": "Doble suelo — el precio tocó dos mínimos similares sin poder bajar más. Señal de inversión alcista muy fiable.",
        "accion": "Entrada LONG en ruptura del cuello (máximo entre los dos suelos). Objetivo = distancia suelo-cuello.",
    },
    "Multiple Bottom": {
        "emoji": "🏔️🏔️", "bias": "alcista",
        "desc": "Múltiples suelos — tres o más rebotes desde el mismo nivel. Soporte muy fuerte, señal alcista.",
        "accion": "LONG en la zona de soporte con SL justo por debajo. Mayor confiabilidad que el doble suelo.",
    },
    "Head&Shoulders": {
        "emoji": "👤", "bias": "bajista",
        "desc": "Hombro-Cabeza-Hombro — patrón de inversión bajista clásico. Tres picos donde el central es el más alto.",
        "accion": "SHORT en ruptura del cuello con volumen. Objetivo = distancia cabeza-cuello proyectada hacia abajo.",
    },
}

PATTERN_BIAS_COLOR = {
    "alcista": "#00ff88",
    "bajista": "#ff4444",
    "neutro":  "#ffd700",
}

@st.cache_data(ttl=900)
def finviz_patterns() -> dict:
    """
    Scrape los patrones chartistas del homepage de Finviz.
    Retorna {patron: [ticker1, ticker2, ...]}
    """
    result = {}
    if not BS4_OK:
        return result
    try:
        resp = requests.get("https://finviz.com/", headers=FINVIZ_HEADERS, timeout=12)
        if not resp.ok:
            return result
        soup = BeautifulSoup(resp.text, "html.parser")

        # Buscar todas las tablas de patrones
        # Finviz las pone en tablas con celdas de tickers y la última columna es el patrón
        for tabla in soup.find_all("table"):
            filas = tabla.find_all("tr")
            for fila in filas:
                celdas = fila.find_all("td")
                if not celdas:
                    continue
                # La última celda suele ser el enlace del patrón
                ultima = celdas[-1]
                enlace = ultima.find("a")
                if not enlace:
                    continue
                patron_texto = enlace.get_text(strip=True)
                if patron_texto not in PATTERN_INFO:
                    continue
                # Las celdas anteriores son tickers
                tickers_fila = []
                for celda in celdas[:-1]:
                    t_link = celda.find("a")
                    if t_link:
                        sym = t_link.get_text(strip=True).upper()
                        if sym and re.match(r'^[A-Z]{1,5}$', sym):
                            tickers_fila.append(sym)
                if tickers_fila:
                    result.setdefault(patron_texto, []).extend(tickers_fila)

        # Deduplicar
        for k in result:
            result[k] = list(dict.fromkeys(result[k]))
        return result
    except Exception:
        return result

# ══════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN DE TICKER
# ══════════════════════════════════════════════════════════════════════════════

TICKER_REGEX = re.compile(r'^[A-Z0-9\.\^\-]{1,12}$')

def validar_ticker(t: str) -> tuple:
    t = t.strip().upper()
    if not t:
        return False, "El ticker no puede estar vacío."
    if not TICKER_REGEX.match(t):
        return False, f"Ticker inválido: '{t}'. Solo letras, números, puntos, guiones y ^ (máx 12 caracteres)."
    return True, ""

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ QuantumShield Pro")
    st.markdown("---")

    tipo = st.radio("Tipo de activo", ["📈 Acciones", "₿ Criptomonedas", "✏️ Ticker manual"])

    if tipo == "📈 Acciones":
        nombre    = st.selectbox("Selecciona acción", list(ACCIONES.keys()))
        ticker    = ACCIONES[nombre]
        benchmark = BENCHMARK_ACCIONES
        preset_dict = ACCIONES
    elif tipo == "₿ Criptomonedas":
        nombre    = st.selectbox("Selecciona crypto", list(CRYPTOS.keys()))
        ticker    = CRYPTOS[nombre]
        benchmark = BENCHMARK_CRYPTO
        preset_dict = CRYPTOS
    else:
        _ticker_raw = st.text_input("Ticker personalizado", value="NVDA").upper().strip()
        _ok, _msg   = validar_ticker(_ticker_raw)
        if not _ok:
            st.error(_msg)
            ticker = "NVDA"
        else:
            ticker = _ticker_raw
        benchmark   = BENCHMARK_ACCIONES
        preset_dict = ACCIONES

    st.markdown("---")
    period = st.selectbox("Período", ["1mo","3mo","6mo","1y","2y"], index=2)

    st.markdown("---")
    mostrar_bb      = st.checkbox("Bandas de Bollinger",  value=True)
    mostrar_ema     = st.checkbox("EMA 20 / EMA 50",      value=True)
    mostrar_volumen = st.checkbox("Volumen",               value=True)
    mostrar_sr      = st.checkbox("Soporte/Resistencia",   value=True)
    mostrar_vanom   = st.checkbox("Volumen anómalo",       value=True)
    mostrar_fib     = st.checkbox("Fibonacci",             value=False)
    mostrar_divs    = st.checkbox("Divergencias RSI",      value=True)

    st.markdown("---")
    st.subheader("📡 Datos — Twelve Data")
    td_key_input = st.text_input(
        "Twelve Data API Key",
        type="password",
        placeholder="tu_api_key...",
        help="Gratis en twelvedata.com · 800 req/día · Sin tarjeta",
    )
    if td_key_input:
        TD_API_KEY["key"] = td_key_input
        st.success("Twelve Data listo ✓", icon="📡")
    else:
        try:
            TD_API_KEY["key"] = st.secrets["TD_API_KEY"]
            st.success("Twelve Data desde secrets ✓", icon="📡")
        except Exception:
            st.warning("Sin key → usando Yahoo Finance (fallback)", icon="⚠️")
            st.markdown("[Obtener key gratis →](https://twelvedata.com/register)")

    st.markdown("---")
    st.subheader("🤖 IA — Groq (gratis)")
    groq_key_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Gratis en console.groq.com · Sin tarjeta de crédito",
    )
    if groq_key_input:
        GROQ_HEADERS["Authorization"] = f"Bearer {groq_key_input}"
        st.success("Groq listo ✓", icon="🔑")
    else:
        try:
            GROQ_HEADERS["Authorization"] = f"Bearer {st.secrets['GROQ_API_KEY']}"
            st.success("Groq desde secrets ✓", icon="🔑")
        except Exception:
            st.warning("Sin API key — IA desactivada", icon="⚠️")
            st.markdown("[Obtener clave gratis →](https://console.groq.com)")

    st.markdown("---")
    st.subheader("🔄 Auto-Refresh")
    auto_refresh  = st.checkbox("Activar auto-refresh", value=False)
    refresh_secs  = st.selectbox("Intervalo", [30, 60, 120, 300],
                                  format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}min",
                                  index=1)
    if auto_refresh:
        st.info(f"♻️ Recargando cada {refresh_secs}s", icon="⏱️")

    st.markdown("---")
    fuente = "Twelve Data" if TD_API_KEY.get("key") else "Yahoo Finance (fallback)"
    st.caption(f"Datos vía {fuente} · Caché 5 min")

# ══════════════════════════════════════════════════════════════════════════════
# DATOS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

st.title("🛡️ QuantumShield Pro — Trading Terminal")

# ── Auto-refresh ──────────────────────────────────────────────
if auto_refresh:
    import streamlit.components.v1 as components
    # Countdown visible + rerun automático
    components.html(f"""
    <div id="refresh-bar" style="
        background:#1a1a2e;border:1px solid #00cfff33;
        border-radius:8px;padding:8px 16px;
        font-family:monospace;color:#00cfff;font-size:13px;
        display:flex;align-items:center;gap:10px;margin-bottom:4px;">
      <span>⏱️ Auto-refresh en</span>
      <span id="cnt" style="font-weight:bold;color:#00ff88">{refresh_secs}</span>
      <span>segundos</span>
      <div style="flex:1;background:#0d1117;border-radius:4px;height:6px;overflow:hidden;">
        <div id="bar" style="height:100%;background:#00cfff;
             animation:shrink {refresh_secs}s linear forwards;border-radius:4px;"></div>
      </div>
    </div>
    <style>
      @keyframes shrink {{from{{width:100%}}to{{width:0%}}}}
    </style>
    <script>
      var secs = {refresh_secs};
      var iv = setInterval(function(){{
        secs--;
        var el = document.getElementById('cnt');
        if(el) el.textContent = secs;
        if(secs <= 0){{ clearInterval(iv); window.location.reload(); }}
      }}, 1000);
    </script>
    """, height=50)

if not ticker:
    st.warning("Ingresa un ticker válido en el panel izquierdo.")
    st.stop()

with st.spinner(f"Analizando {ticker}..."):
    try:
        df = get_data(ticker, period)
    except Exception as e:
        nombre_e = type(e).__name__
        if "RateLimit" in nombre_e or "TooManyRequests" in nombre_e:
            st.error(
                "⚠️ **Límite de peticiones alcanzado.** "
                "Si usas Twelve Data, revisa tu cuota en twelvedata.com. "
                "Espera unos segundos y recarga la página.",
                icon="🚦",
            )
        else:
            st.error(f"Error al descargar datos para **{ticker}**: {nombre_e}")
        st.stop()

if df.empty:
    st.error(f"No se encontraron datos para **{ticker}**. Verifica que el ticker sea válido.")
    st.stop()

info  = generar_senal(df)
last  = df.iloc[-1]
prev  = df.iloc[-2] if len(df) > 1 else last
price = float(last['Close'])
chg   = ((price / float(prev['Close'])) - 1) * 100 if float(prev['Close']) > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📊 Análisis",
    "🌐 Señales del Mercado",
    "📉 Comparación Relativa",
    "🔗 Correlación",
    "📒 Paper Trading & Notas",
    "🤖 Asistente IA",
    "🧪 Backtesting",
    "⏱️ Multi-Timeframe",
    "🔍 Screener",
    "📰 Noticias & Fundamentales",
    "🕯️ Patrones Chartistas",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANÁLISIS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Registrar señal en historial
    registrar_senal(ticker, info, price)

    # Banner fuente de datos
    es_crypto = ticker in BINANCE_SYMBOLS
    if es_crypto:
        st.success("📡 Datos en tiempo real vía **Binance** (sin delay)", icon="⚡")
    elif TD_API_KEY.get("key"):
        st.info("📡 Datos vía **Twelve Data**", icon="📊")
    else:
        st.warning("📡 Datos vía **Yahoo Finance** (posible delay de 1 día) — agrega Twelve Data key para acciones en tiempo real", icon="⚠️")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Precio",      f"${price:,.4f}",       f"{chg:+.2f}%")
    k2.metric("RSI (14)",    f"{info['rsi']:.1f}")
    k3.metric("ADX (14)",    f"{info['adx']:.1f}",
              delta="Tendencia fuerte" if info['adx'] >= 25 else ("Moderada" if info['adx'] >= 15 else "Sin tendencia"))
    k4.metric("ATR (14)",    f"${info['atr']:,.4f}")
    k5.metric("Stop Loss",   f"${info['sl']:,.4f}")
    k6.metric("Take Profit", f"${info['tp']:,.4f}")

    fecha_str = last.name.strftime('%d/%m/%Y') if hasattr(last.name, 'strftime') else str(last.name)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
        border:1px solid {info['color']};border-radius:12px;
        padding:20px 28px;margin:16px 0;
        box-shadow:0 0 20px {info['color']}33;">
      <h2 style="color:{info['color']};margin:0 0 6px;font-size:1.5rem;">
        Señal de Confluencia: {info['senal']}
      </h2>
      <p style="color:#ccc;margin:0;font-size:.9rem;">
        Puntuación: <strong style="color:{info['color']}">{info['puntos']:+d}</strong> pts
        &nbsp;|&nbsp; {fecha_str}
      </p>
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Detalle de la señal"):
        for r in info['razones']:
            st.markdown(f"- {r}")

    tiene_vol   = mostrar_volumen and 'Volume' in df.columns
    rows_n      = 4 if tiene_vol else 3
    row_heights = [0.48, 0.13, 0.20, 0.19] if tiene_vol else [0.55, 0.22, 0.23]

    fig = make_subplots(
        rows=rows_n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Precio",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ), row=1, col=1)

    if mostrar_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], name="BB Sup",
            line=dict(color="rgba(100,180,255,.6)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_MID'], name="BB Med",
            line=dict(color="rgba(100,180,255,.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], name="BB Inf",
            line=dict(color="rgba(100,180,255,.6)", width=1, dash="dot"),
            fill='tonexty', fillcolor="rgba(100,180,255,.04)"), row=1, col=1)

    if mostrar_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20",
            line=dict(color="#ffd700", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name="EMA50",
            line=dict(color="#ff8c00", width=1.5)), row=1, col=1)

    if mostrar_sr and len(df) >= 25:
        resistencias, soportes = detectar_soportes_resistencias(df)
        for _, nivel in resistencias:
            fig.add_hline(y=nivel, line_color="rgba(255,68,68,0.5)", line_dash="dot",
                          annotation_text=f"R {nivel:,.2f}",
                          annotation_font_color="#ff6666",
                          annotation_position="top right", row=1, col=1)
        for _, nivel in soportes:
            fig.add_hline(y=nivel, line_color="rgba(0,255,136,0.5)", line_dash="dot",
                          annotation_text=f"S {nivel:,.2f}",
                          annotation_font_color="#00ff88",
                          annotation_position="bottom right", row=1, col=1)

    fig.add_hline(y=info['tp'], line_color="#00ff88", line_dash="dash",
                  annotation_text=f"TP {info['tp']:,.2f}", row=1, col=1)
    fig.add_hline(y=info['sl'], line_color="#ff4444", line_dash="dash",
                  annotation_text=f"SL {info['sl']:,.2f}", row=1, col=1)

    if tiene_vol:
        rsi_row  = 3
        macd_row = 4
        colors_vol = []
        for i in range(len(df)):
            is_anom = bool(df['VOL_ANOMALO'].iloc[i]) if 'VOL_ANOMALO' in df.columns else False
            up = float(df['Close'].iloc[i]) >= float(df['Open'].iloc[i])
            if is_anom:
                colors_vol.append("rgba(255,255,255,0.9)" if up else "rgba(255,170,0,0.9)")
            else:
                colors_vol.append("rgba(0,255,136,0.4)" if up else "rgba(255,68,68,0.4)")
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name="Volumen",
            marker_color=colors_vol,
        ), row=2, col=1)
    else:
        rsi_row  = 2
        macd_row = 3

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI",
        line=dict(color="#c77dff", width=1.5)), row=rsi_row, col=1)
    fig.add_hline(y=70, line_color="rgba(255,68,68,.5)",  line_dash="dot", row=rsi_row, col=1)
    fig.add_hline(y=30, line_color="rgba(0,255,136,.5)", line_dash="dot", row=rsi_row, col=1)

    macd_cols = ["#00ff88" if v >= 0 else "#ff4444" for v in df['MACD_HIST'].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name="Hist MACD",
        marker_color=macd_cols), row=macd_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD",
        line=dict(color="#00cfff", width=1.2)), row=macd_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name="Señal",
        line=dict(color="#ff8c00", width=1.2)), row=macd_row, col=1)

    # ── Fibonacci ──────────────────────────────────────────
    if mostrar_fib:
        fib_niveles = calcular_fibonacci(df)
        for nombre, nivel in fib_niveles.items():
            color = FIBONACCI_COLORES.get(nombre, "rgba(255,255,255,0.4)")
            fig.add_hline(
                y=nivel, line_color=color, line_dash="dot", line_width=1,
                annotation_text=f"Fib {nombre} ${nivel:,.2f}",
                annotation_font_color=color,
                annotation_position="top left",
                row=1, col=1,
            )

    # ── Divergencias RSI ───────────────────────────────────
    if mostrar_divs:
        divergencias = detectar_divergencias(df)
        for div in divergencias:
            color_div = "#00ff88" if div["tipo"] == "alcista" else "#ff4444"
            # Línea en precio
            fig.add_shape(type="line",
                x0=div["fecha1"], x1=div["fecha2"],
                y0=div["precio1"], y1=div["precio2"],
                line=dict(color=color_div, width=2, dash="dot"),
                row=1, col=1)
            # Línea en RSI
            fig.add_shape(type="line",
                x0=div["fecha1"], x1=div["fecha2"],
                y0=div["rsi1"], y1=div["rsi2"],
                line=dict(color=color_div, width=2, dash="dot"),
                row=rsi_row, col=1)
            # Anotación
            emoji = "🔺" if div["tipo"] == "alcista" else "🔻"
            fig.add_annotation(
                x=div["fecha2"], y=div["precio2"],
                text=f"{emoji} Div {div['tipo'][:3].upper()}",
                font=dict(color=color_div, size=10),
                showarrow=True, arrowhead=2,
                arrowcolor=color_div, arrowsize=0.8,
                row=1, col=1)

    fig.update_layout(
        template="plotly_dark", height=880,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.01, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    if mostrar_vanom and mostrar_volumen:
        st.caption("⬜ blanco / 🟠 naranja en volumen = anomalía (>2× media 20 períodos) — posible punto de inflexión")

    with st.expander("📊 Datos recientes"):
        cols_show = [c for c in ['Close','EMA20','EMA50','RSI','MACD','BB_UPPER','BB_LOWER','ATR']
                     if c in df.columns]
        st.dataframe(
            df[cols_show].tail(30).sort_index(ascending=False).style.format("{:.4f}"),
            use_container_width=True,
        )
        csv = df[cols_show].to_csv().encode()
        st.download_button("⬇️ Descargar CSV", csv, f"{ticker}_indicadores.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEÑALES EN PARALELO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌐 Señales del Mercado — Vista en Paralelo")
    st.caption(f"Período: {period} · Caché 5 min — ordenado por puntuación")

    todos = {**ACCIONES, **CRYPTOS}
    filas = []

    def _fetch_fila(nm_sym):
        nm, sym = nm_sym
        try:
            d = get_data(sym, period)
            if d.empty or len(d) < 30:
                return None
            inf = generar_senal(d)
            p   = float(d['Close'].iloc[-1])
            pv  = float(d['Close'].iloc[-2]) if len(d) > 1 else p
            chg_row = ((p / pv) - 1) * 100 if pv > 0 else 0
            adx_val = round(float(d['ADX'].iloc[-1]), 1) if 'ADX' in d.columns else 0
            return {
                "Activo":   nm,
                "Ticker":   sym,
                "Precio":   p,
                "Cambio %": round(chg_row, 2),
                "RSI":      round(float(d['RSI'].iloc[-1]), 1),
                "ADX":      adx_val,
                "ATR":      round(float(d['ATR'].iloc[-1]), 4),
                "Señal":    inf['senal'],
                "Puntos":   inf['puntos'],
            }
        except Exception:
            return None

    prog = st.progress(0, text="Cargando señales en paralelo...")
    completados = 0
    total_sym   = len(todos)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futuros = {executor.submit(_fetch_fila, item): item for item in todos.items()}
        for fut in as_completed(futuros):
            completados += 1
            prog.progress(completados / total_sym, text=f"Cargando... {completados}/{total_sym}")
            res = fut.result()
            if res:
                filas.append(res)
    prog.empty()

    if filas:
        df_radar = pd.DataFrame(filas).sort_values("Puntos", ascending=False)

        def color_senal(val):
            if "COMPRA FUERTE" in str(val): return "color:#00ff88;font-weight:bold"
            if "COMPRA DÉBIL"  in str(val): return "color:#ffd700"
            if "VENTA FUERTE"  in str(val): return "color:#ff4444;font-weight:bold"
            if "VENTA DÉBIL"   in str(val): return "color:#ff8c00"
            return "color:#aaaaaa"

        def color_num(val):
            try:
                return "color:#00ff88" if float(val) > 0 else ("color:#ff4444" if float(val) < 0 else "")
            except Exception:
                return ""

        _cell_map = "map" if hasattr(df_radar.style, "map") else "applymap"
        styled = df_radar.style.format({
            "Precio": "{:,.4f}", "Cambio %": "{:+.2f}%",
            "RSI": "{:.1f}", "ADX": "{:.1f}", "ATR": "{:.4f}",
        })
        styled = getattr(styled, _cell_map)(color_senal, subset=["Señal"])
        styled = getattr(styled, _cell_map)(color_num,   subset=["Cambio %", "Puntos"])
        st.dataframe(styled, use_container_width=True, height=620)

        fig_bar = px.bar(
            df_radar, x="Activo", y="Puntos", color="Puntos",
            color_continuous_scale=["#ff4444","#555555","#00ff88"],
            title="Puntuación de Confluencia por Activo",
            template="plotly_dark",
        )
        fig_bar.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                               coloraxis_showscale=False, height=380,
                               xaxis_tickangle=-35)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Mapa de Calor (Treemap estilo Finviz) ──────────────────
        st.markdown("---")
        st.markdown("### 🗺️ Mapa de Calor — Cambio % del Día")

        col_hm1, col_hm2 = st.columns([3, 1])
        with col_hm2:
            hm_tipo = st.radio("Vista", ["Acciones + Crypto", "Solo Acciones", "Solo Crypto"],
                               key="hm_tipo")
            hm_color = st.radio("Color por", ["Cambio % día", "Puntuación señal"],
                                key="hm_color")

        if hm_tipo == "Solo Acciones":
            df_hm = df_radar[~df_radar["Ticker"].str.contains("-USD")]
        elif hm_tipo == "Solo Crypto":
            df_hm = df_radar[df_radar["Ticker"].str.contains("-USD")]
        else:
            df_hm = df_radar.copy()

        if not df_hm.empty:
            df_hm = df_hm.copy()
            df_hm["Sector"] = df_hm["Ticker"].apply(
                lambda t: "Crypto" if "-USD" in t else "Acciones"
            )
            df_hm["Label"] = df_hm.apply(
                lambda r: f"{r['Ticker'].replace('-USD','')}<br>{r['Cambio %']:+.2f}%",
                axis=1
            )

            valor_color = "Cambio %" if hm_color == "Cambio % día" else "Puntos"
            escala = (
                [[0,"#ff2222"],[0.35,"#aa2222"],[0.48,"#333333"],
                 [0.52,"#333333"],[0.65,"#22aa44"],[1,"#00ff88"]]
                if hm_color == "Cambio % día"
                else
                [[0,"#ff4444"],[0.4,"#555555"],[0.5,"#555555"],[1,"#00ff88"]]
            )

            fig_hm = go.Figure(go.Treemap(
                ids=df_hm["Ticker"],
                labels=df_hm["Label"],
                parents=df_hm["Sector"],
                values=[1] * len(df_hm),
                customdata=df_hm[["Precio","Cambio %","Señal","RSI","ADX"]].values,
                hovertemplate=(
                    "<b>%{id}</b><br>"
                    "Precio: $%{customdata[0]:,.4f}<br>"
                    "Cambio: %{customdata[1]:+.2f}%<br>"
                    "Señal: %{customdata[2]}<br>"
                    "RSI: %{customdata[3]:.1f} | ADX: %{customdata[4]:.1f}"
                    "<extra></extra>"
                ),
                marker=dict(
                    colors=df_hm[valor_color],
                    colorscale=escala,
                    cmid=0,
                    line=dict(width=2, color="#0d1117"),
                    pad=dict(t=20, l=3, r=3, b=3),
                ),
                textfont=dict(size=12, color="white"),
                pathbar=dict(visible=True),
                tiling=dict(packing="squarify", pad=4),
            ))

            # Agregar fila raíz vacía para que el treemap agrupe por sector
            fig_hm.add_trace(go.Treemap(
                ids=["Acciones", "Crypto"],
                labels=["Acciones", "Crypto"],
                parents=["", ""],
                values=[1, 1],
                visible=False,
            ))

            fig_hm.update_layout(
                template="plotly_dark",
                height=520,
                paper_bgcolor="#0d1117",
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(color="white"),
            )
            with col_hm1:
                st.plotly_chart(fig_hm, use_container_width=True)
            st.caption(
                "🟢 Verde = subida / señal alcista  ·  🔴 Rojo = bajada / señal bajista  ·  "
                "Hover para ver detalles  ·  Tamaño igual para todos los activos"
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARACIÓN RELATIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"📉 Rendimiento relativo: {ticker} vs benchmark")

    _, col_bench = st.columns([3, 1])
    with col_bench:
        bench_input = st.text_input("Benchmark", value=benchmark).upper().strip()
    bench_usado = bench_input if bench_input else benchmark

    with st.spinner("Cargando comparación..."):
        s_activo = get_close_only(ticker,      period)
        s_bench  = get_close_only(bench_usado, period)

    if s_activo.empty or s_bench.empty:
        st.warning("No se pudieron obtener datos para la comparación.")
    else:
        df_comp = pd.DataFrame({ticker: s_activo, bench_usado: s_bench}).dropna()
        df_norm = df_comp / df_comp.iloc[0] * 100

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=df_norm.index, y=df_norm[ticker],
            name=ticker, line=dict(color="#00cfff", width=2.2)))
        fig_comp.add_trace(go.Scatter(
            x=df_norm.index, y=df_norm[bench_usado],
            name=bench_usado, line=dict(color="#ffd700", width=2, dash="dash")))
        fig_comp.add_hline(y=100, line_color="rgba(255,255,255,0.15)", line_dash="dot")

        diff = df_norm[ticker] - df_norm[bench_usado]
        color_area = "rgba(0,255,136,0.1)" if float(diff.iloc[-1]) >= 0 else "rgba(255,68,68,0.1)"
        base = df_norm[bench_usado]
        fig_comp.add_trace(go.Scatter(
            x=df_norm.index, y=base + diff,
            fill='tonexty', fillcolor=color_area,
            line=dict(width=0), name="Diferencia",
        ))
        fig_comp.update_layout(
            template="plotly_dark", height=460,
            title=f"Rendimiento normalizado (base 100) — {period}",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        ret_a = float(df_norm[ticker].iloc[-1]     - 100)
        ret_b = float(df_norm[bench_usado].iloc[-1] - 100)
        alpha = ret_a - ret_b
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Retorno {ticker}",      f"{ret_a:+.2f}%")
        m2.metric(f"Retorno {bench_usado}", f"{ret_b:+.2f}%")
        m3.metric("Alpha",                  f"{alpha:+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔗 Matriz de Correlación")

    col_tipo_corr = st.radio(
        "Conjunto", ["📈 Acciones", "₿ Criptomonedas"],
        horizontal=True, key="corr_tipo")
    preset_corr = ACCIONES if "Acciones" in col_tipo_corr else CRYPTOS

    with st.spinner("Calculando correlaciones..."):
        series_list = []
        for nm, sym in preset_corr.items():
            s = get_close_only(sym, period)
            if not s.empty:
                s.name = nm.split("(")[0].strip()
                series_list.append(s)

    if len(series_list) < 2:
        st.warning("No hay suficientes datos.")
    else:
        df_corr = pd.concat(series_list, axis=1).dropna()
        corr    = df_corr.pct_change().dropna().corr()
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1, text_auto=".2f",
            title=f"Correlación de retornos diarios — {period}",
            template="plotly_dark",
        )
        fig_corr.update_layout(paper_bgcolor="#0d1117", height=540,
                                coloraxis_colorbar=dict(title="ρ"))
        fig_corr.update_traces(textfont_size=11)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("🟢 Alta correlación positiva · 🔴 Correlación negativa · Útil para diversificación de portafolio")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PAPER TRADING & NOTAS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:

    col_pt, col_notas = st.columns([3, 2])

    with col_pt:
        st.subheader("📒 Paper Trading")
        st.caption("Operaciones simuladas — datos guardados en esta sesión")

        with st.form("nueva_operacion", clear_on_submit=True):
            st.markdown(f"**Activo:** `{ticker}` · Precio actual: `${price:,.4f}`")
            c1, c2, c3 = st.columns(3)
            entrada_pt = c1.number_input("Entrada ($)",     value=round(price, 4),     format="%.4f")
            sl_pt      = c2.number_input("Stop Loss ($)",   value=round(info['sl'],4),  format="%.4f")
            tp_pt      = c3.number_input("Take Profit ($)", value=round(info['tp'],4),  format="%.4f")
            c4, c5     = st.columns(2)
            capital_pt = c4.number_input("Capital ($)", value=1000.0, min_value=1.0, format="%.2f")
            lado_pt    = c5.selectbox("Dirección", ["LONG", "SHORT"])
            nota_op    = st.text_input("Nota (opcional)")
            submitted  = st.form_submit_button("➕ Registrar operación")

        if submitted:
            riesgo_op = abs(entrada_pt - sl_pt)
            reward    = abs(tp_pt - entrada_pt)
            rr        = round(reward / riesgo_op, 2) if riesgo_op > 0 else 0
            tamanio   = round(capital_pt * 0.02 / riesgo_op, 4) if riesgo_op > 0 else 0
            st.session_state.paper_trades.append({
                "Fecha":   datetime.now().strftime("%d/%m/%Y %H:%M"),
                "Ticker":  ticker,
                "Lado":    lado_pt,
                "Entrada": entrada_pt,
                "SL":      sl_pt,
                "TP":      tp_pt,
                "R/R":     rr,
                "Tamaño":  tamanio,
                "Capital": capital_pt,
                "Estado":  "🟡 Abierta",
                "Nota":    nota_op,
            })
            st.success(f"✅ Registrada · R/R: **{rr}** · Tamaño sugerido (2% riesgo): **{tamanio}** unidades")

        trades = st.session_state.paper_trades
        if trades:
            def evaluar_estado(row):
                try:
                    if row['Ticker'] != ticker:
                        return row['Estado']
                    if row['Lado'] == "LONG":
                        if price <= row['SL']: return "🔴 SL tocado"
                        if price >= row['TP']: return "🟢 TP alcanzado"
                    else:
                        if price >= row['SL']: return "🔴 SL tocado"
                        if price <= row['TP']: return "🟢 TP alcanzado"
                except Exception:
                    pass
                return "🟡 Abierta"

            df_trades = pd.DataFrame(trades)
            df_trades['Estado'] = df_trades.apply(evaluar_estado, axis=1)

            # Calcular P&L estimado por operación
            def calc_pnl(row):
                try:
                    precio_actual_op = price if row['Ticker'] == ticker else row['Entrada']
                    if row['Estado'] == "🟢 TP alcanzado":
                        precio_cierre = row['TP']
                    elif row['Estado'] == "🔴 SL tocado":
                        precio_cierre = row['SL']
                    else:
                        precio_cierre = precio_actual_op
                    if row['Lado'] == "LONG":
                        pct = (precio_cierre - row['Entrada']) / row['Entrada'] * 100
                    else:
                        pct = (row['Entrada'] - precio_cierre) / row['Entrada'] * 100
                    pnl_usd = row['Capital'] * pct / 100
                    return round(pnl_usd, 2), round(pct, 2)
                except Exception:
                    return 0.0, 0.0

            pnl_data = df_trades.apply(lambda r: calc_pnl(r), axis=1)
            df_trades['P&L $']  = [x[0] for x in pnl_data]
            df_trades['P&L %']  = [x[1] for x in pnl_data]

            st.dataframe(df_trades, use_container_width=True, height=280)

            total    = len(df_trades)
            ganadas  = (df_trades['Estado'] == "🟢 TP alcanzado").sum()
            perdidas = (df_trades['Estado'] == "🔴 SL tocado").sum()
            winrate  = round(ganadas / (ganadas + perdidas) * 100, 1) if (ganadas + perdidas) > 0 else 0
            pnl_total = df_trades['P&L $'].sum()
            pnl_color = "normal" if pnl_total >= 0 else "inverse"

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Total",          total)
            s2.metric("✅ TP alcanzado", ganadas)
            s3.metric("❌ SL tocado",    perdidas)
            s4.metric("Win Rate",        f"{winrate}%")
            s5.metric("P&L Total",       f"${pnl_total:+,.2f}", delta_color=pnl_color)

            if st.button("🗑️ Limpiar operaciones"):
                st.session_state.paper_trades = []
                st.rerun()
        else:
            st.info("Aún no hay operaciones. Registra una con el formulario de arriba.")

    with col_notas:
        st.subheader("✏️ Notas del Trader")
        st.caption(f"Nota activa: **{ticker}**")

        nota_actual = st.session_state.notas.get(ticker, "")
        nueva_nota  = st.text_area(
            "Tesis, niveles clave, catalizadores...",
            value=nota_actual, height=220, key=f"nota_{ticker}",
        )
        if st.button("💾 Guardar nota"):
            st.session_state.notas[ticker] = nueva_nota
            st.success("Nota guardada ✓")

        notas_otras = {k: v for k, v in st.session_state.notas.items()
                       if v.strip() and k != ticker}
        if notas_otras:
            st.markdown("---")
            st.markdown("**Otras notas guardadas:**")
            for sym_n, texto_n in notas_otras.items():
                with st.expander(f"📌 {sym_n}"):
                    st.write(texto_n)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ASISTENTE IA
# ══════════════════════════════════════════════════════════════════════════════
with tab6:

    ia_activa = _ia_activa()

    if not ia_activa:
        st.warning("Ingresa tu Groq API key en el panel izquierdo para activar las funciones de IA.")
        st.info("Obtén una **gratis** (sin tarjeta) en [console.groq.com](https://console.groq.com)")
        st.stop()

    resumen_tecnico = _resumen_tecnico(ticker, df, info)

    st.subheader(f"🤖 Análisis narrativo — {ticker}")
    cache_key_analisis = f"{ticker}_{period}"
    col_an1, _ = st.columns([1, 5])
    with col_an1:
        regenerar = st.button("🔄 Generar / Actualizar análisis")

    if regenerar or cache_key_analisis not in st.session_state.analisis_cache:
        with st.spinner("Analizando los indicadores..."):
            sistema_analisis = """Eres un analista técnico senior especializado en trading de acciones y criptomonedas.
Tu tarea es leer un resumen de indicadores técnicos y generar un análisis narrativo claro, 
conciso y profesional en español. Estructura tu respuesta así:
1. **Contexto general** (1-2 oraciones sobre la tendencia)
2. **Señales clave** (bullet points de los indicadores más relevantes)
3. **Escenario alcista** (qué necesita pasar para confirmar subida)
4. **Escenario bajista** (qué niveles vigilar para baja)
5. **Conclusión** (1 oración directa)
Sé directo, evita repetir los números del resumen textualmente."""
            analisis_texto = _groq(
                system=sistema_analisis,
                user=f"Analiza este activo:\n\n{resumen_tecnico}",
                max_tokens=700,
            )
            st.session_state.analisis_cache[cache_key_analisis] = analisis_texto

    st.markdown(st.session_state.analisis_cache.get(cache_key_analisis, ""))
    st.markdown("---")

    st.subheader(f"📰 Noticias recientes + Sentimiento — {ticker}")
    col_n1, _ = st.columns([1, 5])
    with col_n1:
        buscar_noticias = st.button("🔍 Buscar noticias y analizar")

    if buscar_noticias or ticker not in st.session_state.noticias_cache:
        with st.spinner("Buscando noticias y analizando sentimiento..."):
            sistema_noticias = """Eres un analista financiero experto en análisis de sentimiento de mercado.
Cuando el usuario te pida analizar noticias de un activo financiero, debes:
1. Resumir el contexto de mercado actual de ese activo (basado en tu conocimiento)
2. Identificar los principales catalizadores positivos y negativos recientes
3. Dar una puntuación de sentimiento del -10 (muy bajista) al +10 (muy alcista) con justificación
4. Mencionar eventos próximos relevantes (earnings, halvings, regulaciones, etc.) si los conoces
Formato: usa emojis, sé visual y directo. Responde en español."""
            noticias_texto = _groq(
                system=sistema_noticias,
                user=f"Analiza el sentimiento actual de mercado para {ticker}. Datos técnicos de contexto:\n{resumen_tecnico}",
                max_tokens=700,
            )
            st.session_state.noticias_cache[ticker] = noticias_texto

    if ticker in st.session_state.noticias_cache:
        st.markdown(st.session_state.noticias_cache[ticker])

    st.markdown("---")

    st.subheader(f"📝 Generador de tesis de inversión — {ticker}")
    col_t1, col_t2, _ = st.columns([1, 1, 4])
    with col_t1:
        horizonte = st.selectbox("Horizonte", ["Corto plazo (días)", "Mediano plazo (semanas)", "Largo plazo (meses)"])
    with col_t2:
        perfil    = st.selectbox("Perfil de riesgo", ["Conservador", "Moderado", "Agresivo"])

    if st.button("✍️ Generar tesis de inversión"):
        with st.spinner("Generando tesis..."):
            sistema_tesis = """Eres un gestor de portafolio profesional. Genera una tesis de inversión 
estructurada, práctica y en español. Usa este formato exacto:

## 🎯 Tesis: [COMPRA/VENTA/NEUTRAL] [Ticker]

**Resumen ejecutivo** (2-3 oraciones)

**📈 Argumentos a favor**
- [bullet]

**📉 Riesgos principales**  
- [bullet]

**🎯 Niveles clave**
- Entrada ideal: $X
- Stop Loss: $X  
- Target 1: $X | Target 2: $X
- R/R: X:1

**📋 Plan de acción**
[1-2 oraciones sobre cuándo y cómo entrar]

Sé específico con los precios. Adapta el tono al perfil de riesgo indicado."""
            tesis_texto = _groq(
                system=sistema_tesis,
                user=f"Genera una tesis de inversión para {ticker}.\nHorizonte: {horizonte}\nPerfil: {perfil}\nDatos técnicos:\n{resumen_tecnico}",
                max_tokens=800,
            )
        st.markdown(tesis_texto)
        if st.button("💾 Guardar tesis como nota del trader"):
            st.session_state.notas[ticker] = tesis_texto
            st.success(f"Tesis guardada en notas de {ticker} ✓")

    st.markdown("---")

    st.subheader(f"💬 Chat con el asistente — {ticker}")
    st.caption("Pregunta lo que quieras sobre este activo, su situación técnica o el mercado en general.")

    if ticker not in st.session_state.chat_history:
        st.session_state.chat_history[ticker] = []

    historial = st.session_state.chat_history[ticker]

    for msg in historial:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt_chat = st.chat_input(f"Pregunta sobre {ticker}...")

    if prompt_chat:
        historial.append({"role": "user", "content": prompt_chat})
        with st.chat_message("user"):
            st.markdown(prompt_chat)

        sistema_chat = f"""Eres un asistente experto en trading y análisis financiero que trabaja dentro de 
QuantumShield Pro, una app de análisis técnico. Tienes acceso al contexto técnico actual del activo {ticker}.

DATOS TÉCNICOS ACTUALES:
{resumen_tecnico}

Responde siempre en español, de forma clara y concisa. Si te preguntan sobre otros activos o temas 
generales de trading, responde con tu conocimiento general. No des consejos financieros definitivos, 
enfatiza que es análisis técnico educativo."""

        mensajes_api = historial[-10:]
        with st.chat_message("assistant"):
            respuesta_placeholder = st.empty()
            respuesta_completa    = ""
            for chunk in _groq_stream(sistema_chat, mensajes_api, max_tokens=600):
                respuesta_completa += chunk
                respuesta_placeholder.markdown(respuesta_completa + "▌")
            respuesta_placeholder.markdown(respuesta_completa)

        historial.append({"role": "assistant", "content": respuesta_completa})
        if len(historial) > 40:
            st.session_state.chat_history[ticker] = historial[-40:]

    if historial and st.button("🗑️ Limpiar chat", key="clear_chat"):
        st.session_state.chat_history[ticker] = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader(f"🧪 Backtesting — {ticker}")
    st.caption("Simulación de la estrategia de confluencia sobre el historial de precio. No garantiza resultados futuros.")

    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
    with col_bt1:
        bt_umbral_compra = st.slider("Umbral de compra (puntos mínimos)", 1, 4, 2)
    with col_bt2:
        bt_umbral_venta  = st.slider("Umbral de salida (puntos máximos)", -4, -1, -1)
    with col_bt3:
        bt_period_sel = st.selectbox("Período de backtest", ["6mo","1y","2y"], index=1, key="bt_period")
    with col_bt4:
        bt_comision = st.number_input(
            "Comisión por operación (%)",
            min_value=0.0, max_value=2.0, value=0.1, step=0.05,
            help="Binance: 0.1% · Broker acciones: 0.05-0.2% · Por los dos lados = entrada+salida",
            format="%.2f",
        )

    if st.button("▶️ Ejecutar Backtest", type="primary"):
        with st.spinner("Simulando estrategia..."):
            df_bt = get_data(ticker, bt_period_sel)
            if df_bt.empty or len(df_bt) < 60:
                st.warning("No hay suficientes datos para el backtest (mínimo 60 velas).")
            else:
                bt = backtest_estrategia(df_bt, bt_umbral_compra, bt_umbral_venta, bt_comision)

                if bt['n_ops'] == 0:
                    st.info("La estrategia no generó ninguna operación. Prueba ajustando los umbrales.")
                else:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Operaciones",       bt['n_ops'])
                    b2.metric("Win Rate",           f"{bt['win_rate']}%")
                    b3.metric("Retorno estrategia", f"{bt['retorno_total']:+.2f}%",
                              delta=f"{bt['retorno_total'] - bt['retorno_bh']:+.2f}% vs B&H")
                    b4.metric("Buy & Hold",         f"{bt['retorno_bh']:+.2f}%")

                    b5, b6, b7, b8 = st.columns(4)
                    b5.metric("Mejor operación",  f"{bt['max_ganancia']:+.2f}%")
                    b6.metric("Peor operación",   f"{bt['max_perdida']:+.2f}%")
                    sharpe_color = "normal" if bt['sharpe'] >= 1 else "inverse"
                    b7.metric("Sharpe Ratio",     f"{bt['sharpe']:.2f}",
                              delta="Bueno ✅" if bt['sharpe'] >= 1 else ("Aceptable ⚠️" if bt['sharpe'] >= 0.5 else "Bajo ❌"),
                              delta_color=sharpe_color)
                    b8.metric("Max Drawdown",     f"{bt['max_drawdown']:+.2f}%",
                              delta_color="inverse")

                    st.markdown("---")
                    df_ops = bt['ops'].copy()
                    df_ops['Retorno acumulado %'] = df_ops['Retorno %'].cumsum()
                    df_ops['Op'] = range(1, len(df_ops) + 1)

                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=df_ops['Op'], y=df_ops['Retorno acumulado %'],
                        mode='lines+markers', name='Retorno acumulado',
                        line=dict(color="#00cfff", width=2),
                        fill='tozeroy', fillcolor='rgba(0,207,255,0.08)',
                    ))
                    fig_eq.add_hline(y=bt['retorno_bh'], line_color="#ffd700", line_dash="dash",
                                     annotation_text=f"Buy & Hold {bt['retorno_bh']:+.1f}%",
                                     annotation_font_color="#ffd700")
                    fig_eq.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
                    # Banda de drawdown máximo
                    eq_vals   = df_ops['Retorno acumulado %'].values
                    peak_vals = np.maximum.accumulate(eq_vals)
                    dd_vals   = eq_vals - peak_vals
                    fig_eq.add_trace(go.Scatter(
                        x=df_ops['Op'], y=dd_vals,
                        name="Drawdown", fill='tozeroy',
                        fillcolor='rgba(255,68,68,0.12)',
                        line=dict(color='rgba(255,68,68,0.5)', width=1),
                    ))
                    fig_eq.update_layout(
                        template="plotly_dark", height=380,
                        title="Curva de equity + Drawdown",
                        xaxis_title="Número de operación",
                        yaxis_title="Retorno acumulado %",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

                    col_h1, col_h2 = st.columns(2)
                    with col_h1:
                        fig_hist = go.Figure()
                        colors_hist = ["#00ff88" if v > 0 else "#ff4444" for v in df_ops['Retorno %']]
                        fig_hist.add_trace(go.Bar(
                            x=df_ops['Op'], y=df_ops['Retorno %'],
                            marker_color=colors_hist, name="Retorno por op."
                        ))
                        fig_hist.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
                        fig_hist.update_layout(
                            template="plotly_dark", height=280,
                            title="Retorno por operación",
                            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    with col_h2:
                        gan = (df_ops['Retorno %'] > 0).sum()
                        per = (df_ops['Retorno %'] <= 0).sum()
                        fig_pie = go.Figure(go.Pie(
                            labels=["Ganadas", "Perdidas"],
                            values=[gan, per],
                            marker_colors=["#00ff88", "#ff4444"],
                            hole=0.55,
                        ))
                        fig_pie.update_layout(
                            template="plotly_dark", height=280,
                            title=f"Win Rate: {bt['win_rate']}%",
                            paper_bgcolor="#0d1117",
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with st.expander("📋 Tabla de operaciones"):
                        def color_retorno(val):
                            try:
                                return "color:#00ff88" if float(val) > 0 else "color:#ff4444"
                            except Exception:
                                return ""
                        _cm = "map" if hasattr(df_ops.style, "map") else "applymap"
                        styled_bt = df_ops.drop(columns=['Op']).style.format({"Retorno %": "{:+.2f}%"})
                        styled_bt = getattr(styled_bt, _cm)(color_retorno, subset=["Retorno %"])
                        st.dataframe(styled_bt, use_container_width=True)

                        csv_bt = df_ops.to_csv(index=False).encode()
                        st.download_button("⬇️ Descargar CSV", csv_bt,
                                           f"{ticker}_backtest.csv", "text/csv")

                    costo_total = round(bt_comision * 2 * bt['n_ops'], 2)
                    st.caption(
                        f"⚠️ Resultados con comisión de **{bt_comision}% por lado** ({bt_comision*2}% por operación completa) · "
                        f"Costo total: **{costo_total:.2f}%** · "
                        "No garantiza rendimientos futuros. Ejecución asumida al cierre del día siguiente."
                    )
    else:
        st.info("Configura los parámetros y haz clic en **▶️ Ejecutar Backtest** para simular la estrategia.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — MULTI-TIMEFRAME
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.subheader(f"⏱️ Análisis Multi-Timeframe — {ticker}")
    st.caption("Compara la señal en 3 marcos temporales. Una señal alineada en los 3 es la más confiable.")

    TFS = ["4H", "1D", "1W"]
    TF_LABELS = {"4H": "4 Horas", "1D": "Diario", "1W": "Semanal"}

    with st.spinner("Cargando 3 timeframes..."):
        mtf_data = {}
        mtf_info = {}
        for tf in TFS:
            d = get_data_mtf(ticker, tf)
            if not d.empty and len(d) >= 30:
                mtf_data[tf] = d
                mtf_info[tf] = generar_senal(d)

    if not mtf_data:
        st.warning("No se pudieron obtener datos multi-timeframe para este activo.")
    else:
        # ── Semáforo de alineación ──────────────────────────────────
        st.markdown("### 🚦 Alineación de Señales")
        cols_tf = st.columns(len(TFS))
        puntos_total = 0
        tfs_disponibles = 0

        for i, tf in enumerate(TFS):
            with cols_tf[i]:
                if tf in mtf_info:
                    inf = mtf_info[tf]
                    puntos_total += inf['puntos']
                    tfs_disponibles += 1
                    st.markdown(f"""
                    <div style="background:#1a1a2e;border:2px solid {inf['color']};
                        border-radius:12px;padding:16px;text-align:center;
                        box-shadow:0 0 15px {inf['color']}44;">
                      <div style="font-size:1.1rem;color:#aaa;margin-bottom:4px">{TF_LABELS[tf]}</div>
                      <div style="font-size:1.3rem;font-weight:bold;color:{inf['color']}">{inf['senal']}</div>
                      <div style="color:#888;font-size:.85rem;margin-top:6px">
                        RSI: {inf['rsi']:.1f} · ADX: {inf['adx']:.1f} · Pts: {inf['puntos']:+d}
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#1a1a2e;border:2px solid #333;
                        border-radius:12px;padding:16px;text-align:center;">
                      <div style="color:#666">{TF_LABELS[tf]}</div>
                      <div style="color:#555">Sin datos</div>
                    </div>""", unsafe_allow_html=True)

        # Alineación general
        st.markdown("---")
        if tfs_disponibles > 0:
            promedio = puntos_total / tfs_disponibles
            if promedio >= 2:
                alin_txt, alin_col = "🟢 ALCISTA — Señal alineada en múltiples timeframes", "#00ff88"
            elif promedio <= -2:
                alin_txt, alin_col = "🔴 BAJISTA — Señal alineada en múltiples timeframes", "#ff4444"
            elif promedio >= 1:
                alin_txt, alin_col = "🟡 TENDENCIA ALCISTA — No confirmada en todos los marcos", "#ffd700"
            elif promedio <= -1:
                alin_txt, alin_col = "🟠 TENDENCIA BAJISTA — No confirmada en todos los marcos", "#ff8c00"
            else:
                alin_txt, alin_col = "⚪ MIXTO — Sin consenso entre timeframes", "#aaaaaa"

            st.markdown(f"""
            <div style="background:#0d1117;border-left:4px solid {alin_col};
                padding:14px 20px;border-radius:0 8px 8px 0;margin:8px 0;">
              <span style="color:{alin_col};font-size:1.1rem;font-weight:bold">{alin_txt}</span>
              <span style="color:#888;font-size:.85rem;margin-left:12px">
                Puntuación promedio: {promedio:+.1f} / {tfs_disponibles} timeframes
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Gráficos de los 3 timeframes ───────────────────────────
        st.markdown("### 📈 Gráficos por Timeframe")

        for tf in TFS:
            if tf not in mtf_data:
                continue
            d   = mtf_data[tf]
            inf = mtf_info[tf]

            with st.expander(f"📊 {TF_LABELS[tf]} — {inf['senal']} (pts: {inf['puntos']:+d})", expanded=(tf == "1D")):
                fig_tf = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.55, 0.22, 0.23],
                )
                # Velas
                fig_tf.add_trace(go.Candlestick(
                    x=d.index, open=d['Open'], high=d['High'],
                    low=d['Low'], close=d['Close'], name="Precio",
                    increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
                ), row=1, col=1)
                # EMAs
                fig_tf.add_trace(go.Scatter(x=d.index, y=d['EMA20'], name="EMA20",
                    line=dict(color="#ffd700", width=1.2)), row=1, col=1)
                fig_tf.add_trace(go.Scatter(x=d.index, y=d['EMA50'], name="EMA50",
                    line=dict(color="#ff8c00", width=1.5)), row=1, col=1)
                # BB
                fig_tf.add_trace(go.Scatter(x=d.index, y=d['BB_UPPER'], name="BB Sup",
                    line=dict(color="rgba(100,180,255,.5)", width=1, dash="dot")), row=1, col=1)
                fig_tf.add_trace(go.Scatter(x=d.index, y=d['BB_LOWER'], name="BB Inf",
                    line=dict(color="rgba(100,180,255,.5)", width=1, dash="dot"),
                    fill='tonexty', fillcolor="rgba(100,180,255,.04)"), row=1, col=1)
                # SL/TP
                fig_tf.add_hline(y=inf['tp'], line_color="#00ff88", line_dash="dash",
                                  annotation_text=f"TP {inf['tp']:,.2f}", row=1, col=1)
                fig_tf.add_hline(y=inf['sl'], line_color="#ff4444", line_dash="dash",
                                  annotation_text=f"SL {inf['sl']:,.2f}", row=1, col=1)
                # RSI
                fig_tf.add_trace(go.Scatter(x=d.index, y=d['RSI'], name="RSI",
                    line=dict(color="#c77dff", width=1.5)), row=2, col=1)
                fig_tf.add_hline(y=70, line_color="rgba(255,68,68,.5)", line_dash="dot", row=2, col=1)
                fig_tf.add_hline(y=30, line_color="rgba(0,255,136,.5)", line_dash="dot", row=2, col=1)
                # MACD
                macd_c = ["#00ff88" if v >= 0 else "#ff4444" for v in d['MACD_HIST'].fillna(0)]
                fig_tf.add_trace(go.Bar(x=d.index, y=d['MACD_HIST'], name="Hist",
                    marker_color=macd_c), row=3, col=1)
                fig_tf.add_trace(go.Scatter(x=d.index, y=d['MACD'], name="MACD",
                    line=dict(color="#00cfff", width=1.2)), row=3, col=1)
                fig_tf.add_trace(go.Scatter(x=d.index, y=d['MACD_SIGNAL'], name="Señal MACD",
                    line=dict(color="#ff8c00", width=1.2)), row=3, col=1)

                fig_tf.update_layout(
                    template="plotly_dark", height=600,
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    legend=dict(orientation="h", y=1.02),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=f"{ticker} — {TF_LABELS[tf]}",
                )
                fig_tf.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                fig_tf.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(fig_tf, use_container_width=True)

        st.caption("💡 Estrategia óptima: buscar señales donde 1D y 1W coinciden, y usar 4H para afinar la entrada.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — SCREENER AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.subheader("🔍 Screener Automático")
    st.caption("Escanea todos los activos y muestra solo los que tienen señal relevante ahora.")

    col_sc1, col_sc2, col_sc3 = st.columns(3)
    with col_sc1:
        filtro_senal = st.multiselect(
            "Filtrar por señal",
            ["🟢 COMPRA FUERTE", "🟡 COMPRA DÉBIL", "🔴 VENTA FUERTE",
             "🟠 VENTA DÉBIL", "⚪ NEUTRAL"],
            default=["🟢 COMPRA FUERTE", "🔴 VENTA FUERTE"],
        )
    with col_sc2:
        filtro_adx = st.slider("ADX mínimo (fuerza tendencia)", 0, 40, 20,
                               help="ADX > 25 = tendencia fuerte")
    with col_sc3:
        filtro_tipo = st.radio("Tipo de activo", ["Todos", "Solo Acciones", "Solo Crypto"],
                               horizontal=True)

    ejecutar_screener = st.button("▶️ Ejecutar Screener", type="primary")

    if ejecutar_screener:
        if filtro_tipo == "Solo Acciones":
            universo = ACCIONES
        elif filtro_tipo == "Solo Crypto":
            universo = CRYPTOS
        else:
            universo = {**ACCIONES, **CRYPTOS}

        resultados_sc = []
        prog_sc = st.progress(0, text="Escaneando mercado...")
        total_sc = len(universo)
        completados_sc = 0

        def _scan(nm_sym):
            nm, sym = nm_sym
            try:
                d = get_data(sym, "3mo")
                if d.empty or len(d) < 30:
                    return None
                inf = generar_senal(d)
                p   = float(d["Close"].iloc[-1])
                pv  = float(d["Close"].iloc[-2]) if len(d) > 1 else p
                chg_sc = ((p / pv) - 1) * 100 if pv > 0 else 0
                return {
                    "Activo":   nm,
                    "Ticker":   sym,
                    "Precio":   round(p, 4),
                    "Cambio %": round(chg_sc, 2),
                    "Señal":    inf["senal"],
                    "Puntos":   inf["puntos"],
                    "RSI":      round(inf["rsi"], 1),
                    "ADX":      round(inf["adx"], 1),
                    "SL":       round(inf["sl"], 4),
                    "TP":       round(inf["tp"], 4),
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as ex:
            futuros_sc = {ex.submit(_scan, item): item for item in universo.items()}
            for fut in as_completed(futuros_sc):
                completados_sc += 1
                prog_sc.progress(completados_sc / total_sc,
                                 text=f"Escaneando... {completados_sc}/{total_sc}")
                r = fut.result()
                if r:
                    resultados_sc.append(r)
        prog_sc.empty()

        if resultados_sc:
            df_sc = pd.DataFrame(resultados_sc)

            # Aplicar filtros
            if filtro_senal:
                df_sc = df_sc[df_sc["Señal"].isin(filtro_senal)]
            if filtro_adx > 0:
                df_sc = df_sc[df_sc["ADX"] >= filtro_adx]

            df_sc = df_sc.sort_values("Puntos", ascending=False)

            if df_sc.empty:
                st.info("Ningún activo cumple los filtros actuales. Prueba reduciendo ADX mínimo o ampliando los tipos de señal.")
            else:
                st.success(f"✅ {len(df_sc)} activos encontrados con los filtros aplicados")

                def _css_senal(val):
                    if "COMPRA FUERTE" in str(val): return "color:#00ff88;font-weight:bold"
                    if "COMPRA DÉBIL"  in str(val): return "color:#ffd700"
                    if "VENTA FUERTE"  in str(val): return "color:#ff4444;font-weight:bold"
                    if "VENTA DÉBIL"   in str(val): return "color:#ff8c00"
                    return "color:#aaa"

                def _css_num(val):
                    try:
                        return "color:#00ff88" if float(val) > 0 else ("color:#ff4444" if float(val) < 0 else "")
                    except Exception:
                        return ""

                _cm = "map" if hasattr(df_sc.style, "map") else "applymap"
                styled_sc = df_sc.style.format({
                    "Precio": "{:,.4f}", "Cambio %": "{:+.2f}%",
                    "RSI": "{:.1f}", "ADX": "{:.1f}",
                    "SL": "{:,.4f}", "TP": "{:,.4f}",
                })
                styled_sc = getattr(styled_sc, _cm)(_css_senal, subset=["Señal"])
                styled_sc = getattr(styled_sc, _cm)(_css_num, subset=["Cambio %", "Puntos"])
                st.dataframe(styled_sc, use_container_width=True, height=500)

                # Mini gráfico de barras
                fig_sc = px.bar(
                    df_sc, x="Activo", y="Puntos", color="Puntos",
                    color_continuous_scale=["#ff4444","#333","#00ff88"],
                    template="plotly_dark", height=300,
                    title="Puntuación de Confluencia — Resultados Screener",
                )
                fig_sc.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                     coloraxis_showscale=False, xaxis_tickangle=-30)
                st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("No se pudieron obtener datos para el screener.")
    else:
        st.info("Configura los filtros y haz clic en **▶️ Ejecutar Screener** para escanear el mercado.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — NOTICIAS & FUNDAMENTALES (Finviz + Historial de Señales)
# ══════════════════════════════════════════════════════════════════════════════
with tab10:
    st.subheader(f"📰 Noticias & Fundamentales — {ticker}")

    col_nf1, col_nf2 = st.columns([3, 2])

    with col_nf1:
        st.markdown("#### 📰 Noticias recientes (Finviz)")

        es_accion_usa = "-USD" not in ticker and not ticker.startswith("^")
        if not BS4_OK:
            st.warning("Instala `beautifulsoup4` en requirements.txt para activar esta función.", icon="⚠️")
        elif not es_accion_usa:
            st.info("Las noticias de Finviz solo están disponibles para acciones USA. Para crypto, usa el Asistente IA.")
        else:
            with st.spinner("Cargando noticias desde Finviz..."):
                fv = finviz_scrape(ticker)

            if fv["ok"] and fv["news"]:
                for noticia in fv["news"][:15]:
                    col_fecha, col_texto = st.columns([1, 4])
                    with col_fecha:
                        st.caption(noticia["fecha"])
                        if noticia["fuente"]:
                            st.caption(f"*{noticia['fuente']}*")
                    with col_texto:
                        st.markdown(f"[{noticia['titulo']}]({noticia['url']})")
                    st.divider()
            else:
                st.info("No se encontraron noticias para este ticker en Finviz. Puede deberse a un bloqueo temporal.")

        st.markdown("---")
        st.markdown("#### 📊 Historial de Señales — Esta Sesión")
        hist_ticker = st.session_state.signal_history.get(ticker, [])
        if hist_ticker:
            df_hist = pd.DataFrame(hist_ticker[::-1])   # más reciente primero

            def _css_hist(val):
                if "COMPRA FUERTE" in str(val): return "color:#00ff88;font-weight:bold"
                if "COMPRA DÉBIL"  in str(val): return "color:#ffd700"
                if "VENTA FUERTE"  in str(val): return "color:#ff4444;font-weight:bold"
                if "VENTA DÉBIL"   in str(val): return "color:#ff8c00"
                return "color:#aaa"

            _cm2 = "map" if hasattr(df_hist.style, "map") else "applymap"
            styled_hist = df_hist.style.format({"Precio": "{:,.4f}", "Puntos": "{:+d}"})
            styled_hist = getattr(styled_hist, _cm2)(_css_hist, subset=["Señal"])
            st.dataframe(styled_hist, use_container_width=True)

            if st.button("🗑️ Limpiar historial", key="clear_hist"):
                st.session_state.signal_history[ticker] = []
                st.rerun()
        else:
            st.info("El historial se llena automáticamente cada vez que la señal cambia. Navega entre activos para verlo aquí.")

    with col_nf2:
        st.markdown("#### 📈 Datos Fundamentales (Finviz)")

        if not BS4_OK:
            st.warning("Requiere `beautifulsoup4`", icon="⚠️")
        elif not es_accion_usa:
            st.info("Solo disponible para acciones USA.")
        else:
            if "fv" not in dir():
                with st.spinner("Cargando fundamentales..."):
                    fv = finviz_scrape(ticker)

            if fv["ok"] and fv["fundamentals"]:
                fund = fv["fundamentals"]
                # Campos más relevantes para trading
                CAMPOS = [
                    ("Market Cap",  "Market Cap"),
                    ("P/E",         "P/E"),
                    ("Fwd P/E",     "Forward P/E"),
                    ("EPS (ttm)",   "EPS (ttm)"),
                    ("EPS next Y",  "EPS next Y"),
                    ("ROE",         "ROE"),
                    ("Beta",        "Beta"),
                    ("52W High",    "52W High"),
                    ("52W Low",     "52W Low"),
                    ("Avg Volume",  "Avg Volume"),
                    ("Short Float", "Short Float"),
                    ("Analyst Rec.", "Recom"),
                    ("Target Price","Target Price"),
                    ("Earnings",    "Earnings"),
                ]
                for label, key in CAMPOS:
                    val = fund.get(key, fund.get(label, "-"))
                    if val and val != "-":
                        col_k, col_v = st.columns([2, 1])
                        col_k.caption(label)
                        col_v.markdown(f"**{val}**")
            else:
                st.info("Datos fundamentales no disponibles para este activo.")

        st.markdown("---")
        st.markdown("#### 📌 Niveles Fibonacci Actuales")
        fib = calcular_fibonacci(df)
        precio_actual_ref = price
        for nombre, nivel in fib.items():
            distancia = ((nivel - precio_actual_ref) / precio_actual_ref * 100)
            color_tag = "🟢" if nivel < precio_actual_ref else "🔴"
            st.write(f"{color_tag} **Fib {nombre}** — ${nivel:,.4f} ({distancia:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — PATRONES CHARTISTAS (Finviz)
# ══════════════════════════════════════════════════════════════════════════════
with tab11:
    st.subheader("🕯️ Patrones Chartistas — Detectados hoy por Finviz")
    st.caption(
        "Patrones técnicos detectados automáticamente por Finviz en el mercado de acciones USA. "
        "Actualización cada 15 minutos. Solo acciones USA — no aplica a crypto."
    )

    col_p_ctrl1, col_p_ctrl2, col_p_ctrl3 = st.columns([1, 1, 2])
    with col_p_ctrl1:
        filtro_bias = st.multiselect(
            "Filtrar por sesgo",
            ["alcista", "bajista", "neutro"],
            default=["alcista", "bajista"],
            key="filt_bias",
        )
    with col_p_ctrl2:
        buscar_patron = st.text_input("Buscar ticker", placeholder="AAPL", key="buscar_patron").upper().strip()
    with col_p_ctrl3:
        mostrar_guia = st.checkbox("Mostrar guía de patrones", value=False, key="mostrar_guia")

    if not BS4_OK:
        st.warning("Instala `beautifulsoup4` en requirements.txt para activar esta función.", icon="⚠️")
        st.stop()

    with st.spinner("Cargando patrones desde Finviz..."):
        patrones = finviz_patterns()

    if not patrones:
        st.info(
            "No se pudieron obtener patrones de Finviz en este momento. "
            "Puede deberse a un bloqueo temporal del servidor. Intenta en unos minutos.",
            icon="⏳",
        )
    else:
        # Construir DataFrame consolidado
        filas_p = []
        for patron, tickers_list in patrones.items():
            info_p = PATTERN_INFO.get(patron, {})
            for t in tickers_list:
                if buscar_patron and buscar_patron not in t:
                    continue
                filas_p.append({
                    "Patrón":  patron,
                    "Emoji":   info_p.get("emoji", "•"),
                    "Sesgo":   info_p.get("bias", "neutro"),
                    "Ticker":  t,
                })

        if not filas_p:
            st.info("No hay resultados con los filtros actuales.")
        else:
            df_p = pd.DataFrame(filas_p)
            if filtro_bias:
                df_p = df_p[df_p["Sesgo"].isin(filtro_bias)]

            # ── Resumen por patrón ─────────────────────────────────
            st.markdown("### 📋 Resumen por Patrón")

            patrones_filtrados = df_p["Patrón"].unique()
            cols_por_fila = 2
            patron_list   = list(patrones_filtrados)

            for row_i in range(0, len(patron_list), cols_por_fila):
                cols_p = st.columns(cols_por_fila)
                for col_i, patron in enumerate(patron_list[row_i:row_i+cols_por_fila]):
                    info_p  = PATTERN_INFO.get(patron, {})
                    bias    = info_p.get("bias", "neutro")
                    color_p = PATTERN_BIAS_COLOR.get(bias, "#aaaaaa")
                    emoji_p = info_p.get("emoji", "•")
                    tickers_patron = df_p[df_p["Patrón"] == patron]["Ticker"].tolist()
                    tickers_str    = "  ·  ".join(tickers_patron)

                    with cols_p[col_i]:
                        st.markdown(f"""
                        <div style="background:#1a1a2e;border-left:4px solid {color_p};
                            border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;">
                          <div style="font-size:1rem;font-weight:bold;color:{color_p}">
                            {emoji_p} {patron}
                            <span style="font-size:.75rem;color:#888;margin-left:8px;
                                background:#0d1117;padding:2px 8px;border-radius:10px;">
                              {bias.upper()}
                            </span>
                          </div>
                          <div style="color:#ccc;font-size:.85rem;margin-top:6px;line-height:1.5">
                            {info_p.get('desc','')[:120]}…
                          </div>
                          <div style="margin-top:8px;font-size:.82rem;color:#00cfff;
                              font-family:monospace;letter-spacing:.5px;">
                            {tickers_str if tickers_str else '—'}
                          </div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Tabla completa ─────────────────────────────────────
            st.markdown("### 📊 Tabla Completa")

            def _css_bias(val):
                if val == "alcista": return "color:#00ff88;font-weight:bold"
                if val == "bajista": return "color:#ff4444;font-weight:bold"
                return "color:#ffd700"

            _cm_p = "map" if hasattr(df_p.style, "map") else "applymap"
            styled_p = df_p[["Emoji","Patrón","Sesgo","Ticker"]].style
            styled_p = getattr(styled_p, _cm_p)(_css_bias, subset=["Sesgo"])
            st.dataframe(styled_p, use_container_width=True, height=420)

            total_encontrados = len(df_p)
            alcistas_n = (df_p["Sesgo"] == "alcista").sum()
            bajistas_n = (df_p["Sesgo"] == "bajista").sum()
            neutros_n  = (df_p["Sesgo"] == "neutro").sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total patrones",  total_encontrados)
            m2.metric("🟢 Alcistas",     alcistas_n)
            m3.metric("🔴 Bajistas",     bajistas_n)
            m4.metric("🟡 Neutros",      neutros_n)

    st.markdown("---")

    # ── Guía de patrones ──────────────────────────────────────────
    if mostrar_guia:
        st.markdown("### 📚 Guía Completa de Patrones")
        st.caption("Qué significa cada patrón y cómo operarlo.")

        GRUPOS = {
            "📈 Patrones Alcistas": ["Trendline Supp.", "Wedge Down", "Triangle Asc.",
                                      "Channel Up", "Double Bottom", "Multiple Bottom"],
            "📉 Patrones Bajistas": ["Trendline Resist.", "Wedge Up", "Triangle Desc.",
                                      "Channel Down", "Double Top", "Multiple Top", "Head&Shoulders"],
            "↔️ Patrones Neutros":  ["Horizontal S/R", "Wedge", "Channel"],
        }

        for grupo, lista_patrones in GRUPOS.items():
            st.markdown(f"#### {grupo}")
            for patron in lista_patrones:
                info_g = PATTERN_INFO.get(patron, {})
                if not info_g:
                    continue
                color_g = PATTERN_BIAS_COLOR.get(info_g.get("bias","neutro"), "#aaa")
                with st.expander(f"{info_g.get('emoji','')} **{patron}**", expanded=False):
                    st.markdown(f"""
                    <div style="border-left:3px solid {color_g};padding:8px 14px;margin:4px 0">
                      <p style="color:#ccc;margin:0 0 8px">{info_g.get('desc','')}</p>
                      <p style="color:{color_g};margin:0">
                        <strong>Cómo operar:</strong> {info_g.get('accion','')}
                      </p>
                    </div>""", unsafe_allow_html=True)
            st.markdown("")

    st.caption(
        "⚠️ Patrones detectados algorítmicamente por Finviz sobre precios de cierre. "
        "No constituyen recomendaciones de inversión. Siempre confirmar con volumen y otros indicadores."
    )
