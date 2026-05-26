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
    """Intenta Twelve Data primero, cae en yfinance si falla o no hay key."""
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

def backtest_estrategia(df: pd.DataFrame, umbral_compra: int = 1, umbral_venta: int = -1) -> dict:
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
                retorno  = (precio_sig - precio_entrada) / precio_entrada * 100
                duracion = (fecha_sig - fecha_entrada).days if hasattr(fecha_sig - fecha_entrada, 'days') else 1
                resultados.append({
                    "Entrada":        fecha_entrada.strftime("%d/%m/%Y") if hasattr(fecha_entrada, 'strftime') else str(fecha_entrada),
                    "Salida":         fecha_sig.strftime("%d/%m/%Y")     if hasattr(fecha_sig,      'strftime') else str(fecha_sig),
                    "Precio entrada": round(precio_entrada, 4),
                    "Precio salida":  round(precio_sig, 4),
                    "Retorno %":      round(retorno, 2),
                    "Días":           duracion,
                    "Señal entrada":  senal_entrada,
                    "Resultado":      "✅ Ganada" if retorno > 0 else "❌ Perdida",
                })
                en_posicion = False

    if not resultados:
        return {"ops": pd.DataFrame(), "win_rate": 0, "retorno_total": 0,
                "retorno_bh": 0, "n_ops": 0, "promedio_op": 0, "max_ganancia": 0, "max_perdida": 0}

    df_ops   = pd.DataFrame(resultados)
    n_ops    = len(df_ops)
    ganadas  = (df_ops['Retorno %'] > 0).sum()
    win_rate = round(ganadas / n_ops * 100, 1) if n_ops > 0 else 0
    ret_bh   = round((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[50]) - 1) * 100, 2)

    return {
        "ops": df_ops,
        "win_rate": win_rate,
        "retorno_total": round(df_ops['Retorno %'].sum(), 2),
        "retorno_bh": ret_bh,
        "n_ops": n_ops,
        "promedio_op": round(df_ops['Retorno %'].mean(), 2),
        "max_ganancia": round(df_ops['Retorno %'].max(), 2),
        "max_perdida":  round(df_ops['Retorno %'].min(), 2),
    }

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
    fuente = "Twelve Data" if TD_API_KEY.get("key") else "Yahoo Finance (fallback)"
    st.caption(f"Datos vía {fuente} · Caché 5 min")

# ══════════════════════════════════════════════════════════════════════════════
# DATOS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

st.title("🛡️ QuantumShield Pro — Trading Terminal")

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Análisis",
    "🌐 Señales del Mercado",
    "📉 Comparación Relativa",
    "🔗 Correlación",
    "📒 Paper Trading & Notas",
    "🤖 Asistente IA",
    "🧪 Backtesting",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANÁLISIS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

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
            st.dataframe(df_trades, use_container_width=True, height=280)

            total    = len(df_trades)
            ganadas  = (df_trades['Estado'] == "🟢 TP alcanzado").sum()
            perdidas = (df_trades['Estado'] == "🔴 SL tocado").sum()
            winrate  = round(ganadas / (ganadas + perdidas) * 100, 1) if (ganadas + perdidas) > 0 else 0

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total",          total)
            s2.metric("✅ TP alcanzado", ganadas)
            s3.metric("❌ SL tocado",    perdidas)
            s4.metric("Win Rate",        f"{winrate}%")

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

    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        bt_umbral_compra = st.slider("Umbral de compra (puntos mínimos)", 1, 4, 2)
    with col_bt2:
        bt_umbral_venta  = st.slider("Umbral de salida (puntos máximos)", -4, -1, -1)
    with col_bt3:
        bt_period_sel = st.selectbox("Período de backtest", ["6mo","1y","2y"], index=1, key="bt_period")

    if st.button("▶️ Ejecutar Backtest", type="primary"):
        with st.spinner("Simulando estrategia..."):
            df_bt = get_data(ticker, bt_period_sel)
            if df_bt.empty or len(df_bt) < 60:
                st.warning("No hay suficientes datos para el backtest (mínimo 60 velas).")
            else:
                bt = backtest_estrategia(df_bt, bt_umbral_compra, bt_umbral_venta)

                if bt['n_ops'] == 0:
                    st.info("La estrategia no generó ninguna operación. Prueba ajustando los umbrales.")
                else:
                    b1, b2, b3, b4, b5, b6 = st.columns(6)
                    b1.metric("Operaciones",        bt['n_ops'])
                    b2.metric("Win Rate",            f"{bt['win_rate']}%")
                    b3.metric("Retorno estrategia",  f"{bt['retorno_total']:+.2f}%",
                              delta=f"{bt['retorno_total'] - bt['retorno_bh']:+.2f}% vs B&H")
                    b4.metric("Buy & Hold",          f"{bt['retorno_bh']:+.2f}%")
                    b5.metric("Mejor operación",     f"{bt['max_ganancia']:+.2f}%")
                    b6.metric("Peor operación",      f"{bt['max_perdida']:+.2f}%")

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
                    fig_eq.update_layout(
                        template="plotly_dark", height=340,
                        title="Curva de equity acumulada",
                        xaxis_title="Número de operación",
                        yaxis_title="Retorno acumulado %",
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
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

                    st.caption(
                        "⚠️ El backtesting usa datos históricos y no garantiza rendimientos futuros. "
                        "Los resultados asumen ejecución al precio de cierre del día siguiente a la señal, "
                        "sin comisiones ni slippage."
                    )
    else:
        st.info("Configura los parámetros y haz clic en **▶️ Ejecutar Backtest** para simular la estrategia.")
