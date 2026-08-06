import os
import time
import json
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Dependencias Opcionales con Fallbacks
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_OK = True
except ImportError:
    ML_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantumShield Pro — Quant Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PRESETS DE ACTIVOS
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
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Coca-Cola (KO)":     "KO",
    "American Express (AXP)": "AXP",
    "Bank of America (BAC)": "BAC",
    "Chevron (CVX)":      "CVX",
    "Occidental Petroleum (OXY)": "OXY",
    "Kraft Heinz (KHC)":  "KHC",
    "Moody's (MCO)":      "MCO",
    "Visa (V)":           "V",
    "Mastercard (MA)":    "MA",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Procter & Gamble (PG)": "PG",
    "Walmart (WMT)":      "WMT",
    "Home Depot (HD)":    "HD",
    "McDonald's (MCD)":   "MCD",
    "Colgate-Palmolive (CL)": "CL",
    "PepsiCo (PEP)":      "PEP",
    "Chubb Limited (CB)": "CB",
    "UnitedHealth (UNH)": "UNH",
    "Verizon (VZ)":       "VZ",
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
PORTFOLIO_FILE     = "qsp_portfolio.json"

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCIA LOCAL Y GESTIÓN DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
def _cargar_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "paper_trades":   data.get("paper_trades", []),
                    "notas":          data.get("notas", {}),
                    "signal_history": data.get("signal_history", {}),
                }
        except (json.JSONDecodeError, OSError) as e:
            st.error(f"Error al cargar archivo de portafolio local: {e}")
    return {"paper_trades": [], "notas": {}, "signal_history": {}}

def guardar_portfolio() -> bool:
    try:
        data = {
            "paper_trades":   st.session_state.get("paper_trades", []),
            "notas":          st.session_state.get("notas", {}),
            "signal_history": st.session_state.get("signal_history", {}),
            "guardado_en":     datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except OSError:
        return False

def portfolio_a_json() -> str:
    data = {
        "paper_trades":   st.session_state.get("paper_trades", []),
        "notas":          st.session_state.get("notas", {}),
        "signal_history": st.session_state.get("signal_history", {}),
        "exportado_en":   datetime.now().strftime("%d/%m/%Y %H:%M"),
        "version":        "QuantumShield Pro 2.0 Pro",
    }
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)

def importar_portfolio_json(contenido: str) -> tuple:
    try:
        data = json.loads(contenido)
        if "paper_trades" not in data:
            return False, "Archivo inválido: formato no reconocido."
        st.session_state.paper_trades   = data.get("paper_trades", [])
        st.session_state.notas          = data.get("notas", {})
        st.session_state.signal_history = data.get("signal_history", {})
        guardar_portfolio()
        return True, f"Importado: {len(st.session_state.paper_trades)} posiciones, {len(st.session_state.notas)} notas."
    except json.JSONDecodeError as e:
        return False, f"Error de formato JSON: {e}"

_datos_guardados = _cargar_portfolio()
for key, val in [
    ("paper_trades", _datos_guardados["paper_trades"]),
    ("notas", _datos_guardados["notas"]),
    ("signal_history", _datos_guardados["signal_history"]),
    ("chat_history", {}),
    ("analisis_cache", {}),
    ("noticias_cache", {}),
    ("finviz_cache", {})
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════════════════
# SERVICIO DE ALERTAS DE TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════
def enviar_alerta_telegram(mensaje: str) -> bool:
    token = st.session_state.get("telegram_token", "").strip()
    chat_id = st.session_state.get("telegram_chat_id", "").strip()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.ok
    except requests.RequestException:
        return False

# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS Y MÉTRICAS CUANTITATIVAS AVANZADAS
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
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0
    
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr_s    = tr.rolling(periodo).mean()
    plus_di  = 100 * (plus_dm.rolling(periodo).mean() / atr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(periodo).mean() / atr_s.replace(0, np.nan))
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(periodo).mean()

def calcular_stoch_rsi(serie: pd.Series, periodo_rsi=14, periodo_stoch=14, suavizado_k=3, suavizado_d=3):
    rsi     = calcular_rsi(serie, periodo_rsi)
    min_rsi = rsi.rolling(periodo_stoch).min()
    max_rsi = rsi.rolling(periodo_stoch).max()
    rango   = (max_rsi - min_rsi).replace(0, np.nan)
    k_raw   = (rsi - min_rsi) / rango * 100
    k       = k_raw.rolling(suavizado_k).mean()
    d       = k.rolling(suavizado_d).mean()
    return k, d

def calcular_sortino_ratio(retornos: pd.Series, rf_rate: float = 0.0) -> float:
    retornos_limpios = retornos.dropna()
    if len(retornos_limpios) == 0:
        return 0.0
    retorno_medio = retornos_limpios.mean() - rf_rate
    retornos_negativos = retornos_limpios[retornos_limpios < 0]
    downside_std = retornos_negativos.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return round(float(retorno_medio / downside_std * np.sqrt(len(retornos_limpios))), 2)

def calcular_var_95(retornos: pd.Series, capital: float = 10000.0) -> tuple:
    retornos_limpios = retornos.dropna()
    if len(retornos_limpios) < 5:
        return 0.0, 0.0
    var_pct = np.percentile(retornos_limpios, 5)
    var_usd = abs(capital * (var_pct / 100.0))
    return round(float(abs(var_pct)), 2), round(float(var_usd), 2)

def calcular_profit_factor(df_ops: pd.DataFrame) -> float:
    if df_ops.empty or 'Retorno %' not in df_ops.columns:
        return 0.0
    ganancias = df_ops[df_ops['Retorno %'] > 0]['Retorno %'].sum()
    perdidas  = abs(df_ops[df_ops['Retorno %'] < 0]['Retorno %'].sum())
    if perdidas == 0:
        return round(float(ganancias), 2) if ganancias > 0 else 0.0
    return round(float(ganancias / perdidas), 2)

def calcular_criterio_kelly(win_rate_pct: float, avg_win: float, avg_loss: float) -> float:
    if win_rate_pct <= 0 or avg_loss <= 0:
        return 0.0
    p = win_rate_pct / 100.0
    q = 1.0 - p
    b = abs(avg_win / avg_loss)
    kelly = (b * p - q) / b
    return round(float(max(0.0, kelly * 100.0)), 2)

# ══════════════════════════════════════════════════════════════════════════════
# MODELO MACHINE LEARNING DE PROBABILIDAD DE ACIERTO
# ══════════════════════════════════════════════════════════════════════════════
def predecir_probabilidad_ml(df: pd.DataFrame) -> tuple:
    if not ML_OK or len(df) < 60:
        return None, "ML no disponible o datos insuficientes."

    df_ml = df.copy()
    df_ml['Retorno_Futuro'] = (df_ml['Close'].shift(-3) - df_ml['Close']) / df_ml['Close']
    df_ml['Target'] = (df_ml['Retorno_Futuro'] > 0.01).astype(int)

    features = ['RSI', 'ADX', 'MACD_HIST', 'ATR']
    df_ml['EMA_DIFF'] = (df_ml['Close'] - df_ml['EMA50']) / df_ml['EMA50']
    features.append('EMA_DIFF')

    df_ml.dropna(subset=features + ['Target'], inplace=True)
    if len(df_ml) < 40:
        return None, "Insuficientes muestras limpias."

    X = df_ml[features].values[:-1]
    y = df_ml['Target'].values[:-1]
    X_actual = df_ml[features].values[-1:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_act_scaled = scaler.transform(X_actual)

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_scaled, y)

    prob = clf.predict_proba(X_act_scaled)[0][1] * 100.0
    return round(float(prob), 1), "Modelo IA RandomForest calibrado ✓"

# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE PATRONES DE VELAS Y ESTRUCTURA DE PRECIO
# ══════════════════════════════════════════════════════════════════════════════
def detectar_velas_japonesas(df: pd.DataFrame) -> list:
    if len(df) < 5:
        return []
    patrones = []
    o = df['Open'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)
    idx = df.index

    for i in range(max(1, len(df)-5), len(df)):
        cuerpo     = abs(c[i] - o[i])
        rango_v    = h[i] - l[i]
        if rango_v == 0: 
            continue
        sombra_sup = h[i] - max(c[i], o[i])
        sombra_inf = min(c[i], o[i]) - l[i]
        alcista    = c[i] > o[i]
        rel_cuerpo = cuerpo / rango_v

        if rel_cuerpo < 0.1:
            patrones.append({"nombre": "Doji", "fecha": idx[i], "tipo": "neutro",
                "desc": "Apertura y cierre casi iguales. Indecisión del mercado.",
                "accion": "Esperar confirmación en la siguiente vela."})
            continue

        if sombra_inf > cuerpo * 2 and sombra_sup < cuerpo * 0.5 and i > 0 and c[i-1] < o[i-1]:
            patrones.append({"nombre": "Hammer 🔨", "fecha": idx[i], "tipo": "alcista",
                "desc": "Sombra inferior larga tras tendencia bajista. Compradores rechazaron mínimos.",
                "accion": "Señal de reversión alcista. Buscar entrada LONG."})

        elif sombra_sup > cuerpo * 2 and sombra_inf < cuerpo * 0.5 and i > 0 and c[i-1] > o[i-1]:
            patrones.append({"nombre": "Shooting Star ⭐", "fecha": idx[i], "tipo": "bajista",
                "desc": "Sombra superior larga tras tendencia alcista. Vendedores rechazaron máximos.",
                "accion": "Señal de reversión bajista. Vigilar entrada SHORT."})

        elif alcista and rel_cuerpo > 0.9:
            patrones.append({"nombre": "Marubozu Alcista", "fecha": idx[i], "tipo": "alcista",
                "desc": "Vela sin sombras. Dominio total comprador.",
                "accion": "Continuación probable. Mantener LONG."})

        elif not alcista and rel_cuerpo > 0.9:
            patrones.append({"nombre": "Marubozu Bajista", "fecha": idx[i], "tipo": "bajista",
                "desc": "Vela sin sombras. Dominio total vendedor.",
                "accion": "Continuación probable. Evitar compras."})

        if i > 0 and alcista and not (c[i-1] > o[i-1]):
            if c[i] > o[i-1] and o[i] < c[i-1]:
                patrones.append({"nombre": "Engulfing Alcista", "fecha": idx[i], "tipo": "alcista",
                    "desc": "Vela alcista envuelve la bajista anterior.",
                    "accion": "Alta probabilidad de reversión LONG."})

        elif i > 0 and not alcista and (c[i-1] > o[i-1]):
            if o[i] > c[i-1] and c[i] < o[i-1]:
                patrones.append({"nombre": "Engulfing Bajista", "fecha": idx[i], "tipo": "bajista",
                    "desc": "Vela bajista envuelve la alcista anterior.",
                    "accion": "Alta probabilidad de reversión SHORT."})

    return patrones[-6:]

def calcular_volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return pd.DataFrame()
    precio_min = float(df['Low'].min())
    precio_max = float(df['High'].max())
    rangos     = np.linspace(precio_min, precio_max, bins + 1)
    vol_por_nivel = np.zeros(bins)
    
    for _, row in df.iterrows():
        idx_low  = np.searchsorted(rangos, float(row['Low']),  side='left')
        idx_high = np.searchsorted(rangos, float(row['High']), side='right')
        idx_low  = max(0, min(idx_low,  bins - 1))
        idx_high = max(0, min(idx_high, bins))
        n_niveles = max(1, idx_high - idx_low)
        vol_por_nivel[idx_low:idx_high] += float(row['Volume']) / n_niveles
        
    precio_mid = (rangos[:-1] + rangos[1:]) / 2
    poc_idx    = np.argmax(vol_por_nivel)
    return pd.DataFrame({
        'precio': precio_mid,
        'volumen': vol_por_nivel,
        'poc': [i == poc_idx for i in range(bins)],
    })

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

def calcular_puntos(fila) -> tuple:
    puntos  = 0
    razones = []

    if float(fila['Close']) > float(fila['EMA50']):
        puntos += 1
        razones.append("✅ Precio sobre EMA50 (alcista)")
    else:
        puntos -= 1
        razones.append("❌ Precio bajo EMA50 (bajista)")

    rsi_val = float(fila['RSI'])
    if rsi_val < 30:
        puntos += 2
        razones.append(f"✅ RSI sobrevendido ({rsi_val:.1f})")
    elif rsi_val > 70:
        puntos -= 2
        razones.append(f"❌ RSI sobrecomprado ({rsi_val:.1f})")
    elif 40 <= rsi_val <= 60:
        puntos += 1
        razones.append(f"✅ RSI neutral ({rsi_val:.1f})")

    if float(fila['MACD']) > float(fila['MACD_SIGNAL']):
        puntos += 1
        razones.append("✅ MACD sobre señal (alcista)")
    else:
        puntos -= 1
        razones.append("❌ MACD bajo señal (bajista)")

    if float(fila['Close']) < float(fila['BB_LOWER']):
        puntos += 1
        razones.append("✅ Precio bajo BB inferior")
    elif float(fila['Close']) > float(fila['BB_UPPER']):
        puntos -= 1
        razones.append("❌ Precio sobre BB superior")

    adx_val = float(fila['ADX']) if 'ADX' in fila.index and not pd.isna(fila['ADX']) else 0
    if adx_val >= 25:
        razones.append(f"✅ ADX {adx_val:.1f} — tendencia fuerte")
    elif adx_val >= 15:
        razones.append(f"⚠️ ADX {adx_val:.1f} — tendencia moderada")
    else:
        puntos = max(-1, min(1, puntos))
        razones.append(f"⚠️ ADX {adx_val:.1f} — sin tendencia clara")

    return puntos, razones, rsi_val, adx_val

def generar_senal(df: pd.DataFrame) -> dict:
    ultima = df.iloc[-1]
    puntos, razones, rsi_val, adx_val = calcular_puntos(ultima)

    if   puntos >= 3:  senal, color = "🟢 COMPRA FUERTE", "#00ff88"
    elif puntos >= 1:  senal, color = "🟡 COMPRA DÉBIL",  "#ffd700"
    elif puntos <= -3: senal, color = "🔴 VENTA FUERTE",  "#ff4444"
    elif puntos <= -1: senal, color = "🟠 VENTA DÉBIL",   "#ff8c00"
    else:              senal, color = "⚪ NEUTRAL",        "#aaaaaa"

    atr_val = float(ultima['ATR']) if not pd.isna(ultima['ATR']) else 0
    precio  = float(ultima['Close'])

    return {
        "senal": senal, "color": color, "puntos": puntos,
        "razones": razones,
        "sl": precio - 2 * atr_val,
        "tp": precio + 3 * atr_val,
        "atr": atr_val, "rsi": rsi_val, "adx": adx_val,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE - INGESTIÓN MULTIFUENTE & ORDER BOOK
# ══════════════════════════════════════════════════════════════════════════════
PERIOD_DAYS     = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 732}
PERIOD_OUTPUT   = {"1mo": 35, "3mo": 95, "6mo": 190, "1y": 370, "2y": 740}
TD_BASE_URL     = "https://api.twelvedata.com/time_series"
BINANCE_BASE    = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVALS = {"1mo": ("1d", 35), "3mo": ("1d", 95), "6mo": ("1d", 190), "1y": ("1d", 370), "2y": ("1d", 740)}
BINANCE_MTF = {"4H": ("4h", 180), "1D": ("1d", 200), "1W": ("1w", 104)}
YF_MTF      = {"4H": ("4h", "60d"), "1D": ("1d", "1y"), "1W": ("1wk", "2y")}

def _binance_download(ticker: str, period: str) -> pd.DataFrame:
    sym = BINANCE_SYMBOLS.get(ticker)
    if not sym:
        return pd.DataFrame()
    interval, limit = BINANCE_INTERVALS.get(period, ("1d", 190))
    try:
        resp = requests.get(BINANCE_BASE, params={"symbol": sym, "interval": interval, "limit": limit}, timeout=10)
        if not resp.ok:
            return pd.DataFrame()
        raw = resp.json()
        if not isinstance(raw, list) or not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=[
            "open_time","Open","High","Low","Close","Volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        for c in ["Open","High","Low","Close","Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        df.index.name = None
        return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    except requests.RequestException:
        return pd.DataFrame()

def obtener_order_book_binance(ticker: str) -> pd.DataFrame:
    sym = BINANCE_SYMBOLS.get(ticker)
    if not sym:
        return pd.DataFrame(), pd.DataFrame()
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={sym}&limit=20"
        resp = requests.get(url, timeout=5)
        if not resp.ok:
            return pd.DataFrame(), pd.DataFrame()
        data = resp.json()
        bids = pd.DataFrame(data['bids'], columns=['Precio', 'Cantidad'], dtype=float)
        asks = pd.DataFrame(data['asks'], columns=['Precio', 'Cantidad'], dtype=float)
        bids['Acumulado'] = bids['Cantidad'].cumsum()
        asks['Acumulado'] = asks['Cantidad'].cumsum()
        return bids, asks
    except requests.RequestException:
        return pd.DataFrame(), pd.DataFrame()

def _ticker_td(ticker: str) -> str:
    t = ticker.replace("-USD", "/USD").replace("-", "/")
    return t[1:] if t.startswith("^") else t

def _td_download(ticker: str, period: str, td_key: str = "") -> pd.DataFrame:
    key = (td_key or "").strip()
    if not key:
        return pd.DataFrame()
    sym      = _ticker_td(ticker)
    outsize  = PERIOD_OUTPUT.get(period, 190)
    params   = {"symbol": sym, "interval": "1day", "outputsize": outsize, "apikey": key, "format": "JSON", "order": "ASC"}
    try:
        resp = requests.get(TD_BASE_URL, params=params, timeout=15)
        if not resp.ok:
            return pd.DataFrame()
        data = resp.json()
        if data.get("status") == "error" or "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.index.name = None
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
        return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    except requests.RequestException:
        return pd.DataFrame()

def _yf_fallback(ticker: str, period: str) -> pd.DataFrame:
    dias  = PERIOD_DAYS.get(period, 183)
    today = datetime.today()
    start = (today - pd.Timedelta(days=dias)).strftime("%Y-%m-%d")
    end   = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    for intento in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, timeout=15)
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
            if "RateLimit" in type(e).__name__ and intento < 2:
                time.sleep(2 ** (intento + 1))
                continue
            break
    return pd.DataFrame()

def _descargar_raw(ticker: str, period: str, td_key: str = "") -> pd.DataFrame:
    if ticker in BINANCE_SYMBOLS:
        df = _binance_download(ticker, period)
        if not df.empty:
            return df
    df = _td_download(ticker, period, td_key)
    if df.empty:
        df = _yf_fallback(ticker, period)
    return df

@st.cache_data(ttl=300)
def get_data(ticker: str, period: str, td_key: str = "") -> pd.DataFrame:
    try:
        df = _descargar_raw(ticker, period, td_key)
    except Exception:
        return pd.DataFrame()
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
    df["STOCH_K"], df["STOCH_D"]                   = calcular_stoch_rsi(close)
    if "Volume" in df.columns:
        df["VOL_ANOMALO"] = volumen_anomalo(df)
    return df

@st.cache_data(ttl=300)
def get_close_only(ticker: str, period: str, td_key: str = "") -> pd.Series:
    df = get_data(ticker, period, td_key)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    return df["Close"].rename(ticker)

@st.cache_data(ttl=120)
def get_data_mtf(ticker: str, tf: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if ticker in BINANCE_SYMBOLS:
        sym = BINANCE_SYMBOLS[ticker]
        interval, limit = BINANCE_MTF[tf]
        try:
            resp = requests.get(BINANCE_BASE, params={"symbol": sym, "interval": interval, "limit": limit}, timeout=10)
            if resp.ok:
                raw = resp.json()
                if isinstance(raw, list) and raw:
                    df = pd.DataFrame(raw, columns=["open_time","Open","High","Low","Close","Volume","close_time","qav","trades","tbbav","tbqav","ignore"])
                    for col in ["Open","High","Low","Close","Volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.index = pd.to_datetime(df["open_time"], unit="ms")
                    df.index.name = None
                    df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
        except requests.RequestException:
            pass

    if df.empty:
        yf_interval, yf_period = YF_MTF[tf]
        try:
            raw = yf.download(ticker, period=yf_period, interval=yf_interval, progress=False, auto_adjust=True)
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
# SERVICIO IA - GROQ PROVIDER
# ══════════════════════════════════════════════════════════════════════════════
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

def _groq_headers() -> dict:
    h = {"Content-Type": "application/json"}
    k = st.session_state.get("groq_api_key", "").strip()
    if k:
        h["Authorization"] = f"Bearer {k}"
    return h

def _ia_activa() -> bool:
    return bool(st.session_state.get("groq_api_key", "").strip())

def _groq(system: str, user: str, max_tokens: int = 1000) -> str:
    if not _ia_activa():
        return "❌ API key no configurada."
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=_groq_headers(), json=payload, timeout=30)
        if not resp.ok:
            if resp.status_code == 401: return "❌ API key inválida."
            if resp.status_code == 429: return "⚠️ Rate limit alcanzado."
            return f"❌ Error HTTP {resp.status_code}"
        return resp.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"❌ Error de conexión: {e}"

def _groq_stream(system: str, messages: list, max_tokens: int = 1000):
    if not _ia_activa():
        yield "❌ API key no configurada."
        return
    mensajes_limpios = [
        {"role": m["role"], "content": str(m["content"])}
        for m in messages if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + mensajes_limpios,
        "stream": True,
    }
    try:
        with requests.post(GROQ_API_URL, headers=_groq_headers(), json=payload, timeout=60, stream=True) as resp:
            if not resp.ok:
                yield f"\n❌ Error {resp.status_code}"
                return
            for line in resp.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]": break
                    try:
                        chunk = json.loads(data_str)
                        yield chunk["choices"][0].get("delta", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue
    except requests.RequestException as e:
        yield f"\n❌ Error de conexión: {e}"

def _resumen_tecnico(ticker: str, df: pd.DataFrame, info: dict) -> str:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg  = ((float(last["Close"]) / float(prev["Close"])) - 1) * 100
    return f"""
Ticker: {ticker}
Precio: ${float(last["Close"]):,.4f} ({chg:+.2f}% hoy)
EMA20: {float(last["EMA20"]):,.4f} | EMA50: {float(last["EMA50"]):,.4f}
RSI(14): {float(last["RSI"]):.1f} | ADX(14): {float(last["ADX"]):.1f}
MACD: {float(last["MACD"]):.4f} | Señal: {float(last["MACD_SIGNAL"]):.4f}
BB Sup: {float(last["BB_UPPER"]):,.4f} | BB Inf: {float(last["BB_LOWER"]):,.4f}
ATR(14): {float(last["ATR"]):,.4f}
SL sugerido: ${info["sl"]:,.4f} | TP sugerido: ${info["tp"]:,.4f}
Señal: {info["senal"]} (puntuación: {info["puntos"]:+d})
Razones: {"; ".join(info["razones"])}
""".strip()

# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
def backtest_estrategia(df: pd.DataFrame, umbral_compra: int = 1, umbral_venta: int = -1,
                        comision_pct: float = 0.1, usar_sl_tp: bool = True,
                        mult_sl: float = 2.0, mult_tp: float = 3.0) -> dict:
    resultados = []
    en_posicion = False
    precio_entrada, fecha_entrada, senal_entrada, idx_entrada = 0.0, None, "", -1
    sl_nivel, tp_nivel = None, None

    puntos_serie = []
    for i in range(len(df)):
        if i < 50:
            puntos_serie.append(0)
            continue
        try:
            p, _, _, _ = calcular_puntos(df.iloc[i])
        except Exception:
            p = 0
        puntos_serie.append(p)

    df = df.copy()
    df['_puntos'] = puntos_serie

    def _cerrar(precio_out, fecha_out, motivo):
        retorno_bruto = (precio_out - precio_entrada) / precio_entrada * 100
        retorno  = retorno_bruto - (comision_pct * 2)
        duracion = (fecha_out - fecha_entrada).days if hasattr(fecha_out - fecha_entrada, 'days') else 1
        resultados.append({
            "Entrada": fecha_entrada.strftime("%d/%m/%Y") if hasattr(fecha_entrada, 'strftime') else str(fecha_entrada),
            "Salida": fecha_out.strftime("%d/%m/%Y") if hasattr(fecha_out, 'strftime') else str(fecha_out),
            "Precio entrada": round(precio_entrada, 4),
            "Precio salida": round(precio_out, 4),
            "Retorno bruto %": round(retorno_bruto, 2),
            "Retorno %": round(retorno, 2),
            "Días": duracion,
            "Señal entrada": senal_entrada,
            "Salida por": motivo,
            "Resultado": "✅ Ganada" if retorno > 0 else "❌ Perdida",
        })

    for i in range(51, len(df) - 1):
        pts_hoy    = df['_puntos'].iloc[i]
        precio_sig = float(df['Close'].iloc[i + 1])
        fecha_sig  = df.index[i + 1]

        if not en_posicion:
            if pts_hoy >= umbral_compra:
                en_posicion = True
                precio_entrada = precio_sig
                fecha_entrada  = fecha_sig
                senal_entrada  = f"+{pts_hoy}"
                idx_entrada    = i + 1
                sl_nivel = tp_nivel = None
                if usar_sl_tp and 'ATR' in df.columns:
                    atr_e = df['ATR'].iloc[i + 1]
                    if not pd.isna(atr_e) and float(atr_e) > 0:
                        sl_nivel = precio_entrada - mult_sl * float(atr_e)
                        tp_nivel = precio_entrada + mult_tp * float(atr_e)
        else:
            if sl_nivel is not None and i > idx_entrada:
                lo, hi = float(df['Low'].iloc[i]), float(df['High'].iloc[i])
                if lo <= sl_nivel:
                    _cerrar(sl_nivel, df.index[i], "🛑 Stop Loss")
                    en_posicion = False
                    continue
                if hi >= tp_nivel:
                    _cerrar(tp_nivel, df.index[i], "🎯 Take Profit")
                    en_posicion = False
                    continue
            if pts_hoy <= umbral_venta or i == len(df) - 2:
                motivo = "📉 Señal contraria" if pts_hoy <= umbral_venta else "🏁 Fin de datos"
                _cerrar(precio_sig, fecha_sig, motivo)
                en_posicion = False

    if not resultados:
        return {"ops": pd.DataFrame(), "win_rate": 0, "retorno_total": 0, "retorno_bh": 0, "n_ops": 0, "promedio_op": 0, "max_ganancia": 0, "max_perdida": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0, "max_drawdown": 0}

    df_ops   = pd.DataFrame(resultados)
    n_ops    = len(df_ops)
    ganadas  = (df_ops['Retorno %'] > 0).sum()
    win_rate = round(ganadas / n_ops * 100, 1) if n_ops > 0 else 0
    ret_bh   = round((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[50]) - 1) * 100, 2)

    retornos = df_ops['Retorno %']
    sharpe   = round(retornos.mean() / retornos.std() * np.sqrt(n_ops), 2) if retornos.std() > 0 else 0
    sortino  = calcular_sortino_ratio(retornos)
    pfactor  = calcular_profit_factor(df_ops)
    equity   = (1 + retornos / 100).cumprod()
    ret_comp = round(float(equity.iloc[-1] - 1) * 100, 2)
    drawdown = (equity / equity.cummax() - 1) * 100
    max_dd   = round(float(drawdown.min()), 2)

    return {
        "ops": df_ops, "win_rate": win_rate, "retorno_total": ret_comp, "retorno_bh": ret_bh,
        "n_ops": n_ops, "promedio_op": round(retornos.mean(), 2), "max_ganancia": round(retornos.max(), 2),
        "max_perdida": round(retornos.min(), 2), "sharpe": sharpe, "sortino": sortino, "profit_factor": pfactor, "max_drawdown": max_dd,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SCRAPING EN FINVIZ CON RESILIENCIA HTTP
# ══════════════════════════════════════════════════════════════════════════════
FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

@st.cache_data(ttl=600)
def finviz_scrape(ticker: str) -> dict:
    result = {"news": [], "fundamentals": {}, "ok": False}
    if not BS4_OK or "-USD" in ticker or ticker.startswith("^"):
        return result
    try:
        url  = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
        resp = requests.get(url, headers=FINVIZ_HEADERS, timeout=10)
        if not resp.ok:
            return result
        soup = BeautifulSoup(resp.text, "html.parser")

        fundamentals = {}
        tabla = soup.find("table", class_="snapshot-table2") or soup.find("table", {"class": lambda c: c and "snapshot" in c})
        if tabla:
            celdas = tabla.find_all("td")
            for i in range(0, len(celdas) - 1, 2):
                key = celdas[i].get_text(strip=True)
                val = celdas[i+1].get_text(strip=True)
                if key and val:
                    fundamentals[key] = val

        noticias = []
        tabla_news = soup.find("table", id="news-table")
        if tabla_news:
            fecha_actual = ""
            for fila in tabla_news.find_all("tr")[:20]:
                celdas = fila.find_all("td")
                if len(celdas) < 2: continue
                fecha_td = celdas[0].get_text(strip=True)
                if any(c.isalpha() for c in fecha_td):
                    partes = fecha_td.split()
                    fecha_actual = partes[0] if len(partes) > 1 else fecha_actual
                    hora = partes[-1] if len(partes) > 1 else fecha_td
                else:
                    hora = fecha_td
                enlace = celdas[1].find("a")
                fuente = celdas[1].find("span")
                if enlace:
                    noticias.append({
                        "fecha": f"{fecha_actual} {hora}".strip(),
                        "titulo": enlace.get_text(strip=True),
                        "url": enlace.get("href", "#"),
                        "fuente": fuente.get_text(strip=True) if fuente else "",
                    })

        result["news"] = noticias
        result["fundamentals"] = fundamentals
        result["ok"] = bool(fundamentals or noticias)
        return result
    except requests.RequestException:
        return result

def _parse_finviz_num(val: str):
    if not val or val == "-": return None
    v = val.strip().replace("%", "")
    mult = 1.0
    if v and v[-1] in ("B", "M", "K"):
        mult = {"B": 1e9, "M": 1e6, "K": 1e3}[v[-1]]
        v = v[:-1]
    try:
        return float(v) * mult
    except (ValueError, TypeError):
        return None

def analizar_buffett(fundamentals: dict) -> dict:
    pe     = _parse_finviz_num(fundamentals.get("P/E", "-"))
    roe    = _parse_finviz_num(fundamentals.get("ROE", "-"))
    deuda  = _parse_finviz_num(fundamentals.get("Debt/Eq", "-"))
    margen = _parse_finviz_num(fundamentals.get("Profit Margin", "-"))

    score = 0
    score += 30 if pe and 0 < pe < 15 else (20 if pe and pe < 25 else (10 if pe and pe < 35 else 0))
    score += 30 if roe and roe >= 20 else (20 if roe and roe >= 15 else (10 if roe and roe >= 10 else 0))
    score += 20 if deuda is not None and deuda <= 0.3 else (15 if deuda is not None and deuda <= 0.5 else (5 if deuda is not None and deuda <= 1.0 else 0))
    score += 20 if margen and margen >= 20 else (10 if margen and margen >= 10 else (5 if margen and margen >= 0 else 0))

    veredicto = "🏛️ Cumple criterios Buffett" if score >= 70 else ("🟡 Parcialmente sólido" if score >= 40 else "🔴 No cumple")
    return {"P/E": pe, "ROE %": roe, "Deuda/Patrim.": deuda, "Margen Neto %": margen, "Score Buffett": score, "Veredicto": veredicto}

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE PATRONES CHARTISTAS Y DIVERGENCIAS (FECHAS FORMATEADAS)
# ══════════════════════════════════════════════════════════════════════════════
def calcular_fibonacci(df: pd.DataFrame) -> dict:
    maximo, minimo = float(df['High'].max()), float(df['Low'].min())
    rango = maximo - minimo
    return {
        "0%": maximo, "23.6%": maximo - 0.236 * rango, "38.2%": maximo - 0.382 * rango,
        "50%": maximo - 0.500 * rango, "61.8%": maximo - 0.618 * rango,
        "78.6%": maximo - 0.786 * rango, "100%": minimo,
    }

FIBONACCI_COLORES = {
    "0%": "rgba(255,255,255,0.5)", "23.6%": "rgba(255,215,0,0.6)", "38.2%": "rgba(0,207,255,0.6)",
    "50%": "rgba(0,255,136,0.7)", "61.8%": "rgba(0,207,255,0.6)", "78.6%": "rgba(255,140,0,0.6)", "100%": "rgba(255,255,255,0.5)"
}

def detectar_divergencias(df: pd.DataFrame, ventana: int = 5) -> list:
    if 'RSI' not in df.columns or len(df) < ventana * 3: return []
    close, rsi, idx = df['Close'].values, df['RSI'].values, df.index
    divs = []
    idx_min_precio = argrelextrema(close, np.less, order=ventana)[0]
    idx_max_precio = argrelextrema(close, np.greater, order=ventana)[0]

    for i in range(1, len(idx_min_precio)):
        i1, i2 = idx_min_precio[i-1], idx_min_precio[i]
        if close[i2] < close[i1] and rsi[i2] > rsi[i1] and not np.isnan(rsi[i1]) and not np.isnan(rsi[i2]):
            divs.append({"tipo": "alcista", "fecha1": idx[i1], "fecha2": idx[i2], "precio1": close[i1], "precio2": close[i2], "rsi1": rsi[i1], "rsi2": rsi[i2]})

    for i in range(1, len(idx_max_precio)):
        i1, i2 = idx_max_precio[i-1], idx_max_precio[i]
        if close[i2] > close[i1] and rsi[i2] < rsi[i1] and not np.isnan(rsi[i1]) and not np.isnan(rsi[i2]):
            divs.append({"tipo": "bajista", "fecha1": idx[i1], "fecha2": idx[i2], "precio1": close[i1], "precio2": close[i2], "rsi1": rsi[i1], "rsi2": rsi[i2]})

    return [d for d in divs if d["tipo"] == "alcista"][-3:] + [d for d in divs if d["tipo"] == "bajista"][-3:]

def registrar_senal(ticker: str, info: dict, precio: float):
    hist = st.session_state.signal_history.setdefault(ticker, [])
    nueva_senal = info["senal"]
    if not hist or hist[-1]["Señal"] != nueva_senal:
        hist.append({
            "Fecha": datetime.now().strftime("%d/%m/%Y %H:%M"), "Ticker": ticker,
            "Señal": nueva_senal, "Puntos": info["puntos"], "Precio": round(precio, 4),
            "RSI": round(info["rsi"], 1), "ADX": round(info["adx"], 1),
        })
        st.session_state.signal_history[ticker] = hist[-50:]
        
        # Disparar alerta automática de Telegram si la señal es fuerte
        if "FUERTE" in nueva_senal and st.session_state.get("auto_telegram_alerts", False):
            msg = f"🚨 *NUEVA SEÑAL CUANTITATIVA*\n\nActivo: `{ticker}`\nSeñal: *{nueva_senal}*\nPrecio: `${precio:,.4f}`\nRSI: `{info['rsi']:.1f}` | ADX: `{info['adx']:.1f}`"
            enviar_alerta_telegram(msg)

PATTERN_INFO = {
    "Trendline Supp.": {"emoji": "📈", "bias": "alcista", "desc": "Soporte dinámico ascendente.", "accion": "LONG cerca de la línea."},
    "Trendline Resist.": {"emoji": "📉", "bias": "bajista", "desc": "Resistencia dinámica descendente.", "accion": "SHORT o venta."},
    "Horizontal S/R": {"emoji": "↔️", "bias": "neutro", "desc": "Nivel horizontal clave.", "accion": "Esperar ruptura."},
    "Wedge Up": {"emoji": "🔺", "bias": "bajista", "desc": "Cuña ascendente de agotamiento.", "accion": "Esperar ruptura inferior."},
    "Wedge": {"emoji": "🔷", "bias": "neutro", "desc": "Compresión simétrica.", "accion": "Operar dirección de ruptura."},
    "Wedge Down": {"emoji": "🔻", "bias": "alcista", "desc": "Cuña descendente de agotamiento.", "accion": "Esperar ruptura superior."},
    "Triangle Asc.": {"emoji": "△", "bias": "alcista", "desc": "Resistencia horizontal y mínimos crecientes.", "accion": "LONG en ruptura."},
    "Triangle Desc.": {"emoji": "▽", "bias": "bajista", "desc": "Soporte horizontal y máximos decrecientes.", "accion": "SHORT en ruptura."},
    "Channel Up": {"emoji": "📊", "bias": "alcista", "desc": "Canal paralelo ascendente.", "accion": "LONG en soporte del canal."},
    "Channel": {"emoji": "📊", "bias": "neutro", "desc": "Rango lateral.", "accion": "Comprar soporte, vender resistencia."},
    "Channel Down": {"emoji": "📉", "bias": "bajista", "desc": "Canal paralelo descendente.", "accion": "Evitar compras."},
    "Double Top": {"emoji": "🔔", "bias": "bajista", "desc": "Doble techo de inversión.", "accion": "SHORT en ruptura del cuello."},
    "Multiple Top": {"emoji": "🔔🔔", "bias": "bajista", "desc": "Múltiples rechazos en resistencia.", "accion": "SHORT en resistencia."},
    "Double Bottom": {"emoji": "🏔️", "bias": "alcista", "desc": "Doble suelo de inversión.", "accion": "LONG en ruptura del cuello."},
    "Multiple Bottom": {"emoji": "🏔️🏔️", "bias": "alcista", "desc": "Múltiples soportes validados.", "accion": "LONG en soporte."},
    "Head&Shoulders": {"emoji": "👤", "bias": "bajista", "desc": "Hombro-Cabeza-Hombro.", "accion": "SHORT en ruptura de línea de cuello."},
}
PATTERN_BIAS_COLOR = {"alcista": "#00ff88", "bajista": "#ff4444", "neutro": "#ffd700"}

def _linreg(y: np.ndarray):
    x = np.arange(len(y), dtype=float)
    m = (len(x) * np.dot(x, y) - x.sum() * y.sum()) / (len(x) * (x**2).sum() - x.sum()**2 + 1e-12)
    b = (y.sum() - m * x.sum()) / len(x)
    return m, b

def detectar_patrones(df: pd.DataFrame) -> list:
    if len(df) < 30: 
        return []
    
    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    idx   = df.index
    n     = len(close)
    orden = max(5, n // 20)
    patrones = []

    def fmt_fecha(ts):
        if hasattr(ts, 'strftime'):
            return ts.strftime("%d/%m/%Y")
        return str(ts).split()[0]

    idx_max = argrelextrema(close, np.greater, order=orden)[0]
    idx_min = argrelextrema(close, np.less, order=orden)[0]

    if len(idx_max) >= 2:
        t1, t2 = idx_max[-2], idx_max[-1]
        if abs(close[t1] - close[t2]) / close[t1] < 0.03 and t2 - t1 > orden:
            if (close[t1] - close[t1:t2].min()) / close[t1] > 0.04:
                patrones.append({
                    "patron": "Double Top", 
                    "confianza": "Alta", 
                    "desc": f"Dos techos en ${close[t1]:,.2f} y ${close[t2]:,.2f}.", 
                    "fecha_ini": fmt_fecha(idx[t1]), 
                    "fecha_fin": fmt_fecha(idx[t2])
                })

    if len(idx_min) >= 2:
        b1, b2 = idx_min[-2], idx_min[-1]
        if abs(close[b1] - close[b2]) / close[b1] < 0.03 and b2 - b1 > orden:
            if (close[b1:b2].max() - close[b1]) / close[b1] > 0.04:
                patrones.append({
                    "patron": "Double Bottom", 
                    "confianza": "Alta", 
                    "desc": f"Dos suelos en ${close[b1]:,.2f} y ${close[b2]:,.2f}.", 
                    "fecha_ini": fmt_fecha(idx[b1]), 
                    "fecha_fin": fmt_fecha(idx[b2])
                })

    ventana = min(60, n)
    mh, _ = _linreg(high[-ventana:])
    ml, _ = _linreg(low[-ventana:])
    if (high[-ventana:].mean() - low[-ventana:].mean()) / close[-ventana:].mean() < 0.15:
        if mh > 0.001 and ml > 0.001:
            patrones.append({
                "patron": "Channel Up", 
                "confianza": "Media", 
                "desc": "Canal alcista estructurado.", 
                "fecha_ini": fmt_fecha(idx[-ventana]), 
                "fecha_fin": fmt_fecha(idx[-1])
            })
        elif mh < -0.001 and ml < -0.001:
            patrones.append({
                "patron": "Channel Down", 
                "confianza": "Media", 
                "desc": "Canal bajista estructurado.", 
                "fecha_ini": fmt_fecha(idx[-ventana]), 
                "fecha_fin": fmt_fecha(idx[-1])
            })

    return patrones

TICKER_REGEX = re.compile(r'^[A-Z0-9\.\^\-]{1,12}$')
def validar_ticker(t: str) -> tuple:
    t = t.strip().upper()
    if not t: return False, "Ticker vacío."
    if not TICKER_REGEX.match(t): return False, f"Ticker inválido: '{t}'."
    return True, ""

# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ QuantumShield Pro")
    st.caption("Terminal Cuantitativo & IA")
    st.markdown("---")

    tipo = st.radio("Tipo de activo", ["📈 Acciones", "₿ Criptomonedas", "✏️ Ticker manual"])
    if tipo == "📈 Acciones":
        nombre = st.selectbox("Selecciona acción", list(ACCIONES.keys()))
        ticker = ACCIONES[nombre]
        benchmark = BENCHMARK_ACCIONES
    elif tipo == "₿ Criptomonedas":
        nombre = st.selectbox("Selecciona crypto", list(CRYPTOS.keys()))
        ticker = CRYPTOS[nombre]
        benchmark = BENCHMARK_CRYPTO
    else:
        _ticker_raw = st.text_input("Ticker personalizado", value="NVDA").upper().strip()
        _ok, _msg = validar_ticker(_ticker_raw)
        ticker = _ticker_raw if _ok else "NVDA"
        if not _ok: st.error(_msg)
        benchmark = BENCHMARK_ACCIONES

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
    mostrar_velas   = st.checkbox("Velas japonesas",       value=True)
    mostrar_stoch   = st.checkbox("Stochastic RSI",        value=True)
    mostrar_vp      = st.checkbox("Volume Profile",        value=False)

    st.markdown("---")
    st.subheader("📡 Datos — Twelve Data")
    td_key_input = st.text_input("API Key", type="password")
    if td_key_input:
        st.session_state["td_api_key"] = td_key_input
    else:
        st.session_state["td_api_key"] = st.secrets.get("TD_API_KEY", "")

    st.markdown("---")
    st.subheader("🤖 IA — Groq")
    groq_key_input = st.text_input("Groq Key", type="password")
    if groq_key_input:
        st.session_state["groq_api_key"] = groq_key_input
    else:
        st.session_state["groq_api_key"] = st.secrets.get("GROQ_API_KEY", "")

    st.markdown("---")
    st.subheader("📲 Alertas Telegram")
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat  = st.text_input("Chat ID")
    if tg_token: st.session_state["telegram_token"] = tg_token
    if tg_chat:  st.session_state["telegram_chat_id"] = tg_chat
    st.session_state["auto_telegram_alerts"] = st.checkbox("Alertas Auto en Compras/Ventas Fuertes", value=False)
    if st.button("🔔 Probar Alerta Telegram"):
        if enviar_alerta_telegram("🧪 Alerta de prueba desde *QuantumShield Pro* ✓"):
            st.success("¡Alerta enviada!")
        else:
            st.error("Error al enviar alerta Telegram.")

    st.markdown("---")
    auto_refresh = st.checkbox("Activar auto-refresh", value=False)
    refresh_secs = st.selectbox("Intervalo", [30, 60, 120, 300], index=1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER & DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
TD_KEY = st.session_state.get("td_api_key", "")
st.title("🛡️ QuantumShield Pro — Terminal Quant & IA")

if auto_refresh:
    import streamlit.components.v1 as components
    components.html(f"""
    <script>
      setTimeout(function(){{ window.location.reload(); }}, {refresh_secs * 1000});
    </script>
    <div style="color:#00cfff;font-family:monospace;">⏱️ Auto-refresh activo ({refresh_secs}s)</div>
    """, height=30)

with st.spinner(f"Cargando análisis para {ticker}..."):
    df = get_data(ticker, period, td_key=TD_KEY)

if df.empty:
    st.error(f"Sin datos accesibles para {ticker}.")
    st.stop()

info  = generar_senal(df)
last  = df.iloc[-1]
prev  = df.iloc[-2] if len(df) > 1 else last
price = float(last['Close'])
chg   = ((price / float(prev['Close'])) - 1) * 100 if float(prev['Close']) > 0 else 0.0

prob_ml, msg_ml = predecir_probabilidad_ml(df)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "📊 Análisis", "🌐 Señales", "📉 Comparativa", "🔗 Correlación",
    "📒 Paper Trading", "🤖 Asistente IA", "🧪 Backtest", "⏱️ Multi-TF",
    "🔍 Screener", "📰 Noticias", "🕯️ Patrones", "💼 Portafolio"
])

# ── TAB 1: ANÁLISIS TÉCNICO & ML ──────────────────────────────────────────────
with tab1:
    registrar_senal(ticker, info, price)
    
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Precio", f"${price:,.4f}", f"{chg:+.2f}%")
    k2.metric("RSI (14)", f"{info['rsi']:.1f}")
    k3.metric("ADX (14)", f"{info['adx']:.1f}")
    k4.metric("ATR (14)", f"${info['atr']:,.4f}")
    k5.metric("Stop Loss", f"${info['sl']:,.4f}")
    k6.metric("Take Profit", f"${info['tp']:,.4f}")
    k7.metric("Prob. ML Acierto", f"{prob_ml}%" if prob_ml else "N/A", delta="RandomForest")

    st.markdown(f"""
    <div style="background:#16213e;border:1px solid {info['color']};border-radius:8px;padding:12px;margin:10px 0;">
      <h3 style="color:{info['color']};margin:0;">Señal: {info['senal']} ({info['puntos']:+d} pts)</h3>
    </div>""", unsafe_allow_html=True)

    rows_n = 3 + (1 if 'Volume' in df.columns and mostrar_volumen else 0) + (1 if mostrar_stoch else 0)
    fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
    if mostrar_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color="#ffd700", width=1), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color="#ff8c00", width=1), name="EMA50"), row=1, col=1)

    curr_row = 2
    if 'Volume' in df.columns and mostrar_volumen:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volumen"), row=curr_row, col=1)
        curr_row += 1

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="#c77dff"), name="RSI"), row=curr_row, col=1)
    curr_row += 1

    fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name="MACD Hist"), row=curr_row, col=1)
    
    fig.update_layout(template="plotly_dark", height=750, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if ticker in BINANCE_SYMBOLS:
        st.markdown("### 📊 Profundidad de Mercado (Order Book - Binance)")
        bids, asks = obtener_order_book_binance(ticker)
        if not bids.empty and not asks.empty:
            fig_ob = go.Figure()
            fig_ob.add_trace(go.Scatter(x=bids['Precio'], y=bids['Acumulado'], fill='tozeroy', name='Bids (Compra)', line=dict(color='#00ff88')))
            fig_ob.add_trace(go.Scatter(x=asks['Precio'], y=asks['Acumulado'], fill='tozeroy', name='Asks (Venta)', line=dict(color='#ff4444')))
            fig_ob.update_layout(template="plotly_dark", height=280, title="Profundidad de Órdenes en Tiempo Real", xaxis_title="Precio USD", yaxis_title="Volumen Acumulado")
            st.plotly_chart(fig_ob, use_container_width=True)

# ── TAB 2: SEÑALES ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🌐 Radar Global de Mercado")
    todos = {**ACCIONES, **CRYPTOS}
    filas = []

    def _fetch_fila(item):
        nm, sym = item
        d = get_data(sym, period, td_key=TD_KEY)
        if d.empty: return None
        inf = generar_senal(d)
        p = float(d['Close'].iloc[-1])
        pv = float(d['Close'].iloc[-2]) if len(d) > 1 else p
        return {
            "Activo": nm, "Ticker": sym, "Precio": p,
            "Cambio %": round(((p/pv)-1)*100, 2), "RSI": round(float(d['RSI'].iloc[-1]), 1),
            "ADX": round(float(d['ADX'].iloc[-1]), 1) if 'ADX' in d.columns else 0,
            "Señal": inf['senal'], "Puntos": inf['puntos']
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_fetch_fila, item) for item in todos.items()]
        for f in as_completed(futures):
            r = f.result()
            if r: filas.append(r)

    if filas:
        df_radar = pd.DataFrame(filas).sort_values("Puntos", ascending=False)
        st.dataframe(df_radar, use_container_width=True)

# ── TAB 3: COMPARATIVA RELATIVA ───────────────────────────────────────────────
with tab3:
    st.subheader(f"📉 Rendimiento Relativo vs {benchmark}")
    s_act = get_close_only(ticker, period, td_key=TD_KEY)
    s_ben = get_close_only(benchmark, period, td_key=TD_KEY)
    if not s_act.empty and not s_ben.empty:
        df_c = pd.DataFrame({ticker: s_act, benchmark: s_ben}).dropna()
        df_n = df_c / df_c.iloc[0] * 100
        fig_comp = px.line(df_n, template="plotly_dark", title="Rendimiento Normalizado (Base 100)")
        st.plotly_chart(fig_comp, use_container_width=True)

# ── TAB 4: CORRELACIÓN ────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔗 Matriz de Correlaciones")
    series_l = []
    for nm, sym in ACCIONES.items():
        s = get_close_only(sym, period, td_key=TD_KEY)
        if not s.empty:
            s.name = sym
            series_l.append(s)
    if series_l:
        df_corr = pd.concat(series_l, axis=1).dropna().pct_change().corr()
        fig_corr = px.imshow(df_corr, text_auto=".2f", template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

# ── TAB 5: PAPER TRADING & KELLY ──────────────────────────────────────────────
with tab5:
    st.subheader("📒 Gestión de Posiciones Simuladas")
    
    col_pt1, col_pt2 = st.columns([3, 2])
    with col_pt1:
        with st.form("trade_form"):
            c1, c2, c3 = st.columns(3)
            p_ent = c1.number_input("Precio Entrada", value=price)
            p_sl  = c2.number_input("Stop Loss", value=info['sl'])
            p_tp  = c3.number_input("Take Profit", value=info['tp'])
            c_cap = st.number_input("Capital ($)", value=1000.0)
            c_dir = st.selectbox("Dirección", ["LONG", "SHORT"])
            if st.form_submit_button("Guardar Operación"):
                st.session_state.paper_trades.append({
                    "Fecha": datetime.now().strftime("%d/%m/%Y"), "Ticker": ticker,
                    "Lado": c_dir, "Entrada": p_ent, "SL": p_sl, "TP": p_tp,
                    "Capital": c_cap, "Estado": "🟡 Abierta"
                })
                guardar_portfolio()
                st.success("Posición agregada.")
    
    with col_pt2:
        st.markdown("#### 📐 Cálculo de Tamaño de Posición (Kelly)")
        win_rate_input = st.slider("Win Rate Estimado %", 10, 90, 55)
        reward_ratio = abs(info['tp'] - price) / abs(price - info['sl']) if abs(price - info['sl']) > 0 else 1.5
        f_kelly = calcular_criterio_kelly(win_rate_input, reward_ratio, 1.0)
        st.metric("Criterio de Kelly (Óptimo)", f"{f_kelly}% del Capital")
        st.caption("Recomendación de gestión de riesgo: Aplicar 'Half-Kelly' (mitad de la cifra) para reducir la volatilidad.")

    if st.session_state.paper_trades:
        st.dataframe(pd.DataFrame(st.session_state.paper_trades), use_container_width=True)

# ── TAB 6: ASISTENTE IA ───────────────────────────────────────────────────────
with tab6:
    st.subheader("🤖 Consultoría Algorítmica & Tesis")
    if _ia_activa():
        if st.button("Generar Informe"):
            res = _groq("Eres un analista cuant.", f"Analiza {ticker} con estos datos:\n{_resumen_tecnico(ticker, df, info)}")
            st.markdown(res)
    else:
        st.info("Agrega tu Groq Key en el menú lateral.")

# ── TAB 7: BACKTESTING & MÉTRICAS CUANT ──────────────────────────────────────
with tab7:
    st.subheader("🧪 Simulación de Estrategia Cuantitativa")
    col1, col2 = st.columns(2)
    u_c = col1.slider("Umbral Compra", 1, 4, 2)
    u_v = col2.slider("Umbral Venta", -4, -1, -1)
    if st.button("Ejecutar Backtest"):
        bt = backtest_estrategia(df, umbral_compra=u_c, umbral_venta=u_v)
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Win Rate", f"{bt['win_rate']}%")
        m2.metric("Retorno Total", f"{bt['retorno_total']}%")
        m3.metric("Buy & Hold", f"{bt['retorno_bh']}%")
        m4.metric("Sortino Ratio", f"{bt['sortino']}")
        m5.metric("Profit Factor", f"{bt['profit_factor']}")
        m6.metric("Max Drawdown", f"{bt['max_drawdown']}%")

        if not bt['ops'].empty:
            st.dataframe(bt['ops'], use_container_width=True)

# ── TAB 8: MULTI-TIMEFRAME ────────────────────────────────────────────────────
with tab8:
    st.subheader("⏱️ Análisis Multitemporal (4H - 1D - 1W)")
    cols = st.columns(3)
    for i, tf in enumerate(["4H", "1D", "1W"]):
        d_tf = get_data_mtf(ticker, tf)
        with cols[i]:
            if not d_tf.empty:
                inf_tf = generar_senal(d_tf)
                st.metric(f"Marco {tf}", inf_tf['senal'], f"{inf_tf['puntos']} pts")

# ── TAB 9: SCREENER ───────────────────────────────────────────────────────────
with tab9:
    st.subheader("🔍 Escáner Filtro Cuantitativo")
    if st.button("Lanzar Escáner"):
        res = []
        for nm, sym in {**ACCIONES, **CRYPTOS}.items():
            d = get_data(sym, "3mo", td_key=TD_KEY)
            if not d.empty:
                inf = generar_senal(d)
                res.append({"Activo": nm, "Ticker": sym, "Señal": inf['senal'], "Puntos": inf['puntos']})
        st.dataframe(pd.DataFrame(res).sort_values("Puntos", ascending=False), use_container_width=True)

# ── TAB 10: NOTICIAS Y FUNDAMENTALES ──────────────────────────────────────────
with tab10:
    st.subheader("📰 Datos de Mercado y Sentimiento")
    data_fv = finviz_scrape(ticker)
    if data_fv["ok"]:
        st.json(data_fv["fundamentals"])

# ── TAB 11: PATRONES CHARTISTAS (VISUALIZACIÓN CORREGIDA) ─────────────────────
with tab11:
    st.subheader(f"🕯️ Reconocimiento de Patrones — {ticker}")
    st.caption("Detección algorítmica sobre la estructura de precios.")

    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        pats = detectar_patrones(df)
        if not pats:
            st.info("Sin patrones complejos detectados en la ventana actual.")
        else:
            for p in pats:
                nombre = p["patron"]
                info_p = PATTERN_INFO.get(nombre, {})
                bias   = info_p.get("bias", "neutro")
                color_p = PATTERN_BIAS_COLOR.get(bias, "#aaaaaa")
                emoji_p = info_p.get("emoji", "📌")
                conf_col = {"Alta": "#00ff88", "Media": "#ffd700", "Baja": "#ff8c00"}.get(p["confianza"], "#aaa")

                st.markdown(f"""
                <div style="background:#1a1a2e; border-left:4px solid {color_p}; border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.1rem; font-weight:bold; color:{color_p};">
                            {emoji_p} {nombre}
                        </span>
                        <span style="font-size:0.8rem; background:#0d1117; padding:2px 8px; border-radius:10px; color:{conf_col}; border:1px solid {conf_col}44;">
                            Confianza: {p['confianza']}
                        </span>
                    </div>
                    <div style="color:#e0e0e0; font-size:0.9rem; margin-top:6px;">
                        {p['desc']}
                    </div>
                    <div style="color:#888; font-size:0.8rem; margin-top:4px;">
                        📅 Período: {p['fecha_ini']} ➔ {p['fecha_fin']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if info_p.get("accion"):
                    st.caption(f"💡 **Estrategia sugerida:** {info_p['accion']}")

    with col_p2:
        st.markdown("#### 📚 Leyenda de Sesgos")
        for bias_type, color in PATTERN_BIAS_COLOR.items():
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>■ {bias_type.upper()}</span>", unsafe_allow_html=True)

# ── TAB 12: PORTAFOLIO CONSOLIDADO Y RIESGO (VaR) ────────────────────────────
with tab12:
    st.subheader("💼 Resumen Global de Portafolio & Análisis de Riesgo")
    if st.session_state.paper_trades:
        df_p = pd.DataFrame(st.session_state.paper_trades)
        cap_total = df_p['Capital'].sum()
        
        # Cálculo de VaR
        retornos_p = df['Close'].pct_change() * 100
        var_pct, var_usd = calcular_var_95(retornos_p, cap_total)
        
        p1, p2, p3 = st.columns(3)
        p1.metric("Total Invertido", f"${cap_total:,.2f}")
        p2.metric("Value at Risk (VaR 95% Diario)", f"${var_usd:,.2f}", f"-{var_pct}% del portafolio")
        p3.metric("Posiciones Activas", len(df_p))
        
        st.dataframe(df_p, use_container_width=True)
        
        st.markdown("#### 📄 Exportación de Reporte de Portafolio")
        reporte_str = f"REPORTE INSTITUCIONAL QUANTUMSHIELD PRO\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nCapital Total: ${cap_total:,.2f}\nVaR 95%: ${var_usd:,.2f} ({var_pct}%)\nPosiciones:\n" + df_p.to_string()
        st.download_button("Descargar Reporte (TXT / PDF Ready)", data=reporte_str, file_name=f"Reporte_QuantumShield_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain")
    else:
        st.info("No hay activos en el portafolio.")
