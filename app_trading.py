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
from datetime import datetime, timezone
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Dependencias Opcionales
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    ML_OK = True
except ImportError:
    ML_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA (SIN SIDEBAR — FULL WORKSTATION MODE)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QSP // QUANT WORKSTATION",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# MOTOR GRÁFICO CSS — ESTÉTICA KOYFIN / BLOOMBERG WORKSTATION
# ══════════════════════════════════════════════════════════════════════════════
WORKSTATION_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-main: #06080c;
    --bg-panel: #0b0e14;
    --bg-header: #0f131c;
    --border: #1c222d;
    --border-bright: #2a3241;
    --text-primary: #e2e8f0;
    --text-muted: #64748b;
    --accent-amber: #f59e0b;
    --accent-blue: #3b82f6;
    --bull: #10b981;
    --bear: #f43f5e;
}

html, body, .stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Eliminar cromado de Streamlit y márgenes desperdiciados */
#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}
section[data-testid="stSidebar"] {
    display: none !important;
}
.block-container {
    padding: 0.4rem 0.8rem 1rem 0.8rem !important;
    max-width: 100% !important;
}

/* Reducir espacio vertical entre bloques de Streamlit */
div[data-testid="stVerticalBlock"] > div {
    gap: 0.35rem !important;
}

/* Barra superior institucional */
.ws-topbar {
    background: var(--bg-header);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent-amber);
    padding: 6px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-bottom: 4px;
}
.ws-brand {
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 10px;
}
.ws-tag {
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-amber);
    border: 1px solid rgba(245, 158, 11, 0.4);
    padding: 1px 6px;
    font-size: 0.65rem;
    font-weight: 600;
}

/* Grilla densa de telemetría (reemplaza st.metric) */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    background: var(--border);
    gap: 1px;
    border: 1px solid var(--border);
    margin-bottom: 6px;
}
.kpi-cell {
    background: var(--bg-panel);
    padding: 7px 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.63rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 2px;
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.98rem;
    font-weight: 700;
    color: #f8fafc;
    font-variant-numeric: tabular-nums;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-variant-numeric: tabular-nums;
    margin-top: 1px;
}

/* Paneles laterales y tablas */
.ws-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    padding: 10px 12px;
    height: 100%;
}
.ws-panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
}

/* Filas de datos tipo Orderbook / Tearsheet */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid #131822;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
}
.data-row-label {
    color: var(--text-muted);
}
.data-row-val {
    color: var(--text-primary);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

/* Tabs estilo Terminal */
div[data-baseweb="tab-list"] {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    padding: 0 6px !important;
    gap: 2px !important;
}
button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 7px 12px !important;
    margin: 0 !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-amber) !important;
    border-bottom: 2px solid var(--accent-amber) !important;
    background-color: rgba(245, 158, 11, 0.06) !important;
}

/* Inputs compactos en la barra de comandos */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 2px !important;
    min-height: 32px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
}
div[data-testid="stPopover"] > button {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 2px !important;
    min-height: 32px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    color: var(--text-primary) !important;
    width: 100%;
}

/* Botones de acción */
.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    border: 1px solid var(--border-bright) !important;
    background-color: #131822 !important;
    color: var(--text-primary) !important;
    min-height: 32px !important;
    padding: 2px 12px !important;
}
.stButton > button:hover {
    border-color: var(--accent-amber) !important;
    color: var(--accent-amber) !important;
}
.stButton > button[kind="primary"] {
    background-color: var(--accent-amber) !important;
    color: #06080c !important;
    border-color: var(--accent-amber) !important;
    font-weight: 700 !important;
}
</style>
"""
st.markdown(WORKSTATION_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSO DE ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════
ACCIONES = {
    "NVDA — NVIDIA Corp":          "NVDA",
    "AAPL — Apple Inc":            "AAPL",
    "MSFT — Microsoft Corp":       "MSFT",
    "TSLA — Tesla Inc":            "TSLA",
    "AMZN — Amazon.com":           "AMZN",
    "META — Meta Platforms":       "META",
    "GOOGL — Alphabet Inc":        "GOOGL",
    "JPM — JPMorgan Chase":        "JPM",
    "SPY — S&P 500 ETF":           "SPY",
    "IPSA — S&P CLX IPSA":         "^IPSA",
    "BRK-B — Berkshire Hathaway":  "BRK-B",
    "KO — Coca-Cola Co":           "KO",
    "AXP — American Express":      "AXP",
    "BAC — Bank of America":       "BAC",
    "CVX — Chevron Corp":          "CVX",
    "OXY — Occidental Petrol":     "OXY",
    "MCO — Moody's Corp":          "MCO",
    "V — Visa Inc":                "V",
    "MA — Mastercard Inc":         "MA",
    "JNJ — Johnson & Johnson":     "JNJ",
    "PG — Procter & Gamble":       "PG",
    "WMT — Walmart Inc":           "WMT",
    "HD — Home Depot":             "HD",
    "MCD — McDonald's Corp":       "MCD",
    "PEP — PepsiCo Inc":           "PEP",
    "UNH — UnitedHealth":          "UNH",
}

CRYPTOS = {
    "BTC — Bitcoin Spot":    "BTC-USD",
    "ETH — Ethereum Spot":   "ETH-USD",
    "SOL — Solana Spot":     "SOL-USD",
    "BNB — Binance Coin":    "BNB-USD",
    "XRP — Ripple Network":  "XRP-USD",
    "ADA — Cardano Spot":    "ADA-USD",
    "AVAX — Avalanche Spot": "AVAX-USD",
    "DOGE — Dogecoin Spot":  "DOGE-USD",
    "LINK — Chainlink Spot": "LINK-USD",
    "DOT — Polkadot Spot":   "DOT-USD",
}

BINANCE_SYMBOLS = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT", "XRP-USD": "XRPUSDT", "ADA-USD": "ADAUSDT",
    "AVAX-USD": "AVAXUSDT", "DOGE-USD": "DOGEUSDT", "LINK-USD": "LINKUSDT", "DOT-USD": "DOTUSDT",
}

BENCHMARK_ACCIONES = "SPY"
BENCHMARK_CRYPTO   = "BTC-USD"
PORTFOLIO_DIR      = "qsp_data"
os.makedirs(PORTFOLIO_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCIA POR SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
def _session_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else "default"
    except Exception:
        return "default"

def _portfolio_path() -> str:
    return os.path.join(PORTFOLIO_DIR, f"portfolio_{_session_id()}.json")

def _cargar_portfolio() -> dict:
    path = _portfolio_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "paper_trades":   data.get("paper_trades", []),
                    "notas":          data.get("notas", {}),
                    "signal_history": data.get("signal_history", {}),
                }
        except Exception:
            pass
    return {"paper_trades": [], "notas": {}, "signal_history": {}}

def guardar_portfolio() -> bool:
    try:
        data = {
            "paper_trades":   st.session_state.get("paper_trades", []),
            "notas":          st.session_state.get("notas", {}),
            "signal_history": st.session_state.get("signal_history", {}),
            "guardado_en":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        with open(_portfolio_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except OSError:
        return False

_datos_guardados = _cargar_portfolio()
for key, val in [
    ("paper_trades", _datos_guardados["paper_trades"]),
    ("notas", _datos_guardados["notas"]),
    ("signal_history", _datos_guardados["signal_history"]),
    ("chat_history", {}),
    ("last_bt", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = val

def enviar_alerta_telegram(mensaje: str) -> bool:
    token = st.session_state.get("telegram_token", "").strip()
    chat_id = st.session_state.get("telegram_chat_id", "").strip()
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"},
            timeout=5
        )
        return resp.ok
    except requests.RequestException:
        return False

# ══════════════════════════════════════════════════════════════════════════════
# MOTOR MATEMÁTICO E INDICADORES (SUAVIZADO DE WILDER)
# ══════════════════════════════════════════════════════════════════════════════
def calcular_rsi(serie: pd.Series, periodo: int = 14) -> pd.Series:
    delta    = serie.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1/periodo, min_periods=periodo, adjust=False).mean()
    perdida  = (-delta.clip(upper=0)).ewm(alpha=1/periodo, min_periods=periodo, adjust=False).mean()
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
    return tr.ewm(alpha=1/periodo, min_periods=periodo, adjust=False).mean()

def calcular_adx(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    alpha    = 1 / periodo
    atr_s    = tr.ewm(alpha=alpha, min_periods=periodo, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=alpha, min_periods=periodo, adjust=False).mean() / atr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=periodo, adjust=False).mean() / atr_s.replace(0, np.nan))
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=alpha, min_periods=periodo, adjust=False).mean()

def calcular_stoch_rsi(serie: pd.Series, periodo_rsi=14, periodo_stoch=14, suavizado_k=3, suavizado_d=3):
    rsi     = calcular_rsi(serie, periodo_rsi)
    min_rsi = rsi.rolling(periodo_stoch).min()
    max_rsi = rsi.rolling(periodo_stoch).max()
    rango   = (max_rsi - min_rsi).replace(0, np.nan)
    k       = ((rsi - min_rsi) / rango * 100).rolling(suavizado_k).mean()
    d       = k.rolling(suavizado_d).mean()
    return k, d

def calcular_vwap(df: pd.DataFrame) -> pd.Series:
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return pd.Series(np.nan, index=df.index)
    pt = (df['High'] + df['Low'] + df['Close']) / 3
    return (pt * df['Volume']).cumsum() / df['Volume'].cumsum()

def calcular_adr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    return (((df['High'] / df['Low']) - 1) * 100).rolling(window=periodo).mean()

def calcular_sortino_ratio(retornos: pd.Series, rf_rate: float = 0.0, factor_anual: float = 252.0) -> float:
    r = retornos.dropna()
    if len(r) < 2: return 0.0
    ret_med = r.mean() - rf_rate
    neg = r[r < 0]
    if len(neg) == 0: return 99.0 if ret_med > 0 else 0.0
    dd = np.sqrt((neg ** 2).mean())
    return round(float((ret_med / dd) * np.sqrt(factor_anual)), 2) if dd > 0 else 0.0

def calcular_var_cvar_95(retornos: pd.Series, capital: float = 10000.0) -> tuple:
    r = retornos.dropna()
    if len(r) < 5: return 0.0, 0.0, 0.0, 0.0
    var_pct = np.percentile(r, 5)
    cola = r[r <= var_pct]
    cvar_pct = cola.mean() if len(cola) > 0 else var_pct
    return (
        round(float(abs(var_pct)), 2),
        round(float(abs(capital * (var_pct / 100.0))), 2),
        round(float(abs(cvar_pct)), 2),
        round(float(abs(capital * (cvar_pct / 100.0))), 2),
    )

def calcular_profit_factor(df_ops: pd.DataFrame) -> float:
    if df_ops.empty or 'Retorno %' not in df_ops.columns: return 0.0
    g = df_ops[df_ops['Retorno %'] > 0]['Retorno %'].sum()
    p = abs(df_ops[df_ops['Retorno %'] < 0]['Retorno %'].sum())
    return round(float(g / p), 2) if p > 0 else (round(float(g), 2) if g > 0 else 0.0)

def calcular_criterio_kelly(win_rate_pct: float, avg_win: float, avg_loss: float) -> float:
    if win_rate_pct <= 0 or avg_loss <= 0 or avg_win <= 0: return 0.0
    p = win_rate_pct / 100.0
    b = abs(avg_win / avg_loss)
    return round(float(max(0.0, ((b * p - (1.0 - p)) / b) * 100.0)), 2)

def calcular_volume_profile(df: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    if 'Volume' not in df.columns or df['Volume'].sum() == 0: return pd.DataFrame()
    p_min, p_max = float(df['Low'].min()), float(df['High'].max())
    rangos = np.linspace(p_min, p_max, bins + 1)
    vol_nivel = np.zeros(bins)
    for _, row in df.iterrows():
        il = max(0, min(np.searchsorted(rangos, float(row['Low']), side='left'), bins - 1))
        ih = max(0, min(np.searchsorted(rangos, float(row['High']), side='right'), bins))
        vol_nivel[il:ih] += float(row['Volume']) / max(1, ih - il)
    p_mid = (rangos[:-1] + rangos[1:]) / 2
    poc_idx = np.argmax(vol_nivel)
    return pd.DataFrame({'precio': p_mid, 'volumen': vol_nivel, 'poc': [i == poc_idx for i in range(bins)]})

def detectar_soportes_resistencias(df: pd.DataFrame, orden: int = 10):
    close = df['Close'].values
    imax = argrelextrema(close, np.greater, order=orden)[0]
    imin = argrelextrema(close, np.less, order=orden)[0]
    return [(df.index[i], close[i]) for i in imax][-5:], [(df.index[i], close[i]) for i in imin][-5:]

def volumen_anomalo(df: pd.DataFrame, factor: float = 2.0) -> pd.Series:
    if 'Volume' not in df.columns: return pd.Series(False, index=df.index)
    return df['Volume'] > (factor * df['Volume'].rolling(20).mean())

# ══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE SEÑALES MULTIFACTOR
# ══════════════════════════════════════════════════════════════════════════════
def calcular_puntos(fila) -> tuple:
    puntos, razones = 0, []
    if 'VWAP' in fila.index and not pd.isna(fila['VWAP']):
        if float(fila['Close']) > float(fila['VWAP']):
            puntos += 1; razones.append(("VWAP Spread", "+1", "#10b981"))
        else:
            puntos -= 1; razones.append(("VWAP Spread", "-1", "#f43f5e"))

    if 'SMA200' in fila.index and not pd.isna(fila['SMA200']):
        if float(fila['Close']) > float(fila['SMA200']):
            puntos += 1; razones.append(("Macro > SMA200", "+1", "#10b981"))
        else:
            puntos -= 1; razones.append(("Macro < SMA200", "-1", "#f43f5e"))

    if float(fila['Close']) > float(fila['EMA50']):
        puntos += 1; razones.append(("Trend > EMA50", "+1", "#10b981"))
    else:
        puntos -= 1; razones.append(("Trend < EMA50", "-1", "#f43f5e"))

    rsi_val = float(fila['RSI'])
    if rsi_val < 30:
        puntos += 2; razones.append((f"RSI Oversold ({rsi_val:.1f})", "+2", "#10b981"))
    elif rsi_val > 70:
        puntos -= 2; razones.append((f"RSI Overbought ({rsi_val:.1f})", "-2", "#f43f5e"))
    elif 40 <= rsi_val <= 60:
        puntos += 1; razones.append((f"RSI Neutral ({rsi_val:.1f})", "+1", "#3b82f6"))

    if float(fila['MACD']) > float(fila['MACD_SIGNAL']):
        puntos += 1; razones.append(("MACD Momentum", "+1", "#10b981"))
    else:
        puntos -= 1; razones.append(("MACD Momentum", "-1", "#f43f5e"))

    if float(fila['Close']) < float(fila['BB_LOWER']):
        puntos += 1; razones.append(("Bollinger -2σ", "+1", "#10b981"))
    elif float(fila['Close']) > float(fila['BB_UPPER']):
        puntos -= 1; razones.append(("Bollinger +2σ", "-1", "#f43f5e"))

    adx_val = float(fila['ADX']) if 'ADX' in fila.index and not pd.isna(fila['ADX']) else 0
    if adx_val >= 25:
        razones.append((f"ADX Trend ({adx_val:.1f})", "STRONG", "#10b981"))
    elif adx_val >= 15:
        razones.append((f"ADX Regime ({adx_val:.1f})", "MOD", "#f59e0b"))
    else:
        puntos = max(-1, min(1, puntos))
        razones.append((f"ADX Chop ({adx_val:.1f})", "FLAT", "#64748b"))

    return puntos, razones, rsi_val, adx_val

def generar_senal(df: pd.DataFrame) -> dict:
    ultima = df.iloc[-1]
    puntos, razones, rsi_val, adx_val = calcular_puntos(ultima)
    if   puntos >= 4:  senal, color = "STRONG LONG",  "#10b981"
    elif puntos >= 2:  senal, color = "ACCUMULATE",   "#3b82f6"
    elif puntos <= -4: senal, color = "STRONG SHORT", "#f43f5e"
    elif puntos <= -2: senal, color = "DISTRIBUTE",   "#f59e0b"
    else:              senal, color = "NEUTRAL",      "#94a3b8"

    atr_val = float(ultima['ATR']) if not pd.isna(ultima['ATR']) else 0
    precio  = float(ultima['Close'])
    return {
        "senal": senal, "color": color, "puntos": puntos, "razones": razones,
        "sl": precio - 2 * atr_val, "tp": precio + 3 * atr_val,
        "atr": atr_val, "rsi": rsi_val, "adx": adx_val,
    }

@st.cache_data(ttl=300, show_spinner=False)
def predecir_probabilidad_ml(df: pd.DataFrame) -> dict:
    if not ML_OK or len(df) < 60:
        return {"prob": None, "acc": None, "importances": {}, "msg": "Insufficient sample"}
    df_ml = df.copy()
    df_ml['Retorno_Futuro'] = (df_ml['Close'].shift(-3) - df_ml['Close']) / df_ml['Close']
    df_ml['Target'] = (df_ml['Retorno_Futuro'] > 0.01).astype(int)
    df_ml['EMA_DIFF'] = (df_ml['Close'] - df_ml['EMA50']) / df_ml['EMA50']
    df_ml['BB_WIDTH'] = (df_ml['BB_UPPER'] - df_ml['BB_LOWER']) / df_ml['BB_MID'].replace(0, np.nan)
    df_ml['RET_5D']   = df_ml['Close'].pct_change(5)

    features = ['RSI', 'ADX', 'MACD_HIST', 'ATR', 'EMA_DIFF', 'BB_WIDTH', 'RET_5D']
    df_valid = df_ml.dropna(subset=features).copy()
    if len(df_valid) < 45:
        return {"prob": None, "acc": None, "importances": {}, "msg": "Insufficient clean rows"}

    X_actual = df_valid[features].values[-1:]
    df_hist  = df_valid.iloc[:-3].dropna(subset=['Target'])
    if len(df_hist) < 35:
        return {"prob": None, "acc": None, "importances": {}, "msg": "Short history"}

    X, y = df_hist[features].values, df_hist['Target'].values
    split_idx = max(20, int(len(X) * 0.8))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s  = scaler.transform(X[split_idx:]) if len(X[split_idx:]) > 0 else X_train_s

    clf = RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=3, random_state=42)
    clf.fit(X_train_s, y[:split_idx])
    acc = round(float(accuracy_score(y[split_idx:], clf.predict(X_test_s)) * 100.0), 1) if len(y[split_idx:]) > 0 else 50.0

    X_all_s = scaler.fit_transform(X)
    clf.fit(X_all_s, y)
    X_act_s = scaler.transform(X_actual)
    clases = list(clf.classes_)
    prob = clf.predict_proba(X_act_s)[0][clases.index(1)] * 100.0 if 1 in clases else 0.0
    importances = dict(zip(features, np.round(clf.feature_importances_ * 100, 1)))
    return {"prob": round(float(prob), 1), "acc": acc, "importances": importances, "msg": f"OOS Acc: {acc}%"}

# ══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE
# ══════════════════════════════════════════════════════════════════════════════
PERIOD_DAYS       = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 732}
PERIOD_OUTPUT     = {"1mo": 35, "3mo": 95, "6mo": 190, "1y": 370, "2y": 740}
TD_BASE_URL       = "https://api.twelvedata.com/time_series"
BINANCE_BASE      = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVALS = {"1mo": ("1d", 35), "3mo": ("1d", 95), "6mo": ("1d", 190), "1y": ("1d", 370), "2y": ("1d", 740)}
BINANCE_MTF       = {"4H": ("4h", 180), "1D": ("1d", 200), "1W": ("1w", 104)}
YF_MTF            = {"4H": ("4h", "60d"), "1D": ("1d", "1y"), "1W": ("1wk", "2y")}

def _binance_download(ticker: str, period: str) -> pd.DataFrame:
    sym = BINANCE_SYMBOLS.get(ticker)
    if not sym: return pd.DataFrame()
    interval, limit = BINANCE_INTERVALS.get(period, ("1d", 190))
    try:
        resp = requests.get(BINANCE_BASE, params={"symbol": sym, "interval": interval, "limit": limit}, timeout=10)
        if not resp.ok: return pd.DataFrame()
        raw = resp.json()
        if not isinstance(raw, list) or not raw: return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["open_time","Open","High","Low","Close","Volume","close_time","qav","trades","tbbav","tbqav","ignore"])
        for c in ["Open","High","Low","Close","Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        df.index.name = None
        return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    except requests.RequestException:
        return pd.DataFrame()

def obtener_order_book_binance(ticker: str) -> tuple:
    sym = BINANCE_SYMBOLS.get(ticker)
    if not sym: return pd.DataFrame(), pd.DataFrame()
    try:
        resp = requests.get(f"https://api.binance.com/api/v3/depth?symbol={sym}&limit=15", timeout=5)
        if not resp.ok: return pd.DataFrame(), pd.DataFrame()
        data = resp.json()
        bids = pd.DataFrame(data['bids'], columns=['Precio', 'Cantidad'], dtype=float)
        asks = pd.DataFrame(data['asks'], columns=['Precio', 'Cantidad'], dtype=float)
        bids['Acumulado'] = bids['Cantidad'].cumsum()
        asks['Acumulado'] = asks['Cantidad'].cumsum()
        return bids, asks
    except requests.RequestException:
        return pd.DataFrame(), pd.DataFrame()

def _td_download(ticker: str, period: str, td_key: str = "") -> pd.DataFrame:
    key = (td_key or "").strip()
    if not key: return pd.DataFrame()
    t = ticker.replace("-USD", "/USD").replace("-", "/")
    sym = t[1:] if t.startswith("^") else t
    params = {"symbol": sym, "interval": "1day", "outputsize": PERIOD_OUTPUT.get(period, 190), "apikey": key, "format": "JSON", "order": "ASC"}
    try:
        resp = requests.get(TD_BASE_URL, params=params, timeout=12)
        if not resp.ok: return pd.DataFrame()
        data = resp.json()
        if data.get("status") == "error" or "values" not in data: return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.index.name = None
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
        return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
    except requests.RequestException:
        return pd.DataFrame()

def _yf_fallback(ticker: str, period: str) -> pd.DataFrame:
    dias = PERIOD_DAYS.get(period, 183)
    today = datetime.today()
    start = (today - pd.Timedelta(days=dias)).strftime("%Y-%m-%d")
    end   = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for intento in range(2):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, timeout=12)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                return df[[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]]
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

def _agregar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    df["EMA20"]  = close.ewm(span=20, adjust=False).mean()
    df["EMA50"]  = close.ewm(span=50, adjust=False).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["VWAP"]   = calcular_vwap(df)
    df["ADR_PCT"]= calcular_adr(df, 14)
    df["RSI"]    = calcular_rsi(close)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = calcular_macd(close)
    df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"]   = calcular_bollinger(close)
    df["ATR"]    = calcular_atr(df)
    df["ADX"]    = calcular_adx(df)
    df["STOCH_K"], df["STOCH_D"] = calcular_stoch_rsi(close)
    if "Volume" in df.columns:
        df["VOL_ANOMALO"] = volumen_anomalo(df)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def get_data(ticker: str, period: str, td_key: str = "") -> pd.DataFrame:
    df = pd.DataFrame()
    if ticker in BINANCE_SYMBOLS:
        df = _binance_download(ticker, period)
    if df.empty:
        df = _td_download(ticker, period, td_key)
    if df.empty:
        df = _yf_fallback(ticker, period)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]].copy()
    df.dropna(subset=["Close"], inplace=True)
    return _agregar_indicadores(df)

@st.cache_data(ttl=300, show_spinner=False)
def get_close_only(ticker: str, period: str, td_key: str = "") -> pd.Series:
    df = get_data(ticker, period, td_key)
    return df["Close"].rename(ticker) if not df.empty else pd.Series(dtype=float, name=ticker)

@st.cache_data(ttl=120, show_spinner=False)
def get_data_mtf(ticker: str, tf: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if ticker in BINANCE_SYMBOLS:
        sym = BINANCE_SYMBOLS[ticker]
        interval, limit = BINANCE_MTF[tf]
        try:
            resp = requests.get(BINANCE_BASE, params={"symbol": sym, "interval": interval, "limit": limit}, timeout=8)
            if resp.ok:
                raw = resp.json()
                if isinstance(raw, list) and raw:
                    df = pd.DataFrame(raw, columns=["open_time","Open","High","Low","Close","Volume","close_time","qav","trades","tbbav","tbqav","ignore"])
                    for col in ["Open","High","Low","Close","Volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.index = pd.to_datetime(df["open_time"], unit="ms")
                    df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
        except requests.RequestException:
            pass
    if df.empty:
        yf_interval, yf_period = YF_MTF[tf]
        try:
            raw = yf.download(ticker, period=yf_period, interval=yf_interval, progress=False, auto_adjust=True)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [col[0] for col in raw.columns]
                df = raw[[c for c in ["Open","High","Low","Close","Volume"] if c in raw.columns]].dropna(subset=["Close"])
        except Exception:
            pass
    if df.empty: return pd.DataFrame()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return _agregar_indicadores(df)

@st.cache_data(ttl=300, show_spinner=False)
def obtener_radar_completo(period: str, td_key: str) -> pd.DataFrame:
    todos = {**ACCIONES, **CRYPTOS}
    filas = []
    def _fetch(item):
        nm, sym = item
        d = get_data(sym, period, td_key=td_key)
        if d.empty: return None
        inf = generar_senal(d)
        p = float(d['Close'].iloc[-1])
        pv = float(d['Close'].iloc[-2]) if len(d) > 1 else p
        return {
            "Instrument": nm, "Ticker": sym,
            "Class": "CRYPTO" if sym in CRYPTOS.values() else "EQUITY",
            "Last Px": round(p, 4),
            "1D Chg %": round(((p / pv) - 1) * 100, 2),
            "RSI(14)": round(float(d['RSI'].iloc[-1]), 1),
            "ADX(14)": round(float(d['ADX'].iloc[-1]), 1) if 'ADX' in d.columns else 0,
            "Stance": inf['senal'], "Score": inf['puntos']
        }
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(_fetch, i) for i in todos.items()]):
            r = f.result()
            if r: filas.append(r)
    return pd.DataFrame(filas).sort_values("Score", ascending=False) if filas else pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def obtener_matriz_correlacion(tickers_tuple: tuple, period: str, td_key: str) -> pd.DataFrame:
    series_l = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(get_close_only, s, period, td_key) for s in tickers_tuple]):
            s = f.result()
            if not s.empty: series_l.append(s)
    return pd.concat(series_l, axis=1).dropna().pct_change().corr() if series_l else pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def optimizar_markowitz(tickers_tuple: tuple, period: str, td_key: str, rf_rate: float = 0.04, n_portfolios: int = 2500) -> dict:
    series_dict = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(get_close_only, sym, period, td_key): sym for sym in tickers_tuple}
        for f in as_completed(futures):
            s = f.result()
            if not s.empty: series_dict[futures[f]] = s
    valid_tickers = [t for t in tickers_tuple if t in series_dict]
    if len(valid_tickers) < 2:
        return {"ok": False, "msg": "Select at least 2 valid instruments."}
    df_prices = pd.DataFrame({t: series_dict[t] for t in valid_tickers}).dropna()
    if len(df_prices) < 20:
        return {"ok": False, "msg": "Insufficient overlapping price history."}
    ret = df_prices.pct_change().dropna()
    mean_ret = ret.mean().values * 252.0
    cov_mat  = ret.cov().values * 252.0
    rng = np.random.default_rng(42)
    weights = rng.random((n_portfolios, len(valid_tickers)))
    weights /= weights.sum(axis=1)[:, np.newaxis]
    port_ret = np.dot(weights, mean_ret) * 100.0
    port_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_mat, weights)) * 100.0
    port_sr  = np.where(port_vol > 0, (port_ret - rf_rate * 100.0) / port_vol, 0.0)
    ims, imv = int(np.argmax(port_sr)), int(np.argmin(port_vol))
    return {
        "ok": True, "tickers": valid_tickers,
        "nube": pd.DataFrame({"Volatilidad %": np.round(port_vol, 2), "Retorno Esperado %": np.round(port_ret, 2), "Sharpe": np.round(port_sr, 2)}),
        "mean_ret": mean_ret * 100.0, "cov_mat": cov_mat * 10000.0,
        "max_sharpe": {"weights": dict(zip(valid_tickers, np.round(weights[ims]*100, 2))), "retorno": round(float(port_ret[ims]), 2), "vol": round(float(port_vol[ims]), 2), "sharpe": round(float(port_sr[ims]), 2)},
        "min_vol":    {"weights": dict(zip(valid_tickers, np.round(weights[imv]*100, 2))), "retorno": round(float(port_ret[imv]), 2), "vol": round(float(port_vol[imv]), 2), "sharpe": round(float(port_sr[imv]), 2)},
    }

# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTER, FINVIZ, PATRONES & GROQ
# ══════════════════════════════════════════════════════════════════════════════
def backtest_estrategia(df: pd.DataFrame, umbral_compra: int = 2, umbral_venta: int = -2,
                        comision_pct: float = 0.1, usar_sl_tp: bool = True, es_crypto: bool = False) -> dict:
    resultados, en_posicion = [], False
    precio_entrada, fecha_entrada, senal_entrada, idx_entrada = 0.0, None, "", -1
    sl_nivel, tp_nivel = None, None
    pts = [calcular_puntos(df.iloc[i])[0] if i >= 50 else 0 for i in range(len(df))]
    df = df.copy()
    df['_puntos'] = pts

    def _cerrar(p_out, f_out, motivo):
        rb = (p_out - precio_entrada) / precio_entrada * 100
        r  = rb - (comision_pct * 2)
        d  = max(1, (f_out - fecha_entrada).days) if hasattr(f_out - fecha_entrada, 'days') else 1
        resultados.append({
            "Entry Date": fecha_entrada.strftime("%Y-%m-%d") if hasattr(fecha_entrada, 'strftime') else str(fecha_entrada),
            "Exit Date": f_out.strftime("%Y-%m-%d") if hasattr(f_out, 'strftime') else str(f_out),
            "Fecha_Salida_DT": f_out, "Entry Px": round(precio_entrada, 4), "Exit Px": round(p_out, 4),
            "Gross %": round(rb, 2), "Retorno %": round(r, 2), "Days": d,
            "Score": senal_entrada, "Trigger": motivo, "Result": "WIN" if r > 0 else "LOSS"
        })

    for i in range(51, len(df) - 1):
        ph, psig, fsig = df['_puntos'].iloc[i], float(df['Open'].iloc[i + 1]), df.index[i + 1]
        if not en_posicion:
            if ph >= umbral_compra:
                en_posicion, precio_entrada, fecha_entrada, senal_entrada, idx_entrada = True, psig, fsig, f"{ph:+d}", i + 1
                sl_nivel = tp_nivel = None
                if usar_sl_tp and 'ATR' in df.columns:
                    atr_p = df['ATR'].iloc[i]
                    if not pd.isna(atr_p) and float(atr_p) > 0:
                        sl_nivel, tp_nivel = precio_entrada - 2.0 * float(atr_p), precio_entrada + 3.0 * float(atr_p)
        else:
            if sl_nivel is not None and i >= idx_entrada:
                lo, hi = float(df['Low'].iloc[i]), float(df['High'].iloc[i])
                if lo <= sl_nivel: _cerrar(sl_nivel, df.index[i], "STOP_LOSS"); en_posicion = False; continue
                if hi >= tp_nivel: _cerrar(tp_nivel, df.index[i], "TAKE_PROFIT"); en_posicion = False; continue
            if ph <= umbral_venta or i == len(df) - 2:
                _cerrar(psig, fsig, "REVERSAL" if ph <= umbral_venta else "END_SAMPLE"); en_posicion = False

    if not resultados:
        return {"ops": pd.DataFrame(), "equity_df": pd.DataFrame(), "win_rate": 0, "retorno_total": 0, "retorno_bh": 0, "n_ops": 0, "sharpe": 0, "sortino": 0, "calmar": 0, "profit_factor": 0, "max_drawdown": 0}

    df_ops = pd.DataFrame(resultados)
    n_ops = len(df_ops)
    win_rate = round((df_ops['Retorno %'] > 0).sum() / n_ops * 100, 1)
    ret_bh = round((float(df['Close'].iloc[-1]) / float(df['Close'].iloc[50]) - 1) * 100, 2)
    retornos = df_ops['Retorno %']
    dias_tot = max(1, (df.index[-1] - df.index[50]).days)
    t_anio = max(1.0, n_ops * ((365.0 if es_crypto else 252.0) / dias_tot))
    sharpe = round(float(retornos.mean() / retornos.std() * np.sqrt(t_anio)), 2) if len(retornos) > 1 and retornos.std() > 0 else 0.0
    sortino = calcular_sortino_ratio(retornos, 0.0, t_anio)
    pfactor = calcular_profit_factor(df_ops)
    equity = (1 + retornos / 100).cumprod()
    ret_comp = round(float(equity.iloc[-1] - 1) * 100, 2)
    max_dd = round(float(((equity / equity.cummax() - 1) * 100).min()), 2)
    calmar = round(float(ret_comp / abs(max_dd)), 2) if max_dd < 0 else 0.0

    eq_df = pd.DataFrame(index=df.index[50:])
    eq_df["Buy & Hold"] = (df['Close'].iloc[50:] / float(df['Close'].iloc[50])) * 100.0
    eq_s = pd.Series(index=df.index[50:], dtype=float)
    eq_s.iloc[0] = 100.0
    for idx_op, row_op in df_ops.iterrows():
        if row_op["Fecha_Salida_DT"] in eq_s.index:
            eq_s.loc[row_op["Fecha_Salida_DT"]] = float(equity.iloc[idx_op]) * 100.0
    eq_df["Estrategia"] = eq_s.ffill().fillna(100.0)
    eq_df["Drawdown %"] = (eq_df["Estrategia"] / eq_df["Estrategia"].cummax() - 1.0) * 100.0
    return {
        "ops": df_ops.drop(columns=["Fecha_Salida_DT"]), "equity_df": eq_df,
        "win_rate": win_rate, "retorno_total": ret_comp, "retorno_bh": ret_bh, "n_ops": n_ops,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "profit_factor": pfactor, "max_drawdown": max_dd
    }

@st.cache_data(ttl=600, show_spinner=False)
def finviz_scrape(ticker: str) -> dict:
    res = {"news": [], "fundamentals": {}, "ok": False}
    if not BS4_OK or "-USD" in ticker or ticker.startswith("^"): return res
    try:
        resp = requests.get(f"https://finviz.com/quote.ashx?t={ticker}&p=d", headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if not resp.ok: return res
        soup = BeautifulSoup(resp.text, "html.parser")
        tabla = soup.find("table", class_="snapshot-table2")
        if tabla:
            celdas = tabla.find_all("td")
            for i in range(0, len(celdas) - 1, 2):
                res["fundamentals"][celdas[i].get_text(strip=True)] = celdas[i+1].get_text(strip=True)
        t_news = soup.find("table", id="news-table")
        if t_news:
            f_act = ""
            for fila in t_news.find_all("tr")[:15]:
                c = fila.find_all("td")
                if len(c) < 2: continue
                ft = c[0].get_text(strip=True)
                if any(ch.isalpha() for ch in ft):
                    p = ft.split()
                    f_act = p[0] if len(p) > 1 else f_act
                    hr = p[-1] if len(p) > 1 else ft
                else: hr = ft
                a = c[1].find("a")
                sp = c[1].find("span")
                if a:
                    res["news"].append({"fecha": f"{f_act} {hr}".strip(), "titulo": a.get_text(strip=True), "url": a.get("href", "#"), "fuente": sp.get_text(strip=True) if sp else ""})
        res["ok"] = bool(res["fundamentals"] or res["news"])
        return res
    except Exception:
        return res

def _parse_finviz_num(val: str):
    if not val or val == "-": return None
    v = val.strip().replace("%", "")
    mult = {"B": 1e9, "M": 1e6, "K": 1e3}.get(v[-1], 1.0) if v else 1.0
    if mult != 1.0: v = v[:-1]
    try: return float(v) * mult
    except Exception: return None

def analizar_buffett(f: dict) -> dict:
    pe, roe, deuda, margen = _parse_finviz_num(f.get("P/E", "-")), _parse_finviz_num(f.get("ROE", "-")), _parse_finviz_num(f.get("Debt/Eq", "-")), _parse_finviz_num(f.get("Profit Margin", "-"))
    s = (30 if pe and 0 < pe < 15 else (20 if pe and pe < 25 else (10 if pe and pe < 35 else 0))) + \
        (30 if roe and roe >= 20 else (20 if roe and roe >= 15 else (10 if roe and roe >= 10 else 0))) + \
        (20 if deuda is not None and deuda <= 0.3 else (15 if deuda is not None and deuda <= 0.5 else 0)) + \
        (20 if margen and margen >= 20 else (10 if margen and margen >= 10 else 0))
    return {"P/E": pe, "ROE %": roe, "Deuda/Patrim.": deuda, "Margen Neto %": margen, "Score": s, "Tier": "TIER 1 // PRIME" if s >= 70 else ("TIER 2 // CORE" if s >= 40 else "TIER 3 // SPEC")}

def actualizar_paper_trades(td_key: str):
    cambios = False
    for t in st.session_state.paper_trades:
        if "OPEN" in str(t.get("Estado", "")) or str(t.get("Estado", "")).startswith("🟡"):
            t["Estado"] = "OPEN"
            s = get_close_only(t["Ticker"], "1mo", td_key=td_key)
            if s.empty: continue
            p_now = float(s.iloc[-1])
            ent, sl, tp, lado = t["Entrada"], t["SL"], t["TP"], t.get("Lado", "LONG")
            pnl = ((p_now - ent) / ent * 100) if lado == "LONG" else ((ent - p_now) / ent * 100)
            if (lado == "LONG" and p_now <= sl) or (lado == "SHORT" and p_now >= sl): t["Estado"] = "CLOSED // SL"
            elif (lado == "LONG" and p_now >= tp) or (lado == "SHORT" and p_now <= tp): t["Estado"] = "CLOSED // TP"
            t["Precio actual"], t["PnL % actual"], t["PnL USD"] = round(p_now, 4), round(pnl, 2), round(t.get("Capital", 0.0) * (pnl / 100.0), 2)
            cambios = True
    if cambios: guardar_portfolio()

def _groq(messages: list) -> str:
    k = st.session_state.get("groq_api_key", "").strip()
    if not k: return "AUTH_ERR // Configure Groq API Key in SYSTEM CONFIG."
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Content-Type": "application/json", "Authorization": f"Bearer {k}"}, json={"model": "openai/gpt-oss-120b", "max_tokens": 1100, "messages": messages}, timeout=25)
        return r.json()["choices"][0]["message"]["content"] if r.ok else f"HTTP_ERR // {r.status_code}"
    except Exception as e:
        return f"NET_ERR // {e}"

# ══════════════════════════════════════════════════════════════════════════════
# 1. TOP INSTITUTIONAL STATUS BAR
# ══════════════════════════════════════════════════════════════════════════════
utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="ws-topbar">
    <div class="ws-brand">
        <span>QUANTUMSHIELD <span style="color:#f59e0b;">PRO</span></span>
        <span class="ws-tag">INSTITUTIONAL DESK v4.0</span>
        <span style="color:#64748b;">// MULTI-ASSET EXECUTION & RISK ENGINE</span>
    </div>
    <div style="color:#94a3b8; display:flex; gap:18px;">
        <span>FEED: <strong style="color:#10b981;">ACTIVE</strong></span>
        <span>ENGINE: <strong style="color:#e2e8f0;">WILDER-RF</strong></span>
        <span>{utc_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. COMMAND BAR INTEGRADA (REEMPLAZA EL SIDEBAR DE STREAMLIT)
# ══════════════════════════════════════════════════════════════════════════════
cmd1, cmd2, cmd3, cmd4, cmd5 = st.columns([1.3, 2.2, 1.0, 1.3, 1.2])

with cmd1:
    clase_activo = st.selectbox("ASSET CLASS", ["EQUITIES", "CRYPTO", "CUSTOM"], label_visibility="collapsed")

with cmd2:
    if clase_activo == "EQUITIES":
        sel_nombre = st.selectbox("INSTRUMENT", list(ACCIONES.keys()), label_visibility="collapsed")
        ticker = ACCIONES[sel_nombre]
        benchmark = BENCHMARK_ACCIONES
    elif clase_activo == "CRYPTO":
        sel_nombre = st.selectbox("INSTRUMENT", list(CRYPTOS.keys()), label_visibility="collapsed")
        ticker = CRYPTOS[sel_nombre]
        benchmark = BENCHMARK_CRYPTO
    else:
        ticker = st.text_input("SYMBOL", value="NVDA", label_visibility="collapsed").upper().strip() or "NVDA"
        benchmark = BENCHMARK_CRYPTO if "-USD" in ticker else BENCHMARK_ACCIONES

with cmd3:
    period = st.selectbox("HORIZON", ["1mo", "3mo", "6mo", "1y", "2y"], index=2, label_visibility="collapsed")

with cmd4:
    with st.popover("⚙️ CHART OVERLAYS"):
        mostrar_vwap    = st.checkbox("VWAP Institutional", value=True)
        mostrar_ema     = st.checkbox("EMA 20 / EMA 50", value=True)
        mostrar_sma     = st.checkbox("SMA 200 Macro", value=False)
        mostrar_bb      = st.checkbox("Bollinger Bands (2σ)", value=True)
        mostrar_sr      = st.checkbox("Horizontal S/R Levels", value=True)
        mostrar_volumen = st.checkbox("Volume & Spikes", value=True)
        mostrar_stoch   = st.checkbox("Stochastic RSI", value=False)
        mostrar_vp      = st.checkbox("Volume Profile (POC)", value=False)

with cmd5:
    with st.popover("🔑 API & TELEMETRY"):
        td_in = st.text_input("Twelve Data Key", type="password")
        gq_in = st.text_input("Groq LLM Key", type="password")
        tg_tk = st.text_input("Telegram Bot Token", type="password")
        tg_id = st.text_input("Telegram Chat ID")
        if td_in: st.session_state["td_api_key"] = td_in
        elif "td_api_key" not in st.session_state:
            try: st.session_state["td_api_key"] = st.secrets.get("TD_API_KEY", "")
            except Exception: st.session_state["td_api_key"] = ""
        if gq_in: st.session_state["groq_api_key"] = gq_in
        elif "groq_api_key" not in st.session_state:
            try: st.session_state["groq_api_key"] = st.secrets.get("GROQ_API_KEY", "")
            except Exception: st.session_state["groq_api_key"] = ""
        if tg_tk: st.session_state["telegram_token"] = tg_tk
        if tg_id: st.session_state["telegram_chat_id"] = tg_id

TD_KEY = st.session_state.get("td_api_key", "")
df = get_data(ticker, period, td_key=TD_KEY)
if df.empty:
    st.error(f"FEED_ERROR // Unable to retrieve market data for {ticker}.")
    st.stop()

info   = generar_senal(df)
last   = df.iloc[-1]
prev   = df.iloc[-2] if len(df) > 1 else last
price  = float(last['Close'])
chg    = ((price / float(prev['Close'])) - 1) * 100 if float(prev['Close']) > 0 else 0.0
ml_res = predecir_probabilidad_ml(df)
vwap_v = float(last.get('VWAP', price))
vwap_s = ((price / vwap_v) - 1) * 100 if vwap_v > 0 else 0.0
chg_col = "#10b981" if chg >= 0 else "#f43f5e"
ml_str  = f"{ml_res['prob']}%" if ml_res['prob'] is not None else "N/A"
ml_sub  = ml_res['msg']

# ══════════════════════════════════════════════════════════════════════════════
# 3. DENSE 8-CELL TELEMETRY RIBBON (CERO TARJETAS NATIVAS DE STREAMLIT)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="kpi-strip">
    <div class="kpi-cell" style="border-left: 3px solid {chg_col};">
        <span class="kpi-label">{ticker} // SPOT PX</span>
        <span class="kpi-val">${price:,.4f}</span>
        <span class="kpi-sub" style="color:{chg_col};">{chg:+.2f}% (1D)</span>
    </div>
    <div class="kpi-cell" style="border-left: 3px solid {info['color']};">
        <span class="kpi-label">QUANT STANCE</span>
        <span class="kpi-val" style="color:{info['color']};">{info['senal']}</span>
        <span class="kpi-sub" style="color:#94a3b8;">CONVICTION: {info['puntos']:+d} PTS</span>
    </div>
    <div class="kpi-cell">
        <span class="kpi-label">VWAP SPREAD</span>
        <span class="kpi-val">${vwap_v:,.4f}</span>
        <span class="kpi-sub" style="color:{'#10b981' if vwap_s>=0 else '#f43f5e'};">{vwap_s:+.2f}% VS SPOT</span>
    </div>
    <div class="kpi-cell">
        <span class="kpi-label">RSI (14W) // ADX</span>
        <span class="kpi-val">{info['rsi']:.1f} <span style="color:#64748b;font-weight:400;">/</span> {info['adx']:.1f}</span>
        <span class="kpi-sub" style="color:#94a3b8;">{'TRENDING' if info['adx']>=25 else 'MEAN-REVERTING'}</span>
    </div>
    <div class="kpi-cell">
        <span class="kpi-label">VOLATILITY (ATR / ADR)</span>
        <span class="kpi-val">${info['atr']:,.3f}</span>
        <span class="kpi-sub" style="color:#94a3b8;">ADR(14): {last.get('ADR_PCT', 0):.2f}%</span>
    </div>
    <div class="kpi-cell">
        <span class="kpi-label">VOL STOP (-2 ATR)</span>
        <span class="kpi-val" style="color:#f43f5e;">${info['sl']:,.4f}</span>
        <span class="kpi-sub" style="color:#64748b;">RISK: {((info['sl']/price - 1)*100):.2f}%</span>
    </div>
    <div class="kpi-cell">
        <span class="kpi-label">TARGET (+3 ATR)</span>
        <span class="kpi-val" style="color:#10b981;">${info['tp']:,.4f}</span>
        <span class="kpi-sub" style="color:#64748b;">UPSIDE: {((info['tp']/price - 1)*100):+.2f}%</span>
    </div>
    <div class="kpi-cell">
        <span class="kpi-label">RF ML (+3D PROB)</span>
        <span class="kpi-val" style="color:#f59e0b;">{ml_str}</span>
        <span class="kpi-sub" style="color:#94a3b8;">{ml_sub}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. WORKSTATION NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════
t_work, t_scan, t_bt, t_port, t_corr, t_fund, t_ai = st.tabs([
    "01 // TRADING WORKSTATION",
    "02 // ALPHA SCANNER & SCREENER",
    "03 // BACKTEST ENGINE",
    "04 // OMS & MARKOWITZ RISK",
    "05 // RELVAL & CORRELATION",
    "06 // FUNDAMENTALS & WIRE",
    "07 // QUANT RESEARCH AI"
])

# ── TAB 01: TRADING WORKSTATION (SPLIT LAYOUT 72% CHART / 28% TEARSHEET) ─────
with t_work:
    col_chart, col_desk = st.columns([2.7, 1.0])

    with col_chart:
        rows_n = 3 + (1 if 'Volume' in df.columns and mostrar_volumen else 0) + (1 if mostrar_stoch else 0)
        row_h  = [0.58, 0.14, 0.14, 0.14][:rows_n] if rows_n == 4 else ([0.52, 0.12, 0.12, 0.12, 0.12] if rows_n == 5 else [0.68, 0.16, 0.16])
        fig = make_subplots(rows=rows_n, cols=1, shared_xaxes=True, vertical_spacing=0.015, row_heights=row_h)

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#10b981', decreasing_line_color='#f43f5e',
            increasing_fillcolor='#10b981', decreasing_fillcolor='#f43f5e', name="PRICE"
        ), row=1, col=1)

        if mostrar_vwap and 'VWAP' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color="#3b82f6", width=1.4), name="VWAP"), row=1, col=1)
        if mostrar_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color="#f59e0b", width=1.1), name="EMA 20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color="#a855f7", width=1.1), name="EMA 50"), row=1, col=1)
        if mostrar_sma and 'SMA200' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color="#e2e8f0", width=1.3, dash="dash"), name="SMA 200"), row=1, col=1)
        if mostrar_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color="rgba(148,163,184,0.25)", width=1, dash="dot"), name="BB +2σ"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color="rgba(148,163,184,0.25)", width=1, dash="dot"), fill='tonexty', fillcolor="rgba(148,163,184,0.03)", name="BB -2σ"), row=1, col=1)
        if mostrar_sr:
            res_l, sop_l = detectar_soportes_resistencias(df)
            for _, v in res_l: fig.add_hline(y=v, line_dash="dot", line_color="rgba(244,63,94,0.35)", row=1, col=1)
            for _, v in sop_l: fig.add_hline(y=v, line_dash="dot", line_color="rgba(16,185,129,0.35)", row=1, col=1)

        curr_r = 2
        if 'Volume' in df.columns and mostrar_volumen:
            c_vol = np.where(df.get('VOL_ANOMALO', False), '#f59e0b', np.where(df['Close'] >= df['Open'], 'rgba(16,185,129,0.45)', 'rgba(244,63,94,0.45)'))
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="VOL", marker_color=c_vol), row=curr_r, col=1)
            curr_r += 1

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="#a855f7", width=1.2), name="RSI"), row=curr_r, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(244,63,94,0.3)", row=curr_r, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(16,185,129,0.3)", row=curr_r, col=1)
        curr_r += 1

        c_macd = np.where(df['MACD_HIST'] >= 0, 'rgba(16,185,129,0.6)', 'rgba(244,63,94,0.6)')
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name="MACD", marker_color=c_macd), row=curr_r, col=1)

        if mostrar_stoch:
            curr_r += 1
            fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_K'], line=dict(color="#10b981", width=1), name="STOCH K"), row=curr_r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_D'], line=dict(color="#f43f5e", width=1), name="STOCH D"), row=curr_r, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#06080c",
            font=dict(family="JetBrains Mono, monospace", size=10, color="#64748b"),
            height=640, margin=dict(l=8, r=52, t=24, b=8),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=10)),
            xaxis_rangeslider_visible=False, hovermode="x unified"
        )
        fig.update_xaxes(showgrid=True, gridcolor="#111621", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#111621", zeroline=False, side="right")
        st.plotly_chart(fig, use_container_width=True)

    with col_desk:
        # Panel Derecho: Factor Attribution & Niveles Clave
        filas_factores = "".join([
            f"<div class='data-row'><span class='data-row-label'>{lbl}</span><span class='data-row-val' style='color:{col};'>{val}</span></div>"
            for lbl, val, col in info["razones"]
        ])
        fibs = calcular_volume_profile(df)
        poc_px = float(fibs.loc[fibs['poc'], 'precio'].iloc[0]) if not fibs.empty and fibs['poc'].any() else price

        st.markdown(f"""
        <div class="ws-panel">
            <div class="ws-panel-title"><span>QUANT FACTOR DECOMPOSITION</span><span>SCORE: {info['puntos']:+d}</span></div>
            {filas_factores}
            <div class="ws-panel-title" style="margin-top:14px;"><span>KEY INSTITUTIONAL LEVELS</span><span>SPOT: ${price:,.2f}</span></div>
            <div class="data-row"><span class="data-row-label">Bollinger Upper (+2σ)</span><span class="data-row-val">${float(last['BB_UPPER']):,.4f}</span></div>
            <div class="data-row"><span class="data-row-label">EMA 20 (Short Term)</span><span class="data-row-val">${float(last['EMA20']):,.4f}</span></div>
            <div class="data-row"><span class="data-row-label">EMA 50 (Medium Term)</span><span class="data-row-val">${float(last['EMA50']):,.4f}</span></div>
            <div class="data-row"><span class="data-row-label">SMA 200 (Macro Trend)</span><span class="data-row-val">${float(last.get('SMA200', 0)):,.4f}</span></div>
            <div class="data-row"><span class="data-row-label">Volume POC Level</span><span class="data-row-val" style="color:#f59e0b;">${poc_px:,.4f}</span></div>
            <div class="data-row"><span class="data-row-label">Bollinger Lower (-2σ)</span><span class="data-row-val">${float(last['BB_LOWER']):,.4f}</span></div>
        </div>
        """, unsafe_allow_html=True)

        if ml_res["importances"]:
            st.markdown("<div class='ws-panel' style='margin-top:6px;'><div class='ws-panel-title'><span>RF FEATURE IMPORTANCE</span><span>WALK-FORWARD</span></div>", unsafe_allow_html=True)
            for k_f, v_f in sorted(ml_res["importances"].items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"""
                <div style="margin-bottom:5px; font-family:'JetBrains Mono',monospace; font-size:0.68rem;">
                    <div style="display:flex; justify-content:space-between; color:#94a3b8;"><span>{k_f}</span><span>{v_f}%</span></div>
                    <div style="background:#131822; height:4px; width:100%; margin-top:2px;">
                        <div style="background:#f59e0b; height:4px; width:{min(100, v_f*2.2)}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 02: ALPHA SCANNER & SCREENER ──────────────────────────────────────────
with t_scan:
    sc1, sc2, sc3, sc4, sc5 = st.columns([1.2, 1.5, 1.5, 1.5, 1.2])
    f_cls = sc1.selectbox("CLASS FILTER", ["ALL", "EQUITY", "CRYPTO"])
    f_rsi = sc2.slider("RSI BAND", 0, 100, (0, 100))
    f_adx = sc3.slider("MIN ADX", 0, 60, 10)
    f_pts = sc4.slider("MIN SCORE", -6, 6, -6)
    if sc5.button("RUN UNIVERSE SCAN", type="primary", use_container_width=True):
        with st.spinner("SCANNING 36 INSTRUMENTS..."):
            st.session_state["radar_df"] = obtener_radar_completo(period, TD_KEY)

    df_r = st.session_state.get("radar_df", pd.DataFrame())
    if not df_r.empty:
        m = (df_r["RSI(14)"] >= f_rsi[0]) & (df_r["RSI(14)"] <= f_rsi[1]) & (df_r["ADX(14)"] >= f_adx) & (df_r["Score"] >= f_pts)
        if f_cls != "ALL": m = m & (df_r["Class"] == f_cls)
        st.dataframe(df_r[m], use_container_width=True, height=540)
    else:
        st.info("READY // Click 'RUN UNIVERSE SCAN' to compute factor scores across all Equities & Crypto.")

# ── TAB 03: BACKTEST ENGINE ───────────────────────────────────────────────────
with t_bt:
    b1, b2, b3, b4, b5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    u_c     = b1.slider("LONG ENTRY SCORE", 1, 6, 2)
    u_v     = b2.slider("EXIT SCORE", -6, -1, -2)
    com_pct = b3.number_input("FEE / SLIPPAGE (%)", value=0.10, step=0.05)
    usar_sl = b4.checkbox("ATR VOLATILITY STOPS", value=True)
    if b5.button("RUN SIMULATION", type="primary", use_container_width=True):
        st.session_state["last_bt"][ticker] = backtest_estrategia(df, u_c, u_v, com_pct, usar_sl, "-USD" in ticker)

    bt = st.session_state.get("last_bt", {}).get(ticker)
    if bt and not bt["equity_df"].empty:
        st.markdown(f"""
        <div class="kpi-strip" style="grid-template-columns: repeat(7, 1fr);">
            <div class="kpi-cell"><span class="kpi-label">WIN RATE</span><span class="kpi-val">{bt['win_rate']}%</span><span class="kpi-sub">N = {bt['n_ops']} TRADES</span></div>
            <div class="kpi-cell"><span class="kpi-label">STRATEGY RETURN</span><span class="kpi-val" style="color:{'#10b981' if bt['retorno_total']>=0 else '#f43f5e'};">{bt['retorno_total']:+.2f}%</span><span class="kpi-sub">NET OF FEES</span></div>
            <div class="kpi-cell"><span class="kpi-label">BUY & HOLD</span><span class="kpi-val">{bt['retorno_bh']:+.2f}%</span><span class="kpi-sub">BENCHMARK</span></div>
            <div class="kpi-cell"><span class="kpi-label">SHARPE RATIO</span><span class="kpi-val">{bt['sharpe']}</span><span class="kpi-sub">ANNUALIZED</span></div>
            <div class="kpi-cell"><span class="kpi-label">SORTINO RATIO</span><span class="kpi-val">{bt['sortino']}</span><span class="kpi-sub">DOWNSIDE RISK</span></div>
            <div class="kpi-cell"><span class="kpi-label">PROFIT FACTOR</span><span class="kpi-val">{bt['profit_factor']}</span><span class="kpi-sub">GROSS W/L</span></div>
            <div class="kpi-cell"><span class="kpi-label">MAX DRAWDOWN</span><span class="kpi-val" style="color:#f43f5e;">{bt['max_drawdown']}%</span><span class="kpi-sub">CALMAR: {bt['calmar']}</span></div>
        </div>
        """, unsafe_allow_html=True)

        eq_df = bt["equity_df"]
        fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig_bt.add_trace(go.Scatter(x=eq_df.index, y=eq_df["Estrategia"], name="QUANT STRATEGY", line=dict(color="#10b981", width=1.8)), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=eq_df.index, y=eq_df["Buy & Hold"], name="BUY & HOLD", line=dict(color="#3b82f6", width=1.2, dash="dot")), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=eq_df.index, y=eq_df["Drawdown %"], name="DRAWDOWN %", fill="tozeroy", line=dict(color="#f43f5e", width=1)), row=2, col=1)
        fig_bt.update_layout(template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#06080c", font=dict(family="JetBrains Mono, monospace", size=10), height=440, margin=dict(l=10, r=40, t=15, b=10))
        st.plotly_chart(fig_bt, use_container_width=True)
        st.dataframe(bt["ops"], use_container_width=True, height=240)

# ── TAB 04: OMS & MARKOWITZ RISK ──────────────────────────────────────────────
with t_port:
    actualizar_paper_trades(TD_KEY)
    om1, om2 = st.columns([1.2, 2.5])

    with om1:
        st.markdown("<div class='ws-panel'><div class='ws-panel-title'><span>ORDER ENTRY (OMS)</span><span>KELLY SIZING</span></div></div>", unsafe_allow_html=True)
        with st.form("oms_form"):
            f_side = st.selectbox("SIDE", ["LONG", "SHORT"])
            f_ent  = st.number_input("ENTRY PX ($)", value=price, format="%.4f")
            f_sl   = st.number_input("STOP LOSS ($)", value=info['sl'], format="%.4f")
            f_tp   = st.number_input("TAKE PROFIT ($)", value=info['tp'], format="%.4f")
            f_cap  = st.number_input("NOTIONAL ($)", value=10000.0, step=1000.0)
            if st.form_submit_button("TRANSMIT ORDER", use_container_width=True):
                st.session_state.paper_trades.append({
                    "Fecha": datetime.now().strftime("%Y-%m-%d"), "Ticker": ticker, "Lado": f_side,
                    "Entrada": f_ent, "SL": f_sl, "TP": f_tp, "Capital": f_cap, "Estado": "OPEN"
                })
                guardar_portfolio()
                st.rerun()

    with om2:
        df_p = pd.DataFrame(st.session_state.paper_trades) if st.session_state.paper_trades else pd.DataFrame()
        df_ab = df_p[df_p['Estado'].astype(str).str.contains('OPEN')] if not df_p.empty else pd.DataFrame()
        cap_act = float(df_ab['Capital'].sum()) if not df_ab.empty else 0.0

        ret_pond = []
        for _, fila in df_ab.iterrows():
            s = get_close_only(fila['Ticker'], period, td_key=TD_KEY)
            if not s.empty and cap_act > 0:
                sg = -1.0 if fila.get('Lado', 'LONG') == 'SHORT' else 1.0
                ret_pond.append(s.pct_change().dropna() * 100.0 * sg * (float(fila['Capital']) / cap_act))
        ret_port = pd.concat(ret_pond, axis=1).sum(axis=1) if ret_pond else pd.Series(dtype=float)
        vpct, vusd, cvpct, cvusd = calcular_var_cvar_95(ret_port, cap_act)

        st.markdown(f"""
        <div class="kpi-strip" style="grid-template-columns: repeat(4, 1fr);">
            <div class="kpi-cell"><span class="kpi-label">ACTIVE EXPOSURE</span><span class="kpi-val">${cap_act:,.2f}</span><span class="kpi-sub">OPEN POSITIONS: {len(df_ab)}</span></div>
            <div class="kpi-cell"><span class="kpi-label">1D 95% VaR</span><span class="kpi-val" style="color:#f59e0b;">${vusd:,.2f}</span><span class="kpi-sub">-{vpct}% OF BOOK</span></div>
            <div class="kpi-cell"><span class="kpi-label">1D 95% CVaR (TAIL)</span><span class="kpi-val" style="color:#f43f5e;">${cvusd:,.2f}</span><span class="kpi-sub">-{cvpct}% SHORTFALL</span></div>
            <div class="kpi-cell"><span class="kpi-label">OPEN PNL AVG</span><span class="kpi-val">{df_ab['PnL % actual'].mean() if not df_ab.empty and 'PnL % actual' in df_ab.columns else 0.0:+.2f}%</span><span class="kpi-sub">MARK-TO-MARKET</span></div>
        </div>
        """, unsafe_allow_html=True)

        if not df_p.empty:
            st.dataframe(df_p, use_container_width=True, height=210)
            if st.button("PURGE OMS BLOTTER"):
                st.session_state.paper_trades = []
                guardar_portfolio()
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    mk1, mk2 = st.columns([3, 1])
    univ_all = {**ACCIONES, **CRYPTOS}
    sel_mk = mk1.multiselect("MARKOWITZ EFFICIENT FRONTIER BASKET:", list(univ_all.values()), default=["NVDA", "AAPL", "MSFT", "SPY", "BTC-USD"])
    if mk2.button("SOLVE FRONTIER (N=2,500)", type="primary", use_container_width=True) and len(sel_mk) >= 2:
        st.session_state["mk_res"] = optimizar_markowitz(tuple(sel_mk), period, TD_KEY)

    mk_r = st.session_state.get("mk_res")
    if mk_r and mk_r.get("ok"):
        ms, mv, n_df = mk_r["max_sharpe"], mk_r["min_vol"], mk_r["nube"]
        c_ef1, c_ef2 = st.columns([1.8, 1.2])
        with c_ef1:
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=n_df["Volatilidad %"], y=n_df["Retorno Esperado %"], mode="markers", marker=dict(size=4, color=n_df["Sharpe"], colorscale="Cividis", showscale=True), name="SIMULATED"))
            fig_ef.add_trace(go.Scatter(x=[ms["vol"]], y=[ms["retorno"]], mode="markers+text", text=["MAX SHARPE"], textposition="top center", marker=dict(symbol="star", size=14, color="#10b981"), name="MAX SHARPE"))
            fig_ef.add_trace(go.Scatter(x=[mv["vol"]], y=[mv["retorno"]], mode="markers+text", text=["MIN VAR"], textposition="bottom center", marker=dict(symbol="diamond", size=12, color="#3b82f6"), name="MIN VAR"))
            fig_ef.update_layout(template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#06080c", font=dict(family="JetBrains Mono, monospace", size=10), height=340, margin=dict(l=10, r=10, t=15, b=10), xaxis_title="Annualized Volatility (%)", yaxis_title="Expected Return (%)")
            st.plotly_chart(fig_ef, use_container_width=True)
        with c_ef2:
            reb_rows = [{"Ticker": t, "Max Sharpe Wt %": ms["weights"][t], "Min Var Wt %": mv["weights"][t]} for t in mk_r["tickers"]]
            st.dataframe(pd.DataFrame(reb_rows), use_container_width=True, height=340)

# ── TAB 05: RELVAL & CORRELATION ──────────────────────────────────────────────
with t_corr:
    rc1, rc2 = st.columns(2)
    with rc1:
        s_a = get_close_only(ticker, period, td_key=TD_KEY)
        s_b = get_close_only(benchmark, period, td_key=TD_KEY)
        if not s_a.empty and not s_b.empty:
            df_c = pd.DataFrame({ticker: s_a, benchmark: s_b}).dropna()
            df_n = df_c / df_c.iloc[0] * 100
            fig_rv = px.line(df_n, template="plotly_dark", color_discrete_sequence=["#f59e0b", "#3b82f6"])
            fig_rv.update_layout(paper_bgcolor="#0b0e14", plot_bgcolor="#06080c", font=dict(family="JetBrains Mono, monospace", size=10), height=460, title=f"RELATIVE PERFORMANCE // {ticker} VS {benchmark} (BASE 100)")
            st.plotly_chart(fig_rv, use_container_width=True)
    with rc2:
        corr_syms = ("NVDA", "AAPL", "MSFT", "TSLA", "SPY", "BTC-USD", "ETH-USD", "^IPSA")
        df_cm = obtener_matriz_correlacion(corr_syms, period, TD_KEY)
        if not df_cm.empty:
            fig_cm = px.imshow(df_cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_dark")
            fig_cm.update_layout(paper_bgcolor="#0b0e14", plot_bgcolor="#06080c", font=dict(family="JetBrains Mono, monospace", size=10), height=460, title="CROSS-ASSET CORRELATION MATRIX")
            st.plotly_chart(fig_cm, use_container_width=True)

# ── TAB 06: FUNDAMENTALS & WIRE ───────────────────────────────────────────────
with t_fund:
    fv = finviz_scrape(ticker)
    if fv["ok"]:
        bf = analizar_buffett(fv["fundamentals"])
        st.markdown(f"""
        <div class="kpi-strip" style="grid-template-columns: repeat(5, 1fr);">
            <div class="kpi-cell"><span class="kpi-label">QUALITY SCORE</span><span class="kpi-val" style="color:#f59e0b;">{bf['Score']}/100</span><span class="kpi-sub">{bf['Tier']}</span></div>
            <div class="kpi-cell"><span class="kpi-label">P/E MULTIPLE</span><span class="kpi-val">{bf['P/E'] if bf['P/E'] else 'N/A'}</span><span class="kpi-sub">VALUATION</span></div>
            <div class="kpi-cell"><span class="kpi-label">RETURN ON EQUITY</span><span class="kpi-val">{bf['ROE %'] if bf['ROE %'] else 'N/A'}%</span><span class="kpi-sub">PROFITABILITY</span></div>
            <div class="kpi-cell"><span class="kpi-label">DEBT / EQUITY</span><span class="kpi-val">{bf['Deuda/Patrim.'] if bf['Deuda/Patrim.'] is not None else 'N/A'}x</span><span class="kpi-sub">LEVERAGE</span></div>
            <div class="kpi-cell"><span class="kpi-label">NET MARGIN</span><span class="kpi-val">{bf['Margen Neto %'] if bf['Margen Neto %'] else 'N/A'}%</span><span class="kpi-sub">EFFICIENCY</span></div>
        </div>
        """, unsafe_allow_html=True)
        for n in fv["news"]:
            st.markdown(f"<div class='data-row'><span style='color:#f59e0b;'>{n['fecha']}</span><a href='{n['url']}' target='_blank' style='color:#e2e8f0;text-decoration:none;'>{n['titulo']}</a><span style='color:#64748b;'>[{n['fuente']}]</span></div>", unsafe_allow_html=True)
    else:
        st.info("FUNDAMENTALS_UNAVAILABLE // Only US-listed equities support Finviz fundamental extraction.")

# ── TAB 07: QUANT RESEARCH AI ─────────────────────────────────────────────────
with t_ai:
    sys_p = f"You are a Senior Quantitative Portfolio Manager at a Wall Street fund. Respond in Spanish with concise, institutional analysis. Current telemetry for {ticker}: Spot=${price:,.4f}, Stance={info['senal']} ({info['puntos']} pts), RSI={info['rsi']:.1f}, ADX={info['adx']:.1f}, VWAP=${vwap_v:,.4f}."
    h_chat = st.session_state.chat_history.setdefault(ticker, [])
    if st.button("GENERATE INSTITUTIONAL INVESTMENT MEMO", type="primary"):
        h_chat.append({"role": "user", "content": f"Elabora un Investment Memo ejecutivo sobre {ticker} cubriendo estructura de mercado, escenarios Long/Short con niveles exactos y dimensionamiento de riesgo."})
        h_chat.append({"role": "assistant", "content": _groq([{"role": "system", "content": sys_p}] + h_chat)})
    for m in h_chat:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    q_u = st.chat_input(f"Ask Quant Desk about {ticker}...")
    if q_u:
        h_chat.append({"role": "user", "content": q_u})
        with st.chat_message("user"): st.markdown(q_u)
        with st.chat_message("assistant"):
            r_ai = _groq([{"role": "system", "content": sys_p}] + h_chat)
            st.markdown(r_ai)
            h_chat.append({"role": "assistant", "content": r_ai})
