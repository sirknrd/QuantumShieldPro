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
    st.session_state.chat_history = {}   # {ticker: [{"role":..,"content":..}]}
if "analisis_cache" not in st.session_state:
    st.session_state.analisis_cache = {}  # {ticker+period: texto}
if "noticias_cache" not in st.session_state:
    st.session_state.noticias_cache = {}  # {ticker: texto}

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
    precio  = float(ultima['Close'])
    return {
        "senal": senal, "color": color, "puntos": puntos,
        "razones": razones,
        "sl": precio - 2 * atr_val,
        "tp": precio + 3 * atr_val,
        "atr": atr_val, "rsi": rsi_val,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def _yf_download_safe(ticker: str, period: str, intentos: int = 3) -> pd.DataFrame:
    """Descarga con retry y manejo explícito de YFRateLimitError."""
    for intento in range(intentos):
        try:
            df = yf.download(ticker, period=period, progress=False,
                             auto_adjust=True, timeout=15)
            if not df.empty:
                return df
            # Fallback a Ticker.history
            df2 = yf.Ticker(ticker).history(period=period)
            if not df2.empty:
                return df2
            return pd.DataFrame()
        except Exception as e:
            nombre_error = type(e).__name__
            if "RateLimit" in nombre_error or "TooManyRequests" in nombre_error:
                if intento < intentos - 1:
                    espera = 2 ** (intento + 1)   # 2s, 4s, 8s
                    time.sleep(espera)
                    continue
                # Agotamos los reintentos
                raise
            # Cualquier otro error: re-lanzar
            raise
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_data(ticker: str, period: str) -> pd.DataFrame:
    df = _yf_download_safe(ticker, period)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]].copy()
    df.dropna(subset=['Close'], inplace=True)

    close = df['Close']
    df['EMA20']                                    = close.ewm(span=20, adjust=False).mean()
    df['EMA50']                                    = close.ewm(span=50, adjust=False).mean()
    df['RSI']                                      = calcular_rsi(close)
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = calcular_macd(close)
    df['BB_UPPER'], df['BB_MID'], df['BB_LOWER']   = calcular_bollinger(close)
    df['ATR']                                      = calcular_atr(df)
    if 'Volume' in df.columns:
        df['VOL_ANOMALO'] = volumen_anomalo(df)
    return df

@st.cache_data(ttl=300)
def get_close_only(ticker: str, period: str) -> pd.Series:
    try:
        df = _yf_download_safe(ticker, period)
    except Exception:
        return pd.Series(dtype=float, name=ticker)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df['Close'].rename(ticker)


# ══════════════════════════════════════════════════════════════════════════════
# IA — Claude API helpers
# ══════════════════════════════════════════════════════════════════════════════

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-haiku-4-5-20251001"   # rápido y económico para trading
CLAUDE_HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}

def _claude(system: str, user: str, max_tokens: int = 1000) -> str:
    """Llamada simple a Claude. Retorna texto o mensaje de error."""
    if "x-api-key" not in CLAUDE_HEADERS or not CLAUDE_HEADERS["x-api-key"]:
        return "❌ API key no configurada."
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        resp = requests.post(CLAUDE_API_URL, headers=CLAUDE_HEADERS,
                             json=payload, timeout=30)
        if not resp.ok:
            try:
                detalle = resp.json().get("error", {}).get("message", resp.text[:200])
            except Exception:
                detalle = resp.text[:200]
            if resp.status_code == 401:
                return "❌ API key inválida o sin permisos. Verifica en console.anthropic.com"
            if resp.status_code == 400:
                return f"❌ Solicitud inválida: {detalle}"
            return f"❌ Error HTTP {resp.status_code}: {detalle}"
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        return f"❌ Error de conexión: {e}"

def _claude_stream(system: str, messages: list, max_tokens: int = 1000):
    """Streaming para el chat. Yield de chunks de texto."""
    if "x-api-key" not in CLAUDE_HEADERS or not CLAUDE_HEADERS["x-api-key"]:
        yield "❌ API key no configurada."
        return
    # Filtrar mensajes: solo role user/assistant, content string
    mensajes_limpios = [
        {"role": m["role"], "content": str(m["content"])}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not mensajes_limpios:
        yield "❌ Sin mensajes para enviar."
        return
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": mensajes_limpios,
        "stream": True,
    }
    try:
        with requests.post(CLAUDE_API_URL, headers=CLAUDE_HEADERS,
                           json=payload, timeout=60, stream=True) as resp:
            if not resp.ok:
                try:
                    detalle = resp.json().get("error", {}).get("message", resp.text[:200])
                except Exception:
                    detalle = resp.text[:200]
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
                        if chunk.get("type") == "content_block_delta":
                            yield chunk["delta"].get("text", "")
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"\n❌ Error de conexión: {e}"

def _resumen_tecnico(ticker: str, df: pd.DataFrame, info: dict) -> str:
    """Construye un resumen compacto del estado técnico para pasarle a Claude."""
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
        ticker    = st.text_input("Ticker personalizado", value="NVDA").upper().strip()
        benchmark = BENCHMARK_ACCIONES
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
    st.subheader("🤖 IA")
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Necesaria para las funciones de IA. Obtén la tuya en console.anthropic.com",
    )
    if api_key_input:
        CLAUDE_HEADERS["x-api-key"] = api_key_input
        st.success("API key cargada ✓", icon="🔑")
    else:
        # Intentar desde st.secrets
        try:
            CLAUDE_HEADERS["x-api-key"] = st.secrets["ANTHROPIC_API_KEY"]
            st.success("API key desde secrets ✓", icon="🔑")
        except Exception:
            st.warning("Sin API key — IA desactivada", icon="⚠️")

    st.markdown("---")
    st.caption("Datos vía Yahoo Finance · Caché 5 min")

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
                "⚠️ **Yahoo Finance está limitando las peticiones** desde este servidor. "
                "Esto es un límite temporal de la IP de Streamlit Cloud, no un error del código. "
                "Espera 30-60 segundos y recarga la página, o prueba con otro ticker.",
                icon="🚦",
            )
        else:
            st.error(f"Error al descargar datos para **{ticker}**: {nombre_e}")
        st.stop()

if df.empty:
    st.error(
        f"No se encontraron datos para **{ticker}**. "
        "Verifica que el ticker sea válido (ej: AAPL, BTC-USD, ^IPSA)."
    )
    st.stop()

info  = generar_senal(df)
last  = df.iloc[-1]
prev  = df.iloc[-2] if len(df) > 1 else last
price = float(last['Close'])
chg   = ((price / float(prev['Close'])) - 1) * 100 if float(prev['Close']) > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Análisis",
    "🌐 Señales del Mercado",
    "📉 Comparación Relativa",
    "🔗 Correlación",
    "📒 Paper Trading & Notas",
    "🤖 Asistente IA",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANÁLISIS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Precio",      f"${price:,.4f}",       f"{chg:+.2f}%")
    k2.metric("RSI (14)",    f"{info['rsi']:.1f}")
    k3.metric("ATR (14)",    f"${info['atr']:,.4f}")
    k4.metric("Stop Loss",   f"${info['sl']:,.4f}")
    k5.metric("Take Profit", f"${info['tp']:,.4f}")

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

    # Gráfico — con volumen: 4 filas (velas|vol|RSI|MACD), sin volumen: 3 filas (velas|RSI|MACD)
    tiene_vol   = mostrar_volumen and 'Volume' in df.columns
    rows_n      = 4 if tiene_vol else 3
    row_heights = [0.48, 0.13, 0.20, 0.19] if tiene_vol else [0.55, 0.22, 0.23]

    fig = make_subplots(
        rows=rows_n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Velas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Precio",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ), row=1, col=1)

    # Bollinger
    if mostrar_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], name="BB Sup",
            line=dict(color="rgba(100,180,255,.6)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_MID'], name="BB Med",
            line=dict(color="rgba(100,180,255,.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], name="BB Inf",
            line=dict(color="rgba(100,180,255,.6)", width=1, dash="dot"),
            fill='tonexty', fillcolor="rgba(100,180,255,.04)"), row=1, col=1)

    # EMAs
    if mostrar_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20",
            line=dict(color="#ffd700", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name="EMA50",
            line=dict(color="#ff8c00", width=1.5)), row=1, col=1)

    # Soporte / Resistencia
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

    # SL / TP
    fig.add_hline(y=info['tp'], line_color="#00ff88", line_dash="dash",
                  annotation_text=f"TP {info['tp']:,.2f}", row=1, col=1)
    fig.add_hline(y=info['sl'], line_color="#ff4444", line_dash="dash",
                  annotation_text=f"SL {info['sl']:,.2f}", row=1, col=1)

    # Con volumen: velas(1)|vol(2)|RSI(3)|MACD(4) — sin volumen: velas(1)|RSI(2)|MACD(3)
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

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI",
        line=dict(color="#c77dff", width=1.5)), row=rsi_row, col=1)
    fig.add_hline(y=70, line_color="rgba(255,68,68,.5)",  line_dash="dot", row=rsi_row, col=1)
    fig.add_hline(y=30, line_color="rgba(0,255,136,.5)", line_dash="dot", row=rsi_row, col=1)

    # MACD
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

    todos  = {**ACCIONES, **CRYPTOS}
    filas  = []
    prog   = st.progress(0)
    for i, (nm, sym) in enumerate(todos.items()):
        prog.progress((i + 1) / len(todos))
        try:
            d = get_data(sym, period)
            if d.empty or len(d) < 30:
                continue
            inf = generar_senal(d)
            p   = float(d['Close'].iloc[-1])
            pv  = float(d['Close'].iloc[-2]) if len(d) > 1 else p
            chg_row = ((p / pv) - 1) * 100 if pv > 0 else 0
            filas.append({
                "Activo":   nm,
                "Ticker":   sym,
                "Precio":   p,
                "Cambio %": round(chg_row, 2),
                "RSI":      round(float(d['RSI'].iloc[-1]), 1),
                "ATR":      round(float(d['ATR'].iloc[-1]), 4),
                "Señal":    inf['senal'],
                "Puntos":   inf['puntos'],
            })
        except Exception:
            continue
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

        # pandas >= 2.1 renombró applymap → map (por elemento)
        _cell_map = "map" if hasattr(df_radar.style, "map") else "applymap"
        styled = df_radar.style.format({
            "Precio": "{:,.4f}", "Cambio %": "{:+.2f}%",
            "RSI": "{:.1f}", "ATR": "{:.4f}",
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
            x=df_norm.index,
            y=base + diff,
            fill='tonexty',
            fillcolor=color_area,
            line=dict(width=0),
            name="Diferencia",
        ))

        fig_comp.update_layout(
            template="plotly_dark", height=460,
            title=f"Rendimiento normalizado (base 100) — {period}",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        ret_a = float(df_norm[ticker].iloc[-1]    - 100)
        ret_b = float(df_norm[bench_usado].iloc[-1] - 100)
        alpha = ret_a - ret_b

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Retorno {ticker}",     f"{ret_a:+.2f}%")
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
            corr,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=".2f",
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

    # Paper Trading
    with col_pt:
        st.subheader("📒 Paper Trading")
        st.caption("Operaciones simuladas — datos guardados en esta sesión")

        with st.form("nueva_operacion", clear_on_submit=True):
            st.markdown(f"**Activo:** `{ticker}` · Precio actual: `${price:,.4f}`")
            c1, c2, c3 = st.columns(3)
            entrada_pt = c1.number_input("Entrada ($)",    value=round(price, 4),    format="%.4f")
            sl_pt      = c2.number_input("Stop Loss ($)",  value=round(info['sl'],4), format="%.4f")
            tp_pt      = c3.number_input("Take Profit ($)", value=round(info['tp'],4), format="%.4f")
            c4, c5     = st.columns(2)
            capital_pt = c4.number_input("Capital ($)", value=1000.0, min_value=1.0, format="%.2f")
            lado_pt    = c5.selectbox("Dirección", ["LONG", "SHORT"])
            nota_op    = st.text_input("Nota (opcional)")
            submitted  = st.form_submit_button("➕ Registrar operación")

        if submitted:
            riesgo  = abs(entrada_pt - sl_pt)
            reward  = abs(tp_pt - entrada_pt)
            rr      = round(reward / riesgo, 2) if riesgo > 0 else 0
            tamanio = round(capital_pt * 0.02 / riesgo, 4) if riesgo > 0 else 0
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
            s1.metric("Total",         total)
            s2.metric("✅ TP alcanzado", ganadas)
            s3.metric("❌ SL tocado",   perdidas)
            s4.metric("Win Rate",       f"{winrate}%")

            if st.button("🗑️ Limpiar operaciones"):
                st.session_state.paper_trades = []
                st.rerun()
        else:
            st.info("Aún no hay operaciones. Registra una con el formulario de arriba.")

    # Notas del Trader
    with col_notas:
        st.subheader("✏️ Notas del Trader")
        st.caption(f"Nota activa: **{ticker}**")

        nota_actual = st.session_state.notas.get(ticker, "")
        nueva_nota  = st.text_area(
            "Tesis, niveles clave, catalizadores...",
            value=nota_actual,
            height=220,
            key=f"nota_{ticker}",
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

    ia_activa = "x-api-key" in CLAUDE_HEADERS and CLAUDE_HEADERS["x-api-key"]

    if not ia_activa:
        st.warning("Ingresa tu Anthropic API key en el panel izquierdo para activar las funciones de IA.")
        st.info("Obtén una en [console.anthropic.com](https://console.anthropic.com) · El tier gratuito es suficiente para esta app.")
        st.stop()

    resumen_tecnico = _resumen_tecnico(ticker, df, info)

    # ── Sección 1: Análisis narrativo ──────────────────────────────────────
    st.subheader(f"🤖 Análisis narrativo — {ticker}")

    cache_key_analisis = f"{ticker}_{period}"
    col_an1, col_an2 = st.columns([1, 5])
    with col_an1:
        regenerar = st.button("🔄 Generar / Actualizar análisis")

    if regenerar or cache_key_analisis not in st.session_state.analisis_cache:
        with st.spinner("Claude está analizando los indicadores..."):
            sistema_analisis = """Eres un analista técnico senior especializado en trading de acciones y criptomonedas.
Tu tarea es leer un resumen de indicadores técnicos y generar un análisis narrativo claro, 
conciso y profesional en español. Estructura tu respuesta así:
1. **Contexto general** (1-2 oraciones sobre la tendencia)
2. **Señales clave** (bullet points de los indicadores más relevantes)
3. **Escenario alcista** (qué necesita pasar para confirmar subida)
4. **Escenario bajista** (qué niveles vigilar para baja)
5. **Conclusión** (1 oración directa)
Sé directo, evita repetir los números del resumen textualmente."""

            analisis_texto = _claude(
                system=sistema_analisis,
                user=f"Analiza este activo:\n\n{resumen_tecnico}",
                max_tokens=700,
            )
            st.session_state.analisis_cache[cache_key_analisis] = analisis_texto

    st.markdown(st.session_state.analisis_cache.get(cache_key_analisis, ""))

    st.markdown("---")

    # ── Sección 2: Noticias + Sentimiento ─────────────────────────────────
    st.subheader(f"📰 Noticias recientes + Sentimiento — {ticker}")

    col_n1, col_n2 = st.columns([1, 5])
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

            noticias_texto = _claude(
                system=sistema_noticias,
                user=f"Analiza el sentimiento actual de mercado para {ticker}. Datos técnicos de contexto:\n{resumen_tecnico}",
                max_tokens=700,
            )
            st.session_state.noticias_cache[ticker] = noticias_texto

    if ticker in st.session_state.noticias_cache:
        st.markdown(st.session_state.noticias_cache[ticker])

    st.markdown("---")

    # ── Sección 3: Generador de tesis ──────────────────────────────────────
    st.subheader(f"📝 Generador de tesis de inversión — {ticker}")

    col_t1, col_t2, col_t3 = st.columns([1, 1, 4])
    with col_t1:
        horizonte = st.selectbox("Horizonte", ["Corto plazo (días)", "Mediano plazo (semanas)", "Largo plazo (meses)"])
    with col_t2:
        perfil    = st.selectbox("Perfil de riesgo", ["Conservador", "Moderado", "Agresivo"])

    generar_tesis = st.button("✍️ Generar tesis de inversión")

    if generar_tesis:
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

            tesis_texto = _claude(
                system=sistema_tesis,
                user=f"Genera una tesis de inversión para {ticker}.\nHorizonte: {horizonte}\nPerfil: {perfil}\nDatos técnicos:\n{resumen_tecnico}",
                max_tokens=800,
            )

        st.markdown(tesis_texto)

        # Ofrecer guardar como nota del trader
        if st.button("💾 Guardar tesis como nota del trader"):
            st.session_state.notas[ticker] = tesis_texto
            st.success(f"Tesis guardada en notas de {ticker} ✓")

    st.markdown("---")

    # ── Sección 4: Chat con el gráfico ─────────────────────────────────────
    st.subheader(f"💬 Chat con el asistente — {ticker}")
    st.caption("Pregunta lo que quieras sobre este activo, su situación técnica o el mercado en general.")

    # Inicializar historial del ticker
    if ticker not in st.session_state.chat_history:
        st.session_state.chat_history[ticker] = []

    historial = st.session_state.chat_history[ticker]

    # Mostrar mensajes anteriores
    for msg in historial:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario
    prompt_chat = st.chat_input(f"Pregunta sobre {ticker}...")

    if prompt_chat:
        # Añadir mensaje del usuario
        historial.append({"role": "user", "content": prompt_chat})
        with st.chat_message("user"):
            st.markdown(prompt_chat)

        # Contexto del sistema para el chat
        sistema_chat = f"""Eres un asistente experto en trading y análisis financiero que trabaja dentro de 
QuantumShield Pro, una app de análisis técnico. Tienes acceso al contexto técnico actual del activo {ticker}.

DATOS TÉCNICOS ACTUALES:
{resumen_tecnico}

Responde siempre en español, de forma clara y concisa. Si te preguntan sobre otros activos o temas 
generales de trading, responde con tu conocimiento general. No des consejos financieros definitivos, 
enfatiza que es análisis técnico educativo."""

        # Construir mensajes para la API (últimos 10 para no saturar contexto)
        mensajes_api = historial[-10:]

        # Streaming de respuesta
        with st.chat_message("assistant"):
            respuesta_placeholder = st.empty()
            respuesta_completa    = ""
            for chunk in _claude_stream(sistema_chat, mensajes_api, max_tokens=600):
                respuesta_completa += chunk
                respuesta_placeholder.markdown(respuesta_completa + "▌")
            respuesta_placeholder.markdown(respuesta_completa)

        historial.append({"role": "assistant", "content": respuesta_completa})

        # Limpiar chat
        if len(historial) > 40:
            st.session_state.chat_history[ticker] = historial[-40:]

    if historial and st.button("🗑️ Limpiar chat", key="clear_chat"):
        st.session_state.chat_history[ticker] = []
        st.rerun()
