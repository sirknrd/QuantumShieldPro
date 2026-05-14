import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumShield Pro",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Quantum Shield Pro — Financial Trading Terminal")
st.caption("Motor de confluencia · Gestión de riesgo ATR · Análisis técnico avanzado")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

MERCADOS = {
    "🌐 Criptomonedas": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "📈 Acciones Internacionales": ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "META"],
    "🇨🇱 Mercado Chileno (IPSA)": ["FALABELLA.SN", "COPEC.SN", "BCI.SN", "CMPC.SN", "CHILE.SN"],
    "✏️ Personalizado": [],
}

mercado = st.sidebar.selectbox("Mercado", list(MERCADOS.keys()))

if mercado == "✏️ Personalizado":
    ticker = st.sidebar.text_input("Ticker", value="NVDA").upper().strip()
else:
    ticker = st.sidebar.selectbox("Activo", MERCADOS[mercado])

period = st.sidebar.selectbox("Período", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Intervalo", ["1d", "1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Parámetros de Riesgo")
atr_sl_mult = st.sidebar.slider("Multiplicador ATR Stop Loss", 1.0, 4.0, 1.5, 0.25)
atr_tp_mult = st.sidebar.slider("Multiplicador ATR Take Profit", 1.5, 6.0, 3.0, 0.25)
capital = st.sidebar.number_input("Capital disponible (USD)", min_value=100, value=10_000, step=500)
riesgo_pct = st.sidebar.slider("Riesgo por operación (%)", 0.5, 5.0, 1.0, 0.5)


# ─────────────────────────────────────────────
# INDICADORES TÉCNICOS
# ─────────────────────────────────────────────
def calcular_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calcular_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calcular_ema(series, fast)
    ema_slow = calcular_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calcular_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calcular_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calcular_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    sma = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = sma + std * sigma
    lower = sma - std * sigma
    return upper, sma, lower


# ─────────────────────────────────────────────
# MOTOR DE CONFLUENCIA
# ─────────────────────────────────────────────
def motor_confluencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera señales BUY/SELL cuando confluyen:
      - EMA50: precio cruza la EMA desde abajo (BUY) / arriba (SELL)
      - RSI: sale de sobreventa (<30 → BUY) o sobrecompra (>70 → SELL)
      - MACD: cruce alcista (BUY) o bajista (SELL) de la señal
    Una señal se valida solo cuando al menos 2 de 3 condiciones coinciden.
    """
    close = df["Close"]

    # ── Indicadores ──────────────────────────────
    df["EMA50"] = calcular_ema(close, 50)
    df["RSI"] = calcular_rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calcular_macd(close)
    df["ATR"] = calcular_atr(df)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = calcular_bollinger(close)

    # ── Condiciones individuales ──────────────────
    # EMA50
    price_above_ema = close > df["EMA50"]
    ema_buy  = (~price_above_ema.shift(1).fillna(False)) & price_above_ema   # cruce alcista
    ema_sell = price_above_ema.shift(1).fillna(False) & (~price_above_ema)   # cruce bajista

    # RSI
    rsi = df["RSI"]
    rsi_buy  = (rsi.shift(1) < 30) & (rsi >= 30)   # sale de sobreventa
    rsi_sell = (rsi.shift(1) > 70) & (rsi <= 70)   # sale de sobrecompra

    # MACD
    macd = df["MACD"]
    sig  = df["MACD_Signal"]
    macd_buy  = (macd.shift(1) < sig.shift(1)) & (macd >= sig)
    macd_sell = (macd.shift(1) > sig.shift(1)) & (macd <= sig)

    # ── Confluencia (≥2 condiciones) ─────────────
    buy_score  = ema_buy.astype(int)  + rsi_buy.astype(int)  + macd_buy.astype(int)
    sell_score = ema_sell.astype(int) + rsi_sell.astype(int) + macd_sell.astype(int)

    df["Signal"]    = "NEUTRAL"
    df["Score_Buy"] = buy_score
    df["Score_Sell"] = sell_score
    df.loc[buy_score  >= 2, "Signal"] = "BUY"
    df.loc[sell_score >= 2, "Signal"] = "SELL"

    # ── Gestión de riesgo ATR ─────────────────────
    df["SL_Buy"]  = close - atr_sl_mult * df["ATR"]
    df["TP_Buy"]  = close + atr_tp_mult * df["ATR"]
    df["SL_Sell"] = close + atr_sl_mult * df["ATR"]
    df["TP_Sell"] = close - atr_tp_mult * df["ATR"]

    return df


# ─────────────────────────────────────────────
# DESCARGA Y PROCESAMIENTO
# ─────────────────────────────────────────────
if not ticker:
    st.warning("Ingresa un ticker válido en el sidebar.")
    st.stop()

with st.spinner(f"Descargando datos de {ticker}..."):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)

        # yfinance puede devolver MultiIndex (ticker en nivel 1 con auto_adjust)
        if isinstance(df.columns, pd.MultiIndex):
            # Intentamos extraer el ticker; si falla, colapsamos el nivel superior
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df.columns = df.columns.get_level_values(0)

        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            raise ValueError(f"Columnas inesperadas: {list(df.columns)}")

        df = df[list(needed)].copy()
        df.dropna(subset=["Close"], inplace=True)

        if len(df) < 55:
            st.error(f"Datos insuficientes para calcular indicadores ({len(df)} velas). "
                     "Prueba con un período más largo.")
            st.stop()

        df = motor_confluencia(df)

    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        st.info("Prueba con otro ticker o período distinto.")
        st.stop()


# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
last   = float(df["Close"].iloc[-1])
prev   = float(df["Close"].iloc[-2])
change = (last / prev - 1) * 100
rsi_v  = float(df["RSI"].iloc[-1])
atr_v  = float(df["ATR"].iloc[-1])
signal = df["Signal"].iloc[-1]
last_date = df.index[-1].strftime("%d/%m/%Y")

# Señal más reciente con score alto
last_signals = df[df["Signal"] != "NEUTRAL"].tail(5)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Precio Actual", f"${last:,.2f}", f"{change:+.2f}%")
col2.metric("RSI (14)", f"{rsi_v:.1f}",
            "Sobrevendido" if rsi_v < 30 else ("Sobrecomprado" if rsi_v > 70 else "Neutral"))
col3.metric("ATR (14)", f"${atr_v:,.2f}", "Volatilidad actual")
col4.metric("Última vela", last_date)

signal_color = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}
col5.metric("Señal Confluencia", f"{signal_color.get(signal,'⚪')} {signal}")


# ─────────────────────────────────────────────
# GESTIÓN DE RIESGO ATR (última señal)
# ─────────────────────────────────────────────
if signal != "NEUTRAL":
    st.markdown("---")
    st.subheader("📐 Gestión de Riesgo — Señal Actual")

    if signal == "BUY":
        sl = float(df["SL_Buy"].iloc[-1])
        tp = float(df["TP_Buy"].iloc[-1])
    else:
        sl = float(df["SL_Sell"].iloc[-1])
        tp = float(df["TP_Sell"].iloc[-1])

    riesgo_usd   = capital * (riesgo_pct / 100)
    sl_dist      = abs(last - sl)
    posicion_sz  = riesgo_usd / sl_dist if sl_dist > 0 else 0
    rr_ratio     = abs(tp - last) / sl_dist if sl_dist > 0 else 0

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Stop Loss",   f"${sl:,.2f}", f"{((sl/last)-1)*100:+.2f}%")
    r2.metric("Take Profit", f"${tp:,.2f}", f"{((tp/last)-1)*100:+.2f}%")
    r3.metric("Tamaño Posición", f"{posicion_sz:,.4f} unidades",
              f"Riesgo: ${riesgo_usd:,.0f}")
    r4.metric("Ratio R:R", f"1 : {rr_ratio:.2f}",
              "✅ Favorable" if rr_ratio >= 2 else "⚠️ Ajustar")


# ─────────────────────────────────────────────
# GRÁFICO PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📈 Análisis Técnico — {ticker}")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2],
    vertical_spacing=0.04,
    subplot_titles=("Precio · Bollinger · EMA50", "RSI (14)", "MACD (12/26/9)"),
)

# ── Panel 1: Velas + Bollinger + EMA50 ───────
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"], high=df["High"],
    low=df["Low"],   close=df["Close"],
    name="Precio",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)

# Área de volatilidad Bollinger
fig.add_trace(go.Scatter(
    x=df.index, y=df["BB_Upper"],
    line=dict(color="rgba(100,180,255,0.4)", width=1),
    name="BB Superior", showlegend=False,
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["BB_Lower"],
    fill="tonexty",
    fillcolor="rgba(100,180,255,0.08)",
    line=dict(color="rgba(100,180,255,0.4)", width=1),
    name="Bandas Bollinger",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["BB_Mid"],
    line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
    name="SMA 20",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["EMA50"],
    line=dict(color="#FFA726", width=1.5),
    name="EMA 50",
), row=1, col=1)

# Señales de compra/venta
buys  = df[df["Signal"] == "BUY"]
sells = df[df["Signal"] == "SELL"]

fig.add_trace(go.Scatter(
    x=buys.index, y=buys["Low"] * 0.993,
    mode="markers",
    marker=dict(symbol="triangle-up", size=12, color="#00E676"),
    name="Señal COMPRA",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=sells.index, y=sells["High"] * 1.007,
    mode="markers",
    marker=dict(symbol="triangle-down", size=12, color="#FF1744"),
    name="Señal VENTA",
), row=1, col=1)

# ── Panel 2: RSI ─────────────────────────────
fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI"],
    line=dict(color="#AB47BC", width=1.5),
    name="RSI",
), row=2, col=1)

fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.05, row=2, col=1)
fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.05, row=2, col=1)

# ── Panel 3: MACD ────────────────────────────
colors_hist = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"]]

fig.add_trace(go.Bar(
    x=df.index, y=df["MACD_Hist"],
    marker_color=colors_hist,
    name="Histograma MACD",
    showlegend=False,
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD"],
    line=dict(color="#42A5F5", width=1.5),
    name="MACD",
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD_Signal"],
    line=dict(color="#FF7043", width=1.5),
    name="Señal MACD",
), row=3, col=1)

fig.update_layout(
    template="plotly_dark",
    height=800,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0),
)
fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="MACD", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# HISTORIAL DE SEÑALES
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Historial de Señales de Confluencia")

signal_df = df[df["Signal"] != "NEUTRAL"][
    ["Close", "ATR", "Signal", "Score_Buy", "Score_Sell",
     "SL_Buy", "TP_Buy", "SL_Sell", "TP_Sell"]
].copy().tail(20).iloc[::-1]

def format_signal_row(row):
    sig = row["Signal"]
    sl  = row["SL_Buy"]  if sig == "BUY" else row["SL_Sell"]
    tp  = row["TP_Buy"]  if sig == "BUY" else row["TP_Sell"]
    score = row["Score_Buy"] if sig == "BUY" else row["Score_Sell"]
    return pd.Series({
        "Fecha":        row.name.strftime("%d/%m/%Y"),
        "Señal":        f"{'🟢 COMPRA' if sig == 'BUY' else '🔴 VENTA'}",
        "Precio":       f"${float(row['Close']):,.2f}",
        "Stop Loss":    f"${float(sl):,.2f}",
        "Take Profit":  f"${float(tp):,.2f}",
        "ATR":          f"${float(row['ATR']):,.2f}",
        "Confluencia":  f"{'⭐' * int(score)} ({int(score)}/3)",
    })

if not signal_df.empty:
    tabla = signal_df.apply(format_signal_row, axis=1)
    st.dataframe(tabla, use_container_width=True, hide_index=True)
else:
    st.info("No se detectaron señales de confluencia en el período seleccionado. "
            "Prueba extendiendo el período o cambiando el activo.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Aviso legal:** Esta herramienta es solo para fines educativos y de análisis. "
    "No constituye asesoramiento financiero. Opera siempre con gestión de riesgo adecuada."
)
