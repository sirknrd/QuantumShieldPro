import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ Quantum Shield Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .signal-buy  { background:#0d3b1e; border-left:4px solid #00ff88;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .signal-sell { background:#3b0d0d; border-left:4px solid #ff4444;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .signal-hold { background:#1a1a2e; border-left:4px solid #f0a500;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .metric-card { background:#1e1e2e; border-radius:8px;
                   padding:12px 16px; text-align:center; margin:4px; }
    h1 { color:#00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  INDICADORES
# ─────────────────────────────────────────
def calcular_ema(series, periodo):
    return series.ewm(span=periodo, adjust=False).mean()

def calcular_rsi(series, periodo=14):
    delta = series.diff()
    ganancia = delta.clip(lower=0)
    perdida  = -delta.clip(upper=0)
    avg_g = ganancia.ewm(com=periodo - 1, adjust=False).mean()
    avg_p = perdida.ewm(com=periodo - 1, adjust=False).mean()
    rs = avg_g / avg_p.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calcular_macd(series, rapido=12, lento=26, senal=9):
    ema_r = calcular_ema(series, rapido)
    ema_l = calcular_ema(series, lento)
    macd_line   = ema_r - ema_l
    signal_line = calcular_ema(macd_line, senal)
    histograma  = macd_line - signal_line
    return macd_line, signal_line, histograma

def calcular_bollinger(series, periodo=20, desviaciones=2):
    media  = series.rolling(periodo).mean()
    std    = series.rolling(periodo).std()
    upper  = media + desviaciones * std
    lower  = media - desviaciones * std
    return upper, media, lower

def calcular_atr(df, periodo=14):
    h_l  = df['High'] - df['Low']
    h_cp = (df['High'] - df['Close'].shift()).abs()
    l_cp = (df['Low']  - df['Close'].shift()).abs()
    tr   = pd.concat([h_l, h_cp, l_cp], axis=1).max(axis=1)
    return tr.ewm(com=periodo - 1, adjust=False).mean()

# ─────────────────────────────────────────
#  MOTOR DE CONFLUENCIA
# ─────────────────────────────────────────
def motor_confluencia(df):
    """
    Genera señal BUY / SELL / HOLD cruzando EMA50, RSI y MACD.
    Retorna señal, score (0-3) y detalle de cada condición.
    """
    close  = df['Close']
    ema50  = calcular_ema(close, 50).iloc[-1]
    rsi    = calcular_rsi(close).iloc[-1]
    macd, senal, hist = calcular_macd(close)
    macd_v = macd.iloc[-1]
    sen_v  = senal.iloc[-1]
    hist_v = hist.iloc[-1]
    precio = close.iloc[-1]

    # Condiciones alcistas
    c1_buy = precio > ema50
    c2_buy = rsi < 70 and rsi > 40
    c3_buy = macd_v > sen_v and hist_v > 0

    # Condiciones bajistas
    c1_sell = precio < ema50
    c2_sell = rsi > 30 and rsi < 60
    c3_sell = macd_v < sen_v and hist_v < 0

    score_buy  = sum([c1_buy, c2_buy, c3_buy])
    score_sell = sum([c1_sell, c2_sell, c3_sell])

    if score_buy >= 2:
        señal = "🟢 COMPRA"
        score = score_buy
        css   = "signal-buy"
    elif score_sell >= 2:
        señal = "🔴 VENTA"
        score = score_sell
        css   = "signal-sell"
    else:
        señal = "🟡 ESPERAR"
        score = max(score_buy, score_sell)
        css   = "signal-hold"

    detalle = {
        "EMA 50":  ("✅" if c1_buy else "⬜") + f" Precio {'>' if c1_buy else '<'} EMA50 (${ema50:,.2f})",
        "RSI":     ("✅" if c2_buy else "⬜") + f" RSI = {rsi:.1f}",
        "MACD":    ("✅" if c3_buy else "⬜") + f" MACD {'>' if c3_buy else '<'} Señal (hist {hist_v:+.4f})",
    }
    return señal, score, css, detalle, rsi, macd_v, sen_v, hist_v

# ─────────────────────────────────────────
#  GESTIÓN DE RIESGO ATR
# ─────────────────────────────────────────
def gestion_riesgo(df, multiplicador_sl=1.5, multiplicador_tp=3.0):
    atr      = calcular_atr(df).iloc[-1]
    precio   = float(df['Close'].iloc[-1])
    sl_long  = precio - multiplicador_sl * atr
    tp_long  = precio + multiplicador_tp * atr
    sl_short = precio + multiplicador_sl * atr
    tp_short = precio - multiplicador_tp * atr
    rr_ratio = multiplicador_tp / multiplicador_sl
    return {
        "atr": atr, "precio": precio,
        "sl_long": sl_long, "tp_long": tp_long,
        "sl_short": sl_short, "tp_short": tp_short,
        "rr": rr_ratio
    }

# ─────────────────────────────────────────
#  DESCARGA DE DATOS
# ─────────────────────────────────────────
PERIODOS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}

MERCADOS = {
    "🌐 Internacional / Acciones": {
        "NVDA": "NVIDIA", "AAPL": "Apple", "MSFT": "Microsoft",
        "GOOGL": "Alphabet", "TSLA": "Tesla", "AMZN": "Amazon"
    },
    "₿ Criptomonedas": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
        "SOL-USD": "Solana", "BNB-USD": "BNB"
    },
    "🇨🇱 Mercado Chileno (IPSA)": {
        "SQM-B.SN": "SQM-B", "FALABELLA.SN": "Falabella",
        "COPEC.SN": "Copec", "CMPC.SN": "CMPC",
        "ENELCHILE.SN": "Enel Chile", "BANCO.SN": "Banco de Chile"
    }
}

@st.cache_data(ttl=300)
def descargar_datos(ticker, dias):
    today = datetime.today()
    start = today - timedelta(days=dias)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True
    )
    if df.empty:
        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=(today + timedelta(days=1)).strftime("%Y-%m-%d")
        )
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance nuevo: nivel 0 = campo, nivel 1 = ticker
        # yfinance viejo: nivel 0 = ticker, nivel 1 = campo
        if df.columns.get_level_values(0)[0] in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df.columns = df.columns.get_level_values(0)   # nuevo formato
        else:
            df.columns = df.columns.get_level_values(1)   # viejo formato
    cols_disponibles = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[cols_disponibles].dropna()
    return df

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("🛡️ Quantum Shield Pro")
st.sidebar.markdown("---")

mercado = st.sidebar.selectbox("📊 Mercado", list(MERCADOS.keys()))
tickers_mercado = MERCADOS[mercado]

modo = st.sidebar.radio("Modo ticker", ["Lista rápida", "Ingreso manual"])
if modo == "Lista rápida":
    ticker_label = st.sidebar.selectbox(
        "Activo", [f"{k} — {v}" for k, v in tickers_mercado.items()]
    )
    ticker = ticker_label.split(" — ")[0]
else:
    ticker = st.sidebar.text_input("Ticker (ej: AAPL, BTC-USD)", value="NVDA").upper().strip()

periodo = st.sidebar.selectbox("📅 Período", list(PERIODOS.keys()), index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parámetros de Riesgo")
mult_sl = st.sidebar.slider("Multiplicador Stop Loss (ATR)", 1.0, 3.0, 1.5, 0.1)
mult_tp = st.sidebar.slider("Multiplicador Take Profit (ATR)", 1.5, 6.0, 3.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Indicadores visibles")
mostrar_ema   = st.sidebar.checkbox("EMA 20 / 50 / 200", value=True)
mostrar_bb    = st.sidebar.checkbox("Bandas de Bollinger", value=True)
mostrar_vol   = st.sidebar.checkbox("Volumen", value=True)

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
st.title("🛡️ Quantum Shield Pro — Financial Trading Terminal")

if not ticker:
    st.warning("Ingresa o selecciona un ticker válido.")
    st.stop()

with st.spinner(f"Descargando datos de **{ticker}**..."):
    try:
        df = descargar_datos(ticker, PERIODOS[periodo])
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        st.stop()

if df.empty or len(df) < 60:
    st.error(f"No hay suficientes datos para **{ticker}**. Prueba otro ticker o un período mayor.")
    st.stop()

# ── KPIs ──────────────────────────────────
precio_actual = float(df['Close'].iloc[-1])
precio_prev   = float(df['Close'].iloc[-2])
cambio_pct    = (precio_actual / precio_prev - 1) * 100
precio_max    = float(df['High'].max())
precio_min    = float(df['Low'].min())
volumen_ult   = int(df['Volume'].iloc[-1])
ultima_vela   = df.index[-1]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Precio Actual",   f"${precio_actual:,.2f}", f"{cambio_pct:+.2f}%")
col2.metric("📈 Máximo período",  f"${precio_max:,.2f}")
col3.metric("📉 Mínimo período",  f"${precio_min:,.2f}")
col4.metric("📦 Volumen (ult.)",  f"{volumen_ult:,}")
col5.metric("🗓️ Última vela",    ultima_vela.strftime("%d/%m/%Y"))

st.markdown("---")

# ── SEÑAL DE CONFLUENCIA ─────────────────
señal, score, css, detalle, rsi_v, macd_v, sen_v, hist_v = motor_confluencia(df)
riesgo = gestion_riesgo(df, mult_sl, mult_tp)

col_sig, col_riesgo = st.columns([1, 1])

with col_sig:
    st.subheader("🎯 Señal de Confluencia")
    st.markdown(f"""
    <div class="{css}">
        <h2 style="margin:0">{señal}</h2>
        <p style="margin:4px 0 0 0; opacity:.8">Confluencia: {score}/3 indicadores alineados</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Detalle de indicadores:**")
    for nombre, valor in detalle.items():
        st.write(f"&nbsp;&nbsp;{valor}")

with col_riesgo:
    st.subheader("🛡️ Gestión de Riesgo (ATR)")
    st.write(f"**ATR actual:** `${riesgo['atr']:,.4f}`")
    st.write(f"**R/R Ratio:** `{riesgo['rr']:.1f}x`")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**🟢 LARGO (Compra)**")
        st.success(f"TP: ${riesgo['tp_long']:,.2f}")
        st.error(  f"SL: ${riesgo['sl_long']:,.2f}")
    with r2:
        st.markdown("**🔴 CORTO (Venta)**")
        st.error(  f"SL: ${riesgo['sl_short']:,.2f}")
        st.success(f"TP: ${riesgo['tp_short']:,.2f}")

st.markdown("---")

# ── GRÁFICO PRINCIPAL ─────────────────────
close = df['Close']
ema20  = calcular_ema(close, 20)
ema50  = calcular_ema(close, 50)
ema200 = calcular_ema(close, 200)
bb_up, bb_mid, bb_low = calcular_bollinger(close)
macd_line, sig_line, histograma = calcular_macd(close)
rsi_series = calcular_rsi(close)
vol_colors = ['#ef5350' if df['Close'].iloc[i] < df['Open'].iloc[i]
              else '#26a69a' for i in range(len(df))]

# Subplots: precio | RSI | MACD (+ volumen si activo)
n_filas = 3 + (1 if mostrar_vol else 0)
altura_filas = [0.55, 0.15, 0.15] + ([0.15] if mostrar_vol else [])
subplot_titles = ["Precio", "RSI", "MACD"] + (["Volumen"] if mostrar_vol else [])

fig = make_subplots(
    rows=n_filas, cols=1,
    shared_xaxes=True,
    row_heights=altura_filas,
    subplot_titles=subplot_titles,
    vertical_spacing=0.03
)

# ── Velas japonesas ──
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'],   close=df['Close'],
    name="Precio",
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350'
), row=1, col=1)

# ── Bandas de Bollinger ──
if mostrar_bb:
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_up, name="BB Superior",
        line=dict(color='rgba(100,149,237,0.6)', width=1, dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_low, name="BB Inferior",
        fill='tonexty',
        fillcolor='rgba(100,149,237,0.07)',
        line=dict(color='rgba(100,149,237,0.6)', width=1, dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_mid, name="BB Media",
        line=dict(color='rgba(100,149,237,0.4)', width=1)
    ), row=1, col=1)

# ── EMAs ──
if mostrar_ema:
    fig.add_trace(go.Scatter(
        x=df.index, y=ema20, name="EMA 20",
        line=dict(color='#FFD700', width=1.2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ema50, name="EMA 50",
        line=dict(color='#FF8C00', width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ema200, name="EMA 200",
        line=dict(color='#FF4500', width=1.8)
    ), row=1, col=1)

# ── Líneas SL/TP ──
precio_entry = riesgo['precio']
fig.add_hline(y=precio_entry,      line=dict(color='white',   width=1, dash='dot'),  row=1, col=1)
fig.add_hline(y=riesgo['tp_long'], line=dict(color='#26a69a', width=1, dash='dash'), row=1, col=1)
fig.add_hline(y=riesgo['sl_long'], line=dict(color='#ef5350', width=1, dash='dash'), row=1, col=1)

# ── RSI ──
fig.add_trace(go.Scatter(
    x=df.index, y=rsi_series, name="RSI",
    line=dict(color='#E040FB', width=1.5)
), row=2, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,83,80,0.1)',  line_width=0, row=2, col=1)
fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(38,166,154,0.1)', line_width=0, row=2, col=1)
fig.add_hline(y=70, line=dict(color='#ef5350', width=0.8, dash='dot'), row=2, col=1)
fig.add_hline(y=30, line=dict(color='#26a69a', width=0.8, dash='dot'), row=2, col=1)

# ── MACD ──
hist_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in histograma]
fig.add_trace(go.Bar(
    x=df.index, y=histograma, name="Histograma",
    marker_color=hist_colors, opacity=0.7
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=macd_line, name="MACD",
    line=dict(color='#2196F3', width=1.5)
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=sig_line, name="Señal",
    line=dict(color='#FF9800', width=1.5)
), row=3, col=1)

# ── Volumen ──
if mostrar_vol:
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name="Volumen",
        marker_color=vol_colors, opacity=0.7
    ), row=4, col=1)

fig.update_layout(
    template="plotly_dark",
    height=900,
    xaxis_rangeslider_visible=False,
    paper_bgcolor='#0e1117',
    plot_bgcolor='#0e1117',
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
    font=dict(color='#c8c8d4')
)
fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')

st.plotly_chart(fig, use_container_width=True)

# ── TABLA DE INDICADORES ─────────────────
st.markdown("---")
st.subheader("📊 Resumen de Indicadores")

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    st.markdown("**Medias Móviles**")
    st.write(f"EMA 20:  `${float(ema20.iloc[-1]):,.2f}`")
    st.write(f"EMA 50:  `${float(ema50.iloc[-1]):,.2f}`")
    st.write(f"EMA 200: `${float(ema200.iloc[-1]):,.2f}`")
    trend = "📈 Alcista" if precio_actual > float(ema50.iloc[-1]) > float(ema200.iloc[-1]) \
            else "📉 Bajista" if precio_actual < float(ema50.iloc[-1]) < float(ema200.iloc[-1]) \
            else "↔️ Lateral"
    st.write(f"Tendencia: **{trend}**")

with col_t2:
    st.markdown("**Momentum**")
    rsi_label = "Sobrecomprado 🔴" if rsi_v > 70 else "Sobrevendido 🟢" if rsi_v < 30 else "Neutral 🟡"
    st.write(f"RSI (14): `{rsi_v:.1f}` — {rsi_label}")
    st.write(f"MACD:     `{macd_v:+.4f}`")
    st.write(f"Señal:    `{sen_v:+.4f}`")
    st.write(f"Histograma: `{hist_v:+.4f}`")

with col_t3:
    st.markdown("**Volatilidad (Bollinger)**")
    bb_w = float((bb_up.iloc[-1] - bb_low.iloc[-1]) / bb_mid.iloc[-1] * 100)
    st.write(f"BB Superior: `${float(bb_up.iloc[-1]):,.2f}`")
    st.write(f"BB Media:    `${float(bb_mid.iloc[-1]):,.2f}`")
    st.write(f"BB Inferior: `${float(bb_low.iloc[-1]):,.2f}`")
    st.write(f"Ancho BB:    `{bb_w:.1f}%`")

st.markdown("---")
st.caption("⚠️ **Aviso legal:** Esta herramienta es solo para análisis técnico educativo. No constituye asesoramiento financiero. Opera siempre con gestión de riesgo propia.")
