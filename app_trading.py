import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Quantum Shield Pro | Terminal",
    page_icon="🛡️",
    layout="wide"
)

# Estilo CSS personalizado para mejorar la legibilidad
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161B22; padding: 15px; border-radius: 10px; border: 1px solid #262730; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (Panel de Control) ---
st.sidebar.title("🛡️ Configuración")
ticker = st.sidebar.text_input("Símbolo del Activo", value="BTC-USD").upper()
temporalidad = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d", "1wk"], index=2)
periodo = st.sidebar.slider("Días de historial", min_value=10, max_value=730, value=180)

# --- FUNCIONES DE DATOS ---
@st.cache_data(ttl=300) # Caché de 5 minutos
def descargar_datos(symbol, days, interval):
    try:
        fin = datetime.now()
        inicio = fin - timedelta(days=days)
        data = yf.download(symbol, start=inicio, end=fin, interval=interval)
        return data
    except Exception as e:
        return None

# --- MOTOR DE INDICADORES ---
def calcular_indicadores(df):
    # Tendencia
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
    # Momentum & Volatilidad
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # Bandas de Bollinger
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    return df

# --- EJECUCIÓN PRINCIPAL ---
st.title(f"📊 Terminal Quantum Shield Pro: {ticker}")

df = descargar_datos(ticker, periodo, temporalidad)

if df is not None and not df.empty:
    df = calcular_indicadores(df)
    
    # Valores actuales para el tablero
    precio_actual = df['Close'].iloc[-1]
    rsi_actual = df['RSI'].iloc[-1]
    ema50_actual = df['EMA_50'].iloc[-1]
    atr_actual = df['ATR'].iloc[-1]
    
    # --- LÓGICA DE SEÑALES (CONFLUENCIA) ---
    # Compra: Precio > EMA50 Y RSI > 45 Y MACD Histograma > 0
    # Venta: Precio < EMA50 Y RSI < 55 Y MACD Histograma < 0
    macd_cols = [col for col in df.columns if 'MACDh' in col]

if macd_cols:
    macd_hist = df[macd_cols[0]].iloc[-1]
    macd_val = df[[c for c in df.columns if 'MACD_' in col and 's' not in col][0]].iloc[-1]
else:
    st.warning("Indicadores MACD en proceso de cálculo...")
    macd_hist = 0
    
    if precio_actual > ema50_actual and rsi_actual > 45 and macd_hist > 0:
        signal_text = "COMPRA FUERTE"
        signal_color = "#00FFA3"
    elif precio_actual < ema50_actual and rsi_actual < 55 and macd_hist < 0:
        signal_text = "VENTA FUERTE"
        signal_color = "#FF4B4B"
    else:
        signal_text = "NEUTRAL / ESPERAR"
        signal_color = "#FFA500"

    # --- MÉTRICAS SUPERIORES ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precio Actual", f"${precio_actual:,.2f}")
    col2.markdown(f"<div class='stMetric'><b>SEÑAL IA:</b><br><span style='color:{signal_color}; font-size:20px;'>{signal_text}</span></div>", unsafe_allow_html=True)
    col3.metric("RSI (14)", f"{rsi_actual:.2f}")
    col4.metric("Volatilidad (ATR)", f"{atr_actual:.2f}")

    # --- GESTIÓN DE RIESGO ---
    st.subheader("🛡️ Plan de Gestión de Riesgo")
    sl = precio_actual - (atr_actual * 2) if signal_text == "COMPRA FUERTE" else precio_actual + (atr_actual * 2)
    tp = precio_actual + (atr_actual * 4) if signal_text == "COMPRA FUERTE" else precio_actual - (atr_actual * 4)
    
    r_col1, r_col2 = st.columns(2)
    r_col1.warning(f"**STOP LOSS SUGERIDO:** ${sl:,.2f}")
    r_col2.success(f"**TAKE PROFIT SUGERIDO:** ${tp:,.2f}")

    # --- GRÁFICO INTERACTIVO ---
    st.subheader("📈 Análisis Técnico en Vivo")
    fig = go.Figure()

    # Velas Japonesas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Precio", increasing_line_color='#00FFA3', decreasing_line_color='#FF4B4B'
    ))

    # Medias Móviles y Bandas
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name="Trend (EMA 50)", line=dict(color='white', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name="Bollinger Sup", line=dict(color='rgba(173, 216, 230, 0.2)', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name="Bollinger Inf", line=dict(color='rgba(173, 216, 230, 0.2)', dash='dot'), fill='tonexty'))

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- TABLA DE DATOS ---
    with st.expander("Ver Datos Históricos"):
        st.dataframe(df.tail(20), use_container_width=True)

else:
    st.error("No se pudo cargar el activo. Verifica que el símbolo sea correcto (ej: BTC-USD o AAPL).")
