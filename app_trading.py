import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Shield | Selector Pro", page_icon="🛡️", layout="wide")

# --- 2. LÓGICA DE RECOMENDACIÓN (6 Capas) ---
def obtener_recomendacion(last):
    try:
        score = 0
        close, ema200 = float(last['Close']), float(last['EMA_200'])
        rsi, mfi, adx = float(last['RSI']), float(last['MFI']), float(last['ADX_14'])
        
        # Confluencia técnica
        if close > ema200: score += 1
        else: score -= 1
        if rsi > 55: score += 1
        elif rsi < 45: score -= 1
        if mfi > 55: score += 1
        elif mfi < 45: score -= 1
        if adx > 25: score += 1
        
        m_h_col = [c for c in last.index if 'MACDh' in str(c)][0]
        if float(last[m_h_col]) > 0: score += 1
        else: score -= 1

        if score >= 3: return "COMPRA FUERTE 🚀", "#00FFA3", "rgba(0, 255, 163, 0.1)"
        elif score <= -3: return "VENTA FUERTE 📉", "#FF4B4B", "rgba(255, 75, 75, 0.1)"
        else: return "MANTENER / NEUTRAL ⚖️", "#FFA500", "rgba(255, 165, 0, 0.1)"
    except: return "ESPERANDO DATOS...", "#8B949E", "transparent"

# --- 3. BARRA LATERAL (TU SELECTOR) ---
st.sidebar.title("🛡️ Selector de Activos")
st.sidebar.markdown("---")

# Buscador manual
ticker_input = st.sidebar.text_input("Escribe el Símbolo (Ej: NVDA, BTC, SQM-B.SN)", value="AAPL").upper()

# Sugerencias rápidas (Botones)
st.sidebar.subheader("Favoritos Rápidos")
if st.sidebar.button("NVIDIA (IA)"): ticker_input = "NVDA"
if st.sidebar.button("BITCOIN"): ticker_input = "BTC-USD"
if st.sidebar.button("APPLE"): ticker_input = "AAPL"
if st.sidebar.button("BANCO CHILE"): ticker_input = "CHILE.SN"

# Temporalidad
tf = st.sidebar.selectbox("Temporalidad del Análisis", ["1h", "4h", "1d"], index=2)

# --- 4. MOTOR DE ANÁLISIS ---
# Limpieza automática de nombres para Cripto
ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL", "DOT"] else ticker_input

df_raw = yf.download(ticker, period="365d", interval=tf, auto_adjust=True)

if not df_raw.empty:
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    # Indicadores
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df = pd.concat([df, ta.macd(df['Close'])], axis=1).fillna(0)
    
    last = df.iloc[-1]
    rec, color, bg = obtener_recomendacion(last)

    # --- 5. INTERFAZ ---
    st.markdown(f"""<div style="background-color: {bg}; border: 2px solid {color}; padding: 20px; border-radius: 15px; text-align: center;">
        <h3 style="color: white; margin:0;">ACTIVO SELECCIONADO: {ticker}</h3>
        <h1 style="color: {color}; margin: 10px 0; font-size: 45px;">{rec}</h1>
    </div>""", unsafe_allow_html=True)

    # Gráfico interactivo
    fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name="Institucional (EMA 200)", line=dict(color='red', width=2)))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(f"No se encontraron datos para '{ticker}'. Verifica el símbolo en Yahoo Finance.")
