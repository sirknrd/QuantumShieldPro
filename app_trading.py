import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Shield Pro | Bollinger Edition", page_icon="🛡️", layout="wide")

# --- 2. LÓGICA DE CONFLUENCIA (Modo Institucional) ---
def calcular_señal_pro(last):
    try:
        puntos = 0
        close = float(last['Close'])
        ema200 = float(last['EMA_200'])
        bb_up = float(last['BBU_20_2.0'])
        bb_low = float(last['BBL_20_2.0'])
        
        # Test 1: Filtro de Tendencia (EMA 200)
        if close > ema200: puntos += 1
        else: puntos -= 1
        
        # Test 2: Bollinger Rejection (Sobrecompra/Sobreventa)
        if close > bb_up: puntos -= 1  # Riesgo de caída (muy caro)
        if close < bb_low: puntos += 1 # Oportunidad de rebote (muy barato)
        
        # Test 3: Momentum (RSI)
        if 50 < float(last['RSI']) < 70: puntos += 1
        
        # Test 4: Fuerza (ADX)
        if float(last['ADX_14']) > 25: puntos += 1

        if puntos >= 2: return "COMPRA CONFIRMADA ✅", "#00FFA3", "rgba(0, 255, 163, 0.1)"
        elif puntos <= -2: return "VENTA / TOMA GANANCIA ⚠️", "#FF4B4B", "rgba(255, 75, 75, 0.1)"
        else: return "MANTENER / NEUTRAL ⚖️", "#FFA500", "rgba(255, 165, 0, 0.1)"
    except: return "ESPERANDO...", "#8B949E", "transparent"

# --- 3. INTERFAZ ---
st.sidebar.title("🛡️ Quantum Pro")
ticker_input = st.sidebar.text_input("Activo", value="BTC").upper()

# Limpieza de Símbolo
if len(ticker_input) <= 4 and "-" not in ticker_input and "." not in ticker_input:
    ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL"] else ticker_input
else: ticker = ticker_input

df = yf.download(ticker, period="365d", interval="1d", auto_adjust=True)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # INDICADORES DE ALTA CONFIANZA
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    
    # Bandas de Bollinger (20 periodos, 2 desviaciones)
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1).fillna(0)
    
    last = df.iloc[-1]
    rec, color, bg = calcular_señal_pro(last)

    # PANEL DE CONTROL
    st.markdown(f"""
        <div style="background-color: {bg}; border: 3px solid {color}; padding: 25px; border-radius: 20px; text-align: center;">
            <h1 style="color: {color}; font-size: 45px; margin: 0;">{rec}</h1>
            <p style="color: white; font-size: 18px;">Estrategia: EMA 200 + Bollinger + RSI</p>
        </div>
    """, unsafe_allow_html=True)

    # GRÁFICO PRO
    fig = go.Figure()
    # Velas
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    
    # Bandas de Bollinger (Sombreadas)
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(color='rgba(173, 216, 230, 0.4)'), name="Bollinger Sup"))
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)', line=dict(color='rgba(173, 216, 230, 0.4)'), name="Bollinger Inf"))
    
    # EMA 200 Mejorada (Roja y Gruesa)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF0000', width=3), name="Institucional (EMA 200)"))

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=650)
    st.plotly_chart(fig, use_container_width=True)

else: st.error("Símbolo no encontrado.")
