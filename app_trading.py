import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Shield Pro | Ultra", page_icon="🛡️", layout="wide")

# --- 2. MOTOR DE RECOMENDACIÓN (Scoring Pro) ---
def calcular_señal(last, df_cols):
    try:
        puntos = 0
        close = float(last['Close'])
        ema200 = float(last['EMA_200'])
        
        # Buscar columnas de Bollinger dinámicamente
        bbu_col = [c for c in df_cols if 'BBU' in str(c)][0]
        bbl_col = [c for c in df_cols if 'BBL' in str(c)][0]
        
        # Regla 1: Tendencia Institucional
        if close > ema200: puntos += 1
        else: puntos -= 1
        
        # Regla 2: Bandas de Bollinger (Sobrecompra/Sobreventa)
        if close > float(last[bbu_col]): puntos -= 1 # Muy caro
        if close < float(last[bbl_col]): puntos += 1 # Muy barato
        
        # Regla 3: RSI (Fuerza de precio)
        if 50 < float(last['RSI']) < 70: puntos += 1
        
        if puntos >= 1: return "COMPRA CONFIRMADA ✅", "#00FFA3", "rgba(0, 255, 163, 0.1)"
        elif puntos <= -1: return "VENTA / PRECAUCIÓN ⚠️", "#FF4B4B", "rgba(255, 75, 75, 0.1)"
        else: return "MANTENER / NEUTRAL ⚖️", "#FFA500", "rgba(255, 165, 0, 0.1)"
    except:
        return "ANALIZANDO...", "#8B949E", "transparent"

# --- 3. BARRA LATERAL ---
st.sidebar.title("🛡️ Quantum Pro")
ticker_input = st.sidebar.text_input("Activo (Ej: BTC, AAPL, SQM-B.SN)", value="BTC").upper()

# Limpieza de Símbolo (Crypto + Stocks)
if len(ticker_input) <= 4 and "-" not in ticker_input and "." not in ticker_input:
    ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL", "DOT"] else ticker_input
else: ticker = ticker_input

# --- 4. DESCARGA Y CÁLCULOS ---
df_raw = yf.download(ticker, period="365d", interval="1d", auto_adjust=True)

if not df_raw.empty:
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Indicadores
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    
    # Bandas de Bollinger
    bb = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1).fillna(0)
    
    # Identificar columnas de Bollinger para el gráfico
    bbu_name = [c for c in df.columns if 'BBU' in str(c)][0]
    bbl_name = [c for c in df.columns if 'BBL' in str(c)][0]
    
    last = df.iloc[-1]
    rec, color, bg = calcular_señal(last, df.columns)

    # --- 5. INTERFAZ ---
    st.markdown(f"""
        <div style="background-color: {bg}; border: 3px solid {color}; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: {color}; margin: 0; font-size: 40px;">{rec}</h1>
            <p style="color: white; font-size: 16px;">Activo: {ticker} | Estrategia de Alta Confianza</p>
        </div>
    """, unsafe_allow_html=True)

    # Gráfico Profesional
    fig = go.Figure()
    # Velas
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    
    # Bandas de Bollinger (Sombreado)
    fig.add_trace(go.Scatter(x=df.index, y=df[bbu_name], line=dict(color='rgba(173, 216, 230, 0.3)'), name="B. Superior"))
    fig.add_trace(go.Scatter(x=df.index, y=df[bbl_name], fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)', line=dict(color='rgba(173, 216, 230, 0.3)'), name="B. Inferior"))
    
    # EMA 200 (Línea Roja Institucional)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF0000', width=3), name="EMA 200 (Muro)"))

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

else: st.error("No se encontraron datos para este símbolo.")
