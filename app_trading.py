import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Shield | Ichimoku Pro", page_icon="🛡️", layout="wide")

# --- 2. MOTOR DE RECOMENDACIÓN ---
def calcular_señal_maestra(last, df_cols):
    try:
        puntos = 0
        close = float(last['Close'])
        # Buscar columnas de Ichimoku dinámicamente
        isa_col = [c for c in df_cols if 'ISA' in str(c)][0]
        isb_col = [c for c in df_cols if 'ISB' in str(c)][0]
        
        span_a = float(last[isa_col])
        span_b = float(last[isb_col])
        
        # Test 1: Proyección Ichimoku (Precio vs Nube)
        if close > span_a and close > span_b: puntos += 2
        elif close < span_a and close < span_b: puntos -= 2
        
        # Test 2: RSI
        if 50 < float(last['RSI']) < 70: puntos += 1
        
        if puntos >= 2: return "COMPRA FUERTE", "#00FFA3"
        elif puntos <= -2: return "VENTA / ALERTA", "#FF4B4B"
        else: return "NEUTRAL / ESPERAR", "#FFA500"
    except: return "ANALIZANDO", "#8B949E"

# --- 3. BARRA LATERAL ---
st.sidebar.title("🛡️ Quantum Pro")
ticker_input = st.sidebar.text_input("Activo", value="BTC").upper()

if len(ticker_input) <= 4 and "-" not in ticker_input and "." not in ticker_input:
    ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL"] else ticker_input
else: ticker = ticker_input

# --- 4. DATOS Y CÁLCULOS ---
df_raw = yf.download(ticker, period="400d", interval="1d", auto_adjust=True)

if not df_raw.empty:
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Bandas de Bollinger e Ichimoku
    bb = ta.bbands(df['Close'], length=20, std=2)
    ichi = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
    
    # --- CORRECCIÓN DEL ERROR LINEA 56 ---
    df = pd.concat([df, bb, ichi], axis=1).ffill() 
    # -------------------------------------
    
    # Nombres de columnas dinámicos
    bbu = [c for c in df.columns if 'BBU' in str(c)][0]
    bbl = [c for c in df.columns if 'BBL' in str(c)][0]
    isa = [c for c in df.columns if 'ISA' in str(c)][0]
    isb = [c for c in df.columns if 'ISB' in str(c)][0]
    
    last = df.iloc[-1]
    rec, color = calcular_señal_maestra(last, df.columns)

    # --- 5. INTERFAZ (Texto Negro sobre Color) ---
    st.markdown(f"""
        <div style="background-color: {color}; border-radius: 15px; padding: 25px; text-align: center; margin-bottom: 25px; box-shadow: 0px 4px 10px rgba(0,0,0,0.3);">
            <h1 style="color: #000000 !important; margin: 0; font-weight: 900; font-size: 50px; line-height: 1.2;">{rec}</h1>
            <p style="color: #000000 !important; margin: 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Proyección Técnica: {ticker}</p>
        </div>
    """, unsafe_allow_html=True)

    # Gráfico con Proyección Futura
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[isa], line=dict(color='rgba(0, 255, 163, 0.2)'), name="Senkou A"))
    fig.add_trace(go.Scatter(x=df.index, y=df[isb], line=dict(color='rgba(255, 75, 75, 0.2)'), fill='tonexty', name="Nube Kumo"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF0000', width=3), name="EMA 200"))
    
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=700)
    st.plotly_chart(fig, use_container_width=True)

else: st.error("Símbolo no encontrado.")
