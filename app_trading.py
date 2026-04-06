import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Shield | Dashboard Pro", page_icon="🛡️", layout="wide")

# --- 2. MOTOR DE RECOMENDACIÓN ---
def calcular_señal_maestra(last, df_cols):
    try:
        puntos = 0
        close = float(last['Close'])
        isa = float(last[[c for c in df_cols if 'ISA' in str(c)][0]])
        isb = float(last[[c for c in df_cols if 'ISB' in str(c)][0]])
        
        if close > isa and close > isb: puntos += 2
        elif close < isa and close < isb: puntos -= 2
        if 50 < float(last['RSI']) < 70: puntos += 1
        
        if puntos >= 2: return "COMPRA FUERTE", "#00FFA3"
        elif puntos <= -2: return "VENTA / ALERTA", "#FF4B4B"
        else: return "NEUTRAL / ESPERAR", "#FFA500"
    except: return "ANALIZANDO", "#8B949E"

# --- 3. BARRA LATERAL (SELECTORES) ---
st.sidebar.title("🛡️ Configuración")
ticker_input = st.sidebar.text_input("Activo", value="BTC").upper()

# --- NUEVO: SELECTOR DE TEMPORALIDAD ---
tf_map = {"1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d", "1 Semana": "1wk"}
tf_label = st.sidebar.selectbox("Temporalidad", list(tf_map.keys()), index=2)
tf = tf_map[tf_label]

if len(ticker_input) <= 4 and "-" not in ticker_input and "." not in ticker_input:
    ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL"] else ticker_input
else: ticker = ticker_input

# --- 4. DATOS Y CÁLCULOS ---
df_raw = yf.download(ticker, period="400d", interval=tf, auto_adjust=True)

if not df_raw.empty:
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    bb = ta.bbands(df['Close'], length=20, std=2)
    ichi = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
    df = pd.concat([df, bb, ichi], axis=1).ffill()
    
    # Identificación de columnas
    last = df.iloc[-1]
    prev = df.iloc[-2]
    bbu_col = [c for c in df.columns if 'BBU' in str(c)][0]
    bbl_col = [c for c in df.columns if 'BBL' in str(c)][0]
    
    # Variación %
    cambio = float(last['Close']) - float(prev['Close'])
    variacion_pct = (cambio / float(prev['Close'])) * 100
    
    rec, color = calcular_señal_maestra(last, df.columns)

    # --- 5. INTERFAZ: TABLERO DE MÉTRICAS ---
    st.markdown(f"""
        <div style="background-color: {color}; border-radius: 10px; padding: 15px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: #000000 !important; margin: 0; font-weight: 900;">{rec}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Celdas de Valores Relevantes
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precio Actual", f"${float(last['Close']):,.2f}", f"{variacion_pct:.2f}%")
    m2.metric("RSI (14)", f"{float(last['RSI']):.1f}")
    m3.metric("EMA 200", f"{float(last['EMA_200']):,.1f}")
    m4.metric("Banda Superior", f"{float(last[bbu_col]):,.1f}")
    m5.metric("Banda Inferior", f"{float(last[bbl_col]):,.1f}")

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF0000', width=2), name="EMA 200"))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

else: st.error("No se pudo cargar el activo.")
