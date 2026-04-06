import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Shield Pro | Terminal", page_icon="🛡️", layout="wide")

# --- 2. MOTOR DE RECOMENDACIÓN ---
def obtener_recomendacion(last):
    score = 0
    # Usamos comparaciones seguras de un solo valor
    if float(last['Close']) > float(last['EMA_200']): score += 1
    else: score -= 1
    
    if float(last['RSI']) > 55: score += 1
    elif float(last['RSI']) < 45: score -= 1
    
    if float(last['MFI']) > 55: score += 1
    elif float(last['MFI']) < 45: score -= 1
    
    if float(last['ADX_14']) > 25: score += 1
    
    m_h_col = [c for c in last.index if 'MACDh' in c][0]
    if float(last[m_h_col]) > 0: score += 1
    else: score -= 1

    if score >= 3: return "COMPRA FUERTE 🚀", "#00FFA3", "rgba(0, 255, 163, 0.1)"
    elif score <= -3: return "VENTA FUERTE 📉", "#FF4B4B", "rgba(255, 75, 75, 0.1)"
    else: return "MANTENER / NEUTRAL ⚖️", "#FFA500", "rgba(255, 165, 0, 0.1)"

# --- 3. SIDEBAR INTELIGENTE (CRYPTO + STOCKS) ---
st.sidebar.title("🛡️ Quantum Finder")
raw_ticker = st.sidebar.text_input("Activo (Ej: BTC, ETH, AAPL, SQM-B.SN)", value="BTC").upper()

# Auto-corrección para Cripto
if len(raw_ticker) <= 4 and "-" not in raw_ticker and ".SN" not in raw_ticker:
    ticker = f"{raw_ticker}-USD"
else:
    ticker = raw_ticker

tf = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=1)

# --- 4. EJECUCIÓN ---
df_raw = yf.download(ticker, period="365d", interval=tf)

if not df_raw.empty:
    df = df_raw.copy()
    
    # Limpieza de MultiIndex (Para evitar el ValueError)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Indicadores
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # Asegurar que no hay NaNs en la última fila para la lógica
    df = df.fillna(0)
    last = df.iloc[-1]
    
    rec, color, bg = obtener_recomendacion(last)

    # PANEL VISUAL
    st.markdown(f"""
        <div style="background-color: {bg}; border: 2px solid {color}; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 20px;">
            <h3 style="color: white; margin: 0;">ACTIVO: {ticker}</h3>
            <h1 style="color: {color}; margin: 10px 0; font-size: 40px;">{rec}</h1>
            <p style="color: gray;">Basado en confluencia técnica institucional</p>
        </div>
    """, unsafe_allow_html=True)

    # MÉTRICAS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${float(last['Close']):,.2f}")
    c2.metric("RSI", f"{float(last['RSI']):.1f}")
    c3.metric("MFI (Flujo)", f"{float(last['MFI']):.1f}")
    c4.metric("ADX (Fuerza)", f"{float(last['ADX_14']):.1f}")

    # GRÁFICO
    fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name="EMA 200", line=dict(color='red')))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(f"No se encontraron datos para {ticker}. Revisa el símbolo.")
