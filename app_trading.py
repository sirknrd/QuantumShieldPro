import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime  # <-- ESTA ES LA LÍNEA QUE FALTABA

# --- 1. CONFIGURACIÓN DE UI ---
st.set_page_config(page_title="Quantum Shield Ultra v6.1", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #000000; }
    .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #222; border-radius: 8px; padding: 10px; }
    [data-testid="stTable"] { background-color: #000000; color: #ffffff !important; }
    table { border: 1px solid #333 !important; color: #ffffff !important; width: 100%; }
    thead tr th { background-color: #111 !important; color: #8b949e !important; text-align: left; }
    td { color: #ffffff !important; border-bottom: 1px solid #222 !important; padding: 8px !important; }
    h1, h2, h3, p, span { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO ---
def obtener_datos(ticker):
    try:
        # Descargamos 2 años para que la SMA 200 tenga datos suficientes
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # Medias exactas (Sincronizadas con la tabla final)
        periodos_tabla = [5, 20, 50, 200]
        for n in periodos_tabla:
            df[f'SMA{n}'] = ta.sma(df['Close'], length=n)
            df[f'EMA{n}'] = ta.ema(df['Close'], length=n)

        # Indicadores Clave
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0]
        
        # Pivotes Clásicos
        h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        df['PP'] = (h + l + c) / 3
        df['R1'], df['S1'] = (2 * df['PP']) - l, (2 * df['PP']) - h
        df['R2'], df['S2'] = df['PP'] + (h - l), df['PP'] - (h - l)
        
        # Volumen
        df['Vol_Avg'] = ta.sma(df['Volume'], length=20)
        df['Rel_Vol'] = df['Volume'] / df['Vol_Avg']
        
        return df.ffill()
    except: return None

# --- 3. LÓGICA DE SENTIMIENTO ---
def calcular_sentimiento(last):
    buy, sell = 0, 0
    # Comprobación de Medias
    for n in [5, 20, 50, 200]:
        if last['Close'] > last[f'SMA{n}']: buy += 1
        else: sell += 1
    
    # Comprobación de RSI
    if last['RSI'] > 55: buy += 1
    if last['RSI'] < 45: sell += 1
    
    if sell > buy: return "VENTA FUERTE 🔴", "#FF4B4B"
    if buy > sell: return "COMPRA FUERTE 🟢", "#00FF88"
    return "NEUTRAL ⚖️", "#888888"

# --- 4. INTERFAZ ---
st.sidebar.title("🛡️ QUANTUM V6.1")
ticker_principal = st.sidebar.text_input("ACTIVO", value="CHILE.SN").upper()
radar_input = st.sidebar.text_area("RADAR", value="CHILE.SN, SQM-B.SN, COPEC.SN, AAPL, BTC-USD")
watch_list = [x.strip().upper() for x in radar_input.split(",")]

df = obtener_datos(ticker_principal)

if df is not None:
    last = df.iloc[-1]
    status_text, status_color = calcular_sentimiento(last)
    
    # Banner Principal (Corregido con la fecha actual)
    st.markdown(f"""
        <div style="border-left: 10px solid {status_color}; background-color: #0a0a0a; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin:0; font-size: 2.8rem;">{ticker_principal}: {status_text}</h1>
            <p style="margin:0; opacity: 0.7;">Análisis Técnico Profesional | {datetime.now().strftime('%d/%m/%Y')}</p>
        </div>
    """, unsafe_allow_html=True)

    # Métricas Superiores
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRECIO", f"${last['Close']:,.2f}")
    c2.metric("RSI (14D)", f"{last['RSI']:.1f}")
    c3.metric("FUERZA ADX", f"{last['ADX']:.1f}")
    c4.metric("VOL. RELATIVO", f"{last['Rel_Vol']:.1f}x")

    # Gráfico Profesional
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index[-120:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df['SMA50'][-120:], line=dict(color='#00BFFF', width=1), name="MA 50"))
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df['SMA200'][-120:], line=dict(color='#FF4B4B', width=2), name="MA 200"))
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='black', plot_bgcolor='black')
    st.plotly_chart(fig, use_container_width=True)

    # --- TABLAS TÉCNICAS ---
    st.write("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📈 Medias Móviles")
        # Aseguramos simetría absoluta en las listas
        periodos = ["5", "20", "50", "200"]
        smas = [f"{last[f'SMA{n}']:,.2f}" for n in [5, 20, 50, 200]]
        emas = [f"{last[f'EMA{n}']:,.2f}" for n in [5, 20, 50, 200]]
        
        medias
