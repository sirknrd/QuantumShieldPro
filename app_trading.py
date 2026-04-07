import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="Quantum Shield | Volume Scanner", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; }
    .stMetric { background-color: #111111; border: 1px solid #222222; padding: 15px; border-radius: 12px; }
    [data-testid="stTable"] { background-color: #111111; border-radius: 10px; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO AVANZADO ---
def analizar_mercado(ticker_name):
    try:
        d = yf.download(ticker_name, period="250d", interval="1d", progress=False, auto_adjust=True)
        if d.empty: return 0, "N/A", False
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        
        # Datos actuales
        close = d['Close'].iloc[-1]
        vol_hoy = d['Volume'].iloc[-1]
        vol_med = d['Volume'].rolling(20).mean().iloc[-1]
        ema200 = ta.ema(d['Close'], length=200).iloc[-1]
        rsi = ta.rsi(d['Close'], length=14).iloc[-1]
        
        # Alerta de Fuego (Volumen > 150% del promedio)
        volumen_explosivo = vol_hoy > (vol_med * 1.5)
        
        # Scoring
        s = 0
        if close > ema200: s += 50
        if 40 < rsi < 65: s += 30
        if close > d['Close'].iloc[-10]: s += 20
        
        status = "COMPRA" if s >= 70 else "MANTENER" if s >= 45 else "VENDER"
        return s, status, volumen_explosivo
    except:
        return 0, "Error", False

# --- 3. BARRA LATERAL ---
st.sidebar.title("🛡️ Quantum Shield")
ticker_principal = st.sidebar.text_input("Activo Principal", value="BTC-USD").upper()
activos_radar = st.sidebar.text_area("Radar Personalizado (Comas)", 
                                     value="ETH-USD, SOL-USD, NVDA, AAPL, SQM-B.SN, CHILE.SN, COPEC.SN")
watch_list = [x.strip().upper() for x in activos_radar.split(",")]

# --- 4. PANEL PRINCIPAL ---
data = yf.download(ticker_principal, period="350d", interval="1d", auto_adjust=True)

if not data.empty:
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    df = data.copy()
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    last = df.iloc[-1]
    color_banner = "#00FF88" if last['Close'] > last['EMA_200'] else "#FF4B4B"

    st.markdown(f"""
        <div style="background-color: {color_banner}; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: #000; margin: 0; font-size: 2.5rem; font-weight: 900;">{ticker_principal}</h1>
            <p style="color: #000; margin: 0; font-weight: 600;">Analizando Tendencia Diaria</p>
        </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=2), name="EMA 200"))
    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 5. SECCIÓN: RADAR CON INDICADOR DE FUEGO 🔥 ---
st.write("---")
st.subheader("🚀 Radar de Oportunidades 🔥")

with st.spinner("Escaneando volumen e indicadores..."):
    results = []
    for t in watch_list:
        score, status, fuego = analizar_mercado(t)
        nombre_display = f"{t} 🔥" if fuego else t
        results.append({"Activo": nombre_display, "Puntaje": score, "Sugerencia": status})
    
    reporte_df = pd.DataFrame(results).sort_values(by="Puntaje", ascending=False)

# Estilo de tabla
def style_status(val):
    color = '#00FF88' if val == "COMPRA" else '#FF4B4B' if val == "VENDER" else '#FFC107'
    return f'color: {color}; font-weight: bold'

# Usamos .map() para compatibilidad con Pandas 2.0+
st.table(reporte_df.style.map(style_status, subset=['Sugerencia']))

st.caption("🔥 = Volumen inusual detectado hoy (Instituciones entrando).")
