import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="Quantum Shield | Pro Radar", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; }
    /* Forzar texto blanco en tablas y métricas */
    [data-testid="stTable"] { 
        background-color: #111111; 
        border-radius: 10px; 
        color: #FFFFFF !important; 
    }
    th { color: #8b949e !important; }
    td { color: #FFFFFF !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO ---
def analizar_mercado_pro(ticker_name):
    try:
        d = yf.download(ticker_name, period="250d", interval="1d", progress=False, auto_adjust=True)
        if d.empty: return 0, "N/A", False, 0
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        
        close = d['Close'].iloc[-1]
        vol_hoy = d['Volume'].iloc[-1]
        vol_med = d['Volume'].rolling(20).mean().iloc[-1]
        ema200 = ta.ema(d['Close'], length=200).iloc[-1]
        
        # Alerta de Fuego (Volumen > 150%)
        fuego = vol_hoy > (vol_med * 1.5)
        
        # Stop Loss Sugerido (Mínimo de 3 días)
        stop_loss = d['Low'].iloc[-3:].min()
        
        # Score
        s = 0
        if close > ema200: s += 50
        if close > d['Close'].iloc[-10]: s += 50
        
        status = "COMPRA" if s >= 70 else "MANTENER" if s >= 50 else "VENDER"
        return s, status, fuego, stop_loss
    except:
        return 0, "Error", False, 0

# --- 3. INTERFAZ LATERAL ---
st.sidebar.title("🛡️ Quantum Pro")
ticker_principal = st.sidebar.text_input("Activo Principal", value="BTC-USD").upper()
activos_radar = st.sidebar.text_area("Lista Radar (Comas)", 
                                     value="ETH-USD, SOL-USD, NVDA, AAPL, SQM-B.SN, CHILE.SN")
watch_list = [x.strip().upper() for x in activos_radar.split(",")]

# --- 4. PANEL DE CONTROL ---
data = yf.download(ticker_principal, period="300d", interval="1d", auto_adjust=True)
if not data.empty:
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    df = data.copy()
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    label = "ALCISTA" if df['Close'].iloc[-1] > df['EMA_200'].iloc[-1] else "BAJISTA"
    color_label = "#00FF88" if label == "ALCISTA" else "#FF4B4B"

    st.markdown(f"<h1 style='text-align:center; color:{color_label};'>{ticker_principal} | {label}</h1>", unsafe_allow_html=True)
    
    fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=2)))
    fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- 5. TABLA DE OPORTUNIDADES (TEXTO BLANCO) ---
st.write("---")
st.subheader("🚀 Radar de Oportunidades 🔥")

with st.spinner("Analizando..."):
    results = []
    for t in watch_list:
        score, status, fuego, sl = analizar_mercado_pro(t)
        nombre = f"{t} 🔥" if fuego else t
        results.append({
            "Activo": nombre, 
            "Puntaje": f"{score}%", 
            "Sugerencia": status,
            "Stop Loss": f"${sl:,.2f}"
        })
    
    reporte_df = pd.DataFrame(results)

# Aplicar estilos: Texto blanco general, color solo en 'Sugerencia'
def style_results(df):
    # Crear una copia para aplicar estilos
    styled = df.style.set_properties(**{'color': 'white'})
    
    # Aplicar color solo a la columna de Sugerencia
    def color_status(val):
        if val == "COMPRA": return 'color: #00FF88; font-weight: bold'
        if val == "VENDER": return 'color: #FF4B4B; font-weight: bold'
        return 'color: #FFC107; font-weight: bold'
    
    return styled.map(color_status, subset=['Sugerencia'])

st.table(style_results(reporte_df))

st.caption("Nota: El Stop Loss se calcula basado en el mínimo técnico de las últimas 72 horas.")
