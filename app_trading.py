import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="Quantum Shield | Market Scanner", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; }
    .stMetric { background-color: #111111; border: 1px solid #222222; padding: 15px; border-radius: 12px; }
    h3 { color: #8b949e !important; padding-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO ---
def calcular_score(ticker_name):
    try:
        d = yf.download(ticker_name, period="200d", interval="1d", progress=False, auto_adjust=True)
        if d.empty: return 0, "N/A"
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        
        close = d['Close'].iloc[-1]
        ema200 = ta.ema(d['Close'], length=200).iloc[-1]
        rsi = ta.rsi(d['Close'], length=14).iloc[-1]
        
        s = 0
        if close > ema200: s += 50
        if 40 < rsi < 65: s += 30
        if close > d['Close'].iloc[-5]: s += 20 # Momentum 5 días
        
        status = "COMPRA" if s >= 70 else "MANTENER" if s >= 40 else "VENDER"
        return s, status
    except:
        return 0, "Error"

# --- 3. BARRA LATERAL ---
st.sidebar.title("🛡️ Quantum Shield")
ticker_principal = st.sidebar.text_input("Analizar Activo", value="BTC-USD").upper()
tf = st.sidebar.selectbox("Frecuencia", ["1h", "4h", "1d"], index=2)

# --- 4. PANEL PRINCIPAL (ANÁLISIS INDIVIDUAL) ---
data = yf.download(ticker_principal, period="350d", interval=tf, auto_adjust=True)

if not data.empty:
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    # Indicadores rápidos
    df = data.copy()
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    last = df.iloc[-1]
    score_p = 0
    if last['Close'] > last['EMA_200']: score_p += 50
    if 40 < last['RSI'] < 65: score_p += 50
    
    # Colores Dinámicos
    if score_p >= 70: label, col_b, col_t = "COMPRAR ✅", "#00FF88", "#000000"
    elif score_p >= 40: label, col_b, col_t = "MANTENER ⚖️", "#FFC107", "#000000"
    else: label, col_b, col_t = "VENDER ⚠️", "#FF4B4B", "#FFFFFF"

    st.markdown(f"""
        <div style="background-color: {col_b}; padding: 25px; border-radius: 15px; text-align: center;">
            <h1 style="color: {col_t} !important; margin: 0; font-size: 3rem; font-weight: 900;">{label}</h1>
            <p style="color: {col_t}; margin: 0; font-weight: 600;">{ticker_principal} | Score: {score_p}%</p>
        </div>
    """, unsafe_allow_html=True)

    # Gráfico Simple
    fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=2), name="EMA 200"))
    fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- 5. NUEVA SECCIÓN: ESCÁNER DE OPORTUNIDADES ---
st.write("---")
st.subheader("🚀 Sugerencias del Mercado (Top Oportunidades)")

# Lista de seguimiento (Criptos + Chile + USA)
watch_list = ["ETH-USD", "SOL-USD", "AAPL", "NVDA", "SQM-B.SN", "CHILE.SN", "COPEC.SN"]

with st.spinner("Escaneando mercado..."):
    results = []
    for t in watch_list:
        val_score, val_status = calcular_score(t)
        results.append({"Activo": t, "Puntaje": val_score, "Sugerencia": val_status})
    
    reporte_df = pd.DataFrame(results).sort_values(by="Puntaje", ascending=False)

# Mostrar en formato de tabla limpia para celular
def color_status(val):
    color = '#00FF88' if val == "COMPRA" else '#FF4B4B' if val == "VENDER" else '#FFC107'
    return f'color: {color}; font-weight: bold'

st.table(reporte_df.style.applymap(color_status, subset=['Sugerencia']))

st.caption("Nota: El escáner utiliza temporalidad diaria para mayor precisión en la tendencia.")
