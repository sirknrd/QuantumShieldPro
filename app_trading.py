import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN DE PANTALLA ---
st.set_page_config(page_title="Quantum Shield Ultra | Inteligencia Adaptativa", page_icon="🛡️", layout="wide")

# --- 2. MOTOR DE IA ADAPTATIVA (Lógica de Alta Confianza) ---
def calcular_señal_ultra(last, prev, df_cols):
    try:
        puntos = 0
        close = float(last['Close'])
        adx = float(last['ADX_14'])
        rsi = float(last['RSI'])
        ema200 = float(last['EMA_200'])
        
        # Identificación de columnas dinámicas
        bbu = float(last[[c for c in df_cols if 'BBU' in str(c)][0]])
        bbl = float(last[[c for c in df_cols if 'BBL' in str(c)][0]])
        isa = float(last[[c for c in df_cols if 'ISA' in str(c)][0]])
        isb = float(last[[c for c in df_cols if 'ISB' in str(c)][0]])

        # --- NIVEL 1: INTERRUPTOR DE RÉGIMEN (ADX) ---
        if adx > 25:
            # MODO TENDENCIA: Seguimos a las instituciones
            if close > ema200: puntos += 2
            if close > isa and close > isb: puntos += 1
            # Filtro de sobreextensión
            if rsi > 75: puntos -= 1 
        else:
            # MODO RANGO: Reversión a la media (Bollinger)
            if close < bbl: puntos += 2 # Compra en soporte de volatilidad
            if close > bbu: puntos -= 2 # Venta en resistencia de volatilidad

        # --- NIVEL 2: FILTRO DE DIVERGENCIA ---
        # Si el precio sube pero el RSI baja, restamos confianza
        if close > float(prev['Close']) and rsi < float(prev['RSI']):
            puntos -= 1

        # --- NIVEL 3: RESULTADO ---
        if puntos >= 2: return "COMPRA (ALTA CONFIANZA)", "#00FFA3"
        elif puntos <= -2: return "VENTA / PRECAUCIÓN", "#FF4B4B"
        else: return "MANTENER / NEUTRAL", "#FFA500"
    except:
        return "CALCULANDO...", "#8B949E"

# --- 3. BARRA LATERAL (SELECTORES) ---
st.sidebar.title("🛡️ Sistema Quantum")
ticker_input = st.sidebar.text_input("Activo (Ej: BTC, SQM-B.SN, NVDA)", value="BTC").upper()

tf_map = {"1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d"}
tf_label = st.sidebar.selectbox("Temporalidad", list(tf_map.keys()), index=2)
tf = tf_map[tf_label]

# Formateo automático de Ticker
if len(ticker_input) <= 4 and "-" not in ticker_input and "." not in ticker_input:
    ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL"] else ticker_input
else: ticker = ticker_input

# --- 4. OBTENCIÓN DE DATOS ---
df_raw = yf.download(ticker, period="400d", interval=tf, auto_adjust=True)

if not df_raw.empty:
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- INDICADORES AVANZADOS ---
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    
    # Bollinger + Ichimoku
    df = pd.concat([df, ta.bbands(df['Close'], length=20, std=2), ta.ichimoku(df['High'], df['Low'], df['Close'])[0]], axis=1).ffill()
    
    # Columnas para visualización
    last = df.iloc[-1]
    prev = df.iloc[-2]
    bbu_col = [c for c in df.columns if 'BBU' in str(c)][0]
    bbl_col = [c for c in df.columns if 'BBL' in str(c)][0]
    isa_col = [c for c in df.columns if 'ISA' in str(c)][0]
    isb_col = [c for c in df.columns if 'ISB' in str(c)][0]
    
    # Variación y Señal
    cambio = float(last['Close']) - float(prev['Close'])
    var_pct = (cambio / float(prev['Close'])) * 100
    rec, color = calcular_señal_ultra(last, prev, df.columns)

    # --- 5. INTERFAZ: DASHBOARD EJECUTIVO ---
    st.markdown(f"""
        <div style="background-color: {color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; border: 2px solid rgba(0,0,0,0.1);">
            <h1 style="color: #000000 !important; margin: 0; font-weight: 900; font-size: 45px; text-transform: uppercase;">{rec}</h1>
            <p style="color: #000000 !important; margin: 0; font-weight: 700; opacity: 0.8;">SISTEMA ADAPTATIVO QUANTUM SHIELD</p>
        </div>
    """, unsafe_allow_html=True)

    # Celdas de Datos en Tiempo Real
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio Actual", f"${float(last['Close']):,.2f}", f"{var_pct:.2f}%")
    m2.metric("Fuerza Tendencia (ADX)", f"{float(last['ADX_14']):.1f}")
    m3.metric("RSI (Momento)", f"{float(last['RSI']):.1f}")
    m4.metric("Estado Mercado", "Tendencia" if float(last['ADX_14']) > 25 else "Rango / Lateral")

    # --- 6. GRÁFICO INTEGRAL AVANZADO ---
    fig = go.Figure()

    # Capa 1: Nube Ichimoku (Proyección de Tendencia)
    fig.add_trace(go.Scatter(x=df.index, y=df[isa_col], line=dict(color='rgba(0, 255, 163, 0.15)'), name="Senkou A"))
    fig.add_trace(go.Scatter(x=df.index, y=df[isb_col], line=dict(color='rgba(255, 75, 75, 0.15)'), fill='tonexty', name="Nube Kumo"))

    # Capa 2: Bandas de Bollinger (Volatilidad)
    fig.add_trace(go.Scatter(x=df.index, y=df[bbu_col], line=dict(color='rgba(173, 216, 230, 0.2)', dash='dot'), name="B. Superior"))
    fig.add_trace(go.Scatter(x=df.index, y=df[bbl_col], line=dict(color='rgba(173, 216, 230, 0.2)', dash='dot'), name="B. Inferior"))

    # Capa 3: Velas Japonesas
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))

    # Capa 4: EMA 200 (Institucional)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF0000', width=3), name="EMA 200 (Muro)"))

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=750, 
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # Info Técnica Inferior
    with st.expander("Ver detalle de indicadores"):
        st.write(f"**Límite Superior Bollinger:** ${float(last[bbu_col]):,.2f}")
        st.write(f"**Límite Inferior Bollinger:** ${float(last[bbl_col]):,.2f}")
        st.write(f"**Valor Nube Ichimoku:** ${float(last[isa_col]):,.2f}")

else:
    st.error("Error crítico: No se pudieron obtener datos del mercado.")
