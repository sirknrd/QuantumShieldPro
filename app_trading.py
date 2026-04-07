import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DE INTERFAZ PROFESIONAL ---
st.set_page_config(page_title="Quantum Shield Pro | Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR TÉCNICO (Blindado contra KeyErrors) ---
def realizar_analisis_profundo(df_input):
    df = df_input.copy()
    
    # EMAs y RSI
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # ADX con manejo dinámico de columnas
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx_df.iloc[:, 0] # Toma la primera columna (ADX) sin importar el nombre
    
    # Volumen Relativo
    df['Vol_Avg'] = ta.sma(df['Volume'], length=20)
    df['Rel_Vol'] = df['Volume'] / df['Vol_Avg']
    
    # Pivot Points (Cálculo Manual para evitar errores de librerías)
    h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
    df['PP'] = (h + l + c) / 3
    df['R1'] = (2 * df['PP']) - l
    df['S1'] = (2 * df['PP']) - h
    
    # Ichimoku
    ichi = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
    
    return pd.concat([df, ichi], axis=1).ffill()

# --- 3. BARRA LATERAL ---
st.sidebar.title("🛡️ Quantum Shield v4.0")
ticker = st.sidebar.text_input("Activo (Ej: BTC-USD, AAPL, SQM-B.SN)", value="BTC-USD").upper()
tf = st.sidebar.selectbox("Periodicidad", ["1h", "4h", "1d"], index=2)

# --- 4. PROCESAMIENTO DE DATOS ---
data = yf.download(ticker, period="400d", interval=tf, auto_adjust=True)

if not data.empty:
    # Limpieza de MultiIndex (Yahoo Finance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    df = realizar_analisis_profundo(data)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- CÁLCULO DE SCORE DE CONFIANZA (0-100) ---
    score = 0
    if last['Close'] > last['EMA_200']: score += 25
    if last['Close'] > last['EMA_50']: score += 15
    if 45 < last['RSI'] < 65: score += 20
    if last['Rel_Vol'] > 1.1: score += 20
    if last['ADX'] > 20: score += 20
    
    color_score = "#00FFA3" if score >= 70 else "#FFA500" if score >= 45 else "#FF4B4B"

    # --- 5. DASHBOARD EJECUTIVO ---
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, #161b22 0%, {color_score}22 100%); 
                    border-left: 8px solid {color_score}; padding: 25px; border-radius: 10px; margin-bottom: 20px;">
            <p style="color: #8b949e; margin: 0; font-weight: 600;">CONFIANZA DEL ALGORITMO</p>
            <h1 style="color: {color_score}; margin: 0; font-size: 4rem;">{score}%</h1>
        </div>
    """, unsafe_allow_html=True)

    # Métricas de Alta Densidad
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio", f"${last['Close']:,.2f}", f"{(last['Close']/prev['Close']-1)*100:.2f}%")
    m2.metric("Vol. Relativo", f"{last['Rel_Vol']:.2f}x", "Inyección" if last['Rel_Vol'] > 1.2 else "Normal")
    m3.metric("Fuerza (ADX)", f"{last['ADX']:.1f}")
    m4.metric("RSI", f"{last['RSI']:.1f}")

    # --- 6. GRÁFICO TÉCNICO ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF0000', width=2.5), name="EMA 200 (Muro)"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#00FFA3', width=1.5), name="EMA 50 (Cruce)"))
    
    # Resistencia y Soporte de hoy
    fig.add_hline(y=last['R1'], line_dash="dot", line_color="#ff4b4b", annotation_text="Resistencia")
    fig.add_hline(y=last['S1'], line_dash="dot", line_color="#00ffa3", annotation_text="Soporte")

    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- 7. PANEL DE DECISIÓN ---
    st.subheader("🔭 Análisis de Situación")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Fortalezas:**")
        if last['Close'] > last['EMA_200']: st.success("✅ Precio sobre el promedio institucional (Alcista).")
        if last['ADX'] > 25: st.success("✅ Tendencia con fuerza real detectada.")
    with col_b:
        st.write("**Riesgos:**")
        if last['RSI'] > 70: st.error("⚠️ Sobrecompra extrema. Riesgo de corrección.")
        if last['Close'] < last['S1']: st.error("⚠️ Soporte quebrado. Posible caída libre.")

else:
    st.error("Símbolo no válido o sin datos. Prueba con BTC-USD o AAPL.")
