import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN DE UI PRO ---
st.set_page_config(page_title="Quantum Shield | Terminal Pro", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .metric-card { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LÓGICA DE ANALÍTICA AVANZADA ---
def realizar_analisis_profundo(df):
    # Indicadores Core
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    
    # Volumen Relativo (Media de 20 periodos)
    df['Vol_Avg'] = ta.sma(df['Volume'], length=20)
    df['Rel_Vol'] = df['Volume'] / df['Vol_Avg']
    
    # Pivot Points (Standard)
    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = (2 * df['PP']) - df['Low'].shift(1)
    df['S1'] = (2 * df['PP']) - df['High'].shift(1)
    
    # Bandas e Ichimoku
    bbands = ta.bbands(df['Close'], length=20, std=2)
    ichi = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
    return pd.concat([df, bbands, ichi], axis=1).ffill()

# --- 3. BARRA LATERAL ---
st.sidebar.title("🛡️ Quantum Pro v3.0")
ticker_symbol = st.sidebar.text_input("Activo", value="BTC-USD").upper()
intervalo = st.sidebar.selectbox("Frecuencia", ["1h", "4h", "1d"], index=2)

# --- 4. PROCESAMIENTO ---
data = yf.download(ticker_symbol, period="400d", interval=intervalo, auto_adjust=True)

if not data.empty:
    df = realizar_analisis_profundo(data)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- CÁLCULO DE SCORE DE CONFIANZA ---
    score = 0
    if last['Close'] > last['EMA_200']: score += 20
    if last['Close'] > last['EMA_50']: score += 15
    if 40 < last['RSI'] < 65: score += 15
    if last['Rel_Vol'] > 1.2: score += 20
    if last['ADX'] > 25: score += 15
    if last['Close'] > last['ISA_9']: score += 15
    
    color_score = "#00FFA3" if score > 60 else "#FFA500" if score > 40 else "#FF4B4B"

    # --- 5. VISUALIZACIÓN DASHBOARD ---
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, #161b22 0%, {color_score}33 100%); border-left: 5px solid {color_score}; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
            <h1 style="margin:0; font-size: 1.5rem; color: #8b949e;">CONFIANZA DEL SISTEMA</h1>
            <h1 style="margin:0; font-size: 3.5rem; color: {color_score};">{score}%</h1>
        </div>
    """, unsafe_allow_html=True)

    # Métricas Pro
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precio", f"${last['Close']:,.2f}", f"{(last['Close']/prev['Close']-1)*100:.2f}%")
    with c2: st.metric("Vol. Relativo", f"{last['Rel_Vol']:.2x}", "Alto" if last['Rel_Vol'] > 1 else "Bajo")
    with c3: st.metric("RSI", f"{last['RSI']:.1f}", "Sobrecompra" if last['RSI'] > 70 else "Ok")
    with c4: st.metric("Tendencia (ADX)", f"{last['ADX']:.0f}", "Fuerte" if last['ADX'] > 25 else "Débil")

    # --- 6. GRÁFICO PROFESIONAL ---
    fig = go.Figure()
    # Velas
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF4B4B', width=2), name="EMA 200"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#00FFA3', width=1.5), name="EMA 50"))
    # Pivots (Líneas horizontales en el último precio)
    fig.add_hline(y=last['R1'], line_dash="dot", line_color="#888", annotation_text="Resistencia")
    fig.add_hline(y=last['S1'], line_dash="dot", line_color="#888", annotation_text="Soporte")

    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- 7. PANEL DE INTELIGENCIA ---
    st.subheader("🔭 Análisis de Escenario")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Señales Técnicas:**")
        if last['Close'] > last['R1']: st.warning("⚠️ Precio rompiendo Resistencia (R1). Posible FOMO.")
        if last['Rel_Vol'] > 2: st.info("🔥 Volumen inusual detectado. Instituciones operando.")
        if last['EMA_50'] > last['EMA_200']: st.success("📈 Cruce Dorado activo (Tendencia Alcista).")

    with col_b:
        st.markdown("**Estrategia Sugerida:**")
        if score > 70: st.write("✅ **ENTRADA:** Confirmación múltiple. Ratio Riesgo/Beneficio óptimo.")
        elif score > 40: st.write("⚖️ **OBSERVAR:** Esperar a que el volumen confirme el movimiento.")
        else: st.write("❌ **EVITAR:** Alta probabilidad de trampa o caída libre.")

else:
    st.error("Ticker no encontrado o sin conexión a Yahoo Finance.")
