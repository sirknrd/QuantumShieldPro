import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. ESTILIZACIÓN DE INTERFAZ (DARK MODE PRO) ---
st.set_page_config(page_title="Quantum Shield | High Contrast", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 800 !important; color: #FFFFFF !important; }
    [data-testid="stMetricDelta"] { font-weight: 600 !important; }
    .stMetric { background-color: #111111; border: 1px solid #222222; padding: 15px; border-radius: 12px; }
    div[data-testid="stExpander"] { background-color: #111111; border: none; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR TÉCNICO BLINDADO ---
def realizar_analisis_pro(df_input):
    df = df_input.copy()
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # ADX Robusto
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx_df.iloc[:, 0]
    
    # Volumen y Pivotes
    df['Vol_Avg'] = ta.sma(df['Volume'], length=20)
    df['Rel_Vol'] = df['Volume'] / df['Vol_Avg']
    
    h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
    df['PP'] = (h + l + c) / 3
    df['R1'] = (2 * df['PP']) - l
    df['S1'] = (2 * df['PP']) - h
    
    # Ichimoku
    ichi = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
    return pd.concat([df, ichi], axis=1).ffill()

# --- 3. BARRA LATERAL ---
st.sidebar.title("🛡️ Quantum Shield")
ticker = st.sidebar.text_input("Activo", value="BTC-USD").upper()
tf = st.sidebar.selectbox("Frecuencia", ["1h", "4h", "1d"], index=2)

# --- 4. PROCESAMIENTO ---
data = yf.download(ticker, period="350d", interval=tf, auto_adjust=True)

if not data.empty:
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    df = realizar_analisis_pro(data)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- LÓGICA DE ETIQUETAS Y COLORES ---
    score = 0
    if last['Close'] > last['EMA_200']: score += 30
    if last['Close'] > last['EMA_50']: score += 20
    if 45 < last['RSI'] < 68: score += 20
    if last['Rel_Vol'] > 1.1: score += 15
    if last['ADX'] > 22: score += 15
    
    if score >= 70:
        etiqueta, color_bg, color_txt = "COMPRAR ✅", "#00FF88", "#000000"
    elif score >= 40:
        etiqueta, color_bg, color_txt = "MANTENER ⚖️", "#FFC107", "#000000"
    else:
        etiqueta, color_bg, color_txt = "VENDER ⚠️", "#FF4B4B", "#FFFFFF"

    # --- 5. VISUALIZACIÓN DEL SCORE ---
    st.markdown(f"""
        <div style="background-color: {color_bg}; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 20px; box-shadow: 0px 10px 30px {color_bg}33;">
            <p style="color: {color_txt}; margin: 0; font-weight: 700; letter-spacing: 2px; font-size: 0.9rem; opacity: 0.8;">SISTEMA QUANTUM SHIELD</p>
            <h1 style="color: {color_txt} !important; margin: 0; font-size: 4.5rem; font-weight: 900; line-height: 1;">{etiqueta}</h1>
            <p style="color: {color_txt}; margin: 5px 0 0 0; font-weight: 600; font-size: 1.2rem;">Confianza Técnica: {score}%</p>
        </div>
    """, unsafe_allow_html=True)

    # Métricas Críticas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio", f"${last['Close']:,.2f}", f"{(last['Close']/prev['Close']-1)*100:.2f}%")
    m2.metric("Vol. Rel.", f"{last['Rel_Vol']:.2f}x")
    m3.metric("RSI", f"{last['RSI']:.1f}")
    m4.metric("ADX", f"{last['ADX']:.0f}")

    # --- 6. GRÁFICO PROFESIONAL ---
    fig = go.Figure()
    # Velas con colores mejorados
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                 increasing_line_color='#00FF88', decreasing_line_color='#FF4B4B', name="Precio"))
    # Indicadores
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF4B4B', width=2), name="EMA 200"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#00FF88', width=1.5), name="EMA 50"))
    
    # Niveles de Soporte/Resistencia
    fig.add_hline(y=last['R1'], line_dash="dash", line_color="#555", annotation_text="Resistencia")
    fig.add_hline(y=last['S1'], line_dash="dash", line_color="#555", annotation_text="Soporte")

    fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False, 
                      margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- 7. PANEL DE INTELIGENCIA RÁPIDA ---
    with st.container():
        st.subheader("🔭 Análisis de Situación")
        col_a, col_b = st.columns(2)
        with col_a:
            if last['Close'] > last['EMA_200']: st.success("✅ Tendencia Primaria Alcista")
            if last['Rel_Vol'] > 1.2: st.info("🔥 Volumen superior al promedio")
        with col_b:
            if last['RSI'] > 70: st.error("⚠️ Alerta de Sobrecompra")
            if last['Close'] < last['S1']: st.error("⚠️ Ruptura de Soporte")

else:
    st.error("Error al obtener datos. Verifica el Ticker.")
