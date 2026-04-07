import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Pattern Expert", page_icon="🛡️", layout="wide")

# Ocultar elementos de Streamlit para que parezca App Nativa
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE DETECCIÓN DE PATRONES ---
def detectar_patrones(df):
    # Usamos pandas_ta para reconocer patrones de velas
    # Retorna 100 para alcista, -100 para bajista, 0 para nada
    patrones = df.ta.cdl_pattern(name=["doji", "engulfing", "hammer", "morningstar", "eveningstar"])
    return patrones

# --- 3. LÓGICA DE DECISIÓN MEJORADA ---
def calcular_señal_expert(last, df_cols, patrones_last):
    try:
        puntos = 0
        close = float(last['Close'])
        adx = float(last['ADX_14'])
        ema200 = float(last['EMA_200'])
        
        # Filtro de Tendencia/Rango
        if adx > 25:
            if close > ema200: puntos += 2
            else: puntos -= 2
        
        # SUMAR PUNTOS POR PATRONES DE VELAS
        # Buscamos cualquier patrón activo en la última vela
        for col in patrones_last.index:
            val = patrones_last[col]
            if val > 0: puntos += 2  # Patrón Alcista detectado
            if val < 0: puntos -= 2  # Patrón Bajista detectado

        if puntos >= 2: return "COMPRA CONFIRMADA ✅", "#00FFA3"
        elif puntos <= -2: return "VENTA / SALIDA ⚠️", "#FF4B4B"
        return "ESPERAR / NEUTRAL ⚖️", "#FFA500"
    except: return "ANALIZANDO...", "#8B949E"

# --- 4. INTERFAZ Y DATOS ---
st.sidebar.title("🛡️ Quantum Shield")
ticker_input = st.sidebar.text_input("Activo", value="BTC").upper()
tf = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=2)

ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL"] and "-" not in ticker_input else ticker_input

df = yf.download(ticker, period="350d", interval=tf, auto_adjust=True)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Cálculo de Indicadores
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    
    # Bandas y Nube
    df = pd.concat([df, ta.bbands(df['Close']), ta.ichimoku(df['High'], df['Low'], df['Close'])[0]], axis=1).ffill()
    
    # Detección de Patrones
    df_patrones = detectar_patrones(df)
    
    last = df.iloc[-1]
    last_patron = df_patrones.iloc[-1]
    rec, color = calcular_señal_expert(last, df.columns, last_patron)

    # --- 5. VISUALIZACIÓN ---
    st.markdown(f"""
        <div style="background-color: {color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 15px;">
            <h1 style="color: #000000 !important; margin: 0; font-weight: 850; font-size: 35px;">{rec}</h1>
        </div>
    """, unsafe_allow_html=True)

    # Métricas Críticas
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio", f"${float(last['Close']):,.2f}")
    c2.metric("RSI", f"{float(last['RSI']):.1f}")
    c3.metric("ADX", f"{float(last['ADX_14']):.1f}")

    # Lista de Patrones Detectados hoy
    patrones_activos = [col for col, val in last_patron.items() if val != 0]
    if patrones_activos:
        st.info(f"📊 **Patrones Detectados:** {', '.join(patrones_activos).replace('CDL_', '')}")

    # --- 6. GRÁFICO CON ANOTACIONES ---
    fig = go.Figure()
    
    # Nube Ichimoku
    isa = [c for c in df.columns if 'ISA' in str(c)][0]
    isb = [c for c in df.columns if 'ISB' in str(c)][0]
    fig.add_trace(go.Scatter(x=df.index, y=df[isa], line=dict(color='rgba(0, 255, 163, 0.1)'), name="Cloud"))
    fig.add_trace(go.Scatter(x=df.index, y=df[isb], line=dict(color='rgba(255, 75, 75, 0.1)'), fill='tonexty'))
    
    # Velas
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
    
    # EMA 200
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='#FF0000', width=2), name="EMA200"))

    # Marcar donde hubo patrones en el pasado (puntos sobre las velas)
    for col in df_patrones.columns:
        hits = df_patrones[df_patrones[col] != 0]
        if not hits.empty:
            fig.add_trace(go.Scatter(x=hits.index, y=df.loc[hits.index, 'High'] * 1.02, 
                                     mode='markers', marker=dict(symbol='diamond', size=8), 
                                     name=col.replace('CDL_', '')))

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, 
                      margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    st.error("Activo no encontrado")
