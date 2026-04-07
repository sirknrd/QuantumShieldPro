import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN VISUAL (MODO DARK PRO) ---
st.set_page_config(page_title="Quantum Shield | Terminal Completa", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; }
    [data-testid="stTable"] { background-color: #111111; border-radius: 10px; color: #FFFFFF !important; }
    td { color: #FFFFFF !important; font-size: 1rem; }
    th { color: #8b949e !important; }
    .stMetric { background-color: #111111; border: 1px solid #222222; padding: 15px; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. DICCIONARIO DE PATRONES (RECUPERADO) ---
INFO_PATRONES = {
    "doji": "⚖️ Doji: Indecisión total. El mercado no sabe hacia dónde ir. Espera confirmación.",
    "hammer": "🔨 Martillo: Rechazo de precios bajos. Posible rebote alcista inminente.",
    "engulfing": "🌊 Envolvente: Cambio de fuerza. La tendencia actual ha sido superada.",
    "morningstar": "🌅 Estrella: Patrón de giro alcista de alta fiabilidad.",
    "eveningstar": "🌇 Estrella: Patrón de giro bajista. Agotamiento de la subida."
}

# --- 3. MOTORES DE CÁLCULO ---
def procesar_datos_completos(ticker_name, t_frame="1d"):
    try:
        d = yf.download(ticker_name, period="300d", interval=t_frame, progress=False, auto_adjust=True)
        if d.empty: return None
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        
        df = d.copy()
        # Indicadores
        df['EMA_200'] = ta.ema(df['Close'], length=200)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
        
        # Pivotes y Volumen
        h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        df['PP'] = (h + l + c) / 3
        df['R1'] = (2 * df['PP']) - l
        df['S1'] = (2 * df['PP']) - h
        df['Vol_Avg'] = ta.sma(df['Volume'], length=20)
        
        # Patrones de Velas
        df_pat = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name=["doji", "hammer", "engulfing", "morningstar", "eveningstar"])
        
        return pd.concat([df, df_pat], axis=1).ffill()
    except: return None

# --- 4. INTERFAZ LATERAL ---
st.sidebar.title("🛡️ Quantum Shield Pro")
ticker_main = st.sidebar.text_input("Activo a Graficar", value="BTC-USD").upper()
intervalo = st.sidebar.selectbox("Frecuencia", ["1h", "4h", "1d"], index=2)
radar_input = st.sidebar.text_area("Radar (Separar por Comas)", value="ETH-USD, SOL-USD, NVDA, SQM-B.SN, CHILE.SN, AAPL")
watch_list = [x.strip().upper() for x in radar_input.split(",")]

# --- 5. PANEL DE ANÁLISIS PRINCIPAL ---
df = procesar_datos_completos(ticker_main, intervalo)

if df is not None:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Determinar Etiqueta
    score = 0
    if last['Close'] > last['EMA_200']: score += 50
    if 40 < last['RSI'] < 65: score += 50
    label = "COMPRAR ✅" if score >= 70 else "MANTENER ⚖️" if score >= 45 else "VENDER ⚠️"
    color_main = "#00FF88" if "COMPRAR" in label else "#FFC107" if "MANTENER" in label else "#FF4B4B"

    # Cartel Principal
    st.markdown(f"""
        <div style="background-color: {color_main}; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 15px;">
            <h1 style="color: #000 !important; margin: 0; font-size: 3.5rem; font-weight: 900;">{label}</h1>
            <p style="color: #000; margin: 0; font-weight: 700;">{ticker_main} | SCORE: {score}%</p>
        </div>
    """, unsafe_allow_html=True)

    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${last['Close']:,.2f}", f"{(last['Close']/prev['Close']-1)*100:.2f}%")
    c2.metric("RSI", f"{last['RSI']:.1f}")
    c3.metric("Fuerza ADX", f"{last['ADX']:.0f}")
    c4.metric("Vol. Rel.", f"{last['Volume']/last['Vol_Avg']:.1f}x")

    # Gráfico con Pivotes
    fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=2), name="EMA 200"))
    fig.add_hline(y=last['R1'], line_dash="dot", line_color="#555", annotation_text="Resistencia")
    fig.add_hline(y=last['S1'], line_dash="dot", line_color="#555", annotation_text="Soporte")
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- EXPLICACIÓN DE PATRONES DETECTADOS ---
    cols_pat = [c for c in df.columns if "CDL_" in c]
    patrones_hoy = [c for c in cols_pat if last[c] != 0]
    if patrones_hoy:
        st.subheader("🔍 Patrones de Velas Detectados")
        for p in patrones_hoy:
            key = p.replace("CDL_", "").lower()
            if key in INFO_PATRONES: st.info(INFO_PATRONES[key])

# --- 6. RADAR DE OPORTUNIDADES CON TEXTO BLANCO ---
st.write("---")
st.subheader("🚀 Radar de Oportunidades 🔥")

with st.spinner("Escaneando mercado..."):
    results = []
    for t in watch_list:
        d_r = procesar_datos_completos(t, "1d")
        if d_r is not None:
            l_r = d_r.iloc[-1]
            fuego = l_r['Volume'] > (l_r['Vol_Avg'] * 1.5)
            s_r = 50 if l_r['Close'] > l_r['EMA_200'] else 0
            s_r += 50 if 40 < l_r['RSI'] < 65 else 0
            
            results.append({
                "Activo": f"{t} 🔥" if fuego else t,
                "Puntaje": f"{s_r}%",
                "Sugerencia": "COMPRA" if s_r >= 70 else "MANTENER" if s_r >= 50 else "VENTA",
                "Stop Loss": f"${d_r['Low'].iloc[-3:].min():,.2f}"
            })
    
    reporte_df = pd.DataFrame(results)

def style_white_table(df):
    styled = df.style.set_properties(**{'color': 'white'})
    def color_stat(val):
        color = '#00FF88' if val == "COMPRA" else '#FF4B4B' if val == "VENTA" else '#FFC107'
        return f'color: {color}; font-weight: bold'
    return styled.map(color_stat, subset=['Sugerencia'])

st.table(style_white_table(reporte_df))
