import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DE INTERFAZ ELITE ---
st.set_page_config(page_title="Quantum Shield Ultra", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; }
    [data-testid="stTable"] { background-color: #0c0c0c; border: 1px solid #222; color: #FFF !important; }
    td { color: #FFFFFF !important; font-size: 0.95rem; border-bottom: 1px solid #222 !important; }
    th { background-color: #111 !important; color: #8b949e !important; text-transform: uppercase; font-size: 0.8rem; }
    .stMetric { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE ANÁLISIS TÉCNICO AVANZADO ---
def realizar_escaneo_pro(ticker_name):
    try:
        df = yf.download(ticker_name, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # Medias Móviles (Igual al reporte que enviaste)
        for n in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{n}'] = ta.sma(df['Close'], length=n)
            df[f'EMA_{n}'] = ta.ema(df['Close'], length=n)

        # Indicadores Osciladores
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)

        # Puntos Pivote Clásicos
        h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        df['PP'] = (h + l + c) / 3
        df['R1'] = (2 * df['PP']) - l
        df['S1'] = (2 * df['PP']) - h
        df['R2'] = df['PP'] + (h - l)
        df['S2'] = df['PP'] - (h - l)

        return df.ffill()
    except: return None

# --- 3. LOGICA DE DECISIÓN (RESUMEN TÉCNICO) ---
def obtener_resumen(last):
    buy, sell = 0, 0
    # Medias Móviles
    for n in [5, 10, 20, 50, 100, 200]:
        if last['Close'] > last[f'SMA_{n}']: buy += 1
        else: sell += 1
    # Osciladores
    if last['RSI'] < 30: buy += 1
    elif last['RSI'] > 70: sell += 1
    if last['ADX'] > 25 and last['Close'] > last['SMA_50']: buy += 1
    
    if sell >= 10: return "VENTA FUERTE 🔴", "#FF4B4B"
    if sell > buy: return "VENTA 🟠", "#FF8C00"
    if buy >= 10: return "COMPRA FUERTE 🟢", "#00FF88"
    if buy > sell: return "COMPRA 🔵", "#00BFFF"
    return "NEUTRAL ⚪", "#888888"

# --- 4. INTERFAZ ---
st.sidebar.title("🛡️ Quantum Ultra v5")
ticker_input = st.sidebar.text_input("Activo", value="CHILE.SN").upper()
watch_list_input = st.sidebar.text_area("Radar", value="CHILE.SN, SQM-B.SN, COPEC.SN, AAPL, BTC-USD")
watch_list = [x.strip() for x in watch_list_input.split(",")]

df = realizar_escaneo_pro(ticker_input)

if df is not None:
    last = df.iloc[-1]
    resumen, color_res = obtener_resumen(last)

    # BANNER DE DECISIÓN (IGUAL A INVESTING)
    st.markdown(f"""
        <div style="background-color: {color_res}; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
            <p style="color: #000; margin: 0; font-weight: 800; opacity: 0.7;">RESUMEN TÉCNICO DIARIO</p>
            <h1 style="color: #000 !important; margin: 0; font-size: 3rem; font-weight: 900;">{resumen}</h1>
        </div>
    """, unsafe_allow_html=True)

    # MÉTRICAS CLAVE
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio Actual", f"${last['Close']:,.2f}")
    m2.metric("RSI (14)", f"{last['RSI']:.2f}")
    m3.metric("ADX (Tendencia)", f"{last['ADX']:.2f}")
    m4.metric("Punto Pivote", f"${last['PP']:.2f}")

    # GRÁFICO
    fig = go.Figure(go.Candlestick(x=df.index[-100:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA_50'][-100:], line=dict(color='#00FF88'), name="SMA 50"))
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA_200'][-100:], line=dict(color='#FF4B4B'), name="SMA 200"))
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # TABLAS DE DETALLE (Pivotes y Medias)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Puntos Pivote (Clásico)")
        pivots = {
            "Nivel": ["S2", "S1", "Punto Pivote", "R1", "R2"],
            "Precio": [f"{last['S2']:.2f}", f"{last['S1']:.2f}", f"{last['PP']:.2f}", f"{last['R1']:.2f}", f"{last['R2']:.2f}"]
        }
        st.table(pd.DataFrame(pivots))

    with col2:
        st.subheader("📈 Medias Móviles")
        medias = {
            "Periodo": ["MA5", "MA20", "MA50", "MA200"],
            "Simple (SMA)": [f"{last['SMA_5']:.2f}", f"{last['SMA_20']:.2f}", f"{last['SMA_50']:.2f}", f"{last['SMA_200']:.2f}"],
            "Exponencial (EMA)": [f"{last['EMA_5']:.2f}", f"{last['EMA_20']:.2f}", f"{last['EMA_50']:.2f}", f"{last['EMA_200']:.2f}"]
        }
        st.table(pd.DataFrame(medias))

# RADAR DE OPORTUNIDADES
st.write("---")
st.subheader("🚀 Radar Multimercado (Letras Blancas)")
results = []
for t in watch_list:
    d_r = realizar_escaneo_pro(t)
    if d_r is not None:
        l_r = d_r.iloc[-1]
        res, col = obtener_resumen(l_r)
        results.append({"Activo": t, "Precio": f"${l_r['Close']:,.2f}", "Sugerencia": res, "RSI": f"{l_r['RSI']:.1f}"})

res_df = pd.DataFrame(results)
st.table(res_df.style.set_properties(**{'color': 'white'}).map(lambda x: f"color: {color_res}; font-weight: bold" if "COMPRA" in str(x) else "", subset=['Sugerencia']))
