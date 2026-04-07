import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN DE UI (TERMINAL BLACK ULTRA) ---
st.set_page_config(page_title="Quantum Shield Ultra v6.2", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #000000; }
    .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #222; border-radius: 8px; padding: 10px; }
    [data-testid="stTable"] { background-color: #000000; color: #ffffff !important; }
    table { border: 1px solid #333 !important; color: #ffffff !important; width: 100%; }
    thead tr th { background-color: #111 !important; color: #8b949e !important; text-align: left; }
    td { color: #ffffff !important; border-bottom: 1px solid #222 !important; padding: 8px !important; }
    h1, h2, h3, p, span { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO PROFESIONAL ---
def obtener_datos_completos(ticker):
    try:
        # Descarga de datos extendida para indicadores de largo plazo
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # Cálculo Sincronizado de Medias
        per = [5, 20, 50, 200]
        for n in per:
            df[f'SMA{n}'] = ta.sma(df['Close'], length=n)
            df[f'EMA{n}'] = ta.ema(df['Close'], length=n)

        # Indicadores de Fuerza y Momentum
        df['RSI'] = ta.rsi(df['Close'], length=14)
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx_data.iloc[:, 0]
        
        # Puntos Pivote Diarios
        h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        df['PP'] = (h + l + c) / 3
        df['R1'], df['S1'] = (2 * df['PP']) - l, (2 * df['PP']) - h
        df['R2'], df['S2'] = df['PP'] + (h - l), df['PP'] - (h - l)
        
        # Volumen Relativo
        df['Vol_Avg'] = ta.sma(df['Volume'], length=20)
        df['Rel_Vol'] = df['Volume'] / df['Vol_Avg']
        
        return df.ffill()
    except: return None

# --- 3. LÓGICA DE SENTIMIENTO (VOTACIÓN TÉCNICA) ---
def calcular_sentimiento_pro(last):
    buy_v, sell_v = 0, 0
    # Evaluación vs Medias
    for n in [5, 20, 50, 200]:
        if last['Close'] > last[f'SMA{n}']: buy_v += 1
        else: sell_v += 1
    
    # Evaluación Osciladores
    if last['RSI'] > 55: buy_v += 1
    if last['RSI'] < 45: sell_v += 1
    
    if sell_v > buy_v: return "VENTA FUERTE 🔴", "#FF4B4B"
    if buy_v > sell_v: return "COMPRA FUERTE 🟢", "#00FF88"
    return "NEUTRAL ⚖️", "#888888"

# --- 4. DASHBOARD DE MANDO ---
st.sidebar.title("🛡️ QUANTUM V6.2")
ticker_target = st.sidebar.text_input("ACTIVO", value="CHILE.SN").upper()
lista_radar = st.sidebar.text_area("RADAR", value="CHILE.SN, SQM-B.SN, COPEC.SN, AAPL, BTC-USD")
radar_items = [x.strip().upper() for x in lista_radar.split(",")]

df_main = obtener_datos_completos(ticker_target)

if df_main is not None:
    last_row = df_main.iloc[-1]
    status_msg, status_hex = calcular_sentimiento_pro(last_row)
    
    # BANNER DE ESTADO
    st.markdown(f"""
        <div style="border-left: 10px solid {status_hex}; background-color: #0a0a0a; padding: 25px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin:0; font-size: 3rem;">{ticker_target}: {status_msg}</h1>
            <p style="margin:0; opacity: 0.7; font-weight: bold;">Quantum Terminal | {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
    """, unsafe_allow_html=True)

    # MÉTRICAS TIPO BLOOMBERG
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ÚLTIMO PRECIO", f"${last_row['Close']:,.2f}")
    c2.metric("RSI (14D)", f"{last_row['RSI']:.1f}")
    c3.metric("ADX (FUERZA)", f"{last_row['ADX']:.1f}")
    c4.metric("VOL. RELATIVO", f"{last_row['Rel_Vol']:.1f}x")

    # GRÁFICO PROFESIONAL
    fig_pro = go.Figure()
    fig_pro.add_trace(go.Candlestick(x=df_main.index[-100:], open=df_main['Open'], high=df_main['High'], low=df_main['Low'], close=df_main['Close'], name="Candles"))
    fig_pro.add_trace(go.Scatter(x=df_main.index[-100:], y=df_main['SMA50'][-100:], line=dict(color='#00BFFF', width=1), name="SMA 50"))
    fig_pro.add_trace(go.Scatter(x=df_main.index[-100:], y=df_main['SMA200'][-100:], line=dict(color='#FF4B4B', width=2), name="SMA 200"))
    fig_pro.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='black', plot_bgcolor='black')
    st.plotly_chart(fig_pro, use_container_width=True)

    # --- TABLAS DE ANÁLISIS ---
    st.write("---")
    t_left, t_right = st.columns(2)
    
    with t_left:
        st.subheader("📈 Resumen de Medias")
        # Generación de tabla blindada contra NameErrors
        m_data = {
            "Periodo": ["5 (Corta)", "20 (Media)", "50 (Tendencia)", "200 (Institucional)"],
            "SMA": [f"{last_row[f'SMA{n}']:,.2f}" for n in [5, 20, 50, 200]],
            "EMA": [f"{last_row[f'EMA{n}']:,.2f}" for n in [5, 20, 50, 200]]
        }
        st.table(pd.DataFrame(m_data))

    with t_right:
        st.subheader("📍 Pivotes de Acción")
        piv_data = {
            "Nivel": ["R2 (Techo)", "R1 (Resistencia)", "Punto Pivote (PP)", "S1 (Soporte)", "S2 (Suelo)"],
            "Precio": [f"{last_row['R2']:,.2f}", f"{last_row['R1']:,.2f}", f"{last_row['PP']:,.2f}", f"{last_row['S1']:,.2f}", f"{last_row['S2']:,.2f}"]
        }
        st.table(pd.DataFrame(piv_data))

    # RADAR GLOBAL
    st.write("---")
    st.subheader("🚀 Radar de Seguimiento")
    radar_final = []
    for item in radar_items:
        d_radar = obtener_datos_completos(item)
        if d_radar is not None:
            l_radar = d_radar.iloc[-1]
            txt_res, _ = calcular_sentimiento_pro(l_radar)
            fuego_icon = "🔥" if l_radar['Rel_Vol'] > 1.5 else ""
            radar_final.append({
                "Activo": f"{item} {fuego_icon}", 
                "Precio": f"${l_radar['Close']:,.2f}", 
                "Señal": txt_res,
                "Stop sugerido": f"${d_radar['Low'].iloc[-3:].min():,.2f}"
            })
    
    if radar_final:
        st.table(pd.DataFrame(radar_final).style.set_properties(**{'color': 'white'}))

else:
    st.error("No se pudo conectar con el mercado. Verifica el Ticker o la conexión.")
