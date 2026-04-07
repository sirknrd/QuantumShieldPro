import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURACIÓN DE UI DE ALTA DENSIDAD (BLACK MODE) ---
st.set_page_config(page_title="Quantum Shield Ultra v6", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #000000; }
    .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #222; border-radius: 8px; padding: 10px; }
    [data-testid="stTable"] { background-color: #000000; color: #ffffff !important; }
    table { border: 1px solid #333 !important; color: #ffffff !important; }
    thead tr th { background-color: #111 !important; color: #8b949e !important; }
    td { color: #ffffff !important; border-bottom: 1px solid #222 !important; }
    h1, h2, h3, p, span { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #111; border-radius: 4px 4px 0 0; padding: 10px 20px; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO DE GRADO INSTITUCIONAL ---
def obtener_analisis_profundo(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 2.1 Medias Móviles (SMA y EMA)
        for n in [5, 10, 20, 50, 100, 200]:
            df[f'SMA{n}'] = ta.sma(df['Close'], length=n)
            df[f'EMA{n}'] = ta.ema(df['Close'], length=n)

        # 2.2 Osciladores Técnicos
        df['RSI'] = ta.rsi(df['Close'], length=14)
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx_df.iloc[:, 0]
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        # 2.3 Puntos Pivote (Clásicos)
        h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        df['PP'] = (h + l + c) / 3
        df['R1'], df['S1'] = (2 * df['PP']) - l, (2 * df['PP']) - h
        df['R2'], df['S2'] = df['PP'] + (h - l), df['PP'] - (h - l)
        
        # 2.4 Volumen Relativo
        df['Vol_Avg'] = ta.sma(df['Volume'], length=20)
        df['Rel_Vol'] = df['Volume'] / df['Vol_Avg']
        
        return df.ffill()
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return None

# --- 3. LÓGICA DE DECISIÓN "ESTILO INVESTING" ---
def calcular_sentimiento(last):
    buy_votes, sell_votes = 0, 0
    
    # Votos por Medias Móviles
    for n in [5, 10, 20, 50, 100, 200]:
        if last['Close'] > last[f'SMA{n}']: buy_votes += 1
        else: sell_votes += 1
        if last['Close'] > last[f'EMA{n}']: buy_votes += 1
        else: sell_votes += 1
        
    # Votos por Osciladores
    if last['RSI'] < 30: buy_votes += 2  # Sobreventa
    elif last['RSI'] > 70: sell_votes += 2 # Sobrecompra
    
    if last['ADX'] > 25 and last['Close'] > last['SMA50']: buy_votes += 1
    elif last['ADX'] > 25 and last['Close'] < last['SMA50']: sell_votes += 1

    # Resultado Final
    total = buy_votes + sell_votes
    if sell_votes > (total * 0.7): return "VENTA FUERTE 🔴", "#FF4B4B"
    if sell_votes > buy_votes: return "VENTA 🟠", "#FF8C00"
    if buy_votes > (total * 0.7): return "COMPRA FUERTE 🟢", "#00FF88"
    if buy_votes > sell_votes: return "COMPRA 🔵", "#00BFFF"
    return "NEUTRAL ⚖️", "#888888"

# --- 4. INTERFAZ Y SIDEBAR ---
st.sidebar.title("🛡️ QUANTUM ULTRA v6")
ticker_principal = st.sidebar.text_input("ACTIVO PRINCIPAL", value="CHILE.SN").upper()
radar_input = st.sidebar.text_area("RADAR DE SEGUIMIENTO (Comas)", value="CHILE.SN, SQM-B.SN, COPEC.SN, AAPL, BTC-USD, ETH-USD")
watch_list = [x.strip().upper() for x in radar_input.split(",")]

# --- 5. DASHBOARD EJECUTIVO ---
df = obtener_analisis_profundo(ticker_principal)

if df is not None:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    status_text, status_color = calcular_sentimiento(last)
    
    # Banner de Decisión Inmediata
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, #000 0%, {status_color}55 100%); 
                    border-left: 10px solid {status_color}; padding: 25px; border-radius: 10px; margin-bottom: 25px;">
            <h3 style="margin:0; opacity: 0.6; font-size: 1rem;">ANÁLISIS TÉCNICO EN TIEMPO REAL</h3>
            <h1 style="margin:0; font-size: 3.5rem; letter-spacing: -2px;">{ticker_principal}: {status_text}</h1>
            <p style="margin:0; font-weight: bold; color: {status_color} !important;">Resumen basado en 14 indicadores técnicos y medias móviles.</p>
        </div>
    """, unsafe_allow_html=True)

    # Métricas de la Terminal
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRECIO", f"${last['Close']:,.2f}", f"{(last['Close']/prev['Close']-1)*100:.2f}%")
    c2.metric("RSI (14D)", f"{last['RSI']:.2f}")
    c3.metric("FUERZA (ADX)", f"{last['ADX']:.2f}")
    c4.metric("VOL. RELATIVO", f"{last['Rel_Vol']:.2f}x")

    # --- 6. GRÁFICO PROFESIONAL ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index[-120:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df['SMA50'][-120:], line=dict(color='#00BFFF', width=1.5), name="SMA 50"))
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df['SMA200'][-120:], line=dict(color='#FF4B4B', width=2), name="SMA 200"))
    
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, 
                      margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='black', plot_bgcolor='black')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- 7. TABS DE PROFUNDIDAD ---
    tab1, tab2, tab3 = st.tabs(["📊 Puntos Pivote", "📈 Medias Móviles", "🔥 Radar de Oportunidades"])
    
    with tab1:
        st.markdown("#### Niveles Clave para Operar Hoy")
        piv_data = {
            "Nivel": ["R2 (Techo)", "R1 (Resistencia)", "Punto Pivote (PP)", "S1 (Soporte)", "S2 (Suelo)"],
            "Precio CLP/USD": [f"{last['R2']:,.2f}", f"{last['R1']:,.2f}", f"{last['PP']:,.2f}", f"{last['S1']:,.2f}", f"{last['S2']:,.2f}"]
        }
        st.table(pd.DataFrame(piv_data))

    with tab2:
        st.markdown("#### Comparativa de Medias (SMA vs EMA)")
        medias_df = pd.DataFrame({
            "Periodo": ["5", "20", "50", "100", "200"],
            "Simple (SMA)": [f"{last['SMA5']:.2f}", f"{last['SMA20']:.2f}", f"{last['SMA50']:.2f}", f"{last['SMA200']:.2f}"],
            "Exponencial (EMA)": [f"{last['EMA5']:.2f}", f"{last['EMA20']:.2f}", f"{last['EMA50']:.2f}", f"{last['EMA200']:.2f}"]
        })
        st.table(medias_df)

    with tab3:
        st.markdown("#### Escáner de Mercado con Letras Blancas")
        radar_results = []
        for t in watch_list:
            d_r = obtener_analisis_profundo(t)
            if d_r is not None:
                l_r = d_r.iloc[-1]
                res_r, col_r = calcular_sentimiento(l_r)
                fuego = "🔥" if l_r['Rel_Vol'] > 1.5 else ""
                radar_results.append({
                    "Activo": f"{t} {fuego}",
                    "Precio": f"{l_r['Close']:,.2f}",
                    "Sugerencia": res_r,
                    "Stop Loss": f"{d_r['Low'].iloc[-3:].min():,.2f}"
                })
        
        radar_df = pd.DataFrame(radar_results)
        def color_radar(val):
            if "COMPRA" in str(val): return 'color: #00FF88; font-weight: bold'
            if "VENTA" in str(val): return 'color: #FF4B4B; font-weight: bold'
            return 'color: #FFC107; font-weight: bold'
            
        st.table(radar_df.style.set_properties(**{'color': 'white'}).map(color_radar, subset=['Sugerencia']))

else:
    st.error("No hay conexión con los mercados. Revisa el Ticker.")
