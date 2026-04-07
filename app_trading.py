import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. SETUP DE INTERFAZ ULTRA-NEGRA ---
st.set_page_config(page_title="Quantum Shield Ultra", layout="wide", initial_sidebar_state="collapsed")

# CSS para eliminar espacios en blanco y forzar el modo oscuro real
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .main { background-color: #000000; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Estilo para métricas */
    [data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #222;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Forzar texto blanco en todo */
    h1, h2, h3, p, span, div, td, th { color: #ffffff !important; font-family: 'Inter', sans-serif; }
    
    /* Tablas estilo Bloomberg */
    .stTable { 
        background-color: #000000; 
        border: 1px solid #333;
        border-radius: 5px;
    }
    thead tr th { background-color: #111 !important; border-bottom: 2px solid #333 !important; }
    tbody tr td { border-bottom: 1px solid #222 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR TÉCNICO DE ALTA PRECISIÓN ---
def get_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Medias e Indicadores
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['SMA200'] = ta.sma(df['Close'], length=200)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Pivotes Clásicos
        h, l, c = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        df['PP'] = (h + l + c) / 3
        df['R1'], df['S1'] = (2 * df['PP']) - l, (2 * df['PP']) - h
        df['R2'], df['S2'] = df['PP'] + (h - l), df['PP'] - (h - l)
        
        return df.ffill()
    except: return None

# --- 3. DASHBOARD PRINCIPAL ---
st.sidebar.title("🛡️ Quantum Config")
ticker = st.sidebar.text_input("Activo", value="CHILE.SN").upper()
watch_input = st.sidebar.text_area("Radar", "CHILE.SN, SQM-B.SN, COPEC.SN, AAPL, BTC-USD")
watch_list = [x.strip() for x in watch_input.split(",")]

df = get_data(ticker)

if df is not None:
    last = df.iloc[-1]
    
    # Lógica de Color de Señal
    score = 0
    if last['Close'] > last['SMA200']: score += 50
    if 40 < last['RSI'] < 65: score += 50
    
    sig_text = "COMPRA FUERTE" if score == 100 else "MANTENER / VENTA" if score == 50 else "VENTA FUERTE"
    sig_col = "#00FF88" if score == 100 else "#FFC107" if score == 50 else "#FF4B4B"

    # BANNER SUPERIOR DE IMPACTO
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, {sig_col} 0%, #000 100%); padding: 20px; border-radius: 10px; border-left: 10px solid {sig_col}; margin-bottom: 20px;">
            <h1 style="margin:0; font-size: 2.5rem;">{ticker} : {sig_text}</h1>
            <p style="margin:0; opacity: 0.8; font-weight: bold;">Score Algorítmico: {score}% | RSI: {last['RSI']:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

    # MÉTRICAS TIPO TERMINAL
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ÚLTIMO PRECIO", f"${last['Close']:,.2f}")
    m2.metric("PUNTO PIVOTE", f"${last['PP']:.2f}")
    m3.metric("RESISTENCIA (R1)", f"${last['R1']:.2f}")
    m4.metric("SOPORTE (S1)", f"${last['S1']:.2f}")

    # --- 4. GRÁFICO PROFESIONAL (PLOTLY BLACK) ---
    fig = go.Figure()
    # Velas Japonesas
    fig.add_trace(go.Candlestick(
        x=df.index[-120:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#00FF88', decreasing_line_color='#FF4B4B', name="Precio"
    ))
    # Medias Móviles
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df['SMA50'][-120:], line=dict(color='#00BFFF', width=2), name="Media 50 (Corta)"))
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df['SMA200'][-120:], line=dict(color='#FF0000', width=2), name="Media 200 (Institucional)"))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- 5. SECCIÓN DE TABLAS TÉCNICAS ---
    st.markdown("### 📊 Profundidad Técnica")
    t1, t2 = st.columns(2)
    
    with t1:
        st.markdown("**Niveles de Pivote Clásicos**")
        piv_df = pd.DataFrame({
            "Nivel": ["R2 (Techo)", "R1 (Resistencia)", "Punto Pivote", "S1 (Soporte)", "S2 (Suelo)"],
            "Valor": [f"{last['R2']:.2f}", f"{last['R1']:.2f}", f"{last['PP']:.2f}", f"{last['S1']:.2f}", f"{last['S2']:.2f}"]
        })
        st.table(piv_df)

    with t2:
        st.markdown("**Radar de Oportunidades (Mercado Abierto)**")
        radar_results = []
        for t_name in watch_list:
            d_r = get_data(t_name)
            if d_r is not None:
                l_r = d_r.iloc[-1]
                st_r = "COMPRA" if l_r['Close'] > l_r['SMA200'] else "VENTA"
                radar_results.append({"Activo": t_name, "Precio": f"${l_r['Close']:.2f}", "Señal": st_r})
        
        radar_df = pd.DataFrame(radar_results)
        st.table(radar_df)

else:
    st.error("No se pudieron cargar los datos. Revisa la conexión o el Ticker.")
