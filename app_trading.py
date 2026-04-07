import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- 1. DICCIONARIO DE CONOCIMIENTO (Interpretación de Patrones) ---
DICCIONARIO_PATRONES = {
    "doji": {
        "titulo": "Doji (Indecisión) ⚖️",
        "significado": "El precio de apertura y cierre fue casi igual. Indica que ni compradores ni vendedores tienen el control.",
        "accion": "Espera a la siguiente vela. Si la siguiente rompe arriba, es alcista; si rompe abajo, es bajista."
    },
    "hammer": {
        "titulo": "Martillo (Hammer) 🔨",
        "significado": "Vela con cuerpo pequeño y mecha larga inferior. Indica que los compradores rechazaron precios bajos.",
        "accion": "Señal de rebote alcista. Ideal si aparece cerca de la EMA 200 o la Banda Inferior."
    },
    "engulfing": {
        "titulo": "Envolvente (Engulfing) 🌊",
        "significado": "La vela actual 'se traga' por completo a la anterior.",
        "accion": "Si es verde, es una señal de compra fuerte. Si es roja, es una señal de venta inmediata."
    },
    "morningstar": {
        "titulo": "Estrella del Amanecer 🌅",
        "significado": "Patrón de 3 velas que marca el fin de una caída.",
        "accion": "Alta probabilidad de cambio de tendencia a alcista. Excelente señal de entrada."
    },
    "eveningstar": {
        "titulo": "Estrella del Atardecer 🌇",
        "significado": "Patrón de 3 velas que marca el fin de una subida.",
        "accion": "Precaución: El precio está agotado. Considera tomar ganancias o salir."
    }
}

# --- 2. CONFIGURACIÓN ---
st.set_page_config(page_title="Quantum Expert Guide", page_icon="🛡️", layout="wide")

# --- 3. LÓGICA TÉCNICA ---
def detectar_y_explicar(df):
    # Detectamos los 5 patrones más confiables
    patrones = df.ta.cdl_pattern(name=["doji", "engulfing", "hammer", "morningstar", "eveningstar"])
    return patrones

# --- 4. INTERFAZ ---
st.sidebar.title("🛡️ Quantum Shield")
ticker_input = st.sidebar.text_input("Activo", value="BTC").upper()
tf = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"], index=2)

ticker = f"{ticker_input}-USD" if ticker_input in ["BTC", "ETH", "SOL"] and "-" not in ticker_input else ticker_input

df = yf.download(ticker, period="350d", interval=tf, auto_adjust=True)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Indicadores Base
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    
    # Patrones
    df_patrones = detectar_y_explicar(df)
    last_p = df_patrones.iloc[-1]
    
    # Determinación de Señal
    close = float(df['Close'].iloc[-1])
    ema = float(df['EMA_200'].iloc[-1])
    color = "#00FFA3" if close > ema else "#FF4B4B"
    estado = "COMPRA" if close > ema else "VENTA / PRECAUCIÓN"

    # --- VISUALIZACIÓN ---
    st.markdown(f"""
        <div style="background-color: {color}; border-radius: 10px; padding: 15px; text-align: center; margin-bottom: 10px;">
            <h1 style="color: #000000 !important; margin: 0; font-size: 30px;">{estado}</h1>
        </div>
    """, unsafe_allow_html=True)

    # Gráfico
    fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Precio"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=2), name="EMA 200"))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- NUEVA SECCIÓN: EXPLICACIÓN DE PATRONES ---
    st.subheader("📊 Análisis de Velas Actual")
    
    encontrado = False
    for col in last_p.index:
        if last_p[col] != 0:
            nombre_clave = col.replace("CDL_", "").lower()
            if nombre_clave in DICCIONARIO_PATRONES:
                info = DICCIONARIO_PATRONES[nombre_clave]
                st.success(f"**{info['titulo']}**")
                st.write(f"**¿Qué significa?** {info['significado']}")
                st.info(f"**Acción Sugerida:** {info['accion']}")
                encontrado = True
    
    if not encontrado:
        st.write("No se detectan patrones de giro claros en la vela actual. Sigue la tendencia de la EMA 200.")

else:
    st.error("Error al cargar datos.")
