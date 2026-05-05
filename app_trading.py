import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="QuantumShield Pro", layout="wide")
st.title("QuantumShield Pro — Trading Terminal")

ticker = st.sidebar.text_input("Ticker", value="NVDA").upper()
period = st.sidebar.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)

if ticker:
    with st.spinner(f"Descargando {ticker}..."):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            
            if df.empty:
                df = yf.Ticker(ticker).history(period=period)
            
            if df.empty:
                st.error(f"No se pudieron obtener datos para {ticker}")
                st.stop()
                
            st.success(f"✅ Datos cargados: {len(df)} velas")
            
            # KPIs
            last_price = float(df['Close'].iloc[-1])
            change = (last_price / float(df['Close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
            
            col1, col2 = st.columns(2)
            col1.metric("Precio", f"${last_price:,.2f}", f"{change:+.2f}%")
            
            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, 
                                       open=df['Open'], 
                                       high=df['High'], 
                                       low=df['Low'], 
                                       close=df['Close']))
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Yahoo Finance a veces tiene problemas temporales. Intenta de nuevo en unos minutos.")

else:
    st.warning("Ingresa un ticker")
