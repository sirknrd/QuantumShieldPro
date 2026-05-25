import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="QuantumShield Pro", layout="wide")
st.title("QuantumShield Pro — Trading Terminal")

ticker = st.sidebar.text_input("Ticker", value="NVDA").upper().strip()
period = st.sidebar.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)

PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}

if ticker:
    with st.spinner(f"Descargando datos de {ticker}..."):
        try:
            today = datetime.today()
            start = today - timedelta(days=PERIOD_DAYS[period])

            # Usar start/end explícitos en lugar de period para evitar caché
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),  # +1 para incluir hoy
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                df = yf.Ticker(ticker).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=(today + timedelta(days=1)).strftime("%Y-%m-%d")
                )

            if df.empty:
                st.error(f"No se pudieron obtener datos para {ticker}")
                st.stop()

            # LIMPIEZA DE COLUMNAS
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            st.success(f"✅ Datos cargados correctamente ({len(df)} velas)")

            last_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else last_price
            change = ((last_price / prev_price - 1) * 100) if prev_price > 0 else 0.0

            col1, col2 = st.columns(2)
            col1.metric("Precio Actual", f"${last_price:,.2f}", f"{change:+.2f}%")
            col2.metric("Última vela", df.index[-1].strftime("%d/%m/%Y"))

            st.subheader(f"📈 Gráfico — {ticker}")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ))
            fig.update_layout(
                template="plotly_dark",
                height=700,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Intenta con otro ticker o refresca la página.")
else:
    st.warning("Ingresa un ticker válido")
