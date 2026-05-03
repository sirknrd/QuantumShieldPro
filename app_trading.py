from __future__ import annotations

import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

APP_TITLE = "QuantumShield Pro — Trading Terminal"

def _is_valid_ticker(ticker: str) -> bool:
    return bool(re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker.strip().upper()))

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    
    try:
        # Versión más robusta de descarga
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, auto_adjust=True, timeout=15)
        
        if df.empty:
            # Intento alternativo
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True, timeout=15)
        
        if df.empty:
            return pd.DataFrame()

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error descargando {ticker}: {str(e)[:100]}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    # Indicadores principales
    for n in [9, 21, 50, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    out["MACD_HIST"] = ta.macd(close).iloc[:, 2] if ta.macd(close) is not None else np.nan
    adx_df = ta.adx(high, low, close)
    out["ADX14"] = adx_df.iloc[:, 0] if adx_df is not None else np.nan

    return out.ffill()


def recommend(df: pd.DataFrame):
    if df.empty or len(df) < 20:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": 0}

    last = df.iloc[-1]
    trend = 0
    if not pd.isna(last.get("EMA50")) and not pd.isna(last.get("EMA200")):
        trend += 1 if last["Close"] > last["EMA50"] else -1
        trend += 1 if last["EMA50"] > last["EMA200"] else -1

    score = trend * 40
    score = max(-100, min(100, score))

    if score >= 60:
        return {"label": "COMPRA FUERTE", "color": "#00D18F", "score": score}
    elif score >= 25:
        return {"label": "COMPRA", "color": "#2F81F7", "score": score}
    elif score <= -60:
        return {"label": "VENTA FUERTE", "color": "#FF4B4B", "score": score}
    elif score <= -25:
        return {"label": "VENTA", "color": "#FFA657", "score": score}
    else:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": score}


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)

    if not _is_valid_ticker(ticker):
        st.error("Ticker inválido")
        st.stop()

    df = load_ohlcv(ticker, period)
    
    if df.empty:
        st.error(f"❌ No se pudieron descargar datos para **{ticker}**. Intenta con otro ticker (ej: MSFT, NVDA, TSLA).")
        st.info("Si el problema persiste, es posible que Yahoo Finance tenga restricciones temporales.")
        st.stop()

    dfi = compute_indicators(df)
    rec = recommend(dfi)

    last_price = dfi["Close"].iloc[-1]
    change = (last_price / dfi["Close"].iloc[-2] - 1) * 100 if len(dfi) > 1 else 0

    col1, col2, col3 = st.columns([1,1,1])
    col1.metric("Precio Actual", f"${last_price:,.2f}", f"{change:+.2f}%")
    col2.metric("Recomendación", rec["label"])
    col3.metric("Score", f"{rec['score']:+.0f}")

    # Gráfico
    st.subheader("Gráfico de Precios")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dfi.index,
        open=dfi["Open"],
        high=dfi["High"],
        low=dfi["Low"],
        close=dfi["Close"],
        name="Precio"
    ))
    fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ Análisis completado para **{ticker}**")


if __name__ == "__main__":
    main()
