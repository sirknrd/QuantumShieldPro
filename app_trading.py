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

st.set_page_config(page_title="QuantumShield Pro", layout="wide")

APP_TITLE = "QuantumShield Pro — Trading Terminal"
st.markdown(f"<h1 style='text-align:center;color:#00D18F;'>{APP_TITLE}</h1>", unsafe_allow_html=True)


def _is_valid_ticker(ticker: str) -> bool:
    return bool(re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker.strip().upper()))


@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, timeout=15)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        return df
    except:
        try:
            df = yf.Ticker(ticker).history(period=period)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy() if not df.empty else pd.DataFrame()
        except:
            return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return df
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    for n in [9, 21, 50, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    macd = ta.macd(close)
    if macd is not None and not macd.empty:
        out["MACD_HIST"] = macd.iloc[:, 2]

    adx = ta.adx(high, low, close)
    if adx is not None and not adx.empty:
        out["ADX14"] = adx.iloc[:, 0]

    return out.ffill()


def get_recommendation(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": 0, "confidence": 50}

    last = df.iloc[-1]

    score = 0
    # Tendencia
    if last["Close"] > last.get("EMA50", 0):
        score += 35
    if last.get("EMA50", 0) > last.get("EMA200", 0):
        score += 30
    if last["Close"] > last.get("EMA200", 0):
        score += 25

    # Momentum RSI
    rsi = last.get("RSI14")
    if rsi is not None:
        if 40 < rsi < 70:
            score += 20
        elif rsi < 35:
            score -= 25
        elif rsi > 75:
            score -= 20

    score = max(-100, min(100, score))

    if score >= 65:
        return {"label": "COMPRA FUERTE", "color": "#00D18F", "score": int(score), "confidence": 85}
    elif score >= 30:
        return {"label": "COMPRA", "color": "#2F81F7", "score": int(score), "confidence": 70}
    elif score <= -65:
        return {"label": "VENTA FUERTE", "color": "#FF4B4B", "score": int(score), "confidence": 85}
    elif score <= -30:
        return {"label": "VENTA", "color": "#FFA657", "score": int(score), "confidence": 70}
    else:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": int(score), "confidence": 55}


def main():
    with st.sidebar:
        st.header("🔍 Terminal")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)
        st.markdown("---")
        st.caption("Datos de Yahoo Finance")

    if not ticker:
        st.warning("Ingresa un ticker")
        return

    df = load_ohlcv(ticker, period)

    if df.empty:
        st.error(f"❌ No se pudieron descargar datos para **{ticker}**")
        st.info("Prueba con: AAPL, MSFT, NVDA, TSLA, GOOGL")
        return

    dfi = compute_indicators(df)
    rec = get_recommendation(dfi)

    # KPIs
    last_price = float(dfi["Close"].iloc[-1])
    change = (last_price / float(dfi["Close"].iloc[-2]) - 1) * 100 if len(dfi) > 1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${last_price:,.2f}", f"{change:+.2f}%")
    c2.metric("Señal", rec["label"], help=f"Score: {rec['score']}")
    c3.metric("RSI 14", f"{dfi['RSI14'].iloc[-1]:.1f}" if "RSI14" in dfi.columns else "—")
    c4.metric("ADX 14", f"{dfi['ADX14'].iloc[-1]:.1f}" if "ADX14" in dfi.columns else "—")

    # Gráfico
    st.subheader(f"📈 {ticker} - Evolución")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfi.index, open=dfi["Open"], high=dfi["High"], low=dfi["Low"], close=dfi["Close"]))

    for ema_col in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        if ema_col in dfi.columns:
            fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema_col], name=ema_col, line=dict(width=1.8)))

    fig.update_layout(template="plotly_dark", height=680, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    # Análisis
    st.subheader("📊 Análisis Técnico")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Recomendación:** <span style='color:{rec['color']};font-weight:bold;font-size:1.3em;'>{rec['label']}</span>", unsafe_allow_html=True)
        st.metric("Score Final", f"{rec['score']}/100", f"Confianza: {rec['confidence']}%")
    with col2:
        st.info("**Interpretación:**\n" + 
                ("Fuerte tendencia alcista con momentum sano." if rec['score'] >= 65 else
                 "Señal alcista moderada." if rec['score'] >= 30 else
                 "Fuerte señal bajista." if rec['score'] <= -65 else "Señal bajista moderada." if rec['score'] <= -30 else "Mercado en rango / neutral."))

    st.caption(f"QuantumShield Pro • {datetime.now().strftime('%d/%m/%Y %H:%M')} • Datos en tiempo real vía Yahoo Finance")


if __name__ == "__main__":
    main()
