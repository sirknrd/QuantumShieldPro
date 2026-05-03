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
st.markdown(f"<h1 style='text-align:center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)

# ====================== CONFIG ======================
def _is_valid_ticker(ticker: str) -> bool:
    return bool(re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker.strip().upper()))

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, timeout=20)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        return df
    except:
        try:
            df = yf.Ticker(ticker).history(period=period)
            if not df.empty:
                return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        except:
            pass
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return df
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    for n in [9, 21, 50, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    macd = ta.macd(close)
    if macd is not None:
        out["MACD_HIST"] = macd.iloc[:, 2]
    adx = ta.adx(high, low, close)
    if adx is not None:
        out["ADX14"] = adx.iloc[:, 0]

    return out.ffill()


def get_recommendation(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": 0, "confidence": 50}

    last = df.iloc[-1]
    score = 0

    # Tendencia
    if last["Close"] > last.get("EMA50", 0):
        score += 40
    if last.get("EMA50", 0) > last.get("EMA200", 0):
        score += 30
    if last["Close"] > last.get("EMA200", 0):
        score += 30

    # Momentum
    if "RSI14" in last and 40 < last["RSI14"] < 70:
        score += 20
    elif "RSI14" in last and last["RSI14"] < 30:
        score -= 30

    score = max(-100, min(100, score))

    if score >= 65:
        return {"label": "COMPRA FUERTE", "color": "#00D18F", "score": score, "confidence": 85}
    elif score >= 30:
        return {"label": "COMPRA", "color": "#2F81F7", "score": score, "confidence": 70}
    elif score <= -65:
        return {"label": "VENTA FUERTE", "color": "#FF4B4B", "score": score, "confidence": 85}
    elif score <= -30:
        return {"label": "VENTA", "color": "#FFA657", "score": score, "confidence": 70}
    else:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": score, "confidence": 50}


def main():
    with st.sidebar:
        st.header("Terminal")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        period = st.selectbox("Periodo histórico", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh cada 5 min", value=False)

    if not ticker:
        st.warning("Ingresa un ticker válido")
        return

    df = load_ohlcv(ticker, period)
    if df.empty:
        st.error(f"❌ No se pudieron obtener datos para **{ticker}**")
        st.info("Prueba con: AAPL, MSFT, NVDA, TSLA, AMZN")
        return

    dfi = compute_indicators(df)
    rec = get_recommendation(dfi)

    # Header KPIs
    last_price = dfi["Close"].iloc[-1]
    change = _safe_pct(last_price, dfi["Close"].iloc[-2]) if len(dfi) > 1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Último Precio", f"${last_price:,.2f}", f"{change:+.2f}%")
    c2.metric("Señal", rec["label"], delta=f"Score: {rec['score']:+.0f}")
    c3.metric("RSI (14)", f"{dfi['RSI14'].iloc[-1]:.1f}" if "RSI14" in dfi.columns else "—")
    c4.metric("ADX (14)", f"{dfi['ADX14'].iloc[-1]:.1f}" if "ADX14" in dfi.columns else "—")

    # Gráfico principal
    st.subheader(f"Gráfico — {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dfi.index, open=dfi["Open"], high=dfi["High"],
        low=dfi["Low"], close=dfi["Close"], name="OHLC"
    ))
    # Añadir EMAs
    for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        if ema in dfi.columns:
            fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema], name=ema, line=dict(width=1.5)))

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # Análisis
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Resumen Técnico")
        st.success(f"**{rec['label']}** — Confianza: {rec['confidence']}%")
        st.metric("Score Final", f"{rec['score']:+.0f}/100")

    with colB:
        st.subheader("Indicadores Actuales")
        metrics = {
            "RSI 14": f"{dfi['RSI14'].iloc[-1]:.1f}" if "RSI14" in dfi else "—",
            "ADX 14": f"{dfi['ADX14'].iloc[-1]:.1f}" if "ADX14" in dfi else "—",
            "Precio vs EMA50": "Alcista" if dfi["Close"].iloc[-1] > dfi["EMA50"].iloc[-1] else "Bajista",
        }
        st.json(metrics)

    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

    st.caption(f"QuantumShield Pro • Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')} • Datos de Yahoo Finance")


def _safe_pct(a: float, b: float) -> float:
    if b == 0 or math.isnan(b):
        return 0.0
    return (a / b - 1.0) * 100.0


if __name__ == "__main__":
    main()
