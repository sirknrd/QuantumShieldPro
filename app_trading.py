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


def safe_float(x) -> float:
    try:
        return float(x) if not pd.isna(x) and x is not None else 0.0
    except:
        return 0.0


def _is_valid_ticker(ticker: str) -> bool:
    return bool(re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker.strip().upper()))


@st.cache_data(ttl=300)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except:
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return df
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    vol = out["Volume"]

    # EMAs y SMAs
    for n in [9, 21, 50, 100, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)
        out[f"SMA{n}"] = ta.sma(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    out["ADX14"] = ta.adx(high, low, close).iloc[:, 0] if ta.adx(high, low, close) is not None else np.nan

    macd = ta.macd(close)
    if macd is not None:
        out["MACD_HIST"] = macd.iloc[:, 2]

    bb = ta.bbands(close)
    if bb is not None:
        out["BBP"] = bb.iloc[:, 3]

    out["ATR14"] = ta.atr(high, low, close, length=14)
    out["REL_VOL"] = vol / ta.sma(vol, length=20)

    # Supertrend
    try:
        st_df = ta.supertrend(high, low, close, length=10, multiplier=3.0)
        if st_df is not None:
            out = pd.concat([out, st_df], axis=1)
    except:
        pass

    return out.ffill()


def get_recommendation(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 40:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": 0, "confidence": 40, "reason": "Datos insuficientes"}

    last = df.iloc[-1]

    close = safe_float(last.get("Close"))
    ema50 = safe_float(last.get("EMA50"))
    ema200 = safe_float(last.get("EMA200"))
    rsi = safe_float(last.get("RSI14"))
    adx = safe_float(last.get("ADX14"))
    macd_hist = safe_float(last.get("MACD_HIST"))
    rel_vol = safe_float(last.get("REL_VOL"))

    score = 0.0
    reasons = []

    if close > ema50 > ema200:
        score += 45
        reasons.append("Tendencia Alcista Fuerte")
    elif close < ema50 < ema200:
        score -= 45
        reasons.append("Tendencia Bajista Fuerte")

    if 40 < rsi < 70:
        score += 20
    elif rsi < 35:
        score -= 25
    elif rsi > 75:
        score -= 25

    if adx > 25:
        score += 15 if close > ema50 else -15

    if macd_hist > 0:
        score += 12
    if rel_vol > 1.5:
        score += 8

    score = max(-100, min(100, score))

    if score >= 65:
        return {"label": "COMPRA FUERTE", "color": "#00D18F", "score": int(score), "confidence": 82, "reason": "Señales alcistas alineadas"}
    elif score >= 30:
        return {"label": "COMPRA", "color": "#2F81F7", "score": int(score), "confidence": 68, "reason": "Señales alcistas moderadas"}
    elif score <= -65:
        return {"label": "VENTA FUERTE", "color": "#FF4B4B", "score": int(score), "confidence": 82, "reason": "Señales bajistas alineadas"}
    elif score <= -30:
        return {"label": "VENTA", "color": "#FFA657", "score": int(score), "confidence": 68, "reason": "Señales bajistas moderadas"}
    else:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": int(score), "confidence": 55, "reason": "Sin tendencia clara"}


def main():
    with st.sidebar:
        st.header("Terminal")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        period = st.selectbox("Periodo", ["3mo", "6mo", "1y"], index=1)

    df = load_ohlcv(ticker, period)

    if df.empty:
        st.error(f"❌ No se pudieron descargar datos para **{ticker}**")
        return

    dfi = compute_indicators(df)
    rec = get_recommendation(dfi)

    last_price = safe_float(dfi["Close"].iloc[-1])
    change = (last_price / safe_float(dfi["Close"].iloc[-2]) - 1) * 100 if len(dfi) > 1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${last_price:,.2f}", f"{change:+.2f}%")
    c2.metric("Señal", rec["label"])
    c3.metric("RSI 14", f"{safe_float(dfi.get('RSI14', pd.Series([50])).iloc[-1]):.1f}")
    c4.metric("ADX 14", f"{safe_float(dfi.get('ADX14', pd.Series([20])).iloc[-1]):.1f}")

    st.subheader(f"Gráfico — {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfi.index, open=dfi["Open"], high=dfi["High"], low=dfi["Low"], close=dfi["Close"]))

    for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        if ema in dfi.columns:
            fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema], name=ema))

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recomendación del Sistema")
    st.markdown(f"<h2 style='color:{rec['color']};'>{rec['label']}</h2>", unsafe_allow_html=True)
    st.metric("Score", f"{rec['score']}/100", f"Confianza {rec['confidence']}%")
    st.info(rec["reason"])


if __name__ == "__main__":
    main()
