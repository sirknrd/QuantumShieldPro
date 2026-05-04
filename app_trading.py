from __future__ import annotations

import math
import re
from datetime import datetime

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
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, timeout=20)
        if df.empty:
            return pd.DataFrame()
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
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
    if df.empty or len(df) < 40:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": 0, "confidence": 40, "reason": "Datos insuficientes"}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    score = 0.0
    reasons = []

    close = float(last["Close"])
    ema50 = float(last.get("EMA50", 0))
    ema200 = float(last.get("EMA200", 0))
    rsi = float(last.get("RSI14", 50))
    adx = float(last.get("ADX14", 0))
    macd_hist = float(last.get("MACD_HIST", 0))

    # 1. Tendencia (peso alto)
    if close > ema50 and ema50 > ema200:
        score += 40
        reasons.append("Tendencia alcista fuerte (EMA)")
    elif close < ema50 and ema50 < ema200:
        score -= 40
        reasons.append("Tendencia bajista fuerte (EMA)")

    # 2. Momentum RSI
    if 45 < rsi < 70:
        score += 20
        reasons.append("Momentum sano")
    elif rsi < 35:
        score -= 25
        reasons.append("Sobreventa extrema")
    elif rsi > 75:
        score -= 25
        reasons.append("Sobrecompra extrema")

    # 3. ADX + Tendencia
    if adx > 25:
        if close > ema50:
            score += 15
        else:
            score -= 15
        reasons.append(f"Tendencia fuerte (ADX {adx:.1f})")

    # 4. MACD
    if macd_hist > 0:
        score += 15
        reasons.append("MACD alcista")
    elif macd_hist < -0.5:
        score -= 15
        reasons.append("MACD bajista")

    score = max(-100, min(100, score))

    if score >= 70:
        return {"label": "COMPRA FUERTE", "color": "#00D18F", "score": int(score), "confidence": 85, "reason": " • ".join(reasons[:3])}
    elif score >= 35:
        return {"label": "COMPRA", "color": "#2F81F7", "score": int(score), "confidence": 70, "reason": " • ".join(reasons[:3])}
    elif score <= -70:
        return {"label": "VENTA FUERTE", "color": "#FF4B4B", "score": int(score), "confidence": 85, "reason": " • ".join(reasons[:3])}
    elif score <= -35:
        return {"label": "VENTA", "color": "#FFA657", "score": int(score), "confidence": 70, "reason": " • ".join(reasons[:3])}
    else:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": int(score), "confidence": 50, "reason": "Mercado en rango"}


def main():
    with st.sidebar:
        st.header("🔍 Terminal")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        period = st.selectbox("Periodo", ["3mo", "6mo", "1y", "2y"], index=1)

    if not ticker:
        return

    df = load_ohlcv(ticker, period)
    if df.empty:
        st.error(f"❌ No se pudieron descargar datos para **{ticker}**")
        return

    dfi = compute_indicators(df)
    rec = get_recommendation(dfi)

    last_price = float(dfi["Close"].iloc[-1])
    change = (last_price / float(dfi["Close"].iloc[-2]) - 1) * 100 if len(dfi) > 1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${last_price:,.2f}", f"{change:+.2f}%")
    c2.metric("Señal", rec["label"])
    c3.metric("RSI", f"{float(dfi['RSI14'].iloc[-1]):.1f}")
    c4.metric("ADX", f"{float(dfi['ADX14'].iloc[-1]):.1f}")

    st.subheader(f"📈 {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfi.index, open=dfi["Open"], high=dfi["High"], low=dfi["Low"], close=dfi["Close"]))
    for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        if ema in dfi.columns:
            fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema], name=ema))
    fig.update_layout(template="plotly_dark", height=680, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Recomendación")
    st.markdown(f"<h2 style='color:{rec['color']};'>{rec['label']}</h2>", unsafe_allow_html=True)
    st.metric("Score", f"{rec['score']}/100", f"Confianza: {rec['confidence']}%")
    st.info(rec["reason"])

    st.caption(f"QuantumShield Pro • {datetime.now().strftime('%d/%m/%Y %H:%M')}")


if __name__ == "__main__":
    main()
