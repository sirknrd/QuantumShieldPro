from __future__ import annotations

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


def safe_float(x) -> float:
    try:
        if x is None or pd.isna(x):
            return 0.0
        return float(x)
    except:
        return 0.0


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
    if df.empty or len(df) < 30:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": 0, "confidence": 40, "reason": "Datos insuficientes"}

    last = df.iloc[-1]

    close = safe_float(last.get("Close"))
    ema50 = safe_float(last.get("EMA50"))
    ema200 = safe_float(last.get("EMA200"))
    rsi = safe_float(last.get("RSI14"))
    adx = safe_float(last.get("ADX14"))
    macd_h = safe_float(last.get("MACD_HIST"))

    score = 0.0
    reasons = []

    if close > ema50 and ema50 > 0:
        score += 35
        reasons.append("Precio sobre EMA50")
    if ema50 > ema200 and ema200 > 0:
        score += 30
        reasons.append("EMA50 sobre EMA200")
    if close > ema200 and ema200 > 0:
        score += 20
        reasons.append("Precio sobre EMA200")

    if 40 < rsi < 70:
        score += 20
        reasons.append("RSI saludable")
    elif rsi < 35:
        score -= 25
        reasons.append("Sobreventa")
    elif rsi > 75:
        score -= 25
        reasons.append("Sobrecompra")

    if adx > 25:
        score += 12 if close > ema50 else -12
        reasons.append(f"Tendencia fuerte (ADX {adx:.1f})")

    if macd_h > 0:
        score += 15
        reasons.append("MACD alcista")

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
        return {"label": "NEUTRAL", "color": "#8B949E", "score": int(score), "confidence": 50, "reason": "Mercado sin tendencia clara"}


def main():
    with st.sidebar:
        st.header("🔍 Terminal")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        period = st.selectbox("Periodo", ["3mo", "6mo", "1y"], index=1)

    df = load_ohlcv(ticker, period)

    if df.empty:
        st.error(f"❌ No se pudieron descargar datos para **{ticker}**")
        st.info("Prueba con: AAPL, MSFT, NVDA, TSLA")
        return

    dfi = compute_indicators(df)
    rec = get_recommendation(dfi)

    last_price = safe_float(dfi["Close"].iloc[-1])
    prev_price = safe_float(dfi["Close"].iloc[-2]) if len(dfi) > 1 else last_price
    change = (last_price / prev_price - 1) * 100 if prev_price != 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Actual", f"${last_price:,.2f}", f"{change:+.2f}%")
    c2.metric("Señal", rec["label"])
    c3.metric("RSI 14", f"{safe_float(dfi['RSI14'].iloc[-1] if 'RSI14' in dfi.columns else 50):.1f}")
    c4.metric("ADX 14", f"{safe_float(dfi['ADX14'].iloc[-1] if 'ADX14' in dfi.columns else 20):.1f}")

    st.subheader(f"📈 {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfi.index, open=dfi["Open"], high=dfi["High"], low=dfi["Low"], close=dfi["Close"]))
    for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        if ema in dfi.columns:
            fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema], name=ema))
    fig.update_layout(template="plotly_dark", height=680, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Recomendación Final")
    st.markdown(f"<h2 style='color:{rec['color']}; text-align:center;'>{rec['label']}</h2>", unsafe_allow_html=True)
    st.metric("Score", f"{rec['score']}/100", f"Confianza: {rec['confidence']}%")
    st.info(rec["reason"])


if __name__ == "__main__":
    main()
