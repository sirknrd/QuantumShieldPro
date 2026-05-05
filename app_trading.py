from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="QuantumShield Pro", layout="wide")

st.markdown("<h1 style='text-align:center;color:#00D18F;'>QuantumShield Pro — Trading Terminal</h1>", unsafe_allow_html=True)


def safe_float(x) -> float:
    try:
        return float(x) if not pd.isna(x) and x is not None else 0.0
    except:
        return 0.0


@st.cache_data(ttl=60)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Versión ultra robusta de carga"""
    if not re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker.strip().upper()):
        return pd.DataFrame()
    
    st.info(f"Descargando datos de {ticker}...")  # Para debug
    
    try:
        # Método 1 - yfinance download
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, timeout=10)
        if not df.empty and len(df) > 10:
            st.success(f"✅ Datos cargados correctamente ({len(df)} velas)")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Método 2 - Ticker history
        df = yf.Ticker(ticker).history(period=period, timeout=10)
        if not df.empty and len(df) > 10:
            st.success(f"✅ Datos cargados correctamente ({len(df)} velas)")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        st.error("No se obtuvieron datos")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al descargar {ticker}: {str(e)[:80]}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    for n in [9, 21, 50, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    adx = ta.adx(high, low, close)
    if adx is not None:
        out["ADX14"] = adx.iloc[:, 0]

    return out.ffill()


def get_recommendation(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 20:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": 0, "confidence": 30, "reason": "Datos insuficientes"}

    last = df.iloc[-1]
    close = safe_float(last.get("Close"))
    ema50 = safe_float(last.get("EMA50"))
    ema200 = safe_float(last.get("EMA200"))
    rsi = safe_float(last.get("RSI14"))

    score = 0.0

    if close > ema50 > ema200:
        score += 45
    elif close < ema50 < ema200:
        score -= 45

    if 40 < rsi < 70:
        score += 20
    elif rsi < 35:
        score -= 25
    elif rsi > 75:
        score -= 25

    score = max(-100, min(100, score))

    if score >= 60:
        return {"label": "COMPRA FUERTE", "color": "#00D18F", "score": int(score), "confidence": 80, "reason": "Tendencia alcista fuerte"}
    elif score >= 25:
        return {"label": "COMPRA", "color": "#2F81F7", "score": int(score), "confidence": 65, "reason": "Señal alcista"}
    elif score <= -60:
        return {"label": "VENTA FUERTE", "color": "#FF4B4B", "score": int(score), "confidence": 80, "reason": "Tendencia bajista fuerte"}
    elif score <= -25:
        return {"label": "VENTA", "color": "#FFA657", "score": int(score), "confidence": 65, "reason": "Señal bajista"}
    else:
        return {"label": "NEUTRAL", "color": "#8B949E", "score": int(score), "confidence": 50, "reason": "Sin tendencia clara"}


def main():
    with st.sidebar:
        st.header("⚙️ Terminal")
        ticker = st.text_input("Ticker", value="NVDA").strip().upper()
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)

    df = load_ohlcv(ticker, period)

    if df.empty:
        st.error(f"❌ No se pudieron obtener datos para **{ticker}**")
        st.info("Prueba con NVDA, AAPL, TSLA o MSFT")
        st.stop()

    dfi = compute_indicators(df)
    rec = get_recommendation(dfi)

    last_price = safe_float(dfi["Close"].iloc[-1])
    prev_price = safe_float(dfi["Close"].iloc[-2]) if len(dfi) > 1 else last_price
    change = ((last_price / prev_price - 1) * 100) if prev_price > 0.01 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio Actual", f"${last_price:,.2f}", f"{change:+.2f}%")
    c2.metric("Señal", rec["label"])
    c3.metric("RSI 14", f"{safe_float(dfi.get('RSI14', pd.Series([50])).iloc[-1]):.1f}")

    st.subheader(f"📈 Gráfico — {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfi.index, open=dfi["Open"], high=dfi["High"], low=dfi["Low"], close=dfi["Close"]))
    for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        if ema in dfi.columns:
            fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema], name=ema))
    fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Recomendación")
    st.markdown(f"<h2 style='color:{rec['color']};'>{rec['label']}</h2>", unsafe_allow_html=True)
    st.metric("Score", f"{rec['score']}/100", f"Confianza: {rec['confidence']}%")
    st.info(rec["reason"])


if __name__ == "__main__":
    main()
