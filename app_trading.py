from __future__ import annotations

import concurrent.futures
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="QuantumShield Pro", layout="wide")

st.markdown("""
<style>
    .header {font-size: 2.8rem; font-weight: bold; color: #00D18F; text-align: center;}
    .rec-strong {font-size: 2.4rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

APP_TITLE = "QuantumShield Pro — Trading Terminal"
st.markdown(f"<h1 class='header'>{APP_TITLE}</h1>", unsafe_allow_html=True)


@dataclass
class Recommendation:
    label: str
    color: str
    score: float
    confidence: int
    reason: str


def safe_float(x) -> float:
    try:
        return float(x) if not pd.isna(x) and x is not None else 0.0
    except:
        return 0.0


def _is_valid_ticker(ticker: str) -> bool:
    return bool(re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker.strip().upper()))


# ==================== CARGAS ====================
@st.cache_data(ttl=180)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
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
    vol = out["Volume"]

    for n in [9, 21, 50, 100, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    out["ADX14"] = ta.adx(high, low, close).iloc[:, 0] if ta.adx(high, low, close) is not None else np.nan
    macd = ta.macd(close)
    if macd is not None:
        out["MACD_HIST"] = macd.iloc[:, 2]
    bb = ta.bbands(close)
    if bb is not None:
        out["BBP"] = bb.iloc[:, 3]
    out["ATR14"] = ta.atr(high, low, close, length=14)

    vol_sma = ta.sma(vol, length=20)
    if vol_sma is not None:
        out["REL_VOL"] = vol / vol_sma.where(vol_sma > 0)

    try:
        st_df = ta.supertrend(high, low, close, length=10, multiplier=3)
        if st_df is not None:
            out = pd.concat([out, st_df], axis=1)
    except:
        pass

    return out.ffill()


def get_recommendation(df: pd.DataFrame) -> Recommendation:
    if df.empty or len(df) < 40:
        return Recommendation("NEUTRAL", "#8B949E", 0, 40, "Datos insuficientes")

    last = df.iloc[-1]
    close = safe_float(last.get("Close"))
    ema50 = safe_float(last.get("EMA50"))
    ema200 = safe_float(last.get("EMA200"))
    rsi = safe_float(last.get("RSI14"))
    adx = safe_float(last.get("ADX14"))
    macd_hist = safe_float(last.get("MACD_HIST"))

    score = 0.0
    reasons = []

    if close > ema50 > ema200:
        score += 48
        reasons.append("Tendencia Alcista Fuerte")
    elif close < ema50 < ema200:
        score -= 48
        reasons.append("Tendencia Bajista Fuerte")

    if 42 < rsi < 68:
        score += 22
    elif rsi < 35:
        score -= 30
    elif rsi > 73:
        score -= 30

    if adx > 25:
        score += 18 if close > ema50 else -18
    if macd_hist > 0:
        score += 15

    score = max(-100, min(100, score))

    if score >= 65:
        return Recommendation("COMPRA FUERTE", "#00D18F", int(score), 85, " • ".join(reasons))
    elif score >= 30:
        return Recommendation("COMPRA", "#2F81F7", int(score), 72, " • ".join(reasons))
    elif score <= -65:
        return Recommendation("VENTA FUERTE", "#FF4B4B", int(score), 85, " • ".join(reasons))
    elif score <= -30:
        return Recommendation("VENTA", "#FFA657", int(score), 72, " • ".join(reasons))
    else:
        return Recommendation("NEUTRAL", "#8B949E", int(score), 55, "Mercado en rango")


def main():
    with st.sidebar:
        st.header("⚙️ Terminal")
        ticker = st.text_input("Ticker Principal", value="NVDA").strip().upper()
        period = st.selectbox("Periodo", ["3mo", "6mo", "1y"], index=1)

    df = load_ohlcv(ticker, period)
    if df.empty:
        st.error(f"❌ No se pudieron descargar datos para **{ticker}**")
        st.stop()

    dfi = compute_indicators(df)
    rec = get_recommendation(dfi)

    last_price = safe_float(dfi["Close"].iloc[-1])
    change = ((last_price / safe_float(dfi["Close"].iloc[-2]) - 1) * 100) if len(dfi) > 1 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${last_price:,.2f}", f"{change:+.2f}%")
    c2.metric("Señal", rec.label)
    c3.metric("RSI 14", f"{safe_float(dfi.get('RSI14', pd.Series([50])).iloc[-1]):.1f}")
    c4.metric("ADX 14", f"{safe_float(dfi.get('ADX14', pd.Series([20])).iloc[-1]):.1f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Gráfico", "📊 Análisis", "⚠️ Riesgo", "📰 Noticias", "🔦 Radar"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=dfi.index, open=dfi["Open"], high=dfi["High"], low=dfi["Low"], close=dfi["Close"]))
        for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
            if ema in dfi.columns:
                fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema], name=ema))
        fig.update_layout(template="plotly_dark", height=720, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Recomendación")
        st.markdown(f"<h2 style='color:{rec.color};' class='rec-strong'>{rec.label}</h2>", unsafe_allow_html=True)
        st.metric("Score", f"{rec.score}/100", f"Confianza: {rec.confidence}%")
        st.info(rec.reason)

    with tab3:
        st.subheader("Gestión de Riesgo (ATR)")
        atr = safe_float(dfi["ATR14"].iloc[-1])
        if atr > 0:
            st.write(f"**Stop Loss:** ${last_price - 1.5*atr:,.2f}")
            st.write(f"**TP1:** ${last_price + 2*atr:,.2f}")
            st.write(f"**TP2:** ${last_price + 4*atr:,.2f}")

    with tab4:
        st.subheader("📰 Noticias y Sentimiento")
        st.info("Módulo de Noticias + Sentimiento (próxima actualización)")

    with tab5:
        st.subheader("🔦 Radar Avanzado")
        st.info("Radar multi-ticker (próxima actualización)")

    st.caption(f"QuantumShield Pro • {datetime.now().strftime('%d/%m/%Y %H:%M')}")


if __name__ == "__main__":
    main()
