from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

APP_TITLE = "QuantumShield Pro — Trading Terminal"

_TICKER_RE = re.compile(r"^[A-Z0-9\-\.]{1,10}$")


@dataclass(frozen=True)
class Recommendation:
    label: str
    color: str
    score: float
    confidence: int


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_pct(a: float, b: float) -> float:
    if b == 0 or math.isnan(b) or math.isinf(b):
        return float("nan")
    return (a / b - 1.0) * 100.0


def _is_valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_RE.match(ticker.strip().upper()))


@st.cache_data(ttl=300)
def load_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        return df
    except:
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    for n in [9, 21, 50, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    out["ATR14"] = ta.atr(high, low, close, length=14)

    macd = ta.macd(close)
    if macd is not None:
        out["MACD_HIST"] = macd.iloc[:, 2]

    adx = ta.adx(high, low, close)
    if adx is not None:
        out["ADX14"] = adx.iloc[:, 0]

    bb = ta.bbands(close)
    if bb is not None:
        out["BBP"] = bb.iloc[:, 3]

    return out.ffill()


def _signal_trend(last):
    def above(a, b):
        return 1.0 if last.get(a, 0) > last.get(b, 0) else -1.0

    score = (above("Close", "EMA200") + above("EMA50", "EMA200") + above("Close", "EMA50")) / 3
    return score


def recommend(df: pd.DataFrame):
    if df.empty or len(df) < 20:
        return Recommendation("NEUTRAL", "#8B949E", 0, 50), pd.DataFrame()

    last = df.iloc[-1]
    adx = float(last.get("ADX14", 0))
    trend_score = _signal_trend(last)
    final_score = _clamp(trend_score * 80, -100, 100)

    if final_score >= 60:
        rec = Recommendation("COMPRA FUERTE", "#00D18F", final_score, 80)
    elif final_score >= 25:
        rec = Recommendation("COMPRA", "#2F81F7", final_score, 70)
    elif final_score <= -60:
        rec = Recommendation("VENTA FUERTE", "#FF4B4B", final_score, 80)
    elif final_score <= -25:
        rec = Recommendation("VENTA", "#FFA657", final_score, 70)
    else:
        rec = Recommendation("NEUTRAL", "#8B949E", final_score, 50)

    return rec, pd.DataFrame([{"Score Final": final_score, "ADX": adx}])


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        ticker = st.text_input("Activo (Ticker)", "AAPL").strip().upper()
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)

    if not _is_valid_ticker(ticker):
        st.error("Ticker inválido")
        st.stop()

    df = load_ohlcv(ticker, period)
    if df.empty:
        st.error(f"No se pudieron descargar datos para {ticker}")
        st.stop()

    dfi = compute_indicators(df)
    rec, summary = recommend(dfi)

    # KPIs
    last_price = dfi["Close"].iloc[-1]
    change = _safe_pct(last_price, dfi["Close"].iloc[-2])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio", f"${last_price:.2f}", f"{change:+.2f}%")
    c2.metric("Recomendación", rec.label, f"Score: {rec.score:+.0f}")
    c3.metric("RSI", f"{dfi['RSI14'].iloc[-1]:.1f}" if "RSI14" in dfi else "—")
    c4.metric("ADX", f"{dfi['ADX14'].iloc[-1]:.1f}" if "ADX14" in dfi else "—")

    tab1, tab2 = st.tabs(["Gráfico", "Detalles"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dfi.index, open=dfi["Open"], high=dfi["High"],
            low=dfi["Low"], close=dfi["Close"], name="Precio"
        ))
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Resumen Técnico")
        st.dataframe(summary, use_container_width=True)
        st.info("Esta es una versión mejorada y estable del terminal.")

    st.caption(f"QuantumShield Pro • {datetime.now().strftime('%d/%m/%Y %H:%M')}")


if __name__ == "__main__":
    main()
