from __future__ import annotations

import concurrent.futures
import math
import re
import urllib.request
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

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
    if b == 0 or (isinstance(b, float) and (math.isnan(b) or math.isinf(b))):
        return float("nan")
    return (a / b - 1.0) * 100.0


def _is_valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_RE.match(ticker.strip().upper()))


@st.cache_data(ttl=60 * 5, show_spinner=False)
def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        t = tables[0]
        tickers = t["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip().tolist()
        return sorted(list(dict.fromkeys([x for x in tickers if x])))
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_most_active_sp500(top_n: int = 15) -> pd.DataFrame:
    tickers = load_sp500_tickers()
    try:
        df = yf.download(" ".join(tickers[:100]), period="5d", interval="1d", progress=False, threads=True)
        if df is None or df.empty:
            return pd.DataFrame()
        last = df.tail(2)
        rows = []
        for t in tickers[:100]:
            try:
                close = float(last["Close"][t].iloc[-1])
                prev = float(last["Close"][t].iloc[-2]) if len(last) >= 2 else float("nan")
                vol = float(last["Volume"][t].iloc[-1]) if "Volume" in last.columns.get_level_values(0) else float("nan")
                if math.isnan(close) or math.isnan(vol):
                    continue
                rows.append({
                    "Ticker": t,
                    "Precio": close,
                    "Cambio %": _safe_pct(close, prev),
                    "$ Volumen": close * vol,
                })
            except Exception:
                continue
        out = pd.DataFrame(rows)
        return out.sort_values("$ Volumen", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def regime_label(adx: float) -> str:
    if pd.isna(adx) or adx == 0:
        return "—"
    if adx >= 35: return "Tendencia fuerte"
    if adx >= 25: return "Tendencia"
    if adx >= 15: return "Mixto"
    return "Rango"


def _signal_trend(last: pd.Series) -> tuple[float, dict]:
    details: dict[str, float] = {}
    def above(a: str, b: str) -> float:
        va = last.get(a)
        vb = last.get(b)
        if pd.isna(va) or pd.isna(vb):
            return 0.0
        return 1.0 if float(va) > float(vb) else -1.0

    details["Price vs EMA200"] = above("Close", "EMA200")
    details["EMA50 vs EMA200"] = above("EMA50", "EMA200")
    details["Price vs EMA50"] = above("Close", "EMA50")

    st_dir = 0.0
    for k in last.index:
        if str(k).startswith("SUPERTd_"):
            v = last.get(k)
            if v is not None and not pd.isna(v):
                st_dir = 1.0 if float(v) > 0 else -1.0
            break
    details["Supertrend"] = st_dir

    vals = np.array(list(details.values()), dtype="float64")
    return float(np.nanmean(vals)) if vals.size else 0.0, details


def recommend(df: pd.DataFrame) -> tuple[Recommendation, pd.DataFrame, pd.DataFrame, bool, float]:
    if df.empty or len(df) < 10:
        return Recommendation("NEUTRAL", "#8B949E", 0, 0), pd.DataFrame(), pd.DataFrame(), False, 0.0

    last = df.iloc[-1]
    adx_v = float(last.get("ADX14", 0)) if not pd.isna(last.get("ADX14")) else 0.0
    trending = adx_v >= 25

    trend_s, trend_d = _signal_trend(last)
    score = _clamp(trend_s * 70, -100, 100)

    if score >= 60:
        rec = Recommendation("COMPRA FUERTE", "#00D18F", score, 80)
    elif score >= 20:
        rec = Recommendation("COMPRA", "#2F81F7", score, 65)
    elif score <= -60:
        rec = Recommendation("VENTA FUERTE", "#FF4B4B", score, 80)
    elif score <= -20:
        rec = Recommendation("VENTA", "#FFA657", score, 65)
    else:
        rec = Recommendation("NEUTRAL", "#8B949E", score, 50)

    expl = pd.DataFrame([{"Grupo": "Trend", "Peso": 70, "Score": round(trend_s,2), "Contribución": round(trend_s*70,1)}])
    details = pd.DataFrame([{"Indicador": k, "Señal": round(v,2)} for k,v in trend_d.items()])

    return rec, expl, details, trending, adx_v


def build_chart(df: pd.DataFrame):
    view = df.tail(200)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"]))
    fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False, title="Gráfico")
    return fig


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        ticker = st.text_input("Ticker", "AAPL").strip().upper()
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)

    if not _is_valid_ticker(ticker):
        st.error("Ticker inválido")
        return

    df = load_ohlcv(ticker, period, "1d")
    if df.empty:
        st.error(f"No se pudieron descargar datos para {ticker}")
        return

    dfi = df  # En versión simplificada usamos df directamente
    rec, expl, details, _, adx = recommend(dfi)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precio", f"{dfi['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Recomendación", rec.label)
    with col3:
        st.metric("ADX", f"{adx:.1f}", regime_label(adx))

    tab1, tab2 = st.tabs(["Gráfico", "Análisis"])
    with tab1:
        fig = build_chart(dfi)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Señales")
        st.dataframe(details, use_container_width=True)
        st.subheader("Más Activas")
        act = load_most_active_sp500(10)
        if not act.empty:
            st.dataframe(act, use_container_width=True)

    st.caption("QuantumShield Pro - Versión Corregida")


if __name__ == "__main__":
    main()
