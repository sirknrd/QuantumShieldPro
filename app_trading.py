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


@st.cache_data(ttl=120)
def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if not re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker.strip().upper()):
        return pd.DataFrame()
    
    st.info(f"Descargando datos de {ticker}...")
    
    try:
        # Método 1: download
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, timeout=15)
        
        # Método 2: Ticker history (más confiable a veces)
        if df.empty:
            df = yf.Ticker(ticker).history(period=period, timeout=15)
        
        if df.empty:
            st.error("No se recibieron datos")
            return pd.DataFrame()

        # LIMPIEZA DE COLUMNAS - CRÍTICO
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)
        
        # Asegurar columnas estándar
        keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available = [col for col in keep_cols if col in df.columns]
        df = df[available].copy()
        
        st.success(f"✅ Datos cargados correctamente ({len(df)} velas)")
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

    for n in [9, 21, 50, 200]:
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    adx = ta.adx(high, low, close)
    if adx is not None and not adx.empty:
        out["ADX14"] = adx.iloc[:, 0]

    return out.ffill()


def main():
    with st.sidebar:
        st.header("⚙️ Terminal")
        ticker = st.text_input("Ticker", value="NVDA").strip().upper()
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y"], index=2)

    df = load_ohlcv(ticker, period)

    if df.empty:
        st.error(f"❌ No se pudieron obtener datos para **{ticker}**")
        st.stop()

    dfi = compute_indicators(df)

    last_price = safe_float(dfi["Close"].iloc[-1])
    prev_price = safe_float(dfi["Close"].iloc[-2]) if len(dfi) > 1 else last_price
    change = ((last_price / prev_price - 1) * 100) if prev_price > 0.01 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Precio Actual", f"${last_price:,.2f}", f"{change:+.2f}%")
    col2.metric("RSI 14", f"{safe_float(dfi.get('RSI14', pd.Series([50])).iloc[-1]):.1f}")
    col3.metric("ADX 14", f"{safe_float(dfi.get('ADX14', pd.Series([20])).iloc[-1]):.1f}")

    st.subheader(f"📈 Gráfico — {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dfi.index, open=dfi["Open"], high=dfi["High"], low=dfi["Low"], close=dfi["Close"]))

    for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        if ema in dfi.columns:
            fig.add_trace(go.Scatter(x=dfi.index, y=dfi[ema], name=ema, line=dict(width=2)))

    fig.update_layout(template="plotly_dark", height=680, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ Análisis completado para **{ticker}**")


if __name__ == "__main__":
    main()
