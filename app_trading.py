from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


APP_TITLE = "QuantumShield Pro — Trading Terminal"
VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}
VALID_INTERVALS = {"1d", "1h", "30m", "15m"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recommendation:
    label: str
    color: str
    score: float  # -100..+100
    confidence: int  # 0..100


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_pct(a: float, b: float) -> float:
    try:
        b_val = float(b)
    except (TypeError, ValueError):
        return float("nan")
    if b_val == 0 or not np.isfinite(b_val):
        return float("nan")
    return (float(a) / b_val - 1.0) * 100.0


@st.cache_data(ttl=60 * 5, show_spinner=False)
def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not ticker:
        return pd.DataFrame()
    if period not in VALID_PERIODS or interval not in VALID_INTERVALS:
        return pd.DataFrame()
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        logger.exception("Error downloading OHLCV for ticker=%s period=%s interval=%s", ticker, period, interval)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_sp500_tickers() -> list[str]:
    """
    Pull S&P 500 tickers from Wikipedia (cached daily).
    Falls back to a small static list if the fetch fails.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        # First table usually contains Symbol + Security
        t = tables[0]
        if "Symbol" not in t.columns:
            raise ValueError("No Symbol column")
        tickers = (
            t["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (Yahoo format)
            .str.strip()
            .tolist()
        )
        # Keep only non-empty symbols for resilient fallback behavior.
        tickers = [x for x in tickers if x]
        return sorted(list(dict.fromkeys(tickers)))
    except Exception:
        # Minimal fallback (keeps feature working offline-ish)
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH"]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_most_active_sp500(top_n: int = 20) -> pd.DataFrame:
    """
    Compute "most active" by dollar volume (Close * Volume) on last daily bar.
    Downloads in one batch for speed.
    """
    if top_n <= 0:
        return pd.DataFrame()
    tickers = load_sp500_tickers()
    # Limit to avoid huge payloads if Wikipedia table grows or network is slow.
    tickers = tickers[:505]
    if not tickers:
        return pd.DataFrame()

    try:
        df = yf.download(
            " ".join(tickers),
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        logger.exception("Error downloading S&P 500 activity data")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    # When multiple tickers, yfinance returns MultiIndex columns: (Field, Ticker)
    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()

    last = df.tail(2)  # for change %
    rows = []
    for t in tickers:
        try:
            close = float(last["Close"][t].iloc[-1])
            prev = float(last["Close"][t].iloc[-2]) if len(last) >= 2 else float("nan")
            vol = float(last["Volume"][t].iloc[-1]) if "Volume" in last.columns.get_level_values(0) else float("nan")
            if math.isnan(close) or math.isnan(vol):
                continue
            dol = close * vol
            chg = _safe_pct(close, prev) if not math.isnan(prev) else float("nan")
            rows.append(
                {
                    "Ticker": t,
                    "Precio": close,
                    "Cambio %": chg,
                    "Volumen": vol,
                    "$ Volumen": dol,
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("$ Volumen", ascending=False).head(int(top_n)).reset_index(drop=True)
    return out


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()

    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    vol = out["Volume"] if "Volume" in out.columns else pd.Series(index=out.index, dtype="float64")

    for n in (10, 20, 50, 100, 200):
        out[f"SMA{n}"] = ta.sma(close, length=n)
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["MACD"] = macd.iloc[:, 0]
        out["MACD_SIGNAL"] = macd.iloc[:, 1]
        out["MACD_HIST"] = macd.iloc[:, 2]

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and not adx.empty:
        out["ADX14"] = adx.iloc[:, 0]
        out["DMP14"] = adx.iloc[:, 1]
        out["DMN14"] = adx.iloc[:, 2]

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        out["BBL"] = bb.iloc[:, 0]
        out["BBM"] = bb.iloc[:, 1]
        out["BBU"] = bb.iloc[:, 2]
        out["BBP"] = bb.iloc[:, 3]
        out["BBW"] = bb.iloc[:, 4]

    out["ATR14"] = ta.atr(high, low, close, length=14)

    stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None and not stochrsi.empty:
        out["STOCHRSI_K"] = stochrsi.iloc[:, 0]
        out["STOCHRSI_D"] = stochrsi.iloc[:, 1]

    sup = ta.supertrend(high, low, close, length=10, multiplier=3.0)
    if sup is not None and not sup.empty:
        # pandas_ta names are like SUPERT_10_3.0, SUPERTd_10_3.0
        for c in sup.columns:
            out[c] = sup[c]

    ichi = ta.ichimoku(high, low, close)
    if isinstance(ichi, tuple) and len(ichi) >= 1:
        ichi_df = ichi[0]
        if ichi_df is not None and not ichi_df.empty:
            for c in ichi_df.columns:
                out[c] = ichi_df[c]

    out["VOL_SMA20"] = ta.sma(vol, length=20) if not vol.empty else np.nan
    out["REL_VOL"] = out["Volume"] / out["VOL_SMA20"] if "Volume" in out.columns else np.nan

    return out.ffill()


def _signal_trend(last: pd.Series) -> tuple[float, dict[str, float]]:
    """Return score contribution (-1..+1) and detail signals."""
    details: dict[str, float] = {}

    def gt(a: str, b: str) -> float:
        if pd.isna(last.get(a)) or pd.isna(last.get(b)):
            return 0.0
        return 1.0 if last[a] > last[b] else -1.0

    def above(a: str, b: str) -> float:
        if pd.isna(last.get(a)) or pd.isna(last.get(b)):
            return 0.0
        return 1.0 if last[a] > last[b] else -1.0

    details["Price vs EMA200"] = above("Close", "EMA200")
    details["EMA50 vs EMA200"] = gt("EMA50", "EMA200")
    details["Price vs EMA50"] = above("Close", "EMA50")

    # Supertrend direction column (SUPERTd_10_3.0) is usually 1/-1
    st_dir = 0.0
    for k in last.index:
        if str(k).startswith("SUPERTd_"):
            v = last.get(k)
            if pd.isna(v):
                st_dir = 0.0
            else:
                st_dir = 1.0 if float(v) > 0 else -1.0
            break
    details["Supertrend"] = st_dir

    # Ichimoku cloud (if available): close above cloud bullish, below bearish
    ichi_score = 0.0
    span_a = last.get("ISA_9") if "ISA_9" in last.index else last.get("ICHISA_9")
    span_b = last.get("ISB_26") if "ISB_26" in last.index else last.get("ICHISB_26")
    if span_a is not None and span_b is not None and not (pd.isna(span_a) or pd.isna(span_b)):
        top = max(float(span_a), float(span_b))
        bot = min(float(span_a), float(span_b))
        if not pd.isna(last.get("Close")):
            if float(last["Close"]) > top:
                ichi_score = 1.0
            elif float(last["Close"]) < bot:
                ichi_score = -1.0
            else:
                ichi_score = 0.0
    details["Ichimoku Cloud"] = ichi_score

    vals = np.array(list(details.values()), dtype="float64")
    if vals.size == 0:
        return 0.0, details
    return float(np.nanmean(vals)), details


def _signal_momentum(last: pd.Series) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}
    rsi = last.get("RSI14")
    if rsi is not None and not pd.isna(rsi):
        r = float(rsi)
        # map 30..70 to -1..+1 (oversold->bullish, overbought->bearish)
        details["RSI14"] = _clamp((r - 50.0) / 20.0, -1.0, 1.0)
    else:
        details["RSI14"] = 0.0

    macd_hist = last.get("MACD_HIST")
    if macd_hist is not None and not pd.isna(macd_hist):
        details["MACD_HIST"] = float(np.tanh(float(macd_hist) * 5.0))
    else:
        details["MACD_HIST"] = 0.0

    k = last.get("STOCHRSI_K")
    if k is not None and not pd.isna(k):
        kk = float(k)
        details["StochRSI"] = _clamp((kk - 50.0) / 25.0, -1.0, 1.0)
    else:
        details["StochRSI"] = 0.0

    vals = np.array(list(details.values()), dtype="float64")
    return float(np.nanmean(vals)), details


def _signal_volatility(last: pd.Series) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}
    atr = last.get("ATR14")
    close = last.get("Close")
    atrp = np.nan
    if atr is not None and close is not None and not (pd.isna(atr) or pd.isna(close)) and float(close) != 0:
        atrp = float(atr) / float(close) * 100.0

    # lower ATR% -> better risk-adjusted entries, but too low can mean chop; keep neutral band
    if not pd.isna(atrp):
        if atrp < 1.0:
            details["ATR%"] = 0.2
        elif atrp < 2.5:
            details["ATR%"] = 0.0
        else:
            details["ATR%"] = -0.2
    else:
        details["ATR%"] = 0.0

    bbp = last.get("BBP")
    if bbp is not None and not pd.isna(bbp):
        # BBP ~0 -> near lower band (potential bounce), ~1 near upper band
        b = float(bbp)
        details["BB%"] = _clamp((b - 0.5) * 2.0, -1.0, 1.0)
    else:
        details["BB%"] = 0.0

    vals = np.array(list(details.values()), dtype="float64")
    return float(np.nanmean(vals)), details


def _signal_volume(last: pd.Series) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}
    rv = last.get("REL_VOL")
    if rv is None or pd.isna(rv):
        details["RelVol"] = 0.0
    else:
        # >1 supports breakout/trend continuation; cap effect
        details["RelVol"] = _clamp((float(rv) - 1.0) / 1.0, -0.5, 0.8)
    return float(np.nanmean(list(details.values()))), details


def recommend(df: pd.DataFrame) -> tuple[Recommendation, pd.DataFrame, pd.DataFrame, bool, float]:
    """Return recommendation, group explanation table, indicator detail table, trend regime flag and ADX value."""
    if df.empty:
        empty_rec = Recommendation("NEUTRAL", "#8B949E", 0.0, 0)
        return empty_rec, pd.DataFrame(), pd.DataFrame(), False, 0.0
    last = df.iloc[-1]

    adx = last.get("ADX14")
    adx_v = float(adx) if adx is not None and not pd.isna(adx) else 0.0
    # ADX regime: >=25 is a more conservative "trend" threshold
    trending = adx_v >= 25.0

    trend_s, trend_d = _signal_trend(last)
    mom_s, mom_d = _signal_momentum(last)
    vol_s, vol_d = _signal_volatility(last)
    v_s, v_d = _signal_volume(last)

    # Weights depending on regime
    if trending:
        w = {"Trend": 0.45, "Momentum": 0.30, "Volatility": 0.10, "Volume": 0.15}
    else:
        w = {"Trend": 0.30, "Momentum": 0.35, "Volatility": 0.20, "Volume": 0.15}

    comp = {
        "Trend": float(trend_s),
        "Momentum": float(mom_s),
        "Volatility": float(vol_s),
        "Volume": float(v_s),
    }
    raw = sum(w[k] * comp[k] for k in comp.keys())
    score = _clamp(raw * 100.0, -100.0, 100.0)

    # Confidence combines regime strength + signal alignment
    alignment = float(np.mean([abs(trend_s), abs(mom_s), abs(vol_s), abs(v_s)]))
    regime = _clamp((adx_v - 10.0) / 25.0, 0.0, 1.0)
    confidence = int(round(_clamp((0.55 * alignment + 0.45 * regime) * 100.0, 0.0, 100.0)))

    if score >= 60:
        rec = Recommendation("COMPRA FUERTE", "#00D18F", score, confidence)
    elif score >= 20:
        rec = Recommendation("COMPRA", "#2F81F7", score, confidence)
    elif score <= -60:
        rec = Recommendation("VENTA FUERTE", "#FF4B4B", score, confidence)
    elif score <= -20:
        rec = Recommendation("VENTA", "#FFA657", score, confidence)
    else:
        rec = Recommendation("NEUTRAL", "#8B949E", score, confidence)

    rows = []
    for group, weight in w.items():
        rows.append(
            {
                "Grupo": group,
                "Peso": weight,
                "Score (-1..+1)": comp[group],
                "Contribución": weight * comp[group],
            }
        )
    expl = pd.DataFrame(rows)

    # Add indicator-level details as a second table-like block
    detail_rows = []
    for name, val in {**trend_d, **mom_d, **vol_d, **v_d}.items():
        detail_rows.append({"Indicador": name, "Señal (-1..+1)": float(val)})
    details = pd.DataFrame(detail_rows).sort_values("Indicador")

    return rec, expl, details, trending, adx_v


def regime_label(adx_v: float) -> str:
    if not adx_v:
        return "—"
    if adx_v >= 35:
        return "Tendencia fuerte"
    if adx_v >= 25:
        return "Tendencia"
    if adx_v >= 15:
        return "Mixto"
    return "Rango"


def format_price(x: float) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if abs(float(x)) >= 1000:
        return f"{float(x):,.2f}"
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def build_chart(df: pd.DataFrame, overlays: Iterable[str]) -> go.Figure:
    view = df.tail(220).copy()
    x = view.index

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            name="Precio",
            increasing_line_color="#00D18F",
            decreasing_line_color="#FF4B4B",
        )
    )

    if "Volumen" in overlays and "Volume" in view.columns:
        fig.add_trace(
            go.Bar(
                x=x,
                y=view["Volume"],
                name="Volumen",
                marker_color="rgba(139,148,158,0.35)",
                yaxis="y2",
            )
        )

    if "EMA 50/200" in overlays:
        for n, col, w in [(50, "#2F81F7", 1), (200, "#FFA657", 2)]:
            k = f"EMA{n}"
            if k in view.columns:
                fig.add_trace(go.Scatter(x=x, y=view[k], name=f"EMA {n}", line=dict(color=col, width=w)))

    if "Bandas Bollinger" in overlays and all(k in view.columns for k in ("BBL", "BBM", "BBU")):
        fig.add_trace(go.Scatter(x=x, y=view["BBU"], name="BB Upper", line=dict(color="rgba(139,148,158,0.45)", width=1)))
        fig.add_trace(go.Scatter(x=x, y=view["BBM"], name="BB Mid", line=dict(color="rgba(139,148,158,0.30)", width=1)))
        fig.add_trace(go.Scatter(x=x, y=view["BBL"], name="BB Lower", line=dict(color="rgba(139,148,158,0.45)", width=1)))

    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Precio"),
        yaxis2=dict(title="Vol", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.markdown(
        """
<style>
  /* Reduce chances of overlap: avoid structural HTML wrappers around Streamlit widgets. */
  [data-testid="stAppViewContainer"] { overflow-x: hidden; }
  .qsp-title {font-size:18px; font-weight:700; letter-spacing:0.2px; margin: 0;}
  .qsp-sub {font-size:12px; color: rgba(230,237,243,0.75); margin: 0;}
  .qsp-pill {padding:6px 10px; border-radius:999px; font-weight:700; font-size:12px; border:1px solid rgba(139,148,158,0.25); display:inline-block;}
  .qsp-rec {
    border-radius: 12px;
    padding: 12px 12px 10px 12px;
    border: 1px solid rgba(139,148,158,0.18);
    background: rgba(15,23,34,0.35);
  }
  .qsp-rec-label { font-size: 14px; font-weight: 800; letter-spacing: 0.4px; margin: 0 0 6px 0; }
  .qsp-rec-score { font-size: 28px; font-weight: 800; margin: 0; line-height: 1.1; }
  .qsp-rec-sub { font-size: 12px; color: rgba(230,237,243,0.75); margin: 4px 0 0 0; }

  /* KPI cards (Streamlit-native components) */
  div[data-testid="stMetric"] {
    background: rgba(15,23,34,0.35);
    border: 1px solid rgba(139,148,158,0.18);
    border-radius: 12px;
    padding: 12px 12px 10px 12px;
  }
  div[data-testid="stMetric"] > label { margin-bottom: 4px; }

  /* Compact spacing so rows don't collide on small widths */
  div[data-testid="stVerticalBlock"] { gap: 0.75rem; }
  .block-container { padding-top: 1.0rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Terminal")
        ticker = st.text_input("Activo (Ticker)", value="AAPL").strip().upper()
        colp = st.columns(2)
        with colp[0]:
            period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        with colp[1]:
            interval = st.selectbox("Intervalo", ["1d", "1h", "30m", "15m"], index=0)

        st.markdown("---")
        st.markdown("### Gráfico")
        overlays = st.multiselect(
            "Overlays",
            ["Volumen", "EMA 50/200", "Bandas Bollinger"],
            default=["Volumen", "EMA 50/200"],
        )

        st.markdown("---")
        st.markdown("### Radar")
        radar_raw = st.text_area(
            "Lista (coma)",
            value="AAPL, MSFT, NVDA, TSLA, BTC-USD",
            height=90,
        )
        radar_items = [x.strip().upper() for x in radar_raw.split(",") if x.strip()]

    # Header (Streamlit layout; no raw HTML containers)
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    hL, hR = st.columns([0.78, 0.22], vertical_alignment="center")
    with hL:
        st.markdown(f"<div class='qsp-title'>{ticker} <span class='qsp-sub'>• {APP_TITLE}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='qsp-sub'>Actualizado: {now} • Fuente: Yahoo Finance</div>", unsafe_allow_html=True)
    with hR:
        st.empty()

    df = load_ohlcv(ticker, period=period, interval=interval)
    if df.empty:
        st.error("No pude descargar datos. Revisa el ticker o tu conexión.")
        return

    dfi = compute_indicators(df)
    last = dfi.iloc[-1]
    prev_close = dfi["Close"].iloc[-2] if len(dfi) >= 2 else float("nan")
    chg = _safe_pct(float(last["Close"]), float(prev_close)) if not pd.isna(prev_close) else float("nan")

    rec, expl, details, trending, adx_v = recommend(dfi)

    # KPI row (Investing-like)
    k1, k2, k3, k4, k5 = st.columns([1.2, 1, 1, 1, 1.2])
    with k1:
        st.metric("Último", format_price(float(last["Close"])))
        st.caption(f"Cambio: {chg:.2f}%" if not pd.isna(chg) else "Cambio: —")
    with k2:
        st.metric("RSI (14)", f"{float(last.get('RSI14', np.nan)):.1f}" if not pd.isna(last.get("RSI14")) else "—")
        st.caption("Momentum")
    with k3:
        st.metric("ADX (14)", f"{adx_v:.1f}" if adx_v else "—")
        st.caption(f"Régimen: {regime_label(adx_v)}")
    with k4:
        atr = last.get("ATR14")
        atrp = float(atr) / float(last["Close"]) * 100.0 if atr is not None and not pd.isna(atr) and float(last["Close"]) != 0 else np.nan
        st.metric("ATR% (14)", f"{atrp:.2f}%" if not pd.isna(atrp) else "—")
        st.caption("Volatilidad/Riesgo")
    with k5:
        # Make recommendation card more visible (without wrapping Streamlit widgets in HTML)
        tint = rec.color
        bg = "rgba(139,148,158,0.10)"
        if rec.label.startswith("COMPRA FUERTE"):
            bg = "rgba(0, 209, 143, 0.16)"
        elif rec.label.startswith("COMPRA"):
            bg = "rgba(47, 129, 247, 0.16)"
        elif rec.label.startswith("VENTA FUERTE"):
            bg = "rgba(255, 75, 75, 0.16)"
        elif rec.label.startswith("VENTA"):
            bg = "rgba(255, 166, 87, 0.16)"

        st.markdown(
            f"""
<div class="qsp-rec" style="border-color: {tint}; background: {bg};">
  <div class="qsp-rec-label" style="color:{tint};">RECOMENDACIÓN</div>
  <div class="qsp-rec-score" style="color:{tint};">{rec.label}</div>
  <div class="qsp-rec-sub">Score: <b>{rec.score:+.0f}</b>/100 • Confianza: <b>{rec.confidence}%</b></div>
</div>
            """.strip(),
            unsafe_allow_html=True,
        )

    t_overview, t_chart, t_tech, t_radar = st.tabs(["Resumen", "Gráfico", "Técnicos", "Radar"])

    with t_overview:
        cL, cR = st.columns([1.05, 0.95])
        with cL:
            st.markdown("### Snapshot")
            snap = {
                "Open": last.get("Open"),
                "High": last.get("High"),
                "Low": last.get("Low"),
                "Close": last.get("Close"),
                "Volumen": last.get("Volume"),
                "RelVol (20)": last.get("REL_VOL"),
            }
            snap_df = pd.DataFrame(
                [{"Campo": k, "Valor": format_price(float(v)) if v is not None and not pd.isna(v) else "—"} for k, v in snap.items()]
            )
            st.dataframe(snap_df, use_container_width=True, hide_index=True)

        with cR:
            st.markdown("### Explicación del score")
            st.dataframe(
                expl.assign(
                    **{
                        "Peso": (expl["Peso"] * 100).round(0).astype(int).astype(str) + "%",
                        "Score (-1..+1)": expl["Score (-1..+1)"].round(3),
                        "Contribución": (expl["Contribución"] * 100).round(1),
                    }
                )[["Grupo", "Peso", "Score (-1..+1)", "Contribución"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Contribución está en puntos porcentuales del score total (aprox.).")

        st.markdown("---")
        st.markdown("### Más activas del S&P 500 (por $ volumen)")
        act = load_most_active_sp500(top_n=15)
        if act.empty:
            st.warning("No se pudo cargar el ranking de más activas (sin datos o sin conexión).")
        else:
            show = act.copy()
            show["Precio"] = show["Precio"].map(lambda x: format_price(float(x)))
            show["Cambio %"] = show["Cambio %"].map(lambda x: f"{float(x):.2f}%" if not pd.isna(x) else "—")
            show["Volumen"] = show["Volumen"].map(lambda x: f"{float(x):,.0f}")
            show["$ Volumen"] = show["$ Volumen"].map(lambda x: f"{float(x):,.0f}")
            st.dataframe(show, use_container_width=True, hide_index=True)

    with t_chart:
        st.markdown("### Gráfico")
        fig = build_chart(dfi, overlays=overlays)
        st.plotly_chart(fig, use_container_width=True)

    with t_tech:
        st.markdown("### Señales por indicador")
        st.dataframe(
            details.assign(**{"Señal (-1..+1)": details["Señal (-1..+1)"].round(3)}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Interpretación rápida: +1 bullish, -1 bearish, 0 neutral.")

    with t_radar:
        st.markdown("### Radar multi-activo")
        if not radar_items:
            st.info("Agrega tickers en la barra lateral para ver el radar.")
        else:
            rows = []
            for it in radar_items[:30]:
                d0 = load_ohlcv(it, period=period, interval=interval)
                if d0.empty:
                    continue
                di = compute_indicators(d0)
                r, _, _, tr, ax = recommend(di)
                l0 = di.iloc[-1]
                p = float(l0["Close"])
                pc = float(di["Close"].iloc[-2]) if len(di) >= 2 else float("nan")
                ch = _safe_pct(p, pc) if not pd.isna(pc) else float("nan")
                rows.append(
                    {
                        "Activo": it,
                        "Precio": format_price(p),
                        "Cambio %": f"{ch:.2f}%" if not pd.isna(ch) else "—",
                        "Recomendación": r.label,
                        "Score": f"{r.score:+.0f}",
                        "Confianza": f"{r.confidence}%",
                        "ADX": f"{ax:.1f}" if ax else "—",
                        "Régimen": regime_label(ax),
                        "RelVol": f"{float(l0.get('REL_VOL', np.nan)):.2f}x" if not pd.isna(l0.get("REL_VOL")) else "—",
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No se pudieron cargar activos del radar (tickers inválidos o sin datos).")


if __name__ == "__main__":
    main()
