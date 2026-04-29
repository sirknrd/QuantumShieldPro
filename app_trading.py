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

# Regex para validar tickers
_TICKER_RE = re.compile(r"^[A-Z0-9\-\.]{1,10}$")


@dataclass(frozen=True)
class Recommendation:
    label: str
    color: str
    score: float   # -100..+100
    confidence: int  # 0..100


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_pct(a: float, b: float) -> float:
    if b == 0 or (isinstance(b, float) and (math.isnan(b) or math.isinf(b))):
        return float("nan")
    return (a / b - 1.0) * 100.0


def _is_valid_ticker(ticker: str) -> bool:
    """Valida formato de ticker"""
    return bool(_TICKER_RE.match(ticker.strip().upper()))


@st.cache_data(ttl=60 * 5, show_spinner=False)
def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=False, group_by="column", threads=True
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
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
        tickers = (
            t["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        return sorted(list(dict.fromkeys([x for x in tickers if x])))
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH"]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_most_active_sp500(top_n: int = 20) -> pd.DataFrame:
    """Versión corregida y robusta"""
    tickers = load_sp500_tickers()[:505]
    if not tickers:
        return pd.DataFrame()

    try:
        df = yf.download(
            " ".join(tickers), period="5d", interval="1d",
            progress=False, auto_adjust=False, group_by="column", threads=True
        )
        if df is None or df.empty:
            return pd.DataFrame()

        last = df.tail(2)
        rows = []

        if isinstance(df.columns, pd.MultiIndex):
            close_col = last["Close"]
            vol_col = last.get("Volume", pd.DataFrame())
        else:
            close_col = last[["Close"]] if "Close" in last.columns else pd.DataFrame()
            vol_col = last[["Volume"]] if "Volume" in last.columns else pd.DataFrame()

        for t in tickers:
            try:
                if t not in close_col.columns:
                    continue
                close = float(close_col[t].iloc[-1])
                prev = float(close_col[t].iloc[-2]) if len(last) >= 2 else float("nan")
                vol = float(vol_col[t].iloc[-1]) if not vol_col.empty and t in vol_col.columns else float("nan")

                if math.isnan(close) or math.isnan(vol) or vol == 0:
                    continue

                rows.append({
                    "Ticker": t,
                    "Precio": close,
                    "Cambio %": _safe_pct(close, prev) if not math.isnan(prev) else float("nan"),
                    "Volumen": vol,
                    "$ Volumen": close * vol,
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        return out.sort_values("$ Volumen", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def regime_label(adx: float) -> str:
    """Función agregada - estaba faltando (error crítico)"""
    if pd.isna(adx) or adx == 0:
        return "—"
    if adx >= 35:
        return "Tendencia fuerte"
    if adx >= 25:
        return "Tendencia"
    if adx >= 15:
        return "Mixto"
    return "Rango"


# ====================== FIN PARTE 1/4 ======================
print("✅ Parte 1/4 cargada correctamente")
# ====================== PARTE 2/4 ======================

# ====================== CONTEXTO MACRO ======================

_SECTOR_ETFS = {
    "XLK":  "Tecnología",
    "XLF":  "Financiero",
    "XLV":  "Salud",
    "XLE":  "Energía",
    "XLY":  "Consumo Discrecional",
    "XLP":  "Consumo Básico",
    "XLI":  "Industrial",
    "XLB":  "Materiales",
    "XLRE": "Inmobiliario",
    "XLU":  "Utilities",
    "XLC":  "Comunicaciones",
}

_MACRO_REFS = {
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
    "DIA":  "Dow Jones",
    "IWM":  "Russell 2000",
    "VIX":  "VIX (Miedo)",
    "TLT":  "Bonos 20Y",
    "GLD":  "Oro",
    "UUP":  "Dólar (DXY)",
}


@st.cache_data(ttl=60 * 15, show_spinner=False)
def load_macro_context(ticker: str, period: str = "1y") -> dict:
    refs = list(_MACRO_REFS.keys())
    batch = [ticker] + refs

    try:
        raw = yf.download(
            " ".join(batch), period=period, interval="1d",
            progress=False, auto_adjust=True, group_by="column", threads=True
        )
        if raw is None or raw.empty:
            return {}

        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"].copy()
        else:
            close_df = raw[["Close"]].copy()
            close_df.columns = [ticker]

        rets = close_df.pct_change().dropna()
        result: dict = {}

        # Beta y correlación con SPY
        if ticker in rets.columns and "SPY" in rets.columns:
            common = rets[[ticker, "SPY"]].dropna()
            if len(common) > 20:
                cov = common.cov()
                var_sp = float(cov.loc["SPY", "SPY"])
                beta = float(cov.loc[ticker, "SPY"]) / var_sp if var_sp != 0 else float("nan")
                corr = float(common[ticker].corr(common["SPY"]))
                result["beta_spy"] = beta
                result["corr_spy"] = corr

        # Correlación con QQQ
        if ticker in rets.columns and "QQQ" in rets.columns:
            common_q = rets[[ticker, "QQQ"]].dropna()
            if len(common_q) > 20:
                result["corr_qqq"] = float(common_q[ticker].corr(common_q["QQQ"]))

        # Fuerza relativa vs SPY
        if ticker in close_df.columns and "SPY" in close_df.columns:
            for days, label in [(21, "1m"), (63, "3m"), (126, "6m")]:
                if len(close_df) > days:
                    tk_ret = _safe_pct(float(close_df[ticker].iloc[-1]), float(close_df[ticker].iloc[-(days+1)]))
                    spy_ret = _safe_pct(float(close_df["SPY"].iloc[-1]), float(close_df["SPY"].iloc[-(days+1)]))
                    result[f"rs_{label}"] = tk_ret - spy_ret

        # Régimen de mercado
        if "SPY" in close_df.columns:
            spy_s = close_df["SPY"].dropna()
            if len(spy_s) >= 200:
                sma200 = float(spy_s.rolling(200).mean().iloc[-1])
                spy_last = float(spy_s.iloc[-1])
                result["spy_vs_sma200"] = _safe_pct(spy_last, sma200)
                result["market_regime"] = "Bull" if spy_last > sma200 else "Bear"
            if len(spy_s) >= 2:
                result["spy_1d"] = _safe_pct(float(spy_s.iloc[-1]), float(spy_s.iloc[-2]))

        # VIX
        if "VIX" in close_df.columns:
            vix_s = close_df["VIX"].dropna()
            if not vix_s.empty:
                vix_v = float(vix_s.iloc[-1])
                result["vix"] = vix_v
                result["vix_regime"] = "Pánico" if vix_v > 30 else "Elevado" if vix_v > 20 else "Normal"

        # Snapshots
        snapshots = {}
        for sym, name in _MACRO_REFS.items():
            if sym in close_df.columns:
                s = close_df[sym].dropna()
                if len(s) >= 2:
                    chg = _safe_pct(float(s.iloc[-1]), float(s.iloc[-2]))
                    snapshots[sym] = {"name": name, "price": float(s.iloc[-1]), "chg1d": chg}
        result["snapshots"] = snapshots

        return result
    except Exception:
        return {}


@st.cache_data(ttl=60 * 15, show_spinner=False)
def guess_sector_etf(ticker: str) -> tuple[str, str]:
    _SECTOR_TO_ETF = {
        "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
        "Energy": "XLE", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
        "Industrials": "XLI", "Basic Materials": "XLB", "Real Estate": "XLRE",
        "Utilities": "XLU", "Communication Services": "XLC",
    }
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "")
        etf = _SECTOR_TO_ETF.get(sector, "")
        return etf, sector
    except Exception:
        return "", ""


@st.cache_data(ttl=60 * 15, show_spinner=False)
def load_sector_rs(ticker: str, sector_etf: str, period: str = "6mo") -> dict:
    if not sector_etf:
        return {}
    try:
        raw = yf.download(f"{ticker} {sector_etf}", period=period, interval="1d",
                          progress=False, auto_adjust=True, group_by="column", threads=True)
        if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
            return {}
        close = raw["Close"].dropna()
        if ticker not in close.columns or sector_etf not in close.columns:
            return {}
        tk_r = _safe_pct(float(close[ticker].iloc[-1]), float(close[ticker].iloc[0]))
        sect_r = _safe_pct(float(close[sector_etf].iloc[-1]), float(close[sector_etf].iloc[0]))
        return {"ticker_return": tk_r, "sector_return": sect_r, "rs_vs_sector": tk_r - sect_r}
    except Exception:
        return {}


# ====================== FIN PARTE 2/4 ======================
print("✅ Parte 2/4 cargada correctamente (Macro + Sector)")

# ====================== PARTE 3/4 ======================

# ====================== ALERTAS + NEWS + SENTIMIENTO ======================

_ALERT_CONDITIONS = {
    "RSI < umbral (sobreventa)":         lambda last, v: _safe_val(last, "RSI14") < v,
    "RSI > umbral (sobrecompra)":        lambda last, v: _safe_val(last, "RSI14") > v,
    "Precio cruza EMA50 al alza":        lambda last, v: _safe_val(last, "Close") > _safe_val(last, "EMA50"),
    "Precio cruza EMA200 al alza":       lambda last, v: _safe_val(last, "Close") > _safe_val(last, "EMA200"),
    "MACD histograma positivo":          lambda last, v: _safe_val(last, "MACD_HIST") > 0,
    "ADX > umbral (tendencia fuerte)":   lambda last, v: _safe_val(last, "ADX14") > v,
    "Volumen relativo > umbral":         lambda last, v: _safe_val(last, "REL_VOL") > v,
    "Score técnico > umbral":            lambda last, v: False,
    "Score técnico < umbral":            lambda last, v: False,
    "Supertrend alcista":                lambda last, v: _supertrend_dir(last) == 1,
    "Supertrend bajista":                lambda last, v: _supertrend_dir(last) == -1,
    "BB%: precio cerca de banda baja":   lambda last, v: _safe_val(last, "BBP") < 0.1,
    "BB%: precio cerca de banda alta":   lambda last, v: _safe_val(last, "BBP") > 0.9,
    "CCI < -100 (sobreventa extrema)":   lambda last, v: _safe_val(last, "CCI14") < -100,
    "CCI > +100 (sobrecompra extrema)":  lambda last, v: _safe_val(last, "CCI14") > 100,
}


def _safe_val(last: pd.Series, col: str) -> float:
    v = last.get(col)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _supertrend_dir(last: pd.Series) -> int:
    for k in last.index:
        if str(k).startswith("SUPERTd_"):
            v = last.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return 1 if float(v) > 0 else -1
    return 0


@dataclass
class Alert:
    id: str
    ticker: str
    condition: str
    threshold: float
    active: bool = True
    triggered: bool = False
    trigger_ts: str = ""
    note: str = ""


def _init_alerts() -> None:
    if "qsp_alerts" not in st.session_state:
        st.session_state["qsp_alerts"] = []
    if "qsp_alert_log" not in st.session_state:
        st.session_state["qsp_alert_log"] = []


def evaluate_alerts(dfi: pd.DataFrame, ticker: str, score: float) -> list[Alert]:
    _init_alerts()
    alerts = st.session_state["qsp_alerts"]
    log = st.session_state["qsp_alert_log"]
    fired: list[Alert] = []
    last = dfi.iloc[-1]

    for a in alerts:
        if not a.active or a.ticker.upper() != ticker.upper():
            continue

        fn = _ALERT_CONDITIONS.get(a.condition)
        triggered = False

        if fn is not None:
            if "Score técnico >" in a.condition:
                triggered = score > a.threshold
            elif "Score técnico <" in a.condition:
                triggered = score < a.threshold
            else:
                try:
                    triggered = bool(fn(last, a.threshold))
                except Exception:
                    triggered = False

        if triggered and not a.triggered:
            a.triggered = True
            a.trigger_ts = datetime.now().strftime("%d/%m %H:%M")
            log.append({
                "ts": a.trigger_ts, "ticker": a.ticker, "condición": a.condition,
                "umbral": a.threshold, "nota": a.note
            })
            fired.append(a)
        elif not triggered:
            a.triggered = False

    return fired


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_news(ticker: str, max_items: int = 15) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(ticker)}"
    items: list[dict] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        channel = root.find("channel")
        if channel is None:
            return items
        for item in channel.findall("item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title and link:
                try:
                    dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
                    ago = f"hace {(datetime.now(tz=timezone.utc) - dt).days}d" if (datetime.now(tz=timezone.utc) - dt).days > 0 else "hoy"
                except Exception:
                    ago = pub[:16] if pub else "—"
                items.append({"title": title, "link": link, "ago": ago, "source": "Yahoo Finance"})
    except Exception:
        pass
    return items


# ====================== SENTIMIENTO ======================

_BULL_WORDS = {"surge", "rally", "gain", "beat", "upgrade", "buy", "strong", "growth", "bullish", "upside", "positive"}
_BEAR_WORDS = {"fall", "drop", "miss", "downgrade", "sell", "weak", "loss", "bearish", "downside", "negative", "crash"}

def score_headline(title: str) -> float:
    words = set(re.findall(r"[a-z]+", title.lower()))
    bull = len(words & _BULL_WORDS)
    bear = len(words & _BEAR_WORDS)
    total = bull + bear
    return (bull - bear) / total if total > 0 else 0.0


def aggregate_sentiment(news: list[dict]) -> dict:
    if not news:
        return {"score": 0.0, "label": "Neutral", "color": "#8B949E", "bull": 0, "bear": 0, "neutral": 0}
    scores = [score_headline(n["title"]) for n in news]
    avg = float(np.mean(scores))
    bull = sum(1 for s in scores if s > 0.1)
    bear = sum(1 for s in scores if s < -0.1)
    neutral = len(scores) - bull - bear

    if avg > 0.15:
        label, color = "Bullish", "#00D18F"
    elif avg < -0.15:
        label, color = "Bearish", "#FF4B4B"
    else:
        label, color = "Neutral", "#8B949E"

    return {"score": avg, "label": label, "color": color, "bull": bull, "bear": bear, "neutral": neutral}


# ====================== FIN PARTE 3/4 ======================
print("✅ Parte 3/4 cargada correctamente (Alertas + Noticias + Sentimiento)")
# ====================== PARTE 4/4 (FINAL) ======================

# ====================== INDICADORES + RECOMMEND + GRÁFICO ======================

_MACD_COL = "MACD_12_26_9"
_MACDS_COL = "MACDs_12_26_9"
_MACDH_COL = "MACDh_12_26_9"
_ADX_COL = "ADX_14"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
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
        out["MACD"] = macd.get(_MACD_COL, macd.iloc[:, 0])
        out["MACD_SIGNAL"] = macd.get(_MACDS_COL, macd.iloc[:, 1])
        out["MACD_HIST"] = macd.get(_MACDH_COL, macd.iloc[:, 2])

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and not adx.empty:
        out["ADX14"] = adx.get(_ADX_COL, adx.iloc[:, 0])

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        out["BBL"] = bb.iloc[:, 0]
        out["BBM"] = bb.iloc[:, 1]
        out["BBU"] = bb.iloc[:, 2]
        out["BBP"] = bb.iloc[:, 3]

    out["ATR14"] = ta.atr(high, low, close, length=14)
    out["CCI14"] = ta.cci(high, low, close, length=14)

    sup = ta.supertrend(high, low, close, length=10, multiplier=3.0)
    if sup is not None and not sup.empty:
        for c in sup.columns:
            out[c] = sup[c]

    out["VOL_SMA20"] = ta.sma(vol, length=20) if not vol.empty else np.nan
    out["REL_VOL"] = out["Volume"] / out["VOL_SMA20"] if "Volume" in out.columns else np.nan

    indicator_cols = [c for c in out.columns if c not in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    out[indicator_cols] = out[indicator_cols].ffill()

    return out
# ====================== CORRECCIÓN PARA EL ERROR ======================

def _signal_trend(last: pd.Series) -> tuple[float, dict]:
    """Versión corregida y segura"""
    details: dict[str, float] = {}

    def above(a: str, b: str) -> float:
        va = last.get(a)
        vb = last.get(b)
        if pd.isna(va) or pd.isna(vb):
            return 0.0
        return 1.0 if float(va) > float(vb) else -1.0

    details["Price vs EMA200"] = above("Close", "EMA200")
    details["EMA50 vs EMA200"] = above("EMA50", "EMA200")
    details["Price vs EMA50"]  = above("Close", "EMA50")

    # Supertrend direction
    st_dir = 0.0
    for k in last.index:
        if str(k).startswith("SUPERTd_"):
            v = last.get(k)
            if v is not None and not pd.isna(v):
                st_dir = 1.0 if float(v) > 0 else -1.0
            break
    details["Supertrend"] = st_dir

    vals = np.array(list(details.values()), dtype="float64")
    score = float(np.nanmean(vals)) if vals.size > 0 else 0.0
    return score, details


def recommend(df: pd.DataFrame) -> tuple[Recommendation, pd.DataFrame, pd.DataFrame, bool, float]:
    """Versión corregida y estable"""
    if df.empty:
        return Recommendation("NEUTRAL", "#8B949E", 0.0, 0), pd.DataFrame(), pd.DataFrame(), False, 0.0

    last = df.iloc[-1]
    adx_v = float(last.get("ADX14", 0)) if not pd.isna(last.get("ADX14")) else 0.0
    trending = adx_v >= 25.0

    trend_s, trend_d = _signal_trend(last)

    # Score simplificado pero funcional
    raw_score = trend_s * 0.7   # Damos más peso al trend
    score = _clamp(raw_score * 100.0, -100.0, 100.0)

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

    # Explanation table
    expl = pd.DataFrame([
        {"Grupo": "Trend", "Peso": 0.7, "Score (-1..+1)": round(trend_s, 2), "Contribución": round(0.7 * trend_s, 2)}
    ])

    # Details table
    detail_rows = [{"Indicador": name, "Señal (-1..+1)": round(val, 2)} for name, val in trend_d.items()]
    details = pd.DataFrame(detail_rows)

    return rec, expl, details, trending, adx_v
