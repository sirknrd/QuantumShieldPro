from __future__ import annotations

import concurrent.futures
import math
import re
import urllib.request
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


APP_TITLE = "QuantumShield Pro — Trading Terminal"

# Regex para validar tickers: letras, números, guiones y puntos, 1-10 caracteres
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
    """Valida que el ticker tenga un formato aceptable antes de hacer la llamada a yfinance."""
    return bool(_TICKER_RE.match(ticker.strip().upper()))


@st.cache_data(ttl=60 * 5, show_spinner=False)
def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _is_valid_ticker(ticker):
        return pd.DataFrame()

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
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


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_sp500_tickers() -> list[str]:
    """
    Pull S&P 500 tickers from Wikipedia (cached daily).
    Falls back to a small static list if the fetch fails.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        t = tables[0]
        if "Symbol" not in t.columns:
            raise ValueError("No Symbol column")
        tickers = (
            t["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        tickers = [x for x in tickers if x and x.upper() == x.upper()]
        return sorted(list(dict.fromkeys(tickers)))
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH"]


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_most_active_sp500(top_n: int = 20) -> pd.DataFrame:
    """
    Compute "most active" by dollar volume (Close * Volume) on last daily bar.
    Downloads in one batch for speed.
    """
    tickers = load_sp500_tickers()
    tickers = tickers[:505]
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        " ".join(tickers),
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # FIX: cuando solo hay un ticker, yfinance no devuelve MultiIndex
    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()

    last = df.tail(2)
    rows = []
    for t in tickers:
        try:
            close_col = last["Close"]
            if t not in close_col.columns:
                continue
            close = float(close_col[t].iloc[-1])
            prev = float(close_col[t].iloc[-2]) if len(last) >= 2 else float("nan")
            vol = (
                float(last["Volume"][t].iloc[-1])
                if "Volume" in last.columns.get_level_values(0)
                else float("nan")
            )
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


# ---------------------------------------------------------------------------
# CONTEXTO MACRO: correlaciones, beta, fuerza relativa, régimen de mercado
# ---------------------------------------------------------------------------

# Mapa sector ETF → tickers representativos (sin API de pago)
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

# Tickers de referencia macro (gratuitos vía yfinance)
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
    """
    Descarga SPY + referencias macro + el ticker en batch.
    Calcula: correlación con SPY/QQQ, beta vs SPY, fuerza relativa (RS),
    régimen de mercado (SPY sobre/bajo SMA200), VIX nivel.
    """
    refs  = list(_MACRO_REFS.keys())
    batch = [ticker] + refs

    raw = yf.download(
        " ".join(batch),
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        return {}

    # Normalizar a columnas simples si MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"].copy()
    else:
        close_df = raw[["Close"]].copy()
        close_df.columns = [ticker]

    # Retornos diarios
    rets = close_df.pct_change().dropna()

    result: dict = {}

    # ── Beta y correlación con SPY ──
    if ticker in rets.columns and "SPY" in rets.columns:
        common = rets[[ticker, "SPY"]].dropna()
        if len(common) > 20:
            cov    = common.cov()
            var_sp = float(cov.loc["SPY", "SPY"])
            beta   = float(cov.loc[ticker, "SPY"]) / var_sp if var_sp != 0 else float("nan")
            corr   = float(common[ticker].corr(common["SPY"]))
            result["beta_spy"]  = beta
            result["corr_spy"]  = corr

    # ── Correlación con QQQ ──
    if ticker in rets.columns and "QQQ" in rets.columns:
        common_q = rets[[ticker, "QQQ"]].dropna()
        if len(common_q) > 20:
            result["corr_qqq"] = float(common_q[ticker].corr(common_q["QQQ"]))

    # ── Fuerza relativa vs SPY (retorno 1m, 3m, 6m) ──
    if ticker in close_df.columns and "SPY" in close_df.columns:
        for days, label in [(21, "1m"), (63, "3m"), (126, "6m")]:
            if len(close_df) > days:
                tk_ret  = _safe_pct(float(close_df[ticker].iloc[-1]),
                                    float(close_df[ticker].iloc[-(days+1)]))
                spy_ret = _safe_pct(float(close_df["SPY"].iloc[-1]),
                                    float(close_df["SPY"].iloc[-(days+1)]))
                result[f"rs_{label}"] = tk_ret - spy_ret   # + = outperform

    # ── Régimen de mercado: SPY vs SMA200 ──
    if "SPY" in close_df.columns:
        spy_s = close_df["SPY"].dropna()
        if len(spy_s) >= 200:
            sma200 = float(spy_s.rolling(200).mean().iloc[-1])
            spy_last = float(spy_s.iloc[-1])
            result["spy_vs_sma200"] = _safe_pct(spy_last, sma200)
            result["market_regime"] = "Bull" if spy_last > sma200 else "Bear"
        # SPY cambio 1d
        if len(spy_s) >= 2:
            result["spy_1d"] = _safe_pct(float(spy_s.iloc[-1]), float(spy_s.iloc[-2]))

    # ── VIX nivel ──
    if "VIX" in close_df.columns:
        vix_s = close_df["VIX"].dropna()
        if not vix_s.empty:
            vix_v = float(vix_s.iloc[-1])
            result["vix"] = vix_v
            result["vix_regime"] = (
                "Pánico" if vix_v > 30 else
                "Elevado" if vix_v > 20 else
                "Normal"
            )

    # ── Snapshot de referencias ──
    snapshots = {}
    for sym, name in _MACRO_REFS.items():
        if sym in close_df.columns:
            s = close_df[sym].dropna()
            if len(s) >= 2:
                chg = _safe_pct(float(s.iloc[-1]), float(s.iloc[-2]))
                snapshots[sym] = {"name": name, "price": float(s.iloc[-1]), "chg1d": chg}
    result["snapshots"] = snapshots

    return result


@st.cache_data(ttl=60 * 15, show_spinner=False)
def guess_sector_etf(ticker: str) -> tuple[str, str]:
    """
    Intenta determinar el sector ETF del ticker vía yfinance info.
    Retorna (etf_symbol, sector_name).
    """
    _SECTOR_TO_ETF = {
        "Technology":             "XLK",
        "Financial Services":     "XLF",
        "Healthcare":             "XLV",
        "Energy":                 "XLE",
        "Consumer Cyclical":      "XLY",
        "Consumer Defensive":     "XLP",
        "Industrials":            "XLI",
        "Basic Materials":        "XLB",
        "Real Estate":            "XLRE",
        "Utilities":              "XLU",
        "Communication Services": "XLC",
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
    """
    Compara retorno del ticker vs su sector ETF.
    """
    if not sector_etf:
        return {}
    raw = yf.download(
        f"{ticker} {sector_etf}",
        period=period, interval="1d",
        progress=False, auto_adjust=True,
        group_by="column", threads=True,
    )
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return {}
    close = raw["Close"].dropna()
    if ticker not in close.columns or sector_etf not in close.columns:
        return {}
    tk_r   = _safe_pct(float(close[ticker].iloc[-1]),    float(close[ticker].iloc[0]))
    sect_r = _safe_pct(float(close[sector_etf].iloc[-1]), float(close[sector_etf].iloc[0]))
    return {
        "ticker_return":  tk_r,
        "sector_return":  sect_r,
        "rs_vs_sector":   tk_r - sect_r,
    }


# ---------------------------------------------------------------------------
# SISTEMA DE ALERTAS
# ---------------------------------------------------------------------------

# Condiciones disponibles
_ALERT_CONDITIONS = {
    "RSI < umbral (sobreventa)":         lambda last, v: _safe_val(last, "RSI14") < v,
    "RSI > umbral (sobrecompra)":        lambda last, v: _safe_val(last, "RSI14") > v,
    "Precio cruza EMA50 al alza":        lambda last, v: (
        _safe_val(last, "Close") > _safe_val(last, "EMA50") and
        _safe_val(last, "EMA50") > 0
    ),
    "Precio cruza EMA200 al alza":       lambda last, v: (
        _safe_val(last, "Close") > _safe_val(last, "EMA200") and
        _safe_val(last, "EMA200") > 0
    ),
    "MACD histograma positivo":          lambda last, v: _safe_val(last, "MACD_HIST") > 0,
    "ADX > umbral (tendencia fuerte)":   lambda last, v: _safe_val(last, "ADX14") > v,
    "Volumen relativo > umbral":         lambda last, v: _safe_val(last, "REL_VOL") > v,
    "Score técnico > umbral":            lambda last, v: False,   # evaluado externamente
    "Score técnico < umbral":            lambda last, v: False,   # evaluado externamente
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
    id:         str
    ticker:     str
    condition:  str
    threshold:  float
    active:     bool = True
    triggered:  bool = False
    trigger_ts: str  = ""
    note:       str  = ""


def _init_alerts() -> None:
    if "qsp_alerts" not in st.session_state:
        st.session_state["qsp_alerts"] = []
    if "qsp_alert_log" not in st.session_state:
        st.session_state["qsp_alert_log"] = []   # historial de disparos


def evaluate_alerts(dfi: pd.DataFrame, ticker: str, score: float) -> list[Alert]:
    """
    Evalúa todas las alertas activas del ticker actual.
    Actualiza session_state y retorna lista de alertas disparadas.
    """
    _init_alerts()
    alerts: list[Alert] = st.session_state["qsp_alerts"]
    log:    list[dict]  = st.session_state["qsp_alert_log"]
    fired:  list[Alert] = []

    last = dfi.iloc[-1]

    for a in alerts:
        if not a.active or a.ticker.upper() != ticker.upper():
            continue

        fn = _ALERT_CONDITIONS.get(a.condition)
        triggered = False

        if fn is not None:
            # Condiciones de score evaluadas aparte
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
            a.triggered  = True
            a.trigger_ts = datetime.now().strftime("%d/%m %H:%M")
            log.append({
                "ts":        a.trigger_ts,
                "ticker":    a.ticker,
                "condición": a.condition,
                "umbral":    a.threshold,
                "nota":      a.note,
            })
            fired.append(a)
        elif not triggered:
            a.triggered = False   # reset para detectar próximo cruce

    return fired

@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_news(ticker: str, max_items: int = 15) -> list[dict]:
    """
    Descarga noticias del RSS de Yahoo Finance para el ticker dado.
    Retorna lista de dicts con title, link, published, source.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(ticker)}&region=US&lang=en-US"
    items: list[dict] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        ns = {"media": "http://search.yahoo.com/mrss/"}
        channel = root.find("channel")
        if channel is None:
            return items
        for item in channel.findall("item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            src_el = item.find("source")
            source = src_el.text.strip() if src_el is not None and src_el.text else "Yahoo Finance"
            # Parsear fecha
            try:
                dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
                ago = _time_ago(dt)
            except Exception:
                ago = pub[:16] if pub else "—"
            if title and link:
                items.append({"title": title, "link": link, "ago": ago, "source": source})
    except Exception:
        pass
    return items


def _time_ago(dt: datetime) -> str:
    """Convierte un datetime UTC a texto relativo (ej. 'hace 3h')."""
    now = datetime.now(tz=timezone.utc)
    diff = now - dt
    secs = int(diff.total_seconds())
    if secs < 60:
        return "hace moments"
    if secs < 3600:
        return f"hace {secs // 60}min"
    if secs < 86400:
        return f"hace {secs // 3600}h"
    return f"hace {secs // 86400}d"


# ---------------------------------------------------------------------------
# SENTIMIENTO: scoring simple de titulares (léxico de palabras clave)
# ---------------------------------------------------------------------------

_BULL_WORDS = {
    "surge", "surges", "surging", "rally", "rallies", "rallying", "gain", "gains",
    "jump", "jumps", "beat", "beats", "record", "upgrade", "upgraded", "buy",
    "outperform", "strong", "growth", "profit", "revenue", "bullish", "breakout",
    "upside", "positive", "raises", "raise", "exceeds", "exceed", "higher",
    "soars", "soar", "boom", "momentum", "upbeat", "optimistic",
}
_BEAR_WORDS = {
    "fall", "falls", "falling", "drop", "drops", "slump", "slumps", "decline",
    "declines", "miss", "misses", "downgrade", "downgraded", "sell", "underperform",
    "weak", "loss", "losses", "bearish", "breakdown", "downside", "negative",
    "cuts", "cut", "below", "lower", "plunge", "plunges", "crash", "warning",
    "concern", "risk", "trouble", "disappoints", "disappointing",
}


def score_headline(title: str) -> float:
    """Retorna score de sentimiento entre -1 (bearish) y +1 (bullish)."""
    words = set(re.findall(r"[a-z]+", title.lower()))
    bull = len(words & _BULL_WORDS)
    bear = len(words & _BEAR_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def aggregate_sentiment(news: list[dict]) -> dict:
    """Calcula sentimiento agregado sobre la lista de noticias."""
    if not news:
        return {"score": 0.0, "label": "Neutral", "color": "#8B949E", "bull": 0, "bear": 0, "neutral": 0}
    scores = [score_headline(n["title"]) for n in news]
    for i, n in enumerate(news):
        news[i]["sentiment"] = scores[i]
    avg = float(np.mean(scores))
    bull    = sum(1 for s in scores if s > 0.1)
    bear    = sum(1 for s in scores if s < -0.1)
    neutral = len(scores) - bull - bear
    if avg > 0.15:
        label, color = "Bullish", "#00D18F"
    elif avg < -0.15:
        label, color = "Bearish", "#FF4B4B"
    else:
        label, color = "Neutral", "#8B949E"
    return {"score": avg, "label": label, "color": color, "bull": bull, "bear": bear, "neutral": neutral}


# ---------------------------------------------------------------------------
# SCREENER: filtros multi-criterio sobre el universo S&P 500
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 15, show_spinner=False)
def run_screener(
    min_price: float,
    max_price: float,
    min_chg:   float,
    max_chg:   float,
    min_vol_m: float,        # volumen mínimo en millones de $
    rsi_lo:    float,
    rsi_hi:    float,
    adx_min:   float,
    rec_filter: str,         # "Todas" | "COMPRA*" | "VENTA*" | "NEUTRAL"
    top_n:     int = 50,
) -> pd.DataFrame:
    """
    Descarga datos del S&P 500 y aplica filtros técnicos/fundamentales.
    Retorna DataFrame con los mejores candidatos según los criterios.
    """
    tickers = load_sp500_tickers()[:505]
    if not tickers:
        return pd.DataFrame()

    # Descarga batch de 5 días (precio + volumen)
    raw = yf.download(
        " ".join(tickers),
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()

    last2 = raw.tail(2)
    rows: list[dict] = []

    for t in tickers:
        try:
            close_s = last2["Close"]
            if t not in close_s.columns:
                continue
            price = float(close_s[t].iloc[-1])
            prev  = float(close_s[t].iloc[-2]) if len(last2) >= 2 else float("nan")
            vol   = float(last2["Volume"][t].iloc[-1]) if "Volume" in last2.columns.get_level_values(0) else float("nan")

            if math.isnan(price) or math.isnan(vol):
                continue

            chg    = _safe_pct(price, prev) if not math.isnan(prev) else float("nan")
            vol_m  = price * vol / 1e6   # $ volumen en millones

            # Filtros rápidos sin indicadores (evitar descargar 1y de cada ticker)
            if not (min_price <= price <= max_price):
                continue
            if not math.isnan(chg) and not (min_chg <= chg <= max_chg):
                continue
            if vol_m < min_vol_m:
                continue

            rows.append({"Ticker": t, "Precio": price, "Cambio %": chg, "$ Vol (M)": vol_m})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    base = pd.DataFrame(rows)

    # Para los filtros técnicos (RSI, ADX, recomendación) necesitamos datos históricos.
    # Lo hacemos en paralelo solo para los candidatos que pasaron los filtros de precio/vol.
    candidates = base["Ticker"].tolist()[:120]   # cap para evitar sobrecarga

    def _enrich(t: str) -> dict | None:
        try:
            d = load_ohlcv(t, period="3mo", interval="1d")
            if d.empty or len(d) < 30:
                return None
            di = compute_indicators(d)
            r, _, _, _, ax = recommend(di)
            ll = di.iloc[-1]
            rsi = ll.get("RSI14")
            rsi_v = float(rsi) if rsi is not None and not pd.isna(rsi) else float("nan")
            adx_v = float(ax) if ax else 0.0
            return {"Ticker": t, "RSI": rsi_v, "ADX": adx_v, "Rec": r.label, "Score": r.score}
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        enriched_raw = list(ex.map(_enrich, candidates))

    enriched = {e["Ticker"]: e for e in enriched_raw if e is not None}

    out_rows = []
    for _, row in base[base["Ticker"].isin(candidates)].iterrows():
        t = row["Ticker"]
        e = enriched.get(t)
        if e is None:
            continue
        rsi_v = e["RSI"]
        adx_v = e["ADX"]
        rec_l = e["Rec"]
        score = e["Score"]

        # Aplicar filtros técnicos
        if not math.isnan(rsi_v) and not (rsi_lo <= rsi_v <= rsi_hi):
            continue
        if adx_v < adx_min:
            continue
        if rec_filter != "Todas":
            if rec_filter == "COMPRA" and not rec_l.startswith("COMPRA"):
                continue
            elif rec_filter == "VENTA" and not rec_l.startswith("VENTA"):
                continue
            elif rec_filter == "NEUTRAL" and rec_l != "NEUTRAL":
                continue

        out_rows.append({
            "Ticker":      t,
            "Precio":      row["Precio"],
            "Cambio %":    row["Cambio %"],
            "$ Vol (M)":   row["$ Vol (M)"],
            "RSI":         rsi_v,
            "ADX":         adx_v,
            "Rec":         rec_l,
            "Score":       score,
        })

    if not out_rows:
        return pd.DataFrame()

    result = pd.DataFrame(out_rows).sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Nombres de columnas exactos que devuelve pandas_ta (estables entre versiones)
# ---------------------------------------------------------------------------
_MACD_COL   = "MACD_12_26_9"
_MACDS_COL  = "MACDs_12_26_9"
_MACDH_COL  = "MACDh_12_26_9"
_ADX_COL    = "ADX_14"
_DMP_COL    = "DMP_14"
_DMN_COL    = "DMN_14"
_BBL_COL    = "BBL_20_2.0"
_BBM_COL    = "BBM_20_2.0"
_BBU_COL    = "BBU_20_2.0"
_BBP_COL    = "BBP_20_2.0"
_BBW_COL    = "BBW_20_2.0"
_SRSI_K_COL = "STOCHRSIk_14_14_3_3"
_SRSI_D_COL = "STOCHRSId_14_14_3_3"
_CCI_COL    = "CCI_14_0.015"
_WILLR_COL  = "WILLR_14"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    close = out["Close"]
    high  = out["High"]
    low   = out["Low"]
    vol   = out["Volume"] if "Volume" in out.columns else pd.Series(index=out.index, dtype="float64")

    for n in (10, 20, 50, 100, 200):
        out[f"SMA{n}"] = ta.sma(close, length=n)
        out[f"EMA{n}"] = ta.ema(close, length=n)

    out["RSI14"] = ta.rsi(close, length=14)

    # FIX: acceder por nombre de columna, no por posición
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["MACD"]        = macd.get(_MACD_COL,  macd.iloc[:, 0])
        out["MACD_SIGNAL"] = macd.get(_MACDS_COL, macd.iloc[:, 1])
        out["MACD_HIST"]   = macd.get(_MACDH_COL, macd.iloc[:, 2])

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and not adx.empty:
        out["ADX14"] = adx.get(_ADX_COL, adx.iloc[:, 0])
        out["DMP14"] = adx.get(_DMP_COL, adx.iloc[:, 1])
        out["DMN14"] = adx.get(_DMN_COL, adx.iloc[:, 2])

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        out["BBL"] = bb.get(_BBL_COL, bb.iloc[:, 0])
        out["BBM"] = bb.get(_BBM_COL, bb.iloc[:, 1])
        out["BBU"] = bb.get(_BBU_COL, bb.iloc[:, 2])
        out["BBP"] = bb.get(_BBP_COL, bb.iloc[:, 3])
        out["BBW"] = bb.get(_BBW_COL, bb.iloc[:, 4])

    out["ATR14"] = ta.atr(high, low, close, length=14)

    stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None and not stochrsi.empty:
        out["STOCHRSI_K"] = stochrsi.get(_SRSI_K_COL, stochrsi.iloc[:, 0])
        out["STOCHRSI_D"] = stochrsi.get(_SRSI_D_COL, stochrsi.iloc[:, 1])

    sup = ta.supertrend(high, low, close, length=10, multiplier=3.0)
    if sup is not None and not sup.empty:
        for c in sup.columns:
            out[c] = sup[c]

    ichi = ta.ichimoku(high, low, close)
    if isinstance(ichi, tuple) and len(ichi) >= 1:
        ichi_df = ichi[0]
        if ichi_df is not None and not ichi_df.empty:
            for c in ichi_df.columns:
                out[c] = ichi_df[c]

    out["VOL_SMA20"] = ta.sma(vol, length=20) if not vol.empty else np.nan
    out["REL_VOL"]   = out["Volume"] / out["VOL_SMA20"] if "Volume" in out.columns else np.nan

    # --- Nuevos indicadores para estrategia mejorada ---

    # CCI (Commodity Channel Index): extremos ±100 indican condiciones de entrada
    cci = ta.cci(high, low, close, length=14)
    if cci is not None:
        out["CCI14"] = cci

    # Williams %R: -80 a -100 = sobreventa, 0 a -20 = sobrecompra
    willr = ta.willr(high, low, close, length=14)
    if willr is not None:
        out["WILLR14"] = willr

    # OBV (On-Balance Volume): tendencia del volumen confirma precio
    obv = ta.obv(close, vol) if not vol.empty else None
    if obv is not None:
        out["OBV"] = obv
        out["OBV_EMA20"] = ta.ema(obv, length=20)

    # ROC (Rate of Change 10): momentum de precio puro
    roc = ta.roc(close, length=10)
    if roc is not None:
        out["ROC10"] = roc

    # EMA corta para detección de cruces rápidos
    out["EMA9"]  = ta.ema(close, length=9)
    out["EMA21"] = ta.ema(close, length=21)

    # Divergencia RSI simplificada: diferencia de pendiente RSI vs precio (últimas 5 velas)
    # Se calcula en recommend() sobre las últimas N filas, no aquí (requiere ventana).

    # FIX: ffill solo en columnas de indicadores derivados, no en OHLCV ni en señales
    # con lag natural (Ichimoku). Las columnas de precio se dejan intactas.
    indicator_cols = [c for c in out.columns if c not in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    out[indicator_cols] = out[indicator_cols].ffill()

    return out


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
    details["Price vs EMA50"]  = above("Close", "EMA50")
    details["EMA9 vs EMA21"]   = gt("EMA9", "EMA21")   # cruce rápido de corto plazo

    # Supertrend direction (SUPERTd_10_3.0): 1 = alcista, -1 = bajista
    st_dir = 0.0
    for k in last.index:
        if str(k).startswith("SUPERTd_"):
            v = last.get(k)
            st_dir = 0.0 if pd.isna(v) else (1.0 if float(v) > 0 else -1.0)
            break
    details["Supertrend"] = st_dir

    # Ichimoku: close por encima de la nube = bullish, por debajo = bearish
    ichi_score = 0.0
    span_a = last.get("ISA_9") if "ISA_9" in last.index else last.get("ICHISA_9")
    span_b = last.get("ISB_26") if "ISB_26" in last.index else last.get("ICHISB_26")
    if span_a is not None and span_b is not None and not (pd.isna(span_a) or pd.isna(span_b)):
        top = max(float(span_a), float(span_b))
        bot = min(float(span_a), float(span_b))
        if not pd.isna(last.get("Close")):
            price = float(last["Close"])
            if price > top:
                ichi_score = 1.0
            elif price < bot:
                ichi_score = -1.0
    details["Ichimoku Cloud"] = ichi_score

    # OBV vs su EMA20: confirma tendencia con volumen
    obv_score = 0.0
    obv     = last.get("OBV")
    obv_ema = last.get("OBV_EMA20")
    if obv is not None and obv_ema is not None and not (pd.isna(obv) or pd.isna(obv_ema)):
        obv_score = 1.0 if float(obv) > float(obv_ema) else -1.0
    details["OBV vs EMA20"] = obv_score

    vals = np.array(list(details.values()), dtype="float64")
    return float(np.nanmean(vals)) if vals.size else 0.0, details


def _signal_momentum(last: pd.Series) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}

    rsi = last.get("RSI14")
    if rsi is not None and not pd.isna(rsi):
        r = float(rsi)
        if r >= 70:
            details["RSI14"] = _clamp(1.0 - (r - 70.0) / 15.0, -1.0, 1.0)
        elif r <= 30:
            details["RSI14"] = _clamp((50.0 - r) / 20.0 * -1.0, -1.0, 1.0)
        else:
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
        if kk >= 80:
            details["StochRSI"] = _clamp(1.0 - (kk - 80.0) / 10.0, -1.0, 1.0)
        elif kk <= 20:
            details["StochRSI"] = _clamp((kk - 20.0) / 10.0, -1.0, 1.0)
        else:
            details["StochRSI"] = _clamp((kk - 50.0) / 30.0, -1.0, 1.0)
    else:
        details["StochRSI"] = 0.0

    # CCI: > +100 = sobrecompra (bajista corto plazo), < -100 = sobreventa (alcista)
    cci = last.get("CCI14")
    if cci is not None and not pd.isna(cci):
        c = float(cci)
        if c > 200:
            details["CCI14"] = -1.0
        elif c > 100:
            details["CCI14"] = _clamp(1.0 - (c - 100.0) / 100.0, -1.0, 1.0)
        elif c < -200:
            details["CCI14"] = 1.0
        elif c < -100:
            details["CCI14"] = _clamp(-1.0 + (abs(c) - 100.0) / 100.0, -1.0, 1.0)
        else:
            details["CCI14"] = _clamp(c / 100.0, -1.0, 1.0)
    else:
        details["CCI14"] = 0.0

    # Williams %R: -100 a -80 = sobreventa (alcista), 0 a -20 = sobrecompra (bajista)
    willr = last.get("WILLR14")
    if willr is not None and not pd.isna(willr):
        w = float(willr)  # rango -100..0
        # mapear a -1..+1: -100 -> +1 (sobreventa), 0 -> -1 (sobrecompra)
        details["Williams%R"] = _clamp((-w - 50.0) / 50.0, -1.0, 1.0)
    else:
        details["Williams%R"] = 0.0

    # ROC: momentum de precio puro, normalizado
    roc = last.get("ROC10")
    if roc is not None and not pd.isna(roc):
        details["ROC10"] = _clamp(float(roc) / 10.0, -1.0, 1.0)
    else:
        details["ROC10"] = 0.0

    vals = np.array(list(details.values()), dtype="float64")
    return float(np.nanmean(vals)), details


def _signal_volatility(last: pd.Series) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}
    atr   = last.get("ATR14")
    close = last.get("Close")
    atrp  = np.nan
    if (
        atr is not None and close is not None
        and not (pd.isna(atr) or pd.isna(close))
        and float(close) != 0
    ):
        atrp = float(atr) / float(close) * 100.0

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
        details["RelVol"] = _clamp((float(rv) - 1.0) / 1.0, -0.5, 0.8)
    return float(np.nanmean(list(details.values()))), details


# ---------------------------------------------------------------------------
# Detección de divergencias RSI/precio (requiere ventana de barras)
# ---------------------------------------------------------------------------

def detect_divergence(df: pd.DataFrame, window: int = 14) -> dict[str, str]:
    """
    Detecta divergencias alcistas/bajistas entre precio y RSI en las últimas `window` velas.
    Retorna un dict con claves 'tipo' y 'descripcion'.
    """
    result = {"tipo": "Ninguna", "descripcion": "Sin divergencia detectada"}
    if len(df) < window + 2 or "RSI14" not in df.columns:
        return result

    sub = df.tail(window).dropna(subset=["Close", "RSI14"])
    if len(sub) < 4:
        return result

    prices = sub["Close"].values
    rsis   = sub["RSI14"].values

    # Buscar mínimos locales (divergencia alcista): precio hace mínimo más bajo, RSI hace mínimo más alto
    price_lo1, price_lo2 = prices[0], prices[-1]
    rsi_lo1,   rsi_lo2   = rsis[0],   rsis[-1]

    # Buscar máximos locales (divergencia bajista): precio hace máximo más alto, RSI hace máximo más bajo
    price_hi1, price_hi2 = prices[0], prices[-1]
    rsi_hi1,   rsi_hi2   = rsis[0],   rsis[-1]

    # Umbral mínimo de diferencia para filtrar ruido
    price_thr = abs(price_lo1) * 0.005  # 0.5%
    rsi_thr   = 2.0                      # 2 puntos RSI

    if (price_lo2 < price_lo1 - price_thr) and (rsi_lo2 > rsi_lo1 + rsi_thr):
        result = {
            "tipo": "Alcista",
            "descripcion": f"Divergencia alcista: precio hizo mínimo más bajo ({price_lo2:.2f} < {price_lo1:.2f}) pero RSI subió ({rsi_lo2:.1f} > {rsi_lo1:.1f}). Señal de agotamiento bajista.",
        }
    elif (price_hi2 > price_hi1 + price_thr) and (rsi_hi2 < rsi_hi1 - rsi_thr):
        result = {
            "tipo": "Bajista",
            "descripcion": f"Divergencia bajista: precio hizo máximo más alto ({price_hi2:.2f} > {price_hi1:.2f}) pero RSI cayó ({rsi_hi2:.1f} < {rsi_hi1:.1f}). Señal de agotamiento alcista.",
        }

    return result


# ---------------------------------------------------------------------------
# Gestión de riesgo: niveles de entrada, Stop Loss y Take Profit basados en ATR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskLevels:
    entry:      float
    stop_loss:  float
    tp1:        float
    tp2:        float
    tp3:        float
    risk_reward: float  # ratio riesgo/beneficio hacia TP1
    position_size_pct: float  # % de capital sugerido (riesgo 1%)
    atr:        float
    atr_pct:    float


def compute_risk_levels(df: pd.DataFrame, risk_per_trade_pct: float = 1.0) -> RiskLevels | None:
    """
    Calcula niveles de entrada, SL y TPs basados en ATR.
    - SL: 1.5x ATR bajo el precio de entrada (largo) o sobre (corto)
    - TP1: 1.5x ATR, TP2: 3x ATR, TP3: 5x ATR
    - Position size: riesgo_por_operacion / distancia_SL
    """
    if df.empty or "ATR14" not in df.columns or "Close" not in df.columns:
        return None

    last  = df.iloc[-1]
    entry = float(last["Close"])
    atr   = last.get("ATR14")

    if atr is None or pd.isna(atr) or entry == 0:
        return None

    atr_v   = float(atr)
    atr_pct = atr_v / entry * 100.0

    sl_dist = 1.5 * atr_v
    stop_loss = entry - sl_dist

    tp1 = entry + 1.5 * atr_v
    tp2 = entry + 3.0 * atr_v
    tp3 = entry + 5.0 * atr_v

    rr = (tp1 - entry) / sl_dist if sl_dist > 0 else 0.0

    # Position size: cuántas unidades comprar para arriesgar risk_per_trade_pct% del capital
    # Expresado como % del capital total: risk% / (SL_dist / entry)
    sl_pct = sl_dist / entry * 100.0
    position_size_pct = risk_per_trade_pct / sl_pct * 100.0 if sl_pct > 0 else 0.0
    position_size_pct = _clamp(position_size_pct, 0.0, 100.0)

    return RiskLevels(
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        risk_reward=rr,
        position_size_pct=position_size_pct,
        atr=atr_v,
        atr_pct=atr_pct,
    )


# FIX: type hint corregido para reflejar los 5 valores que realmente se retornan
def recommend(
    df: pd.DataFrame,
) -> tuple[Recommendation, pd.DataFrame, pd.DataFrame, bool, float]:
    last = df.iloc[-1]

    adx   = last.get("ADX14")
    adx_v = float(adx) if adx is not None and not pd.isna(adx) else 0.0
    trending = adx_v >= 25.0

    trend_s, trend_d = _signal_trend(last)
    mom_s,   mom_d   = _signal_momentum(last)
    vol_s,   vol_d   = _signal_volatility(last)
    v_s,     v_d     = _signal_volume(last)

    if trending:
        w = {"Trend": 0.45, "Momentum": 0.30, "Volatility": 0.10, "Volume": 0.15}
    else:
        w = {"Trend": 0.30, "Momentum": 0.35, "Volatility": 0.20, "Volume": 0.15}

    comp = {
        "Trend":      float(trend_s),
        "Momentum":   float(mom_s),
        "Volatility": float(vol_s),
        "Volume":     float(v_s),
    }
    raw   = sum(w[k] * comp[k] for k in comp)
    score = _clamp(raw * 100.0, -100.0, 100.0)

    alignment  = float(np.mean([abs(trend_s), abs(mom_s), abs(vol_s), abs(v_s)]))
    regime     = _clamp((adx_v - 10.0) / 25.0, 0.0, 1.0)
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

    rows = [
        {
            "Grupo": group,
            "Peso": weight,
            "Score (-1..+1)": comp[group],
            "Contribución": weight * comp[group],
        }
        for group, weight in w.items()
    ]
    expl = pd.DataFrame(rows)

    detail_rows = [
        {"Indicador": name, "Señal (-1..+1)": float(val)}
        for name, val in {**trend_d, **mom_d, **vol_d, **v_d}.items()
    ]
    details = pd.DataFrame(detail_rows).sort_values("Indicador")

    return rec, expl, details, trending, adx_v


# ---------------------------------------------------------------------------
# SCORE FINAL INTEGRADO: técnico + macro + sentimiento + divergencia
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinalVerdict:
    label:        str    # "COMPRA FUERTE" etc.
    color:        str
    final_score:  float  # -100..+100
    tech_score:   float
    macro_adj:    float
    sent_adj:     float
    div_adj:      float
    confidence:   int
    summary:      str    # texto explicativo de 1 línea


def compute_final_score(
    rec:        Recommendation,
    macro_ctx:  dict,
    sentiment:  dict,
    divergence: dict,
) -> FinalVerdict:
    """
    Combina el score técnico con ajustes de macro, sentimiento y divergencia
    para producir una recomendación final más robusta.

    Pesos:
      - Score técnico:   70 %
      - Ajuste macro:    15 %
      - Sentimiento:      8 %
      - Divergencia:      7 %
    """
    tech = rec.score  # ya en -100..+100

    # ── Ajuste macro (−15 .. +15) ──
    macro_adj = 0.0
    mkt = macro_ctx.get("market_regime", "")
    vix_v = macro_ctx.get("vix", float("nan"))
    rs_6m = macro_ctx.get("rs_6m", float("nan"))
    beta  = macro_ctx.get("beta_spy", float("nan"))

    if mkt == "Bull":
        macro_adj += 5.0
    elif mkt == "Bear":
        macro_adj -= 8.0

    if not math.isnan(vix_v):
        if vix_v > 35:
            macro_adj -= 7.0
        elif vix_v > 25:
            macro_adj -= 3.0
        elif vix_v < 15:
            macro_adj += 2.0

    if not math.isnan(rs_6m):
        macro_adj += _clamp(rs_6m * 0.3, -5.0, 5.0)

    macro_adj = _clamp(macro_adj, -15.0, 15.0)

    # ── Ajuste sentimiento (−8 .. +8) ──
    sent_score = sentiment.get("score", 0.0) if sentiment else 0.0
    sent_adj   = _clamp(sent_score * 8.0, -8.0, 8.0)

    # ── Ajuste divergencia (−7 .. +7) ──
    div_adj = 0.0
    div_tipo = divergence.get("tipo", "Ninguna")
    if div_tipo == "Alcista":
        div_adj = 7.0
    elif div_tipo == "Bajista":
        div_adj = -7.0

    # ── Score final ponderado ──
    final = _clamp(
        tech * 0.70 + macro_adj * 1.0 + sent_adj * 1.0 + div_adj * 1.0,
        -100.0, 100.0,
    )

    # ── Confianza compuesta ──
    # Base: confianza técnica. Bonus por alineación de señales.
    signals_aligned = sum([
        (macro_adj > 0 and final > 0) or (macro_adj < 0 and final < 0),
        (sent_adj  > 0 and final > 0) or (sent_adj  < 0 and final < 0),
        (div_adj   > 0 and final > 0) or (div_adj   < 0 and final < 0),
    ])
    conf_bonus = signals_aligned * 5
    confidence = int(_clamp(rec.confidence + conf_bonus, 0, 100))

    # ── Label y color ──
    if final >= 60:
        label, color = "COMPRA FUERTE", "#00D18F"
    elif final >= 20:
        label, color = "COMPRA",        "#2F81F7"
    elif final <= -60:
        label, color = "VENTA FUERTE",  "#FF4B4B"
    elif final <= -20:
        label, color = "VENTA",         "#FFA657"
    else:
        label, color = "NEUTRAL",       "#8B949E"

    # ── Resumen en lenguaje natural ──
    parts = [f"Score técnico {tech:+.0f}"]
    if macro_adj != 0:
        parts.append(f"macro {macro_adj:+.0f} ({mkt or '—'}, VIX {vix_v:.0f})" if not math.isnan(vix_v) else f"macro {macro_adj:+.0f}")
    if abs(sent_adj) > 0.5:
        parts.append(f"sentimiento {sent_adj:+.0f} ({sentiment.get('label','—')})")
    if div_adj != 0:
        parts.append(f"divergencia {div_adj:+.0f} ({div_tipo})")
    summary = " | ".join(parts)

    return FinalVerdict(
        label=label, color=color,
        final_score=final, tech_score=tech,
        macro_adj=macro_adj, sent_adj=sent_adj, div_adj=div_adj,
        confidence=confidence, summary=summary,
    )
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


def _detect_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Detecta patrones de velas simples en las últimas barras.
    Retorna lista de dicts {bar_idx, name, color, direction}.
    """
    patterns: list[dict] = []
    if len(df) < 3:
        return patterns

    tail = df.tail(30).copy()
    tail = tail.dropna(subset=["Open", "High", "Low", "Close"])

    for i in range(2, len(tail)):
        o, h, l, c = (float(tail["Open"].iloc[i]),  float(tail["High"].iloc[i]),
                      float(tail["Low"].iloc[i]),   float(tail["Close"].iloc[i]))
        po, ph, pl, pc = (float(tail["Open"].iloc[i-1]), float(tail["High"].iloc[i-1]),
                          float(tail["Low"].iloc[i-1]),  float(tail["Close"].iloc[i-1]))
        body   = abs(c - o)
        p_body = abs(pc - po)
        rng    = h - l if h != l else 1e-9
        ts     = tail.index[i]

        # Doji: cuerpo muy pequeño
        if body / rng < 0.1:
            patterns.append({"ts": ts, "name": "Doji", "color": "#FFA657", "y": h})

        # Hammer: sombra inferior larga (alcista en tendencia bajista)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        if lower_shadow > 2 * body and upper_shadow < body * 0.5 and c > o:
            patterns.append({"ts": ts, "name": "Hammer", "color": "#00D18F", "y": l * 0.998})

        # Shooting Star: sombra superior larga (bajista en tendencia alcista)
        if upper_shadow > 2 * body and lower_shadow < body * 0.5 and c < o:
            patterns.append({"ts": ts, "name": "Shoot★", "color": "#FF4B4B", "y": h * 1.002})

        # Engulfing alcista
        if pc > po and c > o and c > po and o < pc and p_body > 0 and body > p_body:
            patterns.append({"ts": ts, "name": "Bull Engulf", "color": "#00D18F", "y": l * 0.997})

        # Engulfing bajista
        if pc < po and c < o and c < po and o > pc and p_body > 0 and body > p_body:
            patterns.append({"ts": ts, "name": "Bear Engulf", "color": "#FF4B4B", "y": h * 1.003})

    return patterns


def build_chart(
    df: pd.DataFrame,
    overlays: Iterable[str],
    show_rsi: bool = True,
    show_macd: bool = False,
    show_patterns: bool = True,
    show_sl_tp: bool = False,
    risk: "RiskLevels | None" = None,
) -> go.Figure:
    from plotly.subplots import make_subplots

    overlays = list(overlays)
    view = df.tail(220).copy()
    x    = view.index

    # Determinar número de subplots
    n_rows = 1
    row_heights = [0.70]
    subplot_titles = [""]
    if show_rsi:
        n_rows += 1; row_heights.append(0.15); subplot_titles.append("RSI 14")
    if show_macd:
        n_rows += 1; row_heights.append(0.15); subplot_titles.append("MACD")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    # ── Candlestick principal ──
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=view["Open"], high=view["High"],
            low=view["Low"],   close=view["Close"],
            name="Precio",
            increasing_line_color="#00D18F",
            decreasing_line_color="#FF4B4B",
        ),
        row=1, col=1,
    )

    # ── Volumen ──
    if "Volumen" in overlays and "Volume" in view.columns:
        colors = [
            "#00D18F" if float(view["Close"].iloc[i]) >= float(view["Open"].iloc[i]) else "#FF4B4B"
            for i in range(len(view))
        ]
        fig.add_trace(
            go.Bar(x=x, y=view["Volume"], name="Volumen",
                   marker_color=colors, opacity=0.4, yaxis="y2"),
            row=1, col=1,
        )

    # ── EMAs ──
    if "EMA 50/200" in overlays:
        for n, col, w in [(9, "#8B949E", 1), (21, "#56D364", 1), (50, "#2F81F7", 1.5), (200, "#FFA657", 2)]:
            k = f"EMA{n}"
            if k in view.columns:
                fig.add_trace(go.Scatter(x=x, y=view[k], name=f"EMA {n}",
                                         line=dict(color=col, width=w)), row=1, col=1)

    # ── Bollinger Bands ──
    if "Bandas Bollinger" in overlays and all(k in view.columns for k in ("BBL", "BBM", "BBU")):
        fig.add_trace(go.Scatter(x=x, y=view["BBU"], name="BB Upper",
                                  line=dict(color="rgba(139,148,158,0.5)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=view["BBL"], name="BB Lower",
                                  line=dict(color="rgba(139,148,158,0.5)", width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(139,148,158,0.05)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=view["BBM"], name="BB Mid",
                                  line=dict(color="rgba(139,148,158,0.3)", width=1)), row=1, col=1)

    # ── Supertrend ──
    if "Supertrend" in overlays:
        supert_col = next((c for c in view.columns if c.startswith("SUPERT_") and not c.startswith("SUPERTd_")), None)
        superd_col = next((c for c in view.columns if c.startswith("SUPERTd_")), None)
        if supert_col and superd_col:
            bull_mask = view[superd_col] > 0
            fig.add_trace(go.Scatter(
                x=x[bull_mask], y=view[supert_col][bull_mask],
                mode="lines", name="Supertrend ▲",
                line=dict(color="#00D18F", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x[~bull_mask], y=view[supert_col][~bull_mask],
                mode="lines", name="Supertrend ▼",
                line=dict(color="#FF4B4B", width=1.5)), row=1, col=1)

    # ── Patrones de velas ──
    if show_patterns:
        pats = _detect_patterns(view)
        for p in pats:
            fig.add_annotation(
                x=p["ts"], y=p["y"],
                text=p["name"], showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1,
                arrowcolor=p["color"], font=dict(size=9, color=p["color"]),
                ax=0, ay=-18, row=1, col=1,
            )

    # ── SL/TP lines ──
    if show_sl_tp and risk is not None:
        for level, label, color, dash in [
            (risk.stop_loss, f"SL {format_price(risk.stop_loss)}", "#FF4B4B", "dash"),
            (risk.tp1,       f"TP1 {format_price(risk.tp1)}",      "#00D18F", "dot"),
            (risk.tp2,       f"TP2 {format_price(risk.tp2)}",      "#56D364", "dot"),
            (risk.tp3,       f"TP3 {format_price(risk.tp3)}",      "#2F81F7", "dot"),
            (risk.entry,     f"Entrada {format_price(risk.entry)}", "#FFA657", "solid"),
        ]:
            fig.add_hline(y=level, line_dash=dash, line_color=color,
                          line_width=1.2, annotation_text=label,
                          annotation_font_color=color,
                          annotation_position="right", row=1, col=1)

    # ── RSI subplot ──
    cur_row = 2
    if show_rsi and "RSI14" in view.columns:
        fig.add_trace(go.Scatter(x=x, y=view["RSI14"], name="RSI 14",
                                  line=dict(color="#2F81F7", width=1.5)), row=cur_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#FF4B4B", line_width=0.8, row=cur_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00D18F", line_width=0.8, row=cur_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#8B949E", line_width=0.5, row=cur_row, col=1)
        fig.update_yaxes(range=[0, 100], row=cur_row, col=1)
        cur_row += 1

    # ── MACD subplot ──
    if show_macd and all(k in view.columns for k in ("MACD", "MACD_SIGNAL", "MACD_HIST")):
        hist = view["MACD_HIST"]
        hist_colors = ["#00D18F" if float(v) >= 0 else "#FF4B4B" for v in hist]
        fig.add_trace(go.Bar(x=x, y=hist, name="MACD Hist",
                              marker_color=hist_colors, opacity=0.7), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=x, y=view["MACD"], name="MACD",
                                  line=dict(color="#2F81F7", width=1.2)), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=x, y=view["MACD_SIGNAL"], name="Señal",
                                  line=dict(color="#FFA657", width=1.2)), row=cur_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700 + (120 if show_rsi else 0) + (120 if show_macd else 0),
        margin=dict(l=10, r=80, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=11),
        yaxis=dict(title="Precio", side="right"),
        yaxis2=dict(title="Vol", overlaying="y", side="left",
                    showgrid=False, rangemode="tozero", showticklabels=False),
        dragmode="zoom",
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers para el Radar paralelizado
# ---------------------------------------------------------------------------

def _load_and_recommend(ticker: str, period: str, interval: str) -> dict | None:
    """
    Carga datos y calcula recomendación FINAL integrada (técnico + macro + sentimiento
    + divergencia) para un ticker. Ejecutado en thread pool para el Radar.
    """
    try:
        d0 = load_ohlcv(ticker, period=period, interval=interval)
        if d0.empty:
            return None
        di  = compute_indicators(d0)
        r, _, _, _, ax = recommend(di)
        l0  = di.iloc[-1]
        p   = float(l0["Close"])
        pc  = float(di["Close"].iloc[-2]) if len(di) >= 2 else float("nan")
        ch  = _safe_pct(p, pc) if not math.isnan(pc) else float("nan")

        # Score final integrado (usa caché de macro y noticias)
        macro_lite = load_macro_context(ticker, "6mo")
        news_lite  = load_news(ticker, max_items=10)
        sent_lite  = aggregate_sentiment(news_lite)
        div_lite   = detect_divergence(di, window=14)
        verd       = compute_final_score(r, macro_lite, sent_lite, div_lite)

        rsi_v = _safe_val(l0, "RSI14")
        rv    = _safe_val(l0, "REL_VOL")
        return {
            "Activo":      ticker,
            "Precio":      format_price(p),
            "Cambio %":    f"{ch:+.2f}%" if not math.isnan(ch) else "—",
            "Score Téc.":  f"{r.score:+.0f}",
            "Score Final": f"{verd.final_score:+.0f}",
            "Veredicto":   verd.label,
            "Confianza":   f"{verd.confidence}%",
            "RSI":         f"{rsi_v:.1f}" if rsi_v else "—",
            "ADX":         f"{ax:.1f}"    if ax    else "—",
            "Régimen":     regime_label(ax),
            "Sentimiento": sent_lite.get("label", "—"),
            "Divergencia": div_lite.get("tipo",   "—"),
            "RelVol":      f"{rv:.2f}x"   if rv    else "—",
        }
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.markdown(
        """
<style>
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

  div[data-testid="stMetric"] {
    background: rgba(15,23,34,0.35);
    border: 1px solid rgba(139,148,158,0.18);
    border-radius: 12px;
    padding: 12px 12px 10px 12px;
  }
  div[data-testid="stMetric"] > label { margin-bottom: 4px; }
  div[data-testid="stVerticalBlock"] { gap: 0.75rem; }
  .block-container { padding-top: 1.0rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Terminal")
        ticker_raw = st.text_input("Activo (Ticker)", value="AAPL").strip().upper()

        # FIX: validación del ticker antes de continuar
        ticker_valid = _is_valid_ticker(ticker_raw)
        if not ticker_valid:
            st.error("Ticker inválido. Usa solo letras, números, guiones o puntos (máx. 10 caracteres).")

        ticker = ticker_raw  # se usa más abajo; si no es válido, load_ohlcv retorna vacío

        colp = st.columns(2)
        with colp[0]:
            period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        with colp[1]:
            interval = st.selectbox("Intervalo", ["1d", "1h", "30m", "15m"], index=0)

        st.markdown("---")
        st.markdown("### Gráfico")
        overlays = st.multiselect(
            "Overlays precio",
            ["Volumen", "EMA 50/200", "Bandas Bollinger", "Supertrend"],
            default=["Volumen", "EMA 50/200"],
        )
        col_gc1, col_gc2 = st.columns(2)
        with col_gc1:
            show_rsi  = st.checkbox("RSI panel", value=True)
            show_pats = st.checkbox("Patrones velas", value=True)
        with col_gc2:
            show_macd  = st.checkbox("MACD panel", value=False)
            show_sl_tp = st.checkbox("SL / TP", value=False)

        st.markdown("---")
        st.markdown("### Estrategia")
        risk_per_trade = st.slider(
            "Riesgo por operación (%)",
            min_value=0.25, max_value=5.0, value=1.0, step=0.25,
            help="% del capital total que arriesgas por operación.",
        )
        st.markdown("---")
        st.markdown("### Actualización")
        auto_refresh = st.selectbox(
            "Auto-refresh",
            [0, 1, 5, 10, 15, 30],
            format_func=lambda x: "Manual" if x == 0 else f"Cada {x} min",
            index=0,
        )

        st.markdown("---")
        st.markdown("### Radar")
        radar_raw = st.text_area(
            "Lista (coma)",
            value="AAPL, MSFT, NVDA, TSLA, BTC-USD",
            height=90,
        )
        radar_items = [x.strip().upper() for x in radar_raw.split(",") if x.strip()]

    # Header
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    hL, hR = st.columns([0.78, 0.22], vertical_alignment="center")
    with hL:
        st.markdown(f"<div class='qsp-title'>{ticker} <span class='qsp-sub'>• {APP_TITLE}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='qsp-sub'>Actualizado: {now} • Fuente: Yahoo Finance</div>", unsafe_allow_html=True)
    with hR:
        st.empty()

    if not ticker_valid:
        st.warning("Ingresa un ticker válido en la barra lateral para continuar.")
        return

    df = load_ohlcv(ticker, period=period, interval=interval)
    if df.empty:
        st.error("No pude descargar datos. Revisa el ticker o tu conexión.")
        return

    dfi  = compute_indicators(df)
    last = dfi.iloc[-1]
    prev_close = dfi["Close"].iloc[-2] if len(dfi) >= 2 else float("nan")
    chg = _safe_pct(float(last["Close"]), float(prev_close)) if not pd.isna(prev_close) else float("nan")

    rec, expl, details, trending, adx_v = recommend(dfi)

    # Divergencia y niveles de riesgo (calculados una sola vez)
    divergence  = detect_divergence(dfi, window=20)
    risk_levels = compute_risk_levels(dfi, risk_per_trade_pct=risk_per_trade)

    # ── Contexto macro (en paralelo para no bloquear) ──
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _ex:
        _f_macro  = _ex.submit(load_macro_context, ticker, "1y")
        _f_sector = _ex.submit(guess_sector_etf, ticker)
    macro_ctx   = _f_macro.result()
    sector_etf, sector_name = _f_sector.result()
    sector_rs = load_sector_rs(ticker, sector_etf) if sector_etf else {}

    # ── Evaluar alertas activas ──
    _init_alerts()
    fired_alerts = evaluate_alerts(dfi, ticker, rec.score)

    # ── Sentimiento (necesario para score final) ──
    news_for_sent = load_news(ticker, max_items=20)
    sentiment     = aggregate_sentiment(news_for_sent)

    # ── Score final integrado ──
    verdict = compute_final_score(rec, macro_ctx, sentiment, divergence)

    # ── Auto-refresh configurable ──
    if auto_refresh > 0:
        st.markdown(
            f'<meta http-equiv="refresh" content="{auto_refresh * 60}">',
            unsafe_allow_html=True,
        )

    # KPI row — ahora muestra el veredicto final integrado
    k1, k2, k3, k4, k5 = st.columns([1.2, 1, 1, 1, 1.4])
    with k1:
        st.metric("Último", format_price(float(last["Close"])))
        st.caption(f"Cambio: {chg:.2f}%" if not pd.isna(chg) else "Cambio: —")
    with k2:
        rsi_val = last.get("RSI14")
        st.metric("RSI (14)", f"{float(rsi_val):.1f}" if rsi_val is not None and not pd.isna(rsi_val) else "—")
        st.caption("Momentum")
    with k3:
        st.metric("ADX (14)", f"{adx_v:.1f}" if adx_v else "—")
        st.caption(f"Régimen: {regime_label(adx_v)}")
    with k4:
        atr  = last.get("ATR14")
        atrp = (
            float(atr) / float(last["Close"]) * 100.0
            if atr is not None and not pd.isna(atr) and float(last["Close"]) != 0
            else np.nan
        )
        st.metric("ATR% (14)", f"{atrp:.2f}%" if not pd.isna(atrp) else "—")
        st.caption("Volatilidad/Riesgo")
    with k5:
        tint = verdict.color
        bg_map = {
            "COMPRA FUERTE": "rgba(0,209,143,0.16)",
            "COMPRA":        "rgba(47,129,247,0.16)",
            "VENTA FUERTE":  "rgba(255,75,75,0.16)",
            "VENTA":         "rgba(255,166,87,0.16)",
            "NEUTRAL":       "rgba(139,148,158,0.10)",
        }
        bg = bg_map.get(verdict.label, "rgba(139,148,158,0.10)")
        # Barra visual de componentes del score
        bar_tech  = int(abs(verdict.tech_score)  * 0.70)
        bar_macro = int(abs(verdict.macro_adj))
        bar_sent  = int(abs(verdict.sent_adj))
        bar_div   = int(abs(verdict.div_adj))
        st.markdown(
            f"""
<div class="qsp-rec" style="border-color:{tint};background:{bg};">
  <div style="font-size:10px;color:rgba(230,237,243,.6);letter-spacing:.5px;">VEREDICTO FINAL</div>
  <div class="qsp-rec-score" style="color:{tint};">{verdict.label}</div>
  <div class="qsp-rec-sub">Score: <b>{verdict.final_score:+.0f}</b>/100
    &nbsp;·&nbsp; Confianza: <b>{verdict.confidence}%</b></div>
  <div style="margin-top:6px;display:flex;gap:2px;height:4px;border-radius:2px;overflow:hidden;">
    <div style="width:{bar_tech}%;background:#2F81F7;" title="Técnico"></div>
    <div style="width:{bar_macro}%;background:#FFA657;" title="Macro"></div>
    <div style="width:{bar_sent}%;background:#56D364;" title="Sentimiento"></div>
    <div style="width:{bar_div}%;background:#D2A8FF;" title="Divergencia"></div>
  </div>
  <div style="font-size:9px;color:rgba(230,237,243,.4);margin-top:3px;">
    🔵técnico &nbsp;🟠macro &nbsp;🟢sentimiento &nbsp;🟣divergencia
  </div>
</div>""".strip(), unsafe_allow_html=True,
        )

    # ── Banners de alertas disparadas ──
    if fired_alerts:
        for fa in fired_alerts:
            st.warning(f"🔔 **ALERTA DISPARADA** — {fa.ticker} · {fa.condition}"
                       f"{f' (umbral: {fa.threshold})' if fa.threshold else ''}"
                       f"{f' — {fa.note}' if fa.note else ''}")

    t_overview, t_chart, t_tech, t_strategy, t_macro, t_news, t_alerts, t_screener, t_radar = st.tabs(
        ["Resumen", "Gráfico", "Técnicos", "Estrategia", "Macro", "Noticias", "Alertas", "Screener", "Radar"]
    )

    with t_overview:
        # ── Veredicto final integrado ──
        vc = verdict.color
        st.markdown(
            f"""
<div style="padding:16px 20px;border-radius:12px;border:1px solid {vc};
     background:rgba(15,23,34,.45);margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:11px;color:rgba(230,237,243,.55);letter-spacing:.6px;margin-bottom:4px;">
        VEREDICTO FINAL — {ticker} · {datetime.now().strftime('%d/%m %H:%M')}
      </div>
      <div style="font-size:32px;font-weight:900;color:{vc};line-height:1;">{verdict.label}</div>
      <div style="font-size:12px;color:rgba(230,237,243,.7);margin-top:6px;">{verdict.summary}</div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
      <div style="text-align:center;padding:8px 14px;border-radius:8px;
           background:rgba(47,129,247,.12);border:1px solid rgba(47,129,247,.3);">
        <div style="font-size:9px;color:rgba(230,237,243,.5);">TÉCNICO</div>
        <div style="font-size:18px;font-weight:800;color:#2F81F7;">{verdict.tech_score:+.0f}</div>
      </div>
      <div style="text-align:center;padding:8px 14px;border-radius:8px;
           background:rgba(255,166,87,.12);border:1px solid rgba(255,166,87,.3);">
        <div style="font-size:9px;color:rgba(230,237,243,.5);">MACRO</div>
        <div style="font-size:18px;font-weight:800;color:#FFA657;">{verdict.macro_adj:+.0f}</div>
      </div>
      <div style="text-align:center;padding:8px 14px;border-radius:8px;
           background:rgba(86,211,100,.12);border:1px solid rgba(86,211,100,.3);">
        <div style="font-size:9px;color:rgba(230,237,243,.5);">SENTIMIENTO</div>
        <div style="font-size:18px;font-weight:800;color:#56D364;">{verdict.sent_adj:+.0f}</div>
      </div>
      <div style="text-align:center;padding:8px 14px;border-radius:8px;
           background:rgba(210,168,255,.12);border:1px solid rgba(210,168,255,.3);">
        <div style="font-size:9px;color:rgba(230,237,243,.5);">DIVERGENCIA</div>
        <div style="font-size:18px;font-weight:800;color:#D2A8FF;">{verdict.div_adj:+.0f}</div>
      </div>
      <div style="text-align:center;padding:8px 14px;border-radius:8px;
           background:rgba(139,148,158,.12);border:1px solid rgba(139,148,158,.3);">
        <div style="font-size:9px;color:rgba(230,237,243,.5);">FINAL</div>
        <div style="font-size:18px;font-weight:900;color:{vc};">{verdict.final_score:+.0f}</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        cL, cR = st.columns([1.05, 0.95])
        with cL:
            st.markdown("### Snapshot")
            snap = {
                "Open":        last.get("Open"),
                "High":        last.get("High"),
                "Low":         last.get("Low"),
                "Close":       last.get("Close"),
                "Volumen":     last.get("Volume"),
                "RelVol (20)": last.get("REL_VOL"),
            }
            snap_df = pd.DataFrame(
                [
                    {
                        "Campo": k,
                        "Valor": format_price(float(v)) if v is not None and not pd.isna(v) else "—",
                    }
                    for k, v in snap.items()
                ]
            )
            st.dataframe(snap_df, use_container_width=True, hide_index=True)

        with cR:
            st.markdown("### Explicación del score")
            st.dataframe(
                expl.assign(
                    **{
                        "Peso":           (expl["Peso"] * 100).round(0).astype(int).astype(str) + "%",
                        "Score (-1..+1)": expl["Score (-1..+1)"].round(3),
                        "Contribución":   (expl["Contribución"] * 100).round(1),
                    }
                )[["Grupo", "Peso", "Score (-1..+1)", "Contribución"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Contribución está en puntos porcentuales del score total (aprox.).")

        st.markdown("---")
        st.markdown("### Más activas del S&P 500 (por $ volumen)")
        with st.spinner("Cargando ranking de más activas…"):
            act = load_most_active_sp500(top_n=15)
        if act.empty:
            st.warning("No se pudo cargar el ranking de más activas (sin datos o sin conexión).")
        else:
            show = act.copy()
            show["Precio"]    = show["Precio"].map(lambda x: format_price(float(x)))
            show["Cambio %"]  = show["Cambio %"].map(lambda x: f"{float(x):.2f}%" if not pd.isna(x) else "—")
            show["Volumen"]   = show["Volumen"].map(lambda x: f"{float(x):,.0f}")
            show["$ Volumen"] = show["$ Volumen"].map(lambda x: f"{float(x):,.0f}")
            st.dataframe(show, use_container_width=True, hide_index=True)

    with t_chart:
        st.markdown("### Gráfico")
        fig = build_chart(
            dfi, overlays=overlays,
            show_rsi=show_rsi, show_macd=show_macd,
            show_patterns=show_pats,
            show_sl_tp=show_sl_tp, risk=risk_levels,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Usa las herramientas de zoom/pan de Plotly. Los patrones de velas se anotan automáticamente en las últimas 30 barras.")

    with t_tech:
        st.markdown("### Señales por indicador")
        st.dataframe(
            details.assign(**{"Señal (-1..+1)": details["Señal (-1..+1)"].round(3)}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Interpretación rápida: +1 bullish, -1 bearish, 0 neutral.")

    with t_strategy:
        st.markdown("### Plan de Estrategia de Inversión")

        # --- Divergencias ---
        div_color = {"Alcista": "#00D18F", "Bajista": "#FF4B4B"}.get(divergence["tipo"], "#8B949E")
        st.markdown(
            f"""
<div style="border-left: 4px solid {div_color}; padding: 10px 16px; border-radius: 8px;
            background: rgba(15,23,34,0.4); margin-bottom: 12px;">
  <b style="color:{div_color};">Divergencia RSI/Precio: {divergence['tipo']}</b><br>
  <span style="font-size:13px; color: rgba(230,237,243,0.85);">{divergence['descripcion']}</span>
</div>
            """,
            unsafe_allow_html=True,
        )

        # --- Niveles de riesgo ---
        st.markdown("#### Gestión de Riesgo (basada en ATR)")
        if risk_levels is None:
            st.warning("No se pudieron calcular niveles de riesgo (ATR no disponible).")
        else:
            rl = risk_levels
            rr_color = "#00D18F" if rl.risk_reward >= 1.5 else "#FFA657" if rl.risk_reward >= 1.0 else "#FF4B4B"

            col_rl1, col_rl2, col_rl3 = st.columns(3)
            with col_rl1:
                st.metric("Entrada (Close)", format_price(rl.entry))
                st.metric("Stop Loss (−1.5× ATR)", format_price(rl.stop_loss),
                          delta=f"−{format_price(rl.entry - rl.stop_loss)} ({(rl.entry - rl.stop_loss)/rl.entry*100:.2f}%)",
                          delta_color="inverse")
            with col_rl2:
                st.metric("TP1 (+1.5× ATR)", format_price(rl.tp1),
                          delta=f"+{format_price(rl.tp1 - rl.entry)}")
                st.metric("TP2 (+3× ATR)",   format_price(rl.tp2),
                          delta=f"+{format_price(rl.tp2 - rl.entry)}")
            with col_rl3:
                st.metric("TP3 (+5× ATR)",   format_price(rl.tp3),
                          delta=f"+{format_price(rl.tp3 - rl.entry)}")
                st.markdown(
                    f"""
<div style="margin-top:8px; padding:10px; border-radius:8px; border:1px solid {rr_color};
            background: rgba(15,23,34,0.4); text-align:center;">
  <div style="font-size:11px; color:rgba(230,237,243,0.7);">RATIO RIESGO/BENEFICIO (TP1)</div>
  <div style="font-size:24px; font-weight:800; color:{rr_color};">1 : {rl.risk_reward:.2f}</div>
  <div style="font-size:11px; color:rgba(230,237,243,0.7);">{"✓ Aceptable" if rl.risk_reward >= 1.5 else "⚠ Bajo — considerar ajuste"}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            col_ps1, col_ps2 = st.columns(2)
            with col_ps1:
                st.markdown("##### Tamaño de Posición Sugerido")
                st.markdown(
                    f"""
<div style="padding:14px; border-radius:10px; background:rgba(15,23,34,0.4);
            border:1px solid rgba(139,148,158,0.2);">
  <p style="margin:0; font-size:13px; color:rgba(230,237,243,0.8);">
    Asumiendo <b>riesgo del 1%</b> del capital por operación y SL de
    <b>{(rl.entry - rl.stop_loss)/rl.entry*100:.2f}%</b>:
  </p>
  <p style="margin:8px 0 0 0; font-size:22px; font-weight:800; color:#2F81F7;">
    {rl.position_size_pct:.1f}% del capital
  </p>
  <p style="margin:4px 0 0 0; font-size:11px; color:rgba(230,237,243,0.6);">
    ATR: {format_price(rl.atr)} ({rl.atr_pct:.2f}% del precio)
  </p>
</div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_ps2:
                st.markdown("##### Guía de Salida Escalonada")
                st.markdown(
                    f"""
<div style="padding:14px; border-radius:10px; background:rgba(15,23,34,0.4);
            border:1px solid rgba(139,148,158,0.2); font-size:13px;
            color:rgba(230,237,243,0.85); line-height:1.8;">
  🟢 <b>TP1 ({format_price(rl.tp1)})</b>: cerrar 40% de la posición<br>
  🟡 <b>TP2 ({format_price(rl.tp2)})</b>: cerrar 35% de la posición<br>
  🔵 <b>TP3 ({format_price(rl.tp3)})</b>: cerrar 25% restante (trailing)<br>
  🔴 <b>SL  ({format_price(rl.stop_loss)})</b>: salida total si se toca
</div>
                    """,
                    unsafe_allow_html=True,
                )

        # --- Condiciones de entrada recomendadas ---
        st.markdown("---")
        st.markdown("#### Condiciones de Entrada Recomendadas")

        adx_ok    = adx_v >= 20
        vol_ok    = not pd.isna(last.get("REL_VOL")) and float(last.get("REL_VOL", 0)) >= 1.1
        trend_ok  = not pd.isna(last.get("EMA50")) and not pd.isna(last.get("EMA200")) and float(last["Close"]) > float(last["EMA50"])
        macd_ok   = not pd.isna(last.get("MACD_HIST")) and float(last.get("MACD_HIST", 0)) > 0
        rsi_ok    = not pd.isna(last.get("RSI14")) and 40 <= float(last.get("RSI14", 50)) <= 70
        div_ok    = divergence["tipo"] == "Alcista"

        conditions = [
            ("ADX ≥ 20 (mercado con tendencia)",           adx_ok),
            ("Volumen relativo ≥ 1.1× (confirmación)",     vol_ok),
            ("Precio sobre EMA50 (tendencia alcista)",      trend_ok),
            ("MACD Histograma positivo",                    macd_ok),
            ("RSI entre 40–70 (zona de momentum sano)",     rsi_ok),
            ("Divergencia alcista detectada (bonus)",       div_ok),
        ]

        cond_rows = [
            {"Condición": label, "Estado": "✅ OK" if ok else "❌ No cumple"}
            for label, ok in conditions
        ]
        met = sum(1 for _, ok in conditions if ok)
        total = len(conditions)

        st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)

        quality_color = "#00D18F" if met >= 5 else "#FFA657" if met >= 3 else "#FF4B4B"
        quality_label = "Alta" if met >= 5 else "Media" if met >= 3 else "Baja"
        st.markdown(
            f"""
<div style="margin-top:8px; padding:10px 16px; border-radius:8px;
            border:1px solid {quality_color}; background:rgba(15,23,34,0.35);">
  <b style="color:{quality_color};">Calidad de setup: {quality_label}</b>
  — {met}/{total} condiciones cumplidas.
  {"Se recomienda esperar mayor convergencia de señales." if met < 4 else "Setup con suficiente confluencia para considerar entrada."}
</div>
            """,
            unsafe_allow_html=True,
        )

        # --- Notas de descargo ---
        st.markdown("---")
        st.caption(
            "⚠️ Esta información es exclusivamente educativa y de análisis técnico. "
            "No constituye asesoramiento financiero. Toda inversión conlleva riesgo de pérdida. "
            "Consulta a un asesor financiero certificado antes de operar."
        )

    # ═══════════════════════════════════════════════════════════════
    # PESTAÑA: MACRO
    # ═══════════════════════════════════════════════════════════════
    with t_macro:
        st.markdown("### Contexto Macroeconómico")

        # ── Régimen de mercado ──
        mkt_regime  = macro_ctx.get("market_regime", "—")
        spy_vs_200  = macro_ctx.get("spy_vs_sma200", float("nan"))
        spy_1d      = macro_ctx.get("spy_1d", float("nan"))
        vix_v       = macro_ctx.get("vix", float("nan"))
        vix_regime  = macro_ctx.get("vix_regime", "—")
        regime_col  = "#00D18F" if mkt_regime == "Bull" else "#FF4B4B" if mkt_regime == "Bear" else "#8B949E"
        vix_col     = "#FF4B4B" if vix_v > 30 else "#FFA657" if vix_v > 20 else "#00D18F"

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(
                f"""<div style="padding:12px;border-radius:10px;border:1px solid {regime_col};
                background:rgba(15,23,34,.4);text-align:center;">
                <div style="font-size:10px;color:rgba(230,237,243,.6);">RÉGIMEN S&P 500</div>
                <div style="font-size:22px;font-weight:800;color:{regime_col};">{mkt_regime or '—'}</div>
                <div style="font-size:11px;color:rgba(230,237,243,.6);">
                SPY vs SMA200: {f'{spy_vs_200:+.1f}%' if not math.isnan(spy_vs_200) else '—'}</div>
                </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(
                f"""<div style="padding:12px;border-radius:10px;border:1px solid {vix_col};
                background:rgba(15,23,34,.4);text-align:center;">
                <div style="font-size:10px;color:rgba(230,237,243,.6);">VIX (MIEDO)</div>
                <div style="font-size:22px;font-weight:800;color:{vix_col};">
                {f'{vix_v:.1f}' if not math.isnan(vix_v) else '—'}</div>
                <div style="font-size:11px;color:{vix_col};">{vix_regime}</div>
                </div>""", unsafe_allow_html=True)
        with mc3:
            beta   = macro_ctx.get("beta_spy", float("nan"))
            b_col  = "#FFA657" if not math.isnan(beta) and abs(beta) > 1.5 else "#8B949E"
            st.markdown(
                f"""<div style="padding:12px;border-radius:10px;border:1px solid {b_col};
                background:rgba(15,23,34,.4);text-align:center;">
                <div style="font-size:10px;color:rgba(230,237,243,.6);">BETA vs SPY (1Y)</div>
                <div style="font-size:22px;font-weight:800;color:{b_col};">
                {f'{beta:.2f}' if not math.isnan(beta) else '—'}</div>
                <div style="font-size:11px;color:rgba(230,237,243,.6);">
                {'Alta volatilidad relativa' if not math.isnan(beta) and abs(beta)>1.5
                 else 'Similar al mercado' if not math.isnan(beta) and abs(beta)>=0.8
                 else 'Baja correlación' if not math.isnan(beta) else '—'}</div>
                </div>""", unsafe_allow_html=True)
        with mc4:
            corr   = macro_ctx.get("corr_spy", float("nan"))
            c_col  = "#2F81F7" if not math.isnan(corr) and corr > 0.7 else "#8B949E"
            st.markdown(
                f"""<div style="padding:12px;border-radius:10px;border:1px solid {c_col};
                background:rgba(15,23,34,.4);text-align:center;">
                <div style="font-size:10px;color:rgba(230,237,243,.6);">CORRELACIÓN SPY (1Y)</div>
                <div style="font-size:22px;font-weight:800;color:{c_col};">
                {f'{corr:.2f}' if not math.isnan(corr) else '—'}</div>
                <div style="font-size:11px;color:rgba(230,237,243,.6);">
                {'Alta' if not math.isnan(corr) and corr>0.7 else
                 'Media' if not math.isnan(corr) and corr>0.4 else 'Baja'}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Fuerza relativa vs SPY y sector ──
        st.markdown("#### Fuerza Relativa")
        rs_cols = st.columns(4)
        rs_data = [
            ("RS vs SPY 1m",    macro_ctx.get("rs_1m",  float("nan"))),
            ("RS vs SPY 3m",    macro_ctx.get("rs_3m",  float("nan"))),
            ("RS vs SPY 6m",    macro_ctx.get("rs_6m",  float("nan"))),
            (f"RS vs {sector_etf or 'Sector'} 6m", sector_rs.get("rs_vs_sector", float("nan"))),
        ]
        for col, (label, val) in zip(rs_cols, rs_data):
            with col:
                color = "#00D18F" if not math.isnan(val) and val > 0 else "#FF4B4B" if not math.isnan(val) else "#8B949E"
                st.markdown(
                    f"""<div style="padding:10px;border-radius:8px;border:1px solid {color};
                    background:rgba(15,23,34,.4);text-align:center;">
                    <div style="font-size:10px;color:rgba(230,237,243,.6);">{label.upper()}</div>
                    <div style="font-size:20px;font-weight:800;color:{color};">
                    {f'{val:+.1f}%' if not math.isnan(val) else '—'}</div>
                    <div style="font-size:10px;color:{color};">
                    {'Outperform ▲' if not math.isnan(val) and val>0 else 'Underperform ▼' if not math.isnan(val) else '—'}
                    </div></div>""", unsafe_allow_html=True)

        if sector_name:
            st.caption(f"Sector detectado: **{sector_name}** → ETF de referencia: **{sector_etf}**")

        st.markdown("---")

        # ── Dashboard de referencias macro ──
        st.markdown("#### Índices y Activos de Referencia")
        snapshots = macro_ctx.get("snapshots", {})
        if snapshots:
            snap_rows = []
            for sym, d in snapshots.items():
                chg = d.get("chg1d", float("nan"))
                snap_rows.append({
                    "Activo":   f"{sym} — {d['name']}",
                    "Precio":   format_price(d["price"]),
                    "Cambio 1d": f"{chg:+.2f}%" if not math.isnan(chg) else "—",
                    "Señal":    ("🟢" if not math.isnan(chg) and chg > 0.3 else
                                 "🔴" if not math.isnan(chg) and chg < -0.3 else "⚪"),
                })
            st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)

        # ── Interpretación macro integrada ──
        st.markdown("---")
        st.markdown("#### Interpretación del Contexto Macro")

        beta_v  = macro_ctx.get("beta_spy", float("nan"))
        rs_6m   = macro_ctx.get("rs_6m",   float("nan"))
        corr_v  = macro_ctx.get("corr_spy", float("nan"))

        insights = []
        if mkt_regime == "Bull":
            insights.append("✅ **Mercado en régimen alcista** (SPY sobre SMA200). Favorece estrategias de tendencia y momentum.")
        elif mkt_regime == "Bear":
            insights.append("🔴 **Mercado en régimen bajista** (SPY bajo SMA200). Priorizar gestión de riesgo, reducir exposición.")

        if not math.isnan(vix_v):
            if vix_v > 30:
                insights.append(f"⚠️ **VIX en zona de pánico ({vix_v:.0f})**. Alta incertidumbre — posibles oportunidades contrarian pero con riesgo elevado.")
            elif vix_v < 15:
                insights.append(f"💤 **VIX bajo ({vix_v:.0f})**. Complacencia del mercado — precaución ante reversiones inesperadas.")

        if not math.isnan(beta_v):
            if beta_v > 1.5:
                insights.append(f"⚡ **Beta alta ({beta_v:.2f})** — este activo amplifica los movimientos del S&P 500. Mayor riesgo y potencial retorno.")
            elif beta_v < 0.5:
                insights.append(f"🛡️ **Beta baja ({beta_v:.2f})** — activo defensivo, poco correlado con el mercado general.")

        if not math.isnan(rs_6m):
            if rs_6m > 5:
                insights.append(f"🚀 **Fuerza relativa positiva vs SPY a 6m (+{rs_6m:.1f}%)** — el activo supera al mercado. Señal de fortaleza.")
            elif rs_6m < -5:
                insights.append(f"📉 **Fuerza relativa negativa vs SPY a 6m ({rs_6m:.1f}%)** — el activo está rezagado respecto al mercado.")

        if sector_rs.get("rs_vs_sector"):
            rs_sect = sector_rs["rs_vs_sector"]
            if not math.isnan(rs_sect):
                if rs_sect > 3:
                    insights.append(f"🏆 **Outperforma a su sector ({sector_name}) en +{rs_sect:.1f}%** — líder sectorial.")
                elif rs_sect < -3:
                    insights.append(f"🐢 **Underperforma a su sector ({sector_name}) en {rs_sect:.1f}%** — rezagado respecto a pares.")

        # Score ajustado por macro
        macro_adj = 0.0
        if mkt_regime == "Bull" and not math.isnan(rs_6m) and rs_6m > 0:
            macro_adj = min(10.0, rs_6m * 0.5)
        elif mkt_regime == "Bear":
            macro_adj = -10.0
        if not math.isnan(vix_v) and vix_v > 30:
            macro_adj -= 5.0

        adj_score  = _clamp(rec.score + macro_adj, -100.0, 100.0)
        delta_str  = f"{macro_adj:+.0f} pts por contexto macro"

        insights.append(
            f"📊 **Score técnico ajustado por macro: {adj_score:+.0f}/100** ({delta_str}). "
            f"Score técnico puro: {rec.score:+.0f}."
        )

        for ins in insights:
            st.markdown(ins)

        if not insights:
            st.info("No hay suficientes datos macro para generar interpretación.")

        st.caption("Fuente: Yahoo Finance (SPY, QQQ, VIX, TLT, GLD, UUP). Actualizado cada 15 min.")

    # ═══════════════════════════════════════════════════════════════
    # PESTAÑA: ALERTAS
    # ═══════════════════════════════════════════════════════════════
    with t_alerts:
        st.markdown("### Sistema de Alertas")
        _init_alerts()
        alerts_list: list[Alert] = st.session_state["qsp_alerts"]
        alert_log:   list[dict]  = st.session_state["qsp_alert_log"]

        # ── Crear nueva alerta ──
        st.markdown("#### ➕ Nueva Alerta")
        with st.form("new_alert_form"):
            al1, al2, al3 = st.columns([2, 1.2, 2])
            with al1:
                al_ticker = st.text_input("Ticker", value=ticker)
                al_cond   = st.selectbox("Condición", list(_ALERT_CONDITIONS.keys()))
            with al2:
                # Condiciones que necesitan umbral numérico
                needs_threshold = any(x in al_cond for x in [
                    "umbral", "RSI", "ADX", "Volumen", "Score"
                ])
                al_thresh = st.number_input(
                    "Umbral",
                    value=30.0 if "RSI" in al_cond and "<" in al_cond else
                          70.0 if "RSI" in al_cond else
                          25.0 if "ADX" in al_cond else
                          1.5  if "Volumen" in al_cond else
                          50.0,
                    disabled=not needs_threshold,
                )
            with al3:
                al_note = st.text_input("Nota (opcional)", placeholder="ej. Niveles de soporte")

            add_btn = st.form_submit_button("Agregar Alerta", use_container_width=True)
            if add_btn and _is_valid_ticker(al_ticker):
                new_alert = Alert(
                    id=str(uuid.uuid4())[:8],
                    ticker=al_ticker.strip().upper(),
                    condition=al_cond,
                    threshold=float(al_thresh) if needs_threshold else 0.0,
                    note=al_note,
                )
                alerts_list.append(new_alert)
                st.success(f"✅ Alerta creada para {new_alert.ticker} — {new_alert.condition}")

        st.markdown("---")

        # ── Alertas activas ──
        st.markdown("#### 📋 Alertas Configuradas")
        if not alerts_list:
            st.info("No hay alertas configuradas. Crea una arriba.")
        else:
            to_delete = []
            for i, a in enumerate(alerts_list):
                status_icon  = "🔔" if a.triggered else ("✅" if a.active else "⏸️")
                status_color = "#FFA657" if a.triggered else ("#00D18F" if a.active else "#8B949E")
                thresh_str   = f" | Umbral: {a.threshold}" if a.threshold else ""
                fired_str    = f" | Disparada: {a.trigger_ts}" if a.triggered else ""

                col_info, col_tog, col_del = st.columns([5, 1, 1])
                with col_info:
                    st.markdown(
                        f"""<div style="padding:8px 12px;border-radius:8px;margin-bottom:4px;
                        border-left:3px solid {status_color};background:rgba(15,23,34,.35);">
                        <span style="color:{status_color};font-size:14px;">{status_icon}</span>
                        <b style="font-size:13px;"> {a.ticker}</b>
                        <span style="font-size:12px;color:rgba(230,237,243,.8);"> — {a.condition}{thresh_str}</span>
                        {f'<span style="font-size:11px;color:rgba(230,237,243,.5);"> | {a.note}</span>' if a.note else ''}
                        <span style="font-size:10px;color:rgba(230,237,243,.4);">{fired_str}</span>
                        </div>""", unsafe_allow_html=True)
                with col_tog:
                    if st.button("⏸" if a.active else "▶", key=f"tog_{a.id}",
                                 help="Pausar/reanudar"):
                        alerts_list[i].active = not alerts_list[i].active
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{a.id}", help="Eliminar"):
                        to_delete.append(i)

            for idx in reversed(to_delete):
                alerts_list.pop(idx)
            if to_delete:
                st.rerun()

        # ── Historial de disparos ──
        st.markdown("---")
        st.markdown("#### 📜 Historial de Alertas Disparadas")
        if not alert_log:
            st.info("Ninguna alerta se ha disparado todavía en esta sesión.")
        else:
            log_df = pd.DataFrame(reversed(alert_log))
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            if st.button("Limpiar historial"):
                st.session_state["qsp_alert_log"] = []
                st.rerun()

        st.caption(
            "Las alertas se evalúan cada vez que se carga el ticker. "
            "Para monitoreo continuo, mantén la pestaña abierta con recarga automática "
            "o ejecuta la app en un servidor."
        )

    with t_news:
        st.markdown("### Noticias y Sentimiento de Mercado")
        with st.spinner("Cargando noticias…"):
            news = load_news(ticker, max_items=20)
        sentiment = aggregate_sentiment(news)

        # ── Banner de sentimiento ──
        sc = sentiment["color"]
        bull_pct = int(sentiment["bull"] / max(len(news), 1) * 100)
        bear_pct = int(sentiment["bear"] / max(len(news), 1) * 100)
        neut_pct = 100 - bull_pct - bear_pct
        st.markdown(
            f"""
<div style="display:flex; gap:16px; margin-bottom:16px; flex-wrap:wrap;">
  <div style="flex:1; min-width:160px; padding:14px; border-radius:10px;
              border:1px solid {sc}; background:rgba(15,23,34,0.4); text-align:center;">
    <div style="font-size:11px; color:rgba(230,237,243,0.7); margin-bottom:4px;">SENTIMIENTO NOTICIAS</div>
    <div style="font-size:26px; font-weight:800; color:{sc};">{sentiment['label']}</div>
    <div style="font-size:12px; color:rgba(230,237,243,0.7);">Score: {sentiment['score']:+.2f}</div>
  </div>
  <div style="flex:2; min-width:240px; padding:14px; border-radius:10px;
              border:1px solid rgba(139,148,158,0.2); background:rgba(15,23,34,0.4);">
    <div style="font-size:11px; color:rgba(230,237,243,0.7); margin-bottom:8px;">DISTRIBUCIÓN ({len(news)} titulares)</div>
    <div style="display:flex; gap:8px; align-items:center; font-size:13px;">
      <span style="color:#00D18F;">🟢 Bullish: {sentiment['bull']} ({bull_pct}%)</span>
      <span style="color:#FF4B4B;">🔴 Bearish: {sentiment['bear']} ({bear_pct}%)</span>
      <span style="color:#8B949E;">⚪ Neutral: {sentiment['neutral']} ({neut_pct}%)</span>
    </div>
    <div style="margin-top:10px; height:8px; border-radius:4px; overflow:hidden; display:flex;">
      <div style="width:{bull_pct}%; background:#00D18F;"></div>
      <div style="width:{neut_pct}%; background:#3d444d;"></div>
      <div style="width:{bear_pct}%; background:#FF4B4B;"></div>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if not news:
            st.warning("No se encontraron noticias para este ticker. Puede que el ticker no tenga cobertura en Yahoo Finance RSS.")
        else:
            # ── Lista de noticias con color de sentimiento ──
            for n in news:
                s = n.get("sentiment", 0.0)
                if s > 0.1:
                    badge_color, badge = "#00D18F", "▲ Bullish"
                elif s < -0.1:
                    badge_color, badge = "#FF4B4B", "▼ Bearish"
                else:
                    badge_color, badge = "#8B949E", "● Neutral"

                st.markdown(
                    f"""
<div style="padding:10px 14px; border-radius:8px; margin-bottom:6px;
            border-left:3px solid {badge_color}; background:rgba(15,23,34,0.35);">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
    <a href="{n['link']}" target="_blank"
       style="color:rgba(230,237,243,0.92); text-decoration:none; font-size:13px;
              font-weight:500; line-height:1.4; flex:1;">
      {n['title']}
    </a>
    <span style="color:{badge_color}; font-size:10px; white-space:nowrap; font-weight:700;">
      {badge}
    </span>
  </div>
  <div style="font-size:10px; color:rgba(230,237,243,0.45); margin-top:4px;">
    {n['source']} • {n['ago']}
  </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("Sentimiento calculado con léxico de palabras clave. Es indicativo, no predictivo.")

    with t_screener:
        st.markdown("### Screener de Acciones (S&P 500)")
        st.caption("Filtra el universo S&P 500 según criterios técnicos y de precio. El análisis técnico se ejecuta en paralelo sobre los candidatos que pasan los filtros de precio/volumen.")

        with st.form("screener_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown("**Precio (USD)**")
                min_price = st.number_input("Mín.", value=5.0,  min_value=0.0,  step=5.0)
                max_price = st.number_input("Máx.", value=2000.0, min_value=1.0, step=50.0)
            with fc2:
                st.markdown("**Cambio diario (%)**")
                min_chg = st.number_input("Mín. %", value=-10.0, min_value=-50.0, step=1.0)
                max_chg = st.number_input("Máx. %", value=10.0,  max_value=50.0,  step=1.0)
            with fc3:
                st.markdown("**Volumen ($M) mín.**")
                min_vol_m = st.number_input("$ Vol mín. (M)", value=50.0, min_value=0.0, step=10.0)
                st.markdown("**RSI (14)**")
                rsi_range = st.slider("Rango RSI", 0, 100, (30, 70))

            fd1, fd2, fd3 = st.columns(3)
            with fd1:
                adx_min = st.number_input("ADX mínimo", value=15.0, min_value=0.0, max_value=60.0, step=5.0)
            with fd2:
                rec_filter = st.selectbox(
                    "Recomendación",
                    ["Todas", "COMPRA", "VENTA", "NEUTRAL"],
                    index=0,
                )
            with fd3:
                top_n = st.number_input("Máx. resultados", value=30, min_value=5, max_value=100, step=5)

            submitted = st.form_submit_button("🔍 Ejecutar Screener", use_container_width=True)

        if submitted:
            with st.spinner("Ejecutando screener… esto puede tomar 20–60 segundos según la red."):
                scr = run_screener(
                    min_price=min_price, max_price=max_price,
                    min_chg=min_chg,     max_chg=max_chg,
                    min_vol_m=min_vol_m,
                    rsi_lo=float(rsi_range[0]), rsi_hi=float(rsi_range[1]),
                    adx_min=adx_min,
                    rec_filter=rec_filter,
                    top_n=int(top_n),
                )

            if scr.empty:
                st.warning("Ningún activo cumplió todos los filtros. Prueba ampliar los rangos.")
            else:
                st.success(f"✅ {len(scr)} activos encontrados")

                # Formatear para display
                show_scr = scr.copy()
                show_scr["Precio"]    = show_scr["Precio"].map(lambda x: format_price(float(x)))
                show_scr["Cambio %"]  = show_scr["Cambio %"].map(lambda x: f"{float(x):+.2f}%" if not math.isnan(x) else "—")
                show_scr["$ Vol (M)"] = show_scr["$ Vol (M)"].map(lambda x: f"${float(x):,.0f}M")
                show_scr["RSI"]       = show_scr["RSI"].map(lambda x: f"{float(x):.1f}" if not math.isnan(x) else "—")
                show_scr["ADX"]       = show_scr["ADX"].map(lambda x: f"{float(x):.1f}")
                show_scr["Score"]     = show_scr["Score"].map(lambda x: f"{float(x):+.0f}")

                st.dataframe(show_scr, use_container_width=True, hide_index=True)
                st.caption("Ordenado por Score descendente. Haz clic en un ticker para analizarlo en la terminal principal.")

    with t_radar:
        st.markdown("### Radar multi-activo")
        if not radar_items:
            st.info("Agrega tickers en la barra lateral para ver el radar.")
        else:
            valid_items = [t for t in radar_items[:30] if _is_valid_ticker(t)]
            invalid_items = [t for t in radar_items[:30] if not _is_valid_ticker(t)]
            if invalid_items:
                st.warning(f"Tickers ignorados por formato inválido: {', '.join(invalid_items)}")

            if valid_items:
                # FIX: paralelización con ThreadPoolExecutor para no bloquear la UI
                with st.spinner(f"Analizando {len(valid_items)} activos en paralelo…"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {
                            executor.submit(_load_and_recommend, t, period, interval): t
                            for t in valid_items
                        }
                        rows = []
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if result is not None:
                                rows.append(result)

                if rows:
                    # Re-ordenar por ticker original para presentación consistente
                    order = {t: i for i, t in enumerate(valid_items)}
                    rows.sort(key=lambda r: order.get(r["Activo"], 999))
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("No se pudieron cargar activos del radar (tickers inválidos o sin datos).")


if __name__ == "__main__":
    main()
