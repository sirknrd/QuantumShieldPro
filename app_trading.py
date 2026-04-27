from __future__ import annotations

import concurrent.futures
import math
import re
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
        fig.add_trace(go.Scatter(x=x, y=view["BBM"], name="BB Mid",   line=dict(color="rgba(139,148,158,0.30)", width=1)))
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


# ---------------------------------------------------------------------------
# Helpers para el Radar paralelizado
# ---------------------------------------------------------------------------

def _load_and_recommend(ticker: str, period: str, interval: str) -> dict | None:
    """Carga datos y calcula recomendación para un ticker. Ejecutado en thread pool."""
    try:
        d0 = load_ohlcv(ticker, period=period, interval=interval)
        if d0.empty:
            return None
        di = compute_indicators(d0)
        r, _, _, tr, ax = recommend(di)
        l0 = di.iloc[-1]
        p  = float(l0["Close"])
        pc = float(di["Close"].iloc[-2]) if len(di) >= 2 else float("nan")
        ch = _safe_pct(p, pc) if not pd.isna(pc) else float("nan")
        return {
            "Activo":         ticker,
            "Precio":         format_price(p),
            "Cambio %":       f"{ch:.2f}%" if not pd.isna(ch) else "—",
            "Recomendación":  r.label,
            "Score":          f"{r.score:+.0f}",
            "Confianza":      f"{r.confidence}%",
            "ADX":            f"{ax:.1f}" if ax else "—",
            "Régimen":        regime_label(ax),
            "RelVol":         f"{float(l0.get('REL_VOL', np.nan)):.2f}x"
                              if not pd.isna(l0.get("REL_VOL")) else "—",
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
            "Overlays",
            ["Volumen", "EMA 50/200", "Bandas Bollinger"],
            default=["Volumen", "EMA 50/200"],
        )

        st.markdown("---")
        st.markdown("### Estrategia")
        risk_per_trade = st.slider(
            "Riesgo por operación (%)",
            min_value=0.25,
            max_value=5.0,
            value=1.0,
            step=0.25,
            help="% del capital total que arriesgas por operación. Se usa para calcular el tamaño de posición.",
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

    # KPI row
    k1, k2, k3, k4, k5 = st.columns([1.2, 1, 1, 1, 1.2])
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

    t_overview, t_chart, t_tech, t_strategy, t_radar = st.tabs(
        ["Resumen", "Gráfico", "Técnicos", "Estrategia", "Radar"]
    )

    with t_overview:
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
