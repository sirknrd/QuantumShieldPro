"""
Quantum Shield Pro — Financial Trading Terminal
Mejoras: Backtesting · Multi-Timeframe · Filtro Volumen · Caché · Intraday · Paper Trading
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumShield Pro",
    page_icon="🛡️",
    layout="wide",
)

# Paper Trading state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {
        "cash": 10_000.0,
        "position": 0.0,
        "entry_price": 0.0,
        "trades": [],
        "equity_curve": [],
    }
if "pt_capital" not in st.session_state:
    st.session_state.pt_capital = 10_000.0

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

MERCADOS = {
    "🌐 Criptomonedas":              ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "📈 Acciones Internacionales":   ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "META"],
    "🇨🇱 Mercado Chileno (IPSA)":   ["FALABELLA.SN", "COPEC.SN", "BCI.SN", "CMPC.SN", "CHILE.SN"],
    "✏️ Personalizado":              [],
}

mercado = st.sidebar.selectbox("Mercado", list(MERCADOS.keys()))
if mercado == "✏️ Personalizado":
    ticker = st.sidebar.text_input("Ticker", value="NVDA").upper().strip()
else:
    ticker = st.sidebar.selectbox("Activo", MERCADOS[mercado])

INTERVAL_OPTIONS = {
    "15 minutos": "15m",
    "30 minutos": "30m",
    "1 hora":     "1h",
    "4 horas":    "4h",
    "Diario":     "1d",
    "Semanal":    "1wk",
}
PERIOD_BY_INTERVAL = {
    "15m": ["1d", "5d", "1mo"],
    "30m": ["1d", "5d", "1mo"],
    "1h":  ["5d", "1mo", "3mo"],
    "4h":  ["1mo", "3mo", "6mo"],
    "1d":  ["3mo", "6mo", "1y", "2y"],
    "1wk": ["1y", "2y", "5y"],
}

interval_label = st.sidebar.selectbox("Intervalo", list(INTERVAL_OPTIONS.keys()), index=4)
interval = INTERVAL_OPTIONS[interval_label]
period   = st.sidebar.selectbox("Período", PERIOD_BY_INTERVAL[interval])

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Parámetros de Riesgo")
atr_sl_mult = st.sidebar.slider("ATR Stop Loss ×",   1.0, 4.0, 1.5, 0.25)
atr_tp_mult = st.sidebar.slider("ATR Take Profit ×", 1.5, 6.0, 3.0, 0.25)

st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Filtro de Volumen")
vol_filter = st.sidebar.checkbox("Activar filtro de volumen", value=True,
                                  help="La señal solo es válida si el volumen supera su media")
vol_mult   = st.sidebar.slider("Volumen mínimo (× media)", 1.0, 3.0, 1.2, 0.1,
                                disabled=not vol_filter)

st.sidebar.markdown("---")
st.sidebar.subheader("💼 Paper Trading")
pt_capital_input = st.sidebar.number_input(
    "Capital inicial (USD)", min_value=100,
    value=int(st.session_state.pt_capital), step=500)
if pt_capital_input != st.session_state.pt_capital:
    st.session_state.pt_capital = float(pt_capital_input)
    st.session_state.portfolio  = {
        "cash": float(pt_capital_input), "position": 0.0,
        "entry_price": 0.0, "trades": [], "equity_curve": [],
    }
pt_size_pct = st.sidebar.slider("Tamaño por operación (%)", 5, 100, 50, 5)
if st.sidebar.button("🔄 Resetear Portfolio"):
    st.session_state.portfolio = {
        "cash": st.session_state.pt_capital, "position": 0.0,
        "entry_price": 0.0, "trades": [], "equity_curve": [],
    }


# ─────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────
def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_macd(s, f=12, sl=26, sig=9):
    ml = calc_ema(s, f) - calc_ema(s, sl)
    sg = calc_ema(ml, sig)
    return ml, sg, ml - sg

def calc_atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_bollinger(s, n=20, k=2.0):
    m   = s.rolling(n).mean()
    std = s.rolling(n).std()
    return m + k * std, m, m - k * std


# ─────────────────────────────────────────────
# MOTOR DE CONFLUENCIA + FILTRO VOLUMEN
# ─────────────────────────────────────────────
def motor_confluencia(df: pd.DataFrame, use_vol: bool, vmult: float) -> pd.DataFrame:
    c = df["Close"]

    df["EMA50"]                              = calc_ema(c, 50)
    df["RSI"]                                = calc_rsi(c)
    df["MACD"], df["MACD_Sig"], df["MACD_Hist"] = calc_macd(c)
    df["ATR"]                                = calc_atr(df)
    df["BB_U"], df["BB_M"], df["BB_L"]       = calc_bollinger(c)
    df["Vol_MA"]                             = df["Volume"].rolling(20).mean()

    above_ema = c > df["EMA50"]
    ema_buy   = (~above_ema.shift(1).fillna(False)) & above_ema
    ema_sell  = above_ema.shift(1).fillna(False) & (~above_ema)

    r = df["RSI"]
    rsi_buy  = (r.shift(1) < 30) & (r >= 30)
    rsi_sell = (r.shift(1) > 70) & (r <= 70)

    m, s = df["MACD"], df["MACD_Sig"]
    macd_buy  = (m.shift(1) < s.shift(1)) & (m >= s)
    macd_sell = (m.shift(1) > s.shift(1)) & (m <= s)

    buy_score  = ema_buy.astype(int)  + rsi_buy.astype(int)  + macd_buy.astype(int)
    sell_score = ema_sell.astype(int) + rsi_sell.astype(int) + macd_sell.astype(int)

    vol_ok = (df["Volume"] >= df["Vol_MA"] * vmult) if use_vol else pd.Series(True, index=df.index)

    df["Signal"]     = "NEUTRAL"
    df["Score_Buy"]  = buy_score
    df["Score_Sell"] = sell_score
    df.loc[(buy_score  >= 2) & vol_ok, "Signal"] = "BUY"
    df.loc[(sell_score >= 2) & vol_ok, "Signal"] = "SELL"

    df["SL_Buy"]  = c - atr_sl_mult * df["ATR"]
    df["TP_Buy"]  = c + atr_tp_mult * df["ATR"]
    df["SL_Sell"] = c + atr_sl_mult * df["ATR"]
    df["TP_Sell"] = c - atr_tp_mult * df["ATR"]

    return df


# ─────────────────────────────────────────────
# DESCARGA CON CACHÉ (TTL 5 min)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Columnas inesperadas: {list(df.columns)}")
    return df[list(needed)].dropna(subset=["Close"]).copy()


# ─────────────────────────────────────────────
# MULTI-TIMEFRAME
# ─────────────────────────────────────────────
MTF_MAP = {
    "15m": ("1h",  "1d"),
    "30m": ("4h",  "1d"),
    "1h":  ("4h",  "1d"),
    "4h":  ("1d",  "1wk"),
    "1d":  ("1wk", None),
    "1wk": (None,  None),
}
MTF_PERIOD = {
    "15m": "5d", "30m": "5d", "1h": "1mo",
    "4h": "3mo", "1d": "6mo", "1wk": "2y",
}

def get_mtf_bias(ticker, tf):
    result = {}
    for sup_tf in MTF_MAP.get(tf, (None, None)):
        if sup_tf is None:
            continue
        try:
            per = MTF_PERIOD.get(sup_tf, "3mo")
            d   = get_data(ticker, per, sup_tf)
            if len(d) < 55:
                continue
            d["EMA50"] = calc_ema(d["Close"], 50)
            d["RSI"]   = calc_rsi(d["Close"])
            lc = float(d["Close"].iloc[-1])
            le = float(d["EMA50"].iloc[-1])
            lr = float(d["RSI"].iloc[-1])
            bias = ("🟢 Alcista" if (lc > le and lr > 50) else
                    "🔴 Bajista" if (lc < le and lr < 50) else "🟡 Neutral")
            result[sup_tf] = {
                "bias": bias, "rsi": round(lr, 1),
                "ema_label": "Sobre EMA50" if lc > le else "Bajo EMA50",
            }
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, init_capital: float = 10_000.0) -> dict:
    cash     = init_capital
    position = 0.0
    entry_px = 0.0
    sl = tp  = 0.0
    trades   = []
    equity   = []

    for idx, row in df.iterrows():
        price = float(row["Close"])

        if position > 0:
            if price <= sl or price >= tp or row["Signal"] == "SELL":
                pnl    = (price - entry_px) * position
                reason = ("SL" if price <= sl else
                          "TP" if price >= tp else "Señal SELL")
                cash  += position * price
                trades.append({
                    "Fecha cierre": idx,
                    "Entrada":  round(entry_px, 4),
                    "Salida":   round(price, 4),
                    "PnL USD":  round(pnl, 2),
                    "PnL %":    round((price / entry_px - 1) * 100, 2),
                    "Resultado": "✅ Win" if pnl > 0 else "❌ Loss",
                    "Cierre por": reason,
                })
                position = 0.0

        if position == 0 and row["Signal"] == "BUY":
            invest   = cash * 0.95
            position = invest / price
            entry_px = price
            sl       = float(row["SL_Buy"])
            tp       = float(row["TP_Buy"])
            cash    -= invest

        equity.append({"Fecha": idx, "Equity": cash + position * price})

    if position > 0:
        price = float(df["Close"].iloc[-1])
        pnl   = (price - entry_px) * position
        cash += position * price
        trades.append({
            "Fecha cierre": df.index[-1],
            "Entrada":  round(entry_px, 4),
            "Salida":   round(price, 4),
            "PnL USD":  round(pnl, 2),
            "PnL %":    round((price / entry_px - 1) * 100, 2),
            "Resultado": "✅ Win" if pnl > 0 else "❌ Loss",
            "Cierre por": "Fin período",
        })
        equity[-1]["Equity"] = cash

    eq_df     = pd.DataFrame(equity).set_index("Fecha")
    total_ret = (cash / init_capital - 1) * 100
    wins      = [t for t in trades if t["PnL USD"] > 0]
    losses    = [t for t in trades if t["PnL USD"] <= 0]
    win_rate  = len(wins) / len(trades) * 100 if trades else 0
    gross_p   = sum(t["PnL USD"] for t in wins)  or 0
    gross_l   = abs(sum(t["PnL USD"] for t in losses)) or 1
    eq_s      = eq_df["Equity"]
    dd        = (eq_s - eq_s.cummax()) / eq_s.cummax() * 100
    rets      = eq_s.pct_change().dropna()
    sharpe    = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    return {
        "trades": trades, "equity_df": eq_df,
        "total_ret": total_ret, "win_rate": win_rate,
        "profit_factor": gross_p / gross_l,
        "max_drawdown": float(dd.min()),
        "sharpe": sharpe,
        "n_trades": len(trades),
        "final_equity": cash,
    }


# ─────────────────────────────────────────────
# CARGA PRINCIPAL
# ─────────────────────────────────────────────
if not ticker:
    st.warning("Ingresa un ticker válido.")
    st.stop()

with st.spinner(f"Cargando {ticker} [{interval_label} · {period}]…"):
    try:
        df = get_data(ticker, period, interval)
        if len(df) < 55:
            st.error(f"Solo {len(df)} velas — extiende el período.")
            st.stop()
        df = motor_confluencia(df, vol_filter, vol_mult)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

sig_icon = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_main, tab_mtf, tab_bt, tab_pt = st.tabs([
    "📈 Terminal", "🔭 Multi-Timeframe", "📊 Backtesting", "💼 Paper Trading"
])


# ══════════════════════════════════════════════
# TAB 1 — TERMINAL
# ══════════════════════════════════════════════
with tab_main:
    st.title(f"🛡️ Quantum Shield Pro — {ticker}")

    last      = float(df["Close"].iloc[-1])
    prev      = float(df["Close"].iloc[-2])
    change    = (last / prev - 1) * 100
    rsi_v     = float(df["RSI"].iloc[-1])
    atr_v     = float(df["ATR"].iloc[-1])
    signal    = df["Signal"].iloc[-1]
    vol_v     = float(df["Volume"].iloc[-1])
    volma_v   = float(df["Vol_MA"].iloc[-1])
    vol_ratio = vol_v / volma_v if volma_v > 0 else 1.0
    ts_fmt    = "%d/%m/%Y %H:%M" if interval in ["15m","30m","1h","4h"] else "%d/%m/%Y"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Precio",  f"${last:,.4f}",   f"{change:+.2f}%")
    c2.metric("RSI (14)", f"{rsi_v:.1f}",
              "Sobrevendido" if rsi_v < 30 else "Sobrecomprado" if rsi_v > 70 else "Neutral")
    c3.metric("ATR (14)", f"${atr_v:,.4f}")
    c4.metric("Volumen",  f"{vol_v:,.0f}",   f"× {vol_ratio:.1f} vs media")
    c5.metric("Última vela", df.index[-1].strftime(ts_fmt))
    c6.metric("Señal",   f"{sig_icon.get(signal,'')} {signal}")

    if signal != "NEUTRAL":
        st.markdown("---")
        st.subheader("📐 Gestión de Riesgo ATR")
        sl = float(df["SL_Buy"  if signal == "BUY" else "SL_Sell"].iloc[-1])
        tp = float(df["TP_Buy"  if signal == "BUY" else "TP_Sell"].iloc[-1])
        sl_dist = abs(last - sl)
        rr = abs(tp - last) / sl_dist if sl_dist > 0 else 0
        riesgo_usd = st.session_state.pt_capital * 0.01
        pos_sz = riesgo_usd / sl_dist if sl_dist > 0 else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Stop Loss",   f"${sl:,.4f}", f"{((sl/last)-1)*100:+.2f}%")
        r2.metric("Take Profit", f"${tp:,.4f}", f"{((tp/last)-1)*100:+.2f}%")
        r3.metric("Posición (1% riesgo)", f"{pos_sz:,.4f} u", f"${riesgo_usd:,.0f}")
        r4.metric("Ratio R:R",   f"1 : {rr:.2f}", "✅" if rr >= 2 else "⚠️")

    # ── Gráfico principal ──
    st.markdown("---")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=("Precio · Bollinger · EMA50", "Volumen", "RSI (14)", "MACD (12/26/9)"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Precio",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"],
        line=dict(color="rgba(100,180,255,0.4)", width=1),
        name="BB Sup", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"],
        fill="tonexty", fillcolor="rgba(100,180,255,0.07)",
        line=dict(color="rgba(100,180,255,0.4)", width=1),
        name="Bollinger"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_M"],
        line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dot"),
        name="SMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"],
        line=dict(color="#FFA726", width=1.5), name="EMA 50"), row=1, col=1)

    buys  = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys.index,  y=buys["Low"]   * 0.992, mode="markers",
        marker=dict(symbol="triangle-up",   size=12, color="#00E676"), name="BUY"),  row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.008, mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="#FF1744"), name="SELL"), row=1, col=1)

    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color=vol_colors, name="Vol", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_MA"],
        line=dict(color="#FFD54F", width=1.2, dash="dash"), name="Vol MA20"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color="#AB47BC", width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.4, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.4, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.04, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.04, row=3, col=1)

    hc = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
        marker_color=hc, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
        line=dict(color="#42A5F5", width=1.5), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Sig"],
        line=dict(color="#FF7043", width=1.5), name="Señal MACD"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", height=860,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=1, xanchor="right"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_yaxes(row=3, col=1, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # Historial + exportar
    st.subheader("📋 Historial de Señales")
    sig_hist = df[df["Signal"] != "NEUTRAL"].tail(30).iloc[::-1]

    def fmt_row(row):
        sig = row["Signal"]
        sl_ = row["SL_Buy"]  if sig == "BUY" else row["SL_Sell"]
        tp_ = row["TP_Buy"]  if sig == "BUY" else row["TP_Sell"]
        sc  = int(row["Score_Buy"] if sig == "BUY" else row["Score_Sell"])
        vr  = row["Volume"] / row["Vol_MA"] if row["Vol_MA"] > 0 else 0
        return pd.Series({
            "Fecha":       row.name.strftime(ts_fmt),
            "Señal":       "🟢 COMPRA" if sig == "BUY" else "🔴 VENTA",
            "Precio":      f"${float(row['Close']):,.4f}",
            "Stop Loss":   f"${float(sl_):,.4f}",
            "Take Profit": f"${float(tp_):,.4f}",
            "Confluencia": "⭐" * sc + f" ({sc}/3)",
            "Vol/Media":   f"× {vr:.2f}",
        })

    if not sig_hist.empty:
        tabla = sig_hist.apply(fmt_row, axis=1)
        st.dataframe(tabla, use_container_width=True, hide_index=True)
        buf = io.StringIO()
        tabla.to_csv(buf, index=False)
        st.download_button("⬇️ Exportar CSV",
                           buf.getvalue(),
                           f"señales_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv")
    else:
        st.info("Sin señales en este período. Extiende el rango o desactiva el filtro de volumen.")


# ══════════════════════════════════════════════
# TAB 2 — MULTI-TIMEFRAME
# ══════════════════════════════════════════════
with tab_mtf:
    st.header(f"🔭 Análisis Multi-Timeframe — {ticker}")
    st.caption("Confirma la dirección en timeframes superiores antes de operar.")

    with st.spinner("Calculando TF superiores…"):
        mtf_data = get_mtf_bias(ticker, interval)

    tf1, tf2 = MTF_MAP.get(interval, (None, None))

    lc   = float(df["Close"].iloc[-1])
    le   = float(df["EMA50"].iloc[-1])
    lr   = float(df["RSI"].iloc[-1])
    b_cur = ("🟢 Alcista" if (lc > le and lr > 50) else
             "🔴 Bajista" if (lc < le and lr < 50) else "🟡 Neutral")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(f"TF Actual ({interval_label})", b_cur)
        st.write(f"RSI: **{lr:.1f}** | {'Sobre' if lc > le else 'Bajo'} EMA50")

    for col, tf in zip([col_b, col_c], [tf1, tf2]):
        if tf and tf in mtf_data:
            with col:
                d = mtf_data[tf]
                st.metric(f"TF Superior ({tf})", d["bias"])
                st.write(f"RSI: **{d['rsi']}** | {d['ema_label']}")
        elif tf:
            with col:
                st.metric(f"TF Superior ({tf})", "⚠️ Sin datos")

    st.markdown("---")
    biases  = [b_cur] + [mtf_data[tf]["bias"] for tf in [tf1, tf2] if tf and tf in mtf_data]
    n_bull  = sum(1 for b in biases if "Alcista" in b)
    n_bear  = sum(1 for b in biases if "Bajista" in b)
    total   = len(biases)

    if n_bull == total:
        verdict = "✅ **ALINEACIÓN ALCISTA TOTAL** — Condiciones óptimas para COMPRA"
    elif n_bear == total:
        verdict = "🚨 **ALINEACIÓN BAJISTA TOTAL** — Condiciones óptimas para VENTA"
    elif n_bull > n_bear:
        verdict = f"🟡 **SESGO ALCISTA PARCIAL** ({n_bull}/{total} TF) — Opera con precaución"
    elif n_bear > n_bull:
        verdict = f"🟡 **SESGO BAJISTA PARCIAL** ({n_bear}/{total} TF) — Opera con precaución"
    else:
        verdict = "⚪ **SIN ALINEACIÓN** — Mercado en rango, mejor esperar"

    st.markdown(f"### Veredicto\n> {verdict}")

    # Mini gráficos por TF
    st.markdown("---")
    st.subheader("Gráficos por Timeframe")
    tfs_show = [(interval_label, interval, period)] + [
        (tf, tf, MTF_PERIOD.get(tf, "3mo")) for tf in [tf1, tf2] if tf
    ]
    cols = st.columns(len(tfs_show))
    for col, (lbl, iv, per) in zip(cols, tfs_show):
        with col:
            st.caption(f"**{lbl}**")
            try:
                d_tf = get_data(ticker, per, iv)
                if len(d_tf) >= 55:
                    d_tf["EMA50"] = calc_ema(d_tf["Close"], 50)
                    mini = go.Figure()
                    mini.add_trace(go.Candlestick(
                        x=d_tf.index[-60:], open=d_tf["Open"][-60:],
                        high=d_tf["High"][-60:], low=d_tf["Low"][-60:],
                        close=d_tf["Close"][-60:], showlegend=False,
                        increasing_line_color="#26a69a",
                        decreasing_line_color="#ef5350",
                    ))
                    mini.add_trace(go.Scatter(
                        x=d_tf.index[-60:], y=d_tf["EMA50"][-60:],
                        line=dict(color="#FFA726", width=1.5), showlegend=False,
                    ))
                    mini.update_layout(
                        template="plotly_dark", height=220,
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=0, r=0, t=5, b=0),
                    )
                    st.plotly_chart(mini, use_container_width=True)
            except Exception:
                st.info("Sin datos suficientes")


# ══════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ══════════════════════════════════════════════
with tab_bt:
    st.header(f"📊 Backtesting — {ticker} [{interval_label} · {period}]")
    st.caption("Simulación histórica de la estrategia de confluencia. Long-only.")

    bt_cap = st.number_input("Capital inicial (USD)", min_value=100,
                              value=10_000, step=500, key="bt_cap")

    if st.button("▶️ Ejecutar Backtest"):
        with st.spinner("Simulando…"):
            bt = run_backtest(df, float(bt_cap))
        st.session_state["bt_results"] = bt

    if "bt_results" in st.session_state:
        bt = st.session_state["bt_results"]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Retorno Total",  f"{bt['total_ret']:+.2f}%",
                  f"${bt['final_equity']:,.0f}")
        k2.metric("Win Rate",       f"{bt['win_rate']:.1f}%",
                  f"{sum(1 for t in bt['trades'] if t['PnL USD']>0)} / {bt['n_trades']} trades")
        k3.metric("Profit Factor",  f"{bt['profit_factor']:.2f}",
                  "✅" if bt['profit_factor'] >= 1.5 else "⚠️")
        k4.metric("Max Drawdown",   f"{bt['max_drawdown']:.2f}%")
        k5.metric("Sharpe",         f"{bt['sharpe']:.2f}",
                  "✅" if bt['sharpe'] >= 1 else "⚠️")

        st.markdown("---")
        eq_df = bt["equity_df"]
        bm    = (df["Close"] / float(df["Close"].iloc[0])) * float(bt_cap)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df.index, y=eq_df["Equity"],
            fill="tozeroy", fillcolor="rgba(38,166,154,0.15)",
            line=dict(color="#26a69a", width=2), name="Estrategia"))
        fig_eq.add_trace(go.Scatter(
            x=bm.index, y=bm,
            line=dict(color="#546E7A", width=1.5, dash="dash"), name="Buy & Hold"))
        fig_eq.update_layout(template="plotly_dark", height=320,
                              title="Equity vs Buy & Hold",
                              margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_eq, use_container_width=True)

        roll_max  = eq_df["Equity"].cummax()
        dd_series = (eq_df["Equity"] - roll_max) / roll_max * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series,
            fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
            line=dict(color="#ef5350", width=1.5), name="Drawdown"))
        fig_dd.update_layout(template="plotly_dark", height=200,
                              title="Drawdown (%)",
                              margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Log de Operaciones")
        if bt["trades"]:
            td = pd.DataFrame(bt["trades"])
            td["Fecha cierre"] = td["Fecha cierre"].apply(
                lambda x: x.strftime(ts_fmt) if hasattr(x, "strftime") else str(x))
            td["Entrada"] = td["Entrada"].apply(lambda x: f"${x:,.4f}")
            td["Salida"]  = td["Salida"].apply(lambda x:  f"${x:,.4f}")
            td["PnL USD"] = td["PnL USD"].apply(lambda x: f"${x:+,.2f}")
            td["PnL %"]   = td["PnL %"].apply(lambda x:   f"{x:+.2f}%")
            st.dataframe(td, use_container_width=True, hide_index=True)

            buf_bt = io.StringIO()
            td.to_csv(buf_bt, index=False)
            st.download_button("⬇️ Exportar trades CSV",
                               buf_bt.getvalue(),
                               f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                               "text/csv")
        else:
            st.info("No se generaron operaciones en este período.")


# ══════════════════════════════════════════════
# TAB 4 — PAPER TRADING
# ══════════════════════════════════════════════
with tab_pt:
    st.header(f"💼 Paper Trading — {ticker}")
    st.caption("Simulación sin dinero real. Los datos tienen el retraso habitual de yfinance.")

    port  = st.session_state.portfolio
    price = float(df["Close"].iloc[-1])
    sig   = df["Signal"].iloc[-1]

    equity_now = port["cash"] + port["position"] * price
    pnl_open   = (price - port["entry_price"]) * port["position"] if port["position"] > 0 else 0.0
    ret_total  = (equity_now / st.session_state.pt_capital - 1) * 100

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Equity",    f"${equity_now:,.2f}", f"{ret_total:+.2f}%")
    p2.metric("Cash",      f"${port['cash']:,.2f}")
    p3.metric("Posición",  f"{port['position']:,.6f} u",
              f"Entry: ${port['entry_price']:,.4f}" if port["position"] > 0 else "—")
    p4.metric("PnL abierto", f"${pnl_open:+,.2f}",
              f"{(pnl_open/(port['entry_price']*port['position'])*100):+.2f}%"
              if port["position"] > 0 else "—")
    p5.metric("Trades", len(port["trades"]))

    st.markdown("---")
    col_b, col_s, col_i = st.columns([1, 1, 2])

    invest_usd = port["cash"] * (pt_size_pct / 100)
    units_buy  = invest_usd / price if price > 0 else 0

    with col_b:
        st.subheader("🟢 COMPRAR")
        st.write(f"Precio: **${price:,.4f}**")
        st.write(f"Invertir: **${invest_usd:,.2f}** ({pt_size_pct}%)")
        st.write(f"Unidades: **{units_buy:,.6f}**")
        if st.button("✅ Ejecutar COMPRA",
                     disabled=port["position"] > 0 or port["cash"] < 1,
                     use_container_width=True):
            port["cash"]        -= invest_usd
            port["position"]    += units_buy
            port["entry_price"]  = price
            port["trades"].append({
                "Tipo": "COMPRA", "Precio": price, "Unidades": units_buy,
                "Total": invest_usd,
                "SL": float(df["SL_Buy"].iloc[-1]),
                "TP": float(df["TP_Buy"].iloc[-1]),
                "Fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "PnL USD": None,
            })
            port["equity_curve"].append({"Fecha": datetime.now(), "Equity": equity_now})
            st.success(f"Compra: {units_buy:,.6f} u @ ${price:,.4f}")
            st.rerun()

    with col_s:
        st.subheader("🔴 VENDER")
        if port["position"] > 0:
            sale_val = port["position"] * price
            pnl_s    = sale_val - port["entry_price"] * port["position"]
            st.write(f"Posición: **{port['position']:,.6f} u**")
            st.write(f"Valor: **${sale_val:,.2f}**")
            st.write(f"PnL est.: **${pnl_s:+,.2f}**")
        else:
            st.write("Sin posición abierta")
        if st.button("🔴 Ejecutar VENTA",
                     disabled=port["position"] == 0,
                     use_container_width=True):
            sale_val = port["position"] * price
            pnl_s    = sale_val - port["entry_price"] * port["position"]
            if port["trades"]:
                port["trades"][-1]["PnL USD"] = round(pnl_s, 2)
            port["cash"]        += sale_val
            port["equity_curve"].append({"Fecha": datetime.now(), "Equity": port["cash"]})
            port["position"]     = 0.0
            port["entry_price"]  = 0.0
            st.success(f"Venta ejecutada — PnL: ${pnl_s:+,.2f}")
            st.rerun()

    with col_i:
        st.subheader("📡 Señal Actual")
        st.info(f"Confluencia: **{sig_icon.get(sig,'')} {sig}**")
        if sig == "BUY":
            st.success("El motor detecta señal de COMPRA — puede ejecutar manualmente.")
        elif sig == "SELL":
            st.warning("El motor detecta señal de VENTA — considera cerrar la posición.")
        else:
            st.write("Sin señal activa — espera confluencia.")

    if port["trades"]:
        st.markdown("---")
        st.subheader("📋 Historial Paper")
        pt_df = pd.DataFrame(port["trades"])
        pt_df["Precio"] = pt_df["Precio"].apply(lambda x: f"${x:,.4f}")
        pt_df["Total"]  = pt_df["Total"].apply(lambda x:  f"${x:,.2f}")
        pt_df["PnL USD"]= pt_df["PnL USD"].apply(
            lambda x: f"${x:+,.2f}" if x is not None else "Abierta")
        st.dataframe(
            pt_df[["Fecha","Tipo","Precio","Unidades","Total","PnL USD"]],
            use_container_width=True, hide_index=True,
        )

        if len(port["equity_curve"]) >= 2:
            eq_pt = pd.DataFrame(port["equity_curve"])
            fig_pt = go.Figure()
            fig_pt.add_trace(go.Scatter(
                x=eq_pt["Fecha"], y=eq_pt["Equity"],
                fill="tozeroy", fillcolor="rgba(171,71,188,0.15)",
                line=dict(color="#AB47BC", width=2), name="Equity Paper"))
            fig_pt.add_hline(y=st.session_state.pt_capital,
                             line_dash="dash", line_color="#546E7A", opacity=0.5)
            fig_pt.update_layout(template="plotly_dark", height=250,
                                 title="Curva de Equity Paper",
                                 margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pt, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ Solo para fines educativos. No constituye asesoramiento financiero.")
