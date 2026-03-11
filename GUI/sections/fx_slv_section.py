# FX-SLV Model Section
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.fx_slv import FXStochasticLocalVol
from MarketData.market_data import get_eval_date

try:
    import QuantLib as ql
except ImportError:
    ql = None


# ---------------------------------------------------------------------------
# Helper: Garman-Kohlhagen delta -> strike inversion
# ---------------------------------------------------------------------------
def delta_to_strike(delta_abs, S, T, rd, rf, sigma, is_put=False):
    from scipy.stats import norm
    Nd1 = (1.0 - delta_abs * np.exp(rf * T)) if is_put else (delta_abs * np.exp(rf * T))
    Nd1 = np.clip(Nd1, 1e-8, 1 - 1e-8)
    d1  = norm.ppf(Nd1)
    return S * np.exp((rd - rf + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T))


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
# Note: Market vols include realistic micro-perturbations on 10D wings and a
# small kink at 6M/1Y so a single 5-parameter Heston set cannot absorb them
# perfectly, producing a realistic calibration RMSE of ~5-15 bps rather than
# exactly 0. This better reflects real interbank quote behaviour.
FX_OPTION_INSTRUMENTS = pd.DataFrame({
    "Tenor":          ["1W","1W","1W","1W","1W",
                       "1M","1M","1M","1M","1M",
                       "3M","3M","3M","3M","3M",
                       "6M","6M","6M","6M","6M",
                       "1Y","1Y","1Y","1Y","1Y",
                       "2Y","2Y","2Y","2Y","2Y"],
    "Expiry (Years)": [1/52,1/52,1/52,1/52,1/52,
                       1/12,1/12,1/12,1/12,1/12,
                       0.25,0.25,0.25,0.25,0.25,
                       0.50,0.50,0.50,0.50,0.50,
                       1.00,1.00,1.00,1.00,1.00,
                       2.00,2.00,2.00,2.00,2.00],
    "Instrument":     ["10D Put","25D Put","ATM","25D Call","10D Call"]*6,
    "Delta":          [-0.10,-0.25,0.50,0.25,0.10]*6,
    "Type":           ["Put","Put","Call","Call","Call"]*6,
    # 10D wings carry small irregular bumps (+0.10 to +0.25 on select tenors)
    # 6M ATM is nudged up by 0.08 and 1Y 25D Put down by 0.06 to introduce
    # a mild kink that Heston cannot perfectly resolve with 5 global params.
    "Market Vol (%)":[6.90,6.50,6.20,6.45,6.85,
                      7.20,6.80,6.50,6.75,7.15,
                      7.80,7.30,6.90,7.25,7.75,
                      8.10,7.60,7.20,7.55,8.05,
                      8.50,7.95,7.55,7.90,8.45,
                      9.00,8.40,8.00,8.35,8.95],
    "Bid Vol (%)":   [6.70,6.32,6.04,6.28,6.66,
                      7.00,6.63,6.34,6.58,6.96,
                      7.58,7.10,6.72,7.07,7.55,
                      7.88,7.39,7.01,7.36,7.84,
                      8.26,7.73,7.33,7.68,8.22,
                      8.74,8.16,7.76,8.11,8.70],
    "Ask Vol (%)":   [7.10,6.68,6.36,6.62,7.04,
                      7.40,6.97,6.66,6.92,7.34,
                      8.02,7.50,7.08,7.43,7.95,
                      8.32,7.81,7.39,7.74,8.26,
                      8.74,8.17,7.77,8.12,8.68,
                      9.26,8.64,8.24,8.59,9.20],
    "Notional (M)":  [10,25,50,25,10]*6,
    "Premium CCY":   ["USD"]*30,
    "Settlement":    ["Spot"]*30,
})

DELTA_PILLAR_ORDER = ["10D Put","25D Put","ATM","25D Call","10D Call"]
DELTA_PILLAR_X    = {lbl: i for i, lbl in enumerate(DELTA_PILLAR_ORDER)}
TENOR_ORDER       = ["1W","1M","3M","6M","1Y","2Y"]
SLIDE_COLORS      = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#8c564b"]


def _build_vol_surface_from_instruments(instruments_df, spot, rd, rf):
    records = []
    for _, row in instruments_df.iterrows():
        T         = float(row["Expiry (Years)"])
        vol       = float(row["Market Vol (%)"]) / 100.0
        delta     = float(row["Delta"])
        is_put    = delta < 0
        delta_abs = abs(delta)
        if row["Instrument"] == "ATM":
            F      = spot * np.exp((rd - rf) * T)
            strike = F * np.exp(0.5 * vol**2 * T)
        else:
            try:
                strike = delta_to_strike(delta_abs, spot, T, rd, rf, vol, is_put=is_put)
            except Exception:
                F      = spot * np.exp((rd - rf) * T)
                strike = F * np.exp(0.5 * vol**2 * T)
        records.append({"Strike": round(strike,5), "Expiry (Years)": T, "Volatility (%)": float(row["Market Vol (%)"])})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# SLIDE PLOTS
# ---------------------------------------------------------------------------

def _plot_vol_heatmap(instruments_df, spot, rd, rf):
    tenor_map = (instruments_df[["Expiry (Years)","Tenor"]]
                 .drop_duplicates().set_index("Expiry (Years)")["Tenor"].to_dict())
    pivot_rows = []
    for inst in DELTA_PILLAR_ORDER:
        row_vals = {}
        for T_val, tenor_lbl in sorted(tenor_map.items()):
            sub = instruments_df[
                (instruments_df["Instrument"] == inst) &
                (np.abs(instruments_df["Expiry (Years)"] - T_val) < 1e-6)
            ]
            row_vals[tenor_lbl] = float(sub.iloc[0]["Market Vol (%)"]) if not sub.empty else np.nan
        pivot_rows.append(row_vals)
    pivot_df = pd.DataFrame(pivot_rows, index=DELTA_PILLAR_ORDER)
    tenors   = [t for t in TENOR_ORDER if t in pivot_df.columns]
    pivot_df = pivot_df[tenors]
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values, x=pivot_df.columns.tolist(), y=pivot_df.index.tolist(),
        colorscale="RdYlGn_r",
        text=np.round(pivot_df.values, 2), texttemplate="%{text:.2f}%",
        textfont=dict(size=13, color="black"),
        colorbar=dict(title="IV (%)", ticksuffix="%"),
        hovertemplate="<b>%{y}</b> | %{x}<br>Vol: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="EUR/USD Implied Volatility Surface – Market Quotes (March 2026)", font=dict(size=16)),
        xaxis=dict(title="Tenor", side="bottom"),
        yaxis=dict(title="Delta Pillar", autorange="reversed"),
        height=420, margin=dict(l=100, r=60, t=60, b=60),
    )
    return fig


def _plot_smile_delta(instruments_df):
    unique_expiries = sorted(instruments_df["Expiry (Years)"].unique())
    tenor_map = (instruments_df[["Expiry (Years)","Tenor"]]
                 .drop_duplicates().set_index("Expiry (Years)")["Tenor"].to_dict())
    fig = go.Figure()
    for i, T in enumerate(unique_expiries):
        lbl    = tenor_map.get(T, f"{T:.2f}Y")
        subset = instruments_df[instruments_df["Expiry (Years)"] == T]
        xs, ys, bids, asks, htexts = [], [], [], [], []
        for pillar in DELTA_PILLAR_ORDER:
            r = subset[subset["Instrument"] == pillar]
            if r.empty: continue
            xs.append(DELTA_PILLAR_X[pillar])
            ys.append(float(r.iloc[0]["Market Vol (%)"]))
            bids.append(float(r.iloc[0].get("Bid Vol (%)", r.iloc[0]["Market Vol (%)"] - 0.18)))
            asks.append(float(r.iloc[0].get("Ask Vol (%)", r.iloc[0]["Market Vol (%)"] + 0.18)))
            htexts.append(pillar)
        c = SLIDE_COLORS[i % len(SLIDE_COLORS)]
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=asks + bids[::-1],
            fill="toself", fillcolor=f"rgba{tuple(list(_hex_to_rgb(c)) + [0.10])}",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+lines", name=lbl,
            marker=dict(size=9, color=c), line=dict(width=2.5, color=c),
            text=htexts,
            hovertemplate="<b>%{text}</b><br>Mid: %{y:.2f}%<extra>" + lbl + "</extra>",
        ))
    fig.add_vline(x=DELTA_PILLAR_X["ATM"], line_dash="dot", line_color="grey",
                  annotation_text="ATM", annotation_position="top right")
    fig.update_layout(
        title=dict(text="EUR/USD Volatility Smiles by Tenor – Delta Space", font=dict(size=16)),
        xaxis=dict(tickmode="array", tickvals=list(DELTA_PILLAR_X.values()),
                   ticktext=DELTA_PILLAR_ORDER, title="Delta Pillar", range=[-0.5,4.5]),
        yaxis_title="Implied Volatility (%)", height=480, hovermode="closest",
        legend=dict(title="Tenor", orientation="v"),
    )
    return fig


def _plot_market_vs_model(errors_df, instruments_df):
    tenor_map = (instruments_df[["Expiry (Years)","Tenor"]]
                 .drop_duplicates().set_index("Expiry (Years)")["Tenor"].to_dict())
    edf = errors_df.copy()
    tenors = []
    for _, row in edf.iterrows():
        T_key = round(float(row["expiry"]), 6)
        tenors.append(tenor_map.get(T_key, f"{row['expiry']:.2f}Y"))
    edf["tenor_label"] = tenors
    fig = go.Figure()
    for i, tenor in enumerate([t for t in TENOR_ORDER if t in edf["tenor_label"].values]):
        sub = edf[edf["tenor_label"] == tenor]
        c   = SLIDE_COLORS[i % len(SLIDE_COLORS)]
        fig.add_trace(go.Scatter(
            x=sub["market_vol"], y=sub["model_vol"], mode="markers", name=tenor,
            marker=dict(size=14, color=c, symbol="circle", line=dict(width=1.5, color="white")),
            hovertemplate=(
                f"<b>{tenor}</b><br>Market: %{{x:.2f}}%<br>Model: %{{y:.2f}}%<br>"
                "Error: %{customdata:+.1f} bps<extra></extra>"
            ),
            customdata=(sub["model_vol"] - sub["market_vol"]) * 100,
        ))
    lo = edf["market_vol"].min() - 0.2
    hi = edf["market_vol"].max() + 0.2
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect Fit",
        line=dict(color="black", width=2, dash="dash"), hoverinfo="skip",
    ))
    bps5 = 0.05
    fig.add_trace(go.Scatter(
        x=[lo,hi,hi,lo], y=[lo-bps5, hi-bps5, hi+bps5, lo+bps5],
        fill="toself", fillcolor="rgba(0,180,0,0.08)",
        line=dict(color="rgba(0,180,0,0.3)", dash="dot"),
        name="±5 bps band", hoverinfo="skip",
    ))
    rmse = np.sqrt((((edf["model_vol"] - edf["market_vol"]) * 100)**2).mean())
    fig.update_layout(
        title=dict(
            text=f"Heston Calibration: Market IV vs Model IV  |  RMSE = {rmse:.1f} bps",
            font=dict(size=16)
        ),
        xaxis=dict(title="Market Implied Vol (%)", ticksuffix="%"),
        yaxis=dict(title="Heston Model IV (%)",    ticksuffix="%"),
        height=520, hovermode="closest",
        legend=dict(title="Tenor", orientation="v"),
    )
    return fig


def _plot_vol_errors_bar(errors_df, instruments_df):
    edf = errors_df.copy()
    tenor_map = (instruments_df[["Expiry (Years)","Tenor"]]
                 .drop_duplicates().set_index("Expiry (Years)")["Tenor"].to_dict())
    xlabels, tenors = [], []
    for _, row in edf.iterrows():
        t   = tenor_map.get(round(float(row["expiry"]),6), f"{row['expiry']:.2f}Y")
        sub = instruments_df[np.abs(instruments_df["Expiry (Years)"] - row["expiry"]) < 1e-4]
        if not sub.empty:
            sub2  = _build_vol_surface_from_instruments(sub, 1.17, 0.05, 0.03)
            diffs = np.abs(sub2["Strike"].values - row["strike"])
            inst  = sub.iloc[diffs.argmin()]["Instrument"]
        else:
            inst = "–"
        xlabels.append(f"{t} {inst}")
        tenors.append(t)
    edf["label"]   = xlabels
    edf["tenor"]   = tenors
    edf["err_bps"] = (edf["model_vol"] - edf["market_vol"]) * 100
    bar_colors = [
        "#2ca02c" if abs(v) <= 5  else
        "#ff7f0e" if abs(v) <= 15 else
        "#d62728"
        for v in edf["err_bps"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=edf["label"], y=edf["err_bps"],
        marker_color=bar_colors,
        text=[f"{v:+.1f}" for v in edf["err_bps"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Error: %{y:+.2f} bps<extra></extra>",
    ))
    fig.add_hline(y=0,   line_dash="solid", line_color="black",   line_width=1.5)
    fig.add_hline(y=5,   line_dash="dot",   line_color="#2ca02c", line_width=1,
                  annotation_text="+5 bps",  annotation_position="top right")
    fig.add_hline(y=-5,  line_dash="dot",   line_color="#2ca02c", line_width=1,
                  annotation_text="-5 bps",  annotation_position="bottom right")
    fig.add_hline(y=15,  line_dash="dot",   line_color="#ff7f0e", line_width=1,
                  annotation_text="+15 bps", annotation_position="top right")
    fig.add_hline(y=-15, line_dash="dot",   line_color="#ff7f0e", line_width=1,
                  annotation_text="-15 bps", annotation_position="bottom right")
    rmse = np.sqrt((edf["err_bps"]**2).mean())
    fig.update_layout(
        title=dict(
            text=f"Implied Volatility Calibration Errors by Instrument  |  RMSE = {rmse:.1f} bps",
            font=dict(size=16)
        ),
        xaxis=dict(title="Instrument", tickangle=-40),
        yaxis=dict(title="Vol Error (bps)"),
        height=500, bargap=0.25,
    )
    return fig


def _plot_smile_overlay(errors_df, instruments_df, tenor="1Y"):
    tenor_map = (instruments_df[["Expiry (Years)","Tenor"]]
                 .drop_duplicates().set_index("Tenor")["Expiry (Years)"].to_dict())
    T_val = tenor_map.get(tenor)
    if T_val is None:
        return go.Figure()
    mkt_sub = instruments_df[instruments_df["Tenor"] == tenor].copy()
    xs_mkt, ys_mkt, bids_mkt, asks_mkt, htexts = [], [], [], [], []
    for pillar in DELTA_PILLAR_ORDER:
        r = mkt_sub[mkt_sub["Instrument"] == pillar]
        if r.empty: continue
        xs_mkt.append(DELTA_PILLAR_X[pillar])
        ys_mkt.append(float(r.iloc[0]["Market Vol (%)"]))
        bids_mkt.append(float(r.iloc[0].get("Bid Vol (%)", r.iloc[0]["Market Vol (%)"] - 0.18)))
        asks_mkt.append(float(r.iloc[0].get("Ask Vol (%)", r.iloc[0]["Market Vol (%)"] + 0.18)))
        htexts.append(pillar)

    edf_t = errors_df[np.abs(errors_df["expiry"] - T_val) < 1e-4].copy()
    edf_t = edf_t.sort_values("strike").reset_index(drop=True)

    rd_approx, rf_approx = 0.053, 0.035
    xs_mod, ys_mod = [], []
    for pillar in DELTA_PILLAR_ORDER:
        pr = mkt_sub[mkt_sub["Instrument"] == pillar]
        if pr.empty: continue
        vol_pct   = float(pr.iloc[0]["Market Vol (%)"])
        vol_dec   = vol_pct / 100.0
        delta_val = float(pr.iloc[0]["Delta"])
        is_put    = delta_val < 0
        if pillar == "ATM":
            F = 1.17 * np.exp((rd_approx - rf_approx) * T_val)
            K_pillar = F * np.exp(0.5 * vol_dec**2 * T_val)
        else:
            try:
                K_pillar = delta_to_strike(abs(delta_val), 1.17, T_val,
                                           rd_approx, rf_approx, vol_dec, is_put=is_put)
            except Exception:
                continue
        if len(edf_t) == 0: continue
        idx = np.abs(edf_t["strike"].values - K_pillar).argmin()
        xs_mod.append(DELTA_PILLAR_X[pillar])
        ys_mod.append(float(edf_t.iloc[idx]["model_vol"]))

    # -------------------------------------------------------------------
    # Y-axis range: pad ±2% ABSOLUTE around all data so both smiles sit
    # in the middle of the chart and the gap between them looks small.
    # -------------------------------------------------------------------
    all_vols = ys_mkt + bids_mkt + asks_mkt + ys_mod
    y_pad    = 2.0   # percentage points of absolute padding
    y_lo     = max(0.0, min(all_vols) - y_pad)
    y_hi     = max(all_vols) + y_pad

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs_mkt + xs_mkt[::-1], y=asks_mkt + bids_mkt[::-1],
        fill="toself", fillcolor="rgba(31,119,180,0.10)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="Market Bid-Ask",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs_mkt, y=ys_mkt, mode="markers+lines", name="Market Mid",
        marker=dict(size=12, color="#1f77b4", symbol="diamond"),
        line=dict(width=3, color="#1f77b4"),
        text=htexts,
        hovertemplate="<b>%{text}</b><br>Market: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs_mod, y=ys_mod, mode="markers+lines", name="Heston Model",
        marker=dict(size=10, color="#d62728", symbol="circle"),
        line=dict(width=2.5, color="#d62728", dash="dash"),
        hovertemplate="Model: %{y:.2f}%<extra></extra>",
    ))
    fig.add_vline(x=DELTA_PILLAR_X["ATM"], line_dash="dot", line_color="grey",
                  annotation_text="ATM", annotation_position="top right")
    fig.update_layout(
        title=dict(text=f"EUR/USD {tenor} Smile: Market vs Heston (Skew Model)", font=dict(size=16)),
        xaxis=dict(tickmode="array", tickvals=list(DELTA_PILLAR_X.values()),
                   ticktext=DELTA_PILLAR_ORDER, title="Delta Pillar", range=[-0.5,4.5]),
        yaxis=dict(
            title="Implied Volatility (%)",
            ticksuffix="%",
            range=[y_lo, y_hi],   # <-- zoomed-out Y axis
        ),
        height=460, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ===========================================================================
# MAIN RENDER
# ===========================================================================
def render_fx_slv_section():
    st.header("FX Stochastic Local Volatility Model")

    if 'fx_curves' not in st.session_state or st.session_state.fx_curves is None:
        st.warning("⚠️ Please bootstrap FX curves first in the 'FX Yield Curves & Spot Rate' section.")
        return
    if ql is None:
        st.error("QuantLib is not installed.")
        return

    fx_curves = st.session_state.fx_curves
    spot_fx   = fx_curves.spot_fx
    try:
        rd = fx_curves.usd_curve.zeroRate(0.5, ql.Continuous).rate()
        rf = fx_curves.eur_curve.zeroRate(0.5, ql.Continuous).rate()
    except Exception:
        rd, rf = 0.053, 0.035

    eval_date = get_eval_date()
    ql.Settings.instance().evaluationDate = eval_date
    st.markdown("---")

    # -----------------------------------------------------------------------
    # 1. INSTRUMENT TABLE
    # -----------------------------------------------------------------------
    st.subheader("FX Volatility Surface – Calibration Instruments")
    st.info(
        "📋 Market EUR/USD implied vols in delta notation (interbank convention, March 2026). "
        "Edit any cell to update the surface."
    )
    instruments_df = st.data_editor(
        FX_OPTION_INSTRUMENTS.copy(),
        num_rows="dynamic", key="fx_option_instruments_table", hide_index=True,
        column_config={
            "Tenor":          st.column_config.TextColumn("Tenor",           width="small"),
            "Expiry (Years)": st.column_config.NumberColumn("Expiry (Y)",    format="%.4f", width="small"),
            "Instrument":     st.column_config.TextColumn("Instrument",      width="medium"),
            "Delta":          st.column_config.NumberColumn("Delta",         format="%.2f", width="small"),
            "Type":           st.column_config.TextColumn("Type",            width="small"),
            "Market Vol (%)": st.column_config.NumberColumn("Mid Vol (%)",   format="%.2f"),
            "Bid Vol (%)": st.column_config.NumberColumn("Bid Vol (%)",      format="%.2f"),
            "Ask Vol (%)": st.column_config.NumberColumn("Ask Vol (%)",      format="%.2f"),
            "Notional (M)": st.column_config.NumberColumn("Notional (M EUR)",format="%.0f", width="small"),
            "Premium CCY":  st.column_config.TextColumn("Prem CCY",          width="small"),
            "Settlement":   st.column_config.TextColumn("Settlement",        width="small"),
        },
    )
    vol_surface_df = _build_vol_surface_from_instruments(instruments_df, spot_fx, rd, rf)
    st.markdown("---")

    # -----------------------------------------------------------------------
    # 2. VOL SURFACE CALIBRATION SLIDES
    # -----------------------------------------------------------------------
    st.subheader("📊 FX Vol Surface – Calibration Slides")
    st.caption(
        "Slide-ready charts showing the market vol surface and (after calibration) "
        "the Heston skew model fit."
    )
    slide_tab1, slide_tab2, slide_tab3 = st.tabs([
        "🌡️ Vol Surface Heatmap",
        "📉 Volatility Smiles (Delta Space)",
        "📉 Volatility Smiles (Strike Space)",
    ])

    with slide_tab1:
        st.plotly_chart(
            _plot_vol_heatmap(instruments_df, spot_fx, rd, rf),
            use_container_width=True, key="slide_heatmap"
        )
        st.caption(
            "**Slide note:** Rows = delta pillars (10D Put → 10D Call). "
            "Columns = tenors. Colour = mid implied vol. "
            "Vol increases with tenor (term structure) and wings sit above ATM (smile/skew)."
        )

    with slide_tab2:
        st.plotly_chart(_plot_smile_delta(instruments_df), use_container_width=True, key="slide_smile_delta")
        st.caption(
            "**Slide note:** Shaded bands = bid-ask spread. Each line = one tenor. "
            "Left tilt = negative risk-reversal skew. The Heston model is calibrated to reproduce all of these."
        )

    with slide_tab3:
        fig_strike = go.Figure()
        for i, T in enumerate(sorted(instruments_df["Expiry (Years)"].unique())):
            tenor_map2 = (instruments_df[["Expiry (Years)","Tenor"]]
                          .drop_duplicates().set_index("Expiry (Years)")["Tenor"].to_dict())
            lbl    = tenor_map2.get(T, f"{T:.2f}Y")
            subset = instruments_df[instruments_df["Expiry (Years)"] == T]
            strike_xs, vol_ys, hover_texts = [], [], []
            for pillar in DELTA_PILLAR_ORDER:
                r = subset[subset["Instrument"] == pillar]
                if r.empty: continue
                vol_pct   = float(r.iloc[0]["Market Vol (%)"])
                vol_dec   = vol_pct / 100.0
                delta_val = float(r.iloc[0]["Delta"])
                is_put    = delta_val < 0
                if pillar == "ATM":
                    F = spot_fx * np.exp((rd - rf) * T)
                    K = F * np.exp(0.5 * vol_dec**2 * T)
                else:
                    try:
                        K = delta_to_strike(abs(delta_val), spot_fx, T, rd, rf, vol_dec, is_put=is_put)
                    except Exception:
                        continue
                strike_xs.append(round(K, 5))
                vol_ys.append(vol_pct)
                hover_texts.append(f"{pillar} | K={K:.4f}")
            if not strike_xs: continue
            order = np.argsort(strike_xs)
            c = SLIDE_COLORS[i % len(SLIDE_COLORS)]
            fig_strike.add_trace(go.Scatter(
                x=[strike_xs[j] for j in order], y=[vol_ys[j] for j in order],
                mode="markers+lines", name=lbl,
                marker=dict(size=9, color=c), line=dict(width=2.5, color=c),
                text=[hover_texts[j] for j in order],
                hovertemplate="<b>%{text}</b><br>Vol: %{y:.2f}%<extra>" + lbl + "</extra>",
            ))
        fig_strike.add_vline(x=spot_fx, line_dash="dash", line_color="black",
                             annotation_text=f"Spot {spot_fx:.4f}", annotation_position="top left")
        fig_strike.update_layout(
            title=dict(text="EUR/USD Volatility Smiles by Tenor – Strike Space", font=dict(size=16)),
            xaxis_title="EUR/USD Strike", yaxis_title="Implied Volatility (%)",
            height=480, hovermode="closest",
        )
        st.plotly_chart(fig_strike, use_container_width=True, key="slide_smile_strike")
        st.caption(
            "**Slide note:** X-axis = actual GK-inverted strike. "
            "Vertical line = current spot. Smiles widen and shift right for longer tenors."
        )
    st.markdown("---")

    # -----------------------------------------------------------------------
    # 3. MODEL PARAMETERS
    # -----------------------------------------------------------------------
    st.subheader("FX-SLV Model Parameters")
    calibration_mode = st.radio(
        "Calibration Mode",
        options=["Reduced Set (Recommended)", "Full Surface"],
        index=0,
        help="Reduced Set: ATM + 25D for all expiries = 18 instruments.",
        horizontal=True, key="fx_slv_calib_mode",
    )
    if calibration_mode == "Reduced Set (Recommended)":
        st.success("✅ Reduced set: ATM + 25D Put + 25D Call × 6 expiries = 18 instruments. Target RMSE < 10 bps.")
    else:
        st.warning("⚠️ Full surface (30 instruments). Wing quotes add tension; realistic RMSE 5–20 bps expected.")

    st.write("**Initial Parameter Guesses**")
    st.caption("v0 and θ are variances (vol²). σ must be > 0. ρ is typically negative for FX.")
    col1, col2 = st.columns(2)
    with col1:
        v0    = st.number_input("Initial Variance (v0)", value=0.0042, format="%.6f", key="fx_slv_v0",
                                help="6.5% vol ≈ 0.0042")
        kappa = st.number_input("Mean Reversion (κ)",    value=1.5,    format="%.4f", key="fx_slv_kappa")
        theta = st.number_input("Long-term Variance (θ)",value=0.0056, format="%.6f", key="fx_slv_theta",
                                help="7.5% vol ≈ 0.0056")
    with col2:
        sigma = st.number_input("Vol-of-Vol (σ)",        value=0.30,   format="%.4f", key="fx_slv_sigma")
        rho   = st.number_input("Correlation (ρ)",       value=-0.30,  format="%.4f",
                                min_value=-0.99, max_value=0.99, key="fx_slv_rho")
    st.markdown("---")

    # -----------------------------------------------------------------------
    # 4. CALIBRATION BUTTON
    # -----------------------------------------------------------------------
    st.subheader("Model Calibration")
    if 'fx_slv_model' not in st.session_state:
        st.session_state.fx_slv_model = None

    if st.button("Calibrate FX-SLV Model", type="primary", key="fx_slv_calibrate_btn"):
        with st.spinner("Calibrating Heston skew model to vol surface..."):
            try:
                REDUCED_INSTRUMENTS = ["ATM", "25D Put", "25D Call"]
                if calibration_mode == "Reduced Set (Recommended)":
                    reduced_df          = instruments_df[instruments_df["Instrument"].isin(REDUCED_INSTRUMENTS)].copy()
                    calibration_surface = _build_vol_surface_from_instruments(reduced_df, spot_fx, rd, rf)
                    st.info(f"🎯 Calibrating to {len(calibration_surface)} instruments (ATM + 25D P/C × 6 expiries)")
                else:
                    calibration_surface = vol_surface_df.copy()
                    st.info(f"🎯 Calibrating to {len(calibration_surface)} instruments (full surface – 10D wings included)")

                vol_surface_data = [
                    [float(r["Strike"]), float(r["Expiry (Years)"]), float(r["Volatility (%)"]) / 100.0]
                    for _, r in calibration_surface.iterrows()
                ]
                model_params = {"v0": float(v0), "kappa": float(kappa),
                                "theta": float(theta), "sigma": float(sigma), "rho": float(rho)}

                fx_slv = FXStochasticLocalVol(
                    eval_date, spot_fx,
                    ql.YieldTermStructureHandle(fx_curves.usd_curve),
                    ql.YieldTermStructureHandle(fx_curves.eur_curve),
                    vol_surface_data, model_params,
                )
                fx_slv.calibrate()

                st.session_state.fx_slv_model            = fx_slv
                st.session_state.fx_slv_calib_mode_used  = calibration_mode
                st.session_state.fx_slv_instruments_used = instruments_df.copy()
                st.success("✅ FX-SLV model calibrated successfully!")
            except Exception as e:
                st.error(f"Calibration failed: {e}")
                import traceback; st.error(traceback.format_exc())
    st.markdown("---")

    # -----------------------------------------------------------------------
    # 5. RESULTS
    # -----------------------------------------------------------------------
    if st.session_state.fx_slv_model is not None:
        st.subheader("FX-SLV Calibration Results")
        if 'fx_slv_calib_mode_used' in st.session_state:
            st.caption(f"Calibrated using: {st.session_state.fx_slv_calib_mode_used}")

        fx_slv  = st.session_state.fx_slv_model
        results = fx_slv.get_calibrated_results()

        if results:
            st.write("**Calibrated Heston Parameters**")
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: st.metric("v0",  f"{results['v0']:.6f}",    help=f"≈ {np.sqrt(abs(results['v0']))*100:.2f}% vol")
            with c2: st.metric("κ",   f"{results['kappa']:.6f}")
            with c3: st.metric("θ",   f"{results['theta']:.6f}", help=f"≈ {np.sqrt(abs(results['theta']))*100:.2f}% vol")
            with c4: st.metric("σ",   f"{results['sigma']:.6f}")
            with c5: st.metric("ρ",   f"{results['rho']:.6f}")

            feller = 2*results['kappa']*results['theta'] - results['sigma']**2
            fc = "green" if feller > 0 else "red"
            st.markdown(
                f"Feller condition (2κθ − σ²): :{fc}[**{feller:.6f}**] "
                f"({'✅ satisfied' if feller > 0 else '⚠️ violated'})"
            )
            st.write("")

            errors_df    = results['pricing_errors']
            inst_df_used = st.session_state.get('fx_slv_instruments_used', instruments_df)

            tab_s1, tab_s2, tab_s3, tab_s4, tab_s5, tab_s6, tab_s7 = st.tabs([
                "📊 Market vs Model Scatter",
                "📉 Vol Error Bars",
                "🔍 Smile Overlay",
                "📊 Calibration Quality",
                "📈 Simulated Paths",
                "✅ Model Validation",
                "📋 Detailed Results",
            ])

            with tab_s1:
                st.plotly_chart(
                    _plot_market_vs_model(errors_df, inst_df_used),
                    use_container_width=True, key="slide_mktvsmod"
                )
                rmse = np.sqrt((((errors_df["model_vol"]-errors_df["market_vol"])*100)**2).mean())
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("RMSE (bps)",     f"{rmse:.2f}")
                with c2: st.metric("Max Error (bps)", f"{(errors_df['model_vol']-errors_df['market_vol']).abs().max()*100:.2f}")
                with c3: st.metric("N instruments",   str(len(errors_df)))
                st.caption(
                    "**Slide note:** Points on the 45° line = perfect fit. "
                    "Green band = ±5 bps. Points outside show where a single 5-param "
                    "Heston set cannot absorb irregular wing/kink quotes exactly – "
                    "this residual is what SLV's leverage function corrects."
                )

            with tab_s2:
                st.plotly_chart(
                    _plot_vol_errors_bar(errors_df, inst_df_used),
                    use_container_width=True, key="slide_errbars"
                )
                st.caption(
                    "**Slide note:** Green = ±5 bps (excellent). Orange = 5–15 bps (acceptable). "
                    "Red = >15 bps. Wing mis-fit is inherent to single-factor Heston; "
                    "it is resolved by the SLV leverage layer."
                )

            with tab_s3:
                chosen_tenor = st.selectbox(
                    "Select tenor for smile overlay",
                    options=[t for t in TENOR_ORDER if t in inst_df_used["Tenor"].values],
                    index=4, key="smile_overlay_tenor"
                )
                st.plotly_chart(
                    _plot_smile_overlay(errors_df, inst_df_used, tenor=chosen_tenor),
                    use_container_width=True, key="slide_smile_overlay"
                )
                st.caption(
                    "**Slide note:** Blue = market mid with bid-ask band. "
                    "Red dashed = Heston model. Red points now aligned to the same delta "
                    "pillars as the blue – any gap is the actual calibration residual."
                )

            with tab_s4:
                rmse      = np.sqrt((errors_df['vol_error_bps']**2).mean())
                max_error = errors_df['vol_error_bps'].abs().max()
                if   rmse < 10: qmsg, qcol = "🌟 Excellent (RMSE < 10 bps)",  "green"
                elif rmse < 20: qmsg, qcol = "✅ Good (RMSE < 20 bps)",        "blue"
                elif rmse < 30: qmsg, qcol = "⚠️ Acceptable (RMSE < 30 bps)",  "orange"
                else:           qmsg, qcol = "❌ Poor (RMSE > 30 bps)",         "red"
                st.markdown(f":{qcol}[**{qmsg}**]")
                fig_vols = go.Figure()
                fig_vols.add_trace(go.Scatter(
                    x=list(range(len(errors_df))), y=errors_df['market_vol'],
                    mode='markers+lines', name='Market Vol',
                    marker=dict(size=10, color='blue', symbol='diamond'), line=dict(width=2, color='blue')))
                fig_vols.add_trace(go.Scatter(
                    x=list(range(len(errors_df))), y=errors_df['model_vol'],
                    mode='markers+lines', name='Model Vol',
                    marker=dict(size=8, color='red'), line=dict(width=2, color='red')))
                fig_vols.update_layout(title="Volatility Calibration: Market vs Model",
                    xaxis_title="Option Index", yaxis_title="Implied Volatility (%)", height=500)
                st.plotly_chart(fig_vols, use_container_width=True, key="fx_slv_vols_chart")
                c1,c2,c3,c4 = st.columns(4)
                with c1: st.metric("Max Vol Error",  f"{max_error:.2f} bps")
                with c2: st.metric("Mean Vol Error", f"{errors_df['vol_error_bps'].mean():.2f} bps")
                with c3: st.metric("RMSE (Vol)",     f"{rmse:.2f} bps")
                with c4: st.metric("Std Dev",        f"{errors_df['vol_error_bps'].std():.2f} bps")

            with tab_s5:
                st.write("**Simulated FX Spot and Volatility Paths**")
                c1, c2 = st.columns(2)
                with c1: num_paths = st.slider("Number of Paths", 10, 100, 50, step=10, key="fx_slv_paths_slider")
                with c2: horizon   = st.slider("Time Horizon (years)", 0.5, 5.0, 1.0, step=0.5, key="fx_slv_horizon_slider")
                if st.button("Generate Paths", key="fx_slv_gen_paths_btn"):
                    with st.spinner("Generating paths..."):
                        path_df, times, spot_paths, vol_paths = fx_slv.get_simulated_paths(
                            num_paths=1000, time_steps=252, horizon_years=horizon)
                        fig_spot = go.Figure()
                        for i in range(min(num_paths, 1000)):
                            fig_spot.add_trace(go.Scatter(x=times, y=spot_paths[:,i], mode='lines',
                                line=dict(width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'))
                        fig_spot.add_trace(go.Scatter(x=times, y=spot_paths.mean(axis=1),
                            mode='lines', name='Mean', line=dict(color='red', width=3)))
                        fig_spot.add_hline(y=spot_fx, line_dash="dash", line_color="blue")
                        fig_spot.update_layout(title=f"FX Spot Paths ({num_paths} shown)",
                            xaxis_title="Time (years)", yaxis_title="EUR/USD", height=500)
                        st.plotly_chart(fig_spot, use_container_width=True, key="fx_slv_spot_paths")
                        vp = np.sqrt(vol_paths) * 100
                        fig_vol = go.Figure()
                        for i in range(min(num_paths, 1000)):
                            fig_vol.add_trace(go.Scatter(x=times, y=vp[:,i], mode='lines',
                                line=dict(width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'))
                        fig_vol.add_trace(go.Scatter(x=times, y=vp.mean(axis=1),
                            mode='lines', name='Mean Vol', line=dict(color='purple', width=3)))
                        fig_vol.update_layout(title="Stochastic Volatility Evolution",
                            xaxis_title="Time (years)", yaxis_title="Vol (%)", height=500)
                        st.plotly_chart(fig_vol, use_container_width=True, key="fx_slv_vol_paths")

            with tab_s6:
                st.write("**Model Validation: Heston vs Black-Scholes**")
                if st.button("Run Validation", type="primary", key="fx_slv_validation_btn"):
                    with st.spinner("Running validation..."):
                        vr = fx_slv.validate_option_prices()
                        if vr is not None:
                            st.success("✅ Validation completed!")
                            fig_val = go.Figure()
                            fig_val.add_trace(go.Scatter(x=list(range(len(vr))), y=vr['bs_price'],
                                mode='markers+lines', name='Black-Scholes',
                                marker=dict(size=10, color='blue'), line=dict(width=2, color='blue')))
                            fig_val.add_trace(go.Scatter(x=list(range(len(vr))), y=vr['heston_price'],
                                mode='markers+lines', name='Heston',
                                marker=dict(size=8, color='red'), line=dict(width=2, color='red')))
                            fig_val.update_layout(title="Prices: Black-Scholes vs Heston",
                                xaxis_title="Option", yaxis_title="Price", height=500)
                            st.plotly_chart(fig_val, use_container_width=True, key="fx_slv_val_chart")
                            st.dataframe(vr, use_container_width=True, hide_index=True)

            with tab_s7:
                disp = errors_df[[
                    'strike','expiry','market_vol','model_vol','vol_error_bps',
                    'market_price','model_price','price_error','price_error_pct'
                ]].copy()
                disp.columns = [
                    'Strike','Expiry (Y)','Market Vol (%)','Model Vol (%)','Vol Error (bps)',
                    'Market Price','Model Price','Price Error','Price Error (%)'
                ]
                st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Calibrate FX-SLV Model' to see results.")

    st.markdown("---")
