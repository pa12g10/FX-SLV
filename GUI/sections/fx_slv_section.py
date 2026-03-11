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
# Helper: convert delta + spot/rates/vol/T -> strike (Garman-Kohlhagen inversion)
# ---------------------------------------------------------------------------
def delta_to_strike(delta_abs, S, T, rd, rf, sigma, is_put=False):
    from scipy.stats import norm
    if is_put:
        Nd1 = 1.0 - delta_abs * np.exp(rf * T)
    else:
        Nd1 = delta_abs * np.exp(rf * T)
    Nd1 = np.clip(Nd1, 1e-8, 1 - 1e-8)
    d1  = norm.ppf(Nd1)
    return S * np.exp((rd - rf + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T))


# ---------------------------------------------------------------------------
# Realistic EUR/USD market data (March 2026 style)
# ---------------------------------------------------------------------------
FX_OPTION_INSTRUMENTS = pd.DataFrame({
    "Tenor":        ["1W",  "1W",  "1W",  "1W",  "1W",
                     "1M",  "1M",  "1M",  "1M",  "1M",
                     "3M",  "3M",  "3M",  "3M",  "3M",
                     "6M",  "6M",  "6M",  "6M",  "6M",
                     "1Y",  "1Y",  "1Y",  "1Y",  "1Y",
                     "2Y",  "2Y",  "2Y",  "2Y",  "2Y"],
    "Expiry (Years)":[1/52, 1/52, 1/52, 1/52, 1/52,
                      1/12, 1/12, 1/12, 1/12, 1/12,
                      0.25, 0.25, 0.25, 0.25, 0.25,
                      0.50, 0.50, 0.50, 0.50, 0.50,
                      1.00, 1.00, 1.00, 1.00, 1.00,
                      2.00, 2.00, 2.00, 2.00, 2.00],
    "Instrument":   ["10D Put","25D Put","ATM","25D Call","10D Call",
                     "10D Put","25D Put","ATM","25D Call","10D Call",
                     "10D Put","25D Put","ATM","25D Call","10D Call",
                     "10D Put","25D Put","ATM","25D Call","10D Call",
                     "10D Put","25D Put","ATM","25D Call","10D Call",
                     "10D Put","25D Put","ATM","25D Call","10D Call"],
    "Delta":        [-0.10,-0.25, 0.50, 0.25, 0.10,
                     -0.10,-0.25, 0.50, 0.25, 0.10,
                     -0.10,-0.25, 0.50, 0.25, 0.10,
                     -0.10,-0.25, 0.50, 0.25, 0.10,
                     -0.10,-0.25, 0.50, 0.25, 0.10,
                     -0.10,-0.25, 0.50, 0.25, 0.10],
    "Type":         ["Put","Put","Call","Call","Call",
                     "Put","Put","Call","Call","Call",
                     "Put","Put","Call","Call","Call",
                     "Put","Put","Call","Call","Call",
                     "Put","Put","Call","Call","Call",
                     "Put","Put","Call","Call","Call"],
    "Market Vol (%)": [6.90, 6.50, 6.20, 6.45, 6.85,
                       7.20, 6.80, 6.50, 6.75, 7.15,
                       7.80, 7.30, 6.90, 7.25, 7.75,
                       8.10, 7.60, 7.20, 7.55, 8.05,
                       8.50, 7.95, 7.55, 7.90, 8.45,
                       9.00, 8.40, 8.00, 8.35, 8.95],
    "Bid Vol (%)":    [6.70, 6.32, 6.04, 6.28, 6.66,
                       7.00, 6.63, 6.34, 6.58, 6.96,
                       7.58, 7.10, 6.72, 7.07, 7.55,
                       7.88, 7.39, 7.01, 7.36, 7.84,
                       8.26, 7.73, 7.33, 7.68, 8.22,
                       8.74, 8.16, 7.76, 8.11, 8.70],
    "Ask Vol (%)":    [7.10, 6.68, 6.36, 6.62, 7.04,
                       7.40, 6.97, 6.66, 6.92, 7.34,
                       8.02, 7.50, 7.08, 7.43, 7.95,
                       8.32, 7.81, 7.39, 7.74, 8.26,
                       8.74, 8.17, 7.77, 8.12, 8.68,
                       9.26, 8.64, 8.24, 8.59, 9.20],
    "Notional (M)":   [10, 25, 50, 25, 10,
                       10, 25, 50, 25, 10,
                       10, 25, 50, 25, 10,
                       10, 25, 50, 25, 10,
                       10, 25, 50, 25, 10,
                       10, 25, 50, 25, 10],
    "Premium CCY":    ["USD"]*30,
    "Settlement":     ["Spot"]*30,
})

DELTA_PILLAR_ORDER = ["10D Put", "25D Put", "ATM", "25D Call", "10D Call"]
DELTA_PILLAR_X    = {lbl: i for i, lbl in enumerate(DELTA_PILLAR_ORDER)}


def _build_vol_surface_from_instruments(instruments_df, spot, rd, rf):
    records = []
    for _, row in instruments_df.iterrows():
        T     = float(row["Expiry (Years)"])
        vol   = float(row["Market Vol (%)"]) / 100.0
        delta = float(row["Delta"])
        is_put = delta < 0
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
        records.append({
            "Strike":         round(strike, 5),
            "Expiry (Years)": T,
            "Volatility (%)":  float(row["Market Vol (%)"])
        })
    return pd.DataFrame(records)


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

    # FIX: use the canonical eval date from market_data, not a hardcoded stale date
    eval_date = get_eval_date()
    ql.Settings.instance().evaluationDate = eval_date

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 1. INSTRUMENT TABLE
    # -----------------------------------------------------------------------
    st.subheader("FX Volatility Surface – Calibration Instruments")
    st.info(
        "📋 Market instruments used for SLV calibration. Vols are mid-market EUR/USD "
        "implied volatilities in **delta notation** (interbank convention, March 2026). "
        "Edit any cell to update the calibration surface."
    )

    instruments_df = st.data_editor(
        FX_OPTION_INSTRUMENTS.copy(),
        num_rows="dynamic",
        key="fx_option_instruments_table",
        hide_index=True,
        column_config={
            "Tenor":           st.column_config.TextColumn("Tenor", width="small"),
            "Expiry (Years)":  st.column_config.NumberColumn("Expiry (Y)", format="%.4f", width="small"),
            "Instrument":      st.column_config.TextColumn("Instrument", width="medium"),
            "Delta":           st.column_config.NumberColumn("Delta", format="%.2f", width="small"),
            "Type":            st.column_config.TextColumn("Type", width="small"),
            "Market Vol (%)":  st.column_config.NumberColumn("Mid Vol (%)", format="%.2f"),
            "Bid Vol (%)":     st.column_config.NumberColumn("Bid Vol (%)", format="%.2f"),
            "Ask Vol (%)":     st.column_config.NumberColumn("Ask Vol (%)", format="%.2f"),
            "Notional (M)":    st.column_config.NumberColumn("Notional (M EUR)", format="%.0f", width="small"),
            "Premium CCY":     st.column_config.TextColumn("Prem CCY", width="small"),
            "Settlement":      st.column_config.TextColumn("Settlement", width="small"),
        },
    )

    vol_surface_df = _build_vol_surface_from_instruments(instruments_df, spot_fx, rd, rf)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 2. VOLATILITY SMILES
    # -----------------------------------------------------------------------
    st.subheader("Volatility Smiles by Expiry")

    if st.checkbox("Show Volatility Smile Plots", value=True, key="fx_smile_plot_check"):
        unique_expiries = sorted(instruments_df["Expiry (Years)"].unique())
        tenor_map = (
            instruments_df[["Expiry (Years)", "Tenor"]]
            .drop_duplicates().set_index("Expiry (Years)")["Tenor"].to_dict()
        )
        colors = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#8c564b"]

        st.write("**Panel 1 – Delta Space**")
        fig_delta = go.Figure()
        for i, T in enumerate(unique_expiries):
            tenor_label = tenor_map.get(T, f"{T:.2f}Y")
            subset = instruments_df[instruments_df["Expiry (Years)"] == T].copy()
            xs, ys, htexts = [], [], []
            for lbl in DELTA_PILLAR_ORDER:
                row = subset[subset["Instrument"] == lbl]
                if row.empty: continue
                xs.append(DELTA_PILLAR_X[lbl])
                ys.append(float(row.iloc[0]["Market Vol (%)"]))
                htexts.append(lbl)
            c = colors[i % len(colors)]
            fig_delta.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+lines", name=tenor_label,
                marker=dict(size=9, color=c), line=dict(width=2, color=c),
                text=htexts,
                hovertemplate="<b>%{text}</b><br>Vol: %{y:.2f}%<extra>" + tenor_label + "</extra>",
            ))
        fig_delta.update_layout(
            title="EUR/USD Volatility Smile – Delta Space",
            xaxis=dict(tickmode="array", tickvals=list(DELTA_PILLAR_X.values()),
                       ticktext=DELTA_PILLAR_ORDER, title="Delta Pillar", range=[-0.5, 4.5]),
            yaxis_title="Implied Volatility (%)", height=480, hovermode="closest",
        )
        fig_delta.add_vline(x=DELTA_PILLAR_X["ATM"], line_dash="dot", line_color="grey",
                            annotation_text="ATM", annotation_position="top right")
        st.plotly_chart(fig_delta, use_container_width=True, key="fx_smile_delta")

        st.write("**Panel 2 – Strike Space**")
        fig_strike = go.Figure()
        for i, T in enumerate(unique_expiries):
            tenor_label = tenor_map.get(T, f"{T:.2f}Y")
            subset = instruments_df[instruments_df["Expiry (Years)"] == T].copy()
            strike_xs, vol_ys, hover_texts = [], [], []
            for lbl in DELTA_PILLAR_ORDER:
                row = subset[subset["Instrument"] == lbl]
                if row.empty: continue
                vol_pct   = float(row.iloc[0]["Market Vol (%)"])
                vol_dec   = vol_pct / 100.0
                delta_val = float(row.iloc[0]["Delta"])
                is_put    = delta_val < 0
                if lbl == "ATM":
                    F = spot_fx * np.exp((rd - rf) * T)
                    K = F * np.exp(0.5 * vol_dec**2 * T)
                else:
                    try:
                        K = delta_to_strike(abs(delta_val), spot_fx, T, rd, rf, vol_dec, is_put=is_put)
                    except Exception:
                        continue
                strike_xs.append(round(K, 5))
                vol_ys.append(vol_pct)
                hover_texts.append(f"{lbl} | K={K:.4f}")
            if not strike_xs: continue
            order = np.argsort(strike_xs)
            c = colors[i % len(colors)]
            fig_strike.add_trace(go.Scatter(
                x=[strike_xs[j] for j in order], y=[vol_ys[j] for j in order],
                mode="markers+lines", name=tenor_label,
                marker=dict(size=9, color=c), line=dict(width=2, color=c),
                text=[hover_texts[j] for j in order],
                hovertemplate="<b>%{text}</b><br>Vol: %{y:.2f}%<extra>" + tenor_label + "</extra>",
            ))
        fig_strike.add_vline(x=spot_fx, line_dash="dash", line_color="black",
                             annotation_text=f"Spot {spot_fx:.4f}", annotation_position="top left")
        fig_strike.update_layout(
            title="EUR/USD Volatility Smile – Strike Space",
            xaxis_title="EUR/USD Strike", yaxis_title="Implied Volatility (%)",
            height=480, hovermode="closest",
        )
        st.plotly_chart(fig_strike, use_container_width=True, key="fx_smile_strike")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Model Configuration
    # -----------------------------------------------------------------------
    st.subheader("FX-SLV Model Parameters")

    st.write("**Calibration Settings**")
    calibration_mode = st.radio(
        "Calibration Mode",
        options=["Reduced Set (Recommended)", "Full Surface"],
        index=0,
        help="Reduced Set: ATM + 25D for all expiries (well-conditioned). Full: all 30 instruments.",
        horizontal=True,
        key="fx_slv_calib_mode",
    )

    if calibration_mode == "Reduced Set (Recommended)":
        st.success("✅ Reduced set: ATM + 25D Put + 25D Call per expiry = 18 instruments. Target RMSE < 10 bps")
    else:
        st.warning("⚠️ Full surface (30 instruments). Heston cannot perfectly fit all wings — RMSE 20-40 bps is normal.")

    st.write("**Initial Parameter Guesses**")
    st.caption(
        "v0 and θ are variances: to get ~7% initial vol enter 0.0049 (= 0.07²). "
        "σ (vol-of-vol) must be > 0. ρ is typically negative for FX."
    )
    col1, col2 = st.columns(2)
    with col1:
        # FIX: defaults changed to match ATM vol level (~6.5% => v0=0.0042)
        v0    = st.number_input("Initial Variance (v0)",  value=0.0042, format="%.6f", key="fx_slv_v0",
                                help="Variance = vol². 6.5% vol => 0.0042")
        kappa = st.number_input("Mean Reversion (κ)",      value=1.5,    format="%.4f", key="fx_slv_kappa")
        theta = st.number_input("Long-term Variance (θ)",  value=0.0056, format="%.6f", key="fx_slv_theta",
                                help="Long-run variance. 7.5% vol => 0.0056")
    with col2:
        sigma = st.number_input("Vol-of-Vol (σ)",          value=0.30,   format="%.4f", key="fx_slv_sigma",
                                help="Must be > 0. Typical FX range: 0.15 – 0.60")
        rho   = st.number_input("Correlation (ρ)",         value=-0.30,  format="%.4f",
                                min_value=-0.99, max_value=0.99, key="fx_slv_rho",
                                help="Typically -0.1 to -0.4 for EUR/USD")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------
    st.subheader("Model Calibration")

    if 'fx_slv_model' not in st.session_state:
        st.session_state.fx_slv_model = None

    if st.button("Calibrate FX-SLV Model", type="primary", key="fx_slv_calibrate_btn"):
        with st.spinner("Calibrating FX-SLV model to volatility surface..."):
            try:
                if calibration_mode == "Reduced Set (Recommended)":
                    # FIX: use ATM + 25D Put + 25D Call for ALL expiries
                    # This gives 18 well-conditioned helpers vs the old 10
                    # and covers both ATM and skew information.
                    REDUCED_INSTRUMENTS = ["ATM", "25D Put", "25D Call"]
                    reduced_df = instruments_df[
                        instruments_df["Instrument"].isin(REDUCED_INSTRUMENTS)
                    ].copy()
                    calibration_surface = _build_vol_surface_from_instruments(
                        reduced_df, spot_fx, rd, rf
                    )
                    st.info(f"🎯 Calibrating to {len(calibration_surface)} instruments "
                            f"(ATM + 25D Put + 25D Call × 6 expiries)")
                else:
                    calibration_surface = vol_surface_df.copy()
                    st.info(f"🎯 Calibrating to {len(calibration_surface)} instruments (full surface)")

                # Vols are stored as % in the DataFrame; divide by 100 for the model
                vol_surface_data = [
                    [float(r["Strike"]),
                     float(r["Expiry (Years)"]),
                     float(r["Volatility (%)"]) / 100.0]
                    for _, r in calibration_surface.iterrows()
                ]

                model_params = {
                    "v0":    float(v0),
                    "kappa": float(kappa),
                    "theta": float(theta),
                    "sigma": float(sigma),
                    "rho":   float(rho),
                }

                domestic_curve_handle = ql.YieldTermStructureHandle(fx_curves.usd_curve)
                foreign_curve_handle  = ql.YieldTermStructureHandle(fx_curves.eur_curve)

                fx_slv = FXStochasticLocalVol(
                    eval_date, spot_fx,
                    domestic_curve_handle, foreign_curve_handle,
                    vol_surface_data, model_params,
                )
                fx_slv.calibrate()

                st.session_state.fx_slv_model            = fx_slv
                st.session_state.fx_slv_calib_mode_used  = calibration_mode
                st.session_state.fx_slv_instruments_used = instruments_df.copy()
                st.success("✅ FX-SLV model calibrated successfully!")

            except Exception as e:
                st.error(f"Calibration failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    if st.session_state.fx_slv_model is not None:
        st.subheader("FX-SLV Calibration Results")

        if 'fx_slv_calib_mode_used' in st.session_state:
            st.caption(f"Calibrated using: {st.session_state.fx_slv_calib_mode_used}")

        fx_slv  = st.session_state.fx_slv_model
        results = fx_slv.get_calibrated_results()

        if results:
            st.write("**Calibrated Heston Parameters**")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("v0",  f"{results['v0']:.6f}",    help=f"≈ {np.sqrt(abs(results['v0']))*100:.2f}% vol")
            with col2: st.metric("κ",   f"{results['kappa']:.6f}")
            with col3: st.metric("θ",   f"{results['theta']:.6f}", help=f"≈ {np.sqrt(abs(results['theta']))*100:.2f}% vol")
            with col4: st.metric("σ",   f"{results['sigma']:.6f}")
            with col5: st.metric("ρ",   f"{results['rho']:.6f}")

            feller = 2 * results['kappa'] * results['theta'] - results['sigma']**2
            feller_color = "green" if feller > 0 else "red"
            st.markdown(f"Feller condition (2κθ − σ²): :{feller_color}[**{feller:.6f}**] "
                        f"({'✅ satisfied' if feller > 0 else '⚠️ violated – variance can hit 0'})")
            st.write("")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Calibration Quality",
                "⚖️ Market vs Model Prices",
                "📈 Simulated Paths",
                "✅ Model Validation",
                "📋 Detailed Results",
            ])

            # -----------------------------------------------------------
            with tab1:
                errors_df = results['pricing_errors']
                rmse      = np.sqrt((errors_df['vol_error_bps']**2).mean())
                max_error = errors_df['vol_error_bps'].abs().max()

                if   rmse < 10: quality_msg, quality_color = "🌟 Excellent (RMSE < 10 bps)",  "green"
                elif rmse < 20: quality_msg, quality_color = "✅ Good (RMSE < 20 bps)",        "blue"
                elif rmse < 30: quality_msg, quality_color = "⚠️ Acceptable (RMSE < 30 bps)",  "orange"
                else:           quality_msg, quality_color = "❌ Poor (RMSE > 30 bps)",         "red"

                st.markdown(f":{quality_color}[**{quality_msg}**]")
                st.write("")

                fig_vols = go.Figure()
                fig_vols.add_trace(go.Scatter(
                    x=list(range(len(errors_df))), y=errors_df['market_vol'],
                    mode='markers+lines', name='Market Vol',
                    marker=dict(size=10, color='blue', symbol='diamond'),
                    line=dict(width=2, color='blue'),
                ))
                fig_vols.add_trace(go.Scatter(
                    x=list(range(len(errors_df))), y=errors_df['model_vol'],
                    mode='markers+lines', name='Model Vol',
                    marker=dict(size=8, color='red', symbol='circle'),
                    line=dict(width=2, color='red'),
                ))
                fig_vols.update_layout(
                    title="Volatility Calibration: Market vs Model",
                    xaxis_title="Option Index", yaxis_title="Implied Volatility (%)",
                    height=500, hovermode='closest'
                )
                st.plotly_chart(fig_vols, use_container_width=True, key="fx_slv_vols_chart")

                fig_vol_errors = go.Figure()
                fig_vol_errors.add_trace(go.Bar(
                    x=[f"K={r['strike']:.4f}, T={r['expiry']:.2f}Y" for _, r in errors_df.iterrows()],
                    y=errors_df['vol_error_bps'], marker_color='orange',
                ))
                fig_vol_errors.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
                fig_vol_errors.update_layout(
                    title="Implied Volatility Errors (Model − Market)",
                    xaxis_title="Option", yaxis_title="Error (bps)",
                    height=500, xaxis_tickangle=-45
                )
                st.plotly_chart(fig_vol_errors, use_container_width=True, key="fx_slv_vol_errors_chart")

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Max Vol Error",  f"{max_error:.2f} bps")
                with col2: st.metric("Mean Vol Error", f"{errors_df['vol_error_bps'].mean():.2f} bps")
                with col3: st.metric("RMSE (Vol)",     f"{rmse:.2f} bps")
                with col4: st.metric("Std Dev",        f"{errors_df['vol_error_bps'].std():.2f} bps")

            # -----------------------------------------------------------
            with tab2:
                st.write("**Market vs Model Option Prices**")
                errors_df    = results['pricing_errors']
                inst_df_used = st.session_state.get('fx_slv_instruments_used', instruments_df)

                labels, tenor_labels = [], []
                for _, row in errors_df.iterrows():
                    T_row = row['expiry']
                    match = inst_df_used[np.abs(inst_df_used['Expiry (Years)'] - T_row) < 1e-4]
                    if not match.empty:
                        br = match.iloc[0]
                        labels.append(f"{br['Tenor']} {br['Instrument']}")
                        tenor_labels.append(br['Tenor'])
                    else:
                        labels.append(f"K={row['strike']:.4f} T={T_row:.2f}Y")
                        tenor_labels.append(f"{T_row:.2f}Y")

                errors_df = errors_df.copy()
                errors_df['label']       = labels
                errors_df['tenor_label'] = tenor_labels

                unique_tenors = errors_df['tenor_label'].unique().tolist()
                colors_tab    = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#8c564b","#e377c2"]

                fig_mv = go.Figure()
                for ti, tenor in enumerate(unique_tenors):
                    sub = errors_df[errors_df['tenor_label'] == tenor].sort_values('market_price')
                    c   = colors_tab[ti % len(colors_tab)]
                    fig_mv.add_trace(go.Scatter(
                        x=sub['market_price'], y=sub['model_price'],
                        mode='lines+markers', name=tenor,
                        marker=dict(size=9, color=c, line=dict(width=1, color='white')),
                        line=dict(width=2, color=c),
                        text=sub['label'],
                        hovertemplate=("<b>%{text}</b><br>Market: %{x:.6f}<br>Model: %{y:.6f}<br>"
                                       "Error: %{customdata:.2f}%<extra></extra>"),
                        customdata=sub['price_error_pct'],
                    ))

                all_prices = pd.concat([errors_df['market_price'], errors_df['model_price']])
                p_min, p_max = all_prices.min(), all_prices.max()
                pad      = (p_max - p_min) * 0.05
                ref_line = [p_min - pad, p_max + pad]
                fig_mv.add_trace(go.Scatter(
                    x=ref_line, y=ref_line, mode='lines', name='Perfect Fit (45°)',
                    line=dict(color='black', width=2, dash='dash'), hoverinfo='skip',
                ))
                fig_mv.add_trace(go.Scatter(
                    x=ref_line + ref_line[::-1],
                    y=[v*1.02 for v in ref_line] + [v*0.98 for v in ref_line[::-1]],
                    fill='toself', fillcolor='rgba(0,200,0,0.08)',
                    line=dict(color='rgba(0,200,0,0.3)', dash='dot'),
                    name='±2% Tolerance', hoverinfo='skip',
                ))
                fig_mv.update_layout(
                    title="Market Price vs Model Price – Arbitrage-Free Check",
                    xaxis_title="Market Price (USD)", yaxis_title="Model Price (USD)",
                    height=560, hovermode='closest', legend=dict(title="Tenor"),
                )
                st.plotly_chart(fig_mv, use_container_width=True, key="fx_mv_scatter")

                bar_colors = ["#2ca02c" if abs(v) <= 2.0 else "#d62728"
                              for v in errors_df['price_error_pct']]
                fig_price_err = go.Figure()
                fig_price_err.add_trace(go.Bar(
                    x=errors_df['label'], y=errors_df['price_error_pct'],
                    marker_color=bar_colors,
                    text=[f"{v:+.2f}%" for v in errors_df['price_error_pct']],
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>Error: %{y:+.2f}%<extra></extra>",
                ))
                fig_price_err.add_hline(y=0,    line_dash="solid", line_color="black",   line_width=1.5)
                fig_price_err.add_hline(y=2.0,  line_dash="dot",   line_color="#2ca02c", line_width=1,
                                        annotation_text="+2%", annotation_position="right")
                fig_price_err.add_hline(y=-2.0, line_dash="dot",   line_color="#2ca02c", line_width=1,
                                        annotation_text="−2%", annotation_position="right")
                fig_price_err.update_layout(
                    title="Option Price Errors (Model − Market)",
                    xaxis_title="Instrument", yaxis_title="Price Error (%)",
                    height=480, xaxis_tickangle=-45, bargap=0.25,
                )
                st.plotly_chart(fig_price_err, use_container_width=True, key="fx_price_err_bar")

                n_outside = (errors_df['price_error_pct'].abs() > 2.0).sum()
                n_total   = len(errors_df)
                if n_outside == 0:
                    st.success(f"✅ All {n_total} instruments within ±2% tolerance.")
                else:
                    st.warning(f"⚠️ {n_outside}/{n_total} instruments outside ±2% tolerance.")

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Max Price Error",         f"{errors_df['price_error_pct'].abs().max():.2f}%")
                with col2: st.metric("Mean Price Error",        f"{errors_df['price_error_pct'].mean():.2f}%")
                with col3: st.metric("RMSE (Price %)",          f"{np.sqrt((errors_df['price_error_pct']**2).mean()):.2f}%")
                with col4: st.metric("Instruments Outside ±2%", f"{n_outside}/{n_total}")

            # -----------------------------------------------------------
            with tab3:
                st.write("**Simulated FX Spot and Volatility Paths**")
                col1, col2 = st.columns(2)
                with col1:
                    num_paths = st.slider("Number of Paths", 10, 100, 50, step=10, key="fx_slv_paths_slider")
                with col2:
                    horizon = st.slider("Time Horizon (years)", 0.5, 5.0, 1.0, step=0.5, key="fx_slv_horizon_slider")

                if st.button("Generate Paths", key="fx_slv_gen_paths_btn"):
                    with st.spinner("Generating Monte Carlo paths..."):
                        path_df, times, spot_paths, vol_paths = fx_slv.get_simulated_paths(
                            num_paths=1000, time_steps=252, horizon_years=horizon
                        )
                        fig_spot = go.Figure()
                        for i in range(min(num_paths, 1000)):
                            fig_spot.add_trace(go.Scatter(x=times, y=spot_paths[:, i], mode='lines',
                                line=dict(width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'))
                        fig_spot.add_trace(go.Scatter(x=times, y=spot_paths.mean(axis=1), mode='lines',
                            name='Mean Path', line=dict(color='red', width=3)))
                        fig_spot.add_hline(y=spot_fx, line_dash="dash", line_color="blue",
                                           annotation_text=f"Initial Spot: {spot_fx:.4f}")
                        fig_spot.update_layout(title=f"FX Spot Simulation ({num_paths} paths)",
                            xaxis_title="Time (years)", yaxis_title="FX Spot Rate", height=500)
                        st.plotly_chart(fig_spot, use_container_width=True, key="fx_slv_spot_paths")

                        vol_paths_pct = np.sqrt(vol_paths) * 100
                        fig_vol = go.Figure()
                        for i in range(min(num_paths, 1000)):
                            fig_vol.add_trace(go.Scatter(x=times, y=vol_paths_pct[:, i], mode='lines',
                                line=dict(width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'))
                        fig_vol.add_trace(go.Scatter(x=times, y=vol_paths_pct.mean(axis=1), mode='lines',
                            name='Mean Volatility', line=dict(color='purple', width=3)))
                        fig_vol.update_layout(title="Stochastic Volatility Evolution",
                            xaxis_title="Time (years)", yaxis_title="Volatility (%)", height=500)
                        st.plotly_chart(fig_vol, use_container_width=True, key="fx_slv_vol_paths")

            # -----------------------------------------------------------
            with tab4:
                st.write("**Model Validation: Heston vs Black-Scholes**")
                if st.button("Run Validation", type="primary", key="fx_slv_validation_btn"):
                    with st.spinner("Running validation..."):
                        validation_results = fx_slv.validate_option_prices()
                        if validation_results is not None:
                            st.success("✅ Validation completed!")
                            fig_val = go.Figure()
                            fig_val.add_trace(go.Scatter(
                                x=list(range(len(validation_results))), y=validation_results['bs_price'],
                                mode='markers+lines', name='Black-Scholes Price',
                                marker=dict(size=10, color='blue'), line=dict(width=2, color='blue')))
                            fig_val.add_trace(go.Scatter(
                                x=list(range(len(validation_results))), y=validation_results['heston_price'],
                                mode='markers+lines', name='Heston Price',
                                marker=dict(size=8, color='red'), line=dict(width=2, color='red')))
                            fig_val.update_layout(title="Option Prices: Black-Scholes vs Heston",
                                xaxis_title="Option Index", yaxis_title="Price", height=500)
                            st.plotly_chart(fig_val, use_container_width=True, key="fx_slv_val_chart")
                            st.dataframe(validation_results, use_container_width=True, hide_index=True)

            # -----------------------------------------------------------
            with tab5:
                display_errors = results['pricing_errors'][[
                    'strike', 'expiry',
                    'market_vol', 'model_vol', 'vol_error_bps',
                    'market_price', 'model_price', 'price_error', 'price_error_pct'
                ]].copy()
                display_errors.columns = [
                    'Strike', 'Expiry (Y)',
                    'Market Vol (%)', 'Model Vol (%)', 'Vol Error (bps)',
                    'Market Price', 'Model Price', 'Price Error', 'Price Error (%)'
                ]
                st.dataframe(display_errors, use_container_width=True, hide_index=True)

    else:
        st.info("Click 'Calibrate FX-SLV Model' to see results.")

    st.markdown("---")
