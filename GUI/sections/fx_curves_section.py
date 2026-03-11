# FX Curves Section
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from MarketData.market_data import (
    get_eval_date,
    get_sofr_deposit_data,
    get_sofr_futures_data,
    get_sofr_swaps_data,
    get_estr_deposit_data,
    get_estr_futures_data,
    get_estr_swaps_data,
    get_fx_spot,
    get_fx_swap_data,
    get_ccy_swaps_data,
)

from Models import FXCurves

try:
    import QuantLib as ql
except ImportError:
    ql = None

AXIS_STYLE = dict(
    title_font=dict(color='black'),
    tickfont=dict(color='black'),
    tickcolor='black',
)


def _rate_chart(df, x_col, y_col, name, color, title, x_title, y_title):
    """Helper: simple scatter+line chart."""
    fig = go.Figure(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode='lines+markers', name=name,
        line=dict(width=3, color=color), marker=dict(size=8)
    ))
    fig.update_layout(
        title=title, height=420, hovermode='x unified',
        xaxis=dict(title_text=x_title, **AXIS_STYLE),
        yaxis=dict(title_text=y_title, **AXIS_STYLE),
    )
    return fig


def render_fx_curves_section():
    st.header("FX Curves & Market Data")

    if ql is None:
        st.error("QuantLib is not installed.")
        return

    eval_date = get_eval_date()
    st.info(f"**Evaluation Date:** {eval_date.dayOfMonth()}/{eval_date.month()}/{eval_date.year()}  |  "
            f"**Spot EUR/USD:** {get_fx_spot()['rate']:.4f}")

    # ================================================================
    # SECTION 1 — DOMESTIC CURVE CONSTRUCTION (USD SOFR)
    # ================================================================
    st.markdown("---")
    st.subheader("1 — Domestic Curve Construction (USD SOFR)")

    dom_tab1, dom_tab2, dom_tab3 = st.tabs(["Deposit", "Futures", "OIS Swaps"])

    with dom_tab1:
        st.write("**SOFR Overnight Deposit**")
        st.dataframe(pd.DataFrame([get_sofr_deposit_data()]),
                     use_container_width=True, hide_index=True)

    with dom_tab2:
        st.write("**SOFR IMM Futures**")
        sofr_fut = get_sofr_futures_data()
        st.dataframe(sofr_fut, use_container_width=True, hide_index=True)
        st.plotly_chart(
            _rate_chart(sofr_fut, 'contract', 'rate', 'SOFR Futures', 'steelblue',
                        'SOFR Futures Implied Rates', 'Contract', 'Rate (%)'),
            use_container_width=True
        )

    with dom_tab3:
        st.write("**SOFR OIS Swaps (2Y – 30Y)**")
        sofr_sw = get_sofr_swaps_data()
        st.dataframe(sofr_sw, use_container_width=True, hide_index=True)
        st.plotly_chart(
            _rate_chart(sofr_sw, 'tenor', 'rate', 'SOFR OIS', 'darkblue',
                        'SOFR OIS Swap Rates', 'Tenor', 'Rate (%)'),
            use_container_width=True
        )

    # ================================================================
    # SECTION 2 — FOREIGN CURVE CONSTRUCTION (EUR ESTR)
    # ================================================================
    st.markdown("---")
    st.subheader("2 — Foreign Curve Construction (EUR ESTR)")

    for_tab1, for_tab2, for_tab3 = st.tabs(["Deposit", "Futures", "OIS Swaps"])

    with for_tab1:
        st.write("**ESTR Overnight Deposit**")
        st.dataframe(pd.DataFrame([get_estr_deposit_data()]),
                     use_container_width=True, hide_index=True)

    with for_tab2:
        st.write("**ESTR IMM Futures**")
        estr_fut = get_estr_futures_data()
        st.dataframe(estr_fut, use_container_width=True, hide_index=True)
        st.plotly_chart(
            _rate_chart(estr_fut, 'contract', 'rate', 'ESTR Futures', 'tomato',
                        'ESTR Futures Implied Rates', 'Contract', 'Rate (%)'),
            use_container_width=True
        )

    with for_tab3:
        st.write("**ESTR OIS Swaps (2Y – 30Y)**")
        estr_sw = get_estr_swaps_data()
        st.dataframe(estr_sw, use_container_width=True, hide_index=True)
        st.plotly_chart(
            _rate_chart(estr_sw, 'tenor', 'rate', 'ESTR OIS', 'darkred',
                        'ESTR OIS Swap Rates', 'Tenor', 'Rate (%)'),
            use_container_width=True
        )

    # ================================================================
    # SECTION 3 — CCY BASIS CURVE CONSTRUCTION
    # ================================================================
    st.markdown("---")
    st.subheader("3 — CCY Basis Curve Construction (EUR/USD)")

    basis_tab1, basis_tab2 = st.tabs(["FX Swaps (O/N – 18M)", "MtM XCcy Swaps (2Y – 30Y)"])

    with basis_tab1:
        st.write("""
        **FX Swaps** are the primary short-end instrument for the EUR/USD basis curve.
        Forward points directly imply the interest-rate differential (and hence the basis)
        via covered interest parity.  The 18M pillar is typically the last FX swap tenor
        before transitioning to cross-currency swaps.
        """)
        fx_sw = get_fx_swap_data()
        st.dataframe(fx_sw, use_container_width=True, hide_index=True)

        fig_fxsw = go.Figure()
        fig_fxsw.add_trace(go.Bar(
            x=fx_sw['tenor'], y=fx_sw['points'],
            name='Forward Points',
            marker_color=['#d62728' if p < 0 else '#2ca02c' for p in fx_sw['points']]
        ))
        fig_fxsw.update_layout(
            title='EUR/USD FX Swap Forward Points',
            height=420, hovermode='x unified',
            xaxis=dict(title_text='Tenor', type='category', tickangle=-35, **AXIS_STYLE),
            yaxis=dict(title_text='Forward Points (pips)', **AXIS_STYLE),
        )
        st.plotly_chart(fig_fxsw, use_container_width=True)

        fig_out = go.Figure(go.Scatter(
            x=fx_sw['tenor'], y=fx_sw['outright'],
            mode='lines+markers', name='Outright Forward',
            line=dict(width=3, color='purple'), marker=dict(size=8)
        ))
        fig_out.add_hline(y=get_fx_spot()['rate'], line_dash='dot', line_color='black',
                          annotation_text=f"Spot {get_fx_spot()['rate']:.4f}")
        fig_out.update_layout(
            title='EUR/USD FX Outright Forward Rates',
            height=400, hovermode='x unified',
            xaxis=dict(title_text='Tenor', type='category', tickangle=-35, **AXIS_STYLE),
            yaxis=dict(title_text='Outright Rate', **AXIS_STYLE),
        )
        st.plotly_chart(fig_out, use_container_width=True)

    with basis_tab2:
        st.write("""
        **Mark-to-Market Cross-Currency Swaps** are the standard long-end instrument.
        Convention: ESTR flat vs SOFR + basis spread (bps) — the EUR notional resets
        at prevailing spot each coupon date, removing FX delta so the quoted spread
        is the pure funding basis.
        """)
        ccy_sw = get_ccy_swaps_data()
        st.dataframe(ccy_sw, use_container_width=True, hide_index=True)

        fig_basis = go.Figure()
        fig_basis.add_trace(go.Scatter(
            x=ccy_sw['tenor'], y=ccy_sw['basis'],
            mode='lines+markers', name='EUR/USD Basis',
            line=dict(width=3, color='darkorange'), marker=dict(size=10)
        ))
        fig_basis.add_hline(y=0, line_dash='dash', line_color='gray',
                            annotation_text='Zero Basis', annotation_position='right')
        fig_basis.update_layout(
            title='EUR/USD Cross-Currency Basis Spreads',
            height=450, hovermode='x unified',
            xaxis=dict(title_text='Tenor', type='category', **AXIS_STYLE),
            yaxis=dict(title_text='Basis Spread (bps)', **AXIS_STYLE),
        )
        st.plotly_chart(fig_basis, use_container_width=True)

        st.info("""
        **Interpretation:** Negative basis means EUR funding is cheaper than implied
        by the interest-rate differential.  The EUR/USD basis has been persistently
        negative since 2008 due to regulatory costs (leverage ratio, G-SIB surcharge)
        preventing full CIP arbitrage.
        """)

    # ================================================================
    # SECTION 4 — FORWARD CURVE GENERATION
    # ================================================================
    st.markdown("---")
    st.subheader("4 — Forward Curve Generation")

    st.info("""
    Forward FX rates are derived from the domestic (SOFR) and foreign (ESTR) discount
    curves via covered interest parity, then adjusted for the CCY basis curve built
    from FX swaps and MtM cross-currency swaps above.
    """)

    if 'fx_curves' not in st.session_state:
        st.session_state.fx_curves = None

    if st.button("Bootstrap All Curves & Generate Forwards", type="primary",
                 use_container_width=True):
        with st.spinner("Bootstrapping curves..."):
            try:
                import io
                from contextlib import redirect_stdout

                fx_curves = FXCurves(eval_date)

                buf = io.StringIO()
                with redirect_stdout(buf):
                    fx_curves.bootstrap_domestic_curves()
                    fx_curves.bootstrap_basis_curve()

                log = buf.getvalue()
                if log.strip():
                    with st.expander("Bootstrap log", expanded=False):
                        st.code(log)

                st.session_state.fx_curves = fx_curves
                st.success("All curves bootstrapped successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Bootstrapping failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.fx_curves is not None:
        fx_curves = st.session_state.fx_curves

        # ── tenor grid: 0.25Y → 30Y at quarterly resolution (120 points) ──
        tenors = np.linspace(0.25, 30, 120)
        forward_curve = fx_curves.get_forward_curve(tenors)

        fwd_tab1, fwd_tab2, fwd_tab3, fwd_tab4 = st.tabs([
            "Zero Rates", "Discount Factors", "Forward FX Curve", "Basis Impact"
        ])

        with fwd_tab1:
            zero_df = fx_curves.get_zero_rate_summary()
            st.dataframe(zero_df, use_container_width=True, hide_index=True)
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=zero_df['Tenor (Years)'], y=zero_df['USD SOFR (%)'],
                                       mode='lines+markers', name='USD SOFR',
                                       line=dict(width=3, color='steelblue')))
            fig_z.add_trace(go.Scatter(x=zero_df['Tenor (Years)'], y=zero_df['EUR ESTR (%)'],
                                       mode='lines+markers', name='EUR ESTR',
                                       line=dict(width=3, color='tomato')))
            fig_z.update_layout(title='Bootstrapped Zero Rate Curves', height=500,
                                hovermode='x unified',
                                xaxis=dict(title_text='Tenor (Years)', **AXIS_STYLE),
                                yaxis=dict(title_text='Zero Rate (%)', **AXIS_STYLE))
            st.plotly_chart(fig_z, use_container_width=True)

        with fwd_tab2:
            df_sum = fx_curves.get_discount_factor_summary()
            st.dataframe(df_sum, use_container_width=True, hide_index=True)
            fig_df = go.Figure()
            fig_df.add_trace(go.Scatter(x=df_sum['Tenor (Years)'], y=df_sum['USD DF'],
                                        mode='lines+markers', name='USD DF',
                                        line=dict(width=3, color='steelblue')))
            fig_df.add_trace(go.Scatter(x=df_sum['Tenor (Years)'], y=df_sum['EUR DF'],
                                        mode='lines+markers', name='EUR DF',
                                        line=dict(width=3, color='tomato')))
            fig_df.update_layout(title='Discount Factor Curves', height=500,
                                 hovermode='x unified',
                                 xaxis=dict(title_text='Tenor (Years)', **AXIS_STYLE),
                                 yaxis=dict(title_text='Discount Factor', **AXIS_STYLE))
            st.plotly_chart(fig_df, use_container_width=True)

        with fwd_tab3:
            st.dataframe(forward_curve, use_container_width=True, hide_index=True)
            fig_fwd = go.Figure()
            fig_fwd.add_trace(go.Scatter(
                x=forward_curve['Tenor (Years)'], y=forward_curve['Standard Forward'],
                mode='lines', name='Standard Forward (no basis)',
                line=dict(width=2, color='gray', dash='dash')
            ))
            fig_fwd.add_trace(go.Scatter(
                x=forward_curve['Tenor (Years)'], y=forward_curve['Adjusted Forward'],
                mode='lines+markers', name='Basis-Adjusted Forward',
                line=dict(width=3, color='green'), marker=dict(size=4)
            ))
            fig_fwd.add_hline(y=fx_curves.spot_fx, line_dash='dot', line_color='black',
                              annotation_text=f"Spot {fx_curves.spot_fx:.4f}")
            fig_fwd.update_layout(
                title='EUR/USD Forward Curve (Basis-Adjusted)',
                height=550, hovermode='x unified',
                xaxis=dict(title_text='Tenor (Years)', **AXIS_STYLE),
                yaxis=dict(title_text='EUR/USD FX Rate', **AXIS_STYLE)
            )
            st.plotly_chart(fig_fwd, use_container_width=True)

        with fwd_tab4:
            fig_bi = go.Figure()
            fig_bi.add_trace(go.Scatter(
                x=forward_curve['Tenor (Years)'], y=forward_curve['Basis (bps)'],
                mode='lines+markers', name='Basis Spread',
                line=dict(width=3, color='darkorange'), marker=dict(size=4), yaxis='y1'
            ))
            fig_bi.add_trace(go.Scatter(
                x=forward_curve['Tenor (Years)'], y=forward_curve['Basis Impact (pips)'],
                mode='lines+markers', name='Impact on Forward (pips)',
                line=dict(width=3, color='purple'), marker=dict(size=4), yaxis='y2'
            ))
            fig_bi.update_layout(
                title='CCY Basis and Impact on FX Forwards',
                height=550, hovermode='x unified',
                xaxis=dict(title_text='Tenor (Years)', **AXIS_STYLE),
                yaxis=dict(title_text='Basis Spread (bps)',
                           title_font=dict(color='black'),
                           tickfont=dict(color='black')),
                yaxis2=dict(title_text='Impact on Forward (pips)', overlaying='y', side='right',
                            title_font=dict(color='black'), tickfont=dict(color='black')),
            )
            st.plotly_chart(fig_bi, use_container_width=True)
    else:
        st.info("Click 'Bootstrap All Curves & Generate Forwards' above to see results.")
