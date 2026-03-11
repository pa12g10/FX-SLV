# Single Barrier Options Section
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Pricing.single_barrier import SingleBarrierOption

try:
    import QuantLib as ql
except ImportError:
    ql = None

def render_single_barrier_section():
    """
    Render the single barrier options pricing section
    """
    st.header("Single Barrier FX Options")

    # -----------------------------------------------------------------------
    # Prerequisites
    # -----------------------------------------------------------------------
    if 'fx_curves' not in st.session_state or st.session_state.fx_curves is None:
        st.warning("⚠️ Please bootstrap FX curves first.")
        return

    if 'fx_slv_model' not in st.session_state or st.session_state.fx_slv_model is None:
        st.warning("⚠️ Please calibrate the FX-SLV model first.")
        return

    if ql is None:
        st.error("QuantLib is not installed.")
        return

    fx_curves = st.session_state.fx_curves
    fx_slv    = st.session_state.fx_slv_model

    # Always derive spot from fx_curves – never rely on a separate session_state key
    spot_fx = fx_curves.spot_fx

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Option Configuration
    # -----------------------------------------------------------------------
    st.subheader("Barrier Option Specification")

    col1, col2, col3 = st.columns(3)

    with col1:
        barrier_type = st.selectbox(
            "Barrier Type",
            ["UpOut", "DownOut", "UpIn", "DownIn"],
            key="sb_barrier_type",
            help="Up/Down: barrier above/below spot; Out/In: knockout/knockin"
        )

    with col2:
        option_type = st.selectbox(
            "Option Type",
            ["Call", "Put"],
            key="sb_option_type"
        )

    with col3:
        expiry_years = st.number_input(
            "Expiry (Years)",
            value=1.0,
            min_value=0.1,
            max_value=10.0,
            format="%.2f",
            key="sb_expiry"
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        strike = st.number_input(
            "Strike",
            value=float(spot_fx),
            format="%.4f",
            help="Strike price for the option",
            key="sb_strike"
        )

    with col2:
        default_barrier = float(spot_fx * 1.15 if "Up" in barrier_type else spot_fx * 0.85)
        barrier = st.number_input(
            "Barrier Level",
            value=default_barrier,
            format="%.4f",
            help="Barrier level (option knocks out/in if spot reaches this level)",
            key="sb_barrier"
        )

    # -----------------------------------------------------------------------
    # Option Setup Visualization
    # -----------------------------------------------------------------------
    st.write("**Option Setup Visualization**")

    fig_setup = go.Figure()

    fig_setup.add_hline(
        y=spot_fx, line_color="blue", line_width=3,
        annotation_text=f"Spot: {spot_fx:.4f}", annotation_position="left"
    )
    fig_setup.add_hline(
        y=strike, line_color="green", line_width=2, line_dash="dash",
        annotation_text=f"Strike: {strike:.4f}", annotation_position="left"
    )
    barrier_color = "red" if "Out" in barrier_type else "orange"
    fig_setup.add_hline(
        y=barrier, line_color=barrier_color, line_width=3,
        annotation_text=f"Barrier: {barrier:.4f} ({barrier_type})",
        annotation_position="right"
    )

    fig_setup.update_layout(
        title=f"{barrier_type} Barrier {option_type} Option Setup",
        yaxis_title="FX Rate",
        xaxis=dict(visible=False),
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig_setup, use_container_width=True, key="sb_setup_chart")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Pricing
    # -----------------------------------------------------------------------
    st.subheader("Pricing")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Price Option (FD Heston)", type="primary", key="sb_price_fd_btn"):
            with st.spinner("Pricing using Finite Difference Heston engine..."):
                try:
                    eval_date = ql.Date(8, 3, 2026)

                    barrier_option = SingleBarrierOption(
                        eval_date, spot_fx, strike, barrier,
                        expiry_years, barrier_type, option_type.lower(),
                        fx_curves.domestic_curve_handle,
                        fx_curves.foreign_curve_handle,
                        fx_slv
                    )

                    price  = barrier_option.price_option()
                    greeks = barrier_option.calculate_greeks()

                    st.session_state.sb_fd_price = price
                    st.session_state.sb_greeks   = greeks

                    st.success(f"✅ Option priced: **{price:.6f}**")

                except Exception as e:
                    st.error(f"Pricing failed: {str(e)}")

    with col2:
        mc_paths = st.number_input(
            "MC Paths",
            value=100000,
            min_value=10000,
            max_value=500000,
            step=10000,
            key="sb_mc_paths"
        )

        if st.button("Price Option (Monte Carlo)", type="primary", key="sb_price_mc_btn"):
            with st.spinner(f"Pricing using Monte Carlo ({mc_paths:,} paths)..."):
                try:
                    eval_date = ql.Date(8, 3, 2026)

                    barrier_option = SingleBarrierOption(
                        eval_date, spot_fx, strike, barrier,
                        expiry_years, barrier_type, option_type.lower(),
                        fx_curves.domestic_curve_handle,
                        fx_curves.foreign_curve_handle,
                        fx_slv
                    )

                    mc_results = barrier_option.monte_carlo_price(
                        num_paths=mc_paths,
                        time_steps=int(252 * expiry_years)
                    )

                    st.session_state.sb_mc_results = mc_results

                    st.success(
                        f"✅ MC Price: **{mc_results['price']:.6f}** "
                        f"(±{mc_results['std_error']:.6f})"
                    )

                except Exception as e:
                    st.error(f"MC pricing failed: {str(e)}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Results Display
    # -----------------------------------------------------------------------
    if 'sb_fd_price' in st.session_state or 'sb_mc_results' in st.session_state:
        st.subheader("Pricing Results")

        tab1, tab2, tab3 = st.tabs(["💰 Prices", "📈 Greeks", "📊 Analysis"])

        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'sb_fd_price' in st.session_state:
                    st.metric("FD Heston Price", f"{st.session_state.sb_fd_price:.6f}",
                              help="Finite Difference Heston engine price")

            with col2:
                if 'sb_mc_results' in st.session_state:
                    mc_r = st.session_state.sb_mc_results
                    st.metric("Monte Carlo Price", f"{mc_r['price']:.6f}",
                              help=f"Standard error: {mc_r['std_error']:.6f}")

            with col3:
                if 'sb_fd_price' in st.session_state and 'sb_mc_results' in st.session_state:
                    diff     = abs(st.session_state.sb_fd_price - st.session_state.sb_mc_results['price'])
                    diff_pct = (diff / st.session_state.sb_fd_price) * 100
                    st.metric("Price Difference", f"{diff:.6f}", delta=f"{diff_pct:.2f}%",
                              help="Absolute difference between FD and MC")

            if 'sb_mc_results' in st.session_state:
                st.write("")
                st.write("**Monte Carlo Statistics**")
                mc_r = st.session_state.sb_mc_results
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Standard Error",      f"{mc_r['std_error']:.6f}")
                with col2: st.metric("Breach Probability",  f"{mc_r['breach_probability']:.2%}")
                with col3: st.metric("95% Confidence Interval", f"±{1.96 * mc_r['std_error']:.6f}")

        with tab2:
            if 'sb_greeks' in st.session_state:
                st.write("**Option Greeks**")
                greeks = st.session_state.sb_greeks
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("Delta", f"{greeks['delta']:.6f}")
                with col2: st.metric("Gamma", f"{greeks['gamma']:.6f}")
                with col3: st.metric("Vega",  f"{greeks['vega']:.6f}")
                with col4: st.metric("Theta", f"{greeks['theta']:.6f}")
                with col5: st.metric("Rho",   f"{greeks['rho']:.6f}")
                st.info("💡 Note: Not all engines support all Greeks. Zero values may indicate unavailable Greeks.")
            else:
                st.info("Price option using FD Heston to see Greeks")

        with tab3:
            st.write("**Barrier Analysis**")
            st.write("**Option Payoff Diagram**")

            spot_range = np.linspace(spot_fx * 0.7, spot_fx * 1.3, 100)
            payoffs = []
            for s in spot_range:
                if   barrier_type == "UpOut"   and s >= barrier: payoff = 0
                elif barrier_type == "DownOut" and s <= barrier: payoff = 0
                elif barrier_type == "UpIn"    and s <  barrier: payoff = 0
                elif barrier_type == "DownIn"  and s >  barrier: payoff = 0
                else:
                    payoff = max(s - strike, 0) if option_type.lower() == "call" else max(strike - s, 0)
                payoffs.append(payoff)

            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(
                x=spot_range, y=payoffs, mode='lines', name='Payoff',
                line=dict(width=3, color='green'),
                fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'
            ))
            fig_payoff.add_vline(x=spot_fx, line_dash="dash", line_color="blue",  annotation_text="Current Spot")
            fig_payoff.add_vline(x=strike,  line_dash="dash", line_color="green", annotation_text="Strike")
            fig_payoff.add_vline(x=barrier, line_dash="solid",line_color="red",   annotation_text="Barrier", line_width=2)
            fig_payoff.update_layout(
                title=f"{barrier_type} {option_type} Payoff at Expiry",
                xaxis_title="FX Spot Rate", yaxis_title="Payoff",
                height=500, hovermode='x'
            )
            st.plotly_chart(fig_payoff, use_container_width=True, key="sb_payoff_chart")

            st.write("**Key Levels**")
            levels_df = pd.DataFrame([
                {"Level": "Current Spot", "Value": spot_fx, "Distance from Spot": "0.00%"},
                {"Level": "Strike",       "Value": strike,  "Distance from Spot": f"{((strike/spot_fx - 1)*100):.2f}%"},
                {"Level": "Barrier",      "Value": barrier, "Distance from Spot": f"{((barrier/spot_fx - 1)*100):.2f}%"},
            ])
            st.dataframe(levels_df, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                moneyness = strike / spot_fx
                if abs(moneyness - 1.0) < 0.02:
                    status = "ATM"
                elif (option_type.lower() == "call" and strike < spot_fx) or \
                     (option_type.lower() == "put"  and strike > spot_fx):
                    status = "ITM"
                else:
                    status = "OTM"
                st.metric("Moneyness", status, help=f"Strike/Spot: {moneyness:.4f}")
            with col2:
                st.metric("Distance to Barrier", f"{abs((barrier - spot_fx) / spot_fx) * 100:.2f}%")
            with col3:
                if 'sb_mc_results' in st.session_state:
                    st.metric("Breach Probability", f"{st.session_state.sb_mc_results['breach_probability']:.2%}")

    else:
        st.info("Price the option to see detailed results")

    st.markdown("---")
