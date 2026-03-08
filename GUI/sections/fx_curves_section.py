# FX Curves Section - Market Data Display & Curve Bootstrapping
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from MarketData import (
    get_eval_date,
    get_sofr_deposit_data,
    get_sofr_futures_data,
    get_sofr_swaps_data,
    get_estr_deposit_data,
    get_estr_futures_data,
    get_estr_swaps_data,
    get_fx_spot,
    get_fx_forwards_data,
    get_ccy_swaps_data
)

try:
    import QuantLib as ql
except ImportError:
    ql = None

def render_fx_curves_section():
    """
    Render the FX Curves section with market data display and curve bootstrapping
    """
    st.header("🌍 FX Curves & Market Data")
    
    if ql is None:
        st.error("QuantLib is not installed. Please install it to use FX-SLV features.")
        return
    
    # Get evaluation date
    eval_date = get_eval_date()
    st.info(f"**Evaluation Date:** {eval_date.dayOfMonth()}/{eval_date.month()}/{eval_date.year()}")
    
    st.markdown("---")
    
    # ========================
    # SECTION 1: FX Spot
    # ========================
    st.subheader("💱 FX Spot Rate")
    
    fx_spot_data = get_fx_spot()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label=f"**{fx_spot_data['pair']}**",
            value=f"{fx_spot_data['rate']:.4f}",
            delta=None
        )
    with col2:
        st.info("Spot rate used for FX forward pricing and cross-currency swap valuation")
    
    st.markdown("---")
    
    # ========================
    # SECTION 2: USD SOFR Curve Instruments
    # ========================
    st.subheader("🇺🇸 USD SOFR Curve Instruments")
    
    # Create tabs for different instrument types
    sofr_tab1, sofr_tab2, sofr_tab3 = st.tabs(["📅 Deposits", "📊 Futures", "💼 OIS Swaps"])
    
    with sofr_tab1:
        st.write("**SOFR Overnight Deposit**")
        deposit_data = get_sofr_deposit_data()
        
        deposit_df = pd.DataFrame([deposit_data])
        st.dataframe(deposit_df, use_container_width=True, hide_index=True)
    
    with sofr_tab2:
        st.write("**SOFR Futures (1M - 18M)**")
        futures_data = get_sofr_futures_data()
        
        st.dataframe(futures_data, use_container_width=True, hide_index=True)
        
        # Plot futures rates
        fig_sofr_fut = go.Figure()
        fig_sofr_fut.add_trace(go.Scatter(
            x=futures_data.index,
            y=futures_data['rate'],
            mode='lines+markers',
            name='SOFR Futures Rate',
            line=dict(width=3, color='blue'),
            marker=dict(size=8)
        ))
        fig_sofr_fut.update_layout(
            title="SOFR Futures Implied Rates",
            xaxis_title="Contract",
            yaxis_title="Rate (%)",
            height=400
        )
        st.plotly_chart(fig_sofr_fut, use_container_width=True)
    
    with sofr_tab3:
        st.write("**SOFR OIS Swaps (2Y - 30Y)**")
        swaps_data = get_sofr_swaps_data()
        
        st.dataframe(swaps_data, use_container_width=True, hide_index=True)
        
        # Plot swap rates
        fig_sofr_swap = go.Figure()
        fig_sofr_swap.add_trace(go.Scatter(
            x=swaps_data.index,
            y=swaps_data['rate'],
            mode='lines+markers',
            name='SOFR OIS Rate',
            line=dict(width=3, color='darkblue'),
            marker=dict(size=8)
        ))
        fig_sofr_swap.update_layout(
            title="SOFR OIS Swap Rates",
            xaxis_title="Tenor",
            yaxis_title="Rate (%)",
            height=400
        )
        st.plotly_chart(fig_sofr_swap, use_container_width=True)
    
    st.markdown("---")
    
    # ========================
    # SECTION 3: EUR ESTR Curve Instruments
    # ========================
    st.subheader("🇪🇺 EUR ESTR Curve Instruments")
    
    # Create tabs for different instrument types
    estr_tab1, estr_tab2, estr_tab3 = st.tabs(["📅 Deposits", "📊 Futures", "💼 OIS Swaps"])
    
    with estr_tab1:
        st.write("**ESTR Overnight Deposit**")
        deposit_data = get_estr_deposit_data()
        
        deposit_df = pd.DataFrame([deposit_data])
        st.dataframe(deposit_df, use_container_width=True, hide_index=True)
    
    with estr_tab2:
        st.write("**ESTR Futures (1M - 18M)**")
        futures_data = get_estr_futures_data()
        
        st.dataframe(futures_data, use_container_width=True, hide_index=True)
        
        # Plot futures rates
        fig_estr_fut = go.Figure()
        fig_estr_fut.add_trace(go.Scatter(
            x=futures_data.index,
            y=futures_data['rate'],
            mode='lines+markers',
            name='ESTR Futures Rate',
            line=dict(width=3, color='red'),
            marker=dict(size=8)
        ))
        fig_estr_fut.update_layout(
            title="ESTR Futures Implied Rates",
            xaxis_title="Contract",
            yaxis_title="Rate (%)",
            height=400
        )
        st.plotly_chart(fig_estr_fut, use_container_width=True)
    
    with estr_tab3:
        st.write("**ESTR OIS Swaps (2Y - 30Y)**"")
        swaps_data = get_estr_swaps_data()
        
        st.dataframe(swaps_data, use_container_width=True, hide_index=True)
        
        # Plot swap rates
        fig_estr_swap = go.Figure()
        fig_estr_swap.add_trace(go.Scatter(
            x=swaps_data.index,
            y=swaps_data['rate'],
            mode='lines+markers',
            name='ESTR OIS Rate',
            line=dict(width=3, color='darkred'),
            marker=dict(size=8)
        ))
        fig_estr_swap.update_layout(
            title="ESTR OIS Swap Rates",
            xaxis_title="Tenor",
            yaxis_title="Rate (%)",
            height=400
        )
        st.plotly_chart(fig_estr_swap, use_container_width=True)
    
    st.markdown("---")
    
    # ========================
    # SECTION 4: FX Forwards
    # ========================
    st.subheader("📈 FX Forward Rates (EUR/USD)")
    
    fx_forwards_data = get_fx_forwards_data()
    st.dataframe(fx_forwards_data, use_container_width=True, hide_index=True)
    
    # Plot forward points
    fig_fwd = go.Figure()
    
    fig_fwd.add_trace(go.Scatter(
        x=fx_forwards_data.index,
        y=fx_forwards_data['points'],
        mode='lines+markers',
        name='Forward Points',
        line=dict(width=3, color='green'),
        marker=dict(size=8),
        yaxis='y1'
    ))
    
    fig_fwd.add_trace(go.Scatter(
        x=fx_forwards_data.index,
        y=fx_forwards_data['outright'],
        mode='lines+markers',
        name='Outright Rate',
        line=dict(width=3, color='purple', dash='dash'),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig_fwd.update_layout(
        title="EUR/USD Forward Points and Outright Rates",
        xaxis_title="Tenor",
        yaxis_title="Forward Points (pips)",
        yaxis2=dict(
            title="Outright Rate",
            overlaying='y',
            side='right'
        ),
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_fwd, use_container_width=True)
    
    st.markdown("---")
    
    # ========================
    # SECTION 5: Cross-Currency Basis Swaps
    # ========================
    st.subheader("🔄 Cross-Currency Basis Swaps (EUR/USD)")
    
    ccy_swaps_data = get_ccy_swaps_data()
    st.dataframe(ccy_swaps_data, use_container_width=True, hide_index=True)
    
    # Plot basis spreads
    fig_basis = go.Figure()
    
    fig_basis.add_trace(go.Scatter(
        x=ccy_swaps_data.index,
        y=ccy_swaps_data['basis'],
        mode='lines+markers',
        name='CCY Basis Spread',
        line=dict(width=3, color='orange'),
        marker=dict(size=10)
    ))
    
    fig_basis.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Zero Basis"
    )
    
    fig_basis.update_layout(
        title="EUR/USD Cross-Currency Basis Spreads",
        xaxis_title="Tenor",
        yaxis_title="Basis Spread (bps)",
        height=500
    )
    
    st.plotly_chart(fig_basis, use_container_width=True)
    
    st.info("""
    **Cross-Currency Basis Interpretation:**
    - Negative basis: EUR funding is cheaper than USD funding (adjusted)
    - The basis compensates for supply/demand imbalances in FX swap market
    - EUR/USD basis has been persistently negative since 2008 financial crisis
    """)
    
    st.markdown("---")
    
    # ========================
    # SECTION 6: Curve Comparison
    # ========================
    st.subheader("📊 SOFR vs ESTR Comparison")
    
    sofr_swaps = get_sofr_swaps_data()
    estr_swaps = get_estr_swaps_data()
    
    # Combine for comparison
    comparison_fig = go.Figure()
    
    comparison_fig.add_trace(go.Scatter(
        x=sofr_swaps.index,
        y=sofr_swaps['rate'],
        mode='lines+markers',
        name='USD SOFR',
        line=dict(width=3, color='blue'),
        marker=dict(size=8)
    ))
    
    comparison_fig.add_trace(go.Scatter(
        x=estr_swaps.index,
        y=estr_swaps['rate'],
        mode='lines+markers',
        name='EUR ESTR',
        line=dict(width=3, color='red'),
        marker=dict(size=8)
    ))
    
    # Calculate spread
    spread = sofr_swaps['rate'] - estr_swaps['rate']
    
    comparison_fig.add_trace(go.Scatter(
        x=sofr_swaps.index,
        y=spread,
        mode='lines',
        name='Spread (SOFR - ESTR)',
        line=dict(width=2, color='green', dash='dot'),
        yaxis='y2'
    ))
    
    comparison_fig.update_layout(
        title="USD SOFR vs EUR ESTR - OIS Swap Rates",
        xaxis_title="Tenor",
        yaxis_title="Rate (%)",
        yaxis2=dict(
            title="Spread (bps)",
            overlaying='y',
            side='right'
        ),
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sofr_5y = sofr_swaps[sofr_swaps['tenor'] == '5Y']['rate'].values[0]
        st.metric("SOFR 5Y", f"{sofr_5y:.2f}%")
    with col2:
        estr_5y = estr_swaps[estr_swaps['tenor'] == '5Y']['rate'].values[0]
        st.metric("ESTR 5Y", f"{estr_5y:.2f}%")
    with col3:
        spread_5y = sofr_5y - estr_5y
        st.metric("5Y Spread", f"{spread_5y:.2f}%", f"{spread_5y*100:.0f} bps")
    with col4:
        fx_fwd_5y = fx_forwards_data[fx_forwards_data['tenor'] == '5Y']['points'].values[0]
        st.metric("FX 5Y Fwd Pts", f"{fx_fwd_5y:.1f} pips")
    
    st.markdown("---")
    
    # ========================
    # SECTION 7: Bootstrap Curves (Placeholder)
    # ========================
    st.subheader("⚙️ Curve Bootstrapping")
    
    st.info("""
    **Next Step: Curve Construction**
    
    The market data above will be used to bootstrap:
    1. **USD SOFR Curve** - Using deposits, futures, and OIS swaps
    2. **EUR ESTR Curve** - Using deposits, futures, and OIS swaps  
    3. **FX Forward Curve** - From covered interest parity with FX forwards
    4. **CCY Basis Curve** - Incorporating cross-currency basis spreads
    
    Click the button below to proceed with curve construction.
    """)
    
    if st.button("🚀 Bootstrap All Curves", type="primary", use_container_width=True):
        st.warning("Curve bootstrapping functionality will be implemented in the next section.")
        st.info("This will use the Pricing modules (DepositPricer, FuturesPricer, SwapPricer, etc.) to build yield curves.")
