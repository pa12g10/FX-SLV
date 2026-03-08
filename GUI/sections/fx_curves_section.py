# FX Curves Section
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.fx_curves import FXCurves

try:
    import QuantLib as ql
except ImportError:
    ql = None

def render_fx_curves_section():
    """
    Render the FX Curves calibration section
    """
    st.header("FX Yield Curves & Spot Rate")
    
    if ql is None:
        st.error("QuantLib is not installed. Please install it to use FX-SLV features.")
        return
    
    st.markdown("---")
    
    # Market Data Input
    st.subheader("Market Data Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spot_fx = st.number_input(
            "Spot FX Rate (USD/EUR)",
            value=1.10,
            format="%.4f",
            help="Current spot exchange rate",
            key="fx_spot"
        )
    
    with col2:
        domestic_currency = st.text_input(
            "Domestic Currency",
            value="USD",
            key="fx_dom_ccy"
        )
    
    with col3:
        foreign_currency = st.text_input(
            "Foreign Currency",
            value="EUR",
            key="fx_for_ccy"
        )
    
    st.write("")
    
    # Yield Curves Input
    st.subheader("Yield Curve Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{domestic_currency} (Domestic) Rates**")
        
        # Default domestic rates (USD)
        default_domestic = pd.DataFrame({
            'Tenor (Years)': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
            'Rate (%)': [4.50, 4.60, 4.70, 4.50, 4.30, 4.00, 3.90, 3.80, 3.70, 3.65, 3.60]
        })
        
        domestic_rates_df = st.data_editor(
            default_domestic,
            num_rows="dynamic",
            key="fx_domestic_rates",
            hide_index=True
        )
    
    with col2:
        st.write(f"**{foreign_currency} (Foreign) Rates**")
        
        # Default foreign rates (EUR)
        default_foreign = pd.DataFrame({
            'Tenor (Years)': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
            'Rate (%)': [3.00, 3.10, 3.20, 3.00, 2.90, 2.70, 2.65, 2.60, 2.55, 2.50, 2.45]
        })
        
        foreign_rates_df = st.data_editor(
            default_foreign,
            num_rows="dynamic",
            key="fx_foreign_rates",
            hide_index=True
        )
    
    st.markdown("---")
    
    # Calibration Section
    st.subheader("Curve Calibration")
    
    # Store FX curves in session state
    if 'fx_curves' not in st.session_state:
        st.session_state.fx_curves = None
    
    if st.button("Bootstrap FX Curves", type="primary", key="fx_curves_calibrate_btn"):
        with st.spinner("Bootstrapping yield curves..."):
            try:
                # Set evaluation date
                eval_date = ql.Date(8, 3, 2026)
                ql.Settings.instance().evaluationDate = eval_date
                
                # Prepare rate data
                domestic_rates = [[row['Tenor (Years)'], row['Rate (%)']/100] 
                                for _, row in domestic_rates_df.iterrows()]
                foreign_rates = [[row['Tenor (Years)'], row['Rate (%)']/100] 
                               for _, row in foreign_rates_df.iterrows()]
                
                # Create FX curves object
                fx_curves = FXCurves(
                    eval_date,
                    domestic_rates,
                    foreign_rates,
                    domestic_currency,
                    foreign_currency
                )
                
                # Bootstrap curves
                fx_curves.bootstrap_curves()
                
                # Store in session state
                st.session_state.fx_curves = fx_curves
                st.session_state.spot_fx = spot_fx
                
                st.success(f"✅ FX curves bootstrapped successfully for {domestic_currency}/{foreign_currency}!")
                
            except Exception as e:
                st.error(f"Curve bootstrapping failed: {str(e)}")
    
    st.markdown("---")
    
    # Results Section
    if st.session_state.fx_curves is not None:
        st.subheader("FX Curves Results")
        
        fx_curves = st.session_state.fx_curves
        spot = st.session_state.spot_fx
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs([
            "📊 Zero Rates",
            "💰 Discount Factors",
            "📈 Forward FX Rates"
        ])
        
        with tab1:
            st.write("**Zero Rate Curves**")
            
            # Generate points for plotting
            tenors = np.linspace(0.1, 30, 100)
            domestic_zeros = []
            foreign_zeros = []
            
            for t in tenors:
                try:
                    dom_rate = fx_curves.domestic_curve.zeroRate(t, ql.Continuous).rate() * 100
                    for_rate = fx_curves.foreign_curve.zeroRate(t, ql.Continuous).rate() * 100
                    domestic_zeros.append(dom_rate)
                    foreign_zeros.append(for_rate)
                except:
                    domestic_zeros.append(None)
                    foreign_zeros.append(None)
            
            # Plot zero curves
            fig_zero = go.Figure()
            
            fig_zero.add_trace(go.Scatter(
                x=tenors,
                y=domestic_zeros,
                mode='lines',
                name=f'{fx_curves.domestic_currency} (Domestic)',
                line=dict(width=3, color='blue')
            ))
            
            fig_zero.add_trace(go.Scatter(
                x=tenors,
                y=foreign_zeros,
                mode='lines',
                name=f'{fx_curves.foreign_currency} (Foreign)',
                line=dict(width=3, color='red')
            ))
            
            # Add market input points
            fig_zero.add_trace(go.Scatter(
                x=domestic_rates_df['Tenor (Years)'],
                y=domestic_rates_df['Rate (%)'],
                mode='markers',
                name=f'{fx_curves.domestic_currency} Market Points',
                marker=dict(size=10, color='blue', symbol='diamond')
            ))
            
            fig_zero.add_trace(go.Scatter(
                x=foreign_rates_df['Tenor (Years)'],
                y=foreign_rates_df['Rate (%)'],
                mode='markers',
                name=f'{fx_curves.foreign_currency} Market Points',
                marker=dict(size=10, color='red', symbol='diamond')
            ))
            
            fig_zero.update_layout(
                title="Zero Rate Curves",
                xaxis_title="Tenor (Years)",
                yaxis_title="Zero Rate (%)",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_zero, use_container_width=True, key="fx_zero_chart")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                dom_1y = fx_curves.domestic_curve.zeroRate(1.0, ql.Continuous).rate() * 100
                st.metric(f"{fx_curves.domestic_currency} 1Y Rate", f"{dom_1y:.4f}%")
            with col2:
                for_1y = fx_curves.foreign_curve.zeroRate(1.0, ql.Continuous).rate() * 100
                st.metric(f"{fx_curves.foreign_currency} 1Y Rate", f"{for_1y:.4f}%")
            with col3:
                rate_diff = dom_1y - for_1y
                st.metric("Rate Differential (1Y)", f"{rate_diff:.4f}%")
        
        with tab2:
            st.write("**Discount Factor Curves**")
            
            # Generate discount factors
            tenors = np.linspace(0.1, 30, 100)
            domestic_dfs, foreign_dfs = fx_curves.get_discount_factors(tenors)
            
            # Plot discount factors
            fig_df = go.Figure()
            
            fig_df.add_trace(go.Scatter(
                x=tenors,
                y=domestic_dfs,
                mode='lines',
                name=f'{fx_curves.domestic_currency} (Domestic)',
                line=dict(width=3, color='blue')
            ))
            
            fig_df.add_trace(go.Scatter(
                x=tenors,
                y=foreign_dfs,
                mode='lines',
                name=f'{fx_curves.foreign_currency} (Foreign)',
                line=dict(width=3, color='red')
            ))
            
            fig_df.update_layout(
                title="Discount Factor Curves",
                xaxis_title="Tenor (Years)",
                yaxis_title="Discount Factor",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_df, use_container_width=True, key="fx_df_chart")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                dom_df_10y = fx_curves.domestic_curve.discount(10.0)
                st.metric(f"{fx_curves.domestic_currency} 10Y DF", f"{dom_df_10y:.6f}")
            with col2:
                for_df_10y = fx_curves.foreign_curve.discount(10.0)
                st.metric(f"{fx_curves.foreign_currency} 10Y DF", f"{for_df_10y:.6f}")
            with col3:
                df_ratio = for_df_10y / dom_df_10y
                st.metric("DF Ratio (For/Dom)", f"{df_ratio:.6f}")
        
        with tab3:
            st.write("**Forward FX Rates**")
            st.info("Forward FX rate calculated using covered interest rate parity: F = S × (DF_foreign / DF_domestic)")
            
            # Generate forward FX rates
            tenors = np.linspace(0.1, 10, 50)
            forward_fx_rates = [fx_curves.get_forward_fx(spot, t) for t in tenors]
            
            # Plot forward FX curve
            fig_fwd = go.Figure()
            
            fig_fwd.add_trace(go.Scatter(
                x=tenors,
                y=forward_fx_rates,
                mode='lines',
                name='Forward FX Rate',
                line=dict(width=3, color='green')
            ))
            
            # Add spot rate line
            fig_fwd.add_hline(
                y=spot,
                line_dash="dash",
                line_color="blue",
                line_width=2,
                annotation_text=f"Spot: {spot:.4f}",
                annotation_position="right"
            )
            
            fig_fwd.update_layout(
                title=f"Forward FX Curve ({fx_curves.domestic_currency}/{fx_curves.foreign_currency})",
                xaxis_title="Tenor (Years)",
                yaxis_title=f"FX Rate ({fx_curves.domestic_currency}/{fx_curves.foreign_currency})",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_fwd, use_container_width=True, key="fx_fwd_chart")
            
            # Forward points calculation
            st.write("**Forward Points Analysis**")
            
            sample_tenors = [0.25, 0.5, 1, 2, 3, 5, 10]
            forward_data = []
            
            for tenor in sample_tenors:
                if tenor <= 10:
                    fwd_rate = fx_curves.get_forward_fx(spot, tenor)
                    fwd_points = (fwd_rate - spot) * 10000  # in pips
                    forward_data.append({
                        'Tenor': f"{tenor}Y",
                        'Spot': spot,
                        'Forward': fwd_rate,
                        'Forward Points (pips)': fwd_points
                    })
            
            forward_df = pd.DataFrame(forward_data)
            st.dataframe(forward_df, use_container_width=True, hide_index=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                fwd_1y = fx_curves.get_forward_fx(spot, 1.0)
                st.metric("1Y Forward", f"{fwd_1y:.4f}")
            with col2:
                fwd_5y = fx_curves.get_forward_fx(spot, 5.0)
                st.metric("5Y Forward", f"{fwd_5y:.4f}")
            with col3:
                premium = ((fwd_1y - spot) / spot) * 100
                st.metric("1Y Forward Premium", f"{premium:.4f}%")
    
    else:
        st.info("Click 'Bootstrap FX Curves' to see results")
    
    st.markdown("---")
