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

try:
    import QuantLib as ql
except ImportError:
    ql = None

def render_fx_slv_section():
    """
    Render the FX-SLV model section
    """
    st.header("FX Stochastic Local Volatility Model")
    
    # Check if FX curves are available
    if 'fx_curves' not in st.session_state or st.session_state.fx_curves is None:
        st.warning("⚠️ Please bootstrap FX curves first in the 'FX Yield Curves & Spot Rate' section.")
        return
    
    if ql is None:
        st.error("QuantLib is not installed. Please install it to use FX-SLV features.")
        return
    
    fx_curves = st.session_state.fx_curves
    spot_fx = fx_curves.spot_fx  # Get spot_fx from fx_curves object
    
    st.markdown("---")
    
    # Volatility Surface Input
    st.subheader("FX Volatility Surface")
    
    st.write("**Market Implied Volatilities**")
    
    # Default volatility surface data
    default_vol_surface = pd.DataFrame({
        'Strike': [1.00, 1.05, 1.10, 1.15, 1.20,
                  1.00, 1.05, 1.10, 1.15, 1.20,
                  1.00, 1.05, 1.10, 1.15, 1.20,
                  1.00, 1.05, 1.10, 1.15, 1.20],
        'Expiry (Years)': [0.25, 0.25, 0.25, 0.25, 0.25,
                           0.5, 0.5, 0.5, 0.5, 0.5,
                           1.0, 1.0, 1.0, 1.0, 1.0,
                           2.0, 2.0, 2.0, 2.0, 2.0],
        'Volatility (%)': [12.0, 11.5, 11.0, 11.5, 12.5,
                          13.0, 12.5, 12.0, 12.5, 13.5,
                          14.0, 13.5, 13.0, 13.5, 14.5,
                          15.0, 14.5, 14.0, 14.5, 15.5]
    })
    
    vol_surface_df = st.data_editor(
        default_vol_surface,
        num_rows="dynamic",
        key="fx_vol_surface",
        hide_index=True
    )
    
    st.write("")
    
    # Visualize volatility surface
    if st.checkbox("Show Volatility Surface Plot", value=True, key="fx_vol_surface_plot_check"):
        # Prepare data for surface plot
        strikes = sorted(vol_surface_df['Strike'].unique())
        expiries = sorted(vol_surface_df['Expiry (Years)'].unique())
        
        # Create volatility matrix
        vol_matrix = np.zeros((len(expiries), len(strikes)))
        for i, expiry in enumerate(expiries):
            for j, strike in enumerate(strikes):
                matching = vol_surface_df[
                    (vol_surface_df['Strike'] == strike) & 
                    (vol_surface_df['Expiry (Years)'] == expiry)
                ]
                if not matching.empty:
                    vol_matrix[i, j] = matching.iloc[0]['Volatility (%)']
        
        # Create 3D surface plot
        fig_vol_surface = go.Figure(data=[go.Surface(
            x=strikes,
            y=expiries,
            z=vol_matrix,
            colorscale='Viridis',
            colorbar=dict(title="Vol (%)")
        )])
        
        fig_vol_surface.update_layout(
            title="FX Implied Volatility Surface",
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Expiry (Years)",
                zaxis_title="Volatility (%)"
            ),
            height=600
        )
        
        st.plotly_chart(fig_vol_surface, use_container_width=True, key="fx_vol_surface_3d")
        
        # 2D smile plot
        st.write("**Volatility Smiles by Expiry**")
        
        fig_smiles = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, expiry in enumerate(expiries):
            expiry_data = vol_surface_df[vol_surface_df['Expiry (Years)'] == expiry].sort_values('Strike')
            color = colors[i % len(colors)]
            
            fig_smiles.add_trace(go.Scatter(
                x=expiry_data['Strike'],
                y=expiry_data['Volatility (%)'],
                mode='markers+lines',
                name=f'{expiry}Y',
                marker=dict(size=10, color=color),
                line=dict(width=2, color=color)
            ))
        
        # Add ATM line
        fig_smiles.add_vline(
            x=spot_fx,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Spot: {spot_fx:.4f}",
            annotation_position="top"
        )
        
        fig_smiles.update_layout(
            title="Volatility Smiles",
            xaxis_title="Strike",
            yaxis_title="Implied Volatility (%)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_smiles, use_container_width=True, key="fx_vol_smiles")
    
    st.markdown("---")
    
    # Model Configuration
    st.subheader("FX-SLV Model Parameters")
    st.info("💡 The FX-SLV model uses Heston stochastic volatility combined with local volatility (Dupire). Parameters will be calibrated to market volatilities.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        v0 = st.number_input(
            "Initial Variance (v0)",
            value=0.01,
            format="%.4f",
            help="Initial variance (vol^2)",
            key="fx_slv_v0"
        )
        
        kappa = st.number_input(
            "Mean Reversion (κ)",
            value=1.0,
            format="%.4f",
            help="Mean reversion speed of variance",
            key="fx_slv_kappa"
        )
        
        theta = st.number_input(
            "Long-term Variance (θ)",
            value=0.01,
            format="%.4f",
            help="Long-term variance level",
            key="fx_slv_theta"
        )
    
    with col2:
        sigma = st.number_input(
            "Vol-of-Vol (σ)",
            value=0.3,
            format="%.4f",
            help="Volatility of variance (vol-of-vol)",
            key="fx_slv_sigma"
        )
        
        rho = st.number_input(
            "Correlation (ρ)",
            value=-0.7,
            min_value=-1.0,
            max_value=1.0,
            format="%.4f",
            help="Correlation between spot and volatility",
            key="fx_slv_rho"
        )
    
    st.markdown("---")
    
    # Calibration Section
    st.subheader("Model Calibration")
    
    if 'fx_slv_model' not in st.session_state:
        st.session_state.fx_slv_model = None
    
    if st.button("Calibrate FX-SLV Model", type="primary", key="fx_slv_calibrate_btn"):
        with st.spinner("Calibrating FX-SLV model to volatility surface..."):
            try:
                # Prepare volatility surface data
                vol_surface_data = []
                for _, row in vol_surface_df.iterrows():
                    vol_surface_data.append([
                        row['Strike'],
                        row['Expiry (Years)'],
                        row['Volatility (%)'] / 100  # Convert to decimal
                    ])
                
                # Model parameters
                model_params = {
                    'v0': v0,
                    'kappa': kappa,
                    'theta': theta,
                    'sigma': sigma,
                    'rho': rho
                }
                
                # Create and calibrate model
                eval_date = ql.Date(8, 3, 2026)
                
                fx_slv = FXStochasticLocalVol(
                    eval_date,
                    spot_fx,
                    fx_curves.domestic_curve_handle,
                    fx_curves.foreign_curve_handle,
                    vol_surface_data,
                    model_params
                )
                
                fx_slv.calibrate()
                
                st.session_state.fx_slv_model = fx_slv
                
                st.success("✅ FX-SLV model calibrated successfully!")
                
            except Exception as e:
                st.error(f"Calibration failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    st.markdown("---")
    
    # Results Section
    if st.session_state.fx_slv_model is not None:
        st.subheader("FX-SLV Calibration Results")
        
        fx_slv = st.session_state.fx_slv_model
        results = fx_slv.get_calibrated_results()
        
        if results:
            # Display calibrated parameters
            st.write("**Calibrated Heston Parameters**")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("v0", f"{results['v0']:.6f}")
            with col2:
                st.metric("κ", f"{results['kappa']:.6f}")
            with col3:
                st.metric("θ", f"{results['theta']:.6f}")
            with col4:
                st.metric("σ", f"{results['sigma']:.6f}")
            with col5:
                st.metric("ρ", f"{results['rho']:.6f}")
            
            st.write("")
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Calibration Quality",
                "📈 Simulated Paths",
                "✅ Model Validation",
                "📋 Detailed Results"
            ])
            
            with tab1:
                errors_df = results['pricing_errors']
                
                # Market vs Model volatilities
                st.write("**Market vs Model Implied Volatilities**")
                
                fig_vols = go.Figure()
                
                fig_vols.add_trace(go.Scatter(
                    x=list(range(len(errors_df))),
                    y=errors_df['market_vol'],
                    mode='markers',
                    name='Market Vol',
                    marker=dict(size=10, color='blue', symbol='diamond')
                ))
                
                fig_vols.add_trace(go.Scatter(
                    x=list(range(len(errors_df))),
                    y=errors_df['model_vol'],
                    mode='markers',
                    name='Model Vol',
                    marker=dict(size=8, color='red', symbol='circle')
                ))
                
                fig_vols.update_layout(
                    title="Volatility Calibration: Market vs Model",
                    xaxis_title="Option Index",
                    yaxis_title="Implied Volatility (%)",
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_vols, use_container_width=True, key="fx_slv_vols_chart")
                
                # Volatility errors
                st.write("**Volatility Errors**")
                
                fig_vol_errors = go.Figure()
                
                fig_vol_errors.add_trace(go.Bar(
                    x=[f"K={row['strike']:.2f}, T={row['expiry']:.2f}Y" for _, row in errors_df.iterrows()],
                    y=errors_df['vol_error_bps'],
                    marker_color='orange'
                ))
                
                fig_vol_errors.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black",
                    line_width=2
                )
                
                fig_vol_errors.update_layout(
                    title="Implied Volatility Errors (Model - Market)",
                    xaxis_title="Option",
                    yaxis_title="Error (bps)",
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_vol_errors, use_container_width=True, key="fx_slv_vol_errors_chart")
                
                # Error statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max Vol Error", f"{errors_df['vol_error_bps'].abs().max():.2f} bps")
                with col2:
                    st.metric("Mean Vol Error", f"{errors_df['vol_error_bps'].mean():.2f} bps")
                with col3:
                    st.metric("RMSE (Vol)", f"{np.sqrt((errors_df['vol_error_bps']**2).mean()):.2f} bps")
                with col4:
                    st.metric("Std Dev", f"{errors_df['vol_error_bps'].std():.2f} bps")
            
            with tab2:
                st.write("**Simulated FX Spot and Volatility Paths**")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_paths = st.slider("Number of Paths", 10, 100, 50, step=10, key="fx_slv_paths_slider")
                with col2:
                    horizon = st.slider("Time Horizon (years)", 0.5, 5.0, 1.0, step=0.5, key="fx_slv_horizon_slider")
                
                if st.button("Generate Paths", key="fx_slv_gen_paths_btn"):
                    with st.spinner("Generating Monte Carlo paths..."):
                        path_df, times, spot_paths, vol_paths = fx_slv.get_simulated_paths(
                            num_paths=1000,
                            time_steps=252,
                            horizon_years=horizon
                        )
                        
                        # Plot FX spot paths
                        st.write("**FX Spot Rate Paths**")
                        
                        fig_spot = go.Figure()
                        
                        for i in range(min(num_paths, 1000)):
                            fig_spot.add_trace(go.Scatter(
                                x=times,
                                y=spot_paths[:, i],
                                mode='lines',
                                line=dict(width=0.5),
                                opacity=0.3,
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                        
                        # Add mean path
                        mean_spot = spot_paths.mean(axis=1)
                        fig_spot.add_trace(go.Scatter(
                            x=times,
                            y=mean_spot,
                            mode='lines',
                            name='Mean Path',
                            line=dict(color='red', width=3)
                        ))
                        
                        # Add spot line
                        fig_spot.add_hline(
                            y=spot_fx,
                            line_dash="dash",
                            line_color="blue",
                            annotation_text=f"Initial Spot: {spot_fx:.4f}"
                        )
                        
                        fig_spot.update_layout(
                            title=f"FX Spot Simulation ({num_paths} of 1000 paths shown)",
                            xaxis_title="Time (years)",
                            yaxis_title="FX Spot Rate",
                            height=500
                        )
                        
                        st.plotly_chart(fig_spot, use_container_width=True, key="fx_slv_spot_paths")
                        
                        # Plot volatility paths
                        st.write("**Stochastic Volatility Paths**")
                        
                        fig_vol = go.Figure()
                        
                        # Convert variance to volatility
                        vol_paths_pct = np.sqrt(vol_paths) * 100
                        
                        for i in range(min(num_paths, 1000)):
                            fig_vol.add_trace(go.Scatter(
                                x=times,
                                y=vol_paths_pct[:, i],
                                mode='lines',
                                line=dict(width=0.5),
                                opacity=0.3,
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                        
                        # Add mean volatility
                        mean_vol = vol_paths_pct.mean(axis=1)
                        fig_vol.add_trace(go.Scatter(
                            x=times,
                            y=mean_vol,
                            mode='lines',
                            name='Mean Volatility',
                            line=dict(color='purple', width=3)
                        ))
                        
                        fig_vol.update_layout(
                            title="Stochastic Volatility Evolution",
                            xaxis_title="Time (years)",
                            yaxis_title="Volatility (%)",
                            height=500
                        )
                        
                        st.plotly_chart(fig_vol, use_container_width=True, key="fx_slv_vol_paths")
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Spot", f"{spot_paths[0, 0]:.4f}")
                        with col2:
                            st.metric(f"Mean Spot at {horizon}Y", f"{spot_paths[-1, :].mean():.4f}")
                        with col3:
                            st.metric(f"Spot Std Dev at {horizon}Y", f"{spot_paths[-1, :].std():.4f}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Vol", f"{vol_paths_pct[0, 0]:.2f}%")
                        with col2:
                            st.metric(f"Mean Vol at {horizon}Y", f"{vol_paths_pct[-1, :].mean():.2f}%")
                        with col3:
                            st.metric(f"Vol Std Dev at {horizon}Y", f"{vol_paths_pct[-1, :].std():.2f}%")
            
            with tab3:
                st.write("**Model Validation: Option Pricing**")
                st.info("💡 Validates model by comparing prices with Black-Scholes using market volatilities.")
                
                if st.button("Run Validation", type="primary", key="fx_slv_validation_btn"):
                    with st.spinner("Running validation..."):
                        validation_results = fx_slv.validate_option_prices()
                        
                        if validation_results is not None:
                            st.success("✅ Validation completed!")
                            
                            # Price comparison plot
                            fig_val = go.Figure()
                            
                            fig_val.add_trace(go.Scatter(
                                x=list(range(len(validation_results))),
                                y=validation_results['bs_price'],
                                mode='markers+lines',
                                name='Black-Scholes Price',
                                marker=dict(size=10, color='blue')
                            ))
                            
                            fig_val.add_trace(go.Scatter(
                                x=list(range(len(validation_results))),
                                y=validation_results['heston_price'],
                                mode='markers+lines',
                                name='Heston Price',
                                marker=dict(size=8, color='red')
                            ))
                            
                            fig_val.update_layout(
                                title="Option Prices: Black-Scholes vs Heston",
                                xaxis_title="Option Index",
                                yaxis_title="Price",
                                height=500
                            )
                            
                            st.plotly_chart(fig_val, use_container_width=True, key="fx_slv_val_chart")
                            
                            # Error plot
                            fig_val_error = go.Figure()
                            
                            fig_val_error.add_trace(go.Bar(
                                x=[f"K={row['strike']:.2f}, T={row['expiry']:.2f}Y" 
                                   for _, row in validation_results.iterrows()],
                                y=validation_results['price_diff_pct'],
                                marker_color='green'
                            ))
                            
                            fig_val_error.add_hline(y=0, line_dash="dash", line_color="black")
                            
                            fig_val_error.update_layout(
                                title="Price Differences (Heston - BS)",
                                xaxis_title="Option",
                                yaxis_title="Difference (%)",
                                height=500,
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig_val_error, use_container_width=True, key="fx_slv_val_error_chart")
                            
                            # Display table
                            st.write("**Validation Results**")
                            st.dataframe(validation_results, use_container_width=True, hide_index=True)
            
            with tab4:
                st.write("**Detailed Calibration Results**")
                
                display_errors = errors_df[[
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
        st.info("Click 'Calibrate FX-SLV Model' to see results")
    
    st.markdown("---")
