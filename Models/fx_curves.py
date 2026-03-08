# FX Curves Module - Cross-Currency Basis and Forward Curves
import QuantLib as ql
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Models.yield_curve import YieldCurveBuilder, bootstrap_sofr_curve, bootstrap_estr_curve
from Pricing import FXForwardPricer, CCYSwapPricer
from MarketData import get_fx_spot, get_fx_forwards_data, get_ccy_swaps_data

class FXCurves:
    """
    FX Curves Builder - Constructs FX forward curve with cross-currency basis
    
    Uses:
    1. USD SOFR curve (domestic)
    2. EUR ESTR curve (foreign)
    3. FX spot rate
    4. FX forward points
    5. Cross-currency basis swaps
    """
    
    def __init__(self, eval_date):
        """
        Initialize FX curves builder
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Evaluation date
        """
        self.eval_date = eval_date
        ql.Settings.instance().evaluationDate = eval_date
        
        # Yield curves
        self.usd_curve_builder = None
        self.eur_curve_builder = None
        self.usd_curve = None
        self.eur_curve = None
        
        # FX data
        self.spot_fx = None
        self.fx_forwards_data = None
        self.ccy_swaps_data = None
        
        # Basis curve
        self.basis_curve = None
        self.basis_spreads = {}
        
        # Pricers
        self.fx_fwd_pricer = None
        self.ccy_swap_pricer = None
    
    def bootstrap_domestic_curves(self):
        """
        Bootstrap USD SOFR and EUR ESTR curves from market data
        """
        print("\n" + "="*60)
        print("BOOTSTRAPPING DOMESTIC YIELD CURVES")
        print("="*60)
        
        # Bootstrap USD SOFR curve
        self.usd_curve_builder = bootstrap_sofr_curve(self.eval_date)
        self.usd_curve = self.usd_curve_builder.curve
        
        print("\n" + "-"*60)
        
        # Bootstrap EUR ESTR curve
        self.eur_curve_builder = bootstrap_estr_curve(self.eval_date)
        self.eur_curve = self.eur_curve_builder.curve
        
        print("\n" + "="*60)
        print("✅ DOMESTIC CURVES BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)
    
    def bootstrap_basis_curve(self):
        """
        Bootstrap cross-currency basis curve from FX forwards and CCY swaps
        """
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Domestic curves must be bootstrapped first")
        
        print("\n" + "="*60)
        print("BOOTSTRAPPING CROSS-CURRENCY BASIS CURVE")
        print("="*60)
        
        # Get market data
        fx_spot_data = get_fx_spot()
        self.spot_fx = fx_spot_data['rate']
        self.fx_forwards_data = get_fx_forwards_data()
        self.ccy_swaps_data = get_ccy_swaps_data()
        
        print(f"\nFX Spot: {fx_spot_data['pair']} = {self.spot_fx:.4f}")
        
        # Initialize pricers
        self.fx_fwd_pricer = FXForwardPricer(self.eval_date, self.spot_fx)
        self.ccy_swap_pricer = CCYSwapPricer(self.eval_date, self.spot_fx)
        
        # Extract basis spreads from CCY swaps
        print(f"\nExtracting basis spreads from {len(self.ccy_swaps_data)} CCY swaps:")
        
        tenors_list = []
        basis_list = []
        
        for idx, row in self.ccy_swaps_data.iterrows():
            tenor = row['tenor']
            basis_bps = row['basis']
            
            # Convert tenor to years
            tenor_years = self._parse_tenor_to_years(tenor)
            
            tenors_list.append(tenor_years)
            basis_list.append(basis_bps / 10000.0)  # Convert bps to decimal
            
            self.basis_spreads[tenor_years] = basis_bps
            
            print(f"  {tenor}: {basis_bps:.1f} bps")
        
        # Create basis term structure (piecewise linear interpolation)
        dates_list = [self.eval_date + ql.Period(int(t * 365), ql.Days) for t in tenors_list]
        
        self.basis_curve = ql.LinearInterpolation(tenors_list, basis_list)
        
        print(f"\n✅ Basis curve constructed with {len(tenors_list)} points")
        print(f"   Basis range: {min(basis_list)*10000:.1f} to {max(basis_list)*10000:.1f} bps")
        
        # Calculate implied basis from FX forwards (for comparison)
        print(f"\n📊 Comparing FX Forward implied basis vs CCY swap basis:")
        self._compare_forward_vs_basis()
        
        print("\n" + "="*60)
        print("✅ CCY BASIS CURVE BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)
    
    def _parse_tenor_to_years(self, tenor_str):
        """
        Parse tenor string to years
        """
        tenor_str = tenor_str.upper().strip()
        
        if tenor_str.endswith('M'):
            months = int(tenor_str[:-1])
            return months / 12.0
        elif tenor_str.endswith('Y'):
            years = int(tenor_str[:-1])
            return float(years)
        else:
            raise ValueError(f"Invalid tenor format: {tenor_str}")
    
    def _compare_forward_vs_basis(self):
        """
        Compare implied basis from FX forwards vs quoted CCY basis
        """
        # Sample a few tenors
        sample_tenors = ['1Y', '2Y', '5Y', '10Y']
        
        for tenor in sample_tenors:
            tenor_years = self._parse_tenor_to_years(tenor)
            
            # Get market forward from FX forwards data
            fwd_data = self.fx_forwards_data[self.fx_forwards_data['tenor'] == tenor]
            if fwd_data.empty:
                continue
            
            market_forward = fwd_data['outright'].values[0]
            
            # Calculate theoretical forward (no basis)
            theoretical = self.fx_fwd_pricer.calculate_forward_rate(
                tenor, self.usd_curve, self.eur_curve
            )
            theoretical_forward = theoretical['forward_rate']
            
            # Implied basis (rough approximation)
            forward_diff_pips = (market_forward - theoretical_forward) * 10000
            
            # Get quoted basis
            quoted_basis = self.basis_spreads.get(tenor_years, 0)
            
            print(f"  {tenor}:")
            print(f"    Market Fwd: {market_forward:.4f}")
            print(f"    Theoretical: {theoretical_forward:.4f}")
            print(f"    Difference: {forward_diff_pips:.1f} pips")
            print(f"    Quoted Basis: {quoted_basis:.1f} bps")
    
    def get_basis_adjusted_forward(self, tenor_years):
        """
        Calculate basis-adjusted FX forward rate
        
        Parameters:
        -----------
        tenor_years : float
            Tenor in years
        
        Returns:
        --------
        dict: Forward rate with and without basis adjustment
        """
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Curves not bootstrapped yet")
        
        # Get discount factors
        df_usd = self.usd_curve.discount(tenor_years)
        df_eur = self.eur_curve.discount(tenor_years)
        
        # Standard forward (no basis)
        standard_forward = self.spot_fx * df_eur / df_usd
        
        # Get basis spread for this tenor
        if self.basis_curve is not None:
            basis_spread = self.basis_curve(tenor_years)  # In decimal
        else:
            basis_spread = 0.0
        
        # Adjust USD discount factor for basis
        # USD DF adjusted = USD DF / (1 + basis * T)
        df_usd_adjusted = df_usd / (1 + basis_spread * tenor_years)
        
        # Basis-adjusted forward
        adjusted_forward = self.spot_fx * df_eur / df_usd_adjusted
        
        return {
            'tenor_years': tenor_years,
            'spot': self.spot_fx,
            'standard_forward': standard_forward,
            'adjusted_forward': adjusted_forward,
            'basis_spread_bps': basis_spread * 10000,
            'basis_impact_pips': (adjusted_forward - standard_forward) * 10000,
            'df_usd': df_usd,
            'df_eur': df_eur,
            'df_usd_adjusted': df_usd_adjusted
        }
    
    def get_forward_curve(self, tenors):
        """
        Get forward FX curve with basis adjustment
        
        Parameters:
        -----------
        tenors : array-like
            Tenors in years
        
        Returns:
        --------
        pandas.DataFrame: Forward curve data
        """
        forward_data = []
        
        for tenor in tenors:
            try:
                result = self.get_basis_adjusted_forward(tenor)
                forward_data.append({
                    'Tenor (Years)': tenor,
                    'Spot': result['spot'],
                    'Standard Forward': result['standard_forward'],
                    'Adjusted Forward': result['adjusted_forward'],
                    'Basis (bps)': result['basis_spread_bps'],
                    'Basis Impact (pips)': result['basis_impact_pips']
                })
            except:
                pass
        
        return pd.DataFrame(forward_data)
    
    def get_zero_rate_summary(self):
        """
        Get summary of USD and EUR zero rates
        
        Returns:
        --------
        pandas.DataFrame: Zero rates comparison
        """
        if self.usd_curve_builder is None or self.eur_curve_builder is None:
            raise ValueError("Curves not bootstrapped yet")
        
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        
        usd_zeros = self.usd_curve_builder.get_zero_rates(tenors) * 100
        eur_zeros = self.eur_curve_builder.get_zero_rates(tenors) * 100
        
        data = []
        for tenor, usd_rate, eur_rate in zip(tenors, usd_zeros, eur_zeros):
            if not np.isnan(usd_rate) and not np.isnan(eur_rate):
                data.append({
                    'Tenor (Years)': tenor,
                    'USD SOFR (%)': usd_rate,
                    'EUR ESTR (%)': eur_rate,
                    'Spread (bps)': (usd_rate - eur_rate) * 100
                })
        
        return pd.DataFrame(data)
    
    def get_discount_factor_summary(self):
        """
        Get summary of USD and EUR discount factors
        
        Returns:
        --------
        pandas.DataFrame: Discount factors comparison
        """
        if self.usd_curve_builder is None or self.eur_curve_builder is None:
            raise ValueError("Curves not bootstrapped yet")
        
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        
        usd_dfs = self.usd_curve_builder.get_discount_factors(tenors)
        eur_dfs = self.eur_curve_builder.get_discount_factors(tenors)
        
        data = []
        for tenor, usd_df, eur_df in zip(tenors, usd_dfs, eur_dfs):
            if not np.isnan(usd_df) and not np.isnan(eur_df):
                data.append({
                    'Tenor (Years)': tenor,
                    'USD DF': usd_df,
                    'EUR DF': eur_df,
                    'DF Ratio (EUR/USD)': eur_df / usd_df
                })
        
        return pd.DataFrame(data)
