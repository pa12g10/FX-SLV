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
    4. FX forward points (FX swaps, O/N – 18M)
    5. Cross-currency basis swaps (2Y – 30Y)
    """

    def __init__(self, eval_date):
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
        self.basis_spreads = {}   # {tenor_years: quoted_basis_bps}

        # Pricers
        self.fx_fwd_pricer = None
        self.ccy_swap_pricer = None

    # ------------------------------------------------------------------
    # BOOTSTRAPPING
    # ------------------------------------------------------------------

    def bootstrap_domestic_curves(self):
        """Bootstrap USD SOFR and EUR ESTR curves from market data."""
        print("\n" + "="*60)
        print("BOOTSTRAPPING DOMESTIC YIELD CURVES")
        print("="*60)

        self.usd_curve_builder = bootstrap_sofr_curve(self.eval_date)
        self.usd_curve = self.usd_curve_builder.curve

        print("\n" + "-"*60)

        self.eur_curve_builder = bootstrap_estr_curve(self.eval_date)
        self.eur_curve = self.eur_curve_builder.curve

        print("\n" + "="*60)
        print("\u2705 DOMESTIC CURVES BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)

    def bootstrap_basis_curve(self):
        """Bootstrap cross-currency basis curve from FX swaps and CCY swaps."""
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Domestic curves must be bootstrapped first")

        print("\n" + "="*60)
        print("BOOTSTRAPPING CROSS-CURRENCY BASIS CURVE")
        print("="*60)

        fx_spot_data = get_fx_spot()
        self.spot_fx = fx_spot_data['rate']
        self.fx_forwards_data = get_fx_forwards_data()
        self.ccy_swaps_data = get_ccy_swaps_data()

        print(f"\nFX Spot: {fx_spot_data['pair']} = {self.spot_fx:.4f}")

        self.fx_fwd_pricer   = FXForwardPricer(self.eval_date, self.spot_fx)
        self.ccy_swap_pricer = CCYSwapPricer(self.eval_date, self.spot_fx)

        print(f"\nExtracting basis spreads from {len(self.ccy_swaps_data)} CCY swaps:")

        tenors_list = []
        basis_list  = []

        for _, row in self.ccy_swaps_data.iterrows():
            tenor      = row['tenor']
            basis_bps  = row['basis']
            t_years    = self._parse_tenor_to_years(tenor)
            tenors_list.append(t_years)
            basis_list.append(basis_bps / 10_000.0)
            self.basis_spreads[t_years] = basis_bps
            print(f"  {tenor}: {basis_bps:.1f} bps")

        self.basis_curve = ql.LinearInterpolation(tenors_list, basis_list)

        print(f"\n\u2705 Basis curve constructed with {len(tenors_list)} points")
        print(f"   Basis range: {min(basis_list)*10000:.1f} to {max(basis_list)*10000:.1f} bps")

        print(f"\n\U0001f4ca Comparing FX Forward implied basis vs CCY swap basis:")
        self._compare_forward_vs_basis()

        print("\n" + "="*60)
        print("\u2705 CCY BASIS CURVE BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)

    # ------------------------------------------------------------------
    # CALIBRATION ERRORS  (CCY basis instruments)
    # ------------------------------------------------------------------

    def get_basis_calibration_errors(self):
        """
        Return calibration errors for the CCY basis instruments in bps.

        Two instrument types are checked:

        1. FX Swaps (O/N – 18M)
           Market rate  = quoted outright forward rate from fx_forwards_data.
           Model rate   = theoretical forward via CIP using USD/EUR discount curves
                          + basis adjustment at that tenor.
           Error (bps)  = (model_outright - market_outright) * 10,000

        2. CCY Swaps (2Y – 30Y)
           Market rate  = quoted basis spread (bps) from ccy_swaps_data.
           Model rate   = basis_curve(tenor_years) * 10,000  (interpolated value;
                          by construction equals the market pillar at knot points,
                          so knot errors are zero; off-pillar errors show interpolation
                          quality).
           Error (bps)  = model_basis_bps - market_basis_bps

        Returns
        -------
        pandas.DataFrame
            Columns: instrument, instrument_type, market_rate, model_rate, error_bps
        """
        if self.basis_curve is None:
            raise ValueError("Basis curve not bootstrapped yet. Call bootstrap_basis_curve() first.")

        rows = []

        # --- FX Swap instruments ---
        for _, row in self.fx_forwards_data.iterrows():
            tenor = row['tenor']
            # Skip overnight tenors that have no equivalent in the basis curve domain
            if tenor in ('O/N', 'T/N', 'S/N'):
                continue
            market_outright = row['outright']
            try:
                t_years = self._parse_tenor_to_years(tenor)
                model_result    = self.get_basis_adjusted_forward(t_years)
                model_outright  = model_result['adjusted_forward']
                error_bps       = (model_outright - market_outright) * 10_000
                rows.append({
                    'instrument':      f"FX Swap {tenor}",
                    'instrument_type': 'FX Swaps',
                    'market_rate':     market_outright,
                    'model_rate':      model_outright,
                    'error_bps':       error_bps,
                })
            except Exception:
                pass

        # --- CCY Swap instruments ---
        for _, row in self.ccy_swaps_data.iterrows():
            tenor     = row['tenor']
            mkt_basis = row['basis']   # already in bps
            try:
                t_years    = self._parse_tenor_to_years(tenor)
                mdl_basis  = self.basis_curve(t_years) * 10_000   # back to bps
                error_bps  = mdl_basis - mkt_basis
                rows.append({
                    'instrument':      f"CCY Swap {tenor}",
                    'instrument_type': 'CCY Swaps',
                    'market_rate':     mkt_basis,
                    'model_rate':      mdl_basis,
                    'error_bps':       error_bps,
                })
            except Exception:
                pass

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _parse_tenor_to_years(self, tenor_str):
        tenor_str = tenor_str.upper().strip()
        if tenor_str.endswith('M'):
            return int(tenor_str[:-1]) / 12.0
        elif tenor_str.endswith('Y'):
            return float(tenor_str[:-1])
        else:
            raise ValueError(f"Invalid tenor format: {tenor_str}")

    def _compare_forward_vs_basis(self):
        sample_tenors = ['1Y', '2Y', '5Y', '10Y']
        for tenor in sample_tenors:
            t_years  = self._parse_tenor_to_years(tenor)
            fwd_data = self.fx_forwards_data[self.fx_forwards_data['tenor'] == tenor]
            if fwd_data.empty:
                continue
            market_forward = fwd_data['outright'].values[0]
            theoretical    = self.fx_fwd_pricer.calculate_forward_rate(
                tenor, self.usd_curve, self.eur_curve
            )
            theoretical_forward = theoretical['forward_rate']
            forward_diff_pips   = (market_forward - theoretical_forward) * 10_000
            quoted_basis        = self.basis_spreads.get(t_years, 0)
            print(f"  {tenor}:")
            print(f"    Market Fwd:   {market_forward:.4f}")
            print(f"    Theoretical:  {theoretical_forward:.4f}")
            print(f"    Difference:   {forward_diff_pips:.1f} pips")
            print(f"    Quoted Basis: {quoted_basis:.1f} bps")

    # ------------------------------------------------------------------
    # CURVE OUTPUT METHODS
    # ------------------------------------------------------------------

    def get_basis_adjusted_forward(self, tenor_years):
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Curves not bootstrapped yet")
        df_usd = self.usd_curve.discount(tenor_years)
        df_eur = self.eur_curve.discount(tenor_years)
        standard_forward = self.spot_fx * df_eur / df_usd
        basis_spread     = self.basis_curve(tenor_years) if self.basis_curve is not None else 0.0
        df_usd_adjusted  = df_usd / (1 + basis_spread * tenor_years)
        adjusted_forward = self.spot_fx * df_eur / df_usd_adjusted
        return {
            'tenor_years':      tenor_years,
            'spot':             self.spot_fx,
            'standard_forward': standard_forward,
            'adjusted_forward': adjusted_forward,
            'basis_spread_bps': basis_spread * 10_000,
            'basis_impact_pips': (adjusted_forward - standard_forward) * 10_000,
            'df_usd':           df_usd,
            'df_eur':           df_eur,
            'df_usd_adjusted':  df_usd_adjusted,
        }

    def get_forward_curve(self, tenors):
        forward_data = []
        for tenor in tenors:
            try:
                r = self.get_basis_adjusted_forward(tenor)
                forward_data.append({
                    'Tenor (Years)':     tenor,
                    'Spot':              r['spot'],
                    'Standard Forward':  r['standard_forward'],
                    'Adjusted Forward':  r['adjusted_forward'],
                    'Basis (bps)':       r['basis_spread_bps'],
                    'Basis Impact (pips)': r['basis_impact_pips'],
                })
            except Exception:
                pass
        return pd.DataFrame(forward_data)

    def get_zero_rate_summary(self):
        if self.usd_curve_builder is None or self.eur_curve_builder is None:
            raise ValueError("Curves not bootstrapped yet")
        tenors    = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        usd_zeros = self.usd_curve_builder.get_zero_rates(tenors) * 100
        eur_zeros = self.eur_curve_builder.get_zero_rates(tenors) * 100
        data = []
        for t, u, e in zip(tenors, usd_zeros, eur_zeros):
            if not np.isnan(u) and not np.isnan(e):
                data.append({'Tenor (Years)': t, 'USD SOFR (%)': u,
                             'EUR ESTR (%)': e, 'Spread (bps)': (u - e) * 100})
        return pd.DataFrame(data)

    def get_discount_factor_summary(self):
        if self.usd_curve_builder is None or self.eur_curve_builder is None:
            raise ValueError("Curves not bootstrapped yet")
        tenors  = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        usd_dfs = self.usd_curve_builder.get_discount_factors(tenors)
        eur_dfs = self.eur_curve_builder.get_discount_factors(tenors)
        data = []
        for t, u, e in zip(tenors, usd_dfs, eur_dfs):
            if not np.isnan(u) and not np.isnan(e):
                data.append({'Tenor (Years)': t, 'USD DF': u,
                             'EUR DF': e, 'DF Ratio (EUR/USD)': e / u})
        return pd.DataFrame(data)
