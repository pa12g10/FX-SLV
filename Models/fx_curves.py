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
    FX Curves Builder - Constructs FX forward curve with cross-currency basis.

    Basis curve construction (two segments joined):
      Short end  (1W - 18M) : basis implied from FX swap forward points via CIP inversion
      Long end   (2Y - 30Y) : basis read directly from MtM CCY swap spreads

    CIP inversion for FX swaps
    --------------------------
    Market outright F_mkt = spot * df_eur / df_usd_basis
    => df_usd_basis = spot * df_eur / F_mkt

    Continuously-compounded basis spread b(T) from the basis-adjusted USD discount factor:
      df_usd_basis = df_usd * exp(-b * T)
      => b(T) = -ln(df_usd_basis / df_usd) / T
              = -ln(spot * df_eur / (F_mkt * df_usd)) / T   [in continuous terms]

    Or equivalently in simple (linear) terms consistent with the forward formula used in
    get_basis_adjusted_forward():
      df_usd_adjusted = df_usd / (1 + b * T)
      => 1 + b * T = df_usd / df_usd_basis = df_usd * F_mkt / (spot * df_eur)
      => b(T) = (df_usd * F_mkt / (spot * df_eur) - 1) / T
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

        # Unified basis curve spanning full tenor range
        self.basis_curve   = None
        self._basis_min_t  = None
        self._basis_max_t  = None

        # All basis pillars keyed by source (for reporting)
        # {tenor_years: {'basis_bps': float, 'source': 'FX Swap' | 'CCY Swap'}}
        self.basis_pillars = {}

        # Pricers
        self.fx_fwd_pricer   = None
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
        """
        Build the unified CCY basis curve from two instrument segments:

        1. FX Swaps (1W - 18M)  -> invert CIP to get implied basis spread per tenor
        2. CCY Swaps (2Y - 30Y) -> quoted basis spread directly

        All pillars are sorted by tenor and joined into a single
        ql.LinearInterpolation basis curve.
        """
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Domestic curves must be bootstrapped first")

        print("\n" + "="*60)
        print("BOOTSTRAPPING CROSS-CURRENCY BASIS CURVE")
        print("="*60)

        fx_spot_data = get_fx_spot()
        self.spot_fx = fx_spot_data['rate']
        self.fx_forwards_data = get_fx_forwards_data()
        self.ccy_swaps_data   = get_ccy_swaps_data()

        print(f"\nFX Spot: {fx_spot_data['pair']} = {self.spot_fx:.4f}")

        self.fx_fwd_pricer   = FXForwardPricer(self.eval_date, self.spot_fx)
        self.ccy_swap_pricer = CCYSwapPricer(self.eval_date, self.spot_fx)

        _SKIP = {'O/N', 'T/N', 'S/N'}
        all_pillars = {}   # {t_years: basis_decimal}

        # ---- SEGMENT 1: FX Swap-implied basis (short end) ----
        print("\nSegment 1 - FX Swap implied basis (CIP inversion):")
        print(f"  {'Tenor':<8} {'F_mkt':>10} {'F_cip':>10} {'Basis (bps)':>12}")
        print("  " + "-"*44)

        for _, row in self.fx_forwards_data.iterrows():
            tenor = row['tenor']
            if tenor in _SKIP:
                continue
            F_mkt = row['outright']
            try:
                t = self._parse_tenor_to_years(tenor)
                df_usd = self.usd_curve.discount(t)
                df_eur = self.eur_curve.discount(t)

                # Plain CIP forward (zero basis)
                F_cip = self.spot_fx * df_eur / df_usd

                # Invert to get simple basis spread:
                # b = (df_usd * F_mkt / (spot * df_eur) - 1) / T
                basis = (df_usd * F_mkt / (self.spot_fx * df_eur) - 1.0) / t

                all_pillars[t] = basis
                self.basis_pillars[t] = {
                    'tenor_label': tenor,
                    'basis_bps':   basis * 10_000,
                    'source':      'FX Swap',
                }
                print(f"  {tenor:<8} {F_mkt:>10.5f} {F_cip:>10.5f} {basis*10000:>11.2f}")

            except Exception as exc:
                print(f"  {tenor:<8} SKIPPED: {exc}")

        # ---- SEGMENT 2: CCY Swap quoted basis (long end) ----
        print("\nSegment 2 - CCY Swap quoted basis (long end):")
        print(f"  {'Tenor':<8} {'Basis (bps)':>12}")
        print("  " + "-"*22)

        for _, row in self.ccy_swaps_data.iterrows():
            tenor     = row['tenor']
            basis_bps = row['basis']
            try:
                t     = self._parse_tenor_to_years(tenor)
                basis = basis_bps / 10_000.0
                all_pillars[t] = basis
                self.basis_pillars[t] = {
                    'tenor_label': tenor,
                    'basis_bps':   basis_bps,
                    'source':      'CCY Swap',
                }
                print(f"  {tenor:<8} {basis_bps:>11.1f}")
            except Exception as exc:
                print(f"  {tenor:<8} SKIPPED: {exc}")

        # ---- Build unified interpolation ----
        sorted_tenors = sorted(all_pillars.keys())
        sorted_basis  = [all_pillars[t] for t in sorted_tenors]

        self._basis_min_t = sorted_tenors[0]
        self._basis_max_t = sorted_tenors[-1]
        self.basis_curve  = ql.LinearInterpolation(sorted_tenors, sorted_basis)

        bps_vals = [b * 10_000 for b in sorted_basis]
        print(f"\n\u2705 Unified basis curve: {len(sorted_tenors)} pillars "
              f"({self._basis_min_t:.3f}Y - {self._basis_max_t:.0f}Y)")
        print(f"   Basis range: {min(bps_vals):.1f} to {max(bps_vals):.1f} bps")

        print("\n" + "="*60)
        print("\u2705 CCY BASIS CURVE BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)

    # ------------------------------------------------------------------
    # CALIBRATION ERRORS
    # ------------------------------------------------------------------

    def get_basis_calibration_errors(self):
        """
        Check that all calibration instruments are repriced exactly by the
        unified basis curve.

        FX Swaps (1W - 18M)
        -------------------
        The basis curve was built by inverting these instruments, so the
        reconstructed outright should match the market exactly at pillar tenors.

        Model outright  = spot * df_eur / (df_usd / (1 + b(T)*T))
                        = spot * df_eur * (1 + b(T)*T) / df_usd
        Error (bps)     = (model_outright - market_outright) * 10,000

        CCY Swaps (2Y - 30Y)
        --------------------
        Model basis (bps) = basis_curve(T) * 10,000
        Error (bps)       = model_basis - market_basis

        Both should be ~0 at pillar tenors by construction.
        """
        if self.basis_curve is None:
            raise ValueError("Basis curve not bootstrapped yet.")

        rows  = []
        _SKIP = {'O/N', 'T/N', 'S/N'}

        # --- FX Swap instruments ---
        for _, row in self.fx_forwards_data.iterrows():
            tenor = row['tenor']
            if tenor in _SKIP:
                continue
            market_outright = row['outright']
            try:
                t      = self._parse_tenor_to_years(tenor)
                result = self.get_basis_adjusted_forward(t)
                model_outright = result['adjusted_forward']
                error_bps      = (model_outright - market_outright) * 10_000
                rows.append({
                    'instrument':      f"FX Swap {tenor}",
                    'instrument_type': 'FX Swaps',
                    'market_rate':     market_outright,
                    'model_rate':      model_outright,
                    'error_bps':       error_bps,
                })
            except Exception as exc:
                rows.append({
                    'instrument':      f"FX Swap {tenor}",
                    'instrument_type': 'FX Swaps',
                    'market_rate':     market_outright,
                    'model_rate':      float('nan'),
                    'error_bps':       float('nan'),
                })

        # --- CCY Swap instruments ---
        for _, row in self.ccy_swaps_data.iterrows():
            tenor     = row['tenor']
            mkt_basis = row['basis']
            try:
                t         = self._parse_tenor_to_years(tenor)
                mdl_basis = self.basis_curve(t) * 10_000
                error_bps = mdl_basis - mkt_basis
                rows.append({
                    'instrument':      f"CCY Swap {tenor}",
                    'instrument_type': 'CCY Swaps',
                    'market_rate':     mkt_basis,
                    'model_rate':      mdl_basis,
                    'error_bps':       error_bps,
                })
            except Exception:
                pass

        return pd.DataFrame(rows).dropna(subset=['error_bps'])

    def get_basis_pillars_summary(self):
        """
        Return a DataFrame of all basis curve pillars showing tenor, source
        instrument and implied basis in bps. Useful for inspecting the unified curve.
        """
        rows = []
        for t_years, info in sorted(self.basis_pillars.items()):
            rows.append({
                'Tenor':       info['tenor_label'],
                'Tenor (Y)':   round(t_years, 4),
                'Source':      info['source'],
                'Basis (bps)': round(info['basis_bps'], 3),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _parse_tenor_to_years(self, tenor_str):
        tenor_str = tenor_str.upper().strip()
        if tenor_str.endswith('W'):
            return int(tenor_str[:-1]) / 52.0
        elif tenor_str.endswith('M'):
            return int(tenor_str[:-1]) / 12.0
        elif tenor_str.endswith('Y'):
            return float(tenor_str[:-1])
        else:
            raise ValueError(f"Unsupported tenor format: '{tenor_str}'")

    def _compare_forward_vs_basis(self):
        """Print comparison of market vs CIP forwards at sample tenors."""
        sample_tenors = ['1M', '6M', '12M', '2Y', '5Y', '10Y']
        for tenor in sample_tenors:
            fwd_data = self.fx_forwards_data[self.fx_forwards_data['tenor'] == tenor]
            if fwd_data.empty:
                continue
            try:
                t_years             = self._parse_tenor_to_years(tenor)
                market_forward      = fwd_data['outright'].values[0]
                theoretical         = self.fx_fwd_pricer.calculate_forward_rate(
                                          tenor, self.usd_curve, self.eur_curve)
                theoretical_forward = theoretical['forward_rate']
                forward_diff_pips   = (market_forward - theoretical_forward) * 10_000
                basis_info          = self.basis_pillars.get(t_years, {})
                basis_bps           = basis_info.get('basis_bps', 'N/A')
                print(f"  {tenor}:")
                print(f"    Market Fwd:   {market_forward:.5f}")
                print(f"    CIP Fwd:      {theoretical_forward:.5f}")
                print(f"    Diff (pips):  {forward_diff_pips:.2f}")
                print(f"    Basis (bps):  {basis_bps}")
            except Exception as exc:
                print(f"  {tenor}: skipped ({exc})")

    # ------------------------------------------------------------------
    # CURVE OUTPUT METHODS
    # ------------------------------------------------------------------

    def get_basis_adjusted_forward(self, tenor_years):
        """
        Basis-adjusted FX forward using the unified basis curve.
        Clamps to domain bounds to avoid QL extrapolation errors.
        """
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Curves not bootstrapped yet")

        df_usd           = self.usd_curve.discount(tenor_years)
        df_eur           = self.eur_curve.discount(tenor_years)
        standard_forward = self.spot_fx * df_eur / df_usd

        if (self.basis_curve is not None
                and self._basis_min_t is not None
                and tenor_years >= self._basis_min_t):
            t_clamped    = min(tenor_years, self._basis_max_t)
            basis_spread = self.basis_curve(t_clamped)
        else:
            basis_spread = 0.0

        df_usd_adjusted  = df_usd / (1 + basis_spread * tenor_years)
        adjusted_forward = self.spot_fx * df_eur / df_usd_adjusted

        return {
            'tenor_years':        tenor_years,
            'spot':               self.spot_fx,
            'standard_forward':   standard_forward,
            'adjusted_forward':   adjusted_forward,
            'basis_spread_bps':   basis_spread * 10_000,
            'basis_impact_pips':  (adjusted_forward - standard_forward) * 10_000,
            'df_usd':             df_usd,
            'df_eur':             df_eur,
            'df_usd_adjusted':    df_usd_adjusted,
        }

    def get_forward_curve(self, tenors):
        forward_data = []
        for tenor in tenors:
            try:
                r = self.get_basis_adjusted_forward(tenor)
                forward_data.append({
                    'Tenor (Years)':       tenor,
                    'Spot':                r['spot'],
                    'Standard Forward':    r['standard_forward'],
                    'Adjusted Forward':    r['adjusted_forward'],
                    'Basis (bps)':         r['basis_spread_bps'],
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
