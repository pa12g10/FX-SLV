# FX Curves Module - Cross-Currency Basis and Forward Curves
#
# Bootstrap architecture
# ----------------------
# 1. Domestic curves: USD SOFR, EUR ESTR  (ql.PiecewiseLogLinearDiscount)
#
# 2. CCY basis curve = NEW EUR discount curve embedding the xccy basis:
#    Short end (1W-18M) : ql.FxSwapRateHelper
#                         is_fx_base_currency_collateral_currency=True
#                         USD is collateral; negative fwd pts => df_eur_basis < df_eur
#    Long end  (2Y-30Y) : ql.MtMCrossCurrencyBasisSwapRateHelper
#                         basisOnBase=False, baseResets=True, baseIsCollateral=True
#                         negative basis => EUR basis zero rates > flat ESTR
#    Both segments consistently produce df_eur_basis < df_eur
#    => adjusted_forward < standard_forward (correct for negative EUR/USD basis)
#
# 3. Interpolation: PiecewiseCubicZero
#    Cubic spline on zero rates gives C2 continuity across pillars,
#    smoothing through the FX-swap / CCY-swap seam at 18M->2Y.
#
# 4. Basis spread display: (zr_eur_basis - zr_eur_flat) * 10000
#    => should read ~-20 to -29 bps in the long end
#    => short end driven by FX swap implied basis

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
    FX Curves Builder.

    Produces:
    - self.usd_curve         : ql.PiecewiseLogLinearDiscount  (USD SOFR)
    - self.eur_curve         : ql.PiecewiseLogLinearDiscount  (EUR ESTR, flat)
    - self.eur_basis_curve   : ql.PiecewiseCubicZero          (EUR ESTR + basis)
    """

    def __init__(self, eval_date):
        self.eval_date = eval_date
        ql.Settings.instance().evaluationDate = eval_date

        self.usd_curve_builder = None
        self.eur_curve_builder = None
        self.usd_curve         = None
        self.eur_curve         = None
        self.eur_basis_curve   = None
        self._usd_handle       = None
        self._eur_handle       = None
        self._helpers          = []
        self._helper_meta      = []
        self.spot_fx           = None
        self.fx_forwards_data  = None
        self.ccy_swaps_data    = None
        self.fx_fwd_pricer     = None
        self.ccy_swap_pricer   = None

    # ------------------------------------------------------------------
    # STEP 1 - DOMESTIC CURVES
    # ------------------------------------------------------------------

    def bootstrap_domestic_curves(self):
        """Bootstrap USD SOFR and EUR ESTR via YieldCurveBuilder."""
        print("\n" + "="*60)
        print("BOOTSTRAPPING DOMESTIC YIELD CURVES")
        print("="*60)

        self.usd_curve_builder = bootstrap_sofr_curve(self.eval_date)
        self.usd_curve         = self.usd_curve_builder.curve
        print("\n" + "-"*60)
        self.eur_curve_builder = bootstrap_estr_curve(self.eval_date)
        self.eur_curve         = self.eur_curve_builder.curve

        self._usd_handle = ql.YieldTermStructureHandle(self.usd_curve)
        self._eur_handle = ql.YieldTermStructureHandle(self.eur_curve)

        print("\n" + "="*60)
        print("\u2705 DOMESTIC CURVES BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)

    # ------------------------------------------------------------------
    # STEP 2 - CCY BASIS CURVE
    # ------------------------------------------------------------------

    def bootstrap_basis_curve(self):
        """
        Bootstrap the EUR basis discount curve using PiecewiseCubicZero.

        Cubic spline interpolation on zero rates ensures C2 continuity
        across all pillars, smoothing the 18M->2Y seam between FX swap
        and CCY swap instruments.

        FX Swap helpers: is_fx_base_currency_collateral_currency=True
            USD is collateral; QL bootstraps EUR dfs from forward points.
            Negative fwd pts => EUR basis dfs < flat ESTR dfs.

        CCY Swap helpers: basisOnBase=False, baseResets=True
            Negative basis on EUR leg => EUR basis zero rates > flat ESTR
            => same direction as FX swap region: df_eur_basis < df_eur.
        """
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Call bootstrap_domestic_curves() first.")

        print("\n" + "="*60)
        print("BOOTSTRAPPING CCY BASIS CURVE (QL - PiecewiseCubicZero)")
        print("="*60)

        fx_spot_data          = get_fx_spot()
        self.spot_fx          = fx_spot_data['rate']
        self.fx_forwards_data = get_fx_forwards_data()
        self.ccy_swaps_data   = get_ccy_swaps_data()

        print(f"\nFX Spot: {fx_spot_data['pair']} = {self.spot_fx:.4f}")

        fx_spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))

        self.fx_fwd_pricer   = FXForwardPricer(self.eval_date, self.spot_fx)
        self.ccy_swap_pricer = CCYSwapPricer(self.eval_date, self.spot_fx)

        self._helpers     = []
        self._helper_meta = []
        _SKIP             = {'O/N', 'T/N', 'S/N'}
        used_pillars      = set()

        # ---- FX SWAP helpers (short end: 1W - 18M) ----
        print("\nAdding FX Swap helpers (ql.FxSwapRateHelper, USD collateral):")
        for _, row in self.fx_forwards_data.iterrows():
            tenor = row['tenor']
            if tenor in _SKIP:
                continue
            fwd_points = row['outright'] - self.spot_fx   # negative for EUR/USD
            try:
                helper = self.fx_fwd_pricer.create_ql_helper(
                    tenor            = tenor,
                    forward_points   = fwd_points,
                    usd_curve_handle = self._usd_handle,
                    eur_curve_handle = self._eur_handle,
                    is_fx_base_currency_collateral_currency = True,  # USD is collateral
                )
                pillar = helper.latestDate()
                if pillar not in used_pillars:
                    self._helpers.append(helper)
                    self._helper_meta.append({'label': f"FX Swap {tenor}",
                                              'instrument_type': 'FX Swaps'})
                    used_pillars.add(pillar)
                    print(f"  {tenor:<6}  fwd pts = {fwd_points*10000:+.2f} pips  -> pillar {pillar}")
                else:
                    print(f"  {tenor:<6}  SKIP (duplicate pillar {pillar})")
            except Exception as exc:
                print(f"  {tenor:<6}  ERROR building FX Swap helper: {exc}")

        # ---- CCY SWAP helpers (long end: 2Y - 30Y) ----
        print("\nAdding CCY Swap helpers (ql.MtMCrossCurrencyBasisSwapRateHelper):")
        for _, row in self.ccy_swaps_data.iterrows():
            tenor     = row['tenor']
            basis_bps = row['basis']
            try:
                helper = self.ccy_swap_pricer.create_ql_helper(
                    tenor            = tenor,
                    basis_spread_bps = basis_bps,
                    usd_curve_handle = self._usd_handle,
                    eur_curve_handle = self._eur_handle,
                    fx_spot_handle   = fx_spot_handle,
                )
                pillar = helper.latestDate()
                if pillar not in used_pillars:
                    self._helpers.append(helper)
                    self._helper_meta.append({'label': f"CCY Swap {tenor}",
                                              'instrument_type': 'CCY Swaps'})
                    used_pillars.add(pillar)
                    print(f"  {tenor:<6}  basis = {basis_bps:+.1f} bps  -> pillar {pillar}")
                else:
                    print(f"  {tenor:<6}  SKIP (duplicate pillar {pillar})")
            except Exception as exc:
                print(f"  {tenor:<6}  ERROR: {exc}")
                self._helper_meta.append({'label': f"CCY Swap {tenor}",
                                          'instrument_type': 'CCY Swaps',
                                          'build_error': str(exc)})

        print(f"\nBootstrapping with {len(self._helpers)} helpers...")

        # PiecewiseCubicZero: cubic spline on zero rates, C2 continuity,
        # smooths the kink at the FX-swap / CCY-swap seam (18M -> 2Y).
        self.eur_basis_curve = ql.PiecewiseCubicZero(
            self.eval_date, self._helpers, ql.Actual360()
        )
        self.eur_basis_curve.enableExtrapolation()

        print(f"\n\u2705 EUR basis curve bootstrapped: {len(self._helpers)} pillars")
        print("\nSample EUR basis discount factors (expect df_basis < df_flat_estr):")
        for t in [0.25, 0.5, 1, 2, 5, 10, 30]:
            try:
                df_basis = self.eur_basis_curve.discount(t)
                df_flat  = self.eur_curve.discount(t)
                zr_basis = self.eur_basis_curve.zeroRate(t, ql.Continuous).rate() * 100
                zr_flat  = self.eur_curve.zeroRate(t, ql.Continuous).rate() * 100
                spread   = (zr_basis - zr_flat) * 100
                chk      = '\u2705' if df_basis < df_flat else '\u274c'
                print(f"   {t:>4}Y: df_basis={df_basis:.6f}  df_flat={df_flat:.6f}  "
                      f"spread={spread:+.1f}bps  {chk}")
            except Exception:
                pass

        print("\n" + "="*60)
        print("\u2705 CCY BASIS CURVE BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)

    # ------------------------------------------------------------------
    # CALIBRATION ERRORS
    # ------------------------------------------------------------------

    def get_basis_calibration_errors(self):
        if self.eur_basis_curve is None:
            raise ValueError("Basis curve not bootstrapped yet.")

        rows = []
        helper_idx = 0
        for meta in self._helper_meta:
            if 'build_error' in meta:
                rows.append({'instrument':      meta['label'],
                             'instrument_type': meta['instrument_type'],
                             'market_rate':     float('nan'),
                             'model_rate':      float('nan'),
                             'error_bps':       float('nan'),
                             'note':            f"build error: {meta['build_error']}"})
                continue
            helper = self._helpers[helper_idx]
            helper_idx += 1
            try:
                mkt        = helper.quote().value()
                model      = helper.impliedQuote()
                error_bps  = (model - mkt) * 10_000
                rows.append({'instrument':      meta['label'],
                             'instrument_type': meta['instrument_type'],
                             'market_rate':     mkt   * 10_000,
                             'model_rate':      model * 10_000,
                             'error_bps':       error_bps,
                             'note':            ''})
            except Exception as exc:
                rows.append({'instrument':      meta['label'],
                             'instrument_type': meta['instrument_type'],
                             'market_rate':     float('nan'),
                             'model_rate':      float('nan'),
                             'error_bps':       float('nan'),
                             'note':            f"impliedQuote error: {exc}"})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CURVE OUTPUT METHODS
    # ------------------------------------------------------------------

    def get_basis_adjusted_forward(self, tenor_years):
        """
        F_adj = spot * df_eur_basis / df_usd
        basis_spread_bps = (zr_eur_basis - zr_eur_flat) * 10_000
            => negative since zr_basis > zr_flat when df_basis < df_flat.
        """
        if self.eur_basis_curve is None:
            raise ValueError("Basis curve not bootstrapped yet.")

        df_usd       = self.usd_curve.discount(tenor_years)
        df_eur       = self.eur_curve.discount(tenor_years)
        df_eur_basis = self.eur_basis_curve.discount(tenor_years)

        standard_forward = self.spot_fx * df_eur       / df_usd
        adjusted_forward = self.spot_fx * df_eur_basis / df_usd

        zr_eur_basis = self.eur_basis_curve.zeroRate(tenor_years, ql.Continuous).rate()
        zr_eur_flat  = self.eur_curve.zeroRate(tenor_years, ql.Continuous).rate()
        basis_bps    = (zr_eur_basis - zr_eur_flat) * 10_000

        return {
            'tenor_years':        tenor_years,
            'spot':               self.spot_fx,
            'standard_forward':   standard_forward,
            'adjusted_forward':   adjusted_forward,
            'basis_spread_bps':   basis_bps,
            'basis_impact_pips':  (adjusted_forward - standard_forward) * 10_000,
            'df_usd':             df_usd,
            'df_eur':             df_eur,
            'df_eur_basis':       df_eur_basis,
        }

    def get_forward_curve(self, tenors):
        rows = []
        for t in tenors:
            try:
                r = self.get_basis_adjusted_forward(t)
                rows.append({
                    'Tenor (Years)':       t,
                    'Spot':                r['spot'],
                    'Standard Forward':    r['standard_forward'],
                    'Adjusted Forward':    r['adjusted_forward'],
                    'Basis (bps)':         r['basis_spread_bps'],
                    'Basis Impact (pips)': r['basis_impact_pips'],
                })
            except Exception:
                pass
        return pd.DataFrame(rows)

    def get_zero_rate_summary(self):
        if self.usd_curve_builder is None or self.eur_curve_builder is None:
            raise ValueError("Domestic curves not bootstrapped yet.")
        tenors    = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        usd_zeros = self.usd_curve_builder.get_zero_rates(tenors) * 100
        eur_zeros = self.eur_curve_builder.get_zero_rates(tenors) * 100
        data = []
        for t, u, e in zip(tenors, usd_zeros, eur_zeros):
            if not (np.isnan(u) or np.isnan(e)):
                data.append({'Tenor (Years)': t, 'USD SOFR (%)': u,
                             'EUR ESTR (%)': e, 'Spread (bps)': (u - e) * 100})
        return pd.DataFrame(data)

    def get_discount_factor_summary(self):
        if self.usd_curve_builder is None or self.eur_curve_builder is None:
            raise ValueError("Domestic curves not bootstrapped yet.")
        tenors  = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        usd_dfs = self.usd_curve_builder.get_discount_factors(tenors)
        eur_dfs = self.eur_curve_builder.get_discount_factors(tenors)
        data = []
        for t, u, e in zip(tenors, usd_dfs, eur_dfs):
            if not (np.isnan(u) or np.isnan(e)):
                data.append({'Tenor (Years)': t, 'USD DF': u,
                             'EUR DF': e, 'DF Ratio (EUR/USD)': e / u})
        return pd.DataFrame(data)

    def _parse_tenor_to_years(self, tenor_str):
        tenor_str = tenor_str.upper().strip()
        if tenor_str.endswith('W'):
            return int(tenor_str[:-1]) / 52.0
        elif tenor_str.endswith('M'):
            return int(tenor_str[:-1]) / 12.0
        elif tenor_str.endswith('Y'):
            return float(tenor_str[:-1])
        else:
            raise ValueError(f"Unsupported tenor: '{tenor_str}'")
