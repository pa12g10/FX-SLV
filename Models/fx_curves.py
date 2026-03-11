# FX Curves Module - Cross-Currency Basis and Forward Curves
#
# Bootstrap architecture
# ----------------------
# 1. Domestic curves already built: USD SOFR (ql.PiecewiseLogLinearDiscount)
#                                    EUR ESTR (ql.PiecewiseLogLinearDiscount)
#
# 2. CCY basis curve = a NEW EUR discount curve that embeds the basis:
#
#    Short end (1W - 18M)  FX Swap helpers via FXForwardPricer.create_ql_helper()
#                          -> ql.FxSwapRateHelper
#
#    Long end  (2Y - 30Y)  CCY Swap helpers via CCYSwapPricer.create_ql_helper()
#                          -> ql.CrossCurrencyBasisSwapRateHelper
#
#    All helpers fed into ql.PiecewiseLogLinearDiscount to produce
#    self.eur_basis_curve: a proper QL YieldTermStructure
#
# 3. Calibration errors via helper.impliedQuote() - helper.quote().value()
#    (same pattern as YieldCurveBuilder)
#
# Key QL convention for CrossCurrencyBasisSwapRateHelper:
#   base  = USD  (curve is KNOWN  -> usd_curve_handle passed as baseCcyDiscount)
#   quote = EUR  (curve being BOOTSTRAPPED -> eur_curve_handle passed as quoteCcyDiscount)
# Without quoteCcyDiscount the helper bootstraps USD, not EUR, yielding 1/df.

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
    - self.eur_basis_curve   : ql.PiecewiseLogLinearDiscount  (EUR ESTR + basis)
    """

    def __init__(self, eval_date):
        self.eval_date = eval_date
        ql.Settings.instance().evaluationDate = eval_date

        # Yield curves
        self.usd_curve_builder = None
        self.eur_curve_builder = None
        self.usd_curve         = None   # ql.PiecewiseLogLinearDiscount
        self.eur_curve         = None   # ql.PiecewiseLogLinearDiscount (flat ESTR)

        # Basis curve output
        self.eur_basis_curve   = None   # ql.PiecewiseLogLinearDiscount (ESTR + basis)

        # Handles (lazy-built)
        self._usd_handle       = None
        self._eur_handle       = None

        # Helpers + metadata for calibration error reporting
        self._helpers     = []   # ql rate helpers
        self._helper_meta = []   # [{label, instrument_type}]

        # FX data
        self.spot_fx          = None
        self.fx_forwards_data = None
        self.ccy_swaps_data   = None

        # Pricers
        self.fx_fwd_pricer   = None
        self.ccy_swap_pricer = None

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

        # Build handles once (reused by all helpers)
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
        Bootstrap the EUR basis discount curve using:
          - ql.FxSwapRateHelper                    for FX swap tenors (1W - 18M)
          - ql.CrossCurrencyBasisSwapRateHelper    for CCY swap tenors (2Y - 30Y)

        QL base/quote convention:
          base  = USD (known curve)     -> usd_curve_handle
          quote = EUR (bootstrapped)    -> eur_curve_handle

        Output: self.eur_basis_curve  (ql.PiecewiseLogLinearDiscount)
        """
        if self.usd_curve is None or self.eur_curve is None:
            raise ValueError("Call bootstrap_domestic_curves() first.")

        print("\n" + "="*60)
        print("BOOTSTRAPPING CCY BASIS CURVE (QL)")
        print("="*60)

        # Market data
        fx_spot_data          = get_fx_spot()
        self.spot_fx          = fx_spot_data['rate']
        self.fx_forwards_data = get_fx_forwards_data()
        self.ccy_swaps_data   = get_ccy_swaps_data()

        print(f"\nFX Spot: {fx_spot_data['pair']} = {self.spot_fx:.4f}")

        # Shared FX spot handle
        fx_spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))

        # Pricers
        self.fx_fwd_pricer   = FXForwardPricer(self.eval_date, self.spot_fx)
        self.ccy_swap_pricer = CCYSwapPricer(self.eval_date, self.spot_fx)

        self._helpers     = []
        self._helper_meta = []
        _SKIP             = {'O/N', 'T/N', 'S/N'}
        used_pillars      = set()

        # ---- FX SWAP helpers (short end) ----
        print("\nAdding FX Swap helpers (ql.FxSwapRateHelper):")
        for _, row in self.fx_forwards_data.iterrows():
            tenor = row['tenor']
            if tenor in _SKIP:
                continue
            fwd_points = row['outright'] - self.spot_fx   # F - S
            try:
                helper      = self.fx_fwd_pricer.create_ql_helper(
                    tenor           = tenor,
                    forward_points  = fwd_points,
                    usd_curve_handle = self._usd_handle,
                    eur_curve_handle = self._eur_handle,
                )
                pillar = helper.latestDate()
                if pillar not in used_pillars:
                    self._helpers.append(helper)
                    self._helper_meta.append({
                        'label':           f"FX Swap {tenor}",
                        'instrument_type': 'FX Swaps',
                    })
                    used_pillars.add(pillar)
                    print(f"  {tenor:<6}  fwd pts = {fwd_points*10000:+.2f} pips  "
                          f"-> pillar {pillar}")
                else:
                    print(f"  {tenor:<6}  SKIP (duplicate pillar {pillar})")
            except Exception as exc:
                print(f"  {tenor:<6}  ERROR building FX Swap helper: {exc}")

        # ---- CCY SWAP helpers (long end) ----
        # FIX: eur_curve_handle is now correctly passed as quoteCcyDiscountCurve
        # inside ccy_swap_pricer.create_ql_helper() so QL bootstraps EUR (quote)
        # given USD (base), not the other way around.
        print("\nAdding CCY Swap helpers (ql.CrossCurrencyBasisSwapRateHelper):")
        for _, row in self.ccy_swaps_data.iterrows():
            tenor     = row['tenor']
            basis_bps = row['basis']
            try:
                helper = self.ccy_swap_pricer.create_ql_helper(
                    tenor             = tenor,
                    basis_spread_bps  = basis_bps,
                    usd_curve_handle  = self._usd_handle,
                    eur_curve_handle  = self._eur_handle,
                    fx_spot_handle    = fx_spot_handle,
                )
                pillar = helper.latestDate()
                if pillar not in used_pillars:
                    self._helpers.append(helper)
                    self._helper_meta.append({
                        'label':           f"CCY Swap {tenor}",
                        'instrument_type': 'CCY Swaps',
                    })
                    used_pillars.add(pillar)
                    print(f"  {tenor:<6}  basis = {basis_bps:+.1f} bps  "
                          f"-> pillar {pillar}")
                else:
                    print(f"  {tenor:<6}  SKIP (duplicate pillar {pillar})")
            except Exception as exc:
                print(f"  {tenor:<6}  ERROR building CCY Swap helper: {exc}")
                # Record the failed helper in meta so calibration errors report it
                self._helper_meta.append({
                    'label':           f"CCY Swap {tenor}",
                    'instrument_type': 'CCY Swaps',
                    'build_error':     str(exc),
                })

        # ---- Bootstrap ----
        print(f"\nBootstrapping with {len(self._helpers)} helpers...")
        self.eur_basis_curve = ql.PiecewiseLogLinearDiscount(
            self.eval_date, self._helpers, ql.Actual360()
        )
        self.eur_basis_curve.enableExtrapolation()

        print(f"\n\u2705 EUR basis curve bootstrapped: {len(self._helpers)} pillars")
        print("\nSample EUR basis discount factors:")
        for t in [0.25, 0.5, 1, 2, 5, 10, 30]:
            try:
                df  = self.eur_basis_curve.discount(t)
                zr  = self.eur_basis_curve.zeroRate(t, ql.Continuous).rate() * 100
                print(f"   {t:>4}Y: df = {df:.6f}  z = {zr:.4f}%")
            except Exception:
                pass

        print("\n" + "="*60)
        print("\u2705 CCY BASIS CURVE BOOTSTRAPPED SUCCESSFULLY")
        print("="*60)

    # ------------------------------------------------------------------
    # CALIBRATION ERRORS
    # ------------------------------------------------------------------

    def get_basis_calibration_errors(self):
        """
        Return calibration errors for all CCY basis instruments.

        Uses helper.impliedQuote() vs helper.quote().value() - the same
        pattern as YieldCurveBuilder.get_calibration_errors().

        FX Swaps  : quote = forward points (F-S); unit = price, error in bps
        CCY Swaps : quote = basis spread (decimal); error in bps

        Failed helpers (build errors or impliedQuote exceptions) are reported
        with NaN values so they appear in the output rather than being silently
        dropped.

        Returns
        -------
        pandas.DataFrame
            Columns: instrument, instrument_type, market_rate, model_rate, error_bps
        """
        if self.eur_basis_curve is None:
            raise ValueError("Basis curve not bootstrapped yet.")

        rows = []

        # Index into self._helpers by counting only entries without build_error
        helper_idx = 0
        for meta in self._helper_meta:
            # Skip meta entries that represent failed builds (no helper object)
            if 'build_error' in meta:
                rows.append({
                    'instrument':      meta['label'],
                    'instrument_type': meta['instrument_type'],
                    'market_rate':     float('nan'),
                    'model_rate':      float('nan'),
                    'error_bps':       float('nan'),
                    'note':            f"build error: {meta['build_error']}",
                })
                continue

            helper = self._helpers[helper_idx]
            helper_idx += 1
            inst   = meta['instrument_type']

            try:
                mkt   = helper.quote().value()
                model = helper.impliedQuote()

                if inst == 'FX Swaps':
                    # quotes in forward-point units (outright - spot)
                    # convert to pips for display; error in bps proxy
                    error_bps   = (model - mkt) * 10_000
                    market_disp = mkt   * 10_000
                    model_disp  = model * 10_000
                else:
                    # quotes are decimal basis spreads
                    error_bps   = (model - mkt) * 10_000
                    market_disp = mkt   * 10_000
                    model_disp  = model * 10_000

                rows.append({
                    'instrument':      meta['label'],
                    'instrument_type': inst,
                    'market_rate':     market_disp,
                    'model_rate':      model_disp,
                    'error_bps':       error_bps,
                    'note':            '',
                })
            except Exception as exc:
                rows.append({
                    'instrument':      meta['label'],
                    'instrument_type': inst,
                    'market_rate':     float('nan'),
                    'model_rate':      float('nan'),
                    'error_bps':       float('nan'),
                    'note':            f"impliedQuote error: {exc}",
                })

        df = pd.DataFrame(rows)
        # Return all rows; caller can filter on error_bps.isna() to find failures
        return df

    # ------------------------------------------------------------------
    # CURVE OUTPUT METHODS
    # ------------------------------------------------------------------

    def get_basis_adjusted_forward(self, tenor_years):
        """
        Basis-adjusted FX forward using the bootstrapped EUR basis curve.
        F = spot * df_eur_basis / df_usd

        With USD SOFR > EUR ESTR, df_usd < df_eur at all tenors, so
        the standard forward is above spot (EUR appreciates in fwd terms).
        The basis-adjusted forward adds the negative EUR/USD xccy basis,
        which reduces the EUR discount factor slightly, so the adjusted
        forward should be a touch below the standard forward.
        """
        if self.eur_basis_curve is None:
            raise ValueError("Basis curve not bootstrapped yet.")

        df_usd        = self.usd_curve.discount(tenor_years)
        df_eur        = self.eur_curve.discount(tenor_years)
        df_eur_basis  = self.eur_basis_curve.discount(tenor_years)

        standard_forward = self.spot_fx * df_eur       / df_usd
        adjusted_forward = self.spot_fx * df_eur_basis / df_usd

        zr_usd_basis = self.usd_curve.zeroRate(tenor_years, ql.Continuous).rate()
        zr_eur_basis = self.eur_basis_curve.zeroRate(tenor_years, ql.Continuous).rate()
        basis_bps    = (zr_usd_basis - zr_eur_basis) * 10_000  # rough proxy

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

    # ------------------------------------------------------------------
    # INTERNAL
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
            raise ValueError(f"Unsupported tenor: '{tenor_str}'")
