# Cross-Currency Swap Pricer
import QuantLib as ql
import numpy as np


class CCYSwapPricer:
    """
    Pricer for Mark-to-Market Cross-Currency Basis Swaps (EUR/USD).

    Convention: pay EUR ESTR flat, receive USD SOFR + basis spread.
    EUR notional resets at prevailing spot each coupon date (MtM reset).

    create_ql_helper() produces a ql.CrossCurrencyBasisSwapRateHelper
    for ql.PiecewiseLogLinearDiscount bootstrapping.
    """

    def __init__(self, eval_date, spot_fx):
        self.eval_date = eval_date
        self.spot_fx   = spot_fx
        ql.Settings.instance().evaluationDate = eval_date

    # ------------------------------------------------------------------
    # QL HELPER  (used by FXCurves bootstrapper)
    # ------------------------------------------------------------------

    def create_ql_helper(self, tenor, basis_spread_bps,
                         usd_curve_handle, eur_curve_handle,
                         fx_spot_handle=None,
                         calendar=None, bdc=ql.Following,
                         end_of_month=False,
                         settlement_days=2,
                         is_resetting_notional=True):
        """
        Create a ql.CrossCurrencyBasisSwapRateHelper.

        QuantLib convention
        -------------------
        CrossCurrencyBasisSwapRateHelper bootstraps the *foreign* (EUR)
        discount curve given the domestic (USD) discount curve and the
        quoted basis spread. The helper solves for the EUR discount factor
        that makes each MtM CCY swap reprice at par.

        The curve being bootstrapped is the EUR *basis* discount curve
        (EUR ESTR + basis), not the flat EUR ESTR curve.

        Parameters
        ----------
        tenor : str
            e.g. '2Y', '5Y', '10Y', '30Y'
        basis_spread_bps : float
            Quoted basis spread in basis points (negative for EUR/USD)
        usd_curve_handle : ql.YieldTermStructureHandle
            USD SOFR discount curve
        eur_curve_handle : ql.YieldTermStructureHandle
            EUR ESTR discount curve (flat, no basis)
        fx_spot_handle : ql.QuoteHandle, optional
            EUR/USD spot handle (defaults to self.spot_fx)
        calendar : ql.Calendar, optional
        bdc : ql.BusinessDayConvention
        end_of_month : bool
        settlement_days : int
        is_resetting_notional : bool
            True  -> MtM reset (standard market convention for EUR/USD)
            False -> fixed notional

        Returns
        -------
        ql.CrossCurrencyBasisSwapRateHelper
        """
        if calendar is None:
            calendar = ql.JointCalendar(
                ql.TARGET(),
                ql.UnitedStates(ql.UnitedStates.FederalReserve)
            )

        if fx_spot_handle is None:
            fx_spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))

        basis_quote = ql.QuoteHandle(
            ql.SimpleQuote(basis_spread_bps / 10_000.0)  # decimal
        )

        period = self._parse_tenor(tenor)

        # ql.CrossCurrencyBasisSwapRateHelper signature:
        # (basis, tenor, settlementDays, calendar, bdc, endOfMonth,
        #  baseCcyIdx, baseCcyDiscountCurve,
        #  quoteCcyIdx, quoteCcyDiscountCurve,
        #  fxSpot, resettingNotional=True)
        #
        # base  = USD (the currency whose curve is KNOWN = usd_curve_handle)
        # quote = EUR (the curve being BOOTSTRAPPED)
        #
        # Both OIS indices are used only for their tenor/day-count;
        # we pass flat proxies tied to the known discount curves.

        usd_ois_idx = ql.OvernightIndex(
            'SOFR', settlement_days, ql.USDCurrency(),
            ql.UnitedStates(ql.UnitedStates.FederalReserve),
            ql.Actual360(), usd_curve_handle
        )
        eur_ois_idx = ql.OvernightIndex(
            'ESTR', settlement_days, ql.EURCurrency(),
            ql.TARGET(), ql.Actual360(), eur_curve_handle
        )

        helper = ql.CrossCurrencyBasisSwapRateHelper(
            basis_quote,
            period,
            settlement_days,
            calendar,
            bdc,
            end_of_month,
            usd_ois_idx,          # base ccy index (known curve)
            usd_curve_handle,     # base ccy discount
            eur_ois_idx,          # quote ccy index (being bootstrapped)
            fx_spot_handle,
            is_resetting_notional,
        )
        return helper

    # ------------------------------------------------------------------
    # ANALYTICS
    # ------------------------------------------------------------------

    def price_ccy_swap(self, tenor, notional_eur, basis_spread_bps,
                       usd_curve, eur_curve,
                       coupon_freq=ql.Quarterly,
                       day_count='Actual/360'):
        """
        Analytically price a (simplified) CCY basis swap using QL discount
        curves. Returns NPV in USD.

        Structure
        ---------
        - EUR leg: pay floating ESTR on EUR notional (approximated as
          (df_start/df_end - 1)/T coupons)
        - USD leg: receive floating SOFR + basis on USD notional
        - Initial + final notional exchange
        """
        period        = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        dc            = ql.Actual360() if day_count == 'Actual/360' else ql.Actual365Fixed()
        notional_usd  = notional_eur * self.spot_fx
        basis_dec     = basis_spread_bps / 10_000.0

        freq_period = ql.Period(coupon_freq)
        schedule = ql.Schedule(
            self.eval_date, maturity_date, freq_period,
            ql.JointCalendar(ql.TARGET(),
                             ql.UnitedStates(ql.UnitedStates.FederalReserve)),
            ql.Following, ql.Following,
            ql.DateGeneration.Forward, False
        )

        eur_leg_npv = 0.0
        usd_leg_npv = 0.0

        for i in range(len(schedule) - 1):
            s, e   = schedule[i], schedule[i + 1]
            yf     = dc.yearFraction(s, e)

            # EUR floating coupon
            dfs_e  = eur_curve.discount(s)
            dfe_e  = eur_curve.discount(e)
            fwd_e  = (dfs_e / dfe_e - 1.0) / yf
            eur_leg_npv += notional_eur * fwd_e * yf * dfe_e

            # USD floating + basis coupon
            dfs_u  = usd_curve.discount(s)
            dfe_u  = usd_curve.discount(e)
            fwd_u  = (dfs_u / dfe_u - 1.0) / yf
            usd_leg_npv += notional_usd * (fwd_u + basis_dec) * yf * dfe_u

        df_mat_eur = eur_curve.discount(maturity_date)
        df_mat_usd = usd_curve.discount(maturity_date)

        # Final notional re-exchange
        final_npv  = (notional_usd - notional_eur * self.spot_fx) * df_mat_usd

        npv_eur_usd = eur_leg_npv * self.spot_fx
        total_npv   = -npv_eur_usd + usd_leg_npv + final_npv

        return {
            'notional_eur':       notional_eur,
            'notional_usd':       notional_usd,
            'spot_fx':            self.spot_fx,
            'basis_spread_bps':   basis_spread_bps,
            'eur_leg_npv':        eur_leg_npv,
            'eur_leg_npv_usd':    npv_eur_usd,
            'usd_leg_npv':        usd_leg_npv,
            'final_exchange_npv': final_npv,
            'total_npv':          total_npv,
            'df_maturity_eur':    df_mat_eur,
            'df_maturity_usd':    df_mat_usd,
        }

    def calculate_par_basis(self, tenor, notional_eur,
                            usd_curve, eur_curve,
                            coupon_freq=ql.Quarterly,
                            day_count='Actual/360'):
        """
        Bisection solver: find basis spread (bps) that makes CCY swap NPV = 0.
        """
        lo, hi = -200.0, 200.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            npv = self.price_ccy_swap(
                tenor, notional_eur, mid, usd_curve, eur_curve,
                coupon_freq, day_count
            )['total_npv']
            if abs(npv) < 1e-6:
                return mid
            if npv > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _parse_tenor(self, tenor_str):
        tenor_str = tenor_str.upper().strip()
        if tenor_str.endswith('M'):
            return ql.Period(int(tenor_str[:-1]), ql.Months)
        elif tenor_str.endswith('Y'):
            return ql.Period(int(tenor_str[:-1]), ql.Years)
        else:
            raise ValueError(f"Unsupported tenor: '{tenor_str}'")
