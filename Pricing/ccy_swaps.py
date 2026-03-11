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
        Full QL signature:
            CrossCurrencyBasisSwapRateHelper(
                basis,              # QuoteHandle - basis spread (decimal)
                tenor,              # Period
                settlementDays,     # int
                calendar,           # Calendar
                bdc,                # BusinessDayConvention
                endOfMonth,         # bool
                baseCcyIdx,         # OvernightIndex  <- USD (KNOWN curve)
                baseCcyDiscountCurve,  # YieldTermStructureHandle  <- USD discount
                quoteCcyIdx,        # OvernightIndex  <- EUR (curve being BOOTSTRAPPED)
                quoteCcyDiscountCurve, # YieldTermStructureHandle  <- EUR discount
                fxSpot,             # QuoteHandle
                resettingNotional,  # bool
            )

        base  = USD: the currency whose discount curve is already KNOWN.
        quote = EUR: the currency whose discount curve is being BOOTSTRAPPED.

        Passing eur_curve_handle as quoteCcyDiscountCurve is REQUIRED so that
        QL builds a *relative* curve on top of EUR ESTR; without it QL was
        bootstrapping the USD side and producing 1/df instead of df, causing
        the EUR basis curve to slope steeply downward.

        Parameters
        ----------
        tenor : str
            e.g. '2Y', '5Y', '10Y', '30Y'
        basis_spread_bps : float
            Quoted basis spread in basis points (negative for EUR/USD)
        usd_curve_handle : ql.YieldTermStructureHandle
            USD SOFR discount curve (base - known)
        eur_curve_handle : ql.YieldTermStructureHandle
            EUR ESTR discount curve (quote - being bootstrapped)
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

        # base  = USD (known)  -> usd_ois_idx + usd_curve_handle
        # quote = EUR (bootstrapped) -> eur_ois_idx + eur_curve_handle
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
            usd_ois_idx,          # base  ccy index  (USD - known)
            usd_curve_handle,     # base  ccy discount curve (USD)
            eur_ois_idx,          # quote ccy index  (EUR - bootstrapped)
            eur_curve_handle,     # quote ccy discount curve (EUR)  <-- FIX: was missing
            fx_spot_handle,
            is_resetting_notional,
        )
        return helper

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
