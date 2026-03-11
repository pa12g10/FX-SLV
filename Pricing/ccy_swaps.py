# Cross-Currency Swap Pricer
import QuantLib as ql
import numpy as np


class CCYSwapPricer:
    """
    Pricer for Mark-to-Market Cross-Currency Basis Swaps (EUR/USD).

    Market convention for EUR/USD xccy basis:
        pay EUR ESTR + basis (negative), receive USD SOFR flat
        i.e. the basis spread is ON THE EUR/other leg  -> basisOnBase = False
        The negative basis (-22 to -29 bps) means EUR pays ESTR - 22 bps.

    MtMCrossCurrencyBasisSwapRateHelper flags:
        baseIsCollateral = True   USD is FX base (EUR/USD) and collateral
        basisOnBase      = False  basis is on the EUR/other leg (NOT USD)
        baseResets       = True   USD notional resets each period (MtM)

    Effect: negative basis quote -> EUR basis zero rates ~22-29 bps ABOVE
    flat ESTR -> df_eur_basis < df_eur -> adjusted_forward < standard_forward
    The CCY basis chart should read ~-20 to -29 bps in the long end.
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
                         calendar=None, bdc=ql.ModifiedFollowing,
                         end_of_month=True,
                         settlement_days=2):
        """
        Create a ql.MtMCrossCurrencyBasisSwapRateHelper.

        QL signature (QuantLib >= 1.31 pip bindings):
        -----------------------------------------------
        MtMCrossCurrencyBasisSwapRateHelper(
            basis,              QuoteHandle  decimal spread
            tenor,              Period
            fixingDays,         int
            calendar,           Calendar
            convention,         BusinessDayConvention
            endOfMonth,         bool
            baseIndex,          OvernightIndex  (USD - known)
            otherIndex,         IborIndex       (EUR - bootstrapped)
            collateralCurve,    YieldTermStructureHandle (USD)
            baseIsCollateral,   bool
            basisOnBase,        bool
            baseResets,         bool
            frequency,          ql.Frequency
        )

        Flag values for EUR/USD (basis on EUR leg, negative quote)
        ------------------------------------------------------------
        baseIsCollateral = True   USD is FX base of EUR/USD AND collateral
        basisOnBase      = False  basis is on the EUR (other) leg
                                  negative quote => EUR pays ESTR - |basis|
                                  => EUR basis zero rate > flat ESTR
                                  => df_eur_basis < df_eur (correct)
        baseResets       = True   USD notional resets each period (MtM)

        Parameters
        ----------
        tenor            : str    e.g. '2Y', '5Y', '10Y', '30Y'
        basis_spread_bps : float  Basis spread in bps (negative for EUR/USD)
        usd_curve_handle : ql.YieldTermStructureHandle  USD SOFR (known)
        eur_curve_handle : ql.YieldTermStructureHandle  EUR ESTR (bootstrapped)
        fx_spot_handle   : ignored, kept for call-site API compatibility
        """
        if calendar is None:
            calendar = ql.JointCalendar(
                ql.TARGET(),
                ql.UnitedStates(ql.UnitedStates.FederalReserve)
            )

        basis_quote = ql.QuoteHandle(
            ql.SimpleQuote(basis_spread_bps / 10_000.0)
        )

        period = self._parse_tenor(tenor)

        # USD base index tied to the known USD curve
        sofr_index = ql.Sofr(usd_curve_handle)

        # EUR other index with relinkable handle so QL relinks during bootstrap
        eur_relinkable = ql.RelinkableYieldTermStructureHandle()
        eur_relinkable.linkTo(eur_curve_handle.currentLink())
        euribor_index = ql.Euribor3M(eur_relinkable)

        helper = ql.MtMCrossCurrencyBasisSwapRateHelper(
            basis_quote,
            period,
            settlement_days,
            calendar,
            bdc,
            end_of_month,
            sofr_index,          # baseIndex        (USD - known)
            euribor_index,       # otherIndex       (EUR - bootstrapped)
            usd_curve_handle,    # collateralCurve  (USD)
            True,                # baseIsCollateral (USD is base AND collateral)
            False,               # basisOnBase=False (basis on EUR/other leg)
            True,                # baseResets=True  (USD notional resets - MtM)
            ql.Quarterly,        # frequency
        )
        return helper

    def _parse_tenor(self, tenor_str):
        tenor_str = tenor_str.upper().strip()
        if tenor_str.endswith('M'):
            return ql.Period(int(tenor_str[:-1]), ql.Months)
        elif tenor_str.endswith('Y'):
            return ql.Period(int(tenor_str[:-1]), ql.Years)
        else:
            raise ValueError(f"Unsupported tenor: '{tenor_str}'")
