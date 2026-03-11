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
                         fx_spot_handle=None,        # unused by this helper; kept for API compat
                         calendar=None, bdc=ql.ModifiedFollowing,
                         end_of_month=True,
                         settlement_days=2):
        """
        Create a ql.CrossCurrencyBasisSwapRateHelper.

        Correct QL signature (QuantLib >= 1.15)
        ----------------------------------------
        CrossCurrencyBasisSwapRateHelper(
            basis,                               # QuoteHandle - spread in decimal
            tenor,                               # Period
            fixingDays,                          # int
            calendar,                            # Calendar
            convention,                          # BusinessDayConvention
            endOfMonth,                          # bool
            baseCurrencyIndex,                   # IborIndex  (USD - known curve)
            quoteCurrencyIndex,                  # IborIndex  (EUR - being bootstrapped)
            collateralCurve,                     # YieldTermStructureHandle (USD discount)
            isFxBaseCurrencyCollateralCurrency,  # bool: True  (USD is base of EUR/USD)
            isBasisOnFxBaseCurrencyLeg,          # bool: False (basis on EUR/quote leg)
        )

        The EUR basis curve is bootstrapped by attaching a
        RelinkableYieldTermStructureHandle to the quoteCurrencyIndex;
        QL relinks it internally during the bootstrap.

        Market convention for EUR/USD
        -----------------------------
        - FX base currency = USD  -> isFxBaseCurrencyCollateralCurrency = True
        - Basis quoted on EUR leg -> isBasisOnFxBaseCurrencyLeg = False
        - Quoted basis is negative (EUR trades at a discount vs CIP)

        Parameters
        ----------
        tenor : str  e.g. '2Y', '5Y', '10Y', '30Y'
        basis_spread_bps : float  Basis spread in bps (negative for EUR/USD)
        usd_curve_handle : ql.YieldTermStructureHandle  USD discount (known)
        eur_curve_handle : ql.YieldTermStructureHandle  EUR ESTR discount
        fx_spot_handle   : ignored (kept for call-site API compatibility)
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

        # Ibor indices tied to the respective discount curves.
        # USDLibor(3M) / Euribor3M are the standard QL proxies;
        # with OIS discounting the index projection curve doesn't affect
        # bootstrapping - only the discount curves matter.
        usd_ibor = ql.USDLibor(ql.Period('3M'), usd_curve_handle)

        # EUR: attach a relinkable handle so QL can relink during bootstrap
        eur_relinkable = ql.RelinkableYieldTermStructureHandle()
        eur_relinkable.linkTo(eur_curve_handle.currentLink())
        eur_ibor = ql.Euribor3M(eur_relinkable)

        # collateralCurve = USD (the known/collateral leg in EUR/USD)
        # isFxBaseCurrencyCollateralCurrency = True   (USD is base of EUR/USD)
        # isBasisOnFxBaseCurrencyLeg         = False  (basis on EUR quote leg)
        helper = ql.CrossCurrencyBasisSwapRateHelper(
            basis_quote,
            period,
            settlement_days,
            calendar,
            bdc,
            end_of_month,
            usd_ibor,           # baseCurrencyIndex  (USD - known)
            eur_ibor,           # quoteCurrencyIndex (EUR - bootstrapped)
            usd_curve_handle,   # collateralCurve    (USD discount)
            True,               # isFxBaseCurrencyCollateralCurrency
            False,              # isBasisOnFxBaseCurrencyLeg
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
