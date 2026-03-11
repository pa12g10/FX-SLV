# FX Forward Pricer
import QuantLib as ql
import numpy as np


class FXForwardPricer:
    """
    Pricer for FX Forward contracts and FX Swaps.

    create_ql_helper() produces a ql.FxSwapRateHelper that feeds directly
    into ql.PiecewiseLogLinearDiscount for CCY basis curve bootstrapping.
    """

    def __init__(self, eval_date, spot_fx):
        self.eval_date = eval_date
        self.spot_fx   = spot_fx
        ql.Settings.instance().evaluationDate = eval_date

    # ------------------------------------------------------------------
    # QL HELPER  (used by FXCurves bootstrapper)
    # ------------------------------------------------------------------

    def create_ql_helper(self, tenor, forward_points,
                         usd_curve_handle, eur_curve_handle,
                         spot_date=None, calendar=None,
                         bdc=ql.Following, end_of_month=False,
                         is_fx_base_currency_collateral_currency=False):
        """
        Create a ql.FxSwapRateHelper for bootstrapping the CCY basis curve.

        QuantLib convention for FxSwapRateHelper
        -----------------------------------------
        The helper takes the *forward points* (F - S), the spot quote,
        the tenor, and the two collateral/discount curves.  It solves for
        the discount curve that makes the FX swap reprice exactly.

        Parameters
        ----------
        tenor : str
            e.g. '1W', '1M', '6M', '18M'
        forward_points : float
            Market forward points (outright - spot), e.g. -0.014500 for 1M
        usd_curve_handle : ql.YieldTermStructureHandle
            USD SOFR discount curve (domestic / collateral)
        eur_curve_handle : ql.YieldTermStructureHandle
            EUR ESTR discount curve (foreign)
        spot_date : ql.Date, optional
            Spot date (T+2 by default)
        calendar : ql.Calendar, optional
            Joint USD+EUR calendar (defaults to TARGET + UnitedStates)
        bdc : ql.BusinessDayConvention
        end_of_month : bool
        is_fx_base_currency_collateral_currency : bool
            True  -> USD is collateral (standard for USD-collateralised EUR/USD)
            False -> EUR is collateral

        Returns
        -------
        ql.FxSwapRateHelper
        """
        if calendar is None:
            calendar = ql.JointCalendar(
                ql.TARGET(),
                ql.UnitedStates(ql.UnitedStates.FederalReserve)
            )

        # Spot date: T+2 good business days
        if spot_date is None:
            spot_date = calendar.advance(
                self.eval_date, ql.Period(2, ql.Days), bdc
            )

        period = self._parse_tenor(tenor)

        spot_quote = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))
        fwd_quote  = ql.QuoteHandle(ql.SimpleQuote(forward_points))

        helper = ql.FxSwapRateHelper(
            fwd_quote,
            spot_quote,
            period,
            2,                  # fixing days
            calendar,
            bdc,
            end_of_month,
            is_fx_base_currency_collateral_currency,
            usd_curve_handle,
        )
        return helper

    # ------------------------------------------------------------------
    # ANALYTICS
    # ------------------------------------------------------------------

    def calculate_forward_rate(self, tenor, domestic_curve, foreign_curve,
                               day_count='Actual/360'):
        """
        Theoretical CIP forward: F = S * df_eur / df_usd
        """
        period        = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period

        dc = ql.Actual360() if day_count == 'Actual/360' else ql.Actual365Fixed()
        time_to_mat   = dc.yearFraction(self.eval_date, maturity_date)

        df_domestic = domestic_curve.discount(maturity_date)
        df_foreign  = foreign_curve.discount(maturity_date)

        r_domestic  = domestic_curve.zeroRate(time_to_mat, ql.Continuous).rate()
        r_foreign   = foreign_curve.zeroRate(time_to_mat, ql.Continuous).rate()

        forward_rate     = self.spot_fx * df_foreign / df_domestic
        forward_rate_alt = self.spot_fx * np.exp((r_domestic - r_foreign) * time_to_mat)
        forward_points   = (forward_rate - self.spot_fx) * 10_000

        return {
            'tenor':            tenor,
            'maturity_date':    maturity_date,
            'time_to_maturity': time_to_mat,
            'spot_rate':        self.spot_fx,
            'forward_rate':     forward_rate,
            'forward_rate_alt': forward_rate_alt,
            'forward_points':   forward_points,
            'domestic_rate':    r_domestic * 100,
            'foreign_rate':     r_foreign  * 100,
            'df_domestic':      df_domestic,
            'df_foreign':       df_foreign,
        }

    def price_fx_forward(self, tenor, strike, notional,
                         domestic_curve, foreign_curve,
                         is_buy=True, day_count='Actual/360'):
        """
        Price an FX forward contract (NPV vs struck rate).
        """
        fwd           = self.calculate_forward_rate(tenor, domestic_curve, foreign_curve, day_count)
        fair_forward  = fwd['forward_rate']
        maturity_date = fwd['maturity_date']
        df_domestic   = domestic_curve.discount(maturity_date)

        sign             = 1 if is_buy else -1
        payoff_maturity  = sign * notional * (fair_forward - strike)
        npv              = payoff_maturity * df_domestic

        return {
            'notional':               notional,
            'strike':                 strike,
            'fair_forward':           fair_forward,
            'forward_points_strike':  (strike       - self.spot_fx) * 10_000,
            'forward_points_fair':    fwd['forward_points'],
            'points_diff':            (fair_forward - strike)       * 10_000,
            'payoff_at_maturity':     payoff_maturity,
            'npv':                    npv,
            'df_domestic':            df_domestic,
            'is_buy':                 is_buy,
        }

    def implied_yield_differential(self, tenor, forward_rate, day_count='Actual/360'):
        """
        (r_dom - r_for) = ln(F/S) / T
        """
        period        = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        dc            = ql.Actual360() if day_count == 'Actual/360' else ql.Actual365Fixed()
        time_to_mat   = dc.yearFraction(self.eval_date, maturity_date)
        yield_diff    = np.log(forward_rate / self.spot_fx) / time_to_mat

        return {
            'tenor':              tenor,
            'time_to_maturity':   time_to_mat,
            'spot_rate':          self.spot_fx,
            'forward_rate':       forward_rate,
            'implied_yield_diff': yield_diff * 100,
            'forward_points':     (forward_rate - self.spot_fx) * 10_000,
        }

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _parse_tenor(self, tenor_str):
        tenor_str = tenor_str.upper().strip()
        if tenor_str.endswith('W'):
            return ql.Period(int(tenor_str[:-1]), ql.Weeks)
        elif tenor_str.endswith('M'):
            return ql.Period(int(tenor_str[:-1]), ql.Months)
        elif tenor_str.endswith('Y'):
            return ql.Period(int(tenor_str[:-1]), ql.Years)
        else:
            raise ValueError(f"Unsupported tenor: '{tenor_str}'")
