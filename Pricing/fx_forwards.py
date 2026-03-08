# FX Forward Pricer
import QuantLib as ql
import numpy as np

class FXForwardPricer:
    """
    Pricer for FX Forward contracts using Covered Interest Rate Parity
    Forward = Spot * (1 + r_foreign * T) / (1 + r_domestic * T)
    """
    
    def __init__(self, eval_date, spot_fx):
        """
        Initialize FX forward pricer
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Valuation date
        spot_fx : float
            FX spot rate (e.g., EUR/USD)
        """
        self.eval_date = eval_date
        self.spot_fx = spot_fx
        ql.Settings.instance().evaluationDate = eval_date
    
    def create_helper(self, tenor, outright_rate, domestic_curve, foreign_curve, 
                     day_count='Actual/360'):
        """
        Create FX forward helper for FX curve bootstrapping
        
        Parameters:
        -----------
        tenor : str
            Forward tenor (e.g., '3M', '6M', '1Y')
        outright_rate : float
            Outright forward FX rate
        domestic_curve : QuantLib.YieldTermStructureHandle
            Domestic currency discount curve (USD)
        foreign_curve : QuantLib.YieldTermStructureHandle
            Foreign currency discount curve (EUR)
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Helper information for bootstrapping
        """
        # Parse tenor
        period = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        
        # Day count convention
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        elif day_count == 'Actual/365':
            dc = ql.Actual365Fixed()
        else:
            dc = ql.Actual360()
        
        # Calculate time to maturity
        time_to_maturity = dc.yearFraction(self.eval_date, maturity_date)
        
        return {
            'tenor': tenor,
            'maturity_date': maturity_date,
            'time_to_maturity': time_to_maturity,
            'outright_rate': outright_rate,
            'spot_rate': self.spot_fx,
            'forward_points': (outright_rate - self.spot_fx) * 10000  # pips
        }
    
    def _parse_tenor(self, tenor_str):
        """
        Parse tenor string to QuantLib Period
        """
        tenor_str = tenor_str.upper().strip()
        
        if tenor_str.endswith('M'):
            months = int(tenor_str[:-1])
            return ql.Period(months, ql.Months)
        elif tenor_str.endswith('Y'):
            years = int(tenor_str[:-1])
            return ql.Period(years, ql.Years)
        elif tenor_str.endswith('W'):
            weeks = int(tenor_str[:-1])
            return ql.Period(weeks, ql.Weeks)
        else:
            raise ValueError(f"Invalid tenor format: {tenor_str}")
    
    def calculate_forward_rate(self, tenor, domestic_curve, foreign_curve, day_count='Actual/360'):
        """
        Calculate theoretical forward FX rate using Covered Interest Parity
        
        F = S * (1 + r_for * T) / (1 + r_dom * T)
        Or with continuous rates: F = S * exp((r_dom - r_for) * T)
        
        Parameters:
        -----------
        tenor : str
            Forward tenor
        domestic_curve : QuantLib.YieldTermStructure
            Domestic currency curve (USD)
        foreign_curve : QuantLib.YieldTermStructure
            Foreign currency curve (EUR)
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Forward rate calculation details
        """
        # Parse tenor
        period = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        else:
            dc = ql.Actual365Fixed()
        
        time_to_maturity = dc.yearFraction(self.eval_date, maturity_date)
        
        # Get discount factors
        df_domestic = domestic_curve.discount(maturity_date)
        df_foreign = foreign_curve.discount(maturity_date)
        
        # Get zero rates
        r_domestic = domestic_curve.zeroRate(time_to_maturity, ql.Continuous).rate()
        r_foreign = foreign_curve.zeroRate(time_to_maturity, ql.Continuous).rate()
        
        # Calculate forward using discount factors
        forward_rate = self.spot_fx * df_foreign / df_domestic
        
        # Alternative: using continuous rates
        forward_rate_alt = self.spot_fx * np.exp((r_domestic - r_foreign) * time_to_maturity)
        
        # Forward points
        forward_points = (forward_rate - self.spot_fx) * 10000  # in pips
        
        return {
            'tenor': tenor,
            'maturity_date': maturity_date,
            'time_to_maturity': time_to_maturity,
            'spot_rate': self.spot_fx,
            'forward_rate': forward_rate,
            'forward_rate_alt': forward_rate_alt,
            'forward_points': forward_points,
            'domestic_rate': r_domestic * 100,
            'foreign_rate': r_foreign * 100,
            'df_domestic': df_domestic,
            'df_foreign': df_foreign
        }
    
    def price_fx_forward(self, tenor, strike, notional, domestic_curve, foreign_curve,
                        is_buy=True, day_count='Actual/360'):
        """
        Price an FX forward contract
        
        Parameters:
        -----------
        tenor : str
            Forward tenor
        strike : float
            Forward strike rate (contracted rate)
        notional : float
            Notional in foreign currency (EUR)
        domestic_curve : QuantLib.YieldTermStructure
            Domestic currency curve (USD)
        foreign_curve : QuantLib.YieldTermStructure
            Foreign currency curve (EUR)
        is_buy : bool
            True = buy EUR (receive EUR, pay USD), False = sell EUR
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Forward pricing results with NPV and mark-to-market
        """
        # Calculate fair forward rate
        forward_calc = self.calculate_forward_rate(tenor, domestic_curve, foreign_curve, day_count)
        fair_forward = forward_calc['forward_rate']
        maturity_date = forward_calc['maturity_date']
        
        # Get discount factor to present value
        df_domestic = domestic_curve.discount(maturity_date)
        
        # Calculate payoff at maturity
        if is_buy:
            # Buy EUR forward: receive EUR, pay USD at strike
            # P&L = notional * (fair_forward - strike) in USD
            payoff_at_maturity = notional * (fair_forward - strike)
        else:
            # Sell EUR forward: pay EUR, receive USD at strike
            # P&L = notional * (strike - fair_forward) in USD
            payoff_at_maturity = notional * (strike - fair_forward)
        
        # Present value (NPV)
        npv = payoff_at_maturity * df_domestic
        
        # Points difference
        points_diff = (fair_forward - strike) * 10000
        
        return {
            'notional': notional,
            'strike': strike,
            'fair_forward': fair_forward,
            'forward_points_strike': (strike - self.spot_fx) * 10000,
            'forward_points_fair': forward_calc['forward_points'],
            'points_diff': points_diff,
            'payoff_at_maturity': payoff_at_maturity,
            'npv': npv,
            'df_domestic': df_domestic,
            'is_buy': is_buy
        }
    
    def implied_yield_differential(self, tenor, forward_rate, day_count='Actual/360'):
        """
        Calculate implied yield differential from FX forward
        (r_dom - r_for) = ln(F / S) / T
        
        Parameters:
        -----------
        tenor : str
            Forward tenor
        forward_rate : float
            Forward FX rate
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Implied yield differential
        """
        period = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        else:
            dc = ql.Actual365Fixed()
        
        time_to_maturity = dc.yearFraction(self.eval_date, maturity_date)
        
        # Calculate implied yield differential
        yield_diff = np.log(forward_rate / self.spot_fx) / time_to_maturity
        
        return {
            'tenor': tenor,
            'time_to_maturity': time_to_maturity,
            'spot_rate': self.spot_fx,
            'forward_rate': forward_rate,
            'implied_yield_diff': yield_diff * 100,  # in %
            'forward_points': (forward_rate - self.spot_fx) * 10000
        }
