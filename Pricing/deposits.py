# Deposit Pricer
import QuantLib as ql

class DepositPricer:
    """
    Pricer for Overnight Deposits (SOFR, ESTR, etc.)
    """
    
    def __init__(self, eval_date):
        """
        Initialize deposit pricer
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Valuation date
        """
        self.eval_date = eval_date
        ql.Settings.instance().evaluationDate = eval_date
    
    def create_helper(self, rate, day_count='Actual/360'):
        """
        Create deposit rate helper for curve bootstrapping
        
        Parameters:
        -----------
        rate : float
            Deposit rate in percentage (e.g., 4.58 for 4.58%)
        day_count : str
            Day count convention
        
        Returns:
        --------
        QuantLib.DepositRateHelper
        """
        # Convert rate to decimal
        rate_decimal = rate / 100.0
        
        # Day count convention
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        elif day_count == 'Actual/365':
            dc = ql.Actual365Fixed()
        else:
            dc = ql.Actual360()
        
        # Create deposit rate helper (overnight)
        helper = ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate_decimal)),
            ql.Period(1, ql.Days),  # Overnight
            1,  # Settlement days
            ql.TARGET(),
            ql.Following,
            False,
            dc
        )
        
        return helper
    
    def price_deposit(self, rate, notional, start_date, end_date, day_count='Actual/360'):
        """
        Price a deposit trade
        
        Parameters:
        -----------
        rate : float
            Deposit rate in percentage
        notional : float
            Notional amount
        start_date : QuantLib.Date
            Start date
        end_date : QuantLib.Date
            End date
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Pricing results with interest and maturity value
        """
        # Day count convention
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        elif day_count == 'Actual/365':
            dc = ql.Actual365Fixed()
        else:
            dc = ql.Actual360()
        
        # Calculate year fraction
        year_fraction = dc.yearFraction(start_date, end_date)
        
        # Calculate interest
        rate_decimal = rate / 100.0
        interest = notional * rate_decimal * year_fraction
        maturity_value = notional + interest
        
        return {
            'notional': notional,
            'rate': rate,
            'start_date': start_date,
            'end_date': end_date,
            'year_fraction': year_fraction,
            'interest': interest,
            'maturity_value': maturity_value
        }
