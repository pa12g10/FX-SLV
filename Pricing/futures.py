# Futures Pricer
import QuantLib as ql

class FuturesPricer:
    """
    Pricer for Interest Rate Futures (SOFR, ESTR, etc.)
    """
    
    def __init__(self, eval_date):
        """
        Initialize futures pricer
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Valuation date
        """
        self.eval_date = eval_date
        ql.Settings.instance().evaluationDate = eval_date
    
    def create_helper(self, maturity, price, day_count='Actual/360'):
        """
        Create futures rate helper for curve bootstrapping
        
        Parameters:
        -----------
        maturity : str
            Maturity tenor (e.g., '3M', '6M', '1Y')
        price : float
            Futures price (e.g., 95.25)
        day_count : str
            Day count convention
        
        Returns:
        --------
        QuantLib.FuturesRateHelper
        """
        # Parse maturity tenor
        period = self._parse_tenor(maturity)
        
        # Implied rate from futures price (100 - price)
        implied_rate = (100 - price) / 100.0
        
        # Day count convention
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        elif day_count == 'Actual/365':
            dc = ql.Actual365Fixed()
        else:
            dc = ql.Actual360()
        
        # Calculate IMM date (futures settlement) based on maturity period
        imm_date = self._get_imm_date(period)
        
        # Create futures rate helper
        helper = ql.FuturesRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(price)),
            imm_date,
            3,  # 3-month futures
            ql.TARGET(),
            ql.Following,
            True,
            dc
        )
        
        return helper
    
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
        else:
            raise ValueError(f"Invalid tenor format: {tenor_str}")
    
    def _get_imm_date(self, period):
        """
        Calculate IMM date (third Wednesday) for futures settlement
        Futures settle on IMM dates, but for different tenors we need different IMM dates
        
        For a period (e.g., 3M), find the IMM date that's approximately that far from eval_date
        """
        # Calculate approximate target date
        target_date = self.eval_date + period
        
        # Find the IMM date on or after the target date
        # IMM dates are 3rd Wednesday of Mar, Jun, Sep, Dec
        imm_date = ql.IMM.nextDate(target_date - ql.Period(1, ql.Months))
        
        # If that's too far in the past, get the next one
        while imm_date < self.eval_date:
            imm_date = ql.IMM.nextDate(imm_date + ql.Period(1, ql.Days))
        
        return imm_date
    
    def price_futures(self, price, notional=1000000, tick_value=25):
        """
        Price a futures contract
        
        Parameters:
        -----------
        price : float
            Futures price
        notional : float
            Contract notional (default $1M)
        tick_value : float
            Value per basis point
        
        Returns:
        --------
        dict: Pricing results
        """
        implied_rate = (100 - price) / 100.0
        
        return {
            'price': price,
            'implied_rate': implied_rate * 100,  # As percentage
            'notional': notional,
            'tick_value': tick_value,
            'contract_value': notional * (1 + implied_rate * 0.25)  # Quarterly
        }
