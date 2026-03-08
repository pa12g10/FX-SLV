# Swap Pricer
import QuantLib as ql

class SwapPricer:
    """
    Pricer for OIS Swaps (SOFR, ESTR, etc.)
    """
    
    def __init__(self, eval_date):
        """
        Initialize swap pricer
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Valuation date
        """
        self.eval_date = eval_date
        ql.Settings.instance().evaluationDate = eval_date
    
    def create_helper(self, tenor, rate, curve_handle, fixed_freq='Annual', 
                     float_freq='Annual', day_count='Actual/360'):
        """
        Create OIS rate helper for curve bootstrapping
        
        Parameters:
        -----------
        tenor : str
            Swap tenor (e.g., '2Y', '5Y', '10Y')
        rate : float
            Fixed rate in percentage
        curve_handle : QuantLib.YieldTermStructureHandle
            Discount curve handle
        fixed_freq : str
            Fixed leg frequency
        float_freq : str
            Float leg frequency
        day_count : str
            Day count convention
        
        Returns:
        --------
        QuantLib.OISRateHelper
        """
        # Parse tenor
        period = self._parse_tenor(tenor)
        
        # Convert rate to decimal
        rate_decimal = rate / 100.0
        
        # Payment frequencies
        fixed_leg_freq = self._parse_frequency(fixed_freq)
        
        # Day count convention
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        elif day_count == 'Actual/365':
            dc = ql.Actual365Fixed()
        else:
            dc = ql.Actual360()
        
        # Create OIS rate helper with correct signature:
        # OISRateHelper(settlementDays, tenor, quote, overnightIndex, 
        #               discountingCurve, telescopicValueDates,
        #               paymentLag, paymentConvention, paymentFrequency,
        #               paymentCalendar, forwardStart, overnightSpread, pillarChoice, customPillarDate)
        
        helper = ql.OISRateHelper(
            2,  # Settlement days
            period,  # Tenor as Period
            ql.QuoteHandle(ql.SimpleQuote(rate_decimal)),  # Rate quote
            ql.Sofr(),  # Overnight index (SOFR)
            curve_handle  # Discounting curve
            # Use defaults for remaining optional parameters
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
    
    def _parse_frequency(self, freq_str):
        """
        Parse frequency string to QuantLib Frequency
        """
        freq_map = {
            'Annual': ql.Annual,
            'Semiannual': ql.Semiannual,
            'Quarterly': ql.Quarterly,
            'Monthly': ql.Monthly
        }
        return freq_map.get(freq_str, ql.Annual)
    
    def price_swap(self, fixed_rate, tenor, notional, curve, is_payer=True,
                   fixed_freq='Annual', float_freq='Annual', day_count='Actual/360'):
        """
        Price an OIS swap
        
        Parameters:
        -----------
        fixed_rate : float
            Fixed rate in percentage
        tenor : str
            Swap tenor
        notional : float
            Notional amount
        curve : QuantLib.YieldTermStructure
            Discount curve
        is_payer : bool
            True for payer swap (pay fixed, receive float)
        fixed_freq : str
            Fixed leg frequency
        float_freq : str
            Float leg frequency
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Pricing results with NPV, fixed leg, float leg
        """
        # Parse inputs
        period = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        
        fixed_leg_freq = self._parse_frequency(fixed_freq)
        float_leg_freq = self._parse_frequency(float_freq)
        
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        else:
            dc = ql.Actual365Fixed()
        
        # Create schedule
        schedule = ql.Schedule(
            self.eval_date,
            maturity_date,
            ql.Period(fixed_leg_freq),
            ql.TARGET(),
            ql.Following,
            ql.Following,
            ql.DateGeneration.Forward,
            False
        )
        
        # Create OIS swap
        swap_type = ql.OvernightIndexedSwap.Payer if is_payer else ql.OvernightIndexedSwap.Receiver
        
        ois_index = ql.Sofr(ql.YieldTermStructureHandle(curve))
        
        ois_swap = ql.OvernightIndexedSwap(
            swap_type,
            notional,
            schedule,
            fixed_rate / 100.0,
            dc,
            ois_index
        )
        
        # Set pricing engine
        engine = ql.DiscountingSwapEngine(ql.YieldTermStructureHandle(curve))
        ois_swap.setPricingEngine(engine)
        
        # Get pricing results
        npv = ois_swap.NPV()
        fixed_leg_npv = ois_swap.fixedLegNPV()
        float_leg_npv = ois_swap.overnightLegNPV()
        fair_rate = ois_swap.fairRate() * 100  # As percentage
        
        return {
            'npv': npv,
            'fixed_leg_npv': fixed_leg_npv,
            'float_leg_npv': float_leg_npv,
            'fair_rate': fair_rate,
            'notional': notional,
            'fixed_rate': fixed_rate
        }
