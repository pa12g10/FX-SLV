# Cross-Currency Swap Pricer
import QuantLib as ql
import numpy as np

class CCYSwapPricer:
    """
    Pricer for Cross-Currency Basis Swaps
    EUR 3M EURIBOR vs USD 3M SOFR + basis spread
    """
    
    def __init__(self, eval_date, spot_fx):
        """
        Initialize cross-currency swap pricer
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Valuation date
        spot_fx : float
            FX spot rate (EUR/USD)
        """
        self.eval_date = eval_date
        self.spot_fx = spot_fx
        ql.Settings.instance().evaluationDate = eval_date
    
    def create_helper(self, tenor, basis_spread, domestic_curve, foreign_curve,
                     eur_freq='Quarterly', usd_freq='Quarterly', day_count='Actual/360'):
        """
        Create CCY swap helper for basis curve bootstrapping
        
        Parameters:
        -----------
        tenor : str
            Swap tenor (e.g., '1Y', '5Y', '10Y')
        basis_spread : float
            Cross-currency basis spread in basis points
        domestic_curve : QuantLib.YieldTermStructureHandle
            USD discount curve
        foreign_curve : QuantLib.YieldTermStructureHandle
            EUR discount curve
        eur_freq : str
            EUR leg payment frequency
        usd_freq : str
            USD leg payment frequency
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Helper information for bootstrapping
        """
        period = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        else:
            dc = ql.Actual365Fixed()
        
        time_to_maturity = dc.yearFraction(self.eval_date, maturity_date)
        
        # Basis spread in decimal
        basis_decimal = basis_spread / 10000.0
        
        return {
            'tenor': tenor,
            'maturity_date': maturity_date,
            'time_to_maturity': time_to_maturity,
            'basis_spread_bps': basis_spread,
            'basis_spread_decimal': basis_decimal,
            'eur_freq': eur_freq,
            'usd_freq': usd_freq
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
        return freq_map.get(freq_str, ql.Quarterly)
    
    def calculate_basis_adjusted_forward(self, tenor, basis_spread, domestic_curve, 
                                         foreign_curve, day_count='Actual/360'):
        """
        Calculate basis-adjusted FX forward rate
        Incorporates cross-currency basis into forward calculation
        
        Parameters:
        -----------
        tenor : str
            Swap tenor
        basis_spread : float
            Basis spread in basis points
        domestic_curve : QuantLib.YieldTermStructure
            USD curve
        foreign_curve : QuantLib.YieldTermStructure
            EUR curve
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: Basis-adjusted forward calculation
        """
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
        
        # Standard forward (without basis)
        standard_forward = self.spot_fx * df_foreign / df_domestic
        
        # Basis adjustment
        basis_decimal = basis_spread / 10000.0
        basis_adjusted_df_domestic = df_domestic / (1 + basis_decimal * time_to_maturity)
        
        # Basis-adjusted forward
        adjusted_forward = self.spot_fx * df_foreign / basis_adjusted_df_domestic
        
        return {
            'tenor': tenor,
            'time_to_maturity': time_to_maturity,
            'basis_spread_bps': basis_spread,
            'standard_forward': standard_forward,
            'adjusted_forward': adjusted_forward,
            'basis_impact': (adjusted_forward - standard_forward) * 10000,  # in pips
            'df_domestic': df_domestic,
            'df_foreign': df_foreign
        }
    
    def price_ccy_swap(self, tenor, notional_eur, basis_spread, domestic_curve, 
                       foreign_curve, eur_freq='Quarterly', usd_freq='Quarterly',
                       day_count='Actual/360'):
        """
        Price a cross-currency basis swap
        
        Structure:
        - EUR leg: Pay/Receive EUR 3M EURIBOR on EUR notional
        - USD leg: Receive/Pay USD 3M SOFR + basis on USD notional
        - Initial and final exchange of notionals at spot FX rate
        
        Parameters:
        -----------
        tenor : str
            Swap tenor
        notional_eur : float
            EUR notional amount
        basis_spread : float
            Basis spread in basis points (added to USD leg)
        domestic_curve : QuantLib.YieldTermStructure
            USD discount curve
        foreign_curve : QuantLib.YieldTermStructure
            EUR discount curve
        eur_freq : str
            EUR leg frequency
        usd_freq : str
            USD leg frequency
        day_count : str
            Day count convention
        
        Returns:
        --------
        dict: CCY swap pricing with NPV breakdown
        """
        period = self._parse_tenor(tenor)
        maturity_date = self.eval_date + period
        
        eur_frequency = self._parse_frequency(eur_freq)
        usd_frequency = self._parse_frequency(usd_freq)
        
        if day_count == 'Actual/360':
            dc = ql.Actual360()
        else:
            dc = ql.Actual365Fixed()
        
        # USD notional
        notional_usd = notional_eur * self.spot_fx
        
        # Create payment schedules
        eur_schedule = ql.Schedule(
            self.eval_date,
            maturity_date,
            ql.Period(eur_frequency),
            ql.TARGET(),
            ql.Following,
            ql.Following,
            ql.DateGeneration.Forward,
            False
        )
        
        usd_schedule = ql.Schedule(
            self.eval_date,
            maturity_date,
            ql.Period(usd_frequency),
            ql.TARGET(),
            ql.Following,
            ql.Following,
            ql.DateGeneration.Forward,
            False
        )
        
        # Calculate leg NPVs
        # EUR leg: floating rate based on EUR curve
        eur_leg_npv = 0.0
        for i in range(len(eur_schedule) - 1):
            start = eur_schedule[i]
            end = eur_schedule[i + 1]
            year_frac = dc.yearFraction(start, end)
            
            # Forward rate
            df_start = foreign_curve.discount(start)
            df_end = foreign_curve.discount(end)
            forward_rate = (df_start / df_end - 1) / year_frac
            
            # Coupon payment
            coupon = notional_eur * forward_rate * year_frac
            eur_leg_npv += coupon * df_end
        
        # USD leg: floating rate + basis spread
        usd_leg_npv = 0.0
        basis_decimal = basis_spread / 10000.0
        
        for i in range(len(usd_schedule) - 1):
            start = usd_schedule[i]
            end = usd_schedule[i + 1]
            year_frac = dc.yearFraction(start, end)
            
            # Forward rate
            df_start = domestic_curve.discount(start)
            df_end = domestic_curve.discount(end)
            forward_rate = (df_start / df_end - 1) / year_frac
            
            # Coupon payment (including basis)
            coupon = notional_usd * (forward_rate + basis_decimal) * year_frac
            usd_leg_npv += coupon * df_end
        
        # Notional exchanges
        df_maturity_eur = foreign_curve.discount(maturity_date)
        df_maturity_usd = domestic_curve.discount(maturity_date)
        
        # Initial exchange: receive EUR, pay USD (both at PV)
        initial_exchange_npv = notional_eur * df_maturity_eur - notional_usd * df_maturity_usd
        
        # Final exchange: pay EUR, receive USD (at maturity, discounted)
        final_exchange_npv = (notional_usd - notional_eur * self.spot_fx) * df_maturity_usd
        
        # Total NPV (from EUR perspective, converted to USD)
        # Positive NPV = favorable to EUR receiver
        npv_eur_leg_in_usd = eur_leg_npv * self.spot_fx
        total_npv = -npv_eur_leg_in_usd + usd_leg_npv + final_exchange_npv
        
        return {
            'notional_eur': notional_eur,
            'notional_usd': notional_usd,
            'spot_fx': self.spot_fx,
            'basis_spread_bps': basis_spread,
            'eur_leg_npv': eur_leg_npv,
            'eur_leg_npv_usd': npv_eur_leg_in_usd,
            'usd_leg_npv': usd_leg_npv,
            'initial_exchange_npv': initial_exchange_npv,
            'final_exchange_npv': final_exchange_npv,
            'total_npv': total_npv,
            'df_maturity_eur': df_maturity_eur,
            'df_maturity_usd': df_maturity_usd
        }
    
    def calculate_par_basis(self, tenor, notional_eur, domestic_curve, foreign_curve,
                           eur_freq='Quarterly', usd_freq='Quarterly', day_count='Actual/360'):
        """
        Calculate the par basis spread that makes the CCY swap NPV = 0
        
        Parameters:
        -----------
        (same as price_ccy_swap)
        
        Returns:
        --------
        float: Par basis spread in basis points
        """
        # Use bisection to find basis that gives NPV ≈ 0
        basis_low = -100  # bps
        basis_high = 100  # bps
        tolerance = 0.01  # 0.01 bps
        
        for _ in range(50):  # Max iterations
            basis_mid = (basis_low + basis_high) / 2
            
            result = self.price_ccy_swap(
                tenor, notional_eur, basis_mid, domestic_curve, foreign_curve,
                eur_freq, usd_freq, day_count
            )
            
            npv = result['total_npv']
            
            if abs(npv) < tolerance:
                return basis_mid
            elif npv > 0:
                basis_high = basis_mid
            else:
                basis_low = basis_mid
        
        return (basis_low + basis_high) / 2
