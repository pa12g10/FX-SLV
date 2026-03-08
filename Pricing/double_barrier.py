# Double Barrier FX Option Pricing
import QuantLib as ql
import pandas as pd
import numpy as np

class DoubleBarrierOption:
    """
    Double Barrier FX Option Pricer
    Supports: Double-Knock-Out, Double-Knock-In
    """
    
    def __init__(self, eval_date, spot_fx, strike, lower_barrier, upper_barrier,
                 expiry_years, barrier_type, option_type, domestic_curve, 
                 foreign_curve, model):
        """
        Initialize double barrier option
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Evaluation date
        spot_fx : float
            Spot FX rate
        strike : float
            Strike price
        lower_barrier : float
            Lower barrier level
        upper_barrier : float
            Upper barrier level
        expiry_years : float
            Time to expiry in years
        barrier_type : str
            'KnockOut' or 'KnockIn'
        option_type : str
            'call' or 'put'
        domestic_curve : YieldTermStructureHandle
            Domestic yield curve
        foreign_curve : YieldTermStructureHandle
            Foreign yield curve
        model : HestonModel or similar
            Calibrated FX model
        """
        self.eval_date = eval_date
        self.spot_fx = spot_fx
        self.strike = strike
        self.lower_barrier = lower_barrier
        self.upper_barrier = upper_barrier
        self.expiry_years = expiry_years
        self.barrier_type = barrier_type
        self.option_type = option_type
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.model = model
        
        self.option = None
        self.price = None
        self.greeks = {}
    
    def price_option(self):
        """
        Price the double barrier option using Monte Carlo simulation
        """
        # Create double barrier option
        expiry_date = self.eval_date + ql.Period(int(self.expiry_years * 365), ql.Days)
        exercise = ql.EuropeanExercise(expiry_date)
        
        # Map barrier types
        barrier_type_map = {
            'KnockOut': ql.DoubleBarrier.KnockOut,
            'KnockIn': ql.DoubleBarrier.KnockIn
        }
        
        option_type_map = {
            'call': ql.Option.Call,
            'put': ql.Option.Put
        }
        
        ql_barrier_type = barrier_type_map[self.barrier_type]
        ql_option_type = option_type_map[self.option_type.lower()]
        
        payoff = ql.PlainVanillaPayoff(ql_option_type, self.strike)
        
        self.option = ql.DoubleBarrierOption(
            ql_barrier_type,
            self.lower_barrier,
            self.upper_barrier,
            0.0,  # rebate
            payoff,
            exercise
        )
        
        # Set up pricing engine
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))
        
        # Use Heston process from model
        if hasattr(self.model, 'heston_model'):
            # For double barriers, use Monte Carlo or FD
            # FD engine for double barriers
            engine = ql.FdHestonDoubleBarrierEngine(
                self.model.heston_model,
                200,  # tGrid
                200,  # xGrid
                100   # vGrid
            )
        else:
            # Fallback to Black-Scholes
            vol = 0.15
            process = ql.BlackScholesMertonProcess(
                spot_handle,
                self.foreign_curve,
                self.domestic_curve,
                ql.BlackVolTermStructureHandle(
                    ql.BlackConstantVol(self.eval_date, ql.TARGET(), vol, ql.Actual365Fixed())
                )
            )
            engine = ql.FdBlackScholesBarrierEngine(process, 200, 200)
        
        self.option.setPricingEngine(engine)
        self.price = self.option.NPV()
        
        return self.price
    
    def calculate_greeks(self):
        """
        Calculate option Greeks
        """
        if not self.option:
            self.price_option()
        
        try:
            self.greeks = {
                'delta': self.option.delta(),
                'gamma': self.option.gamma(),
                'vega': self.option.vega(),
                'theta': self.option.theta(),
                'rho': self.option.rho()
            }
        except:
            self.greeks = {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }
        
        return self.greeks
    
    def monte_carlo_price(self, num_paths=100000, time_steps=252):
        """
        Monte Carlo pricing with path monitoring
        
        Parameters:
        -----------
        num_paths : int
            Number of simulation paths
        time_steps : int
            Number of time steps per year
        
        Returns:
        --------
        dict: Pricing results
        """
        # Get simulated paths from model
        if hasattr(self.model, 'get_simulated_paths'):
            _, times, spot_paths, _ = self.model.get_simulated_paths(
                num_paths=num_paths,
                time_steps=time_steps,
                horizon_years=self.expiry_years
            )
        else:
            return {'price': 0.0, 'std_error': 0.0}
        
        # Check barrier breach (either upper or lower)
        upper_breached = np.max(spot_paths, axis=0) >= self.upper_barrier
        lower_breached = np.min(spot_paths, axis=0) <= self.lower_barrier
        breached = upper_breached | lower_breached
        
        if self.barrier_type == 'KnockOut':
            active = ~breached
        else:  # KnockIn
            active = breached
        
        # Calculate payoffs
        final_spots = spot_paths[-1, :]
        
        if self.option_type.lower() == 'call':
            payoffs = np.maximum(final_spots - self.strike, 0) * active
        else:
            payoffs = np.maximum(self.strike - final_spots, 0) * active
        
        # Discount back
        df = self.domestic_curve.discount(self.expiry_years)
        price = np.mean(payoffs) * df
        std_error = np.std(payoffs) * df / np.sqrt(num_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'breach_probability': np.sum(breached) / num_paths,
            'upper_breach_prob': np.sum(upper_breached) / num_paths,
            'lower_breach_prob': np.sum(lower_breached) / num_paths
        }
