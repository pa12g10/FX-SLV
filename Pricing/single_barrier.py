# Single Barrier FX Option Pricing
import QuantLib as ql
import pandas as pd
import numpy as np

class SingleBarrierOption:
    """
    Single Barrier FX Option Pricer
    Supports: Up-and-Out, Down-and-Out, Up-and-In, Down-and-In
    """
    
    def __init__(self, eval_date, spot_fx, strike, barrier, expiry_years,
                 barrier_type, option_type, domestic_curve, foreign_curve, model):
        """
        Initialize single barrier option
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Evaluation date
        spot_fx : float
            Spot FX rate
        strike : float
            Strike price
        barrier : float
            Barrier level
        expiry_years : float
            Time to expiry in years
        barrier_type : str
            'UpOut', 'DownOut', 'UpIn', 'DownIn'
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
        self.barrier = barrier
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
        Price the barrier option using Monte Carlo simulation
        """
        # Create barrier option
        expiry_date = self.eval_date + ql.Period(int(self.expiry_years * 365), ql.Days)
        exercise = ql.EuropeanExercise(expiry_date)
        
        # Map barrier types
        barrier_type_map = {
            'UpOut': ql.Barrier.UpOut,
            'DownOut': ql.Barrier.DownOut,
            'UpIn': ql.Barrier.UpIn,
            'DownIn': ql.Barrier.DownIn
        }
        
        option_type_map = {
            'call': ql.Option.Call,
            'put': ql.Option.Put
        }
        
        ql_barrier_type = barrier_type_map[self.barrier_type]
        ql_option_type = option_type_map[self.option_type.lower()]
        
        payoff = ql.PlainVanillaPayoff(ql_option_type, self.strike)
        
        self.option = ql.BarrierOption(
            ql_barrier_type,
            self.barrier,
            0.0,  # rebate
            payoff,
            exercise
        )
        
        # Set up pricing engine (FD engine for barriers)
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))
        
        # Use Heston process from model
        if hasattr(self.model, 'heston_model'):
            process = self.model.heston_model.process()
            
            # FD Heston Barrier engine
            engine = ql.FdHestonBarrierEngine(
                self.model.heston_model,
                200,  # tGrid
                200,  # xGrid  
                100   # vGrid
            )
        else:
            # Fallback to Black-Scholes for testing
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
            # Some engines don't support all Greeks
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
        Alternative Monte Carlo pricing with path monitoring
        
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
        
        # Check barrier breach
        if self.barrier_type == 'UpOut':
            breached = np.max(spot_paths, axis=0) >= self.barrier
            active = ~breached
        elif self.barrier_type == 'DownOut':
            breached = np.min(spot_paths, axis=0) <= self.barrier
            active = ~breached
        elif self.barrier_type == 'UpIn':
            breached = np.max(spot_paths, axis=0) >= self.barrier
            active = breached
        elif self.barrier_type == 'DownIn':
            breached = np.min(spot_paths, axis=0) <= self.barrier
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
            'breach_probability': np.sum(breached) / num_paths
        }
