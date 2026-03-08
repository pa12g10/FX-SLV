# FX Curves Module
import QuantLib as ql
import pandas as pd
import numpy as np

class FXCurves:
    """
    FX Yield Curves for domestic and foreign currencies
    """
    
    def __init__(self, eval_date, domestic_rates, foreign_rates, 
                 domestic_currency='USD', foreign_currency='EUR'):
        """
        Initialize FX curves
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Evaluation date
        domestic_rates : list
            Domestic rates [[tenor_years, rate], ...]
        foreign_rates : list
            Foreign rates [[tenor_years, rate], ...]
        domestic_currency : str
            Domestic currency code
        foreign_currency : str
            Foreign currency code
        """
        self.eval_date = eval_date
        self.domestic_rates = domestic_rates
        self.foreign_rates = foreign_rates
        self.domestic_currency = domestic_currency
        self.foreign_currency = foreign_currency
        
        ql.Settings.instance().evaluationDate = eval_date
        
        self.domestic_curve = None
        self.foreign_curve = None
        self.domestic_curve_handle = None
        self.foreign_curve_handle = None
    
    def bootstrap_curves(self):
        """
        Bootstrap yield curves from rate data
        """
        calendar = ql.TARGET()
        day_counter = ql.Actual365Fixed()
        
        # Build domestic curve
        domestic_dates = [self.eval_date + ql.Period(int(t*365), ql.Days) 
                         for t, r in self.domestic_rates]
        domestic_rates_list = [r for t, r in self.domestic_rates]
        
        self.domestic_curve = ql.ZeroCurve(
            domestic_dates,
            domestic_rates_list,
            day_counter,
            calendar
        )
        
        self.domestic_curve_handle = ql.YieldTermStructureHandle(self.domestic_curve)
        
        # Build foreign curve
        foreign_dates = [self.eval_date + ql.Period(int(t*365), ql.Days) 
                        for t, r in self.foreign_rates]
        foreign_rates_list = [r for t, r in self.foreign_rates]
        
        self.foreign_curve = ql.ZeroCurve(
            foreign_dates,
            foreign_rates_list,
            day_counter,
            calendar
        )
        
        self.foreign_curve_handle = ql.YieldTermStructureHandle(self.foreign_curve)
        
        return self.domestic_curve_handle, self.foreign_curve_handle
    
    def get_discount_factors(self, times):
        """
        Get discount factors for given times
        
        Parameters:
        -----------
        times : list
            List of times in years
        
        Returns:
        --------
        tuple: (domestic_dfs, foreign_dfs)
        """
        if not self.domestic_curve or not self.foreign_curve:
            self.bootstrap_curves()
        
        domestic_dfs = [self.domestic_curve.discount(t) for t in times]
        foreign_dfs = [self.foreign_curve.discount(t) for t in times]
        
        return domestic_dfs, foreign_dfs
    
    def get_forward_fx(self, spot_fx, time):
        """
        Calculate forward FX rate
        
        Parameters:
        -----------
        spot_fx : float
            Spot FX rate
        time : float
            Time to forward in years
        
        Returns:
        --------
        float: Forward FX rate
        """
        if not self.domestic_curve or not self.foreign_curve:
            self.bootstrap_curves()
        
        df_domestic = self.domestic_curve.discount(time)
        df_foreign = self.foreign_curve.discount(time)
        
        # Forward = Spot * (DF_foreign / DF_domestic)
        forward_fx = spot_fx * (df_foreign / df_domestic)
        
        return forward_fx
