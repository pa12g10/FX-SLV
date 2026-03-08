# Yield Curve Bootstrapping Module
import QuantLib as ql
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Pricing import DepositPricer, FuturesPricer, SwapPricer

class YieldCurveBuilder:
    """
    Yield Curve Builder for bootstrapping OIS curves (SOFR, ESTR, etc.)
    Uses deposits, futures, and swaps to construct the yield curve
    """
    
    def __init__(self, eval_date, currency='USD'):
        """
        Initialize yield curve builder
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Evaluation date
        currency : str
            Currency code (USD, EUR, etc.)
        """
        self.eval_date = eval_date
        self.currency = currency
        self.curve = None
        self.helpers = []
        
        # Set QuantLib evaluation date
        ql.Settings.instance().evaluationDate = eval_date
        
        # Initialize pricers
        self.deposit_pricer = DepositPricer(eval_date)
        self.futures_pricer = FuturesPricer(eval_date)
        self.swap_pricer = SwapPricer(eval_date)
    
    def bootstrap_curve(self, deposit_data, futures_data, swaps_data):
        """
        Bootstrap yield curve from market instruments
        
        Parameters:
        -----------
        deposit_data : dict
            Deposit data with keys: tenor, rate, day_count
        futures_data : pandas.DataFrame
            Futures data with columns: maturity, price, day_count
        swaps_data : pandas.DataFrame
            Swaps data with columns: tenor, rate, fixed_freq, float_freq, day_count
        
        Returns:
        --------
        QuantLib.YieldTermStructure: Bootstrapped curve
        """
        self.helpers = []
        
        # 1. Add Deposit Helper (Overnight)
        print(f"\n=== Bootstrapping {self.currency} Curve ===")
        print(f"Adding deposit: ON @ {deposit_data['rate']:.2f}%")
        
        deposit_helper = self.deposit_pricer.create_helper(
            rate=deposit_data['rate'],
            day_count=deposit_data['day_count']
        )
        self.helpers.append(deposit_helper)
        
        # 2. Add Futures Helpers
        print(f"\nAdding {len(futures_data)} futures contracts:")
        for idx, row in futures_data.iterrows():
            print(f"  {row['maturity']}: {row['price']:.2f} (implied: {row['rate']:.2f}%)")
            
            futures_helper = self.futures_pricer.create_helper(
                maturity=row['maturity'],
                price=row['price'],
                day_count=row['day_count']
            )
            self.helpers.append(futures_helper)
        
        # 3. Add Swap Helpers
        # For swaps, we need a preliminary curve handle for the swap pricer
        # Create a flat forward curve as initial guess
        print(f"\nAdding {len(swaps_data)} OIS swaps:")
        
        # Use first swap rate as initial flat rate
        initial_rate = swaps_data.iloc[0]['rate'] / 100.0
        flat_forward = ql.FlatForward(
            self.eval_date,
            initial_rate,
            ql.Actual360(),
            ql.Continuous
        )
        curve_handle = ql.YieldTermStructureHandle(flat_forward)
        
        for idx, row in swaps_data.iterrows():
            print(f"  {row['tenor']}: {row['rate']:.2f}%")
            
            swap_helper = self.swap_pricer.create_helper(
                tenor=row['tenor'],
                rate=row['rate'],
                curve_handle=curve_handle,
                fixed_freq=row['fixed_freq'],
                float_freq=row['float_freq'],
                day_count=row['day_count']
            )
            self.helpers.append(swap_helper)
        
        # 4. Bootstrap the curve using PiecewiseLogLinear
        print(f"\nBootstrapping curve with {len(self.helpers)} instruments...")
        
        try:
            self.curve = ql.PiecewiseLogLinearDiscount(
                self.eval_date,
                self.helpers,
                ql.Actual360()
            )
            
            # Enable extrapolation
            self.curve.enableExtrapolation()
            
            print(f"✅ {self.currency} curve bootstrapped successfully!")
            print(f"   Number of nodes: {len(self.helpers)}")
            print(f"   Max maturity: {self._get_max_maturity()} years")
            
            # Print some sample rates
            print(f"\nSample Zero Rates:")
            for tenor in [1, 2, 5, 10, 30]:
                try:
                    zero_rate = self.curve.zeroRate(tenor, ql.Continuous).rate() * 100
                    print(f"   {tenor}Y: {zero_rate:.4f}%")
                except:
                    pass
            
            return self.curve
            
        except Exception as e:
            print(f"❌ Curve bootstrapping failed: {e}")
            raise
    
    def _get_max_maturity(self):
        """
        Get maximum maturity from helpers
        """
        max_date = max([helper.latestDate() for helper in self.helpers])
        return ql.Actual360().yearFraction(self.eval_date, max_date)
    
    def get_zero_rates(self, tenors):
        """
        Get zero rates for specified tenors
        
        Parameters:
        -----------
        tenors : array-like
            Tenors in years
        
        Returns:
        --------
        numpy.array: Zero rates (as decimals)
        """
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        
        zero_rates = []
        for t in tenors:
            try:
                rate = self.curve.zeroRate(t, ql.Continuous).rate()
                zero_rates.append(rate)
            except:
                zero_rates.append(np.nan)
        
        return np.array(zero_rates)
    
    def get_discount_factors(self, tenors):
        """
        Get discount factors for specified tenors
        
        Parameters:
        -----------
        tenors : array-like
            Tenors in years
        
        Returns:
        --------
        numpy.array: Discount factors
        """
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        
        dfs = []
        for t in tenors:
            try:
                df = self.curve.discount(t)
                dfs.append(df)
            except:
                dfs.append(np.nan)
        
        return np.array(dfs)
    
    def get_forward_rates(self, start_tenors, end_tenors):
        """
        Get forward rates between periods
        
        Parameters:
        -----------
        start_tenors : array-like
            Start tenors in years
        end_tenors : array-like
            End tenors in years
        
        Returns:
        --------
        numpy.array: Forward rates (as decimals)
        """
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        
        forward_rates = []
        for t_start, t_end in zip(start_tenors, end_tenors):
            try:
                date_start = self.eval_date + ql.Period(int(t_start * 365), ql.Days)
                date_end = self.eval_date + ql.Period(int(t_end * 365), ql.Days)
                
                forward_rate = self.curve.forwardRate(
                    date_start,
                    date_end,
                    ql.Actual360(),
                    ql.Continuous
                ).rate()
                
                forward_rates.append(forward_rate)
            except:
                forward_rates.append(np.nan)
        
        return np.array(forward_rates)
    
    def get_curve_summary(self, sample_tenors=None):
        """
        Get summary of curve with zero rates and discount factors
        
        Parameters:
        -----------
        sample_tenors : array-like, optional
            Tenors to sample. Defaults to standard tenors.
        
        Returns:
        --------
        pandas.DataFrame: Curve summary
        """
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        
        if sample_tenors is None:
            sample_tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        
        zero_rates = self.get_zero_rates(sample_tenors)
        discount_factors = self.get_discount_factors(sample_tenors)
        
        summary_data = []
        for tenor, zero_rate, df in zip(sample_tenors, zero_rates, discount_factors):
            if not np.isnan(zero_rate):
                summary_data.append({
                    'Tenor (Years)': tenor,
                    'Zero Rate (%)': zero_rate * 100,
                    'Discount Factor': df
                })
        
        return pd.DataFrame(summary_data)
    
    def get_curve_handle(self):
        """
        Get QuantLib curve handle for use in other instruments
        
        Returns:
        --------
        QuantLib.YieldTermStructureHandle
        """
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        
        return ql.YieldTermStructureHandle(self.curve)


def bootstrap_sofr_curve(eval_date):
    """
    Convenience function to bootstrap USD SOFR curve from market data
    
    Parameters:
    -----------
    eval_date : QuantLib.Date
        Evaluation date
    
    Returns:
    --------
    YieldCurveBuilder: Bootstrapped SOFR curve
    """
    from MarketData import get_sofr_deposit_data, get_sofr_futures_data, get_sofr_swaps_data
    
    builder = YieldCurveBuilder(eval_date, currency='USD')
    
    deposit_data = get_sofr_deposit_data()
    futures_data = get_sofr_futures_data()
    swaps_data = get_sofr_swaps_data()
    
    builder.bootstrap_curve(deposit_data, futures_data, swaps_data)
    
    return builder


def bootstrap_estr_curve(eval_date):
    """
    Convenience function to bootstrap EUR ESTR curve from market data
    
    Parameters:
    -----------
    eval_date : QuantLib.Date
        Evaluation date
    
    Returns:
    --------
    YieldCurveBuilder: Bootstrapped ESTR curve
    """
    from MarketData import get_estr_deposit_data, get_estr_futures_data, get_estr_swaps_data
    
    builder = YieldCurveBuilder(eval_date, currency='EUR')
    
    deposit_data = get_estr_deposit_data()
    futures_data = get_estr_futures_data()
    swaps_data = get_estr_swaps_data()
    
    builder.bootstrap_curve(deposit_data, futures_data, swaps_data)
    
    return builder
