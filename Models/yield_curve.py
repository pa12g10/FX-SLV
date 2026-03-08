# Yield Curve Bootstrapping Module
import QuantLib as ql
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

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
        self.selected_instruments = []
        
        # Set QuantLib evaluation date
        ql.Settings.instance().evaluationDate = eval_date
        
        # Initialize pricers
        self.deposit_pricer = DepositPricer(eval_date)
        self.futures_pricer = FuturesPricer(eval_date)
        self.swap_pricer = SwapPricer(eval_date)
    
    def _parse_tenor_to_period(self, tenor_str):
        """
        Parse tenor string to QuantLib Period
        """
        tenor_str = tenor_str.upper().strip()
        
        if tenor_str == 'ON':
            return ql.Period(1, ql.Days)
        elif tenor_str.endswith('D'):
            days = int(tenor_str[:-1])
            return ql.Period(days, ql.Days)
        elif tenor_str.endswith('W'):
            weeks = int(tenor_str[:-1])
            return ql.Period(weeks, ql.Weeks)
        elif tenor_str.endswith('M'):
            months = int(tenor_str[:-1])
            return ql.Period(months, ql.Months)
        elif tenor_str.endswith('Y'):
            years = int(tenor_str[:-1])
            return ql.Period(years, ql.Years)
        else:
            raise ValueError(f"Invalid tenor format: {tenor_str}")
    
    def _calculate_pillar_date(self, tenor_or_date):
        """
        Calculate the pillar date for an instrument
        """
        if isinstance(tenor_or_date, ql.Date):
            return tenor_or_date
        elif isinstance(tenor_or_date, str):
            period = self._parse_tenor_to_period(tenor_or_date)
            # Add 2 settlement days (standard)
            return self.eval_date + ql.Period(2, ql.Days) + period
        elif isinstance(tenor_or_date, ql.Period):
            return self.eval_date + ql.Period(2, ql.Days) + tenor_or_date
        else:
            raise ValueError(f"Cannot calculate pillar date from {type(tenor_or_date)}")
    
    def _select_instruments(self, deposit_data, futures_data, swaps_data):
        """
        Select non-overlapping instruments for curve bootstrapping
        Priority: deposits (short) -> futures (short-med) -> swaps (med-long)
        """
        selected = []
        used_dates = set()
        
        # 1. Add deposit (always ON)
        deposit_date = self._calculate_pillar_date('ON')
        selected.append({
            'type': 'deposit',
            'pillar_date': deposit_date,
            'data': deposit_data,
            'tenor': 'ON'
        })
        used_dates.add(deposit_date)
        
        print(f"\nSelecting instruments for {self.currency} curve:")
        print(f"  Deposit ON: {deposit_date}")
        
        # 2. Add futures (check for conflicts)
        print(f"\n  Futures:")
        for idx, row in futures_data.iterrows():
            maturity_date = self._calculate_pillar_date(row['maturity'])
            
            # Check if date already used
            if maturity_date in used_dates:
                print(f"    {row['maturity']}: SKIP (pillar {maturity_date} already used)")
                continue
            
            # Check if too close to any swap (within 5 days)
            too_close = False
            for _, swap_row in swaps_data.iterrows():
                swap_date = self._calculate_pillar_date(swap_row['tenor'])
                days_diff = abs((swap_date - maturity_date))
                if days_diff <= 5:
                    print(f"    {row['maturity']}: SKIP (too close to swap {swap_row['tenor']})")
                    too_close = True
                    break
            
            if too_close:
                continue
            
            selected.append({
                'type': 'futures',
                'pillar_date': maturity_date,
                'data': row,
                'tenor': row['maturity']
            })
            used_dates.add(maturity_date)
            print(f"    {row['maturity']}: ADD (pillar {maturity_date})")
        
        # 3. Add swaps (these take priority in case of conflicts)
        print(f"\n  Swaps:")
        for idx, row in swaps_data.iterrows():
            maturity_date = self._calculate_pillar_date(row['tenor'])
            
            if maturity_date in used_dates:
                print(f"    {row['tenor']}: SKIP (pillar {maturity_date} already used)")
                continue
            
            selected.append({
                'type': 'swap',
                'pillar_date': maturity_date,
                'data': row,
                'tenor': row['tenor']
            })
            used_dates.add(maturity_date)
            print(f"    {row['tenor']}: ADD (pillar {maturity_date})")
        
        # Sort by pillar date
        selected.sort(key=lambda x: x['pillar_date'])
        
        self.selected_instruments = selected
        return selected
    
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
        print(f"\n{'='*60}")
        print(f"Bootstrapping {self.currency} Curve")
        print(f"{'='*60}")
        
        # Select non-overlapping instruments
        selected = self._select_instruments(deposit_data, futures_data, swaps_data)
        
        # Create flat forward curve for swap helpers
        initial_rate = swaps_data.iloc[0]['rate'] / 100.0
        flat_forward = ql.FlatForward(
            self.eval_date,
            initial_rate,
            ql.Actual360(),
            ql.Continuous
        )
        curve_handle = ql.YieldTermStructureHandle(flat_forward)
        
        # Create helpers
        self.helpers = []
        print(f"\nCreating rate helpers:")
        
        for instrument in selected:
            if instrument['type'] == 'deposit':
                data = instrument['data']
                helper = self.deposit_pricer.create_helper(
                    rate=data['rate'],
                    day_count=data['day_count']
                )
                self.helpers.append(helper)
                print(f"  {instrument['tenor']}: Deposit @ {data['rate']:.2f}%")
            
            elif instrument['type'] == 'futures':
                data = instrument['data']
                helper = self.futures_pricer.create_helper(
                    maturity=data['maturity'],
                    price=data['price'],
                    day_count=data['day_count']
                )
                self.helpers.append(helper)
                print(f"  {instrument['tenor']}: Futures @ {data['price']:.2f} ({data['rate']:.2f}%)")
            
            elif instrument['type'] == 'swap':
                data = instrument['data']
                helper = self.swap_pricer.create_helper(
                    tenor=data['tenor'],
                    rate=data['rate'],
                    curve_handle=curve_handle,
                    fixed_freq=data['fixed_freq'],
                    float_freq=data['float_freq'],
                    day_count=data['day_count']
                )
                self.helpers.append(helper)
                print(f"  {instrument['tenor']}: Swap @ {data['rate']:.2f}%")
        
        # Bootstrap the curve
        print(f"\nBootstrapping with {len(self.helpers)} instruments...")
        
        try:
            self.curve = ql.PiecewiseLogLinearDiscount(
                self.eval_date,
                self.helpers,
                ql.Actual360()
            )
            
            # Enable extrapolation
            self.curve.enableExtrapolation()
            
            print(f"\n✅ {self.currency} curve bootstrapped successfully!")
            print(f"   Instruments used: {len(self.helpers)}")
            print(f"   Max maturity: {self._get_max_maturity():.1f} years")
            
            # Print sample rates
            print(f"\nSample Zero Rates:")
            for tenor in [0.25, 0.5, 1, 2, 5, 10, 30]:
                try:
                    zero_rate = self.curve.zeroRate(tenor, ql.Continuous).rate() * 100
                    print(f"   {tenor}Y: {zero_rate:.4f}%")
                except:
                    pass
            
            print(f"{'='*60}\n")
            
            return self.curve
            
        except Exception as e:
            print(f"\n❌ Curve bootstrapping failed: {e}")
            print(f"{'='*60}\n")
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
