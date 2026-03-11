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
        self.eval_date = eval_date
        self.currency = currency
        self.curve = None
        self.helpers = []
        # Track instrument type per helper for calibration error colouring
        self._helper_meta = []  # list of dicts {label, instrument_type}
        
        ql.Settings.instance().evaluationDate = eval_date
        
        self.deposit_pricer = DepositPricer(eval_date)
        self.futures_pricer = FuturesPricer(eval_date)
        self.swap_pricer = SwapPricer(eval_date)
    
    def bootstrap_curve(self, deposit_data, futures_data, swaps_data):
        """
        Bootstrap yield curve from market instruments.

        Parameters
        ----------
        deposit_data : dict
            Keys: tenor, rate, day_count
        futures_data : pandas.DataFrame
            Columns: contract, tenor, price, day_count
        swaps_data : pandas.DataFrame
            Columns: tenor, rate, fixed_freq, float_freq, day_count
        """
        print(f"\n{'='*60}")
        print(f"Bootstrapping {self.currency} Curve")
        print(f"{'='*60}")
        
        self.helpers = []
        self._helper_meta = []
        used_pillar_dates = set()
        
        # Flat forward used as initial curve for swap helpers
        initial_rate = swaps_data.iloc[0]['rate'] / 100.0
        flat_forward = ql.FlatForward(
            self.eval_date, initial_rate, ql.Actual360(), ql.Continuous
        )
        curve_handle = ql.YieldTermStructureHandle(flat_forward)
        
        # 1. Deposit
        print(f"\nAdding Deposits:")
        try:
            helper = self.deposit_pricer.create_helper(
                rate=deposit_data['rate'],
                day_count=deposit_data['day_count']
            )
            pillar_date = helper.latestDate()
            if pillar_date not in used_pillar_dates:
                self.helpers.append(helper)
                self._helper_meta.append({'label': 'Deposit', 'instrument_type': 'Deposit'})
                used_pillar_dates.add(pillar_date)
                print(f"  ON @ {deposit_data['rate']:.2f}% -> pillar {pillar_date}")
            else:
                print(f"  ON @ {deposit_data['rate']:.2f}% -> SKIP (duplicate {pillar_date})")
        except Exception as e:
            print(f"  ON: ERROR - {e}")
        
        # 2. Futures  (column is 'tenor', was previously 'maturity')
        print(f"\nAdding Futures:")
        for idx, row in futures_data.iterrows():
            tenor_label = row['tenor']          # <-- fixed: was row['maturity']
            try:
                helper = self.futures_pricer.create_helper(
                    maturity=tenor_label,
                    price=row['price'],
                    day_count=row['day_count']
                )
                pillar_date = helper.latestDate()
                
                too_close = False
                for existing_date in used_pillar_dates:
                    days_apart = abs(pillar_date - existing_date)
                    if days_apart <= 7:
                        print(f"  {tenor_label}: SKIP (pillar {pillar_date} within {days_apart}d of {existing_date})")
                        too_close = True
                        break
                
                if not too_close:
                    self.helpers.append(helper)
                    self._helper_meta.append({'label': tenor_label, 'instrument_type': 'Futures'})
                    used_pillar_dates.add(pillar_date)
                    print(f"  {tenor_label} @ {row['price']:.2f} -> pillar {pillar_date}")
            except Exception as e:
                print(f"  {tenor_label}: ERROR - {e}")
        
        # 3. OIS Swaps
        print(f"\nAdding OIS Swaps:")
        for idx, row in swaps_data.iterrows():
            try:
                helper = self.swap_pricer.create_helper(
                    tenor=row['tenor'],
                    rate=row['rate'],
                    curve_handle=curve_handle,
                    fixed_freq=row['fixed_freq'],
                    float_freq=row['float_freq'],
                    day_count=row['day_count']
                )
                pillar_date = helper.latestDate()
                if pillar_date not in used_pillar_dates:
                    self.helpers.append(helper)
                    self._helper_meta.append({'label': f"Swap {row['tenor']}", 'instrument_type': 'Swaps'})
                    used_pillar_dates.add(pillar_date)
                    print(f"  {row['tenor']} @ {row['rate']:.2f}% -> pillar {pillar_date}")
                else:
                    print(f"  {row['tenor']} @ {row['rate']:.2f}% -> SKIP (duplicate {pillar_date})")
            except Exception as e:
                print(f"  {row['tenor']}: ERROR - {e}")
        
        # 4. Bootstrap
        print(f"\nBootstrapping with {len(self.helpers)} instruments...")
        try:
            self.curve = ql.PiecewiseLogLinearDiscount(
                self.eval_date, self.helpers, ql.Actual360()
            )
            self.curve.enableExtrapolation()
            
            print(f"\n\u2705 {self.currency} curve bootstrapped successfully!")
            print(f"   Instruments used: {len(self.helpers)}")
            print(f"   Max maturity: {self._get_max_maturity():.1f} years")
            
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
            print(f"\n\u274c Curve bootstrapping failed: {e}")
            print(f"{'='*60}\n")
            raise
    
    def get_calibration_errors(self):
        """
        Return calibration errors (Model Rate - Market Rate) in bps for each
        bootstrapping instrument, together with instrument type labels.

        Returns
        -------
        pandas.DataFrame
            Columns: instrument, instrument_type, model_rate, market_rate, error_bps
        """
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")

        rows = []
        for helper, meta in zip(self.helpers, self._helper_meta):
            try:
                market_rate = helper.quote().value()
                model_rate  = helper.impliedQuote()
                error_bps   = (model_rate - market_rate) * 10_000   # bps
                rows.append({
                    'instrument':      meta['label'],
                    'instrument_type': meta['instrument_type'],
                    'market_rate':     market_rate * 100,   # percent
                    'model_rate':      model_rate  * 100,   # percent
                    'error_bps':       error_bps,
                })
            except Exception:
                pass

        return pd.DataFrame(rows)

    def _get_max_maturity(self):
        if not self.helpers:
            return 0.0
        max_date = max([h.latestDate() for h in self.helpers])
        return ql.Actual360().yearFraction(self.eval_date, max_date)
    
    def get_zero_rates(self, tenors):
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        zero_rates = []
        for t in tenors:
            try:
                zero_rates.append(self.curve.zeroRate(t, ql.Continuous).rate())
            except:
                zero_rates.append(np.nan)
        return np.array(zero_rates)
    
    def get_discount_factors(self, tenors):
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        dfs = []
        for t in tenors:
            try:
                dfs.append(self.curve.discount(t))
            except:
                dfs.append(np.nan)
        return np.array(dfs)
    
    def get_forward_rates(self, start_tenors, end_tenors):
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        forward_rates = []
        for t_start, t_end in zip(start_tenors, end_tenors):
            try:
                date_start = self.eval_date + ql.Period(int(t_start * 365), ql.Days)
                date_end   = self.eval_date + ql.Period(int(t_end   * 365), ql.Days)
                forward_rates.append(
                    self.curve.forwardRate(date_start, date_end, ql.Actual360(), ql.Continuous).rate()
                )
            except:
                forward_rates.append(np.nan)
        return np.array(forward_rates)
    
    def get_curve_summary(self, sample_tenors=None):
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        if sample_tenors is None:
            sample_tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        zero_rates       = self.get_zero_rates(sample_tenors)
        discount_factors = self.get_discount_factors(sample_tenors)
        summary_data = []
        for tenor, zr, df in zip(sample_tenors, zero_rates, discount_factors):
            if not np.isnan(zr):
                summary_data.append({
                    'Tenor (Years)': tenor,
                    'Zero Rate (%)': zr * 100,
                    'Discount Factor': df
                })
        return pd.DataFrame(summary_data)
    
    def get_curve_handle(self):
        if self.curve is None:
            raise ValueError("Curve not bootstrapped yet. Call bootstrap_curve() first.")
        return ql.YieldTermStructureHandle(self.curve)


def bootstrap_sofr_curve(eval_date):
    from MarketData import get_sofr_deposit_data, get_sofr_futures_data, get_sofr_swaps_data
    builder = YieldCurveBuilder(eval_date, currency='USD')
    builder.bootstrap_curve(
        get_sofr_deposit_data(),
        get_sofr_futures_data(),
        get_sofr_swaps_data()
    )
    return builder


def bootstrap_estr_curve(eval_date):
    from MarketData import get_estr_deposit_data, get_estr_futures_data, get_estr_swaps_data
    builder = YieldCurveBuilder(eval_date, currency='EUR')
    builder.bootstrap_curve(
        get_estr_deposit_data(),
        get_estr_futures_data(),
        get_estr_swaps_data()
    )
    return builder
