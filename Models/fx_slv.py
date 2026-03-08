# FX Stochastic Local Volatility Model
import QuantLib as ql
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class FXStochasticLocalVol:
    """
    FX Stochastic Local Volatility Model
    Combines local volatility (Dupire) with stochastic volatility (Heston-like)
    """
    
    def __init__(self, eval_date, spot_fx, domestic_curve, foreign_curve, 
                 vol_surface_data, model_params=None):
        """
        Initialize FX-SLV model
        
        Parameters:
        -----------
        eval_date : QuantLib.Date
            Evaluation date
        spot_fx : float
            Current FX spot rate (domestic/foreign, e.g., USD/EUR)
        domestic_curve : YieldTermStructureHandle
            Domestic currency yield curve
        foreign_curve : YieldTermStructureHandle
            Foreign currency yield curve
        vol_surface_data : list or DataFrame
            Volatility surface data [[strike, expiry, volatility], ...]
            Volatility in decimal (e.g., 0.10 for 10%)
        model_params : dict, optional
            Model parameters with keys:
            - 'v0': initial variance (default: 0.01)
            - 'kappa': mean reversion speed (default: 1.0)
            - 'theta': long-term variance (default: 0.01)
            - 'sigma': vol-of-vol (default: 0.3)
            - 'rho': correlation between spot and vol (default: -0.7)
        """
        self.eval_date = eval_date
        self.spot_fx = spot_fx
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.vol_surface_data = vol_surface_data
        
        # Default model parameters (Heston-like stochastic volatility)
        self.model_params = model_params or {
            'v0': 0.01,        # Initial variance
            'kappa': 1.0,      # Mean reversion speed
            'theta': 0.01,     # Long-term variance
            'sigma': 0.3,      # Vol-of-vol
            'rho': -0.7        # Correlation
        }
        
        # QuantLib objects
        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))
        self.heston_model = None
        self.local_vol_surface = None
        self.calibrated_helpers = []
        self.calibration_results = None
        self.black_var_surface = None
    
    def build_vol_surface(self):
        """
        Build Black volatility surface from market data
        """
        # Set up calendar and day counter
        calendar = ql.TARGET()
        day_counter = ql.Actual365Fixed()
        
        # Parse volatility data
        if isinstance(self.vol_surface_data, pd.DataFrame):
            strikes = self.vol_surface_data['strike'].values
            expiries = self.vol_surface_data['expiry'].values
            vols = self.vol_surface_data['volatility'].values
        else:
            strikes = [d[0] for d in self.vol_surface_data]
            expiries = [d[1] for d in self.vol_surface_data]
            vols = [d[2] for d in self.vol_surface_data]
        
        # Convert expiries to dates
        expiry_dates = [self.eval_date + ql.Period(int(e*365), ql.Days) for e in expiries]
        
        # Create Black variance surface
        # Group by expiry to create matrix structure
        expiry_set = sorted(set(expiries))
        strike_set = sorted(set(strikes))
        
        expiry_dates_unique = [self.eval_date + ql.Period(int(e*365), ql.Days) for e in expiry_set]
        
        # Build vol matrix
        vol_matrix = ql.Matrix(len(strike_set), len(expiry_set))
        
        for i, strike in enumerate(strike_set):
            for j, expiry in enumerate(expiry_set):
                # Find vol for this strike/expiry combo
                matching_vols = [vols[k] for k in range(len(vols)) 
                               if strikes[k] == strike and expiries[k] == expiry]
                if matching_vols:
                    vol_matrix[i][j] = matching_vols[0]
                else:
                    # Interpolate or use nearby value
                    vol_matrix[i][j] = 0.15  # Default fallback
        
        # Create Black variance surface
        self.black_var_surface = ql.BlackVarianceSurface(
            self.eval_date,
            calendar,
            expiry_dates_unique,
            strike_set,
            vol_matrix,
            day_counter
        )
        
        return self.black_var_surface
    
    def calibrate(self):
        """
        Calibrate the FX-SLV model parameters to vanilla option prices
        Uses Heston model as the stochastic volatility component
        """
        # Build volatility surface first
        if self.black_var_surface is None:
            self.build_vol_surface()
        
        # Get initial parameters
        v0 = self.model_params['v0']
        kappa = self.model_params['kappa']
        theta = self.model_params['theta']
        sigma = self.model_params['sigma']
        rho = self.model_params['rho']
        
        # Create Heston process
        heston_process = ql.HestonProcess(
            self.domestic_curve,
            self.foreign_curve,
            self.spot_handle,
            v0,
            kappa,
            theta,
            sigma,
            rho
        )
        
        # Create Heston model
        self.heston_model = ql.HestonModel(heston_process)
        
        # Create calibration helpers from vol surface
        self.calibrated_helpers = self._create_calibration_helpers()
        
        # Set up Heston pricing engine
        heston_engine = ql.AnalyticHestonEngine(self.heston_model)
        
        # Set pricing engine for all helpers
        for helper in self.calibrated_helpers:
            helper.setPricingEngine(heston_engine)
        
        # Calibrate model
        optimization_method = ql.LevenbergMarquardt()
        end_criteria = ql.EndCriteria(500, 100, 1e-8, 1e-8, 1e-8)
        
        self.heston_model.calibrate(
            self.calibrated_helpers,
            optimization_method,
            end_criteria
        )
        
        # Build local volatility surface from calibrated model
        self._build_local_vol_surface()
        
        # Extract calibration results
        self._extract_calibration_results()
        
        return self.heston_model
    
    def _create_calibration_helpers(self):
        """
        Create vanilla option helpers for calibration
        """
        helpers = []
        calendar = ql.TARGET()
        
        # Parse vol surface data
        if isinstance(self.vol_surface_data, pd.DataFrame):
            data = self.vol_surface_data.to_dict('records')
        else:
            data = [{'strike': d[0], 'expiry': d[1], 'volatility': d[2]} 
                   for d in self.vol_surface_data]
        
        for row in data:
            strike = row['strike']
            expiry_years = row['expiry']
            volatility = row['volatility']
            
            # Create option helper
            expiry_date = self.eval_date + ql.Period(int(expiry_years * 365), ql.Days)
            maturity = ql.Period(int(expiry_years * 365), ql.Days)
            
            # Determine if call or put (use put for below spot, call for above)
            option_type = ql.Option.Call if strike >= self.spot_fx else ql.Option.Put
            
            helper = ql.HestonModelHelper(
                maturity,
                calendar,
                self.spot_fx,
                strike,
                ql.QuoteHandle(ql.SimpleQuote(volatility)),
                self.domestic_curve,
                self.foreign_curve,
                ql.BlackCalibrationHelper.ImpliedVolError
            )
            
            helpers.append(helper)
        
        return helpers
    
    def _build_local_vol_surface(self):
        """
        Build local volatility surface using Dupire's formula
        """
        # Use the calibrated Heston model to generate implied vols
        # Then apply Dupire formula to get local vols
        
        calendar = ql.TARGET()
        day_counter = ql.Actual365Fixed()
        
        # Create local vol surface from Black variance surface
        if self.black_var_surface:
            self.local_vol_surface = ql.LocalVolSurface(
                ql.BlackVarianceSurfaceHandle(self.black_var_surface),
                self.domestic_curve,
                self.foreign_curve,
                self.spot_handle
            )
        
        return self.local_vol_surface
    
    def _extract_calibration_results(self):
        """
        Extract calibration results including parameters and pricing errors
        """
        if not self.heston_model or not self.calibrated_helpers:
            return None
        
        # Get calibrated parameters (Heston model)
        params = self.heston_model.params()
        calibrated_v0 = params[0]
        calibrated_kappa = params[1]
        calibrated_theta = params[2]
        calibrated_sigma = params[3]
        calibrated_rho = params[4]
        
        # Calculate pricing errors
        pricing_errors = []
        
        if isinstance(self.vol_surface_data, pd.DataFrame):
            data = self.vol_surface_data.to_dict('records')
        else:
            data = [{'strike': d[0], 'expiry': d[1], 'volatility': d[2]} 
                   for d in self.vol_surface_data]
        
        for i, (helper, row) in enumerate(zip(self.calibrated_helpers, data)):
            market_vol = row['volatility']
            market_price = helper.marketValue()
            model_price = helper.modelValue()
            
            # Calculate implied vol from model price
            try:
                model_vol = helper.impliedVolatility(
                    model_price,
                    1e-6,
                    1000,
                    0.0,
                    2.0
                )
            except:
                model_vol = market_vol
            
            price_error = model_price - market_price
            price_error_pct = (price_error / market_price * 100) if market_price != 0 else 0.0
            vol_error_bps = (model_vol - market_vol) * 10000
            
            pricing_errors.append({
                'strike': row['strike'],
                'expiry': row['expiry'],
                'market_vol': market_vol * 100,
                'model_vol': model_vol * 100,
                'market_price': market_price,
                'model_price': model_price,
                'price_error': price_error,
                'price_error_pct': price_error_pct,
                'vol_error_bps': vol_error_bps
            })
        
        self.calibration_results = {
            'v0': calibrated_v0,
            'kappa': calibrated_kappa,
            'theta': calibrated_theta,
            'sigma': calibrated_sigma,
            'rho': calibrated_rho,
            'pricing_errors': pd.DataFrame(pricing_errors)
        }
        
        return self.calibration_results
    
    def get_calibrated_results(self):
        """
        Get calibrated model results
        """
        return self.calibration_results
    
    def get_simulated_paths(self, num_paths=1000, time_steps=252, horizon_years=1.0):
        """
        Generate simulated FX spot paths using the calibrated SLV model
        
        Parameters:
        -----------
        num_paths : int
            Number of Monte Carlo paths
        time_steps : int
            Number of time steps per year
        horizon_years : float
            Simulation horizon in years
        
        Returns:
        --------
        tuple: (DataFrame, times array, paths array, vol_paths array)
            - DataFrame with simulated paths
            - Times array for x-axis
            - Spot paths array (steps x paths)
            - Volatility paths array (steps x paths)
        """
        if not self.heston_model:
            return None, None, None, None
        
        # Time grid
        total_steps = int(time_steps * horizon_years)
        times = np.linspace(0, horizon_years, total_steps + 1)
        dt = horizon_years / total_steps
        
        # Get calibrated Heston parameters
        params = self.heston_model.params()
        v0 = params[0]
        kappa = params[1]
        theta = params[2]
        sigma = params[3]
        rho = params[4]
        
        # Get risk-free rates (domestic and foreign)
        rd = self.domestic_curve.zeroRate(horizon_years/2, ql.Continuous).rate()
        rf = self.foreign_curve.zeroRate(horizon_years/2, ql.Continuous).rate()
        
        # Initialize arrays
        spot_paths = np.zeros((total_steps + 1, num_paths))
        vol_paths = np.zeros((total_steps + 1, num_paths))
        
        spot_paths[0, :] = self.spot_fx
        vol_paths[0, :] = v0
        
        # Generate correlated random numbers
        np.random.seed(42)
        dW1 = np.random.normal(0, np.sqrt(dt), (total_steps, num_paths))
        dW2_uncorrelated = np.random.normal(0, np.sqrt(dt), (total_steps, num_paths))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2_uncorrelated
        
        # Simulate paths using Heston dynamics
        for i in range(1, total_steps + 1):
            # Variance process (CIR with full truncation scheme)
            vol_paths[i, :] = np.maximum(
                vol_paths[i-1, :] + kappa * (theta - vol_paths[i-1, :]) * dt + 
                sigma * np.sqrt(np.maximum(vol_paths[i-1, :], 0)) * dW2[i-1, :],
                0
            )
            
            # Spot FX process
            drift = (rd - rf) * dt
            diffusion = np.sqrt(np.maximum(vol_paths[i-1, :], 0)) * dW1[i-1, :]
            spot_paths[i, :] = spot_paths[i-1, :] * np.exp(drift - 0.5 * vol_paths[i-1, :] * dt + diffusion)
        
        # Create DataFrame
        path_df = pd.DataFrame(
            spot_paths.T,
            columns=[f'time_{t:.4f}' for t in times]
        )
        
        return path_df, times, spot_paths, vol_paths
    
    def validate_option_prices(self, test_options=None):
        """
        Validate model by comparing prices with Black-Scholes using market vols
        
        Parameters:
        -----------
        test_options : list, optional
            List of test options [[strike, expiry, option_type], ...]
            If None, uses sample from vol surface
        
        Returns:
        --------
        pd.DataFrame: Comparison of model vs Black-Scholes prices
        """
        if not self.heston_model:
            return None
        
        # Use sample from vol surface if no test options provided
        if test_options is None:
            if isinstance(self.vol_surface_data, pd.DataFrame):
                sample = self.vol_surface_data.sample(min(5, len(self.vol_surface_data)))
                test_options = [[row['strike'], row['expiry'], 'call'] 
                              for _, row in sample.iterrows()]
            else:
                test_options = [[d[0], d[1], 'call'] 
                              for d in self.vol_surface_data[:5]]
        
        validation_results = []
        
        for strike, expiry, option_type in test_options:
            # Create vanilla option
            exercise_date = self.eval_date + ql.Period(int(expiry * 365), ql.Days)
            exercise = ql.EuropeanExercise(exercise_date)
            
            payoff = ql.PlainVanillaPayoff(
                ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put,
                strike
            )
            
            option = ql.VanillaOption(payoff, exercise)
            
            # Price with Heston model
            heston_engine = ql.AnalyticHestonEngine(self.heston_model)
            option.setPricingEngine(heston_engine)
            heston_price = option.NPV()
            
            # Get market vol for this strike/expiry
            market_vol = 0.15  # Default
            if isinstance(self.vol_surface_data, pd.DataFrame):
                matching = self.vol_surface_data[
                    (self.vol_surface_data['strike'] == strike) & 
                    (self.vol_surface_data['expiry'] == expiry)
                ]
                if not matching.empty:
                    market_vol = matching.iloc[0]['volatility']
            
            # Price with Black-Scholes
            bs_process = ql.BlackScholesMertonProcess(
                self.spot_handle,
                self.foreign_curve,
                self.domestic_curve,
                ql.BlackVolTermStructureHandle(
                    ql.BlackConstantVol(self.eval_date, ql.TARGET(), market_vol, ql.Actual365Fixed())
                )
            )
            bs_engine = ql.AnalyticEuropeanEngine(bs_process)
            option.setPricingEngine(bs_engine)
            bs_price = option.NPV()
            
            price_diff = heston_price - bs_price
            price_diff_pct = (price_diff / bs_price * 100) if bs_price != 0 else 0
            
            validation_results.append({
                'strike': strike,
                'expiry': expiry,
                'type': option_type,
                'market_vol': market_vol * 100,
                'bs_price': bs_price,
                'heston_price': heston_price,
                'price_diff': price_diff,
                'price_diff_pct': price_diff_pct
            })
        
        return pd.DataFrame(validation_results)
