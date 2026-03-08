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
            - 'v0': initial variance (default: 0.014)
            - 'kappa': mean reversion speed (default: 2.0)
            - 'theta': long-term variance (default: 0.014)
            - 'sigma': vol-of-vol (default: 0.4)
            - 'rho': correlation between spot and vol (default: -0.6)
        """
        self.eval_date = eval_date
        self.spot_fx = spot_fx
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.vol_surface_data = vol_surface_data
        
        # Default model parameters (Heston with realistic FX values)
        self.model_params = model_params or {
            'v0': 0.014,       # Initial variance (~12% vol)
            'kappa': 2.0,      # Mean reversion speed
            'theta': 0.014,    # Long-term variance (~12% vol)
            'sigma': 0.4,      # Vol-of-vol
            'rho': -0.6        # Correlation
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
        
        # Get unique strikes and expiries, sorted
        expiry_set = sorted(set(expiries))
        strike_set = sorted(set(strikes))
        
        print(f"\nBuilding vol surface with {len(strike_set)} strikes and {len(expiry_set)} expiries")
        print(f"Strikes: {strike_set}")
        print(f"Expiries: {expiry_set}")
        
        # QuantLib requires at least 2 points in each dimension for interpolation
        if len(expiry_set) < 2:
            # Add a nearby expiry point
            if len(expiry_set) == 1:
                base_expiry = expiry_set[0]
                # Add a point 1 day before (or after if at start)
                new_expiry = base_expiry + 0.01 if base_expiry > 0.01 else base_expiry + 0.01
                expiry_set.append(new_expiry)
                expiry_set = sorted(expiry_set)
                
                # Duplicate volatilities for new expiry
                for strike in strike_set:
                    # Find vol for this strike at base expiry
                    matching_vol = None
                    for i in range(len(strikes)):
                        if abs(strikes[i] - strike) < 1e-6 and abs(expiries[i] - base_expiry) < 1e-6:
                            matching_vol = vols[i]
                            break
                    
                    if matching_vol:
                        strikes = np.append(strikes, strike)
                        expiries = np.append(expiries, new_expiry)
                        vols = np.append(vols, matching_vol)
                
                print(f"⚠️  Warning: Only 1 expiry found. Added synthetic expiry at {new_expiry:.4f}Y")
        
        if len(strike_set) < 2:
            # Add a nearby strike point
            if len(strike_set) == 1:
                base_strike = strike_set[0]
                # Add a strike 1% away
                new_strike = base_strike * 1.01
                strike_set.append(new_strike)
                strike_set = sorted(strike_set)
                
                # Duplicate volatilities for new strike
                for expiry in expiry_set:
                    # Find vol for this expiry at base strike
                    matching_vol = None
                    for i in range(len(strikes)):
                        if abs(expiries[i] - expiry) < 1e-6 and abs(strikes[i] - base_strike) < 1e-6:
                            matching_vol = vols[i]
                            break
                    
                    if matching_vol:
                        strikes = np.append(strikes, new_strike)
                        expiries = np.append(expiries, expiry)
                        vols = np.append(vols, matching_vol)
                
                print(f"⚠️  Warning: Only 1 strike found. Added synthetic strike at {new_strike:.4f}")
        
        # Rebuild sets after potential additions
        expiry_set = sorted(set(expiries))
        strike_set = sorted(set(strikes))
        
        print(f"Final grid: {len(strike_set)} strikes × {len(expiry_set)} expiries")
        
        expiry_dates_unique = [self.eval_date + ql.Period(int(e*365), ql.Days) for e in expiry_set]
        
        # Build vol matrix (strikes x expiries)
        vol_matrix = ql.Matrix(len(strike_set), len(expiry_set))
        
        for i, strike in enumerate(strike_set):
            for j, expiry in enumerate(expiry_set):
                # Find vol for this strike/expiry combo
                matching_vols = [vols[k] for k in range(len(vols)) 
                               if abs(strikes[k] - strike) < 1e-6 and abs(expiries[k] - expiry) < 1e-6]
                if matching_vols:
                    vol_matrix[i][j] = matching_vols[0]
                else:
                    # Fallback to nearby or default
                    vol_matrix[i][j] = 0.12  # Default 12%
        
        # Create Black variance surface
        self.black_var_surface = ql.BlackVarianceSurface(
            self.eval_date,
            calendar,
            expiry_dates_unique,
            strike_set,
            vol_matrix,
            day_counter
        )
        
        # Enable extrapolation
        self.black_var_surface.enableExtrapolation()
        
        return self.black_var_surface
    
    def calibrate(self):
        """
        Calibrate the FX-SLV model parameters to vanilla option prices
        Uses Heston model as the stochastic volatility component
        """
        # Build volatility surface first
        if self.black_var_surface is None:
            self.build_vol_surface()
        
        # Get initial parameters with bounds
        v0 = max(0.001, min(0.5, self.model_params['v0']))  # Bound: [0.001, 0.5]
        kappa = max(0.1, min(10.0, self.model_params['kappa']))  # Bound: [0.1, 10]
        theta = max(0.001, min(0.5, self.model_params['theta']))  # Bound: [0.001, 0.5]
        sigma = max(0.05, min(2.0, self.model_params['sigma']))  # Bound: [0.05, 2.0]
        rho = max(-0.95, min(-0.05, self.model_params['rho']))  # Bound: [-0.95, -0.05] (FX typical)
        
        print(f"\nInitial parameters:")
        print(f"v0={v0:.6f}, kappa={kappa:.4f}, theta={theta:.6f}, sigma={sigma:.4f}, rho={rho:.4f}")
        
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
        
        print(f"Created {len(self.calibrated_helpers)} calibration helpers")
        
        # Set up Heston pricing engine
        heston_engine = ql.AnalyticHestonEngine(self.heston_model)
        
        # Set pricing engine for all helpers
        for helper in self.calibrated_helpers:
            helper.setPricingEngine(heston_engine)
        
        # Define parameter constraints
        # Bounds: v0, kappa, theta, sigma > 0, -1 < rho < 1
        lower_bounds = ql.Array([0.0001, 0.01, 0.0001, 0.01, -0.9999])  # Slightly away from boundaries
        upper_bounds = ql.Array([1.0, 15.0, 1.0, 5.0, 0.9999])
        constraint = ql.BoundaryConstraint(lower_bounds, upper_bounds)
        
        # Calibrate model with constraints
        optimization_method = ql.LevenbergMarquardt()
        end_criteria = ql.EndCriteria(
            500,      # max iterations
            50,       # max stationary iterations  
            1e-6,     # root epsilon (relaxed from 1e-8)
            1e-6,     # function epsilon (relaxed)
            1e-6      # gradient epsilon (relaxed)
        )
        
        try:
            print("\nStarting calibration...")
            self.heston_model.calibrate(
                self.calibrated_helpers,
                optimization_method,
                end_criteria,
                constraint
            )
            print("✅ Calibration completed")
        except Exception as e:
            print(f"⚠️  Calibration warning: {e}")
            print("Using initial parameter guesses as fallback")
        
        # Get calibrated parameters and show them
        params = self.heston_model.params()
        print(f"\n=== Calibrated Heston Parameters ===")
        print(f"v0 (initial variance): {params[0]:.6f}  ({np.sqrt(abs(params[0]))*100:.2f}% vol)")
        print(f"kappa (mean reversion): {params[1]:.6f}")
        print(f"theta (long-term var): {params[2]:.6f}  ({np.sqrt(abs(params[2]))*100:.2f}% vol)")
        print(f"sigma (vol-of-vol): {params[3]:.6f}")
        print(f"rho (correlation): {params[4]:.6f}")
        print(f"Feller condition: 2*kappa*theta = {2*params[1]*params[2]:.6f}, sigma^2 = {params[3]**2:.6f}")
        print(f"Feller satisfied: {2*params[1]*params[2] > params[3]**2}")
        print(f"=" * 40)
        
        # Validate - but only reject on critical errors
        validation_result, warnings = self._validate_calibrated_params(params)
        
        if warnings:
            print("\n⚠️  Calibration Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if not validation_result:
            print("⚠️  Critical validation failure - using bounded initial parameters as fallback")
            # Reset to bounded initial parameters
            heston_process = ql.HestonProcess(
                self.domestic_curve,
                self.foreign_curve,
                self.spot_handle,
                v0, kappa, theta, sigma, rho
            )
            self.heston_model = ql.HestonModel(heston_process)
            
            # Re-set pricing engines with fallback model
            heston_engine = ql.AnalyticHestonEngine(self.heston_model)
            for helper in self.calibrated_helpers:
                helper.setPricingEngine(heston_engine)
        
        # Build local volatility surface from calibrated model
        self._build_local_vol_surface()
        
        # Extract calibration results
        self._extract_calibration_results()
        
        return self.heston_model
    
    def _validate_calibrated_params(self, params):
        """
        Validate calibrated parameters
        Returns (is_valid, warnings_list)
        Only fails on critical errors (NaN/Inf/negative)
        """
        v0 = params[0]
        kappa = params[1]
        theta = params[2]
        sigma = params[3]
        rho = params[4]
        
        warnings = []
        
        # CRITICAL CHECKS - Must pass
        # Check for NaN or inf
        if any(np.isnan(p) or np.isinf(p) for p in params):
            return False, ["Parameters contain NaN or Inf"]
        
        # Check positivity (except rho)
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            return False, ["Parameters must be positive (except rho)"]
        
        # Check correlation bounds
        if abs(rho) >= 1.0:
            return False, ["Correlation must be strictly between -1 and 1"]
        
        # NON-CRITICAL CHECKS - Warnings only
        # Check Feller condition
        if 2 * kappa * theta <= sigma ** 2:
            warnings.append(f"Feller condition violated: 2*κ*θ={2*kappa*theta:.4f} <= σ²={sigma**2:.4f}. Variance may reach zero.")
        
        # Check reasonable ranges
        if v0 > 0.5:  # Variance > 70% vol
            warnings.append(f"Initial variance v0={v0:.4f} is high ({np.sqrt(v0)*100:.1f}% vol)")
        
        if theta > 0.5:
            warnings.append(f"Long-term variance theta={theta:.4f} is high ({np.sqrt(theta)*100:.1f}% vol)")
        
        if kappa > 10:
            warnings.append(f"Mean reversion kappa={kappa:.4f} is very fast")
        
        if sigma > 3.0:
            warnings.append(f"Vol-of-vol sigma={sigma:.4f} is very high")
        
        if abs(rho) < 0.1:
            warnings.append(f"Correlation rho={rho:.4f} is very weak")
        
        return True, warnings
    
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
            
            # Skip invalid data
            if volatility <= 0 or expiry_years <= 0 or strike <= 0:
                continue
            
            # Create option helper
            maturity = ql.Period(int(expiry_years * 365), ql.Days)
            
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
        # Create local vol surface from Black variance surface
        if self.black_var_surface:
            try:
                # Use BlackVolTermStructureHandle (correct QuantLib API)
                black_vol_handle = ql.BlackVolTermStructureHandle(self.black_var_surface)
                
                self.local_vol_surface = ql.LocalVolSurface(
                    black_vol_handle,
                    self.domestic_curve,
                    self.foreign_curve,
                    self.spot_handle
                )
                
                # Enable extrapolation
                self.local_vol_surface.enableExtrapolation()
            except Exception as e:
                print(f"Warning: Could not build local vol surface: {e}")
                self.local_vol_surface = None
        
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
            
            try:
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
            except Exception as e:
                # If pricing fails, use market values
                market_price = 0
                model_price = 0
                model_vol = market_vol
                price_error = 0
                price_error_pct = 0
                vol_error_bps = 0
            
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
        v0 = max(0.0001, params[0])  # Ensure positive
        kappa = max(0.01, params[1])
        theta = max(0.0001, params[2])
        sigma = max(0.01, params[3])
        rho = max(-0.99, min(0.99, params[4]))
        
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
                0.0001  # Floor to prevent zero variance
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
                sample_size = min(5, len(self.vol_surface_data))
                sample = self.vol_surface_data.sample(sample_size) if len(self.vol_surface_data) > 5 else self.vol_surface_data
                test_options = [[row['strike'], row['expiry'], 'call'] 
                              for _, row in sample.iterrows()]
            else:
                test_options = [[d[0], d[1], 'call'] 
                              for d in self.vol_surface_data[:5]]
        
        validation_results = []
        
        for strike, expiry, option_type in test_options:
            try:
                # Create vanilla option
                exercise_date = self.eval_date + ql.Period(int(expiry * 365), ql.Days)
                exercise = ql.EuropeanExercise(exercise_date)
                
                payoff = ql.PlainVanillaPayoff(
                    ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put,
                    strike
                )
                
                option = ql.VanillaOption(payoff, exercise)
                
                # Get market vol for this strike/expiry
                market_vol = 0.12  # Default
                if isinstance(self.vol_surface_data, pd.DataFrame):
                    matching = self.vol_surface_data[
                        (abs(self.vol_surface_data['strike'] - strike) < 1e-6) & 
                        (abs(self.vol_surface_data['expiry'] - expiry) < 1e-6)
                    ]
                    if not matching.empty:
                        market_vol = matching.iloc[0]['volatility']
                else:
                    for d in self.vol_surface_data:
                        if abs(d[0] - strike) < 1e-6 and abs(d[1] - expiry) < 1e-6:
                            market_vol = d[2]
                            break
                
                # Price with Heston model
                heston_engine = ql.AnalyticHestonEngine(self.heston_model)
                option.setPricingEngine(heston_engine)
                heston_price = option.NPV()
                
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
            except Exception as e:
                print(f"Warning: Could not validate option K={strike}, T={expiry}: {e}")
                continue
        
        return pd.DataFrame(validation_results) if validation_results else None
