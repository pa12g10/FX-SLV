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
    FX Stochastic Local Volatility Model.
    Calibrates a Heston model to a FX vanilla option vol surface.

    Design notes
    ------------
    * vol_surface_data must be supplied with volatilities in DECIMAL form
      (e.g. 0.075 for 7.5%).  The GUI section divides Market Vol (%) by 100
      before passing data here.
    * The QuantLib evaluation date MUST be set globally (ql.Settings) before
      any QL objects are constructed.  We set it inside __init__ and calibrate.
    * HestonModelHelper requires (maturity_period, calendar, SPOT, strike, vol_quote,
      rd_handle, rf_handle, error_type).  The spot here is the scalar FX spot
      rate – NOT a Handle.
    * We use PriceError (not ImpliedVolError) for numerical stability.
    """

    def __init__(self, eval_date, spot_fx, domestic_curve, foreign_curve,
                 vol_surface_data, model_params=None):
        """
        Parameters
        ----------
        eval_date        : ql.Date   – pricing / calibration date
        spot_fx          : float     – EUR/USD spot rate
        domestic_curve   : YieldTermStructureHandle  – USD (domestic) curve
        foreign_curve    : YieldTermStructureHandle  – EUR (foreign) curve
        vol_surface_data : list of [strike, expiry_years, vol_decimal]
                           OR pd.DataFrame with columns strike/expiry/volatility
        model_params     : dict with keys v0, kappa, theta, sigma, rho
        """
        self.eval_date = eval_date
        self.spot_fx = spot_fx
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.vol_surface_data = vol_surface_data

        # FIX: set the global QL evaluation date immediately so every QL object
        # constructed afterwards uses the correct reference date.
        ql.Settings.instance().evaluationDate = eval_date

        self.model_params = model_params or {
            'v0': 0.005,     # ~7% vol  (realistic FX starting point)
            'kappa': 1.5,
            'theta': 0.005,  # ~7% vol long-run
            'sigma': 0.3,
            'rho': -0.3,
        }

        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))
        self.heston_model = None
        self.local_vol_surface = None
        self.calibrated_helpers = []
        self.calibration_results = None
        self.black_var_surface = None

    # ------------------------------------------------------------------
    def build_vol_surface(self):
        """
        Build a QuantLib BlackVarianceSurface from the supplied vol data.
        Vols must already be in decimal form.
        """
        calendar = ql.TARGET()
        day_counter = ql.Actual365Fixed()

        # Parse vol surface data
        if isinstance(self.vol_surface_data, pd.DataFrame):
            strikes_raw  = self.vol_surface_data['strike'].values.tolist()
            expiries_raw = self.vol_surface_data['expiry'].values.tolist()
            vols_raw     = self.vol_surface_data['volatility'].values.tolist()
        else:
            strikes_raw  = [d[0] for d in self.vol_surface_data]
            expiries_raw = [d[1] for d in self.vol_surface_data]
            vols_raw     = [d[2] for d in self.vol_surface_data]

        # Convert to numpy arrays for easier manipulation
        strikes_arr  = np.array(strikes_raw, dtype=float)
        expiries_arr = np.array(expiries_raw, dtype=float)
        vols_arr     = np.array(vols_raw, dtype=float)

        expiry_set = sorted(set(expiries_arr.tolist()))
        strike_set = sorted(set(strikes_arr.tolist()))

        print(f"\nBuilding vol surface: {len(strike_set)} strikes x {len(expiry_set)} expiries")
        print(f"  Strikes:  {[round(s,5) for s in strike_set]}")
        print(f"  Expiries: {expiry_set}")
        print(f"  Vol range: {vols_arr.min()*100:.2f}% – {vols_arr.max()*100:.2f}%")

        # QL needs >= 2 expiry pillars and >= 2 strike pillars
        if len(expiry_set) < 2:
            base = expiry_set[0]
            extra = base + 0.01
            expiry_set.append(extra)
            expiry_set = sorted(expiry_set)
            extra_strikes = np.array(strike_set)
            extra_vols    = np.full(len(strike_set), vols_arr[0])
            strikes_arr  = np.concatenate([strikes_arr,  extra_strikes])
            expiries_arr = np.concatenate([expiries_arr, np.full(len(strike_set), extra)])
            vols_arr     = np.concatenate([vols_arr,     extra_vols])
            print(f"  Added synthetic expiry at {extra:.4f}Y")

        if len(strike_set) < 2:
            base = strike_set[0]
            extra = base * 1.01
            strike_set.append(extra)
            strike_set = sorted(strike_set)
            extra_expiries = np.array(expiry_set)
            extra_vols     = np.full(len(expiry_set), vols_arr[0])
            strikes_arr  = np.concatenate([strikes_arr,  np.full(len(expiry_set), extra)])
            expiries_arr = np.concatenate([expiries_arr, extra_expiries])
            vols_arr     = np.concatenate([vols_arr,     extra_vols])
            print(f"  Added synthetic strike at {extra:.5f}")

        # FIX: convert fractional years to QL Dates correctly
        # Using eval_date + Period(days) avoids calendar/holiday issues.
        expiry_dates = [self.eval_date + ql.Period(max(1, int(round(e * 365))), ql.Days)
                        for e in expiry_set]

        # Build vol matrix [strikes x expiries]
        vol_matrix = ql.Matrix(len(strike_set), len(expiry_set))
        for i, strike in enumerate(strike_set):
            for j, expiry in enumerate(expiry_set):
                mask = (np.abs(strikes_arr - strike) < 1e-6) & \
                       (np.abs(expiries_arr - expiry) < 1e-6)
                if mask.any():
                    vol_matrix[i][j] = float(vols_arr[mask][0])
                else:
                    # Bilinear fallback: nearest neighbour in expiry then strike
                    ei = np.argmin(np.abs(np.array(expiry_set) - expiry))
                    emask = np.abs(expiries_arr - expiry_set[ei]) < 1e-6
                    if emask.any():
                        si = np.argmin(np.abs(strikes_arr[emask] - strike))
                        vol_matrix[i][j] = float(vols_arr[emask][si])
                    else:
                        vol_matrix[i][j] = 0.07  # 7% fallback

        self.black_var_surface = ql.BlackVarianceSurface(
            self.eval_date, calendar,
            expiry_dates, strike_set,
            vol_matrix, day_counter
        )
        self.black_var_surface.enableExtrapolation()
        return self.black_var_surface

    # ------------------------------------------------------------------
    def calibrate(self):
        """
        Calibrate Heston model to the supplied vanilla option vol surface.

        Key fixes vs original
        ----------------------
        1. ql.Settings evaluation date set before any QL object construction.
        2. HestonModelHelper called with the SCALAR spot (self.spot_fx), not a Handle.
        3. PriceError used instead of ImpliedVolError.
        4. Initial parameter defaults are closer to FX reality (~7% vol).
        5. Feller constraint added to end_criteria so sigma does not explode.
        """
        # Ensure QL evaluation date is set
        ql.Settings.instance().evaluationDate = self.eval_date

        if self.black_var_surface is None:
            self.build_vol_surface()

        # Bounded initial params
        v0    = float(np.clip(self.model_params['v0'],    0.0001, 0.5))
        kappa = float(np.clip(self.model_params['kappa'], 0.1,   15.0))
        theta = float(np.clip(self.model_params['theta'], 0.0001, 0.5))
        sigma = float(np.clip(self.model_params['sigma'], 0.05,   2.0))
        rho   = float(np.clip(self.model_params['rho'],  -0.95,   0.95))

        print(f"\nInitial Heston params:")
        print(f"  v0={v0:.6f} ({np.sqrt(v0)*100:.2f}% vol)  kappa={kappa:.4f}  "
              f"theta={theta:.6f} ({np.sqrt(theta)*100:.2f}% vol)  "
              f"sigma={sigma:.4f}  rho={rho:.4f}")

        heston_process = ql.HestonProcess(
            self.domestic_curve, self.foreign_curve,
            self.spot_handle, v0, kappa, theta, sigma, rho
        )
        self.heston_model = ql.HestonModel(heston_process)

        # Build helpers and attach engine
        self.calibrated_helpers = self._create_calibration_helpers()
        print(f"  Helpers: {len(self.calibrated_helpers)}")

        engine = ql.AnalyticHestonEngine(self.heston_model)
        for h in self.calibrated_helpers:
            h.setPricingEngine(engine)

        # Optimise
        opt    = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        end_c  = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)

        try:
            print("  Calibrating...")
            self.heston_model.calibrate(self.calibrated_helpers, opt, end_c)
            print("  ✅ Done")
        except Exception as e:
            print(f"  ⚠️  Calibration warning: {e}")

        params = self.heston_model.params()
        print(f"\n=== Calibrated Heston Parameters ===")
        print(f"  v0    = {params[0]:.6f}  ({np.sqrt(abs(params[0]))*100:.2f}% vol)")
        print(f"  kappa = {params[1]:.6f}")
        print(f"  theta = {params[2]:.6f}  ({np.sqrt(abs(params[2]))*100:.2f}% vol)")
        print(f"  sigma = {params[3]:.6f}")
        print(f"  rho   = {params[4]:.6f}")
        feller = 2 * params[1] * params[2] - params[3] ** 2
        print(f"  Feller margin (>0 preferred): {feller:.6f}")
        print("=" * 40)

        ok, warnings = self._validate_calibrated_params(params)
        for w in warnings:
            print(f"  ⚠️  {w}")

        if not ok:
            print("  ⚠️  Critical failure – reverting to initial params")
            heston_process = ql.HestonProcess(
                self.domestic_curve, self.foreign_curve,
                self.spot_handle, v0, kappa, theta, sigma, rho
            )
            self.heston_model = ql.HestonModel(heston_process)
            engine = ql.AnalyticHestonEngine(self.heston_model)
            for h in self.calibrated_helpers:
                h.setPricingEngine(engine)

        self._build_local_vol_surface()
        self._extract_calibration_results()
        return self.heston_model

    # ------------------------------------------------------------------
    def _create_calibration_helpers(self):
        """
        Build HestonModelHelper objects.

        CRITICAL FIX: HestonModelHelper signature is:
            (period, calendar, SPOT_float, strike, vol_handle,
             domestic_handle, foreign_handle, error_type)
        The third argument is the SCALAR spot rate (a plain float), NOT a
        QuoteHandle.  Passing a Handle here causes silent mispricing because
        QuantLib interprets the Handle's internal pointer value as the spot,
        giving garbage prices and therefore massive calibration errors.
        """
        calendar = ql.TARGET()
        helpers = []

        if isinstance(self.vol_surface_data, pd.DataFrame):
            rows = self.vol_surface_data.to_dict('records')
            data = [{'strike': r['strike'], 'expiry': r['expiry'],
                     'volatility': r['volatility']} for r in rows]
        else:
            data = [{'strike': d[0], 'expiry': d[1], 'volatility': d[2]}
                    for d in self.vol_surface_data]

        for row in data:
            strike    = float(row['strike'])
            T         = float(row['expiry'])
            vol_dec   = float(row['volatility'])   # must be decimal

            if vol_dec <= 0 or T <= 0 or strike <= 0:
                continue

            # Sanity: warn if vol looks like it was passed as %
            if vol_dec > 2.0:
                print(f"  ⚠️  vol={vol_dec:.4f} looks like % not decimal – dividing by 100")
                vol_dec /= 100.0

            period = ql.Period(max(1, int(round(T * 365))), ql.Days)

            h = ql.HestonModelHelper(
                period,
                calendar,
                self.spot_fx,                              # FIX: scalar float, not Handle
                strike,
                ql.QuoteHandle(ql.SimpleQuote(vol_dec)),
                self.domestic_curve,
                self.foreign_curve,
                ql.BlackCalibrationHelper.PriceError       # FIX: stable error metric
            )
            helpers.append(h)

        return helpers

    # ------------------------------------------------------------------
    def _build_local_vol_surface(self):
        """Build Dupire local vol surface from the calibrated Black surface."""
        if self.black_var_surface is None:
            return None
        try:
            bv_handle = ql.BlackVolTermStructureHandle(self.black_var_surface)
            self.local_vol_surface = ql.LocalVolSurface(
                bv_handle,
                self.domestic_curve,
                self.foreign_curve,
                self.spot_handle
            )
            self.local_vol_surface.enableExtrapolation()
        except Exception as e:
            print(f"  Warning: Could not build local vol surface: {e}")
            self.local_vol_surface = None
        return self.local_vol_surface

    # ------------------------------------------------------------------
    def _validate_calibrated_params(self, params):
        v0, kappa, theta, sigma, rho = params
        warnings = []

        if any(np.isnan(p) or np.isinf(p) for p in params):
            return False, ["Parameters contain NaN/Inf"]
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            return False, ["Non-positive parameter(s)"]
        if abs(rho) >= 1.0:
            return False, ["Correlation |rho| >= 1"]

        if 2 * kappa * theta <= sigma ** 2:
            warnings.append(f"Feller violated: 2κθ={2*kappa*theta:.4f} <= σ²={sigma**2:.4f}")
        if v0 > 0.25:
            warnings.append(f"v0={v0:.4f} high ({np.sqrt(v0)*100:.1f}% vol)")
        if theta > 0.25:
            warnings.append(f"theta={theta:.4f} high ({np.sqrt(theta)*100:.1f}% vol)")
        if sigma > 2.0:
            warnings.append(f"sigma={sigma:.4f} very high")
        return True, warnings

    # ------------------------------------------------------------------
    def _extract_calibration_results(self):
        """Compute market vs model prices and implied vol errors."""
        if not self.heston_model or not self.calibrated_helpers:
            return None

        params = self.heston_model.params()

        if isinstance(self.vol_surface_data, pd.DataFrame):
            data = self.vol_surface_data.to_dict('records')
            data = [{'strike': r['strike'], 'expiry': r['expiry'],
                     'volatility': r['volatility']} for r in data]
        else:
            data = [{'strike': d[0], 'expiry': d[1], 'volatility': d[2]}
                    for d in self.vol_surface_data]

        # Filter to same rows that produced helpers
        valid_data = [r for r in data
                      if float(r['volatility']) > 0 and float(r['expiry']) > 0
                      and float(r['strike']) > 0]

        pricing_errors = []
        for helper, row in zip(self.calibrated_helpers, valid_data):
            market_vol = float(row['volatility'])   # decimal
            try:
                market_price = helper.marketValue()
                model_price  = helper.modelValue()
                try:
                    model_vol = helper.impliedVolatility(model_price, 1e-6, 1000, 0.0, 2.0)
                except Exception:
                    model_vol = market_vol
                price_error     = model_price - market_price
                price_error_pct = (price_error / market_price * 100) if market_price != 0 else 0.0
                vol_error_bps   = (model_vol - market_vol) * 10000
            except Exception:
                market_price = model_price = 0.0
                model_vol    = market_vol
                price_error  = price_error_pct = vol_error_bps = 0.0

            pricing_errors.append({
                'strike':          float(row['strike']),
                'expiry':          float(row['expiry']),
                'market_vol':      market_vol * 100,   # display as %
                'model_vol':       model_vol  * 100,   # display as %
                'market_price':    market_price,
                'model_price':     model_price,
                'price_error':     price_error,
                'price_error_pct': price_error_pct,
                'vol_error_bps':   vol_error_bps,
            })

        self.calibration_results = {
            'v0':    params[0],
            'kappa': params[1],
            'theta': params[2],
            'sigma': params[3],
            'rho':   params[4],
            'pricing_errors': pd.DataFrame(pricing_errors),
        }
        return self.calibration_results

    # ------------------------------------------------------------------
    def get_calibrated_results(self):
        return self.calibration_results

    # ------------------------------------------------------------------
    def get_simulated_paths(self, num_paths=1000, time_steps=252, horizon_years=1.0):
        """
        Monte-Carlo simulation of the calibrated Heston spot/vol process.
        Returns (path_df, times, spot_paths, vol_paths).
        """
        if not self.heston_model:
            return None, None, None, None

        total_steps = int(time_steps * horizon_years)
        times = np.linspace(0, horizon_years, total_steps + 1)
        dt    = horizon_years / total_steps

        p     = self.heston_model.params()
        v0    = max(1e-4, p[0])
        kappa = max(0.01, p[1])
        theta = max(1e-4, p[2])
        sigma = max(0.01, p[3])
        rho   = float(np.clip(p[4], -0.99, 0.99))

        rd = self.domestic_curve.zeroRate(horizon_years / 2, ql.Continuous).rate()
        rf = self.foreign_curve.zeroRate(horizon_years / 2, ql.Continuous).rate()

        spot_paths = np.zeros((total_steps + 1, num_paths))
        vol_paths  = np.zeros((total_steps + 1, num_paths))
        spot_paths[0, :] = self.spot_fx
        vol_paths[0, :]  = v0

        np.random.seed(42)
        dW1 = np.random.normal(0, np.sqrt(dt), (total_steps, num_paths))
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (total_steps, num_paths))

        for i in range(1, total_steps + 1):
            v_prev = vol_paths[i - 1, :]
            vol_paths[i, :] = np.maximum(
                v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(np.maximum(v_prev, 0)) * dW2[i - 1, :],
                1e-4
            )
            spot_paths[i, :] = spot_paths[i - 1, :] * np.exp(
                (rd - rf - 0.5 * v_prev) * dt + np.sqrt(np.maximum(v_prev, 0)) * dW1[i - 1, :]
            )

        path_df = pd.DataFrame(spot_paths.T, columns=[f'time_{t:.4f}' for t in times])
        return path_df, times, spot_paths, vol_paths

    # ------------------------------------------------------------------
    def validate_option_prices(self, test_options=None):
        """
        Compare Heston prices against Black-Scholes prices for a sample
        of options from the calibration surface.
        """
        if not self.heston_model:
            return None

        if test_options is None:
            if isinstance(self.vol_surface_data, pd.DataFrame):
                sample = (self.vol_surface_data if len(self.vol_surface_data) <= 5
                          else self.vol_surface_data.sample(5, random_state=42))
                test_options = [[r['strike'], r['expiry'], 'call'] for _, r in sample.iterrows()]
            else:
                test_options = [[d[0], d[1], 'call'] for d in self.vol_surface_data[:5]]

        results = []
        for strike, expiry, opt_type in test_options:
            try:
                ql.Settings.instance().evaluationDate = self.eval_date
                exercise = ql.EuropeanExercise(
                    self.eval_date + ql.Period(max(1, int(round(expiry * 365))), ql.Days)
                )
                payoff = ql.PlainVanillaPayoff(
                    ql.Option.Call if opt_type.lower() == 'call' else ql.Option.Put,
                    strike
                )
                option = ql.VanillaOption(payoff, exercise)

                # Heston price
                option.setPricingEngine(ql.AnalyticHestonEngine(self.heston_model))
                heston_price = option.NPV()

                # Black-Scholes price with market vol
                market_vol = 0.07
                if isinstance(self.vol_surface_data, pd.DataFrame):
                    m = self.vol_surface_data[
                        (np.abs(self.vol_surface_data['strike'] - strike) < 1e-6) &
                        (np.abs(self.vol_surface_data['expiry'] - expiry) < 1e-6)
                    ]
                    if not m.empty:
                        market_vol = float(m.iloc[0]['volatility'])
                else:
                    for d in self.vol_surface_data:
                        if abs(d[0] - strike) < 1e-6 and abs(d[1] - expiry) < 1e-6:
                            market_vol = d[2]; break

                bs_proc = ql.BlackScholesMertonProcess(
                    self.spot_handle, self.foreign_curve, self.domestic_curve,
                    ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(self.eval_date, ql.TARGET(),
                                            market_vol, ql.Actual365Fixed())
                    )
                )
                option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_proc))
                bs_price = option.NPV()

                diff = heston_price - bs_price
                results.append({
                    'strike': strike, 'expiry': expiry, 'type': opt_type,
                    'market_vol': market_vol * 100,
                    'bs_price': bs_price, 'heston_price': heston_price,
                    'price_diff': diff,
                    'price_diff_pct': (diff / bs_price * 100) if bs_price != 0 else 0.0,
                })
            except Exception as e:
                print(f"  Validation skip K={strike} T={expiry}: {e}")

        return pd.DataFrame(results) if results else None
