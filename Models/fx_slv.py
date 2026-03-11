# FX Stochastic Local Volatility Model
import QuantLib as ql
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class FXStochasticLocalVol:
    """
    FX Stochastic Local Volatility Model.
    Calibrates a Heston model to a FX vanilla option vol surface.

    Design contract
    ---------------
    * vol_surface_data must supply volatilities in DECIMAL form (0.075 = 7.5%).
    * The QuantLib global evaluation date is set inside __init__ and calibrate().
    * HestonModelHelper 3rd argument is the scalar float spot rate.
    * We use PriceError for numerical stability.
    * sigma is kept strictly positive via a constrained optimizer wrapper.
    """

    def __init__(self, eval_date, spot_fx, domestic_curve, foreign_curve,
                 vol_surface_data, model_params=None):
        self.eval_date       = eval_date
        self.spot_fx         = spot_fx
        self.domestic_curve  = domestic_curve
        self.foreign_curve   = foreign_curve
        self.vol_surface_data = vol_surface_data

        # Set global QL date immediately
        ql.Settings.instance().evaluationDate = eval_date

        # Default initial params: v0 / theta close to ATM vol (~7% => 0.0049)
        self.model_params = model_params or {
            'v0':    0.005,
            'kappa': 1.5,
            'theta': 0.005,
            'sigma': 0.3,
            'rho':   -0.3,
        }

        self.spot_handle          = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))
        self.heston_model         = None
        self.local_vol_surface    = None
        self.calibrated_helpers   = []
        self.calibration_results  = None
        self.black_var_surface    = None

    # ------------------------------------------------------------------
    def build_vol_surface(self):
        """
        Build a QuantLib BlackVarianceSurface from the supplied vol data.
        Vols must already be in decimal form.
        """
        calendar    = ql.TARGET()
        day_counter = ql.Actual365Fixed()

        if isinstance(self.vol_surface_data, pd.DataFrame):
            strikes_raw  = self.vol_surface_data['strike'].values.tolist()
            expiries_raw = self.vol_surface_data['expiry'].values.tolist()
            vols_raw     = self.vol_surface_data['volatility'].values.tolist()
        else:
            strikes_raw  = [d[0] for d in self.vol_surface_data]
            expiries_raw = [d[1] for d in self.vol_surface_data]
            vols_raw     = [d[2] for d in self.vol_surface_data]

        strikes_arr  = np.array(strikes_raw,  dtype=float)
        expiries_arr = np.array(expiries_raw, dtype=float)
        vols_arr     = np.array(vols_raw,     dtype=float)

        # Sanity check: detect % vs decimal
        if vols_arr.max() > 2.0:
            print(f"  WARNING: max vol={vols_arr.max():.2f} looks like %, dividing by 100")
            vols_arr /= 100.0

        expiry_set = sorted(set(expiries_arr.tolist()))
        strike_set = sorted(set(strikes_arr.tolist()))

        print(f"\nBuilding vol surface: {len(strike_set)} strikes x {len(expiry_set)} expiries")
        print(f"  Vol range: {vols_arr.min()*100:.2f}% – {vols_arr.max()*100:.2f}%")

        # Ensure >= 2 pillars in each dimension
        if len(expiry_set) < 2:
            extra = expiry_set[0] + 0.01
            expiry_set = sorted(expiry_set + [extra])
            extra_v = np.full(len(strike_set), vols_arr[0])
            for s, v in zip(strike_set, extra_v):
                strikes_arr  = np.append(strikes_arr,  s)
                expiries_arr = np.append(expiries_arr, extra)
                vols_arr     = np.append(vols_arr,     v)

        if len(strike_set) < 2:
            extra = strike_set[0] * 1.01
            strike_set = sorted(strike_set + [extra])
            extra_v = np.full(len(expiry_set), vols_arr[0])
            for e, v in zip(expiry_set, extra_v):
                strikes_arr  = np.append(strikes_arr,  extra)
                expiries_arr = np.append(expiries_arr, e)
                vols_arr     = np.append(vols_arr,     v)

        expiry_dates = [
            self.eval_date + ql.Period(max(1, int(round(e * 365))), ql.Days)
            for e in expiry_set
        ]

        vol_matrix = ql.Matrix(len(strike_set), len(expiry_set))
        for i, strike in enumerate(strike_set):
            for j, expiry in enumerate(expiry_set):
                mask = (np.abs(strikes_arr - strike) < 1e-6) & \
                       (np.abs(expiries_arr - expiry) < 1e-6)
                if mask.any():
                    vol_matrix[i][j] = float(vols_arr[mask][0])
                else:
                    ei   = np.argmin(np.abs(np.array(expiry_set) - expiry))
                    emsk = np.abs(expiries_arr - expiry_set[ei]) < 1e-6
                    si   = np.argmin(np.abs(strikes_arr[emsk] - strike)) if emsk.any() else 0
                    vol_matrix[i][j] = float(vols_arr[emsk][si]) if emsk.any() else 0.07

        self.black_var_surface = ql.BlackVarianceSurface(
            self.eval_date, calendar, expiry_dates, strike_set,
            vol_matrix, day_counter
        )
        self.black_var_surface.enableExtrapolation()
        return self.black_var_surface

    # ------------------------------------------------------------------
    def calibrate(self):
        """
        Calibrate Heston model to the supplied vanilla option vol surface.

        Optimizer strategy
        ------------------
        QuantLib's LevenbergMarquardt does NOT enforce parameter bounds.
        We therefore use a two-stage approach:
          1.  scipy.optimize.least_squares with hard bounds to get a good
              starting point with sigma > 0 guaranteed.
          2.  QuantLib LM from that constrained starting point for final polish.

        Parameter bounds used
        ----------------------
          v0    : [1e-4,  0.50]   (vol 1% – 70%)
          kappa : [0.10,  15.0]
          theta : [1e-4,  0.50]
          sigma : [0.01,   2.0]   ← CRITICAL: lower bound > 0 prevents sign flip
          rho   : [-0.95,  0.95]
        """
        ql.Settings.instance().evaluationDate = self.eval_date

        if self.black_var_surface is None:
            self.build_vol_surface()

        # Clip initial params to bounds
        BOUNDS_LO = np.array([1e-4, 0.10, 1e-4, 0.01, -0.95])
        BOUNDS_HI = np.array([0.50, 15.0, 0.50,  2.0,  0.95])

        p0 = np.array([
            float(np.clip(self.model_params['v0'],    BOUNDS_LO[0], BOUNDS_HI[0])),
            float(np.clip(self.model_params['kappa'], BOUNDS_LO[1], BOUNDS_HI[1])),
            float(np.clip(self.model_params['theta'], BOUNDS_LO[2], BOUNDS_HI[2])),
            float(np.clip(self.model_params['sigma'], BOUNDS_LO[3], BOUNDS_HI[3])),
            float(np.clip(self.model_params['rho'],   BOUNDS_LO[4], BOUNDS_HI[4])),
        ])

        print(f"\nInitial Heston params:")
        print(f"  v0={p0[0]:.6f} ({np.sqrt(p0[0])*100:.2f}% vol)  kappa={p0[1]:.4f}  "
              f"theta={p0[2]:.6f} ({np.sqrt(p0[2])*100:.2f}% vol)  "
              f"sigma={p0[3]:.4f}  rho={p0[4]:.4f}")

        # Build helpers once
        self._make_process_model_engine(p0)
        self.calibrated_helpers = self._create_calibration_helpers()
        print(f"  Helpers: {len(self.calibrated_helpers)}")

        if len(self.calibrated_helpers) == 0:
            raise RuntimeError("No valid calibration helpers could be constructed. "
                               "Check vol surface data (strikes, expiries, vols).")

        # ------------------------------------------------------------------
        # Stage 1: scipy constrained least-squares
        # ------------------------------------------------------------------
        try:
            from scipy.optimize import least_squares

            def residuals_scipy(params):
                try:
                    self._make_process_model_engine(params)
                    res = []
                    for h in self.calibrated_helpers:
                        try:
                            res.append(h.calibrationError())
                        except Exception:
                            res.append(1.0)
                    return np.array(res, dtype=float)
                except Exception:
                    return np.ones(len(self.calibrated_helpers))

            print("  Stage 1: scipy constrained LM...")
            sol = least_squares(
                residuals_scipy, p0,
                bounds=(BOUNDS_LO, BOUNDS_HI),
                method='trf',
                ftol=1e-9, xtol=1e-9, gtol=1e-9,
                max_nfev=2000,
                verbose=0,
            )
            p1 = sol.x
            print(f"  scipy cost={sol.cost:.6e}  nfev={sol.nfev}  status={sol.status}")
        except ImportError:
            print("  scipy not available – skipping Stage 1")
            p1 = p0.copy()

        # ------------------------------------------------------------------
        # Stage 2: QuantLib LM polish from scipy solution
        # ------------------------------------------------------------------
        print("  Stage 2: QuantLib LM polish...")
        self._make_process_model_engine(p1)

        opt   = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        end_c = ql.EndCriteria(2000, 200, 1e-10, 1e-10, 1e-10)

        try:
            self.heston_model.calibrate(self.calibrated_helpers, opt, end_c)
            print("  ✅ QL calibration done")
        except Exception as e:
            print(f"  ⚠️  QL calibration warning: {e}")

        params = list(self.heston_model.params())

        # Post-calibration: if sigma went negative (QL is unconstrained),
        # reflect it back and re-set the model.
        if params[3] < 0:
            print(f"  ⚠️  sigma went negative ({params[3]:.6f}) – reflecting to positive")
            params[3] = abs(params[3])
            self._make_process_model_engine(params)

        # Clip rho hard to (-1, 1)
        if abs(params[4]) >= 0.99:
            params[4] = float(np.clip(params[4], -0.95, 0.95))
            self._make_process_model_engine(params)

        params = list(self.heston_model.params())
        print(f"\n=== Calibrated Heston Parameters ===")
        print(f"  v0    = {params[0]:.6f}  ({np.sqrt(abs(params[0]))*100:.2f}% vol)")
        print(f"  kappa = {params[1]:.6f}")
        print(f"  theta = {params[2]:.6f}  ({np.sqrt(abs(params[2]))*100:.2f}% vol)")
        print(f"  sigma = {params[3]:.6f}")
        print(f"  rho   = {params[4]:.6f}")
        feller = 2 * params[1] * params[2] - params[3] ** 2
        print(f"  Feller margin (2κθ - σ² = {feller:.6f})")
        print("=" * 40)

        ok, warnings = self._validate_calibrated_params(params)
        for w in warnings:
            print(f"  ⚠️  {w}")

        if not ok:
            print("  ⚠️  Validation failed – using scipy solution as fallback")
            self._make_process_model_engine(p1)

        self._build_local_vol_surface()
        self._extract_calibration_results()
        return self.heston_model

    # ------------------------------------------------------------------
    def _make_process_model_engine(self, params):
        """Construct / update HestonProcess, HestonModel and engine in place."""
        v0, kappa, theta, sigma, rho = [
            float(np.clip(params[i], lo, hi))
            for i, (lo, hi) in enumerate([
                (1e-4, 0.50), (0.10, 15.0), (1e-4, 0.50), (1e-4, 2.0), (-0.9999, 0.9999)
            ])
        ]
        process = ql.HestonProcess(
            self.domestic_curve, self.foreign_curve,
            self.spot_handle, v0, kappa, theta, sigma, rho
        )
        self.heston_model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(self.heston_model)
        for h in self.calibrated_helpers:
            h.setPricingEngine(engine)
        return self.heston_model

    # ------------------------------------------------------------------
    def _create_calibration_helpers(self):
        """
        Build HestonModelHelper objects.
        Third argument = scalar float spot (NOT a Handle).
        Error type     = PriceError (stable everywhere, not just near ATM).
        """
        calendar = ql.TARGET()
        helpers  = []

        if isinstance(self.vol_surface_data, pd.DataFrame):
            rows = [{'strike': r['strike'], 'expiry': r['expiry'],
                     'volatility': r['volatility']}
                    for _, r in self.vol_surface_data.iterrows()]
        else:
            rows = [{'strike': d[0], 'expiry': d[1], 'volatility': d[2]}
                    for d in self.vol_surface_data]

        for row in rows:
            strike  = float(row['strike'])
            T       = float(row['expiry'])
            vol_dec = float(row['volatility'])

            if vol_dec <= 0 or T <= 0 or strike <= 0:
                continue
            if vol_dec > 2.0:          # passed as %, divide silently
                vol_dec /= 100.0

            period = ql.Period(max(1, int(round(T * 365))), ql.Days)
            h = ql.HestonModelHelper(
                period,
                calendar,
                self.spot_fx,                               # scalar float
                strike,
                ql.QuoteHandle(ql.SimpleQuote(vol_dec)),
                self.domestic_curve,
                self.foreign_curve,
                ql.BlackCalibrationHelper.PriceError,       # stable metric
            )
            helpers.append(h)

        return helpers

    # ------------------------------------------------------------------
    def _build_local_vol_surface(self):
        if self.black_var_surface is None:
            return None
        try:
            bv_handle = ql.BlackVolTermStructureHandle(self.black_var_surface)
            self.local_vol_surface = ql.LocalVolSurface(
                bv_handle, self.domestic_curve, self.foreign_curve, self.spot_handle
            )
            self.local_vol_surface.enableExtrapolation()
        except Exception as e:
            print(f"  Warning: local vol surface: {e}")
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
            warnings.append(f"Feller violated: 2κθ={2*kappa*theta:.4f} ≤ σ²={sigma**2:.4f}")
        if v0    > 0.25: warnings.append(f"v0={v0:.4f} very high ({np.sqrt(v0)*100:.1f}% vol)")
        if theta > 0.25: warnings.append(f"theta={theta:.4f} very high")
        if sigma > 2.0:  warnings.append(f"sigma={sigma:.4f} very high")
        return True, warnings

    # ------------------------------------------------------------------
    def _extract_calibration_results(self):
        if not self.heston_model or not self.calibrated_helpers:
            return None

        params = self.heston_model.params()

        if isinstance(self.vol_surface_data, pd.DataFrame):
            data = [{'strike': r['strike'], 'expiry': r['expiry'],
                     'volatility': r['volatility']}
                    for _, r in self.vol_surface_data.iterrows()]
        else:
            data = [{'strike': d[0], 'expiry': d[1], 'volatility': d[2]}
                    for d in self.vol_surface_data]

        valid_data = [r for r in data
                      if float(r['volatility']) > 0 and float(r['expiry']) > 0
                      and float(r['strike']) > 0]

        pricing_errors = []
        for helper, row in zip(self.calibrated_helpers, valid_data):
            market_vol = float(row['volatility'])
            if market_vol > 2.0:
                market_vol /= 100.0
            try:
                market_price    = helper.marketValue()
                model_price     = helper.modelValue()
                try:
                    model_vol = helper.impliedVolatility(
                        model_price, 1e-6, 1000, 0.001, 2.0)
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
                'market_vol':      market_vol * 100,
                'model_vol':       model_vol  * 100,
                'market_price':    market_price,
                'model_price':     model_price,
                'price_error':     price_error,
                'price_error_pct': price_error_pct,
                'vol_error_bps':   vol_error_bps,
            })

        self.calibration_results = {
            'v0':    params[0], 'kappa': params[1],
            'theta': params[2], 'sigma': params[3], 'rho': params[4],
            'pricing_errors': pd.DataFrame(pricing_errors),
        }
        return self.calibration_results

    # ------------------------------------------------------------------
    def get_calibrated_results(self):
        return self.calibration_results

    # ------------------------------------------------------------------
    def get_simulated_paths(self, num_paths=1000, time_steps=252, horizon_years=1.0):
        if not self.heston_model:
            return None, None, None, None

        total_steps = int(time_steps * horizon_years)
        times = np.linspace(0, horizon_years, total_steps + 1)
        dt    = horizon_years / total_steps

        p     = self.heston_model.params()
        v0    = max(1e-4, p[0])
        kappa = max(0.01, p[1])
        theta = max(1e-4, p[2])
        sigma = max(0.01, abs(p[3]))
        rho   = float(np.clip(p[4], -0.99, 0.99))

        rd = self.domestic_curve.zeroRate(horizon_years / 2, ql.Continuous).rate()
        rf = self.foreign_curve.zeroRate(horizon_years / 2, ql.Continuous).rate()

        spot_paths = np.zeros((total_steps + 1, num_paths))
        vol_paths  = np.zeros((total_steps + 1, num_paths))
        spot_paths[0, :] = self.spot_fx
        vol_paths[0, :]  = v0

        np.random.seed(42)
        dW1 = np.random.normal(0, np.sqrt(dt), (total_steps, num_paths))
        dW2 = (rho * dW1 +
               np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), (total_steps, num_paths)))

        for i in range(1, total_steps + 1):
            v = vol_paths[i - 1, :]
            vol_paths[i, :] = np.maximum(
                v + kappa * (theta - v) * dt + sigma * np.sqrt(np.maximum(v, 0)) * dW2[i-1, :],
                1e-4
            )
            spot_paths[i, :] = spot_paths[i-1, :] * np.exp(
                (rd - rf - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * dW1[i-1, :]
            )

        path_df = pd.DataFrame(spot_paths.T, columns=[f'time_{t:.4f}' for t in times])
        return path_df, times, spot_paths, vol_paths

    # ------------------------------------------------------------------
    def validate_option_prices(self, test_options=None):
        if not self.heston_model:
            return None

        if test_options is None:
            if isinstance(self.vol_surface_data, pd.DataFrame):
                df = self.vol_surface_data
                sample = df if len(df) <= 5 else df.sample(min(5, len(df)), random_state=42)
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
                    ql.Option.Call if opt_type.lower() == 'call' else ql.Option.Put, strike
                )
                option = ql.VanillaOption(payoff, exercise)

                option.setPricingEngine(ql.AnalyticHestonEngine(self.heston_model))
                heston_price = option.NPV()

                market_vol = 0.07
                if isinstance(self.vol_surface_data, pd.DataFrame):
                    m = self.vol_surface_data[
                        (np.abs(self.vol_surface_data['strike'] - strike) < 1e-6) &
                        (np.abs(self.vol_surface_data['expiry'] - expiry) < 1e-6)
                    ]
                    if not m.empty:
                        market_vol = float(m.iloc[0]['volatility'])
                        if market_vol > 2.0:
                            market_vol /= 100.0
                else:
                    for d in self.vol_surface_data:
                        if abs(d[0] - strike) < 1e-6 and abs(d[1] - expiry) < 1e-6:
                            market_vol = d[2] if d[2] <= 2.0 else d[2] / 100.0
                            break

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
                    'market_vol':     market_vol * 100,
                    'bs_price':       bs_price,
                    'heston_price':   heston_price,
                    'price_diff':     diff,
                    'price_diff_pct': (diff / bs_price * 100) if bs_price != 0 else 0.0,
                })
            except Exception as e:
                print(f"  Validation skip K={strike} T={expiry}: {e}")

        return pd.DataFrame(results) if results else None
