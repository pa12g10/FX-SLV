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

    Calibration design
    ------------------
    We calibrate purely in IMPLIED-VOL space using a direct scipy
    least_squares call.  This avoids ALL of the pathologies that come
    from using QuantLib's PriceError / calibrationError():

      * PriceError residuals are raw option prices whose magnitude grows
        ~100x from 1W to 2Y, so long-dated instruments completely dominate
        the optimizer and short-end constraints are ignored.

      * calibrationError() for PriceError returns  (model_price - market_price)
        which is in the same mis-scaled units.

      * ImpliedVolError requires a vol inversion at every evaluation; when
        v0 is far from the market (e.g. 70% vs 7%) the inversion fails and
        returns NaN, stalling the optimizer.

    The direct approach:
      residual_i = (heston_iv(K_i, T_i) - market_iv_i) * 100   [in % vol]

    All residuals are O(0.1–1) regardless of expiry, giving the optimizer
    a well-conditioned Jacobian from the very first step.

    We also use tight bounds on v0 and theta (max 25% vol = 0.0625) so
    the optimizer cannot escape to the degenerate v0=0.50 region.
    """

    # Hard parameter bounds  [v0, kappa, theta, sigma, rho]
    _LO = np.array([1e-4,  0.10, 1e-4, 0.05, -0.95])
    _HI = np.array([0.0625, 15.0, 0.0625, 1.50,  0.95])  # 0.0625 = (25% vol)^2

    def __init__(self, eval_date, spot_fx, domestic_curve, foreign_curve,
                 vol_surface_data, model_params=None):
        self.eval_date        = eval_date
        self.spot_fx          = spot_fx
        self.domestic_curve   = domestic_curve
        self.foreign_curve    = foreign_curve
        self.vol_surface_data = vol_surface_data

        ql.Settings.instance().evaluationDate = eval_date

        self.model_params = model_params or {
            'v0': 0.0042, 'kappa': 1.5, 'theta': 0.0056, 'sigma': 0.30, 'rho': -0.30
        }

        self.spot_handle         = ql.QuoteHandle(ql.SimpleQuote(self.spot_fx))
        self.heston_model        = None
        self.local_vol_surface   = None
        self.calibrated_helpers  = []
        self.calibration_results = None
        self.black_var_surface   = None

        # Parsed calibration data (set in calibrate)
        self._cal_strikes  = None
        self._cal_expiries = None
        self._cal_vols     = None   # decimal

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def build_vol_surface(self):
        """Build BlackVarianceSurface for Dupire local vol layer."""
        calendar    = ql.TARGET()
        day_counter = ql.Actual365Fixed()

        S, E, V = self._parse_vol_data(self.vol_surface_data)

        expiry_set = sorted(set(E.tolist()))
        strike_set = sorted(set(S.tolist()))

        print(f"\nBuilding vol surface: {len(strike_set)} strikes x {len(expiry_set)} expiries")
        print(f"  Vol range: {V.min()*100:.2f}% - {V.max()*100:.2f}%")

        if len(expiry_set) < 2:
            extra = expiry_set[0] + 0.01
            expiry_set = sorted(expiry_set + [extra])
            for s in strike_set:
                S = np.append(S, s);  E = np.append(E, extra);  V = np.append(V, V[0])
        if len(strike_set) < 2:
            extra = strike_set[0] * 1.01
            strike_set = sorted(strike_set + [extra])
            for e in expiry_set:
                S = np.append(S, extra);  E = np.append(E, e);  V = np.append(V, V[0])

        expiry_dates = [
            self.eval_date + ql.Period(max(1, int(round(e * 365))), ql.Days)
            for e in expiry_set
        ]

        vol_matrix = ql.Matrix(len(strike_set), len(expiry_set))
        for i, strike in enumerate(strike_set):
            for j, expiry in enumerate(expiry_set):
                mask = (np.abs(S - strike) < 1e-6) & (np.abs(E - expiry) < 1e-6)
                if mask.any():
                    vol_matrix[i][j] = float(V[mask][0])
                else:
                    ei   = np.argmin(np.abs(np.array(expiry_set) - expiry))
                    emsk = np.abs(E - expiry_set[ei]) < 1e-6
                    si   = np.argmin(np.abs(S[emsk] - strike)) if emsk.any() else 0
                    vol_matrix[i][j] = float(V[emsk][si]) if emsk.any() else 0.07

        self.black_var_surface = ql.BlackVarianceSurface(
            self.eval_date, calendar, expiry_dates, strike_set, vol_matrix, day_counter
        )
        self.black_var_surface.enableExtrapolation()
        return self.black_var_surface

    def calibrate(self):
        """
        Calibrate Heston model using direct implied-vol residuals.

        Residual vector
        ---------------
        r_i = (heston_iv(K_i, T_i) - sigma_market_i) * 100   [% vol units]

        All residuals O(0.1-1%) regardless of expiry => well-conditioned.

        Strategy
        --------
        1. scipy.least_squares (TRF, bounded) from user initial guess
        2. If cost > threshold, try 3 additional starting points
        3. Keep the best solution; no QL polish (it's unconstrained and corrupts)
        """
        ql.Settings.instance().evaluationDate = self.eval_date

        if self.black_var_surface is None:
            self.build_vol_surface()

        # Parse and store calibration data
        S, E, V = self._parse_vol_data(self.vol_surface_data)
        self._cal_strikes  = S
        self._cal_expiries = E
        self._cal_vols     = V   # decimal

        # Build QL helpers (used only for result extraction / engine attachment)
        self._make_process_and_model(self._clip(np.array([
            self.model_params['v0'],    self.model_params['kappa'],
            self.model_params['theta'], self.model_params['sigma'],
            self.model_params['rho']
        ])))
        self.calibrated_helpers = self._create_helpers()
        print(f"  Helpers: {len(self.calibrated_helpers)}")

        if len(self.calibrated_helpers) == 0:
            raise RuntimeError("No calibration helpers constructed – check vol data.")

        # Initial point
        p0 = self._clip(np.array([
            self.model_params['v0'],    self.model_params['kappa'],
            self.model_params['theta'], self.model_params['sigma'],
            self.model_params['rho']
        ]))

        print(f"\nInitial Heston params:")
        print(f"  v0={p0[0]:.6f} ({np.sqrt(p0[0])*100:.2f}% vol)  kappa={p0[1]:.4f}  "
              f"theta={p0[2]:.6f} ({np.sqrt(p0[2])*100:.2f}% vol)  "
              f"sigma={p0[3]:.4f}  rho={p0[4]:.4f}")

        best_params, best_cost = self._run_scipy(p0)

        # Multi-start: try alternate seeds if first solution is poor
        COST_THRESHOLD = 0.5   # 0.5 means avg |IV error| ~ sqrt(0.5/N) % per helper
        if best_cost > COST_THRESHOLD:
            seeds = [
                # [v0,      kappa, theta,   sigma, rho  ]
                [0.0042,  2.0,   0.0056,  0.40,  -0.50],
                [0.0056,  1.0,   0.0056,  0.20,  -0.20],
                [0.0030,  3.0,   0.0042,  0.50,  -0.40],
                [0.0056,  1.5,   0.0072,  0.35,  -0.30],
            ]
            for seed in seeds:
                p_seed = self._clip(np.array(seed, dtype=float))
                p_try, cost_try = self._run_scipy(p_seed)
                print(f"    multi-start cost={cost_try:.4e}")
                if cost_try < best_cost:
                    best_cost, best_params = cost_try, p_try

        print(f"  Best cost: {best_cost:.6e}")

        # Set model to best solution
        self._make_process_and_model(best_params)
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

        _, warnings = self._validate_params(params)
        for w in warnings:
            print(f"  ⚠️  {w}")

        self._build_local_vol_surface()
        self._extract_results()
        return self.heston_model

    def get_calibrated_results(self):
        return self.calibration_results

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_vol_data(data):
        """Return (strikes, expiries, vols_decimal) as numpy arrays."""
        if isinstance(data, pd.DataFrame):
            S = data['strike'].values.astype(float)
            E = data['expiry'].values.astype(float)
            V = data['volatility'].values.astype(float)
        else:
            arr = np.array(data, dtype=float)
            S, E, V = arr[:, 0], arr[:, 1], arr[:, 2]
        if V.max() > 2.0:
            print(f"  WARNING: vols look like % (max={V.max():.1f}), dividing by 100")
            V = V / 100.0
        return S, E, V

    def _clip(self, p):
        return np.clip(p, self._LO, self._HI)

    def _make_process_and_model(self, params):
        """(Re)build HestonProcess + HestonModel + engine and attach to all helpers."""
        p = self._clip(params)
        v0, kappa, theta, sigma, rho = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        process = ql.HestonProcess(
            self.domestic_curve, self.foreign_curve,
            self.spot_handle, v0, kappa, theta, sigma, rho
        )
        self.heston_model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(self.heston_model)
        for h in self.calibrated_helpers:
            h.setPricingEngine(engine)
        return self.heston_model

    def _heston_iv(self, params):
        """
        Compute Heston implied vols for all calibration instruments.
        Returns array of IV in DECIMAL form, or NaN for failed inversions.
        """
        p = self._clip(params)
        v0, kappa, theta, sigma, rho = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])

        process = ql.HestonProcess(
            self.domestic_curve, self.foreign_curve,
            self.spot_handle, v0, kappa, theta, sigma, rho
        )
        model  = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)

        calendar    = ql.TARGET()
        day_counter = ql.Actual365Fixed()

        ivs = np.full(len(self._cal_strikes), np.nan)
        for i, (K, T, vol_mkt) in enumerate(
                zip(self._cal_strikes, self._cal_expiries, self._cal_vols)):
            try:
                expiry_date = self.eval_date + ql.Period(max(1, int(round(T * 365))), ql.Days)
                payoff      = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
                exercise    = ql.EuropeanExercise(expiry_date)
                option      = ql.VanillaOption(payoff, exercise)
                option.setPricingEngine(engine)
                price = option.NPV()

                # Compute BS forward for moneyness check
                rd = self.domestic_curve.discount(T)
                rf = self.foreign_curve.discount(T)
                F  = self.spot_fx * rf / rd if rd > 0 else self.spot_fx

                # Only try IV inversion if price is meaningful
                intrinsic  = max(0.0, (F * rd - float(K) * rd))   # rough floor
                if price <= 0 or np.isnan(price) or np.isinf(price):
                    ivs[i] = vol_mkt   # neutral: no error contribution
                    continue

                # Black IV inversion
                rd_rate = -np.log(rd) / T if T > 0 else 0.04
                rf_rate = -np.log(rf) / T if T > 0 else 0.025

                bs_proc = ql.BlackScholesMertonProcess(
                    self.spot_handle,
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rf_rate, day_counter)
                    ),
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rd_rate, day_counter)
                    ),
                    ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(self.eval_date, calendar, vol_mkt, day_counter)
                    )
                )
                option2 = ql.VanillaOption(payoff, exercise)
                option2.setPricingEngine(ql.AnalyticEuropeanEngine(bs_proc))
                try:
                    iv = option2.impliedVolatility(
                        price, bs_proc,
                        accuracy=1e-6, maxEvaluations=1000,
                        minVol=0.001, maxVol=2.0
                    )
                    ivs[i] = iv
                except Exception:
                    ivs[i] = vol_mkt   # inversion failed: zero error contribution
            except Exception:
                ivs[i] = vol_mkt

        return ivs

    def _iv_residuals(self, params):
        """
        Residual vector in % vol units (O(0.1-1) for all expiries).
        r_i = (heston_iv_i - market_iv_i) * 100
        """
        ivs = self._heston_iv(params)
        return (ivs - self._cal_vols) * 100.0

    def _run_scipy(self, p0):
        """Run scipy.least_squares from p0 with hard bounds. Returns (params, cost)."""
        from scipy.optimize import least_squares
        sol = least_squares(
            self._iv_residuals,
            self._clip(p0),
            bounds=(self._LO, self._HI),
            method='trf',
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=5000,
            verbose=0,
        )
        print(f"  scipy: cost={sol.cost:.6e}  nfev={sol.nfev}  status={sol.status}")
        return self._clip(sol.x), sol.cost

    def _create_helpers(self):
        """Build HestonModelHelper list (used for result extraction)."""
        calendar = ql.TARGET()
        helpers  = []
        for K, T, vol in zip(self._cal_strikes, self._cal_expiries, self._cal_vols):
            if vol <= 0 or T <= 0 or K <= 0:
                continue
            period = ql.Period(max(1, int(round(T * 365))), ql.Days)
            h = ql.HestonModelHelper(
                period, calendar,
                float(self.spot_fx), float(K),
                ql.QuoteHandle(ql.SimpleQuote(float(vol))),
                self.domestic_curve, self.foreign_curve,
                ql.BlackCalibrationHelper.ImpliedVolError,
            )
            helpers.append(h)
        return helpers

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
            print(f"  Warning local vol surface: {e}")
            self.local_vol_surface = None
        return self.local_vol_surface

    def _validate_params(self, params):
        v0, kappa, theta, sigma, rho = params
        warnings = []
        if any(np.isnan(p) or np.isinf(p) for p in params):
            return False, ["NaN/Inf in params"]
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            return False, ["Non-positive parameter"]
        if abs(rho) >= 1.0:
            return False, ["|rho| >= 1"]
        if 2 * kappa * theta <= sigma**2:
            warnings.append(f"Feller violated: 2κθ={2*kappa*theta:.5f} <= σ²={sigma**2:.5f}")
        return True, warnings

    def _extract_results(self):
        if not self.heston_model:
            return None

        params = list(self.heston_model.params())

        # Recompute Heston IVs from final calibrated model
        heston_ivs = self._heston_iv(params)

        pricing_errors = []
        for i, (K, T, mkt_vol) in enumerate(
                zip(self._cal_strikes, self._cal_expiries, self._cal_vols)):
            model_vol     = float(heston_ivs[i]) if not np.isnan(heston_ivs[i]) else mkt_vol
            vol_error_bps = (model_vol - mkt_vol) * 10000

            # Also compute raw prices for the Market vs Model tab
            try:
                engine = ql.AnalyticHestonEngine(self.heston_model)
                expiry_date = self.eval_date + ql.Period(max(1, int(round(T * 365))), ql.Days)
                option = ql.VanillaOption(
                    ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
                    ql.EuropeanExercise(expiry_date)
                )
                option.setPricingEngine(engine)
                model_price = option.NPV()

                # BS price at market vol
                day_counter = ql.Actual365Fixed()
                rd = self.domestic_curve.discount(T)
                rf = self.foreign_curve.discount(T)
                rd_rate = -np.log(rd) / T if T > 0 and rd > 0 else 0.04
                rf_rate = -np.log(rf) / T if T > 0 and rf > 0 else 0.025
                bs_proc = ql.BlackScholesMertonProcess(
                    self.spot_handle,
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rf_rate, day_counter)),
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rd_rate, day_counter)),
                    ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(self.eval_date, ql.TARGET(), mkt_vol, day_counter))
                )
                option2 = ql.VanillaOption(
                    ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
                    ql.EuropeanExercise(expiry_date)
                )
                option2.setPricingEngine(ql.AnalyticEuropeanEngine(bs_proc))
                market_price    = option2.NPV()
                price_error     = model_price - market_price
                price_error_pct = (price_error / market_price * 100) if market_price > 1e-10 else 0.0
            except Exception:
                market_price = model_price = 0.0
                price_error  = price_error_pct = 0.0

            pricing_errors.append({
                'strike':          float(K),
                'expiry':          float(T),
                'market_vol':      mkt_vol   * 100,
                'model_vol':       model_vol * 100,
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
    # SIMULATION & VALIDATION
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
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), (total_steps, num_paths))

        for i in range(1, total_steps + 1):
            v = vol_paths[i-1, :]
            vol_paths[i, :] = np.maximum(
                v + kappa * (theta - v) * dt + sigma * np.sqrt(np.maximum(v, 0)) * dW2[i-1, :], 1e-4
            )
            spot_paths[i, :] = spot_paths[i-1, :] * np.exp(
                (rd - rf - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * dW1[i-1, :]
            )

        path_df = pd.DataFrame(spot_paths.T, columns=[f'time_{t:.4f}' for t in times])
        return path_df, times, spot_paths, vol_paths

    def validate_option_prices(self, test_options=None):
        if not self.heston_model:
            return None

        if test_options is None:
            if self._cal_strikes is not None:
                n = min(5, len(self._cal_strikes))
                idx = np.linspace(0, len(self._cal_strikes)-1, n, dtype=int)
                test_options = [[self._cal_strikes[i], self._cal_expiries[i], 'call'] for i in idx]
            else:
                return None

        calendar    = ql.TARGET()
        day_counter = ql.Actual365Fixed()
        results     = []

        for strike, expiry, opt_type in test_options:
            try:
                ql.Settings.instance().evaluationDate = self.eval_date
                expiry_date = self.eval_date + ql.Period(max(1, int(round(expiry * 365))), ql.Days)
                payoff  = ql.PlainVanillaPayoff(
                    ql.Option.Call if opt_type.lower() == 'call' else ql.Option.Put, float(strike)
                )
                exercise = ql.EuropeanExercise(expiry_date)

                option = ql.VanillaOption(payoff, exercise)
                option.setPricingEngine(ql.AnalyticHestonEngine(self.heston_model))
                heston_price = option.NPV()

                market_vol = 0.07
                if self._cal_strikes is not None:
                    mask = (np.abs(self._cal_strikes - strike) < 1e-5) & \
                           (np.abs(self._cal_expiries - expiry) < 1e-5)
                    if mask.any():
                        market_vol = float(self._cal_vols[mask][0])

                rd = self.domestic_curve.discount(expiry)
                rf = self.foreign_curve.discount(expiry)
                rd_rate = -np.log(rd) / expiry if expiry > 0 and rd > 0 else 0.04
                rf_rate = -np.log(rf) / expiry if expiry > 0 and rf > 0 else 0.025

                bs_proc = ql.BlackScholesMertonProcess(
                    self.spot_handle,
                    ql.YieldTermStructureHandle(ql.FlatForward(self.eval_date, rf_rate, day_counter)),
                    ql.YieldTermStructureHandle(ql.FlatForward(self.eval_date, rd_rate, day_counter)),
                    ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(self.eval_date, calendar, market_vol, day_counter)
                    )
                )
                option2 = ql.VanillaOption(payoff, exercise)
                option2.setPricingEngine(ql.AnalyticEuropeanEngine(bs_proc))
                bs_price = option2.NPV()

                diff = heston_price - bs_price
                results.append({
                    'strike': strike, 'expiry': expiry, 'type': opt_type,
                    'market_vol':     market_vol * 100,
                    'bs_price':       bs_price,
                    'heston_price':   heston_price,
                    'price_diff':     diff,
                    'price_diff_pct': (diff / bs_price * 100) if bs_price > 1e-10 else 0.0,
                })
            except Exception as e:
                print(f"  Validation skip K={strike} T={expiry}: {e}")

        return pd.DataFrame(results) if results else None
