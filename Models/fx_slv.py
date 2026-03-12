# FX Stochastic Local Volatility Model
import QuantLib as ql
import pandas as pd
import numpy as np
import sys
import os
import traceback as _tb

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class FXStochasticLocalVol:
    """
    FX Stochastic Local Volatility Model.

    No silent fallbacks policy
    --------------------------
    Every failure is logged to stdout with a full description.
    NaN values propagate to the output DataFrame so failures are visible.
    No variable is ever silently set to a placeholder (0.0, 0.07, etc.).
    """

    _LO    = np.array([1e-4,  0.10, 1e-4, 0.05, -0.95])
    _HI    = np.array([0.0625, 15.0, 0.0625, 1.50,  0.95])
    _LAMBDA = 1e-3
    _PRIOR  = np.array([0.0042, 1.5, 0.0056, 0.30, -0.30])

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

        self._cal_strikes  = None
        self._cal_expiries = None
        self._cal_vols     = None
        self._final_ivs    = None   # IVs at convergence, set by calibrate()
        self._best_params  = None   # clipped best params, set by calibrate()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def build_vol_surface(self):
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
                S = np.append(S, s); E = np.append(E, extra); V = np.append(V, V[0])
        if len(strike_set) < 2:
            extra = strike_set[0] * 1.01
            strike_set = sorted(strike_set + [extra])
            for e in expiry_set:
                S = np.append(S, extra); E = np.append(E, e); V = np.append(V, V[0])

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
                    if emsk.any():
                        vol_matrix[i][j] = float(V[emsk][si])
                    else:
                        raise RuntimeError(
                            f"Cannot fill vol matrix at strike={strike}, expiry={expiry}: "
                            "no nearby point found. Check input vol data."
                        )

        self.black_var_surface = ql.BlackVarianceSurface(
            self.eval_date, calendar, expiry_dates, strike_set, vol_matrix, day_counter
        )
        self.black_var_surface.enableExtrapolation()
        return self.black_var_surface

    def calibrate(self):
        ql.Settings.instance().evaluationDate = self.eval_date

        if self.black_var_surface is None:
            self.build_vol_surface()

        S, E, V = self._parse_vol_data(self.vol_surface_data)
        self._cal_strikes  = S
        self._cal_expiries = E
        self._cal_vols     = V

        self._make_process_and_model(self._clip(np.array([
            self.model_params['v0'],    self.model_params['kappa'],
            self.model_params['theta'], self.model_params['sigma'],
            self.model_params['rho'],
        ])))
        self.calibrated_helpers = self._create_helpers()
        print(f"  Helpers: {len(self.calibrated_helpers)}")
        if len(self.calibrated_helpers) == 0:
            raise RuntimeError("No calibration helpers constructed — check vol data.")

        p0 = self._clip(np.array([
            self.model_params['v0'],    self.model_params['kappa'],
            self.model_params['theta'], self.model_params['sigma'],
            self.model_params['rho'],
        ]))
        print(f"\nInitial Heston params:")
        print(f"  v0={p0[0]:.6f} ({np.sqrt(p0[0])*100:.2f}% vol)  kappa={p0[1]:.4f}  "
              f"theta={p0[2]:.6f} ({np.sqrt(p0[2])*100:.2f}% vol)  "
              f"sigma={p0[3]:.4f}  rho={p0[4]:.4f}")
        print(f"  Tikhonov lambda={self._LAMBDA:.0e}")

        best_params, best_cost = self._run_scipy(p0)

        if best_cost > 1.0:
            seeds = [
                [0.0042, 2.0,  0.0056, 0.40, -0.50],
                [0.0056, 1.0,  0.0056, 0.20, -0.20],
                [0.0030, 3.0,  0.0042, 0.50, -0.40],
                [0.0056, 1.5,  0.0072, 0.35, -0.30],
            ]
            for seed in seeds:
                p_seed = self._clip(np.array(seed, dtype=float))
                p_try, cost_try = self._run_scipy(p_seed)
                print(f"    multi-start cost={cost_try:.4e}")
                if cost_try < best_cost:
                    best_cost, best_params = cost_try, p_try

        print(f"  Best cost: {best_cost:.6e}")

        # Always clamp through _clip before storing — guards against
        # floating-point boundary creep from the optimizer (e.g. sigma = -1e-8)
        best_params = self._clip(best_params)
        self._best_params = best_params          # <-- stored for simulation

        # Store final IVs at convergence — _extract_results uses these directly
        self._final_ivs = self._heston_iv(best_params)

        n_nan = int(np.isnan(self._final_ivs).sum())
        if n_nan > 0:
            print(f"  WARNING: {n_nan}/{len(self._final_ivs)} instruments have NaN IV "
                  f"after calibration — IV inversion failed for those strikes. "
                  f"Check Heston price range vs market strikes.")

        self._make_process_and_model(best_params)
        params = list(self.heston_model.params())

        print(f"\n=== Calibrated Heston Parameters ===")
        print(f"  v0    = {params[0]:.6f}  ({np.sqrt(abs(params[0]))*100:.2f}% vol)")
        print(f"  kappa = {params[1]:.6f}")
        print(f"  theta = {params[2]:.6f}  ({np.sqrt(abs(params[2]))*100:.2f}% vol)")
        print(f"  sigma = {params[3]:.6f}")
        print(f"  rho   = {params[4]:.6f}")
        feller = 2 * params[1] * params[2] - params[3] ** 2
        print(f"  Feller margin: 2kappa*theta - sigma^2 = {feller:.6f}")
        print("=" * 40)

        valid_mask = ~np.isnan(self._final_ivs)
        if valid_mask.any():
            vol_errors_bps = (self._final_ivs[valid_mask] - self._cal_vols[valid_mask]) * 10000
            rmse = np.sqrt(np.mean(vol_errors_bps ** 2))
            print(f"  Calibration RMSE (vol, {valid_mask.sum()} instruments): {rmse:.2f} bps")
        else:
            print("  ERROR: All IV inversions failed — RMSE cannot be computed.")

        _, warnings = self._validate_params(params)
        for w in warnings:
            print(f"  WARNING: {w}")

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
        if isinstance(data, pd.DataFrame):
            S = data['strike'].values.astype(float)
            E = data['expiry'].values.astype(float)
            V = data['volatility'].values.astype(float)
        else:
            arr = np.array(data, dtype=float)
            S, E, V = arr[:, 0], arr[:, 1], arr[:, 2]
        if V.max() > 2.0:
            print(f"  WARNING: vols appear to be in % (max={V.max():.1f}), dividing by 100")
            V = V / 100.0
        return S, E, V

    def _clip(self, p):
        return np.clip(p, self._LO, self._HI)

    def _rate_from_discount(self, discount, T, label):
        """Convert discount factor to continuously-compounded rate. Raises if inputs are invalid."""
        if T <= 0:
            raise ValueError(f"T must be > 0 for rate extraction ({label}), got T={T}")
        if discount <= 0 or np.isnan(discount) or np.isinf(discount):
            raise ValueError(f"Invalid discount factor {discount} for {label} at T={T}")
        return -np.log(discount) / T

    def _make_process_and_model(self, params):
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

        Returns array length N in DECIMAL form.
        NaN means IV inversion failed — every failure is logged with reason.
        No value is silently substituted.

        Note: impliedVolatility() is called with positional arguments only.
        The QuantLib Python binding does not accept keyword arguments for this method.
        Signature: impliedVolatility(price, process, accuracy, maxEvals, minVol, maxVol)
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
                if T <= 0:
                    print(f"  [IV] SKIP i={i} K={K:.5f}: T={T} <= 0")
                    continue
                if K <= 0:
                    print(f"  [IV] SKIP i={i} T={T:.4f}: K={K} <= 0")
                    continue

                expiry_date = self.eval_date + ql.Period(max(1, int(round(T * 365))), ql.Days)
                payoff      = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
                exercise    = ql.EuropeanExercise(expiry_date)
                option      = ql.VanillaOption(payoff, exercise)
                option.setPricingEngine(engine)
                price = option.NPV()

                if price is None or np.isnan(price) or np.isinf(price):
                    print(f"  [IV] FAIL i={i} K={K:.5f} T={T:.4f}: Heston price={price} (NaN/Inf)")
                    continue
                if price <= 0:
                    print(f"  [IV] FAIL i={i} K={K:.5f} T={T:.4f}: Heston price={price:.6f} <= 0")
                    continue

                rd_df   = self.domestic_curve.discount(T)
                rf_df   = self.foreign_curve.discount(T)
                rd_rate = self._rate_from_discount(rd_df, T, f"domestic i={i}")
                rf_rate = self._rate_from_discount(rf_df, T, f"foreign i={i}")

                bs_proc = ql.BlackScholesMertonProcess(
                    self.spot_handle,
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rf_rate, day_counter)),
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rd_rate, day_counter)),
                    ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(self.eval_date, calendar, vol_mkt, day_counter))
                )
                option2 = ql.VanillaOption(payoff, exercise)
                option2.setPricingEngine(ql.AnalyticEuropeanEngine(bs_proc))
                try:
                    # Positional args only — QuantLib Python binding rejects keyword args
                    iv = option2.impliedVolatility(price, bs_proc, 1e-6, 1000, 0.001, 2.0)
                    ivs[i] = iv
                except Exception as e_iv:
                    print(f"  [IV] FAIL i={i} K={K:.5f} T={T:.4f}: impliedVolatility error: {e_iv} "
                          f"(Heston price={price:.6f}, mkt_vol={vol_mkt*100:.2f}%)")
            except Exception as e_outer:
                print(f"  [IV] FAIL i={i} K={K:.5f} T={T:.4f}: outer error: {e_outer}")

        return ivs

    def _iv_residuals(self, params):
        """
        Augmented residual vector:
          r[:N]  = (heston_iv_i - market_iv_i) * 100   [% vol units]
          r[N:]  = sqrt(lambda) * (p - prior) / scale  [Tikhonov]
        NaN IVs -> 50 bps penalty (logged by _heston_iv above).
        """
        ivs     = self._heston_iv(params)
        vol_res = np.where(np.isnan(ivs), 50.0, (ivs - self._cal_vols) * 100.0)
        scale   = np.array([0.01, 2.0, 0.01, 0.30, 0.50])
        reg     = np.sqrt(self._LAMBDA) * (params - self._PRIOR) / scale
        return np.concatenate([vol_res, reg])

    def _run_scipy(self, p0):
        from scipy.optimize import least_squares
        sol = least_squares(
            self._iv_residuals,
            self._clip(p0),
            bounds=(self._LO, self._HI),
            method='trf',
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            max_nfev=5000,
            verbose=0,
        )
        print(f"  scipy: cost={sol.cost:.6e}  nfev={sol.nfev}  status={sol.status}")
        return self._clip(sol.x), sol.cost

    def _create_helpers(self):
        calendar = ql.TARGET()
        helpers  = []
        for K, T, vol in zip(self._cal_strikes, self._cal_expiries, self._cal_vols):
            if vol <= 0 or T <= 0 or K <= 0:
                print(f"  [helpers] SKIP K={K} T={T} vol={vol}: non-positive value")
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
            print("  [local vol] Skipped: black_var_surface not built.")
            return None
        try:
            bv_handle = ql.BlackVolTermStructureHandle(self.black_var_surface)
            self.local_vol_surface = ql.LocalVolSurface(
                bv_handle, self.domestic_curve, self.foreign_curve, self.spot_handle
            )
            self.local_vol_surface.enableExtrapolation()
        except Exception as e:
            print(f"  [local vol] FAILED: {e}")
            print(_tb.format_exc())
            self.local_vol_surface = None
        return self.local_vol_surface

    def _validate_params(self, params):
        v0, kappa, theta, sigma, rho = params
        warnings = []
        if any(np.isnan(p) or np.isinf(p) for p in params):
            return False, ["NaN/Inf in calibrated params"]
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            return False, ["Non-positive parameter after calibration"]
        if abs(rho) >= 1.0:
            return False, ["|rho| >= 1"]
        if 2 * kappa * theta <= sigma**2:
            warnings.append(
                f"Feller violated: 2kappa*theta={2*kappa*theta:.5f} <= sigma^2={sigma**2:.5f}")
        return True, warnings

    def _extract_results(self):
        if not self.heston_model:
            return None

        params = list(self.heston_model.params())

        # Use IVs stored at convergence — never re-run with silent NaN hiding
        heston_ivs = self._final_ivs if self._final_ivs is not None else self._heston_iv(params)

        day_counter = ql.Actual365Fixed()
        pricing_errors = []

        for i, (K, T, mkt_vol) in enumerate(
                zip(self._cal_strikes, self._cal_expiries, self._cal_vols)):

            iv = heston_ivs[i]
            if np.isnan(iv):
                # IV inversion already logged in _heston_iv — store NaN visibly
                model_vol_pct = float('nan')
                vol_error_bps = float('nan')
            else:
                model_vol_pct = float(iv) * 100
                vol_error_bps = (float(iv) - mkt_vol) * 10000

            model_price     = float('nan')
            market_price    = float('nan')
            price_error     = float('nan')
            price_error_pct = float('nan')

            try:
                engine = ql.AnalyticHestonEngine(self.heston_model)
                expiry_date = self.eval_date + ql.Period(max(1, int(round(T * 365))), ql.Days)
                option = ql.VanillaOption(
                    ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
                    ql.EuropeanExercise(expiry_date)
                )
                option.setPricingEngine(engine)
                model_price = option.NPV()
                if np.isnan(model_price) or np.isinf(model_price):
                    print(f"  [results] i={i} K={K:.5f} T={T:.4f}: model_price={model_price} (NaN/Inf)")
                    model_price = float('nan')
            except Exception as e:
                print(f"  [results] i={i} K={K:.5f} T={T:.4f}: model pricing failed: {e}")

            try:
                rd_rate = self._rate_from_discount(self.domestic_curve.discount(T), T, f"domestic i={i}")
                rf_rate = self._rate_from_discount(self.foreign_curve.discount(T),  T, f"foreign i={i}")
                bs_proc = ql.BlackScholesMertonProcess(
                    self.spot_handle,
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rf_rate, day_counter)),
                    ql.YieldTermStructureHandle(
                        ql.FlatForward(self.eval_date, rd_rate, day_counter)),
                    ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(self.eval_date, ql.TARGET(), mkt_vol, day_counter))
                )
                expiry_date = self.eval_date + ql.Period(max(1, int(round(T * 365))), ql.Days)
                option2 = ql.VanillaOption(
                    ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
                    ql.EuropeanExercise(expiry_date)
                )
                option2.setPricingEngine(ql.AnalyticEuropeanEngine(bs_proc))
                market_price = option2.NPV()
                if not np.isnan(model_price) and not np.isnan(market_price):
                    price_error = model_price - market_price
                    if market_price > 1e-10:
                        price_error_pct = price_error / market_price * 100
                    else:
                        print(f"  [results] i={i} K={K:.5f} T={T:.4f}: "
                              f"market_price={market_price:.8f} too small for pct error")
            except Exception as e:
                print(f"  [results] i={i} K={K:.5f} T={T:.4f}: market pricing failed: {e}")

            pricing_errors.append({
                'strike':          float(K),
                'expiry':          float(T),
                'market_vol':      mkt_vol * 100,
                'model_vol':       model_vol_pct,
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
            raise RuntimeError("Model not calibrated — call calibrate() first.")

        total_steps = int(time_steps * horizon_years)
        times = np.linspace(0, horizon_years, total_steps + 1)
        dt    = horizon_years / total_steps

        # ----------------------------------------------------------------
        # Always source params from _best_params (clipped at calibration)
        # rather than heston_model.params() which can return raw optimizer
        # values that have not been re-clamped after QuantLib stores them.
        # ----------------------------------------------------------------
        if self._best_params is not None:
            raw = self._best_params
        else:
            raw = np.array(list(self.heston_model.params()))

        # Re-clip as a final safety net (e.g. sigma = -2.8e-2 -> 0.05)
        p     = self._clip(raw)
        v0    = float(p[0])
        kappa = float(p[1])
        theta = float(p[2])
        sigma = float(p[3])
        rho   = float(p[4])

        # Log if clipping actually changed anything
        if not np.allclose(raw, p, atol=1e-10):
            print(f"  [sim] params clamped before simulation:")
            print(f"    raw:     v0={float(raw[0]):.8f} kappa={float(raw[1]):.6f} "
                  f"theta={float(raw[2]):.8f} sigma={float(raw[3]):.8f} rho={float(raw[4]):.6f}")
            print(f"    clamped: v0={v0:.8f} kappa={kappa:.6f} "
                  f"theta={theta:.8f} sigma={sigma:.8f} rho={rho:.6f}")

        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
            raise ValueError(
                f"Invalid Heston params for simulation after clamping: "
                f"v0={v0} kappa={kappa} theta={theta} sigma={sigma}. "
                f"Try recalibrating with tighter bounds or a different initial guess."
            )
        if abs(rho) >= 1.0:
            raise ValueError(f"Invalid rho={rho} for simulation")

        rd = self.domestic_curve.zeroRate(horizon_years / 2, ql.Continuous).rate()
        rf = self.foreign_curve.zeroRate(horizon_years / 2, ql.Continuous).rate()

        spot_paths = np.zeros((total_steps + 1, num_paths))
        vol_paths  = np.zeros((total_steps + 1, num_paths))
        spot_paths[0, :] = self.spot_fx
        vol_paths[0, :]  = v0

        np.random.seed(42)
        dW1 = np.random.normal(0, np.sqrt(dt), (total_steps, num_paths))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(
            0, np.sqrt(dt), (total_steps, num_paths))

        for i in range(1, total_steps + 1):
            v = vol_paths[i-1, :]
            vol_paths[i, :] = np.maximum(
                v + kappa * (theta - v) * dt + sigma * np.sqrt(np.maximum(v, 0)) * dW2[i-1, :],
                1e-4
            )
            spot_paths[i, :] = spot_paths[i-1, :] * np.exp(
                (rd - rf - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * dW1[i-1, :]
            )

        path_df = pd.DataFrame(spot_paths.T, columns=[f'time_{t:.4f}' for t in times])
        return path_df, times, spot_paths, vol_paths

    def validate_option_prices(self, test_options=None):
        if not self.heston_model:
            raise RuntimeError("Model not calibrated — call calibrate() first.")

        if test_options is None:
            if self._cal_strikes is not None:
                n   = min(5, len(self._cal_strikes))
                idx = np.linspace(0, len(self._cal_strikes) - 1, n, dtype=int)
                test_options = [[self._cal_strikes[j], self._cal_expiries[j], 'call'] for j in idx]
            else:
                raise RuntimeError("No test_options provided and no calibration data available.")

        calendar    = ql.TARGET()
        day_counter = ql.Actual365Fixed()
        results     = []

        for strike, expiry, opt_type in test_options:
            ql.Settings.instance().evaluationDate = self.eval_date
            expiry_date = self.eval_date + ql.Period(max(1, int(round(expiry * 365))), ql.Days)
            payoff  = ql.PlainVanillaPayoff(
                ql.Option.Call if opt_type.lower() == 'call' else ql.Option.Put, float(strike)
            )
            exercise = ql.EuropeanExercise(expiry_date)

            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(ql.AnalyticHestonEngine(self.heston_model))
            heston_price = option.NPV()

            if self._cal_strikes is None:
                raise RuntimeError("No calibration data to look up market vol for validation.")
            mask = ((np.abs(self._cal_strikes - strike) < 1e-5) &
                    (np.abs(self._cal_expiries - expiry) < 1e-5))
            if not mask.any():
                print(f"  [validate] SKIP K={strike} T={expiry}: no matching calibration instrument found")
                continue
            market_vol = float(self._cal_vols[mask][0])

            rd_rate = self._rate_from_discount(
                self.domestic_curve.discount(expiry), expiry, f"domestic K={strike}")
            rf_rate = self._rate_from_discount(
                self.foreign_curve.discount(expiry),  expiry, f"foreign K={strike}")

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
                'strike':         strike,
                'expiry':         expiry,
                'type':           opt_type,
                'market_vol':     market_vol * 100,
                'bs_price':       bs_price,
                'heston_price':   heston_price,
                'price_diff':     diff,
                'price_diff_pct': (diff / bs_price * 100) if bs_price > 1e-10 else float('nan'),
            })

        return pd.DataFrame(results) if results else None
