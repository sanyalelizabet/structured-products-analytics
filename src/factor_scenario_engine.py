"""Multi-factor stress engine — vectorised across paths for Monte Carlo.

Common Random Numbers (CRN) policy
----------------------------------
All randomness comes from a :class:`NoiseSampler`.  The engine never
seeds RNGs itself — there is no scenario-dependent hashing of any kind.
Sharing one sampler across scenario runs keeps draws identical so
scenario-to-scenario differences reflect only the scenario parameter
change.  To request a fresh draw, the caller explicitly calls
``sampler.regenerate()`` or ``sampler.regenerate(seed=...)``.

Pipeline
--------
1. Simulate ``n_paths`` factor paths via correlated mean-reverting GBM:

       d log F_k = [(μ_k − ½σ_k²) + κ(θ_k − log F_k)] dt + σ_k dW_k

   Cholesky-correlated innovations come from a :class:`NoiseSampler` so
   that scenario-to-scenario differences reflect the scenario change,
   not the seed (Common Random Numbers).

2. Apply factor-level shocks on shock dates (the deterministic target θ
   is repriced alongside the actual log-F).

3. Project to assets daily:

       r_{i,t} = α_i + Σ_k β_{i,k} r_{F_k,t} + λ ε_{i,t}

4. Build asset price paths by cumulating projected log-returns from the
   product's current spot.

5. For each product, run the reverse-convertible payoff once *per path*,
   then aggregate (mean, median, 5/95-percentile, expected shortfall,
   std).

Output schema (``run_path_scenario``)
-------------------------------------
``{
    "product_df":          pd.DataFrame,    # one row per product, aggregated
    "pf_scenario_per_ccy": pd.DataFrame,    # currency-level aggregated
    "cash_positions":      pd.DataFrame,    # mean cash redemption per ccy
    "delivered_stocks":    pd.DataFrame,    # mean delivered shares per (ccy, name)
    "asset_paths":         dict[isin → DataFrame[date, median, p5, p95, mean]],
    "factor_paths":        dict[code → DataFrame[date, median, p5, p95, mean]],
    "pnl_samples_by_ccy":  dict[ccy → np.ndarray of shape (n_paths,)],
    "n_paths":             int,
}``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor_engine import FACTORS
from src.noise_sampler import NoiseSampler
from src.reverse_convertible import ReverseConvertible


_PERCENTILES = [5, 95]


class FactorScenarioEngine:
    """Multi-factor mean-reverting GBM scenario engine, vectorised across paths."""

    def __init__(
        self,
        portfolio: pd.DataFrame,
        loadings: dict,
        factor_engine,
        risk_free_rates: dict | None = None,
        idio_intensity: float = 0.3,
        mean_reversion_kappa: float = 0.5,
        n_paths: int = 100,
        noise_sampler: NoiseSampler | None = None,
        years_for_factor_stats: int = 3,
        fx_rates: dict | None = None,
        reference_currency: str | None = None,
    ):
        self.portfolio    = portfolio
        self.loadings     = loadings
        self.fe           = factor_engine
        self.risk_free_rates = risk_free_rates or {}
        self.idio_lambda  = float(idio_intensity)
        self.kappa        = float(mean_reversion_kappa)
        self.n_paths      = int(n_paths)
        self.fx_rates           = fx_rates
        self.reference_currency = reference_currency

        self.factor_codes = list(FACTORS.keys())

        # Empirical factor stats — annualised.
        self.factor_vol_ser = self.fe.factor_vol(years=years_for_factor_stats)
        self.factor_corr_df = self.fe.factor_corr(years=years_for_factor_stats)

        rho = self.factor_corr_df.loc[self.factor_codes, self.factor_codes].to_numpy()
        eigvals = np.linalg.eigvalsh(rho)
        if eigvals.min() < 1e-8:
            rho = rho + np.eye(len(self.factor_codes)) * (1e-8 - eigvals.min())
        self.L_factor = np.linalg.cholesky(rho)

        self.noise_sampler = noise_sampler   # may be None — built lazily

    # ────────────────────────────────────────────────────────────── helpers

    def get_loading(self, isin: str, factor_code: str) -> float:
        if isin not in self.loadings:
            return 1.0 if factor_code == "MKT" else 0.0
        return self.loadings[isin]["betas"].get(
            factor_code, 1.0 if factor_code == "MKT" else 0.0
        )

    def get_idio_vol(self, isin: str) -> float:
        if isin not in self.loadings:
            return 0.15
        return float(self.loadings[isin]["idio_vol"])

    def _ensure_sampler(self, n_days: int, all_isins: list[str]) -> NoiseSampler:
        """Return a noise sampler matching the requested dimensions; create
        if missing or incompatible."""
        if (
            self.noise_sampler is None
            or not self.noise_sampler.matches(
                self.n_paths, n_days, self.factor_codes, all_isins
            )
        ):
            self.noise_sampler = NoiseSampler(
                n_paths      = self.n_paths,
                n_days       = n_days,
                factor_codes = self.factor_codes,
                isins        = all_isins,
            )
        return self.noise_sampler

    # ───────────────────────────────────────────────────── factor simulation

    def _simulate_factor_paths(self, scenario: dict, sampler: NoiseSampler):
        """Run vectorised correlated mean-reverting GBM on the K-dim factor
        block across all paths simultaneously.

        Returns
        -------
        date_range     : pd.DatetimeIndex,                len = n_days
        log_F_paths    : np.ndarray, shape (n_paths, n_days, K)
        factor_returns : np.ndarray, shape (n_paths, n_days, K)
        """
        today              = pd.Timestamp.today().normalize()
        portfolio_maturity = pd.to_datetime(self.portfolio["maturity_date"]).max()

        n_shocks           = int(scenario.get("n_shocks", 0))
        shock_in_days      = int(scenario.get("shock_in_days", 0))
        shock_spacing_days = int(scenario.get("shock_spacing_days", 0))
        factor_shock       = scenario.get("factor_shock", {}) or {}
        drift_pre          = scenario.get("factor_drift_pre_pa", {}) or {}
        drift_post         = scenario.get("factor_drift_post_pa", {}) or {}

        date_range = pd.bdate_range(start=today, end=portfolio_maturity)
        n_days     = len(date_range)
        K          = len(self.factor_codes)
        N          = self.n_paths

        T_total = (portfolio_maturity - today).days / 360
        shock_offsets = sorted(
            shock_in_days + i * shock_spacing_days
            for i in range(n_shocks)
            if (shock_in_days + i * shock_spacing_days) / 360 <= T_total
        )
        shock_dates = set()
        for d in shock_offsets:
            target = today + pd.Timedelta(days=d)
            nearest_idx = int(np.argmin(np.abs(date_range - target)))
            shock_dates.add(date_range[nearest_idx])

        T_first_shock = (shock_offsets[0]  / 360) if shock_offsets else T_total
        T_last_shock  = (shock_offsets[-1] / 360) if shock_offsets else 0.0

        vols       = np.array([float(self.factor_vol_ser.loc[c]) for c in self.factor_codes])
        mu_pre     = np.array([float(drift_pre.get(c, 0.0))  for c in self.factor_codes])
        mu_post    = np.array([float(drift_post.get(c, 0.0)) for c in self.factor_codes])
        shock_vec  = np.array([float(factor_shock.get(c, 0.0)) / 100.0
                               for c in self.factor_codes])

        # Cholesky-correlate the cached factor noise: (N, n_days, K) @ (K, K).
        Z_raw  = sampler.factor_noise()
        Z_corr = Z_raw @ self.L_factor.T

        # Vectorised state across paths
        log_F     = np.zeros((N, K))
        log_theta = np.zeros((N, K))
        log_paths = np.zeros((N, n_days, K))
        prev_date = today

        for t_idx, bday in enumerate(date_range):
            dt      = (bday - prev_date).days / 360 if t_idx > 0 else 0.0
            t_years = (bday - today).days / 360

            if t_years <= T_first_shock:
                mu = mu_pre
            elif t_years > T_last_shock:
                mu = mu_post
            else:
                mu = mu_pre   # during shock window — discrete shocks layer on top

            if dt > 0:
                drift_corr = (mu - 0.5 * vols ** 2) * dt          # (K,)
                log_theta += drift_corr                           # broadcast (K,)
                # log_F shape (N, K); each broadcasts cleanly
                log_F += (
                    drift_corr
                    + self.kappa * (log_theta - log_F) * dt
                    + vols * np.sqrt(dt) * Z_corr[:, t_idx, :]
                )

            if bday in shock_dates:
                shock_factor = np.maximum(1.0 + shock_vec, 1e-8)  # (K,)
                log_shock    = np.log(shock_factor)
                log_F     += log_shock
                log_theta += log_shock

            log_paths[:, t_idx, :] = log_F
            prev_date = bday

        # Daily log-returns — vectorised diff along the day axis.
        factor_returns = np.zeros_like(log_paths)
        factor_returns[:, 1:, :] = np.diff(log_paths, axis=1)

        return date_range, log_paths, factor_returns

    # ───────────────────────────────────────────────────── asset projection

    def _project_assets(self, row, factor_returns, sampler: NoiseSampler):
        """Project factor returns to asset price paths for one product.

        Returns
        -------
        price_paths : np.ndarray, shape (n_paths, n_days, n_assets)
            Asset price paths for this product's underlyings.
        isins : list[str]
        spots : np.ndarray, shape (n_assets,)
        """
        isins    = list(row["underlying_isins"])
        spots    = np.array([float(s) for s in row["current_spots"]])
        n_assets = len(isins)
        K        = len(self.factor_codes)

        beta_matrix = np.zeros((n_assets, K))
        idio_vols   = np.zeros(n_assets)
        for i, isin in enumerate(isins):
            for k, code in enumerate(self.factor_codes):
                beta_matrix[i, k] = self.get_loading(isin, code)
            idio_vols[i] = self.get_idio_vol(isin)

        # Systematic returns: factor_returns @ β.T → (N, n_days, n_assets)
        systematic = factor_returns @ beta_matrix.T

        # Idio returns: cached noise × σ × √dt × λ → (N, n_days, n_assets)
        sqrt_dt   = 1.0 / np.sqrt(252)
        idio_raw  = sampler.idio_noise_for(isins)                  # (N, n_days, n_assets)
        idio_term = idio_raw * idio_vols * sqrt_dt * self.idio_lambda

        asset_returns = systematic + idio_term

        # Cumulative log-return → log-prices → prices.
        log_paths   = np.log(spots)[None, None, :] + np.cumsum(asset_returns, axis=1)
        price_paths = np.exp(log_paths)

        return price_paths, isins, spots

    # ─────────────────────────────────────────────────────── per-product run

    def _run_product(self, row, factor_returns, date_range, sampler):
        """Compute aggregated per-product P&L over all paths."""
        price_paths, isins, spots = self._project_assets(row, factor_returns, sampler)
        n_paths, n_days, n_assets = price_paths.shape

        maturity = pd.Timestamp(row["maturity_date"])

        # Index of the first business day at or after the product maturity.
        # All paths share the same business-day grid so this is a scalar.
        mat_mask = np.asarray(date_range >= maturity)
        if mat_mask.any():
            t_idx_terminal = int(np.argmax(mat_mask))
        else:
            t_idx_terminal = n_days - 1

        # Terminal prices per path: (n_paths, n_assets)
        final_prices = price_paths[:, t_idx_terminal, :]

        # Run RC payoff per path
        pnl              = np.zeros(n_paths)
        return_pct       = np.zeros(n_paths)
        total_payoff     = np.zeros(n_paths)
        cash_redemption  = np.zeros(n_paths)
        worst_underlying = np.empty(n_paths, dtype=object)
        settlement_type  = np.empty(n_paths, dtype=object)
        delivered_under  = np.empty(n_paths, dtype=object)
        delivered_shares = np.zeros(n_paths)
        fractional_cash  = np.zeros(n_paths)
        barrier_breached = np.zeros(n_paths, dtype=bool)
        strike_used      = np.zeros(n_paths)
        final_spot_used  = np.zeros(n_paths)

        for p in range(n_paths):
            shocks = [
                (final_prices[p, i] / spots[i] - 1) * 100
                for i in range(n_assets)
            ]
            rc = ReverseConvertible(row, final_levels=shocks)
            s  = rc.summary()

            idx       = row["underlyings"].index(s["worst_underlying"])
            strike    = row["strike"][idx]
            notional  = row["notional"]
            ts        = notional / strike
            price     = final_prices[p, idx]
            physical  = price < strike

            if physical:
                d_shares = int(ts)
                f_cash   = (ts - d_shares) * price
                cash_red = f_cash
                d_under  = s["worst_underlying"]
                stype    = "physical"
            else:
                d_shares = 0
                f_cash   = 0.0
                cash_red = s["total_payoff"]
                d_under  = None
                stype    = "cash"

            pnl[p]              = s["pnl"]
            return_pct[p]       = s["return_pct"]
            total_payoff[p]     = s["total_payoff"]
            cash_redemption[p]  = cash_red
            worst_underlying[p] = s["worst_underlying"]
            settlement_type[p]  = stype
            delivered_under[p]  = d_under
            delivered_shares[p] = d_shares
            fractional_cash[p]  = f_cash
            barrier_breached[p] = bool(s["barrier_breached"])
            strike_used[p]      = strike
            final_spot_used[p]  = price

        # Aggregations
        T_remaining = (maturity - date_range[0]).days / 360

        return {
            # identifiers
            "product_id":         row["product_id"],
            "product_type":       row["product_type"],
            "currency":           row["currency"],
            "position_units":     row["position_units"],
            "notional":           row["notional"],
            "maturity_date":      row["maturity_date"],
            "T_remaining_years":  round(T_remaining, 2),

            # deterministic
            "total_cost":         float(ReverseConvertible(row, final_levels=[0]*n_assets).summary()["total_cost"]),

            # aggregated P&L statistics
            "pnl_mean":   float(pnl.mean()),
            "pnl_median": float(np.median(pnl)),
            "pnl_p5":     float(np.percentile(pnl, _PERCENTILES[0])),
            "pnl_p95":    float(np.percentile(pnl, _PERCENTILES[1])),
            "pnl_es5":    float(pnl[pnl <= np.percentile(pnl, 5)].mean()) if n_paths >= 20 else float(pnl.min()),
            "pnl_std":    float(pnl.std(ddof=1)) if n_paths > 1 else 0.0,
            "return_mean_pct":   float(return_pct.mean()  * 100),
            "return_median_pct": float(np.median(return_pct) * 100),
            "return_p5_pct":     float(np.percentile(return_pct, 5)  * 100),
            "return_p95_pct":    float(np.percentile(return_pct, 95) * 100),

            # modal categoricals
            "worst_underlying":  _mode_string(worst_underlying),
            "settlement_type":   _mode_string(settlement_type),
            "barrier_breach_freq": float(barrier_breached.mean()),

            # mean settlement particulars (for currency aggregation)
            "mean_cash_redemption": float(cash_redemption.mean()),
            "mean_delivered_shares": float(delivered_shares.mean()),
            "mean_fractional_cash":  float(fractional_cash.mean()),
            "mean_final_spot":  float(final_spot_used.mean()),
            "mean_strike":      float(strike_used.mean()),
            "delivered_underlying": _mode_string(delivered_under),
            # Physical-settlement delivery date (item 5).  Equals maturity
            # when at least one path settles physically; None otherwise.
            "delivery_date":    (str(row["maturity_date"])
                                  if (delivered_shares > 0).any() else None),

            # raw samples (for distribution plots)
            "pnl_samples":    pnl,
            "return_samples": return_pct,

            # per-asset path tensor for downstream summarisation
            "_isins":         isins,
            "_price_paths":   price_paths,
        }

    # ─────────────────────────────────────────────────────── portfolio run

    def run_path_scenario(self, scenario: dict) -> dict:
        # Build / reuse the noise sampler for the union of asset universes.
        all_isins = sorted({
            isin for _, row in self.portfolio.iterrows() for isin in row["underlying_isins"]
        })
        # Need n_days first; derive from grid.
        today              = pd.Timestamp.today().normalize()
        portfolio_maturity = pd.to_datetime(self.portfolio["maturity_date"]).max()
        n_days             = len(pd.bdate_range(start=today, end=portfolio_maturity))
        sampler = self._ensure_sampler(n_days=n_days, all_isins=all_isins)

        # Allow per-scenario λ override
        if "idio_intensity" in scenario:
            saved_lambda = self.idio_lambda
            self.idio_lambda = float(scenario["idio_intensity"])
        else:
            saved_lambda = None

        try:
            date_range, log_F_paths, factor_returns = self._simulate_factor_paths(
                scenario, sampler
            )

            # Per-factor path summary (median, p5, p95, mean) base 100
            factor_paths = {}
            for k, code in enumerate(self.factor_codes):
                idx_paths = np.exp(log_F_paths[:, :, k]) * 100.0   # (N, n_days)
                factor_paths[code] = _path_summary_df(date_range, idx_paths)

            # Run each product
            product_results = []
            asset_paths     = {}
            for _, row in self.portfolio.iterrows():
                res = self._run_product(row, factor_returns, date_range, sampler)
                # Extract per-asset path summaries
                isins      = res.pop("_isins")
                price_paths = res.pop("_price_paths")
                for i, isin in enumerate(isins):
                    if isin not in asset_paths:
                        asset_paths[isin] = _path_summary_df(date_range, price_paths[:, :, i])
                product_results.append(res)

            product_df = pd.DataFrame(product_results)

            # ── Currency-level aggregation ───────────────────────────────
            # For each currency, sum P&L per path then aggregate across paths.
            pnl_samples_by_ccy: dict[str, np.ndarray] = {}
            cost_by_ccy: dict[str, float] = {}
            for ccy, sub in product_df.groupby("currency"):
                # Stack per-product samples into (n_products, n_paths) and sum per path.
                per_path = np.stack(list(sub["pnl_samples"].values))  # (P, N)
                per_path_sum = per_path.sum(axis=0)
                pnl_samples_by_ccy[ccy] = per_path_sum
                cost_by_ccy[ccy] = float(sub["total_cost"].sum())

            pf_rows = []
            for ccy, samples in pnl_samples_by_ccy.items():
                cost = cost_by_ccy[ccy]
                pf_rows.append({
                    "currency":           ccy,
                    "n_products":         int((product_df["currency"] == ccy).sum()),
                    "total_cost":         cost,
                    "underlyings":        sorted(set(
                        product_df.loc[product_df["currency"] == ccy, "worst_underlying"]
                    )),
                    "pnl_mean":   float(samples.mean()),
                    "pnl_median": float(np.median(samples)),
                    "pnl_p5":     float(np.percentile(samples, 5)),
                    "pnl_p95":    float(np.percentile(samples, 95)),
                    "pnl_es5":    float(samples[samples <= np.percentile(samples, 5)].mean())
                                    if len(samples) >= 20 else float(samples.min()),
                    "pnl_std":    float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
                    "portfolio_return_mean_pct":   float(samples.mean() / cost * 100) if cost > 0 else 0.0,
                    "portfolio_return_median_pct": float(np.median(samples) / cost * 100) if cost > 0 else 0.0,
                    "portfolio_return_p5_pct":     float(np.percentile(samples, 5) / cost * 100) if cost > 0 else 0.0,
                    "portfolio_return_p95_pct":    float(np.percentile(samples, 95) / cost * 100) if cost > 0 else 0.0,
                })
            pf_scenario_per_ccy = pd.DataFrame(pf_rows)

            # ── Cash positions (mean across paths) ───────────────────────
            cash_positions = (
                product_df.groupby("currency", as_index=False)
                .agg(total_cash=("mean_cash_redemption", "sum"))
            )

            # ── Delivered stocks (mean across paths) ─────────────────────
            delivered_df = product_df[
                (product_df["delivered_underlying"].notna())
                & (product_df["mean_delivered_shares"] > 0)
            ].copy()
            if not delivered_df.empty:
                delivered_stocks = (
                    delivered_df.groupby(["currency", "delivered_underlying"], as_index=False)
                    .agg(
                        total_shares=("mean_delivered_shares", "sum"),
                        total_fractional_cash=("mean_fractional_cash", "sum"),
                        strike=("mean_strike", "mean"),
                        price=("mean_final_spot", "mean"),
                        final_delivery_date=("delivery_date", "max"),
                    )
                )
                delivered_stocks["market_value"] = delivered_stocks["total_shares"] * delivered_stocks["price"]
                delivered_stocks["cost"]         = delivered_stocks["total_shares"] * delivered_stocks["strike"]
                delivered_stocks["total_value_incl_cash"] = delivered_stocks["market_value"] + delivered_stocks["total_fractional_cash"]
                delivered_stocks["pnl"]        = delivered_stocks["total_value_incl_cash"] - delivered_stocks["cost"]
                delivered_stocks["return_pct"] = delivered_stocks["pnl"] / delivered_stocks["cost"]
                delivered_stocks = delivered_stocks[[
                    "delivered_underlying", "total_shares", "strike", "price", "currency",
                    "market_value", "total_fractional_cash", "total_value_incl_cash",
                    "cost", "pnl", "return_pct", "final_delivery_date",
                ]]
            else:
                delivered_stocks = pd.DataFrame()

            # Reference-currency aggregation (item 4)
            from src.scenario_engine import _aggregate_to_reference
            pnl_samples_ref, pf_scenario_ref = _aggregate_to_reference(
                pnl_samples_by_ccy,
                cost_by_ccy,
                self.fx_rates,
                self.reference_currency,
            )

            return {
                "product_df":          product_df,
                "pf_scenario_per_ccy": pf_scenario_per_ccy,
                "pf_scenario_ref":     pf_scenario_ref,
                "cash_positions":      cash_positions,
                "delivered_stocks":    delivered_stocks,
                "asset_paths":         asset_paths,
                "factor_paths":        factor_paths,
                "pnl_samples_by_ccy":  pnl_samples_by_ccy,
                "pnl_samples_ref":     pnl_samples_ref,
                "reference_currency":  self.reference_currency,
                "n_paths":             self.n_paths,
            }
        finally:
            if saved_lambda is not None:
                self.idio_lambda = saved_lambda


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _path_summary_df(date_range, paths_2d: np.ndarray) -> pd.DataFrame:
    """Aggregate a (n_paths, n_days) tensor into a per-date summary DataFrame.

    Columns: ``date, mean, median, p5, p95``.  When n_paths == 1, all four
    statistics collapse to the single path so downstream code can treat
    single-path and multi-path results uniformly.
    """
    return pd.DataFrame({
        "date":   date_range,
        "mean":   paths_2d.mean(axis=0),
        "median": np.median(paths_2d, axis=0),
        "p5":     np.percentile(paths_2d, 5,  axis=0),
        "p95":    np.percentile(paths_2d, 95, axis=0),
    })


def _mode_string(arr: np.ndarray) -> str | None:
    """Return the most common entry in ``arr`` (ignoring None)."""
    vals, counts = np.unique(
        np.array([str(x) for x in arr if x is not None]),
        return_counts=True,
    )
    if len(vals) == 0:
        return None
    return str(vals[counts.argmax()])
