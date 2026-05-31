"""Single-factor mean-reverting GBM scenario engine, vectorised across paths.

Pipeline
--------
1. For each product, simulate ``n_paths`` correlated GBM paths over its
   underlyings using a Schwartz-style mean-reverting drift in log-prices:

       d log S_i = [(μ_i − ½σ_i²) + κ(θ_i − log S_i)] dt + σ_i dW_i

   with CAPM-style drift  μ_i = r_f + β_i (μ_m − r_f) and per-asset
   correlation imposed via Cholesky on the (asset × asset) correlation
   submatrix for that product.

2. Apply β-scaled discrete shocks on shock dates; both actual log-S and
   the OU target θ are repriced.

3. For each product, run the reverse-convertible payoff once *per path*,
   then aggregate (mean, median, 5/95-percentile, ES, std).

Common Random Numbers (CRN) policy
----------------------------------
All randomness comes from a :class:`NoiseSampler`.  The engine never
seeds RNGs itself — there is no scenario-dependent hashing of any kind.

* Sharing one ``NoiseSampler`` across scenario runs keeps the underlying
  Gaussian draws identical, so scenario-to-scenario differences reflect
  only the scenario parameter change (the foundation of clean
  sensitivity / what-if analysis).
* To request a fresh draw, the *caller* explicitly calls
  ``sampler.regenerate()`` (bumps seed deterministically) or
  ``sampler.regenerate(seed=...)`` (sets an explicit seed).  The engine
  itself is a pure consumer of the cached tensors.

Output schema (``run_path_scenario``)
-------------------------------------
Identical to ``FactorScenarioEngine.run_path_scenario`` so the two
stress views can share rendering helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.barrier import sample_knock_in
from src.numerics.linalg import safe_cholesky
from src.numerics.noise_sampler import NoiseSampler
from src.pricing.products.reverse_convertible import (
    ReverseConvertible,
    barrier_levels,
    vectorised_european_rc_summary,
)


_PERCENTILES = [5, 95]


class ScenarioEngine:

    def __init__(
        self,
        portfolio,
        beta_map,
        vol_map,
        risk_free_rates=None,
        scenarios=None,
        mean_reversion_kappa: float = 0.5,
        n_paths: int = 1,
        noise_sampler: NoiseSampler | None = None,
        fx_rates: dict | None = None,
        reference_currency: str | None = None,
    ):
        self.portfolio       = portfolio
        self.beta_map        = beta_map
        self.scenarios       = scenarios
        self.vol             = vol_map
        self.risk_free_rates = risk_free_rates or {}
        # Speed of mean reversion (per year). 0.0 → pure GBM.
        # 0.5 ≈ ~1.4y half-life. Unconditional log-spread σ/√(2κ).
        self.kappa     = float(mean_reversion_kappa)
        self.n_paths   = int(n_paths)
        self.noise_sampler = noise_sampler

        # Optional FX context — when both fx_rates and reference_currency are
        # supplied, run_path_scenario adds `pnl_samples_ref` and
        # `pf_scenario_ref` to its output (aggregated in ref ccy across all
        # native currencies).  fx_rates keys are (from_ccy, ref_ccy) tuples
        # — same convention as PortfolioAnalytics.fx_rates.
        self.fx_rates           = fx_rates
        self.reference_currency = reference_currency

    # ────────────────────────────────────────────────────────────── helpers

    def get_beta(self, isin):
        return self.beta_map.get(isin, 1.0)

    def get_vol(self, isin):
        return self.vol.get(isin, 0.15)

    def _ensure_sampler(self, n_days: int, all_isins: list[str]) -> NoiseSampler:
        """Return a noise sampler matching the requested dimensions; create
        one if missing or incompatible."""
        # ScenarioEngine doesn't use factors → empty factor universe in the sampler.
        if (
            self.noise_sampler is None
            or not self.noise_sampler.matches(self.n_paths, n_days, [], all_isins)
        ):
            self.noise_sampler = NoiseSampler(
                n_paths=self.n_paths, n_days=n_days,
                factor_codes=[], isins=all_isins,
            )
        return self.noise_sampler

    # ───────────────────────────────────────────────────── path simulation

    def build_shock_paths(self, row, scenario, sampler: NoiseSampler,
                          corr_matrix=None):
        """Simulate ``n_paths`` correlated GBM paths for one product.

        Returns
        -------
        price_paths : np.ndarray, shape (n_paths, n_days, n_assets)
        date_range  : pd.DatetimeIndex
        path_summary: dict
        """
        today              = pd.Timestamp.today().normalize()
        portfolio_maturity = pd.to_datetime(self.portfolio["maturity_date"]).max()
        maturity           = pd.Timestamp(row["maturity_date"])

        T_remaining = (maturity - today).days / 360
        T_total     = (portfolio_maturity - today).days / 360

        market_shock       = scenario.get("market_shock", 0)
        n_shocks           = int(scenario.get("n_shocks", 1))
        shock_in_days      = int(scenario.get("shock_in_days", 0))
        shock_spacing_days = int(scenario.get("shock_spacing_days", 0))
        pre_shock_drift    = float(scenario.get("pre_shock_drift_pa", 0.05))
        post_shock_drift   = float(scenario.get("post_shock_drift_pa", 0.05))
        # Optional third regime: after the recovery horizon elapses, drift
        # reverts to ``post_recovery_drift_pa`` (typically the initial
        # market state).  If ``recovery_horizon_years`` is None the engine
        # behaves as before — recovery drift continues until maturity.
        recovery_horizon_years = scenario.get("recovery_horizon_years", None)
        post_recovery_drift    = float(scenario.get("post_recovery_drift_pa",
                                                    pre_shock_drift))

        isins    = list(row["underlying_isins"])
        spots    = np.array([float(s) for s in row["current_spots"]])
        n_assets = len(isins)
        N        = self.n_paths

        vols  = np.array([self.get_vol(isin)  for isin in isins])
        betas = np.array([self.get_beta(isin) for isin in isins])
        r_f   = float(self.risk_free_rates.get(row["currency"], 0.0))

        if corr_matrix is None:
            corr_matrix = np.eye(n_assets)
        # safe_cholesky projects a non-PSD sample matrix to the nearest
        # correlation matrix (Higham) before factorising.
        L = safe_cholesky(corr_matrix)

        date_range = pd.bdate_range(start=today, end=portfolio_maturity)
        n_days     = len(date_range)

        shock_day_offsets = sorted(
            shock_in_days + i * shock_spacing_days
            for i in range(n_shocks)
            if (shock_in_days + i * shock_spacing_days) / 360 <= T_total
        )
        shock_dates = set()
        for d in shock_day_offsets:
            target = today + pd.Timedelta(days=d)
            nearest_idx = int(np.argmin(np.abs(date_range - target)))
            shock_dates.add(date_range[nearest_idx])

        T_first_shock = shock_day_offsets[0]  / 360 if shock_day_offsets else T_total
        T_last_shock  = shock_day_offsets[-1] / 360 if shock_day_offsets else 0.0
        T_post_shock  = max(T_total - T_last_shock, 0.0)

        # CRN noise: per-isin draws; correlate across the asset axis with L.
        Z_raw  = sampler.idio_noise_for(isins)         # (N, n_days, n_assets)
        Z_corr = Z_raw @ L.T                            # (N, n_days, n_assets)

        # Vectorised state across paths
        log_S     = np.broadcast_to(np.log(spots), (N, n_assets)).copy()
        log_theta = np.broadcast_to(np.log(spots), (N, n_assets)).copy()
        log_paths = np.zeros((N, n_days, n_assets))
        prev_date = today

        for t_idx, bday in enumerate(date_range):
            dt      = (bday - prev_date).days / 360 if t_idx > 0 else 0.0
            t_years = (bday - today).days / 360

            if t_years <= T_first_shock:
                mu_m = pre_shock_drift
            elif t_years > T_last_shock:
                # Recovery regime — but only for the recovery horizon.  Past
                # that, drift reverts to the post-recovery rate (normally
                # the initial market state) so the path doesn't keep
                # snapping upward at +57 %/y for years.
                if (
                    recovery_horizon_years is not None
                    and (t_years - T_last_shock) > float(recovery_horizon_years)
                ):
                    mu_m = post_recovery_drift
                else:
                    mu_m = post_shock_drift
            else:
                mu_m = pre_shock_drift   # discrete shocks layer on top during window

            drift = r_f + betas * (mu_m - r_f)             # (n_assets,)
            drift_corr = (drift - 0.5 * vols ** 2) * dt    # (n_assets,)

            if dt > 0:
                log_theta += drift_corr
                log_S += (
                    drift_corr
                    + self.kappa * (log_theta - log_S) * dt
                    + vols * np.sqrt(dt) * Z_corr[:, t_idx, :]
                )

            if bday in shock_dates:
                shock_factor = np.maximum(1 + market_shock / 100 * betas, 1e-8)
                log_shock    = np.log(shock_factor)
                log_S     += log_shock
                log_theta += log_shock

            log_paths[:, t_idx, :] = log_S
            prev_date = bday

        price_paths = np.exp(log_paths)

        path_summary = {
            "maturity_date":       row["maturity_date"],
            "T_remaining_years":   round(T_remaining, 3),
            "T_first_shock_years": round(T_first_shock, 3),
            "T_post_shock_years":  round(T_post_shock, 3),
            "effective_n_shocks":  len(shock_day_offsets),
            "market_shock_pct":    market_shock,
            "pre_shock_drift_pa":  pre_shock_drift,
            "post_shock_drift_pa": post_shock_drift,
            "correlation_used":    True,
        }

        return price_paths, date_range, path_summary

    # ─────────────────────────────────────────────────────── per-product run

    def _american_breach_mask(self, row, price_paths, date_range, isins, sampler):
        """Per-path continuous (American) knock-in mask for a barrier RC.

        The diffusion variance of each step is the one this engine assumes when
        it generates the paths — ``σ_i² Δt`` with ``σ_i`` the per-asset vol and
        ``Δt`` the business-day gap (the mean-reversion and shock terms are
        drift, not diffusion, and are excluded).  Knock-ins are sampled at the
        Brownian-bridge rate using uniforms vended by the sampler, so the run
        stays reproducible under Common Random Numbers.
        """
        vols     = np.array([self.get_vol(i) for i in isins])           # (n_assets,)
        barriers = barrier_levels(row["initial_levels"], row.get("barrier_pct"))
        # Interval year-fractions over the grid (n_days - 1,), ACT/360.
        dt_gap   = np.diff(date_range.values) / np.timedelta64(1, "D") / 360.0
        step_var = (vols[None, :] ** 2) * dt_gap[:, None]               # (n_steps, n_assets)
        uniforms = sampler.knock_in_uniform(row["product_id"], price_paths.shape[0])
        return sample_knock_in(price_paths, barriers, step_var, uniforms)

    def _run_product(self, row, sampler, scenario, corr_df=None):
        corr_matrix = self.get_corr_subset(row, corr_df)
        price_paths, date_range, path_summary = self.build_shock_paths(
            row, scenario, sampler, corr_matrix
        )
        N, n_days, n_assets = price_paths.shape

        isins    = list(row["underlying_isins"])
        spots    = np.array([float(s) for s in row["current_spots"]])
        maturity = pd.Timestamp(row["maturity_date"])

        mat_mask = np.asarray(date_range >= maturity)
        if mat_mask.any():
            t_idx_terminal = int(np.argmax(mat_mask))
        else:
            t_idx_terminal = n_days - 1

        final_prices = price_paths[:, t_idx_terminal, :]   # (N, n_assets)

        # Vectorised per-path payoff (replaces a Python for-loop that
        # called ReverseConvertible.summary() once per path).  Dispatch on
        # ``product_type`` — autocallables need the full path to evaluate
        # their observation dates, while plain BRCs only need the terminal.
        ptype = str(row.get("product_type", "")).upper()
        if ptype == "IC_BRC":
            from src.pricing.products.issuer_callable_reverse_convertible import (
                vectorised_issuer_callable_rc_summary,
            )
            # Issuer call solved by optimal exercise; American knock-in (if any)
            # is sampled continuously, European observed at maturity.
            rf = float(self.risk_free_rates.get(row["currency"], 0.0))
            mask = (
                self._american_breach_mask(row, price_paths, date_range, isins, sampler)
                if str(row.get("type_style", "european")).lower() == "american"
                else None
            )
            v = vectorised_issuer_callable_rc_summary(
                row, price_paths, date_range, rf, breach_mask=mask,
            )
        elif ptype == "AC_BRC":
            from src.pricing.products.autocallable_reverse_convertible import (
                vectorised_autocallable_rc_summary,
            )
            # American autocallable: paths that run to maturity observe the
            # barrier continuously; called paths redeem at par regardless.
            mask = (
                self._american_breach_mask(row, price_paths, date_range, isins, sampler)
                if str(row.get("type_style", "european")).lower() == "american"
                else None
            )
            v = vectorised_autocallable_rc_summary(
                row, price_paths, date_range, uncalled_breach_mask=mask,
            )
        elif ptype == "CPN":
            from src.pricing.products.capital_protection_note import vectorised_cpn_summary
            v = vectorised_cpn_summary(row, final_prices)
        elif str(row.get("type_style", "european")).lower() == "american":
            # Continuously-monitored barrier: knock-in is sampled over the whole
            # path with the model's own per-step diffusion variance (σ_i² Δt),
            # at the bridge-correct rate, then settled at maturity on the worst-of.
            breach_mask = self._american_breach_mask(
                row, price_paths, date_range, isins, sampler,
            )
            v = vectorised_european_rc_summary(row, final_prices, breach_mask=breach_mask)
        else:
            v = vectorised_european_rc_summary(row, final_prices)
        pnl              = v["pnl"]
        return_pct       = v["return_pct"]
        cash_redemption  = v["cash_redemption"]
        worst_underlying = v["worst_underlying"]
        settlement_type  = v["settlement_type"]
        delivered_under  = v["delivered_underlying"]
        delivered_shares = v["delivered_shares"]
        fractional_cash  = v["fractional_cash"]
        barrier_breached = v["barrier_breached"]
        strike_used      = v["strike_used"]
        final_spot_used  = v["final_spot_used"]
        total_cost       = v["total_cost"]

        T_remaining = (maturity - date_range[0]).days / 360

        return {
            "product_id":         row["product_id"],
            "product_type":       row["product_type"],
            "type_style":         row.get("type_style", "European"),
            "currency":           row["currency"],
            "position_units":     row["position_units"],
            "notional":           row["notional"],
            "maturity_date":      row["maturity_date"],
            "T_remaining_years":  round(T_remaining, 2),
            "total_cost":         total_cost,

            "pnl_mean":   float(pnl.mean()),
            "pnl_median": float(np.median(pnl)),
            "pnl_p5":     float(np.percentile(pnl, _PERCENTILES[0])),
            "pnl_p95":    float(np.percentile(pnl, _PERCENTILES[1])),
            "pnl_es5":    float(pnl[pnl <= np.percentile(pnl, 5)].mean())
                            if N >= 20 else float(pnl.min()),
            "pnl_std":    float(pnl.std(ddof=1)) if N > 1 else 0.0,
            "return_mean_pct":   float(return_pct.mean()  * 100),
            "return_median_pct": float(np.median(return_pct) * 100),
            "return_p5_pct":     float(np.percentile(return_pct, 5)  * 100),
            "return_p95_pct":    float(np.percentile(return_pct, 95) * 100),

            "worst_underlying":    _mode_string(worst_underlying),
            "settlement_type":     _mode_string(settlement_type),
            "barrier_breach_freq": float(barrier_breached.mean()),

            "mean_cash_redemption":  float(cash_redemption.mean()),
            "mean_delivered_shares": float(delivered_shares.mean()),
            "mean_fractional_cash":  float(fractional_cash.mean()),
            "mean_final_spot":       float(final_spot_used.mean()),
            "mean_strike":           float(strike_used.mean()),
            "delivered_underlying":  _mode_string(delivered_under),
            # Physical-settlement delivery date — equal to maturity for any
            # path that settles physically.  None when no path is physical
            # (purely cash-settled product on this scenario).
            "delivery_date":         (str(row["maturity_date"])
                                       if (delivered_shares > 0).any() else None),

            "pnl_samples":    pnl,
            "return_samples": return_pct,

            "_isins":       isins,
            "_price_paths": price_paths,
            "path_summary": path_summary,
        }

    # ───────────────────────────────────────────────────────── portfolio run

    def run_path_scenario(self, scenario, corr_df=None):
        all_isins = sorted({
            isin for _, r in self.portfolio.iterrows() for isin in r["underlying_isins"]
        })
        today              = pd.Timestamp.today().normalize()
        portfolio_maturity = pd.to_datetime(self.portfolio["maturity_date"]).max()
        n_days             = len(pd.bdate_range(start=today, end=portfolio_maturity))
        sampler = self._ensure_sampler(n_days=n_days, all_isins=all_isins)

        product_results = []
        asset_paths     = {}
        for _, row in self.portfolio.iterrows():
            res = self._run_product(row, sampler, scenario, corr_df)
            isins       = res.pop("_isins")
            price_paths = res.pop("_price_paths")
            for i, isin in enumerate(isins):
                if isin not in asset_paths:
                    asset_paths[isin] = _path_summary_df(date_range_for_n_days(today, n_days),
                                                         price_paths[:, :, i])
            product_results.append(res)

        product_df = pd.DataFrame(product_results)

        # ── Currency aggregation ────────────────────────────────────────
        pnl_samples_by_ccy: dict[str, np.ndarray] = {}
        cost_by_ccy: dict[str, float] = {}
        for ccy, sub in product_df.groupby("currency"):
            per_path = np.stack(list(sub["pnl_samples"].values))   # (P, N)
            pnl_samples_by_ccy[ccy] = per_path.sum(axis=0)
            cost_by_ccy[ccy] = float(sub["total_cost"].sum())

        pf_rows = []
        for ccy, samples in pnl_samples_by_ccy.items():
            cost = cost_by_ccy[ccy]
            pf_rows.append({
                "currency":     ccy,
                "n_products":   int((product_df["currency"] == ccy).sum()),
                "total_cost":   cost,
                "underlyings":  sorted(
                    str(x)
                    for x in product_df.loc[
                        product_df["currency"] == ccy, "worst_underlying"
                    ].dropna().unique()
                ),
                "pnl_mean":   float(samples.mean()),
                "pnl_median": float(np.median(samples)),
                "pnl_p5":     float(np.percentile(samples, 5)),
                "pnl_p95":    float(np.percentile(samples, 95)),
                "pnl_es5":    float(samples[samples <= np.percentile(samples, 5)].mean())
                              if len(samples) >= 20 else float(samples.min()),
                "pnl_std":    float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
                "portfolio_return_mean_pct":   float(samples.mean()  / cost * 100) if cost > 0 else 0.0,
                "portfolio_return_median_pct": float(np.median(samples) / cost * 100) if cost > 0 else 0.0,
                "portfolio_return_p5_pct":     float(np.percentile(samples, 5)  / cost * 100) if cost > 0 else 0.0,
                "portfolio_return_p95_pct":    float(np.percentile(samples, 95) / cost * 100) if cost > 0 else 0.0,
            })
        pf_scenario_per_ccy = pd.DataFrame(pf_rows)

        cash_positions = (
            product_df.groupby("currency", as_index=False)
            .agg(total_cash=("mean_cash_redemption", "sum"))
        )

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
            delivered_stocks["market_value"]          = delivered_stocks["total_shares"] * delivered_stocks["price"]
            delivered_stocks["cost"]                  = delivered_stocks["total_shares"] * delivered_stocks["strike"]
            delivered_stocks["total_value_incl_cash"] = delivered_stocks["market_value"] + delivered_stocks["total_fractional_cash"]
            delivered_stocks["pnl"]                   = delivered_stocks["total_value_incl_cash"] - delivered_stocks["cost"]
            delivered_stocks["return_pct"]            = delivered_stocks["pnl"] / delivered_stocks["cost"]
            delivered_stocks = delivered_stocks[[
                "delivered_underlying", "total_shares", "strike", "price", "currency",
                "market_value", "total_fractional_cash", "total_value_incl_cash",
                "cost", "pnl", "return_pct", "final_delivery_date",
            ]]
        else:
            delivered_stocks = pd.DataFrame()

        # ── Reference-currency aggregation (item 4) ─────────────────────
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
            "pnl_samples_by_ccy":  pnl_samples_by_ccy,
            "pnl_samples_ref":     pnl_samples_ref,
            "reference_currency":  self.reference_currency,
            "n_paths":             self.n_paths,
        }

    # ──────────────────────────────────────────────────────── correlation

    def get_corr_subset(self, row, corr_df):
        """Extract product-specific correlation submatrix in the order of
        ``row["underlying_isins"]``.

        Returns identity (``np.eye``) when:

        * ``corr_df`` is ``None`` — no correlation data available, or
        * **any** of the product's ISINs is missing from ``corr_df`` — a
          manually-added product whose underlying isn't in the market
          data DB.  We fall back to identity (treats underlyings as
          uncorrelated) instead of raising; partial analytics is more
          useful than a stack trace, and the pre-flight in the
          Streamlit app surfaces which products are affected so the user
          knows their stress-test correlations are simplified.
        """
        isins = list(row["underlying_isins"])

        if corr_df is None:
            return np.eye(len(isins))

        missing = [
            isin for isin in isins
            if isin not in corr_df.index or isin not in corr_df.columns
        ]
        if missing:
            return np.eye(len(isins))

        return corr_df.loc[isins, isins].to_numpy(dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# Helpers (module-private)
# ──────────────────────────────────────────────────────────────────────────

def _path_summary_df(date_range, paths_2d: np.ndarray) -> pd.DataFrame:
    """Aggregate a (n_paths, n_days) tensor into per-date summary stats.

    Includes both percentile (p5/p95) and ±1σ band columns so plot
    helpers can render either confidence-band style.  ``ddof=0`` is used
    for the std so a single path collapses to a degenerate band on the
    median (matching the existing single-path collapse for percentiles).
    """
    median = np.median(paths_2d, axis=0)
    std = paths_2d.std(axis=0, ddof=0)
    return pd.DataFrame({
        "date":      date_range,
        "mean":      paths_2d.mean(axis=0),
        "median":    median,
        "p5":        np.percentile(paths_2d, 5,  axis=0),
        "p95":       np.percentile(paths_2d, 95, axis=0),
        "std":       std,
        "lower_1sd": median - std,
        "upper_1sd": median + std,
    })


def _mode_string(arr: np.ndarray) -> str | None:
    vals, counts = np.unique(
        np.array([str(x) for x in arr if x is not None]),
        return_counts=True,
    )
    if len(vals) == 0:
        return None
    return str(vals[counts.argmax()])


def date_range_for_n_days(today: pd.Timestamp, n_days: int) -> pd.DatetimeIndex:
    """Recover the business-day grid given today and a length — stable inverse."""
    # Using bdate_range with periods is exactly stable.
    return pd.bdate_range(start=today, periods=n_days)


def _aggregate_to_reference(pnl_samples_by_ccy, cost_by_ccy,
                             fx_rates, reference_currency):
    """Aggregate per-currency P&L samples into a single reference-currency
    distribution.

    Returns
    -------
    pnl_samples_ref : np.ndarray | None
        Shape ``(n_paths,)`` — total portfolio P&L per path in ref ccy.
        ``None`` when ``fx_rates`` or ``reference_currency`` is missing.
    pf_scenario_ref : pd.DataFrame
        One-row summary in reference currency (mean / median / p5 / p95 /
        es5 / std + total_cost_ref + portfolio return percentages).
        Empty DataFrame when no FX context.

    FX convention matches ``PortfolioAnalytics.fx_rates``:
    ``fx_rates[(ccy, ref)] = (ccy → ref)`` multiplier.
    """
    if not pnl_samples_by_ccy or fx_rates is None or reference_currency is None:
        return None, pd.DataFrame()

    def _rate(ccy):
        if ccy == reference_currency:
            return 1.0
        rate = fx_rates.get((ccy, reference_currency))
        if rate is None:
            raise ValueError(
                f"Missing FX rate for {ccy} → {reference_currency} in scenario aggregation."
            )
        return float(rate)

    # Sum native-ccy per-path P&L vectors after FX conversion.
    n_paths = next(iter(pnl_samples_by_ccy.values())).shape[0]
    total_ref = np.zeros(n_paths)
    cost_ref = 0.0
    for ccy, samples in pnl_samples_by_ccy.items():
        r = _rate(ccy)
        total_ref += samples * r
        cost_ref  += cost_by_ccy.get(ccy, 0.0) * r

    p5     = float(np.percentile(total_ref, 5))
    p95    = float(np.percentile(total_ref, 95))
    es5    = (float(total_ref[total_ref <= p5].mean())
              if len(total_ref) >= 20 else float(total_ref.min()))
    pf_row = {
        "reference_currency": reference_currency,
        "n_currencies":       len(pnl_samples_by_ccy),
        "total_cost_ref":     cost_ref,
        "pnl_mean":   float(total_ref.mean()),
        "pnl_median": float(np.median(total_ref)),
        "pnl_p5":     p5,
        "pnl_p95":    p95,
        "pnl_es5":    es5,
        "pnl_std":    float(total_ref.std(ddof=1)) if len(total_ref) > 1 else 0.0,
        "portfolio_return_mean_pct":   (total_ref.mean()  / cost_ref * 100) if cost_ref else 0.0,
        "portfolio_return_median_pct": (np.median(total_ref) / cost_ref * 100) if cost_ref else 0.0,
        "portfolio_return_p5_pct":     (p5  / cost_ref * 100) if cost_ref else 0.0,
        "portfolio_return_p95_pct":    (p95 / cost_ref * 100) if cost_ref else 0.0,
    }
    return total_ref, pd.DataFrame([pf_row])
