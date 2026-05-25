"""Monte Carlo pricing engine for structured products.

Risk-neutral GBM under ACT/360. Payoff functions are passed to the pricer as
strategies, so adding a product type means adding a payoff function, not
modifying the pricer.

    simulate_paths()  correlated GBM path tensor (n_paths, n_steps, n_assets)
    price()           simulate, apply payoff_fn, discount
    price_portfolio() dispatch payoff_fn per row, price each product

    payoff_fn(paths, dates, row) -> np.ndarray, shape (n_paths,)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Callable

from src.linalg import safe_cholesky
from src.reverse_convertible import barrier_levels
from src.autocallable_reverse_convertible import (
    vectorised_autocallable_rc_summary,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payoff functions
# ---------------------------------------------------------------------------

def capital_protection_note_payoff(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    row: pd.Series,
) -> np.ndarray:
    """Capital Protection Note payoff at maturity.

    Per certificate (and scaled by ``notional``):

        Payoff(S_T) = N · [π + p · max(S_T/K − 1, 0)]

    where π = ``protection_pct`` and p = ``participation_pct``.  Coupon
    is added unconditionally with the same year-fraction convention used
    by :func:`european_brc_payoff` so the two payoffs sit on the same
    discounting basis.
    """
    notional      = float(row["notional"])
    strike        = float(row["strike"][0])
    protection    = float(row["protection_pct"])
    participation = float(row["participation_pct"])

    terminal = paths[:, -1, 0]                                   # (n_paths,)
    upside   = np.maximum(terminal / strike - 1.0, 0.0)
    redemption = notional * (protection + participation * upside)

    T_total = (
        pd.Timestamp(row["maturity_date"]) -
        pd.Timestamp(row["initial_fixing_date"])
    ).days / 360
    coupon = notional * float(row.get("coupon", 0.0) or 0.0) * T_total

    return redemption + coupon


def european_brc_payoff(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    row: pd.Series,
) -> np.ndarray:
    """
    European Barrier Reverse Convertible payoff at maturity.

    Barrier observation is at the final fixing date only (European style).
    Coupon is unconditional — always added to every path.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, n_steps, n_assets)
        Simulated spot price paths.
    dates : pd.DatetimeIndex
        Business-day grid matching the n_steps axis.
    row : pd.Series
        Portfolio row — needs: strike, notional, coupon,
        initial_fixing_date, maturity_date, underlying_isins.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Total payoff per path (redemption + coupon) in product currency.
    """
    notional = float(row["notional"])
    strikes  = np.array([float(k) for k in row["strike"]])   # (n_assets,)
    barriers = barrier_levels(row["initial_levels"], row.get("barrier_pct"))  # (n_assets,)

    T_total = (
        pd.Timestamp(row["maturity_date"]) -
        pd.Timestamp(row["initial_fixing_date"])
    ).days / 360

    # Terminal spots — last step of the path
    terminal = paths[:, -1, :]                                # (n_paths, n_assets)

    # Performance relative to strike at maturity
    perfs      = terminal / strikes                           # (n_paths, n_assets)
    worst_perf = perfs.min(axis=1)                            # (n_paths,)

    # European barrier breach: any underlying at/below its barrier
    # (= initial_level × barrier_pct) at final fixing.
    breached   = (terminal <= barriers[None, :]).any(axis=1)  # (n_paths,)

    # Redemption: full notional if the barrier held, else worst-of conversion.
    redemptions = np.where(breached, notional * worst_perf, notional)

    # Fixed coupon paid unconditionally at maturity
    coupon = notional * float(row["coupon"]) * T_total

    return redemptions + coupon


# ---------------------------------------------------------------------------
# Fallback defaults — applied when an input is missing. Every use is recorded
# in the per-product ``fallbacks`` provenance so a defaulted figure is never
# silently indistinguishable from one computed on real data.
# ---------------------------------------------------------------------------

DEFAULT_VOL = 0.15               # annualised vol when an ISIN has no vol estimate
DEFAULT_RISK_FREE_RATE = 0.02    # rate when a currency has no rate


def _is_autocallable(row: pd.Series) -> bool:
    """True for autocallable BRCs, which require path-dependent pricing.

    Autocallables redeem early when the worst-of underlying clears the trigger
    on an observation date, so they cannot be priced with the European payoff
    and a single maturity discount; the pricer routes them to a per-path
    discounted valuation instead.
    """
    return str(row.get("product_type", "")).upper() == "AC_BRC"


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

class MonteCarloPricer:
    """Prices any product whose payoff is a function of correlated GBM paths.

    Parameters
    ----------
    n_paths : int
        Number of paths. ~10,000 suffices for European payoffs; use 50,000+
        for path-dependent features.
    seed : int
        RNG seed; fixed for reproducibility and common random numbers.
    """

    def __init__(self, n_paths: int = 10_000, seed: int = 42):
        self.n_paths = n_paths
        self.seed    = seed

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------
    def simulate_paths(
        self,
        row: pd.Series,
        vol_map: dict,
        risk_free_rate: float,
        corr_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Simulate correlated GBM from today to product maturity.

        Uses a business-day grid so path-dependent payoffs (autocalls,
        American barriers) can check exact observation dates by index.

        Parameters
        ----------
        row : pd.Series
            Portfolio row — needs: underlying_isins, current_spots, maturity_date.
        vol_map : dict
            { isin: annualised_vol }
        risk_free_rate : float
            Annualised continuously-compounded risk-free rate.
        corr_matrix : np.ndarray (n_assets, n_assets), optional
            Correlation between underlyings.  Defaults to identity.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps, n_assets)
        dates : pd.DatetimeIndex   — business-day grid, length n_steps
        """
        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])

        isins    = list(row["underlying_isins"])
        spots    = np.array([float(s) for s in row["current_spots"]])
        n_assets = len(isins)

        vols = np.array([vol_map.get(isin, DEFAULT_VOL) for isin in isins])
        r    = float(risk_free_rate)

        # Business-day grid
        dates   = pd.bdate_range(start=today, end=maturity)
        n_steps = len(dates)

        # Correlation. safe_cholesky never raises: a non-PSD matrix (e.g. from
        # pairwise estimation) is projected to the nearest correlation matrix
        # before factorisation.
        if corr_matrix is None or n_assets == 1:
            corr_matrix = np.eye(n_assets)
        L = safe_cholesky(corr_matrix)

        # Draw all random increments at once: (n_paths, n_steps, n_assets)
        rng   = np.random.default_rng(self.seed)
        Z_raw = rng.standard_normal((self.n_paths, n_steps, n_assets))
        Z     = Z_raw @ L.T                                    # correlated

        # Step sizes in years — prepend today as datetime64 to match dates dtype.
        # ACT/360 throughout: the simulation clock must match the discounting and
        # accrual basis (also /360) so the risk-neutral drift and the discount
        # factor share one day-count convention.
        all_dates = np.concatenate([[today.to_datetime64()], dates.values])
        dt = np.diff(all_dates) / np.timedelta64(1, "D") / 360.0
        dt = np.maximum(dt, 0.0)

        # GBM log-increments: (n_steps, n_assets) broadcast with (n_paths, n_steps, n_assets)
        drift_step = (r - 0.5 * vols ** 2) * dt[:, np.newaxis]   # (n_steps, n_assets)
        diff_step  = vols * np.sqrt(dt[:, np.newaxis]) * Z        # (n_paths, n_steps, n_assets)

        log_increments = drift_step[np.newaxis, :, :] + diff_step  # (n_paths, n_steps, n_assets)

        # Cumulative product from initial spots
        log_paths      = np.cumsum(log_increments, axis=1)
        paths          = spots[np.newaxis, np.newaxis, :] * np.exp(log_paths)

        return paths, dates

    # ------------------------------------------------------------------
    # Single-product pricer
    # ------------------------------------------------------------------
    def price(
        self,
        row: pd.Series,
        payoff_fn: Callable,
        vol_map: dict,
        risk_free_rate: float,
        corr_matrix: np.ndarray | None = None,
    ) -> dict:
        """
        Price one product using a provided payoff function.

        Parameters
        ----------
        row : pd.Series
            Portfolio row.
        payoff_fn : callable
            payoff_fn(paths, dates, row) → np.ndarray (n_paths,)
            Returns the total cash payoff per path at maturity.
        vol_map : dict
            { isin: annualised_vol }
        risk_free_rate : float
            Annualised risk-free rate in the product's currency.
        corr_matrix : np.ndarray, optional

        Returns
        -------
        dict
            fair_value         : present value of expected payoff
            fair_value_pct     : fair_value / notional
            std_error          : Monte Carlo standard error
            fallbacks          : provenance tags for any defaulted inputs
        """
        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])

        T_remaining = max((maturity - today).days / 360, 0.0)
        notional    = float(row["notional"])

        # Input provenance: vol defaults, and identity-correlation fallback when
        # a multi-asset product is priced without a correlation matrix. Logged by
        # the portfolio-level callers (price() runs repeatedly inside greeks).
        corr_fell_back = corr_matrix is None and len(row["underlying_isins"]) > 1
        fallbacks = self._input_fallbacks(row, vol_map, corr_fell_back)

        # Already expired — return intrinsic value
        if T_remaining <= 0:
            spots    = np.array([float(s) for s in row["current_spots"]])
            strikes  = np.array([float(k) for k in row["strike"]])
            barriers = barrier_levels(row["initial_levels"], row.get("barrier_pct"))
            T_total  = (maturity - pd.Timestamp(row["initial_fixing_date"])).days / 360
            worst    = (spots / strikes).min()
            breached = bool((spots <= barriers).any())
            redemp   = notional * worst if breached else notional
            coupon   = notional * float(row["coupon"]) * T_total
            fv       = redemp + coupon
            return {
                "fair_value":     fv,
                "fair_value_pct": fv / notional if notional else np.nan,
                "std_error":      0.0,
                "fallbacks":      fallbacks,
            }

        paths, dates = self.simulate_paths(row, vol_map, risk_free_rate, corr_matrix)

        # Discounting is centralised in ``_fv_from_paths`` so that path-dependent
        # products (autocallables) discount each path to its own cashflow date
        # rather than uniformly to maturity.
        fair_value, std_error = self._fv_from_paths(
            paths, dates, payoff_fn, row, risk_free_rate, T_remaining,
        )

        return {
            "fair_value":     fair_value,
            "fair_value_pct": fair_value / notional if notional else np.nan,
            "std_error":      std_error,
            "fallbacks":      fallbacks,
        }

    # ------------------------------------------------------------------
    # Portfolio-level pricer
    # ------------------------------------------------------------------
    def price_portfolio(
        self,
        portfolio: pd.DataFrame,
        vol_map: dict,
        risk_free_rates: dict,
        corr_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Price all products in the portfolio.

        Infers the correct payoff function from the product's type_style field:
          "european" → european_brc_payoff
          (future)    → american_brc_payoff, autocall_payoff

        Parameters
        ----------
        portfolio : pd.DataFrame
        vol_map : dict             { isin: vol }
        risk_free_rates : dict     { currency: rate }
        corr_df : pd.DataFrame     full correlation matrix (ISIN index/columns)

        Returns
        -------
        pd.DataFrame   columns: product_id, fair_value, fair_value_pct,
                       std_error, fallbacks
        """
        rows = []

        for _, row in portfolio.iterrows():
            r = float(risk_free_rates.get(row["currency"], DEFAULT_RISK_FREE_RATE))
            corr_matrix, corr_fb = self._get_corr_subset(row, corr_df)
            payoff_fn   = self._resolve_payoff(row)
            fallbacks   = self._input_fallbacks(row, vol_map, corr_fb, risk_free_rates)
            if fallbacks:
                log.warning("Pricing %s with defaulted inputs: %s",
                            row["product_id"], "; ".join(fallbacks))

            result = self.price(row, payoff_fn, vol_map, r, corr_matrix=corr_matrix)

            rows.append({
                "product_id":     row["product_id"],
                "fair_value":     result["fair_value"],
                "fair_value_pct": result["fair_value_pct"],
                "std_error":      result["std_error"],
                "fallbacks":      "; ".join(fallbacks),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Apply payoff to a precomputed path tensor
    # ------------------------------------------------------------------
    def _fv_from_paths(
        self,
        paths: np.ndarray,
        dates: pd.DatetimeIndex,
        payoff_fn: Callable,
        row: pd.Series,
        risk_free_rate: float,
        T_remaining: float,
    ) -> tuple[float, float]:
        """Discount a payoff applied to a precomputed path tensor.

        Lets :meth:`compute_greeks` reprice for delta (scaled spot) and theta
        (shifted T) without re-running ``simulate_paths``.

        Autocallables are discounted per path to their own redemption date (the
        early-call date, or maturity if never called); every other product is
        discounted uniformly to maturity.  Routing both through this one method
        keeps fair value and the bump-and-reprice Greeks on a single model.

        Returns ``(fair_value, std_error)``.
        """
        if _is_autocallable(row):
            pv_paths = self._autocallable_path_pv(paths, dates, row, risk_free_rate)
        else:
            payoffs  = payoff_fn(paths, dates, row)
            pv_paths = payoffs * np.exp(-risk_free_rate * T_remaining)
        fair_value = float(np.mean(pv_paths))
        std_error  = float(np.std(pv_paths) / np.sqrt(self.n_paths))
        return fair_value, std_error

    def _autocallable_path_pv(
        self,
        paths: np.ndarray,
        dates: pd.DatetimeIndex,
        row: pd.Series,
        risk_free_rate: float,
    ) -> np.ndarray:
        """Per-path present value of an autocallable BRC.

        Evaluates the path-dependent payoff with the same engine the stress
        views use (:func:`vectorised_autocallable_rc_summary`), then discounts
        each path to *its* cashflow date: the early-call date on called paths,
        or maturity on paths that run to the end.  This is what makes the fair
        value methodologically correct for autocallables, rather than pricing
        them as plain European notes.
        """
        summary    = vectorised_autocallable_rc_summary(row, paths, dates)
        payoffs    = np.asarray(summary["total_payoff"], dtype=float)
        autocalled = np.asarray(summary["autocalled"], dtype=bool)
        call_date  = summary["call_date"]                      # object array

        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])

        # Year-fraction (ACT/360, consistent with the rest of the pricer) from
        # today to each path's cashflow date.
        days = np.array(
            [
                ((call_date[p] if autocalled[p] else maturity) - today).days
                for p in range(payoffs.shape[0])
            ],
            dtype=float,
        )
        tau = np.maximum(days / 360.0, 0.0)
        return payoffs * np.exp(-risk_free_rate * tau)

    # ------------------------------------------------------------------
    # Greeks — bump and reprice
    # ------------------------------------------------------------------
    def compute_greeks(
        self,
        row: pd.Series,
        payoff_fn: Callable,
        vol_map: dict,
        risk_free_rate: float,
        corr_matrix: np.ndarray | None = None,
        spot_bump_pct: float = 0.01,   # 1 % spot move
        vol_bump: float = 0.01,        # 1 pp vol move
        rate_bump: float = 0.0001,     # 1 bp rate move
        corr_bump: float = 0.01,       # 1 pp correlation move (MBRC)
        return_base_price: bool = False,
    ) -> dict:
        """Per-product Greeks by central finite difference.

        Base and bumped reprices share the RNG seed (common random numbers),
        so MC noise largely cancels in the difference.

        Greeks are absolute FV changes (product currency) per bump:

          delta     1 % spot move, per underlying
          vega      1 pp (0.01) vol move, per underlying
          theta     one calendar day, T decreasing (approx)
          rho       1 bp (0.0001) rate move
          corr_sens 1 pp uniform correlation shift (MBRC only)

        Returns
        -------
        dict: product_id, isins, underlyings, delta (list), vega (list),
              theta, rho, corr_sens
        """
        # If the product has already matured, fall back to the original
        # price() path which short-circuits on T_remaining <= 0.  No
        # simulation work to optimise.
        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])
        T_remaining = max((maturity - today).days / 360, 0.0)

        if T_remaining <= 0:
            base_result = self.price(row, payoff_fn, vol_map, risk_free_rate, corr_matrix)
            base_fv = base_result["fair_value"]
            isins   = list(row["underlying_isins"])
            names   = list(row["underlyings"])
            n       = len(isins)
            zeros   = [0.0] * n
            result  = {
                "product_id":  row["product_id"],
                "isins":       isins,
                "underlyings": names,
                "delta":       zeros,
                "vega":        zeros,
                "theta":       0.0,
                "rho":         0.0,
                "corr_sens":   None,
            }
            if return_base_price:
                result["fair_value"]     = base_fv
                result["fair_value_pct"] = base_result["fair_value_pct"]
                result["std_error"]      = base_result["std_error"]
            return result

        # Simulate the base path tensor once; delta and theta reuse it
        # rather than re-running simulate_paths (the dominant cost).
        paths_base, dates_base = self.simulate_paths(
            row, vol_map, risk_free_rate, corr_matrix,
        )
        base_fv, base_se = self._fv_from_paths(
            paths_base, dates_base, payoff_fn, row, risk_free_rate, T_remaining,
        )

        isins  = list(row["underlying_isins"])
        names  = list(row["underlyings"])
        spots  = [float(s) for s in row["current_spots"]]
        n      = len(isins)

        # ── Delta ─────────────────────────────────────────────────────────
        # GBM is multiplicative in S_{i,0}: S_{i,t} = S_{i,0}·exp(drift·t + σ_i√t·Z).
        # Bumping S_{i,0} by (1+ε) equals scaling the i-th path column by (1+ε),
        # so delta needs no re-simulation.
        deltas = []
        for i in range(n):
            paths_up = paths_base.copy()
            paths_dn = paths_base.copy()
            paths_up[:, :, i] *= (1.0 + spot_bump_pct)
            paths_dn[:, :, i] *= (1.0 - spot_bump_pct)

            fv_up, _ = self._fv_from_paths(
                paths_up, dates_base, payoff_fn, row, risk_free_rate, T_remaining,
            )
            fv_dn, _ = self._fv_from_paths(
                paths_dn, dates_base, payoff_fn, row, risk_free_rate, T_remaining,
            )
            # FV change for a 1 % spot move (central diff over ±1 %)
            deltas.append((fv_up - fv_dn) / 2)

        # ── Vega ──────────────────────────────────────────────────────────
        vegas = []
        for isin in isins:
            base_vol = vol_map.get(isin, 0.15)
            vol_up = {**vol_map, isin: base_vol + vol_bump}
            vol_dn = {**vol_map, isin: base_vol - vol_bump}

            fv_up = self.price(row, payoff_fn, vol_up, risk_free_rate, corr_matrix)["fair_value"]
            fv_dn = self.price(row, payoff_fn, vol_dn, risk_free_rate, corr_matrix)["fair_value"]

            # FV change per 1 pp vol move
            vegas.append((fv_up - fv_dn) / 2)

        # ── Theta ─────────────────────────────────────────────────────────
        # FV one business day closer to maturity. Re-simulating with a shifted
        # maturity changes the grid size and breaks CRN, so the small daily
        # signal is lost in MC noise. Instead reuse the base tensor: evaluate
        # the payoff at the penultimate step and discount for T_remaining - 1
        # business day (the contractual coupon is unaffected by one day).
        if len(dates_base) >= 2:
            paths_tm1 = paths_base[:, :-1, :]
            dates_tm1 = dates_base[:-1]
            T_tm1     = max((dates_base[-2] - today).days / 360, 0.0)
            fv_tm1, _ = self._fv_from_paths(
                paths_tm1, dates_tm1, payoff_fn, row, risk_free_rate, T_tm1,
            )
            theta     = fv_tm1 - base_fv
        else:
            theta = 0.0

        # ── Rho ───────────────────────────────────────────────────────────
        fv_up = self.price(row, payoff_fn, vol_map, risk_free_rate + rate_bump, corr_matrix)["fair_value"]
        fv_dn = self.price(row, payoff_fn, vol_map, risk_free_rate - rate_bump, corr_matrix)["fair_value"]
        rho = (fv_up - fv_dn) / 2   # per 1 bp

        # ── Correlation sensitivity (MBRC only) ───────────────────────────
        corr_sens = None
        if n > 1 and corr_matrix is not None:
            off_diag = 1 - np.eye(n)

            corr_up = np.clip(corr_matrix + corr_bump * off_diag, -1.0, 1.0)
            corr_dn = np.clip(corr_matrix - corr_bump * off_diag, -1.0, 1.0)

            # Fall back to base if bumped matrix is no longer positive-definite
            for bumped, original in [(corr_up, corr_matrix), (corr_dn, corr_matrix)]:
                try:
                    np.linalg.cholesky(bumped)
                except np.linalg.LinAlgError:
                    bumped[:] = original

            fv_up = self.price(row, payoff_fn, vol_map, risk_free_rate, corr_up)["fair_value"]
            fv_dn = self.price(row, payoff_fn, vol_map, risk_free_rate, corr_dn)["fair_value"]
            corr_sens = (fv_up - fv_dn) / 2   # per 1 pp correlation move

        result = {
            "product_id":  row["product_id"],
            "isins":       isins,
            "underlyings": names,
            "delta":       deltas,    # list[float]
            "vega":        vegas,     # list[float]
            "theta":       theta,     # float
            "rho":         rho,       # float
            "corr_sens":   corr_sens, # float | None
        }
        if return_base_price:
            notional = float(row["notional"])
            result["fair_value"]     = base_fv
            result["fair_value_pct"] = base_fv / notional if notional else float("nan")
            result["std_error"]      = base_se
        return result

    # ------------------------------------------------------------------
    # Portfolio-level Greeks
    # ------------------------------------------------------------------
    def compute_portfolio_greeks(
        self,
        portfolio: pd.DataFrame,
        vol_map: dict,
        risk_free_rates: dict,
        corr_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute Greeks for every product and aggregate delta to portfolio level.

        Returns
        -------
        greeks_df : pd.DataFrame (long format — one row per product × underlying)
            product_id, currency, isin, underlying,
            delta_1pct, vega_1pp, theta, rho, corr_sens

        portfolio_delta : pd.DataFrame (one row per unique underlying)
            isin, underlying, total_delta_1pct
            Sorted by absolute delta descending — shows largest exposures first.

        fair_values : pd.DataFrame (one row per product)
            product_id, fair_value, fair_value_pct, std_error. Reuses the base
            price from ``compute_greeks``, avoiding a separate ``price_portfolio``.
        """
        greeks_rows = []
        fv_rows     = []
        delta_agg   = {}   # isin → {"underlying": str, "currency": str, "total": float}

        for _, row in portfolio.iterrows():
            r = float(risk_free_rates.get(row["currency"], DEFAULT_RISK_FREE_RATE))
            corr_matrix, corr_fb = self._get_corr_subset(row, corr_df)
            payoff_fn   = self._resolve_payoff(row)
            fallbacks   = self._input_fallbacks(row, vol_map, corr_fb, risk_free_rates)
            if fallbacks:
                log.warning("Pricing %s with defaulted inputs: %s",
                            row["product_id"], "; ".join(fallbacks))

            g = self.compute_greeks(row, payoff_fn, vol_map, r, corr_matrix,
                                    return_base_price=True)

            fv_rows.append({
                "product_id":     row["product_id"],
                "fair_value":     g["fair_value"],
                "fair_value_pct": g["fair_value_pct"],
                "std_error":      g["std_error"],
                "fallbacks":      "; ".join(fallbacks),
            })

            for isin, name, delta, vega in zip(
                g["isins"], g["underlyings"], g["delta"], g["vega"]
            ):
                delta_rounded = round(delta, 2)
                greeks_rows.append({
                    "product_id":  row["product_id"],
                    "currency":    row["currency"],
                    "isin":        isin,
                    "underlying":  name,
                    "delta_1pct":  delta_rounded,
                    "vega_1pp":    round(vega, 2),
                    "theta":       round(g["theta"], 2),
                    "rho":         round(g["rho"], 2),
                    "corr_sens":   round(g["corr_sens"], 2) if g["corr_sens"] is not None else None,
                })

                if isin not in delta_agg:
                    delta_agg[isin] = {"underlying": name, "currency": row["currency"], "total": 0.0}
                delta_agg[isin]["total"] += delta_rounded

        greeks_df = pd.DataFrame(greeks_rows)

        portfolio_delta = pd.DataFrame([
            {
                "isin":            isin,
                "underlying":      v["underlying"],
                "currency":        v["currency"],
                "total_delta_1pct": round(v["total"], 2),
            }
            for isin, v in delta_agg.items()
        ])
        portfolio_delta = (
            portfolio_delta
            .assign(abs_delta=lambda d: d["total_delta_1pct"].abs())
            .sort_values("abs_delta", ascending=False)
            .drop(columns="abs_delta")
            .reset_index(drop=True)
        )

        fv_df = pd.DataFrame(fv_rows)
        return greeks_df, portfolio_delta, fv_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_payoff(self, row: pd.Series) -> Callable:
        """Pick the right payoff function for this product.

        Dispatch on ``product_type`` first (so CPN doesn't get priced with
        BRC's worst-of barrier mechanic); fall back to ``type_style`` for
        the legacy BRC/MBRC path.
        """
        ptype = str(row.get("product_type", "")).upper()
        if ptype == "CPN":
            return capital_protection_note_payoff
        # AC_BRC is path-dependent: ``price``/``_fv_from_paths`` detect it via
        # ``_is_autocallable`` and value it with a per-path discounted autocall
        # model, so the payoff function returned here is unused for it.  The
        # European payoff is kept as a harmless default (and is what the
        # autocall engine itself applies to paths that run to maturity).
        style = str(row.get("type_style", "european")).lower()
        if style == "european":
            return european_brc_payoff
        raise NotImplementedError(
            f"No payoff function registered for type_style='{style}'. "
            "Add one in monte_carlo.py and register it here."
        )

    def _get_corr_subset(
        self, row: pd.Series, corr_df: pd.DataFrame | None
    ) -> tuple[np.ndarray, bool]:
        """Return ``(corr_submatrix, fell_back)``.

        ``fell_back`` is True when the product has multiple underlyings but no
        correlation data covers them, so an identity matrix is substituted.
        Single-underlying products need no correlation, so identity there is
        exact and not flagged.
        """
        isins = list(row["underlying_isins"])
        n = len(isins)

        if n == 1:
            return np.eye(1), False

        if corr_df is None or any(i not in corr_df.index for i in isins):
            return np.eye(n), True

        return corr_df.loc[isins, isins].to_numpy(dtype=float), False

    @staticmethod
    def _input_fallbacks(
        row: pd.Series, vol_map: dict, corr_fell_back: bool,
        risk_free_rates: dict | None = None,
    ) -> list[str]:
        """Provenance tags for inputs substituted by defaults for this product."""
        tags: list[str] = []
        if risk_free_rates is not None and row["currency"] not in risk_free_rates:
            tags.append(f"rate:{row['currency']}")
        missing_vols = [i for i in row["underlying_isins"] if i not in vol_map]
        if missing_vols:
            tags.append("vol:" + ",".join(missing_vols))
        if corr_fell_back:
            tags.append("correlation:identity")
        return tags
