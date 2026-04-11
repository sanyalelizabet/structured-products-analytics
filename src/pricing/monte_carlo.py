"""
Monte Carlo pricing engine for structured products.

Architecture (Option A — payoff strategy)
------------------------------------------
  MonteCarloPricer.simulate_paths()   — correlated GBM, returns full path matrix
  MonteCarloPricer.price()            — simulates paths, applies payoff_fn, discounts
  MonteCarloPricer.price_portfolio()  — iterates portfolio, infers payoff_fn from row

  Payoff functions live outside the pricer:
    european_brc_payoff(paths, dates, row) → np.ndarray (n_paths,)

  To add autocalls or American barriers: write a new payoff function and pass it in.
  The pricer itself never changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable


# ---------------------------------------------------------------------------
# Payoff functions
# ---------------------------------------------------------------------------

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

    T_total = (
        pd.Timestamp(row["maturity_date"]) -
        pd.Timestamp(row["initial_fixing_date"])
    ).days / 360

    # Terminal spots — last step of the path
    terminal = paths[:, -1, :]                                # (n_paths, n_assets)

    # Performance relative to strike at maturity
    perfs      = terminal / strikes                           # (n_paths, n_assets)
    worst_perf = perfs.min(axis=1)                            # (n_paths,)

    # Redemption: full notional if worst-of above barrier, else notional * worst_perf
    redemptions = np.where(worst_perf >= 1.0, notional, notional * worst_perf)

    # Fixed coupon paid unconditionally at maturity
    coupon = notional * float(row["coupon"]) * T_total

    return redemptions + coupon


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

class MonteCarloPricer:
    """
    Simulates correlated GBM paths and prices any product whose payoff
    can be expressed as a function of those paths.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths.  10,000 is fast and accurate for
        European payoffs.  Increase to 50,000+ for path-dependent features.
    seed : int
        Base random seed for reproducibility.
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

        vols = np.array([vol_map.get(isin, 0.15) for isin in isins])
        r    = float(risk_free_rate)

        # Business-day grid
        dates   = pd.bdate_range(start=today, end=maturity)
        n_steps = len(dates)

        # Correlation
        if corr_matrix is None or n_assets == 1:
            corr_matrix = np.eye(n_assets)
        L = np.linalg.cholesky(corr_matrix)

        # Draw all random increments at once: (n_paths, n_steps, n_assets)
        rng   = np.random.default_rng(self.seed)
        Z_raw = rng.standard_normal((self.n_paths, n_steps, n_assets))
        Z     = Z_raw @ L.T                                    # correlated

        # Step sizes in years — prepend today as datetime64 to match dates dtype
        all_dates = np.concatenate([[today.to_datetime64()], dates.values])
        dt = np.diff(all_dates) / np.timedelta64(1, "D") / 365.25
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
        """
        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])

        T_remaining = max((maturity - today).days / 360, 0.0)
        notional    = float(row["notional"])

        # Already expired — return intrinsic value
        if T_remaining <= 0:
            spots    = np.array([float(s) for s in row["current_spots"]])
            strikes  = np.array([float(k) for k in row["strike"]])
            T_total  = (maturity - pd.Timestamp(row["initial_fixing_date"])).days / 360
            worst    = (spots / strikes).min()
            redemp   = notional if worst >= 1.0 else notional * worst
            coupon   = notional * float(row["coupon"]) * T_total
            fv       = redemp + coupon
            return {
                "fair_value":     fv,
                "fair_value_pct": fv / notional if notional else np.nan,
                "std_error":      0.0,
            }

        paths, dates = self.simulate_paths(row, vol_map, risk_free_rate, corr_matrix)

        payoffs        = payoff_fn(paths, dates, row)            # (n_paths,)
        discount       = np.exp(-risk_free_rate * T_remaining)
        pv_paths       = payoffs * discount

        fair_value = float(np.mean(pv_paths))
        std_error  = float(np.std(pv_paths) / np.sqrt(self.n_paths))

        return {
            "fair_value":     fair_value,
            "fair_value_pct": fair_value / notional if notional else np.nan,
            "std_error":      std_error,
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
        pd.DataFrame   columns: product_id, fair_value, fair_value_pct, std_error
        """
        rows = []

        for _, row in portfolio.iterrows():
            r           = float(risk_free_rates.get(row["currency"], 0.02))
            corr_matrix = self._get_corr_subset(row, corr_df)
            payoff_fn   = self._resolve_payoff(row)

            result = self.price(row, payoff_fn, vol_map, r, corr_matrix=corr_matrix)

            rows.append({
                "product_id":     row["product_id"],
                "fair_value":     result["fair_value"],
                "fair_value_pct": result["fair_value_pct"],
                "std_error":      result["std_error"],
            })

        return pd.DataFrame(rows)

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
    ) -> dict:
        """
        Per-product Greeks via central finite differences.

        The same RNG seed is used for base and every bumped reprice so that
        the Monte Carlo noise cancels almost entirely in the difference —
        the signal-to-noise ratio of a finite-difference Greek is much better
        than that of the fair value itself.

        All Greeks are expressed as the absolute FV change (in product
        currency) per one unit of the bump:

          delta    — FV change for a 1 % spot move, per underlying
          vega     — FV change for a 1 pp (0.01) vol move, per underlying
          theta    — FV change for 1 calendar day passing (approx)
          rho      — FV change for a 1 bp (0.0001) rate move
          corr_sens— FV change for a 1 pp uniform correlation shift (MBRC only)

        Returns
        -------
        dict with keys: product_id, isins, underlyings,
                        delta (list), vega (list), theta, rho, corr_sens
        """
        base_fv = self.price(row, payoff_fn, vol_map, risk_free_rate, corr_matrix)["fair_value"]

        isins  = list(row["underlying_isins"])
        names  = list(row["underlyings"])
        spots  = [float(s) for s in row["current_spots"]]
        n      = len(isins)

        # ── Delta ─────────────────────────────────────────────────────────
        deltas = []
        for i in range(n):
            bump = spots[i] * spot_bump_pct

            spots_up = spots.copy(); spots_up[i] += bump
            spots_dn = spots.copy(); spots_dn[i] -= bump

            row_up = row.copy(); row_up["current_spots"] = spots_up
            row_dn = row.copy(); row_dn["current_spots"] = spots_dn

            fv_up = self.price(row_up, payoff_fn, vol_map, risk_free_rate, corr_matrix)["fair_value"]
            fv_dn = self.price(row_dn, payoff_fn, vol_map, risk_free_rate, corr_matrix)["fair_value"]

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
        # Theta = FV if we were 1 business day closer to maturity.
        #
        # Naive approach (shift maturity date by -1 day) breaks CRN because
        # the path grid changes size, so Monte Carlo noise swamps the tiny
        # signal (~a few CHF per day).
        #
        # Correct approach: generate paths to full maturity (same grid, CRN
        # preserved), then evaluate the payoff at the penultimate step and
        # discount for T_remaining - 1 business day. The coupon is fixed
        # contractually and is unchanged by 1 day passing.
        today = pd.Timestamp.today().normalize()
        mat   = pd.Timestamp(row["maturity_date"])

        paths_full, dates_full = self.simulate_paths(row, vol_map, risk_free_rate, corr_matrix)

        if len(dates_full) >= 2:
            # Use penultimate step as the terminal value (= 1 business day closer)
            paths_tm1    = paths_full[:, :-1, :]
            dates_tm1    = dates_full[:-1]
            T_tm1        = max((dates_full[-2] - today).days / 360, 0.0)

            payoffs_tm1  = payoff_fn(paths_tm1, dates_tm1, row)   # row dates unchanged → coupon fixed
            fv_tm1       = float(np.mean(payoffs_tm1)) * np.exp(-risk_free_rate * T_tm1)
            theta        = fv_tm1 - base_fv
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

        return {
            "product_id":  row["product_id"],
            "isins":       isins,
            "underlyings": names,
            "delta":       deltas,    # list[float]
            "vega":        vegas,     # list[float]
            "theta":       theta,     # float
            "rho":         rho,       # float
            "corr_sens":   corr_sens, # float | None
        }

    # ------------------------------------------------------------------
    # Portfolio-level Greeks
    # ------------------------------------------------------------------
    def compute_portfolio_greeks(
        self,
        portfolio: pd.DataFrame,
        vol_map: dict,
        risk_free_rates: dict,
        corr_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        """
        greeks_rows = []
        delta_agg   = {}   # isin → {"underlying": str, "currency": str, "total": float}

        for _, row in portfolio.iterrows():
            r           = float(risk_free_rates.get(row["currency"], 0.02))
            corr_matrix = self._get_corr_subset(row, corr_df)
            payoff_fn   = self._resolve_payoff(row)

            g = self.compute_greeks(row, payoff_fn, vol_map, r, corr_matrix)

            for isin, name, delta, vega in zip(
                g["isins"], g["underlyings"], g["delta"], g["vega"]
            ):
                greeks_rows.append({
                    "product_id":  row["product_id"],
                    "currency":    row["currency"],
                    "isin":        isin,
                    "underlying":  name,
                    "delta_1pct":  round(delta, 2),
                    "vega_1pp":    round(vega, 2),
                    "theta":       round(g["theta"], 2),
                    "rho":         round(g["rho"], 2),
                    "corr_sens":   round(g["corr_sens"], 2) if g["corr_sens"] is not None else None,
                })

                if isin not in delta_agg:
                    delta_agg[isin] = {"underlying": name, "currency": row["currency"], "total": 0.0}
                delta_agg[isin]["total"] += delta

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

        return greeks_df, portfolio_delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_payoff(self, row: pd.Series) -> Callable:
        style = str(row.get("type_style", "european")).lower()
        if style == "european":
            return european_brc_payoff
        raise NotImplementedError(
            f"No payoff function registered for type_style='{style}'. "
            "Add one in monte_carlo.py and register it here."
        )

    def _get_corr_subset(
        self, row: pd.Series, corr_df: pd.DataFrame | None
    ) -> np.ndarray:
        isins = list(row["underlying_isins"])

        if corr_df is None:
            return np.eye(len(isins))

        missing = [i for i in isins if i not in corr_df.index]
        if missing:
            return np.eye(len(isins))   # graceful fallback

        return corr_df.loc[isins, isins].to_numpy(dtype=float)
