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

from typing import Optional

from src.numerics.linalg import safe_cholesky
from src.pricing.products.reverse_convertible import barrier_levels
from src.pricing.products.autocallable_reverse_convertible import (
    vectorised_autocallable_rc_summary,
)
from src.pricing.products.issuer_callable_reverse_convertible import (
    vectorised_issuer_callable_rc_summary,
)
from src.pricing.barrier import (
    continuous_survival_prob,
    continuous_survival_prob_from_var,
    sample_knock_in,
)
from src.numerics.noise_sampler import _stable_isin_seed

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


def _is_issuer_callable(row: pd.Series) -> bool:
    """True for issuer-callable BRCs, valued by optimal-exercise (LSM)."""
    return str(row.get("product_type", "")).upper() == "IC_BRC"


_BARRIER_PRODUCT_TYPES = {"BRC", "MBRC", "AC_BRC", "IC_BRC"}


def _is_american(row: pd.Series) -> bool:
    """True for barrier products observed continuously (American style).

    ``type_style == "american"`` selects continuous monitoring; only barrier
    products carry a barrier, so capital-protection notes (CPN) never qualify.
    """
    return (
        str(row.get("type_style", "european")).lower() == "american"
        and str(row.get("product_type", "")).upper() in _BARRIER_PRODUCT_TYPES
    )


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

    def simulate_paths_local_vol(
        self,
        row: pd.Series,
        vol_surfaces: dict,
        fallback_vol_map: dict,
        risk_free_rate: float,
        corr_matrix: np.ndarray | None = None,
        damping_cap: Optional[float] = None,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """Simulate correlated paths under Dupire local volatility.

        The path generation differs from :meth:`simulate_paths` in two
        respects. First, the per-step volatility is no longer a single
        scalar per underlying but a function of the current spot and
        time, sourced from each underlying's :class:`VolSurface` by
        calling :meth:`VolSurface.local_volatility`. Second, the path
        is evolved step by step rather than in a single vectorised
        log-increment cumsum, because the next step's volatility
        depends on the *current* spot.

        The per-step variance returned alongside the path tensor is
        the same variance the bridge utilities must consume to
        compute knock-in probabilities under the same dynamics. This
        is the structural protection against the implied-realised
        inconsistency: one ``sigma_step`` value, two call sites
        (path-step and bridge-step), identical numerical input.

        For any underlying whose surface is unavailable, lies in the
        fallback regime, or has insufficient term-structure
        information at the relevant tenor, the per-step volatility is
        the corresponding entry of ``fallback_vol_map`` for every
        step. The graceful degradation matches the discipline of
        Stage 3 Substage A.

        Parameters
        ----------
        row : pd.Series
            Portfolio row; required fields match :meth:`simulate_paths`.
        vol_surfaces : dict
            ``{ isin: VolSurface }``. Underlyings absent from the map
            fall back to the constant-volatility input.
        fallback_vol_map : dict
            Per-underlying constant volatility used when the surface
            cannot produce a value.
        risk_free_rate : float
            Annualised continuously-compounded risk-free rate.
        corr_matrix : np.ndarray, optional
            Correlation between underlyings; defaults to identity.
        damping_cap : float or None, optional
            Conservative-mode clip applied to every local-volatility
            evaluation. See :meth:`VolSurface.local_volatility`.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps, n_assets)
        dates : pd.DatetimeIndex of length n_steps
        sigma_path : np.ndarray, shape (n_paths, n_steps, n_assets)
            The per-(path, step, asset) volatility used to generate
            each step, suitable as input to the time-varying form of
            the Brownian-bridge survival probability.
        """
        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])
        isins    = list(row["underlying_isins"])
        spots    = np.array([float(s) for s in row["current_spots"]])
        n_assets = len(isins)
        r        = float(risk_free_rate)

        dates   = pd.bdate_range(start=today, end=maturity)
        n_steps = len(dates)

        if corr_matrix is None or n_assets == 1:
            corr_matrix = np.eye(n_assets)
        L = safe_cholesky(corr_matrix)

        rng   = np.random.default_rng(self.seed)
        Z_raw = rng.standard_normal((self.n_paths, n_steps, n_assets))
        Z     = Z_raw @ L.T

        all_dates = np.concatenate([[today.to_datetime64()], dates.values])
        dt = np.diff(all_dates) / np.timedelta64(1, "D") / 360.0
        dt = np.maximum(dt, 0.0)

        # Per-asset surface resolution: keep a reference to the
        # VolSurface (or None) and the legacy fallback sigma per
        # underlying. This lookup is done once before the time loop;
        # the per-step evaluation pays only for the Dupire computation.
        surface_per_asset = [(vol_surfaces or {}).get(isin) for isin in isins]
        fallback_sigma = np.array(
            [float(fallback_vol_map.get(isin, DEFAULT_VOL)) for isin in isins]
        )

        # ``t_start[j]`` is the calendar time elapsed since today at
        # the START of step j. The local volatility for the step is
        # evaluated at the midpoint of the step, ``t_eval[j] =
        # t_start[j] + 0.5 * dt[j]``: this avoids the T=0 singularity
        # of the Dupire formula at the very first step, gives the
        # same first-order accuracy as a left-endpoint Euler scheme,
        # and provides a single, unambiguous value that both the
        # path-step and the bridge-step consume.
        t_end_per_step   = np.cumsum(dt)
        t_start_per_step = np.concatenate([[0.0], t_end_per_step[:-1]])
        t_eval_per_step  = t_start_per_step + 0.5 * dt

        # Pre-allocate output tensors.
        paths      = np.empty((self.n_paths, n_steps, n_assets), dtype=float)
        sigma_path = np.empty((self.n_paths, n_steps, n_assets), dtype=float)

        # Current spot per (path, asset). Broadcast the initial spots
        # to the path-major shape and evolve in place.
        S = np.broadcast_to(spots, (self.n_paths, n_assets)).copy()

        # The Python loop is unavoidable in the local-vol regime because
        # the next step's volatility depends on the *current* spot. The
        # inner work is fully vectorised across paths and assets, so the
        # per-step cost is comparable to a single vectorised constant-vol
        # step and the total runtime overhead relative to the bulk path
        # generator is a constant factor of a few.
        for j in range(n_steps):
            t_eval = float(t_eval_per_step[j])
            sigma_j = np.empty((self.n_paths, n_assets), dtype=float)
            for a, surface in enumerate(surface_per_asset):
                if (
                    surface is None
                    or getattr(surface, "n_svi_slices", 0) == 0
                    or t_eval <= 0.0
                ):
                    sigma_j[:, a] = fallback_sigma[a]
                    continue
                try:
                    sigma_eval = surface.local_volatility(
                        S[:, a], t_eval, damping_cap=damping_cap,
                    )
                    sigma_j[:, a] = np.asarray(sigma_eval, dtype=float)
                except Exception as exc:  # defensive: never let the MC die
                    log.warning(
                        "local_volatility raised on asset %s at t=%.4f; "
                        "falling back to constant vol (%s)",
                        isins[a], t_eval, exc,
                    )
                    sigma_j[:, a] = fallback_sigma[a]

            sigma_path[:, j, :] = sigma_j

            drift_j = (r - 0.5 * sigma_j ** 2) * dt[j]
            diff_j  = sigma_j * np.sqrt(dt[j]) * Z[:, j, :]
            S = S * np.exp(drift_j + diff_j)
            paths[:, j, :] = S

        return paths, dates, sigma_path

    # ------------------------------------------------------------------
    # Local-vol dispatch predicate and uniform simulate wrapper
    # ------------------------------------------------------------------

    def _should_use_local_vol(self, row: pd.Series, vol_surfaces: dict | None) -> bool:
        """Decide whether a product is priced under local-volatility dynamics.

        Three conditions are required: at least one calibrated surface
        is supplied (``vol_surfaces`` non-empty); the product is
        path-dependent under any of the three pricer specialisations
        (autocallable, American-barrier, issuer-callable); and at
        least one of the product's underlyings carries a non-fallback
        surface. European-barrier products and products without any
        calibrated surface stay on the Substage A scalar input.
        """
        if not vol_surfaces:
            return False
        if not (_is_autocallable(row) or _is_american(row) or _is_issuer_callable(row)):
            return False
        for isin in row["underlying_isins"]:
            surf = vol_surfaces.get(isin)
            if surf is not None and getattr(surf, "n_svi_slices", 0) > 0:
                return True
        return False

    def _simulate(
        self,
        row: pd.Series,
        vol_map: dict,
        risk_free_rate: float,
        corr_matrix: np.ndarray | None,
        vol_surfaces: dict | None,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, Optional[np.ndarray]]:
        """Uniform entry point for path generation.

        Returns ``(paths, dates, sigma_path)`` where ``sigma_path`` is
        ``None`` in the constant-volatility regime (Stage 3 Substage A
        or earlier) and a ``(n_paths, n_steps, n_assets)`` tensor in
        the local-volatility regime (Stage 3 Substage B). The
        sigma_path tensor flows through the downstream payoff
        evaluation so that bridge consistency with the path step is
        preserved by construction.
        """
        if self._should_use_local_vol(row, vol_surfaces):
            return self.simulate_paths_local_vol(
                row, vol_surfaces or {}, vol_map, risk_free_rate, corr_matrix,
            )
        paths, dates = self.simulate_paths(row, vol_map, risk_free_rate, corr_matrix)
        return paths, dates, None

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
        vol_surfaces: dict | None = None,
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

        paths, dates, sigma_path = self._simulate(
            row, vol_map, risk_free_rate, corr_matrix, vol_surfaces,
        )

        # Discounting is centralised in ``_fv_from_paths`` so that path-dependent
        # products (autocallables) discount each path to its own cashflow date
        # rather than uniformly to maturity.
        fair_value, std_error = self._fv_from_paths(
            paths, dates, payoff_fn, row, risk_free_rate, T_remaining, vol_map,
            sigma_path=sigma_path,
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
        vol_map: dict,
        sigma_path: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Discount a payoff applied to a precomputed path tensor.

        Lets :meth:`compute_greeks` reprice for delta (scaled spot) and theta
        (shifted T) without re-running ``simulate_paths``.

        Dispatch by observation style, all on one model so that fair value and
        the bump-and-reprice Greeks never diverge:

        * **Autocallable** — discounted per path to its own redemption date
          (early-call date, or maturity if never called).
        * **American barrier** — continuously-monitored knock-in via the
          Brownian-bridge survival probability, discounted to maturity.
        * **European** — the payoff function with a single maturity discount.

        Returns ``(fair_value, std_error)``.
        """
        if _is_issuer_callable(row):
            pv_paths = self._issuer_callable_path_pv(
                paths, dates, row, risk_free_rate, vol_map,
            )
        elif _is_autocallable(row):
            pv_paths = self._autocallable_path_pv(
                paths, dates, row, risk_free_rate, vol_map,
                sigma_path=sigma_path,
            )
        elif _is_american(row):
            pv_paths = self._american_barrier_pv(
                paths, dates, row, risk_free_rate, T_remaining, vol_map,
                sigma_path=sigma_path,
            )
        else:
            payoffs  = payoff_fn(paths, dates, row)
            pv_paths = payoffs * np.exp(-risk_free_rate * T_remaining)
        fair_value = float(np.mean(pv_paths))
        std_error  = float(np.std(pv_paths) / np.sqrt(self.n_paths))
        return fair_value, std_error

    def _american_barrier_pv(
        self,
        paths: np.ndarray,
        dates: pd.DatetimeIndex,
        row: pd.Series,
        risk_free_rate: float,
        T_remaining: float,
        vol_map: dict,
        sigma_path: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-path present value of a continuously-monitored barrier RC.

        The down-barrier is live throughout the life, so the knock-in is valued
        with the Brownian-bridge survival probability
        (:func:`src.barrier.continuous_survival_prob`) rather than a single
        maturity check.  Redemption blends the survived (par) and knocked-in
        (worst-of conversion) outcomes by that probability — a deterministic,
        lower-variance valuation that keeps the Greeks stable under CRN.
        """
        notional = float(row["notional"])
        strikes  = np.array([float(k) for k in row["strike"]])
        barriers = barrier_levels(row["initial_levels"], row.get("barrier_pct"))
        isins    = list(row["underlying_isins"])

        # Monitor over the simulated business-day grid.  Using the grid points
        # (rather than prepending today's spot) keeps every monitored level
        # inside ``paths``, so the spot-bump Greeks — which scale ``paths`` —
        # stay exact; the dropped today→first-grid-day sub-interval is ~1 day
        # and immaterial to the knock-in probability.
        dt = np.maximum(
            np.diff(dates.values) / np.timedelta64(1, "D") / 360.0, 0.0,
        )  # length n_steps - 1, matching prices = paths

        if sigma_path is not None:
            # Local-volatility regime: per-step variance is sourced from
            # the same evaluator that produced the path. The sigma that
            # generated the step from ``dates[i]`` to ``dates[i+1]`` is
            # ``sigma_path[:, i+1, :]``, so the bridge variance for the
            # monitoring interval ``[i, i+1]`` is built from it directly.
            sigma_bridge = sigma_path[:, 1:, :]                   # (n_paths, n_steps-1, n_assets)
            step_var = (sigma_bridge ** 2) * dt[None, :, None]
            p_survive  = continuous_survival_prob_from_var(paths, barriers, step_var)
        else:
            vols     = np.array([vol_map.get(i, DEFAULT_VOL) for i in isins])
            p_survive  = continuous_survival_prob(paths, barriers, vols, dt)

        terminal   = paths[:, -1, :]
        worst_perf = (terminal / strikes).min(axis=1)
        redemption = p_survive * notional + (1.0 - p_survive) * notional * worst_perf

        T_total = (
            pd.Timestamp(row["maturity_date"]) -
            pd.Timestamp(row["initial_fixing_date"])
        ).days / 360.0
        coupon  = notional * float(row["coupon"]) * T_total

        payoff = redemption + coupon
        return payoff * np.exp(-risk_free_rate * T_remaining)

    def _autocallable_path_pv(
        self,
        paths: np.ndarray,
        dates: pd.DatetimeIndex,
        row: pd.Series,
        risk_free_rate: float,
        vol_map: dict,
        sigma_path: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-path present value of an autocallable BRC.

        Evaluates the path-dependent payoff with the same engine the stress
        views use (:func:`vectorised_autocallable_rc_summary`), then discounts
        each path to *its* cashflow date: the early-call date on called paths,
        or maturity on paths that run to the end.  This is what makes the fair
        value methodologically correct for autocallables, rather than pricing
        them as plain European notes.

        Under American (continuous) observation the paths that run to maturity
        carry a continuously-monitored knock-in: it is sampled per path against
        the Brownian-bridge survival probability, using uniforms fixed by the
        pricer seed and product id so the bump-and-reprice Greeks reuse the same
        draws.  An autocalled path redeems at par regardless of the barrier, so
        the mask only affects uncalled paths.
        """
        uncalled_breach_mask = None
        if _is_american(row):
            uncalled_breach_mask = self._sampled_american_mask(
                paths, dates, row, vol_map, sigma_path=sigma_path,
            )

        summary    = vectorised_autocallable_rc_summary(
            row, paths, dates, uncalled_breach_mask=uncalled_breach_mask,
        )
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

    def _issuer_callable_path_pv(
        self,
        paths: np.ndarray,
        dates: pd.DatetimeIndex,
        row: pd.Series,
        risk_free_rate: float,
        vol_map: dict,
    ) -> np.ndarray:
        """Per-path present value of an issuer-callable BRC.

        The barrier knock-in is resolved to a definite per-path indicator
        (continuous bridge-sampled for American observation, terminal for
        European) because the optimal-exercise regression works on realised
        cashflows.  The issuer's optimal call is solved by Longstaff–Schwartz in
        :func:`vectorised_issuer_callable_rc_summary`; each path is then
        discounted to its own termination date (the call date, or maturity).
        """
        if _is_american(row):
            breach_mask = self._sampled_american_mask(paths, dates, row, vol_map)
        else:
            strikes  = np.array([float(k) for k in row["strike"]])  # noqa: F841 (clarity)
            barriers = barrier_levels(row["initial_levels"], row.get("barrier_pct"))
            terminal = paths[:, -1, :]
            breach_mask = (terminal <= barriers[None, :]).any(axis=1)

        summary    = vectorised_issuer_callable_rc_summary(
            row, paths, dates, risk_free_rate, breach_mask=breach_mask,
        )
        payoffs    = np.asarray(summary["total_payoff"], dtype=float)
        called     = np.asarray(summary["issuer_called"], dtype=bool)
        call_date  = summary["call_date"]

        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])
        days = np.array(
            [
                ((call_date[p] if called[p] else maturity) - today).days
                for p in range(payoffs.shape[0])
            ],
            dtype=float,
        )
        tau = np.maximum(days / 360.0, 0.0)
        return payoffs * np.exp(-risk_free_rate * tau)

    def _sampled_american_mask(
        self, paths: np.ndarray, dates: pd.DatetimeIndex, row: pd.Series, vol_map: dict,
        sigma_path: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sampled continuous knock-in mask for fair-value autocallable pricing.

        Uses the model's per-step variance (σ_i² Δt of the pricing GBM) and a
        Brownian-bridge survival probability, then draws one uniform per path.
        The uniforms are fixed by ``(seed, product_id)`` so every bump-and-reprice
        Greek evaluation reuses identical draws.

        When ``sigma_path`` is supplied (Stage 3 Substage B local-vol
        regime), the per-step variance is sourced from that tensor
        rather than from the constant ``vol_map``: the variance for
        the interval between ``dates[i]`` and ``dates[i+1]`` is
        ``sigma_path[:, i+1, :]**2 * dt[i]``, preserving the
        path-step / bridge-step consistency contract.
        """
        barriers = barrier_levels(row["initial_levels"], row.get("barrier_pct"))
        dt       = np.maximum(np.diff(dates.values) / np.timedelta64(1, "D") / 360.0, 0.0)
        if sigma_path is not None:
            sigma_bridge = sigma_path[:, 1:, :]
            step_var = (sigma_bridge ** 2) * dt[None, :, None]
        else:
            isins    = list(row["underlying_isins"])
            vols     = np.array([vol_map.get(i, DEFAULT_VOL) for i in isins])
            step_var = (vols[None, :] ** 2) * dt[:, None]
        rng      = np.random.default_rng(_stable_isin_seed(str(row["product_id"]), self.seed))
        uniforms = rng.random(paths.shape[0])
        return sample_knock_in(paths, barriers, step_var, uniforms)

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
        vol_surfaces: dict | None = None,
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
            base_result = self.price(
                row, payoff_fn, vol_map, risk_free_rate, corr_matrix,
                vol_surfaces=vol_surfaces,
            )
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
        # Under local volatility (Stage 3 Substage B) the per-step
        # variance varies with the path, so we also retain the
        # ``sigma_path`` tensor for downstream bridge consistency.
        paths_base, dates_base, sigma_path_base = self._simulate(
            row, vol_map, risk_free_rate, corr_matrix, vol_surfaces,
        )
        base_fv, base_se = self._fv_from_paths(
            paths_base, dates_base, payoff_fn, row, risk_free_rate, T_remaining,
            vol_map, sigma_path=sigma_path_base,
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
                vol_map, sigma_path=sigma_path_base,
            )
            fv_dn, _ = self._fv_from_paths(
                paths_dn, dates_base, payoff_fn, row, risk_free_rate, T_remaining,
                vol_map, sigma_path=sigma_path_base,
            )
            # FV change for a 1 % spot move (central diff over ±1 %)
            deltas.append((fv_up - fv_dn) / 2)

        # ── Vega ──────────────────────────────────────────────────────────
        vegas = []
        for isin in isins:
            base_vol = vol_map.get(isin, 0.15)
            vol_up = {**vol_map, isin: base_vol + vol_bump}
            vol_dn = {**vol_map, isin: base_vol - vol_bump}

            fv_up = self.price(row, payoff_fn, vol_up, risk_free_rate, corr_matrix,
                               vol_surfaces=vol_surfaces)["fair_value"]
            fv_dn = self.price(row, payoff_fn, vol_dn, risk_free_rate, corr_matrix,
                               vol_surfaces=vol_surfaces)["fair_value"]

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
            sigma_path_tm1 = sigma_path_base[:, :-1, :] if sigma_path_base is not None else None
            fv_tm1, _ = self._fv_from_paths(
                paths_tm1, dates_tm1, payoff_fn, row, risk_free_rate, T_tm1,
                vol_map, sigma_path=sigma_path_tm1,
            )
            theta     = fv_tm1 - base_fv
        else:
            theta = 0.0

        # ── Rho ───────────────────────────────────────────────────────────
        fv_up = self.price(row, payoff_fn, vol_map, risk_free_rate + rate_bump,
                            corr_matrix, vol_surfaces=vol_surfaces)["fair_value"]
        fv_dn = self.price(row, payoff_fn, vol_map, risk_free_rate - rate_bump,
                            corr_matrix, vol_surfaces=vol_surfaces)["fair_value"]
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

            fv_up = self.price(row, payoff_fn, vol_map, risk_free_rate, corr_up,
                                vol_surfaces=vol_surfaces)["fair_value"]
            fv_dn = self.price(row, payoff_fn, vol_map, risk_free_rate, corr_dn,
                                vol_surfaces=vol_surfaces)["fair_value"]
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
        vol_surfaces: dict | None = None,
        valuation_date=None,
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

            # Stage 3 Substage A: if a vol surface is available, resolve a
            # product-specific volatility map evaluated at each underlying's
            # barrier strike at the product's maturity. The legacy global
            # vol_map remains the fallback for any underlying for which the
            # surface cannot produce a value. When no surfaces are supplied
            # the legacy behaviour is preserved exactly.
            if vol_surfaces:
                from src.pricing.vol_surface import build_product_vol_map
                product_vol_map, _surface_diagnostics = build_product_vol_map(
                    row, vol_surfaces, vol_map, valuation_date,
                )
            else:
                product_vol_map = vol_map

            fallbacks   = self._input_fallbacks(row, product_vol_map, corr_fb, risk_free_rates)
            if fallbacks:
                log.warning("Pricing %s with defaulted inputs: %s",
                            row["product_id"], "; ".join(fallbacks))

            g = self.compute_greeks(row, payoff_fn, product_vol_map, r, corr_matrix,
                                    return_base_price=True,
                                    vol_surfaces=vol_surfaces)

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
        # American (continuous) barrier products are valued in
        # ``_american_barrier_pv`` (detected via ``_is_american``); like the
        # autocall case the payoff function returned here is unused for them,
        # but the European payoff is a safe default for any code path that does
        # call it.
        style = str(row.get("type_style", "european")).lower()
        if style in ("european", "american"):
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
