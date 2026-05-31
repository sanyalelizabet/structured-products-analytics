"""
Implied volatility surface — raw SVI parameterisation
=====================================================

This module provides the mathematical core of the implied volatility surface
used by the structured-products pricers. The surface is constructed
slice-by-slice (one slice per listed expiry) and, in later stages, assembled
into a term-structure-consistent object via SSVI. This file contains the
parameterisation and pointwise evaluation only; calibration, arbitrage gates,
and the higher-level ``VolSliceSurface`` wrapper are added in subsequent
tasks of Stage 1.

Motivation
----------
For barrier products (BRC, MBRC, AC_BRC) the volatility that governs the
risk-neutral barrier-hit probability is the volatility at the *barrier
strike*, not the at-the-money volatility. Under the equity skew that
prevails in essentially all liquid markets, the implied volatility at a
typical 60–70 % barrier strike exceeds the ATM volatility by a material
amount (commonly 5–15 volatility points). Pricing the barrier with the ATM
volatility therefore introduces a systematic, signed mis-pricing. The
present module is the foundation for the correction: it gives a smooth,
arbitrage-aware representation of the volatility smile from which the
volatility at any strike — and, after Stages 2 and 3, at any (strike, time)
— may be obtained.

The raw SVI parameterisation
----------------------------
Following Gatheral (2004) the total implied variance ``w(k) ≡ σ²(k) · T``
is parameterised in log-moneyness ``k = ln(K / F)`` as

    w(k; a, b, ρ, m, ς) = a + b · { ρ (k − m) + sqrt[ (k − m)² + ς² ] }

with five free parameters:

    a   ∈ ℝ           vertical level; shifts the total-variance smile up or
                      down. The condition ``a + b ς sqrt(1 − ρ²) ≥ 0`` is
                      imposed to guarantee that ``w(k) ≥ 0`` for every
                      ``k``, so that the implied volatility is always real.
    b   ≥ 0           overall wing slope; controls the angle between the
                      two linear asymptotes of the smile in the wings.
    ρ   ∈ [−1, 1]     skew; sets the asymmetry between the put and call
                      wings. A negative ``ρ`` produces the steeper put
                      wing characteristic of equity smiles.
    m   ∈ ℝ           horizontal translation of the smile minimum along
                      the log-moneyness axis.
    ς   > 0           smoothness near the smile minimum; controls the
                      curvature of the smile around the at-the-money
                      region.

The implied volatility at log-moneyness ``k`` for tenor ``T`` is then
recovered as

    σ(k, T) = sqrt( w(k) / T ).

Two arbitrage conditions constrain the admissible parameter set and are
enforced in Task 3:

* Roger Lee's wing bounds, ``b (1 ± ρ) ≤ 4 / T``, guarantee that the
  asymptotic behaviour of the smile is compatible with the absence of
  static arbitrage on the call-price surface in the limit of extreme
  strikes.
* Durrleman's butterfly condition, the non-negativity of a particular
  density-like function ``g(k)`` derived from ``w(k)`` and its first two
  derivatives, guarantees the absence of butterfly arbitrage across all
  finite strikes.

This file establishes only the parameter container and pointwise
evaluation; calibration to listed chains and the arbitrage gates are
implemented in subsequent files.

References
----------
Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility
parameterization with application to the valuation of volatility
derivatives.* Presentation, Global Derivatives & Risk Management,
Madrid.

Gatheral, J. and Jacquier, A. (2014). *Arbitrage-free SVI volatility
surfaces.* Quantitative Finance 14(1), 59–71.

Roger Lee (2004). *The moment formula for implied volatility at extreme
strikes.* Mathematical Finance 14(3), 469–480.

De Marco, S. and Martini, C. (2009). *Quasi-explicit calibration of
Gatheral's SVI model.* Zeliade Systems white paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar


# Numerical floor for the curvature parameter ς. The raw SVI formula
# contains the term ``sqrt[(k − m)² + ς²]`` and becomes degenerate as
# ς → 0 (the smile collapses to two linear pieces meeting at a kink).
# A small positive floor avoids singular Jacobians during calibration and
# poses no economic loss: any liquid smile has strictly positive
# curvature near the money.
_SIGMA_FLOOR: float = 1.0e-6


@dataclass(frozen=True)
class SVIParams:
    """Parameter container for the raw SVI total-variance parameterisation.

    The five parameters are stored in the order conventional in the
    literature (a, b, ρ, m, ς). The container is immutable so that a
    calibrated slice cannot be silently mutated after the arbitrage
    checks have been applied.

    Attributes
    ----------
    a : float
        Vertical level of total variance.
    b : float
        Overall wing slope; must be non-negative.
    rho : float
        Skew parameter; must lie in the closed interval [−1, 1].
    m : float
        Horizontal translation along log-moneyness.
    sigma : float
        Curvature near the smile minimum; must be strictly positive.

    Notes
    -----
    The constructor performs only the elementary domain checks that
    correspond to the mathematical definition of the parameterisation
    (signs and intervals). The economically meaningful arbitrage
    conditions (Roger Lee wing bounds, Durrleman butterfly) are tenor-
    dependent and are therefore enforced separately in the arbitrage
    gates rather than at construction time.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.a, self.b, self.rho, self.m, self.sigma]).all():
            raise ValueError(
                f"SVIParams must be finite; received "
                f"a={self.a}, b={self.b}, rho={self.rho}, m={self.m}, sigma={self.sigma}"
            )
        if self.b < 0.0:
            raise ValueError(f"SVI parameter b must be non-negative; received {self.b}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"SVI parameter rho must lie in [-1, 1]; received {self.rho}")
        if self.sigma <= 0.0:
            raise ValueError(f"SVI parameter sigma must be strictly positive; received {self.sigma}")

        # Non-negativity of total variance at the smile minimum. The minimum
        # of w(k) is attained at k = m − ρ ς / sqrt(1 − ρ²) for |ρ| < 1, and
        # equals a + b ς sqrt(1 − ρ²). Requiring this to be non-negative is
        # equivalent to requiring w(k) ≥ 0 for every k. The constructor
        # tolerates a small negative numerical residue to remain robust to
        # the finite precision of an optimiser, but rejects parameters that
        # would imply a genuine negative-variance region.
        min_total_variance = self.a + self.b * self.sigma * float(np.sqrt(max(0.0, 1.0 - self.rho ** 2)))
        if min_total_variance < -1.0e-10:
            raise ValueError(
                f"SVI parameters imply a negative total-variance minimum "
                f"({min_total_variance:.3e}); the surface would yield "
                f"imaginary implied volatilities."
            )


def svi_total_variance(k: np.ndarray | float, params: SVIParams) -> np.ndarray | float:
    """Evaluate the SVI total-variance function ``w(k)``.

    The function returns the total implied variance ``w(k) = σ²(k) · T``
    at log-moneyness ``k`` for the calibrated parameter tuple. Because
    the parameterisation is in total variance rather than in volatility,
    the tenor enters only through the recovery of ``σ`` from ``w``
    (see :func:`svi_implied_vol`).

    Parameters
    ----------
    k : array-like or float
        Log-moneyness ``k = ln(K / F)`` at which the total variance is
        evaluated. The function is vectorised over ``k``.
    params : SVIParams
        Calibrated SVI parameters.

    Returns
    -------
    array-like or float
        Total implied variance at the requested log-moneyness, of the
        same shape as ``k``.

    Notes
    -----
    The numerical floor on the curvature parameter ``ς`` keeps the
    square-root term well-behaved at ``k = m``; without it the partial
    derivatives of ``w`` would be discontinuous at the smile minimum
    when ``ς`` is allowed to vanish.
    """
    k_arr = np.asarray(k, dtype=float)
    shifted = k_arr - params.m
    sigma_eff = max(params.sigma, _SIGMA_FLOOR)
    radical = np.sqrt(shifted * shifted + sigma_eff * sigma_eff)
    w = params.a + params.b * (params.rho * shifted + radical)

    # Guard against numerical underflow of the floor condition. The
    # constructor of SVIParams enforces non-negative total variance at
    # the minimum, but a finite-precision evaluation in the wings may
    # still produce a value such as -1e-18. We clip such residues to
    # zero so that subsequent square-roots remain well-defined.
    return np.where(w < 0.0, 0.0, w) if w.ndim else float(max(0.0, float(w)))


def svi_implied_vol(
    k: np.ndarray | float,
    T: float,
    params: SVIParams,
) -> np.ndarray | float:
    """Implied volatility implied by an SVI slice at a given tenor.

    The function inverts the total-variance representation to recover
    the annualised implied volatility ``σ(k, T) = sqrt(w(k) / T)`` at
    log-moneyness ``k``. The tenor ``T`` is the time-to-maturity (in
    years, continuous compounding) of the slice that produced
    ``params``; passing a different tenor is admissible but the result
    no longer corresponds to a calibrated implied volatility — it would
    re-scale the variance under a different time horizon, an operation
    that is mathematically defined but rarely economically meaningful.

    Parameters
    ----------
    k : array-like or float
        Log-moneyness ``k = ln(K / F)``.
    T : float
        Time to maturity in years; must be strictly positive.
    params : SVIParams
        Calibrated SVI parameters of the slice.

    Returns
    -------
    array-like or float
        Annualised implied volatility at the requested log-moneyness.

    Raises
    ------
    ValueError
        If ``T`` is non-positive.
    """
    if T <= 0.0:
        raise ValueError(f"Tenor T must be strictly positive; received {T}")
    w = svi_total_variance(k, params)
    return np.sqrt(np.asarray(w) / T) if np.ndim(w) else float(np.sqrt(float(w) / T))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

# Bounds used by the non-linear optimiser. They are wide enough to accommodate
# every parameter combination ever observed on a real liquid equity smile and
# tight enough to keep the Trust-Region Reflective solver well behaved.
_SVI_LOWER_BOUNDS = (-1.0, 0.0, -0.999, -2.0, _SIGMA_FLOOR)
_SVI_UPPER_BOUNDS = (5.0, 10.0, 0.999, 2.0, 5.0)

# Minimum number of distinct strikes required to identify five SVI parameters.
# Below this threshold the calibration is structurally under-determined and the
# function refuses to fit rather than return a misleadingly precise object.
_MIN_STRIKES_FOR_FIT: int = 5


class SVICalibrationError(RuntimeError):
    """Raised when SVI calibration cannot return a usable parameter set.

    The exception is the signal consumed by the higher-level orchestrator
    (Task #4) to route the affected slice into the chain-proxy fallback
    rather than expose a silently invalid surface to downstream code.
    """


def _vega_proxy_weights(
    bid_asks: Optional[np.ndarray],
    n_strikes: int,
) -> np.ndarray:
    """Per-observation weights for the calibration residuals.

    The objective minimises a weighted sum of squared deviations between
    market and model implied volatilities. The weight on each strike
    reflects the confidence with which the market quote is known: a
    tight bid–ask spread corresponds to a liquid, informative quote and
    receives a higher weight; a wide spread is treated as uninformative
    and is down-weighted accordingly. When the chain does not carry
    bid–ask information the weights collapse to uniform.

    Parameters
    ----------
    bid_asks : ndarray or None
        Absolute bid–ask spread on the *price* of each option, in the
        same units used by the chain. When ``None`` the function
        returns a uniform weight vector of length ``n_strikes``.
    n_strikes : int
        Number of strikes in the slice.

    Returns
    -------
    ndarray
        Weight vector of length ``n_strikes``, normalised so that the
        weights sum to one. The normalisation has no effect on the
        location of the optimum but keeps the residual norm reported
        by the optimiser comparable across slices.
    """
    if bid_asks is None:
        return np.full(n_strikes, 1.0 / n_strikes)
    spreads = np.asarray(bid_asks, dtype=float)
    floor = np.maximum(np.nanmedian(spreads) * 0.10, 1.0e-6)
    raw = 1.0 / np.maximum(spreads, floor)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    total = raw.sum()
    if total <= 0.0:
        return np.full(n_strikes, 1.0 / n_strikes)
    return raw / total


def _quasi_explicit_seed(k: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float, float]:
    """De Marco–Martini quasi-explicit seed for the SVI parameters.

    The five-parameter SVI calibration is non-convex in (a, b, ρ, m, ς)
    but reduces to a *linear* least-squares problem in (a, d, c) once
    the pair (m, ς) is fixed, by virtue of the substitution
    ``c = b ς`` and ``d = b ς ρ``. The seeding procedure of De Marco
    and Martini (2009) exploits this structure: an outer one-dimensional
    search over the curvature parameter ς is performed (with ``m``
    co-determined as the log-moneyness of the empirical smile minimum),
    and for each candidate ς the inner linear problem is solved
    analytically. The resulting (a, b, ρ, m, ς) tuple is used as an
    initial guess for the full non-linear refinement.

    Parameters
    ----------
    k : ndarray
        Log-moneyness of the observed strikes.
    w : ndarray
        Observed total variance ``σ²·T`` at those strikes.

    Returns
    -------
    tuple of float
        Initial guess for (a, b, ρ, m, ς), already clipped to the
        admissible domain.
    """
    # Anchor the smile minimum at the log-moneyness of the empirical
    # variance minimum. This is a more robust choice than k = 0 because
    # liquid equity smiles need not be centred at the forward.
    m_seed = float(k[int(np.argmin(w))])

    def inner_sse(sigma_candidate: float) -> float:
        x = (k - m_seed) / sigma_candidate
        y = np.sqrt(x * x + 1.0)
        # Design matrix for the linear regression w ≈ a + d·x + c·y.
        X = np.column_stack([np.ones_like(x), x, y])
        try:
            beta, *_ = np.linalg.lstsq(X, w, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")
        residual = w - X @ beta
        return float(residual @ residual)

    # One-dimensional search over the curvature parameter. The bracket
    # [0.01, 1.0] spans the full range of empirically reasonable
    # curvatures for equity smiles.
    res = minimize_scalar(inner_sse, bounds=(0.01, 1.0), method="bounded",
                          options={"xatol": 1.0e-3})
    sigma_seed = float(res.x)

    # Recover (a, c, d) at the optimal ς and translate back to (a, b, ρ).
    x = (k - m_seed) / sigma_seed
    y = np.sqrt(x * x + 1.0)
    X = np.column_stack([np.ones_like(x), x, y])
    beta, *_ = np.linalg.lstsq(X, w, rcond=None)
    a_lin, d_lin, c_lin = float(beta[0]), float(beta[1]), float(beta[2])

    # Project the linear solution onto the admissible SVI cone:
    #   c ≥ 0, |d| ≤ c. Violations are absorbed by clipping and the
    #   non-linear refinement is responsible for the final positioning.
    c_clip = max(c_lin, 1.0e-4)
    d_clip = float(np.clip(d_lin, -c_clip + 1.0e-6, c_clip - 1.0e-6))
    b_seed = c_clip / sigma_seed
    rho_seed = d_clip / c_clip
    a_seed = max(a_lin, -b_seed * sigma_seed * float(np.sqrt(1.0 - rho_seed ** 2)) + 1.0e-8)

    # Final clip to the optimiser bounds.
    a_seed = float(np.clip(a_seed, _SVI_LOWER_BOUNDS[0], _SVI_UPPER_BOUNDS[0]))
    b_seed = float(np.clip(b_seed, _SVI_LOWER_BOUNDS[1] + 1.0e-6, _SVI_UPPER_BOUNDS[1]))
    rho_seed = float(np.clip(rho_seed, _SVI_LOWER_BOUNDS[2], _SVI_UPPER_BOUNDS[2]))
    m_seed = float(np.clip(m_seed, _SVI_LOWER_BOUNDS[3], _SVI_UPPER_BOUNDS[3]))
    sigma_seed = float(np.clip(sigma_seed, _SVI_LOWER_BOUNDS[4], _SVI_UPPER_BOUNDS[4]))

    return a_seed, b_seed, rho_seed, m_seed, sigma_seed


def fit_svi_slice(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    T: float,
    forward: float,
    bid_asks: Optional[np.ndarray] = None,
) -> tuple[SVIParams, dict]:
    """Calibrate one SVI slice to a vector of observed implied volatilities.

    The calibration minimises a weighted sum of squared deviations
    between the model and market implied volatilities, expressed in
    *volatility* units rather than total-variance units. Working in
    volatility units keeps the objective dimensionally comparable
    across slices of different maturities and gives equal economic
    weight to a one-volatility-point miss regardless of tenor.

    The procedure proceeds in two stages. First, a quasi-explicit seed
    is obtained by the De Marco–Martini one-dimensional search that
    exploits the partial linearity of the SVI problem in
    (a, b ς ρ, b ς). Second, the seed is refined by a bounded
    non-linear least-squares solver (SciPy's Trust-Region Reflective)
    operating directly on the five physical parameters with the
    domain constraints implemented as box bounds. The two-stage
    approach reliably locates the global optimum on liquid chains and
    avoids the well-documented multimodality of single-stage SVI
    calibration.

    Parameters
    ----------
    strikes : ndarray
        Strike prices of the observed options. Must be strictly
        positive and have at least ``_MIN_STRIKES_FOR_FIT`` distinct
        entries.
    implied_vols : ndarray
        Market mid implied volatilities, annualised, in decimal form.
        Strictly positive entries.
    T : float
        Time to maturity in years; strictly positive.
    forward : float
        Forward price at the slice maturity, ``F = S exp((r − q) T)``.
        Strictly positive. The log-moneyness of each strike is
        computed as ``k = ln(K / F)``.
    bid_asks : ndarray or None, optional
        Absolute bid–ask spread on the option *prices*. When supplied,
        the residuals are weighted by an inverse-spread proxy for
        information content; when omitted, uniform weights are used.

    Returns
    -------
    SVIParams
        Calibrated parameter tuple.
    dict
        Calibration metadata containing:

        * ``rmse`` — root-mean-square deviation between fitted and
          observed implied volatilities, in volatility points.
        * ``max_resid`` — maximum absolute residual, in volatility
          points.
        * ``n_strikes`` — number of strikes used in the calibration.
        * ``k_range`` — tuple ``(k_min, k_max)`` of log-moneyness
          coverage.
        * ``converged`` — boolean status reported by the optimiser.

    Raises
    ------
    SVICalibrationError
        Raised when the input does not satisfy the structural
        preconditions of the calibration (insufficient strike count,
        non-finite or non-positive entries) or when the non-linear
        optimiser fails to converge to a finite parameter tuple.
    """
    strikes = np.asarray(strikes, dtype=float)
    implied_vols = np.asarray(implied_vols, dtype=float)
    if bid_asks is not None:
        bid_asks = np.asarray(bid_asks, dtype=float)

    if T <= 0.0:
        raise SVICalibrationError(f"Tenor T must be strictly positive; received {T}")
    if forward <= 0.0:
        raise SVICalibrationError(f"Forward must be strictly positive; received {forward}")
    if strikes.shape != implied_vols.shape:
        raise SVICalibrationError(
            f"Shape mismatch: strikes {strikes.shape} vs implied_vols {implied_vols.shape}"
        )
    if not np.isfinite(strikes).all() or not np.isfinite(implied_vols).all():
        raise SVICalibrationError("strikes and implied_vols must be finite")
    if (strikes <= 0.0).any():
        raise SVICalibrationError("strikes must be strictly positive")
    if (implied_vols <= 0.0).any():
        raise SVICalibrationError("implied_vols must be strictly positive")
    if strikes.size < _MIN_STRIKES_FOR_FIT:
        raise SVICalibrationError(
            f"At least {_MIN_STRIKES_FOR_FIT} strikes are required; received {strikes.size}"
        )

    # Sort by strike so that downstream consumers can rely on monotone k.
    order = np.argsort(strikes)
    strikes = strikes[order]
    implied_vols = implied_vols[order]
    if bid_asks is not None:
        bid_asks = bid_asks[order]

    k = np.log(strikes / forward)
    w_obs = implied_vols ** 2 * T
    weights = _vega_proxy_weights(bid_asks, strikes.size)
    sqrt_weights = np.sqrt(weights)

    a0, b0, rho0, m0, sigma0 = _quasi_explicit_seed(k, w_obs)

    def residuals(theta: np.ndarray) -> np.ndarray:
        params = SVIParams(a=theta[0], b=theta[1], rho=theta[2], m=theta[3], sigma=theta[4])
        sigma_model = svi_implied_vol(k, T, params)
        return sqrt_weights * (sigma_model - implied_vols)

    try:
        result = least_squares(
            residuals,
            x0=np.array([a0, b0, rho0, m0, sigma0]),
            bounds=(_SVI_LOWER_BOUNDS, _SVI_UPPER_BOUNDS),
            method="trf",
            xtol=1.0e-10,
            ftol=1.0e-10,
            max_nfev=400,
        )
    except ValueError as exc:
        # Triggered when the seed or an intermediate iterate violates
        # the SVIParams domain checks. The seeding heuristic is robust
        # enough that this should be rare on real data.
        raise SVICalibrationError(f"Optimiser raised on intermediate iterate: {exc}") from exc

    if not result.success or not np.isfinite(result.x).all():
        raise SVICalibrationError(
            f"Optimiser did not converge: status={result.status}, message={result.message!r}"
        )

    params = SVIParams(a=float(result.x[0]), b=float(result.x[1]),
                       rho=float(result.x[2]), m=float(result.x[3]),
                       sigma=float(result.x[4]))

    # Residuals are reported in volatility points (per cent of vol), which
    # is the unit a desk reads natively. The weighting used inside the
    # optimiser is intentionally not applied here: the reported quality
    # statistics describe the unweighted fit.
    sigma_fit = svi_implied_vol(k, T, params)
    resid = sigma_fit - implied_vols
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    max_resid = float(np.max(np.abs(resid)))

    meta = {
        "rmse": rmse,
        "max_resid": max_resid,
        "n_strikes": int(strikes.size),
        "k_range": (float(k.min()), float(k.max())),
        "converged": bool(result.success),
    }
    return params, meta


# ---------------------------------------------------------------------------
# Arbitrage and data-quality gates
# ---------------------------------------------------------------------------

# Default thresholds used by the quality gate. The values reflect the
# empirical noise level of Yahoo option chains on liquid underlyings: a
# tighter root-mean-square fit than 1.5 volatility points is rare even
# on SPX, while a wider one is almost always a sign of a contaminated
# slice rather than a real exotic smile feature.
_QUALITY_DEFAULTS = {
    "min_strikes":           5,
    "max_bid_ask_pct":       0.25,   # bid-ask must be at most 25 % of mid
    "max_rmse_vol_points":   0.015,  # 1.5 volatility points
}

# Densities used by the Durrleman butterfly check. The grid spans a wide
# log-moneyness band so that the test catches violations both near the
# money and in the wings where the SVI parameterisation is most likely
# to misbehave.
_DURRLEMAN_GRID_LIMIT: float = 2.5
_DURRLEMAN_GRID_POINTS: int = 401

# Floor on total variance used inside the Durrleman density to avoid a
# 1 / w singularity in numerical evaluation. The constructor of
# SVIParams already enforces ``w(k) ≥ 0``; the floor only guards against
# the boundary case ``w(k) = 0`` and is set well below any plausible
# market value.
_W_FLOOR: float = 1.0e-12


def _svi_derivatives(k: np.ndarray, params: SVIParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First two derivatives of the SVI total-variance function.

    For raw SVI, ``w(k) = a + b{ρ(k − m) + sqrt[(k − m)² + ς²]}``, the
    first and second derivatives have closed-form expressions

        w'(k)  = b{ρ + (k − m) / R}
        w''(k) = b ς² / R³

    where ``R = sqrt[(k − m)² + ς²]``. Evaluating ``w``, ``w'`` and
    ``w''`` jointly in one pass keeps the Durrleman test efficient
    even on the dense grids used by the butterfly check.
    """
    sigma_eff = max(params.sigma, _SIGMA_FLOOR)
    shifted = k - params.m
    R = np.sqrt(shifted * shifted + sigma_eff * sigma_eff)
    w = params.a + params.b * (params.rho * shifted + R)
    w_prime = params.b * (params.rho + shifted / R)
    w_double = params.b * sigma_eff * sigma_eff / (R ** 3)
    return w, w_prime, w_double


def check_durrleman_butterfly(
    params: SVIParams,
    grid_limit: float = _DURRLEMAN_GRID_LIMIT,
    grid_points: int = _DURRLEMAN_GRID_POINTS,
) -> tuple[bool, str]:
    """Verify that an SVI slice is free of butterfly arbitrage.

    Durrleman (2010) showed that the absence of butterfly arbitrage in
    a smile parameterised by total variance ``w(k)`` is equivalent to
    the non-negativity, for every log-moneyness ``k``, of the function

        g(k) = ( 1 − k w'(k) / (2 w(k)) )²
             − ( w'(k)² / 4 ) ( 1 / w(k) + 1 / 4 )
             + w''(k) / 2.

    This quantity is, up to a multiplicative factor, the density of the
    risk-neutral distribution implied by the slice; its non-negativity
    is therefore both necessary and sufficient for that distribution to
    be a probability measure. The check is implemented numerically on
    a dense grid spanning a wide log-moneyness band, which is the
    standard practice for SVI; an analytical characterisation in terms
    of the parameters alone exists only as a set of necessary
    conditions.

    Parameters
    ----------
    params : SVIParams
        Calibrated SVI parameter tuple.
    grid_limit : float, optional
        Maximum absolute log-moneyness on the evaluation grid.
    grid_points : int, optional
        Number of equally spaced grid points used to evaluate ``g``.

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` if ``g(k) ≥ 0`` everywhere on the grid;
        ``(False, reason)`` otherwise, where ``reason`` quotes the
        location and magnitude of the worst violation.
    """
    k = np.linspace(-grid_limit, grid_limit, grid_points)
    w, w_p, w_pp = _svi_derivatives(k, params)
    w_safe = np.maximum(w, _W_FLOOR)
    term1 = (1.0 - k * w_p / (2.0 * w_safe)) ** 2
    term2 = (w_p ** 2 / 4.0) * (1.0 / w_safe + 0.25)
    g = term1 - term2 + 0.5 * w_pp
    min_g = float(np.min(g))
    if min_g < -1.0e-8:
        k_worst = float(k[int(np.argmin(g))])
        return False, (
            f"Durrleman density g(k) negative at k={k_worst:+.3f}: "
            f"g_min={min_g:+.3e}"
        )
    return True, ""


def check_wing_bounds(params: SVIParams, T: float) -> tuple[bool, str]:
    """Verify that an SVI slice satisfies the Roger Lee wing bounds.

    Lee (2004) established that the asymptotic slope of total implied
    variance with respect to log-moneyness is constrained by the
    absence of static arbitrage in the wings of the option price
    surface. For the raw SVI parameterisation the asymptotic slopes
    are ``b(1 + ρ)`` on the right and ``b(1 − ρ)`` on the left; the
    canonical necessary condition for absence of arbitrage is

        b T (1 + |ρ|) ≤ 4,

    equivalent to ``b (1 + |ρ|) ≤ 4 / T``. Violating this bound
    implies that the smile grows too quickly in at least one wing for
    the implied risk-neutral distribution to have a finite second
    moment under the relevant truncation, which in turn would admit
    arbitrage opportunities in the corresponding call price strip.

    Parameters
    ----------
    params : SVIParams
        Calibrated SVI parameter tuple.
    T : float
        Time to maturity of the slice in years.

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` if the wing bound is satisfied;
        ``(False, reason)`` with the violation margin otherwise.
    """
    if T <= 0.0:
        raise ValueError(f"Tenor T must be strictly positive; received {T}")
    lhs = params.b * T * (1.0 + abs(params.rho))
    if lhs > 4.0 + 1.0e-8:
        return False, (
            f"Roger Lee wing bound violated: bT(1+|rho|) = {lhs:.3f} > 4 "
            f"(b={params.b:.3f}, rho={params.rho:+.3f}, T={T:.3f})"
        )
    return True, ""


def quality_gate(
    strikes: np.ndarray,
    bid_asks: Optional[np.ndarray],
    fit_rmse: float,
    mids: Optional[np.ndarray] = None,
    *,
    min_strikes: int = _QUALITY_DEFAULTS["min_strikes"],
    max_bid_ask_pct: float = _QUALITY_DEFAULTS["max_bid_ask_pct"],
    max_rmse_vol_points: float = _QUALITY_DEFAULTS["max_rmse_vol_points"],
) -> tuple[bool, str]:
    """Assess whether a calibrated slice meets the data-quality bar.

    The quality gate is the second of the three filters that determine
    whether a calibrated slice is exposed to downstream pricing code.
    Whereas the arbitrage gates (Durrleman butterfly, Roger Lee wing
    bounds) check the mathematical admissibility of the calibrated
    parameter tuple, the quality gate checks the *informational*
    admissibility of the underlying market data and the closeness of
    the fit. The thresholds are conservative defaults rather than
    universal constants; they are exposed as keyword arguments so that
    a more permissive or more stringent regime may be configured per
    deployment.

    Three conditions are evaluated:

    * **Strike count.** A minimum number of strikes is required to
      identify the five free SVI parameters with any margin against
      overfitting. The default of five corresponds to a calibration
      that is exactly identified; values below this threshold cannot
      produce a stable smile and are rejected outright.
    * **Bid–ask tightness.** Each option's relative spread (absolute
      spread divided by mid price, when both are available) must lie
      below a configurable ceiling. Quotes wider than the ceiling are
      treated as stale or thin and are excluded from the count of
      informative observations.
    * **Fit quality.** The root-mean-square deviation of the fitted
      implied volatilities from the observed mids must be below a
      tolerance expressed in volatility points. The default of 1.5
      points reflects the typical noise level of mid quotes on liquid
      single names; a substantially larger residual is almost always
      diagnostic of a contaminated slice rather than of a genuine
      exotic smile feature.

    Parameters
    ----------
    strikes : ndarray
        Strike prices used in the calibration.
    bid_asks : ndarray or None
        Absolute bid–ask spread per strike, or ``None`` when the data
        source does not provide spreads.
    fit_rmse : float
        Root-mean-square calibration residual reported by
        :func:`fit_svi_slice`, in decimal volatility units.
    mids : ndarray or None, optional
        Option mid prices used to translate absolute spreads into
        relative spreads. When omitted, the bid–ask tightness check
        is skipped with a tag in the returned reason string.
    min_strikes, max_bid_ask_pct, max_rmse_vol_points
        Configurable thresholds; defaults are listed in the module
        constant ``_QUALITY_DEFAULTS``.

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` if every check passes; ``(False, reason)``
        otherwise, where ``reason`` describes the first failing
        condition.
    """
    n_strikes = int(np.asarray(strikes).size)
    if n_strikes < min_strikes:
        return False, f"insufficient strikes: {n_strikes} < {min_strikes}"

    if bid_asks is not None and mids is not None:
        spreads = np.asarray(bid_asks, dtype=float)
        mids_arr = np.asarray(mids, dtype=float)
        valid = (mids_arr > 0.0) & np.isfinite(spreads) & np.isfinite(mids_arr)
        if valid.any():
            rel = spreads[valid] / mids_arr[valid]
            informative = (rel <= max_bid_ask_pct).sum()
            if informative < min_strikes:
                return False, (
                    f"insufficient tight quotes: {informative} strikes "
                    f"with bid/ask <= {max_bid_ask_pct:.0%} of mid "
                    f"(need {min_strikes})"
                )

    if fit_rmse > max_rmse_vol_points:
        return False, (
            f"fit RMSE too large: {fit_rmse*100:.2f} vol points > "
            f"{max_rmse_vol_points*100:.2f} threshold"
        )

    return True, ""


# ---------------------------------------------------------------------------
# Single-slice surface object and fallback chain
# ---------------------------------------------------------------------------

# Constant volatility used as the terminal fallback when no chain data is
# usable for an underlying. Matches the value already employed by the
# Monte-Carlo pricer in ``src/pricing/monte_carlo.py`` so that the
# fallback regime is consistent across the two layers of the system.
DEFAULT_FALLBACK_VOL: float = 0.15


# Status taxonomy exposed by ``VolSliceSurface``. The values are part of the
# public contract of the module: downstream views badge surfaces according
# to these labels and analytics tags record them verbatim.
FIT_STATUS_SVI: str = "svi"
FIT_STATUS_PROXY: str = "proxy"
FIT_STATUS_FALLBACK: str = "fallback"


def nearest_strike_proxy(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    K_target: float,
) -> float:
    """Return the implied volatility of the listed strike closest to the target.

    The function implements the simplest possible single-point fallback
    when an SVI calibration is unavailable or has failed an admissibility
    check. The implied volatility returned is the one observed at the
    strike in the chain whose distance to ``K_target`` is minimal in
    log-moneyness; ties are broken by selecting the lower strike, an
    arbitrary but documented convention that keeps the function
    deterministic.

    The proxy is intentionally not a smooth function of ``K_target``: it
    is a step function with discontinuities at the midpoints between
    consecutive listed strikes. It should therefore only be invoked for
    occasional point queries (typically the strike of a single barrier),
    never as a substitute for a calibrated smile in calculations that
    require differentiability or path-dependent evaluation.

    Parameters
    ----------
    strikes : ndarray
        Listed option strikes, strictly positive.
    implied_vols : ndarray
        Observed implied volatilities at those strikes, strictly
        positive.
    K_target : float
        Strike at which the implied volatility is queried.

    Returns
    -------
    float
        Implied volatility at the closest listed strike.

    Raises
    ------
    ValueError
        If the inputs are empty, mismatched in shape, or non-positive.
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    ivs_arr = np.asarray(implied_vols, dtype=float)
    if strikes_arr.size == 0:
        raise ValueError("nearest_strike_proxy requires at least one strike")
    if strikes_arr.shape != ivs_arr.shape:
        raise ValueError(
            f"shape mismatch: strikes {strikes_arr.shape} vs ivs {ivs_arr.shape}"
        )
    if K_target <= 0.0:
        raise ValueError(f"K_target must be strictly positive; received {K_target}")
    if (strikes_arr <= 0.0).any() or (ivs_arr <= 0.0).any():
        raise ValueError("strikes and implied_vols must be strictly positive")

    distances = np.abs(np.log(strikes_arr / K_target))
    # Stable tie-break: lower strike wins because argmin returns the first
    # occurrence of the minimum.
    order = np.argsort(strikes_arr)
    return float(ivs_arr[order][int(np.argmin(distances[order]))])


class VolSliceSurface:
    """Volatility smile of a single listed expiry for a single underlying.

    The object exposes a single uniform interface, the method
    :meth:`sigma`, regardless of whether the underlying smile is
    represented by a calibrated SVI parameterisation, a chain-proxy
    fallback, or a constant-vol fallback. The choice of representation
    is recorded in :attr:`fit_status` and is exposed verbatim in the
    user interface so that the user can distinguish a quote derived
    from a stable arbitrage-free fit from one derived from a single
    listed strike or, in the worst case, a static default. This
    transparency is required by the no-silent-wrong-model directive
    that governs the project.

    Three internal representations coexist:

    * ``fit_status == "svi"``. A calibrated :class:`SVIParams` tuple is
      stored; :meth:`sigma` evaluates the analytical SVI formula at
      the requested strike. This is the only branch that yields a
      smooth, arbitrage-aware smile.
    * ``fit_status == "proxy"``. The raw chain (strikes and observed
      implied volatilities) is stored; :meth:`sigma` returns the
      observed implied volatility of the strike closest to the query
      in log-moneyness. This branch is used when SVI calibration is
      either structurally impossible (too few strikes) or has produced
      a parameter tuple that fails the arbitrage gates or the quality
      gate.
    * ``fit_status == "fallback"``. A single constant volatility is
      stored; :meth:`sigma` returns that volatility regardless of the
      query strike. This branch is the terminal fallback used when no
      usable chain data exists for the underlying at the requested
      tenor.

    The class is constructed exclusively through the class method
    :meth:`from_chain`, which implements the full fallback decision
    procedure. Direct construction is supported for testing and for
    composition by later stages (SSVI term-structure interpolation
    will build SVI-branch slices directly), but is not the primary
    entry point.
    """

    __slots__ = (
        "isin", "T", "forward",
        "fit_status", "reason",
        "_params",
        "_chain_strikes", "_chain_ivs",
        "_fallback_sigma",
        "n_strikes", "k_range", "rmse", "max_resid",
    )

    def __init__(
        self,
        isin: str,
        T: float,
        forward: float,
        fit_status: str,
        reason: str = "",
        *,
        params: Optional[SVIParams] = None,
        chain_strikes: Optional[np.ndarray] = None,
        chain_ivs: Optional[np.ndarray] = None,
        fallback_sigma: Optional[float] = None,
        n_strikes: int = 0,
        k_range: tuple[float, float] = (0.0, 0.0),
        rmse: Optional[float] = None,
        max_resid: Optional[float] = None,
    ) -> None:
        if T <= 0.0:
            raise ValueError(f"T must be strictly positive; received {T}")
        if forward <= 0.0:
            raise ValueError(f"forward must be strictly positive; received {forward}")
        if fit_status not in (FIT_STATUS_SVI, FIT_STATUS_PROXY, FIT_STATUS_FALLBACK):
            raise ValueError(
                f"fit_status must be one of {{svi, proxy, fallback}}; received {fit_status!r}"
            )

        if fit_status == FIT_STATUS_SVI and params is None:
            raise ValueError("fit_status='svi' requires a non-None SVIParams")
        if fit_status == FIT_STATUS_PROXY and (chain_strikes is None or chain_ivs is None):
            raise ValueError("fit_status='proxy' requires non-None chain_strikes and chain_ivs")
        if fit_status == FIT_STATUS_FALLBACK and fallback_sigma is None:
            raise ValueError("fit_status='fallback' requires a non-None fallback_sigma")

        object.__setattr__(self, "isin", str(isin))
        object.__setattr__(self, "T", float(T))
        object.__setattr__(self, "forward", float(forward))
        object.__setattr__(self, "fit_status", fit_status)
        object.__setattr__(self, "reason", str(reason))
        object.__setattr__(self, "_params", params)
        object.__setattr__(
            self, "_chain_strikes",
            np.asarray(chain_strikes, dtype=float) if chain_strikes is not None else None,
        )
        object.__setattr__(
            self, "_chain_ivs",
            np.asarray(chain_ivs, dtype=float) if chain_ivs is not None else None,
        )
        object.__setattr__(
            self, "_fallback_sigma",
            float(fallback_sigma) if fallback_sigma is not None else None,
        )
        object.__setattr__(self, "n_strikes", int(n_strikes))
        object.__setattr__(self, "k_range", (float(k_range[0]), float(k_range[1])))
        object.__setattr__(self, "rmse", float(rmse) if rmse is not None else None)
        object.__setattr__(self, "max_resid", float(max_resid) if max_resid is not None else None)

    # ------------------------------------------------------------------
    # Public evaluator
    # ------------------------------------------------------------------

    def sigma(self, K: float | np.ndarray) -> float | np.ndarray:
        """Implied volatility at a strike (or array of strikes).

        Dispatches according to :attr:`fit_status`: SVI for the
        calibrated branch, nearest-strike lookup for the proxy branch,
        and a constant for the fallback branch. The query strike is
        expressed in price units (same unit as the underlying's spot
        and forward); log-moneyness is computed internally as
        ``k = ln(K / F)``.

        Parameters
        ----------
        K : float or ndarray
            Strike at which the implied volatility is requested.

        Returns
        -------
        float or ndarray
            Annualised implied volatility, in decimal form, of the
            same shape as ``K``.
        """
        K_arr = np.asarray(K, dtype=float)
        if (K_arr <= 0.0).any() if K_arr.ndim else (K_arr <= 0.0):
            raise ValueError(f"strike(s) must be strictly positive; received {K}")

        if self.fit_status == FIT_STATUS_SVI:
            k = np.log(K_arr / self.forward)
            return svi_implied_vol(k, self.T, self._params)  # type: ignore[arg-type]

        if self.fit_status == FIT_STATUS_PROXY:
            if K_arr.ndim:
                return np.array([
                    nearest_strike_proxy(self._chain_strikes, self._chain_ivs, float(k_i))  # type: ignore[arg-type]
                    for k_i in K_arr
                ])
            return nearest_strike_proxy(
                self._chain_strikes, self._chain_ivs, float(K_arr)  # type: ignore[arg-type]
            )

        # Fallback branch: constant.
        constant = float(self._fallback_sigma)  # type: ignore[arg-type]
        return np.full_like(K_arr, constant) if K_arr.ndim else constant

    def sigma_at_moneyness(self, m: float | np.ndarray) -> float | np.ndarray:
        """Convenience evaluator parameterised by moneyness ``m = K / F``.

        Many display conventions (notably the 90 % and 110 % skew points
        used in the portfolio expander) are expressed in moneyness
        rather than absolute strike. This wrapper translates the
        moneyness query into the absolute strike consumed by
        :meth:`sigma`.
        """
        m_arr = np.asarray(m, dtype=float)
        return self.sigma(m_arr * self.forward)

    # ------------------------------------------------------------------
    # Factory implementing the fallback decision procedure
    # ------------------------------------------------------------------

    @classmethod
    def from_chain(
        cls,
        isin: str,
        T: float,
        forward: float,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        bid_asks: Optional[np.ndarray] = None,
        mids: Optional[np.ndarray] = None,
        *,
        fallback_sigma: float = DEFAULT_FALLBACK_VOL,
    ) -> "VolSliceSurface":
        """Construct a slice surface from an option chain, with fallback.

        The factory executes the full decision procedure that determines
        which of the three internal representations is appropriate for
        a given chain. The procedure is applied in the following order:

        1. **SVI calibration.** :func:`fit_svi_slice` is invoked. On
           success the calibrated parameter tuple is subjected, in
           order, to :func:`check_durrleman_butterfly`,
           :func:`check_wing_bounds`, and :func:`quality_gate`. If all
           three pass, the returned object is in the ``svi`` branch.
        2. **Chain-proxy fallback.** If the SVI calibration cannot be
           performed (e.g. fewer than the minimum number of strikes)
           or if any of the admissibility checks fail, the factory
           retains the raw chain and constructs a proxy-branch slice.
           The reason recorded on the slice quotes the failing check
           verbatim so that the user can interpret the badge.
        3. **Constant fallback.** If the chain is degenerate (empty
           or with no positive implied volatilities), the factory
           constructs a fallback-branch slice carrying the supplied
           ``fallback_sigma`` and a reason indicating that no chain
           data was available.

        Parameters
        ----------
        isin : str
            Identifier of the underlying.
        T : float
            Time to maturity of this slice, in years.
        forward : float
            Forward price ``F`` at the slice maturity. The log-moneyness
            convention is ``k = ln(K / F)``.
        strikes, implied_vols : ndarray
            Listed strikes and the observed mid implied volatilities at
            those strikes.
        bid_asks, mids : ndarray or None, optional
            Bid–ask spreads and mid prices of the options, used to
            weight the calibration residuals and to assess data quality.
            When ``None``, the corresponding checks degrade gracefully.
        fallback_sigma : float, optional
            Constant volatility used in the terminal fallback branch.
            Defaults to :data:`DEFAULT_FALLBACK_VOL`.
        """
        strikes_arr = np.asarray(strikes, dtype=float) if strikes is not None else np.array([])
        ivs_arr = np.asarray(implied_vols, dtype=float) if implied_vols is not None else np.array([])

        # Degenerate chain — proceed directly to the constant fallback.
        valid_mask = (strikes_arr > 0.0) & (ivs_arr > 0.0) & np.isfinite(strikes_arr) & np.isfinite(ivs_arr)
        strikes_arr = strikes_arr[valid_mask]
        ivs_arr = ivs_arr[valid_mask]
        if bid_asks is not None:
            bid_asks = np.asarray(bid_asks, dtype=float)[valid_mask]
        if mids is not None:
            mids = np.asarray(mids, dtype=float)[valid_mask]

        if strikes_arr.size == 0:
            return cls(
                isin=isin, T=T, forward=forward,
                fit_status=FIT_STATUS_FALLBACK,
                reason="no usable chain data for this (isin, tenor)",
                fallback_sigma=fallback_sigma,
            )

        # Attempt the SVI fit.
        params: Optional[SVIParams] = None
        meta: dict = {}
        svi_reason = ""
        try:
            params, meta = fit_svi_slice(
                strikes_arr, ivs_arr, T=T, forward=forward, bid_asks=bid_asks
            )
        except SVICalibrationError as exc:
            svi_reason = f"calibration failed: {exc}"

        if params is not None:
            bf_ok, bf_msg = check_durrleman_butterfly(params)
            wb_ok, wb_msg = check_wing_bounds(params, T)
            qg_ok, qg_msg = quality_gate(
                strikes_arr, bid_asks, fit_rmse=meta["rmse"], mids=mids
            )
            if bf_ok and wb_ok and qg_ok:
                return cls(
                    isin=isin, T=T, forward=forward,
                    fit_status=FIT_STATUS_SVI,
                    reason="",
                    params=params,
                    chain_strikes=strikes_arr,
                    chain_ivs=ivs_arr,
                    n_strikes=meta["n_strikes"],
                    k_range=meta["k_range"],
                    rmse=meta["rmse"],
                    max_resid=meta["max_resid"],
                )
            svi_reason = bf_msg or wb_msg or qg_msg

        # Proxy fallback when at least one strike survived but SVI
        # could not be produced or did not pass admissibility.
        k_range = (
            float(np.log(strikes_arr.min() / forward)),
            float(np.log(strikes_arr.max() / forward)),
        )
        return cls(
            isin=isin, T=T, forward=forward,
            fit_status=FIT_STATUS_PROXY,
            reason=svi_reason or "SVI fit not admissible; using nearest-strike proxy",
            chain_strikes=strikes_arr,
            chain_ivs=ivs_arr,
            n_strikes=int(strikes_arr.size),
            k_range=k_range,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.fit_status == FIT_STATUS_SVI:
            quality = f"rmse={self.rmse*100:.2f}vp" if self.rmse is not None else ""
        elif self.fit_status == FIT_STATUS_PROXY:
            quality = f"{self.n_strikes} strikes, k_range={self.k_range}"
        else:
            quality = f"sigma={self._fallback_sigma}"
        return (
            f"VolSliceSurface(isin={self.isin!r}, T={self.T:.3f}, "
            f"status={self.fit_status}, {quality})"
        )


# ---------------------------------------------------------------------------
# Term-structure assembly (Stage 2)
# ---------------------------------------------------------------------------
#
# The objects defined below assemble the per-slice SVI surfaces of Stage 1
# into a single, term-structure-consistent volatility surface. The
# assembly follows the standard linear-in-total-variance recipe of
# Gatheral (2006, chapter 3), under which the absence of calendar
# arbitrage is preserved between two slices whose total variance is
# monotone increasing in tenor at every log-moneyness. The recipe is
# preferred here to a full SSVI re-calibration because it preserves the
# per-slice quality already achieved in Stage 1 and avoids re-fitting a
# more constrained functional form to chains whose data quality varies
# considerably across the universe of underlyings.


def interpolate_total_variance(
    k: np.ndarray | float,
    T_query: float,
    T_left: float,
    params_left: SVIParams,
    T_right: float,
    params_right: SVIParams,
) -> np.ndarray | float:
    """Linear-in-total-variance interpolation between two calibrated slices.

    For a query tenor ``T`` that lies between two listed expiries
    ``T_left`` and ``T_right`` at which the surface has been calibrated,
    the total implied variance ``w(k, T)`` is defined as the
    convex combination

    .. math::

        w(k, T) = w(k, T_\\mathrm{left}) + \\alpha
                  \\bigl[ w(k, T_\\mathrm{right}) - w(k, T_\\mathrm{left}) \\bigr],
        \\qquad
        \\alpha = \\frac{T - T_\\mathrm{left}}{T_\\mathrm{right} - T_\\mathrm{left}},

    where ``w(k, T_\\bullet) = w_{SVI}(k; \\theta_\\bullet)`` denotes the
    raw SVI total variance evaluated at the calibrated parameter tuple
    of the corresponding slice. The construction is the simplest
    interpolator that preserves calendar arbitrage absence whenever the
    underlying slices themselves satisfy ``w_\\mathrm{left}(k)
    \\leq w_\\mathrm{right}(k)`` for every ``k`` — a condition that is
    verified at construction of the enclosing :class:`VolSurface` and
    flagged as a quality issue when it fails.

    Butterfly arbitrage absence at the intermediate tenor is *not*
    guaranteed by this construction even when the two endpoint slices
    individually satisfy the Durrleman condition, because the convex
    combination of two arbitrage-free smiles is not in general
    arbitrage-free under butterflies. In practice the violation, when
    it occurs, is small in magnitude and localised in moneyness; the
    surface is nevertheless audited by an evaluation-time check
    described in the methodology document, and the user is informed
    when the audit fails.

    Parameters
    ----------
    k : ndarray or float
        Log-moneyness ``k = ln(K / F)`` at which the total variance is
        evaluated.
    T_query : float
        Tenor at which the interpolated value is required. Must satisfy
        ``T_left <= T_query <= T_right``.
    T_left, T_right : float
        Tenors of the bracketing slices, with ``T_left < T_right``.
    params_left, params_right : SVIParams
        Calibrated SVI parameters of the bracketing slices.

    Returns
    -------
    ndarray or float
        Total implied variance at the requested ``(k, T_query)``.

    Raises
    ------
    ValueError
        If the tenors do not satisfy ``T_left < T_right`` or if
        ``T_query`` falls outside the bracketing interval.
    """
    if not (T_left < T_right):
        raise ValueError(
            f"Bracketing tenors must satisfy T_left < T_right; "
            f"received T_left={T_left}, T_right={T_right}"
        )
    if not (T_left - 1.0e-12 <= T_query <= T_right + 1.0e-12):
        raise ValueError(
            f"T_query={T_query} is not in the bracketing interval "
            f"[{T_left}, {T_right}]"
        )
    w_left = svi_total_variance(k, params_left)
    w_right = svi_total_variance(k, params_right)
    alpha = (T_query - T_left) / (T_right - T_left)
    return w_left + alpha * (w_right - w_left)


def extrapolate_atm_scaling(
    k: np.ndarray | float,
    T_query: float,
    T_anchor: float,
    params_anchor: SVIParams,
) -> np.ndarray | float:
    """Extrapolate the surface beyond the listed range by ATM-vol-flat scaling.

    When the query tenor falls outside the interval spanned by the
    listed expiries — either before the shortest or after the longest —
    no listed slice brackets it and an extrapolation is required. The
    convention adopted here, conventional in the absence of an
    independent term-structure model, is to hold the implied volatility
    at every log-moneyness constant in tenor, so that the entire smile
    of the anchor slice is reused unchanged when expressed in
    volatility units. The corresponding total variance scales linearly
    in tenor:

    .. math::

        \\sigma(k, T) = \\sigma(k, T_\\mathrm{anchor})
        \\qquad \\Longleftrightarrow \\qquad
        w(k, T) = w(k, T_\\mathrm{anchor}) \\cdot \\frac{T}{T_\\mathrm{anchor}}.

    The convention is conservative for the downward extrapolation
    (short tenors), under which the assumption that the smile of the
    anchor slice continues to apply tends to understate the very steep
    short-dated skew that listed markets typically exhibit. It is also
    conservative for the upward extrapolation (long tenors) on most
    underlyings, because empirical term structures of ATM implied
    volatility tend to flatten rather than fall as tenor grows; the
    vol-flat convention therefore avoids the silently incorrect choice
    of extrapolating the ATM variance linearly into a regime where the
    listed market disagrees. The user is informed by the surface
    status badge that any value returned through this path is
    extrapolated, not interpolated.

    Parameters
    ----------
    k : ndarray or float
        Log-moneyness ``k = ln(K / F)``.
    T_query : float
        Tenor at which the extrapolated value is required. Strictly
        positive.
    T_anchor : float
        Tenor of the anchor slice (typically the shortest listed
        expiry for ``T_query < T_min`` or the longest for
        ``T_query > T_max``). Strictly positive.
    params_anchor : SVIParams
        Calibrated SVI parameters of the anchor slice.

    Returns
    -------
    ndarray or float
        Total implied variance at ``(k, T_query)`` under the vol-flat
        extrapolation.

    Raises
    ------
    ValueError
        If ``T_query`` or ``T_anchor`` is non-positive.
    """
    if T_query <= 0.0:
        raise ValueError(f"T_query must be strictly positive; received {T_query}")
    if T_anchor <= 0.0:
        raise ValueError(f"T_anchor must be strictly positive; received {T_anchor}")
    w_anchor = svi_total_variance(k, params_anchor)
    return w_anchor * (T_query / T_anchor)


# ---------------------------------------------------------------------------
# Calendar-arbitrage verification
# ---------------------------------------------------------------------------

# Grid used by the calendar-monotonicity audit. The band [-1.5, 1.5] in
# log-moneyness covers the strike range over which any plausible barrier
# product would query the surface, with substantial headroom in both
# wings to surface violations that occur at extreme strikes even when
# the at-the-money region is well-behaved.
_CALENDAR_GRID_LIMIT: float = 1.5
_CALENDAR_GRID_POINTS: int = 121


def verify_calendar_monotone(
    slice_records: list[tuple[float, "SVIParams", "VolSliceSurface"]],
    grid_limit: float = _CALENDAR_GRID_LIMIT,
    grid_points: int = _CALENDAR_GRID_POINTS,
) -> list[dict]:
    """Audit the slice list for calendar-arbitrage monotonicity violations.

    The linear-in-total-variance interpolation used by
    :class:`VolSurface` preserves the absence of calendar arbitrage at
    every intermediate tenor whenever, at every log-moneyness, the
    total variance is non-decreasing between consecutive listed
    expiries. The audit evaluates this condition on a dense
    log-moneyness grid spanning the band most relevant to barrier
    products and reports the location and magnitude of every violation
    encountered.

    A violation is recorded when the total variance of the later slice
    is strictly smaller than that of the earlier slice at some
    log-moneyness ``k`` by more than a small numerical tolerance. The
    record captures the bracketing tenors, the offending ``k``, and the
    magnitude of the variance deficit. The violations are returned as
    a list of dictionaries; an empty list signals that the audit has
    passed.

    Parameters
    ----------
    slice_records : list of tuples ``(T, SVIParams, VolSliceSurface)``
        Sorted (by tenor) list of the calibrated slices that compose
        the surface.
    grid_limit : float, optional
        Maximum absolute log-moneyness in the audit grid.
    grid_points : int, optional
        Number of equally spaced grid points.

    Returns
    -------
    list of dict
        One entry per violated pair-and-strike; empty when the audit
        passes. Each entry carries the keys ``T_left``, ``T_right``,
        ``k_worst``, ``deficit``, where the deficit is positive when
        the condition is violated.
    """
    if len(slice_records) < 2:
        return []
    k_grid = np.linspace(-grid_limit, grid_limit, grid_points)
    violations: list[dict] = []
    for i in range(1, len(slice_records)):
        T_left, p_left, _ = slice_records[i - 1]
        T_right, p_right, _ = slice_records[i]
        w_left = svi_total_variance(k_grid, p_left)
        w_right = svi_total_variance(k_grid, p_right)
        diff = w_right - w_left
        min_diff = float(np.min(diff))
        if min_diff < -1.0e-10:
            k_worst = float(k_grid[int(np.argmin(diff))])
            violations.append({
                "T_left": float(T_left),
                "T_right": float(T_right),
                "k_worst": k_worst,
                "deficit": -min_diff,
            })
    return violations


# ---------------------------------------------------------------------------
# Surface status taxonomy
# ---------------------------------------------------------------------------
#
# Four values complete the badge taxonomy introduced in Stage 1. They are
# part of the public contract of the module: the user interface badges
# the surface query according to these labels and analytics output tags
# carry them verbatim.

SURFACE_STATUS_INTERPOLATED: str = "interpolated"
SURFACE_STATUS_EXTRAPOLATED: str = "extrapolated"
SURFACE_STATUS_SINGLE_SLICE: str = "single_slice"
SURFACE_STATUS_FALLBACK: str = "fallback"


# Numerical guards on the Dupire local-volatility output. The floor of
# one volatility point keeps the local vol economically positive in the
# presence of small numerical residues. The hard cap at two hundred
# volatility points is a *production safety* threshold that suppresses
# only genuine numerical explosions: empirically observed Dupire local
# volatilities on real Yahoo surfaces can comfortably exceed one hundred
# per cent in the deep out-of-the-money put wing of low-volatility names
# without indicating any pathology, and clipping at that level would
# silently suppress an economically meaningful feature of the surface.
# Values outside [floor, hard cap] are clipped and counted by
# :attr:`VolSurface.local_vol_clip_count`.
#
# Two further thresholds, weaker than the hard cap, raise a *warning*
# rather than a clip. They signal regimes in which the local volatility
# is economically large enough that the user should be informed when
# such values participate in a product's pricing path:
#
# * **SIGMA_LV_WARNING** — local volatilities above the threshold are
#   recorded as warning events. The default of one hundred per cent is
#   the level at which the dealer practitioner literature typically
#   describes the local volatility as 'extreme', not because it is
#   spurious but because it dominates the risk-neutral dynamics in the
#   region where it applies.
# * **LV_IV_RATIO_WARNING** — local-to-implied volatility ratios above
#   the threshold are recorded similarly. The ratio reaches three or
#   above only on names whose surface exhibits a particularly steep
#   put-wing skew, in which case the Dupire amplification factor is at
#   the upper end of empirically defensible values.
#
# Both warnings are informational and do not modify the returned local
# volatility. An optional conservative mode is provided through the
# ``damping_cap`` argument of :meth:`VolSurface.local_volatility`,
# which applies a second, tighter clip on top of the production hard
# cap; the resulting marks are labelled in the user interface as a
# 'damped local-vol scenario' rather than as the pure Dupire output.
SIGMA_LV_FLOOR: float = 0.01
SIGMA_LV_HARD_CAP: float = 2.00         # 200 % — production safety threshold
SIGMA_LV_WARNING: float = 1.00          # 100 % — flagged but not clipped
LV_IV_RATIO_WARNING: float = 3.00       # local/implied ratio threshold

# Maximum number of warning events retained in the per-surface event
# buffer. Beyond this size, only the counter is incremented; the buffer
# itself stores a representative sample of the most extreme events so
# that the user interface can surface concrete examples.
_LV_WARNING_BUFFER_SIZE: int = 16

# Backwards compatibility: the previous ``SIGMA_LV_CAP`` name remains
# exposed but now resolves to the same value as the hard cap.
SIGMA_LV_CAP: float = SIGMA_LV_HARD_CAP


class VolSurface:
    """Term-structure-consistent implied volatility surface of one underlying.

    The object exposes a single uniform interface, the method
    :meth:`sigma`, which accepts a strike ``K`` and a tenor ``T`` and
    returns the corresponding annualised implied volatility. Internally
    the surface dispatches between four regimes, recorded in the
    surface status taxonomy:

    * **interpolated** — the query tenor sits strictly between two of
      the listed expiries at which the surface was calibrated. The
      total variance at the query is the linear-in-total-variance
      interpolation between the bracketing slices, as defined by
      :func:`interpolate_total_variance`.
    * **extrapolated** — the query tenor lies outside the convex hull
      of the listed expiries but at least one calibrated slice is
      available. The total variance at the query is obtained by
      vol-flat scaling of the appropriate anchor slice (the shortest
      listed expiry for ``T < T_min``, the longest for ``T > T_max``),
      as defined by :func:`extrapolate_atm_scaling`.
    * **single_slice** — only one SVI slice survives the Stage 1
      arbitrage and quality gates. Every query is answered by
      vol-flat scaling of that single slice. The status is reported
      separately from ``extrapolated`` because the surface provides
      no genuine term-structure information.
    * **fallback** — no SVI slice is available. The surface returns a
      configurable constant volatility regardless of the query, and
      the user interface is informed that no calibrated information
      underlies the result.

    The class is constructed exclusively through the class method
    :meth:`from_slice_map`, which extracts the SVI-branch slices of a
    :class:`VolSliceSurface` dictionary, sorts them by tenor, and
    populates the surface. Direct construction is supported for
    composition by Stage 3 code that builds surfaces from other
    sources but is not the primary entry point.
    """

    __slots__ = (
        "isin", "forward",
        "_slice_records",          # list[tuple[float, SVIParams, VolSliceSurface]]
        "_t_min", "_t_max",
        "_fallback_sigma",
        "n_svi_slices",
        "calendar_violations",     # list[dict]; populated by Task #12
        "local_vol_clip_count",    # int; hard-cap or floor clip events
        "local_vol_warning_count", # int; warning-threshold breaches
        "local_vol_warning_events", # list[dict]; representative warning loci
    )

    def __init__(
        self,
        isin: str,
        forward: float,
        slice_records: list[tuple[float, SVIParams, "VolSliceSurface"]],
        fallback_sigma: float = DEFAULT_FALLBACK_VOL,
    ) -> None:
        if forward <= 0.0:
            raise ValueError(f"forward must be strictly positive; received {forward}")
        if fallback_sigma <= 0.0:
            raise ValueError(
                f"fallback_sigma must be strictly positive; received {fallback_sigma}"
            )
        # Sort defensively in case the caller did not sort by tenor.
        records = sorted(
            [(float(T), p, s) for (T, p, s) in slice_records],
            key=lambda triple: triple[0],
        )
        # Validate strict tenor ordering: two slices at exactly the same
        # tenor would render the interpolation ill-posed.
        for i in range(1, len(records)):
            if records[i][0] <= records[i - 1][0]:
                raise ValueError(
                    f"slice tenors must be strictly increasing; received "
                    f"{records[i - 1][0]} and {records[i][0]}"
                )

        object.__setattr__(self, "isin", str(isin))
        object.__setattr__(self, "forward", float(forward))
        object.__setattr__(self, "_slice_records", records)
        object.__setattr__(self, "_t_min", records[0][0] if records else 0.0)
        object.__setattr__(self, "_t_max", records[-1][0] if records else 0.0)
        object.__setattr__(self, "_fallback_sigma", float(fallback_sigma))
        object.__setattr__(self, "n_svi_slices", len(records))
        # Audit calendar-arbitrage monotonicity across adjacent slices. The
        # audit is informational: violations do not block construction of
        # the surface, but they are recorded so that the user interface
        # can badge the affected term-structure regions with a quality
        # warning. Where the audit fails the linear-in-w interpolation
        # may admit a small calendar-arbitrage opportunity at the
        # specific (k, T) of the violation, but the surface remains
        # well-defined and usable for monitoring purposes.
        object.__setattr__(
            self, "calendar_violations",
            verify_calendar_monotone(records),
        )
        # Stage 3B diagnostic: number of local-volatility clip events
        # accumulated by ``local_volatility`` over the surface's lifetime.
        # Counts every (k, T) cell where the Dupire output was clipped to
        # the admissible range. Reset by the caller if a per-product or
        # per-evaluation count is required.
        object.__setattr__(self, "local_vol_clip_count", 0)
        # Stage 3B warning diagnostics: events where the Dupire output
        # lay inside the admissible range but exceeded the informational
        # warning thresholds. The counter is incremented for every
        # warning regardless of buffer state; the event list retains a
        # representative sample (the most extreme by sigma_LV) up to
        # ``_LV_WARNING_BUFFER_SIZE`` entries.
        object.__setattr__(self, "local_vol_warning_count", 0)
        object.__setattr__(self, "local_vol_warning_events", [])

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def surface_status_at(self, T: float) -> tuple[str, str]:
        """Surface status and reason at a single query tenor.

        Returns
        -------
        tuple of (str, str)
            ``(status, reason)`` where ``status`` is one of
            :data:`SURFACE_STATUS_INTERPOLATED`,
            :data:`SURFACE_STATUS_EXTRAPOLATED`,
            :data:`SURFACE_STATUS_SINGLE_SLICE`,
            :data:`SURFACE_STATUS_FALLBACK`, and ``reason`` is an empty
            string for the interpolated and single-slice cases or a
            human-readable explanation otherwise.
        """
        if not self._slice_records:
            return (SURFACE_STATUS_FALLBACK,
                    "no SVI slice available; constant fallback in use")
        if len(self._slice_records) == 1:
            T_only = self._slice_records[0][0]
            return (SURFACE_STATUS_SINGLE_SLICE,
                    f"only one calibrated slice at T={T_only:.3f}; "
                    f"every query is vol-flat scaled from this slice")
        if T < self._t_min:
            return (SURFACE_STATUS_EXTRAPOLATED,
                    f"T={T:.3f} is shorter than shortest listed expiry "
                    f"T={self._t_min:.3f}; extrapolated by vol-flat scaling")
        if T > self._t_max:
            return (SURFACE_STATUS_EXTRAPOLATED,
                    f"T={T:.3f} exceeds longest listed expiry "
                    f"T={self._t_max:.3f}; extrapolated by vol-flat scaling")
        return (SURFACE_STATUS_INTERPOLATED, "")

    # ------------------------------------------------------------------
    # Public evaluator
    # ------------------------------------------------------------------

    def sigma(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """Implied volatility at (strike, tenor).

        Dispatches according to the status returned by
        :meth:`surface_status_at`. The strike is expressed in price
        units (same unit as the underlying's spot and forward) and the
        log-moneyness ``k = ln(K / F)`` is computed internally.

        Parameters
        ----------
        K : float or ndarray
            Query strike (or array of strikes).
        T : float
            Query tenor, in years. Strictly positive.

        Returns
        -------
        float or ndarray
            Annualised implied volatility, in decimal form, of the
            same shape as ``K``.
        """
        if T <= 0.0:
            raise ValueError(f"T must be strictly positive; received {T}")
        K_arr = np.asarray(K, dtype=float)
        if (K_arr <= 0.0).any() if K_arr.ndim else (K_arr <= 0.0):
            raise ValueError(f"strike(s) must be strictly positive; received {K}")

        # Fallback branch: no SVI slices at all.
        if not self._slice_records:
            constant = self._fallback_sigma
            return np.full_like(K_arr, constant) if K_arr.ndim else float(constant)

        k = np.log(K_arr / self.forward)

        # Single-slice branch: always vol-flat extrapolate from the sole slice.
        if len(self._slice_records) == 1:
            T_anchor, params_anchor, _ = self._slice_records[0]
            w = extrapolate_atm_scaling(k, T, T_anchor, params_anchor)
            sigma_val = np.sqrt(np.asarray(w) / T) if np.ndim(w) else float(np.sqrt(float(w) / T))
            return sigma_val

        # Extrapolation outside the listed range.
        if T < self._t_min:
            T_anchor, params_anchor, _ = self._slice_records[0]
            w = extrapolate_atm_scaling(k, T, T_anchor, params_anchor)
        elif T > self._t_max:
            T_anchor, params_anchor, _ = self._slice_records[-1]
            w = extrapolate_atm_scaling(k, T, T_anchor, params_anchor)
        else:
            # Interpolation between bracketing listed slices.
            # Locate the right bracket by linear scan; the number of
            # slices per underlying is small enough (tens at most) that
            # a more elaborate search structure is unwarranted.
            T_left = params_left = T_right = params_right = None
            for i in range(1, len(self._slice_records)):
                if self._slice_records[i - 1][0] <= T <= self._slice_records[i][0]:
                    T_left = self._slice_records[i - 1][0]
                    params_left = self._slice_records[i - 1][1]
                    T_right = self._slice_records[i][0]
                    params_right = self._slice_records[i][1]
                    break
            assert T_left is not None, "bracket location must succeed inside [T_min, T_max]"
            w = interpolate_total_variance(k, T, T_left, params_left, T_right, params_right)

        return np.sqrt(np.asarray(w) / T) if np.ndim(w) else float(np.sqrt(float(w) / T))

    def sigma_at_moneyness(self, m: float | np.ndarray, T: float) -> float | np.ndarray:
        """Convenience evaluator parameterised by moneyness ``m = K / F``."""
        m_arr = np.asarray(m, dtype=float)
        return self.sigma(m_arr * self.forward, T)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_slice_map(
        cls,
        isin: str,
        slice_map: dict[str, "VolSliceSurface"],
        *,
        fallback_sigma: float = DEFAULT_FALLBACK_VOL,
    ) -> "VolSurface":
        """Assemble a VolSurface from a ``{expiry_iso: VolSliceSurface}`` map.

        Only :data:`FIT_STATUS_SVI` slices participate in the term-
        structure interpolation; slices in the proxy or fallback
        branches are excluded. The forward of the resulting surface is
        inherited from any participating slice; under the Stage 1
        simplification *F = S* every slice for a given underlying
        carries the same forward, so the choice is unambiguous. When
        no SVI slice is present, the surface is constructed in the
        fallback regime and a constant volatility is returned for every
        query.

        Parameters
        ----------
        isin : str
            Identifier of the underlying.
        slice_map : dict
            Mapping ``{expiry_iso: VolSliceSurface}`` as produced by
            :meth:`MarketDataEngine.build_vol_surface_map`.
        fallback_sigma : float, optional
            Constant volatility used in the fallback regime.

        Returns
        -------
        VolSurface
        """
        svi_records: list[tuple[float, SVIParams, "VolSliceSurface"]] = []
        forward: Optional[float] = None
        for _expiry, slice_surface in (slice_map or {}).items():
            if slice_surface.fit_status != FIT_STATUS_SVI:
                continue
            svi_records.append((slice_surface.T, slice_surface._params, slice_surface))  # type: ignore[arg-type]
            if forward is None:
                forward = slice_surface.forward

        if forward is None:
            # No SVI slice survives. We still need a positive forward to
            # satisfy the constructor; any value is admissible because
            # the fallback branch ignores it. Recover one from any
            # available slice (proxy or fallback) if possible; otherwise
            # use unity.
            for slice_surface in (slice_map or {}).values():
                if slice_surface.forward > 0.0:
                    forward = slice_surface.forward
                    break
            if forward is None:
                forward = 1.0

        return cls(isin=isin, forward=forward,
                   slice_records=svi_records, fallback_sigma=fallback_sigma)

    # ------------------------------------------------------------------
    # Dupire local volatility
    # ------------------------------------------------------------------

    def _surface_partials(
        self, k: np.ndarray, T: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(w, dw_dT, dw_dk, d2w_dk2)`` at log-moneyness ``k`` and tenor ``T``.

        The four quantities are required by the Dupire identity. Each is
        derived analytically from the slice-level SVI calibrations and
        the linear-in-total-variance term-structure assembly of
        Section 6 of the methodology document.

        The partials adopt three different forms according to the
        regime in which the query tenor lies:

        * **Interpolated** (``T_left <= T <= T_right``): the total
          variance is the convex combination
          ``w = (1-alpha) w_left + alpha w_right`` with
          ``alpha = (T - T_left) / (T_right - T_left)``. The
          calendar derivative is therefore piecewise constant,
          ``dw_dT = (w_right - w_left) / (T_right - T_left)``, while
          the strike derivatives are linearly combined.
        * **Extrapolated / single-slice** (``T`` outside the listed
          range, or only one SVI slice): the total variance scales
          linearly in tenor from the anchor slice,
          ``w(k, T) = w(k, T_anchor) * (T / T_anchor)``. The strike
          derivatives scale by the same factor and the calendar
          derivative is constant in ``T``.
        * **Fallback** (no SVI slices): Dupire is not well defined.
          The caller is expected to short-circuit on
          :attr:`n_svi_slices == 0` and not invoke this routine.

        Parameters
        ----------
        k : ndarray
            Log-moneyness at which the partials are evaluated.
        T : float
            Query tenor, strictly positive.

        Returns
        -------
        tuple of ndarray
            ``(w, dw_dT, dw_dk, d2w_dk2)``, all of shape compatible
            with ``k``.
        """
        if self.n_svi_slices == 0:
            raise ValueError(
                "Dupire partials are undefined for a surface with no SVI slices"
            )

        k_arr = np.asarray(k, dtype=float)

        if self.n_svi_slices == 1:
            T_anchor, params_anchor, _ = self._slice_records[0]
            w_anchor, dw_dk_a, d2w_dk2_a = _svi_derivatives(k_arr, params_anchor)
            scale = T / T_anchor
            w        = w_anchor * scale
            dw_dT    = w_anchor / T_anchor                       # constant in T
            dw_dk    = dw_dk_a * scale
            d2w_dk2  = d2w_dk2_a * scale
            return w, np.broadcast_to(dw_dT, k_arr.shape).copy(), dw_dk, d2w_dk2

        # Multi-slice path.
        if T < self._t_min:
            T_anchor, params_anchor, _ = self._slice_records[0]
            w_a, dw_a, d2w_a = _svi_derivatives(k_arr, params_anchor)
            scale = T / T_anchor
            return (w_a * scale,
                    np.broadcast_to(w_a / T_anchor, k_arr.shape).copy(),
                    dw_a * scale,
                    d2w_a * scale)
        if T > self._t_max:
            T_anchor, params_anchor, _ = self._slice_records[-1]
            w_a, dw_a, d2w_a = _svi_derivatives(k_arr, params_anchor)
            scale = T / T_anchor
            return (w_a * scale,
                    np.broadcast_to(w_a / T_anchor, k_arr.shape).copy(),
                    dw_a * scale,
                    d2w_a * scale)

        # Strictly interpolated regime: locate the bracketing pair.
        T_left = params_left = T_right = params_right = None
        for i in range(1, len(self._slice_records)):
            if self._slice_records[i - 1][0] <= T <= self._slice_records[i][0]:
                T_left = self._slice_records[i - 1][0]
                params_left = self._slice_records[i - 1][1]
                T_right = self._slice_records[i][0]
                params_right = self._slice_records[i][1]
                break
        assert T_left is not None

        w_l, dw_l, d2w_l = _svi_derivatives(k_arr, params_left)    # type: ignore[arg-type]
        w_r, dw_r, d2w_r = _svi_derivatives(k_arr, params_right)   # type: ignore[arg-type]
        dT = T_right - T_left
        alpha = (T - T_left) / dT
        w        = (1.0 - alpha) * w_l + alpha * w_r
        dw_dT    = (w_r - w_l) / dT                                # vector in k
        dw_dk    = (1.0 - alpha) * dw_l + alpha * dw_r
        d2w_dk2  = (1.0 - alpha) * d2w_l + alpha * d2w_r
        return w, dw_dT, dw_dk, d2w_dk2

    def local_volatility(
        self,
        K: float | np.ndarray,
        T: float,
        damping_cap: Optional[float] = None,
    ) -> float | np.ndarray:
        """Dupire local volatility at (strike, tenor).

        The local volatility is computed by the Dupire identity in
        total-variance form (Gatheral 2006, equation 1.27):

        .. math::

            \\sigma_\\mathrm{LV}^2(K, T) = \\frac{\\partial_T w}
            {1 - \\dfrac{k}{w} \\partial_k w
                 + \\dfrac{1}{4}\\!\\left(-\\dfrac{1}{4} - \\dfrac{1}{w} + \\dfrac{k^2}{w^2}\\right)
                   (\\partial_k w)^2
                 + \\dfrac{1}{2} \\partial_k^2 w}.

        The output is the annualised local volatility in decimal form
        at the supplied strike and tenor; the log-moneyness ``k``
        consumed by the formula is computed internally as
        ``k = ln(K / F)`` against the surface's forward.

        Output range, clipping, and warnings
        ------------------------------------
        Local volatility, derived from an interpolated implied
        surface that is *not* itself guaranteed butterfly-arbitrage-
        free at every intermediate tenor, occasionally produces
        values that are mathematically valid in form but not
        economically meaningful in size: a non-positive numerator
        signals a calendar-arbitrage violation, a non-positive
        denominator signals a butterfly-arbitrage violation, and
        either condition gives a non-real square root. The
        implementation guards against these regimes through a
        layered policy.

        * A **hard cap** at :data:`SIGMA_LV_HARD_CAP` (two hundred
          per cent, by default) is applied unconditionally. Values
          above the cap or below :data:`SIGMA_LV_FLOOR` are clipped
          and counted by :attr:`local_vol_clip_count`. The hard cap
          is a production safety threshold: it suppresses genuine
          numerical explosions while leaving the economically
          meaningful regime untouched. In particular, local
          volatilities of one hundred to two hundred per cent are
          legitimate on the deep out-of-the-money put wing of
          low-implied-volatility names and are *not* clipped at the
          production cap.
        * A **warning threshold** at :data:`SIGMA_LV_WARNING` (one
          hundred per cent) and at a local-to-implied volatility
          ratio of :data:`LV_IV_RATIO_WARNING` (three) record an
          informational event without modifying the returned value.
          The counter :attr:`local_vol_warning_count` is incremented
          for each breach; up to :data:`_LV_WARNING_BUFFER_SIZE`
          representative events are retained on
          :attr:`local_vol_warning_events`. A warning is *not* a
          defect of the surface — it identifies a region in which
          the conditional dynamics of the underlying are
          economically extreme, and informs the user interface that
          the value should be surfaced as such rather than treated
          as a normal volatility input.
        * An optional **damping cap**, supplied through the
          ``damping_cap`` argument, applies a second, tighter clip
          on top of the production hard cap. The damped output is a
          *conservative scenario* rather than the pure Dupire mark,
          and is intended for use when mark-stability across days
          is preferred to theoretical purity. The argument is
          deliberately not given a default value: every caller must
          choose explicitly between the pure Dupire mode
          (``damping_cap=None``) and the damped mode.

        Interpretation
        --------------
        A local volatility well in excess of one hundred per cent at
        a deep out-of-the-money put strike should be read as a
        *conditional* statement: it is the instantaneous volatility
        that the risk-neutral measure assigns to the underlying when
        and if the underlying spot crosses into that region of
        strikes, given the calibrated implied surface. It is *not* a
        statement about the unconditional volatility of the
        underlying observed across all states of the world. The
        distinction is significant for the interpretation of a
        product's mark: a barrier-product fair value computed under
        such a local-vol surface reflects a steep conditional
        downside dynamics that is a feature of the listed implied
        smile, not an exotic assumption introduced by the pricer.

        Parameters
        ----------
        K : float or ndarray
            Strike at which the local volatility is requested. Must
            be strictly positive.
        T : float
            Tenor at which the local volatility is requested. Must
            be strictly positive.
        damping_cap : float or None, optional
            Optional second clip applied after the production hard
            cap. Use sparingly and label the resulting marks as a
            damped scenario.

        Returns
        -------
        float or ndarray
            Local volatility in decimal form, of the same shape as
            ``K``.

        Raises
        ------
        ValueError
            If ``T`` is non-positive, ``K`` is non-positive, or the
            surface is in the fallback regime (no SVI slice
            available). The fallback regime requires the caller to
            handle the situation outside of Dupire — typically by
            reverting to the constant-volatility fallback.
        """
        if T <= 0.0:
            raise ValueError(f"T must be strictly positive; received {T}")
        if self.n_svi_slices == 0:
            raise ValueError(
                "local_volatility is undefined when no SVI slice is available; "
                "the caller should detect surface_status_at(T)=='fallback' and "
                "revert to the constant-volatility fallback."
            )
        K_arr = np.asarray(K, dtype=float)
        if (K_arr <= 0.0).any() if K_arr.ndim else (K_arr <= 0.0):
            raise ValueError(f"strike(s) must be strictly positive; received {K}")

        k = np.log(K_arr / self.forward)
        w, dw_dT, dw_dk, d2w_dk2 = self._surface_partials(k, T)

        # Floor on w to prevent the 1/w terms from diverging. The
        # constructor enforces non-negativity of total variance at the
        # smile minimum on every slice; w can in principle be exactly
        # zero only at the minimum of a slice whose ``a`` parameter
        # was calibrated to the boundary of the admissible set, but
        # the floor remains a defensive guard.
        w_safe = np.maximum(w, _W_FLOOR)

        denominator = (
            1.0
            - (k / w_safe) * dw_dk
            + 0.25 * (-0.25 - 1.0 / w_safe + (k ** 2) / (w_safe ** 2)) * (dw_dk ** 2)
            + 0.5 * d2w_dk2
        )

        numerator = dw_dT

        # Compute the squared local volatility where the formula is
        # well-defined; record non-finite or non-positive intermediates
        # as floor clips.
        valid = (numerator > 0.0) & (denominator > 0.0) & np.isfinite(numerator) & np.isfinite(denominator)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_lv_sq = np.where(valid, numerator / denominator, np.nan)
        sigma_lv = np.where(np.isfinite(sigma_lv_sq) & (sigma_lv_sq > 0.0),
                            np.sqrt(sigma_lv_sq), np.nan)

        # Floor substitution for any non-finite Dupire output.
        floor_clip_mask = ~np.isfinite(sigma_lv)
        sigma_lv = np.where(np.isfinite(sigma_lv), sigma_lv, SIGMA_LV_FLOOR)

        # Production hard-cap clip: legitimate suppression of numerical
        # explosions only. Values in [SIGMA_LV_WARNING, SIGMA_LV_HARD_CAP]
        # are *not* clipped at this stage.
        hard_clip_mask = (sigma_lv > SIGMA_LV_HARD_CAP) | (sigma_lv < SIGMA_LV_FLOOR)
        sigma_lv = np.clip(sigma_lv, SIGMA_LV_FLOOR, SIGMA_LV_HARD_CAP)

        # Warning detection — does not modify the output. We compute the
        # implied volatility at the same (K, T) for the ratio test; the
        # total variance is already in hand from the partials helper.
        sigma_iv = np.sqrt(np.maximum(w_safe, 0.0) / T)
        with np.errstate(divide="ignore", invalid="ignore"):
            lv_iv_ratio = np.where(sigma_iv > 0.0, sigma_lv / sigma_iv, 0.0)
        warning_mask = (sigma_lv > SIGMA_LV_WARNING) | (lv_iv_ratio > LV_IV_RATIO_WARNING)
        n_warnings = int(np.count_nonzero(warning_mask))
        if n_warnings > 0:
            object.__setattr__(
                self, "local_vol_warning_count",
                int(getattr(self, "local_vol_warning_count", 0)) + n_warnings,
            )
            buffer = list(getattr(self, "local_vol_warning_events", []))
            if len(buffer) < _LV_WARNING_BUFFER_SIZE:
                # Retain the loudest unrecorded events first; bounded.
                warned_idx = np.flatnonzero(warning_mask if warning_mask.ndim else np.atleast_1d(warning_mask))
                # Sort by sigma_lv descending and keep top entries that fit.
                sigma_lv_flat = np.atleast_1d(sigma_lv).ravel()
                sigma_iv_flat = np.atleast_1d(sigma_iv).ravel()
                K_flat = np.atleast_1d(K_arr).ravel() if K_arr.ndim else np.atleast_1d(K_arr)
                order = warned_idx[np.argsort(-sigma_lv_flat[warned_idx])]
                for idx in order:
                    if len(buffer) >= _LV_WARNING_BUFFER_SIZE:
                        break
                    buffer.append({
                        "K": float(K_flat[idx] if idx < len(K_flat) else K_arr),
                        "T": float(T),
                        "sigma_lv": float(sigma_lv_flat[idx]),
                        "sigma_iv": float(sigma_iv_flat[idx]),
                        "ratio": float(sigma_lv_flat[idx] / sigma_iv_flat[idx]) if sigma_iv_flat[idx] > 0 else float("inf"),
                        "code": "LV_WARNING_EXTREME_PUT_WING" if (K_arr if not K_arr.ndim else K_flat[idx]) < self.forward else "LV_WARNING_EXTREME_CALL_WING",
                    })
            object.__setattr__(self, "local_vol_warning_events", buffer)

        # Optional damping cap (conservative scenario).
        if damping_cap is not None:
            if damping_cap <= 0.0:
                raise ValueError(
                    f"damping_cap must be strictly positive when provided; "
                    f"received {damping_cap}"
                )
            damp_mask = sigma_lv > damping_cap
            sigma_lv = np.minimum(sigma_lv, float(damping_cap))
            # Damping clips are *not* recorded in ``local_vol_clip_count``,
            # which is reserved for the production-safety hard cap. They
            # are an opt-in transformation rather than a numerical guard.

        # Aggregate clip count for the hard-cap and floor events only.
        all_clip_mask = floor_clip_mask | hard_clip_mask
        n_clips = int(np.count_nonzero(all_clip_mask))
        if n_clips > 0:
            object.__setattr__(
                self, "local_vol_clip_count",
                int(getattr(self, "local_vol_clip_count", 0)) + n_clips,
            )

        return sigma_lv if K_arr.ndim else float(sigma_lv)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._slice_records:
            qual = "fallback"
        elif len(self._slice_records) == 1:
            qual = f"1 SVI slice at T={self._t_min:.3f}"
        else:
            qual = f"{self.n_svi_slices} SVI slices, T in [{self._t_min:.3f}, {self._t_max:.3f}]"
        return f"VolSurface(isin={self.isin!r}, F={self.forward:.3f}, {qual})"


# ---------------------------------------------------------------------------
# Pricer integration (Stage 3 Substage A)
# ---------------------------------------------------------------------------


def build_product_vol_map(
    product_row,
    vol_surfaces: dict,
    fallback_vol_map: dict,
    valuation_date,
) -> tuple[dict, list[dict]]:
    """Resolve a per-underlying volatility map for one product.

    For each underlying referenced in ``product_row``, the function
    evaluates the corresponding :class:`VolSurface` at the strike of the
    product's downside barrier and at the product's residual maturity.
    The barrier strike per underlying is the absolute level
    ``initial_level * barrier_pct`` returned by
    :func:`src.reverse_convertible.barrier_levels`, which is the
    canonical barrier convention of the project. The residual maturity
    is the calendar distance between the supplied valuation date and
    the product's maturity date, expressed in years on a 365.25-day
    basis to match the convention adopted elsewhere in the analytics
    layer.

    For a European-barrier reverse convertible the resulting
    volatility is the strictly correct input to the constant-volatility
    Monte Carlo, because only the marginal distribution of the
    underlying at the product's maturity enters the valuation and that
    marginal is identified, under geometric Brownian motion, by the
    volatility at the corresponding strike. For American-barrier and
    autocallable products the substitution is a material improvement
    over the at-the-money input — the barrier-zone volatility is the
    economically relevant quantity for the dominant payoff feature —
    but the path-dependency of those payoffs is still mis-represented
    by the constant-volatility dynamics; the full correction requires
    the local-volatility Monte Carlo of Stage 3 Substage B.

    When the surface for a given underlying is unavailable, falls back
    to the static :data:`SURFACE_STATUS_FALLBACK` regime, or has no
    chain coverage at the barrier strike at the residual maturity, the
    function reverts to the corresponding entry of ``fallback_vol_map``
    and records the reason on the diagnostics list. The transparency
    requirement is non-negotiable: every resolution path is recorded
    so that the user interface can badge the affected product.

    Parameters
    ----------
    product_row : pd.Series
        One row of the portfolio dataframe. Must expose
        ``underlying_isins``, ``initial_levels``, ``barrier_pct``,
        ``maturity_date``. Other product fields are not consulted.
    vol_surfaces : dict
        Mapping ``{ isin: VolSurface }`` produced by
        :meth:`MarketDataEngine.build_vol_surfaces`. May be empty when
        no surface coverage is available; in that case every
        underlying resolves via the fallback.
    fallback_vol_map : dict
        Mapping ``{ isin: sigma }`` used when the surface cannot
        produce a value. Typically the implied or realised ATM map
        already in use by the existing pricer.
    valuation_date : pd.Timestamp or convertible
        As-of date for the maturity arithmetic.

    Returns
    -------
    vol_map : dict
        Mapping ``{ isin: sigma }`` consumed by the existing pricer
        machinery. One entry per underlying referenced in
        ``product_row``.
    diagnostics : list of dict
        One entry per underlying, each carrying ``isin``,
        ``K_barrier``, ``T``, ``sigma``, ``source`` (``"surface"`` or
        ``"fallback"``), and ``surface_status`` (the verbatim status
        returned by the surface, or ``"no_surface"`` when the
        underlying has no surface at all).
    """
    # Local import of the barrier-level helper to avoid a hard
    # dependency at module import time.
    from src.pricing.products.reverse_convertible import barrier_levels

    valuation_ts = pd.Timestamp(valuation_date) if valuation_date is not None else pd.Timestamp.today().normalize()
    maturity_ts = pd.Timestamp(product_row["maturity_date"])
    T_years = (maturity_ts - valuation_ts).days / 365.25
    if T_years <= 0.0:
        # Matured (or maturing today): no barrier risk left. The pricer
        # treats this case independently; we simply return the fallback
        # map unchanged so that no surface lookup is attempted at a
        # non-positive tenor.
        diagnostics = [
            {"isin": str(isin), "K_barrier": float("nan"), "T": T_years,
             "sigma": fallback_vol_map.get(isin, DEFAULT_FALLBACK_VOL),
             "source": "fallback",
             "surface_status": "tenor_non_positive"}
            for isin in product_row["underlying_isins"]
        ]
        return ({d["isin"]: d["sigma"] for d in diagnostics}, diagnostics)

    isins = list(product_row["underlying_isins"])
    initial_levels = product_row.get("initial_levels", isins)   # safe default for products without one
    barrier_pct = product_row.get("barrier_pct")
    K_barriers = barrier_levels(initial_levels, barrier_pct)

    vol_map: dict[str, float] = {}
    diagnostics: list[dict] = []

    for isin, K_barrier in zip(isins, K_barriers):
        K_barrier = float(K_barrier)
        surface = (vol_surfaces or {}).get(isin)
        if surface is None:
            sigma = float(fallback_vol_map.get(isin, DEFAULT_FALLBACK_VOL))
            diagnostics.append({
                "isin": str(isin), "K_barrier": K_barrier, "T": T_years,
                "sigma": sigma, "source": "fallback",
                "surface_status": "no_surface",
            })
            vol_map[isin] = sigma
            continue

        status, _ = surface.surface_status_at(T_years)
        if status == SURFACE_STATUS_FALLBACK:
            # Surface itself is in the fallback regime (no calibrated
            # slices). Use the legacy fallback map rather than the
            # surface's static default to preserve consistency with
            # the prior pricer behaviour on these underlyings.
            sigma = float(fallback_vol_map.get(isin, DEFAULT_FALLBACK_VOL))
            diagnostics.append({
                "isin": str(isin), "K_barrier": K_barrier, "T": T_years,
                "sigma": sigma, "source": "fallback",
                "surface_status": status,
            })
            vol_map[isin] = sigma
            continue

        try:
            sigma = float(surface.sigma(K_barrier, T_years))
        except Exception:
            sigma = float(fallback_vol_map.get(isin, DEFAULT_FALLBACK_VOL))
            diagnostics.append({
                "isin": str(isin), "K_barrier": K_barrier, "T": T_years,
                "sigma": sigma, "source": "fallback",
                "surface_status": "surface_query_error",
            })
            vol_map[isin] = sigma
            continue

        if not (0.0 < sigma < 5.0) or not np.isfinite(sigma):
            # Defensive: anomalous surface output is replaced by the
            # fallback rather than fed into the pricer.
            sigma_fb = float(fallback_vol_map.get(isin, DEFAULT_FALLBACK_VOL))
            diagnostics.append({
                "isin": str(isin), "K_barrier": K_barrier, "T": T_years,
                "sigma": sigma_fb, "source": "fallback",
                "surface_status": f"out_of_range:{sigma:.3f}",
            })
            vol_map[isin] = sigma_fb
            continue

        diagnostics.append({
            "isin": str(isin), "K_barrier": K_barrier, "T": T_years,
            "sigma": sigma, "source": "surface",
            "surface_status": status,
        })
        vol_map[isin] = sigma

    return vol_map, diagnostics
