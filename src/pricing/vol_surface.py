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
