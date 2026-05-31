"""Barrier observation styles for reverse-convertible-type products.

Two observation conventions are supported:

* **European** — the down-barrier is tested only at final fixing: a path is
  knocked in iff a terminal underlying level sits at or below its barrier.
* **Continuous / American** — the barrier is live throughout the product's
  life.  Between two simulated grid points a Geometric Brownian Motion can dip
  below the barrier and recover, so checking only the grid values systematically
  *under*-detects breaches.  The correction used here is the Brownian-bridge
  survival probability: conditional on the two log-levels at the ends of a step,
  the probability that a single-underlying bridge never touched a down-barrier
  ``B`` is

  .. math::
      1 - \\exp\\!\\Big(-\\tfrac{2\\,\\ln(S_a/B)\\,\\ln(S_b/B)}{\\sigma^2\\,\\Delta t}\\Big),

  for ``S_a, S_b > B`` (and zero if either endpoint is already at/below ``B``).

The continuous routine returns, per path, the probability of *never* knocking in
(the "survival" probability).  For multi-underlying products the implementation
multiplies marginal bridge survivals across underlyings, which is exact under
conditional independence of the within-step bridges and a tractable
approximation otherwise.  Working with this probability rather than a sampled
0/1 indicator keeps the valuation deterministic given the paths — so the
bump-and-reprice Greeks remain stable under common random numbers — and lowers
Monte Carlo variance (a Rao-Blackwellisation of the breach indicator).
"""
from __future__ import annotations

import numpy as np


def european_knock_in(terminal_prices: np.ndarray, barriers: np.ndarray) -> np.ndarray:
    """Worst-of European knock-in mask.

    Parameters
    ----------
    terminal_prices : np.ndarray, shape ``(n_paths, n_assets)``
        Underlying levels at final fixing.
    barriers : np.ndarray, shape ``(n_assets,)``
        Per-underlying down-barrier levels.

    Returns
    -------
    np.ndarray of bool, shape ``(n_paths,)`` — True where any underlying is at
    or below its barrier at maturity.
    """
    terminal_prices = np.asarray(terminal_prices, dtype=float)
    barriers = np.asarray(barriers, dtype=float)
    return (terminal_prices <= barriers[None, :]).any(axis=1)


def continuous_survival_prob_from_var(
    prices: np.ndarray,
    barriers: np.ndarray,
    step_var: np.ndarray,
) -> np.ndarray:
    """Probability of no knock-in, given the per-step diffusion variance.

    The model variance of each monitoring step — ``σ² Δt`` for the underlying
    over that step — is supplied directly as ``step_var`` rather than inferred,
    so the barrier observation uses exactly the variance the simulation engine
    injected when it generated ``prices``.

    Uses the Brownian-bridge correction step by step and combines across steps
    and underlyings.  A worst-of down-barrier knocks in if *any* underlying ever
    crosses *its* barrier, so the implemented survival probability is the
    product over both axes of the per-step marginal bridge survival factors.

    Parameters
    ----------
    prices : np.ndarray, shape ``(n_paths, n_points, n_assets)``
        Monitored underlying levels.
    barriers : np.ndarray, shape ``(n_assets,)``
        Per-underlying down-barrier levels.
    step_var : np.ndarray
        Diffusion variance of each monitoring interval. Accepted in two
        shapes:

        * ``(n_points - 1, n_assets)`` — constant per path, the regime
          consumed by the constant-volatility Monte Carlo.
        * ``(n_paths, n_points - 1, n_assets)`` — path-dependent, the
          regime produced by the local-volatility Monte Carlo. Each
          step's variance is sourced from the same ``sigma_step(S, t)``
          evaluator that generated the corresponding path step, which
          is the structural guarantee that the bridge and the path
          remain internally consistent.

    Returns
    -------
    np.ndarray of float in ``[0, 1]``, shape ``(n_paths,)`` — per-path
    probability of never knocking in.
    """
    prices   = np.asarray(prices, dtype=float)
    barriers = np.asarray(barriers, dtype=float)
    step_var = np.asarray(step_var, dtype=float)

    n_paths, n_points, n_assets = prices.shape
    if n_points < 2:
        # No interval to monitor — fall back to the endpoint check.
        return (~european_knock_in(prices[:, -1, :], barriers)).astype(float)

    B   = barriers[None, None, :]                       # (1,1,n_assets)
    S_a = prices[:, :-1, :]                             # (n_paths, n_steps, n_assets)
    S_b = prices[:, 1:, :]

    # Accept either (n_steps, n_assets) — constant per path — or the
    # path-dependent (n_paths, n_steps, n_assets) shape produced by the
    # local-volatility Monte Carlo. In the constant case we prepend a
    # singleton path axis so that the subsequent broadcasting against
    # (n_paths, n_steps, n_assets) endpoint arrays is identical.
    if step_var.ndim == 2:
        if step_var.shape != (n_points - 1, n_assets):
            raise ValueError(
                f"step_var of shape {step_var.shape} does not match the "
                f"expected (n_steps, n_assets) = ({n_points - 1}, {n_assets})"
            )
        var = step_var[None, :, :]                       # (1, n_steps, n_assets)
    elif step_var.ndim == 3:
        if step_var.shape != (n_paths, n_points - 1, n_assets):
            raise ValueError(
                f"step_var of shape {step_var.shape} does not match the "
                f"expected (n_paths, n_steps, n_assets) = "
                f"({n_paths}, {n_points - 1}, {n_assets})"
            )
        var = step_var
    else:
        raise ValueError(
            f"step_var must be 2-D (n_steps, n_assets) or 3-D "
            f"(n_paths, n_steps, n_assets); received ndim={step_var.ndim}"
        )

    # Guard zero-variance steps (e.g. a zero-length gap): no diffusion → the
    # bridge cannot breach, so survival is governed by the endpoints alone.
    var = np.where(var > 0.0, var, np.inf)

    above = (S_a > B) & (S_b > B)
    la = np.log(np.where(S_a > 0, S_a, np.nan) / B)
    lb = np.log(np.where(S_b > 0, S_b, np.nan) / B)

    # Bridge survival on a step where both endpoints are above the barrier;
    # endpoints at/below the barrier give survival 0 (certain knock-in).
    bridge = 1.0 - np.exp(-2.0 * la * lb / var)
    step_survival = np.where(above, bridge, 0.0)           # (n_paths, n_steps, n_assets)

    # Product over steps and assets → path survival probability.
    return np.prod(step_survival, axis=(1, 2))


def continuous_survival_prob(
    prices: np.ndarray,
    barriers: np.ndarray,
    vols: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Survival probability from constant per-asset vols and step lengths.

    Convenience wrapper over :func:`continuous_survival_prob_from_var` for the
    constant-volatility case: the per-step variance is ``σ_a² · Δt_t``.

    Parameters
    ----------
    prices : np.ndarray, shape ``(n_paths, n_points, n_assets)``
    barriers : np.ndarray, shape ``(n_assets,)``
    vols : np.ndarray, shape ``(n_assets,)`` — annualised volatilities.
    dt : np.ndarray, shape ``(n_points - 1,)`` — interval year-fractions.
    """
    vols = np.asarray(vols, dtype=float)
    dt   = np.asarray(dt, dtype=float)
    step_var = (vols[None, :] ** 2) * dt[:, None]          # (n_steps, n_assets)
    return continuous_survival_prob_from_var(prices, barriers, step_var)


def sample_knock_in(
    prices: np.ndarray,
    barriers: np.ndarray,
    step_var: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Turn per-path U(0,1) draws into definite knock-in outcomes.

    For a stress *distribution* (as opposed to a fair-value *expectation*) each
    path needs a concrete breached/not-breached outcome, not a probability —
    otherwise the breach/no-breach bimodality that drives the downside tail is
    averaged away.  Comparing one uniform per path to the Brownian-bridge
    survival probability samples knock-ins at exactly the continuous-monitoring
    rate while preserving the full outcome distribution.

    The uniforms are supplied by the caller (drawn from the engine's
    :class:`NoiseSampler`, never seeded inside the engine) so stress runs remain
    reproducible under Common Random Numbers.

    Returns
    -------
    np.ndarray of bool, shape ``(n_paths,)`` — True where the path knocks in.
    """
    survival = continuous_survival_prob_from_var(prices, barriers, step_var)
    return np.asarray(uniforms, dtype=float) > survival
