"""Per-factor annualised drifts, conditioned on the equity-market regime.

The market timeline is split — once, jointly — into three economically defined
regimes (bear / flat / bull) using two signals:

* **Trend** — the trailing-12-month MKT log-return:
      < −5 %  bear   ·   −5 % … +10 %  flat   ·   > +10 %  bull
* **Stress (vol-aware)** — when volatility is elevated (realized MKT vol, or an
  external gauge such as VIX, above the stress threshold) the day is forced to
  the bear/stress regime regardless of trend, and bull additionally requires
  calm. This catches drawdown bursts a slow trailing-return trend would miss.

All thresholds and the regime ERP anchors are named, reviewable assumptions —
see ``docs/factor_premium_methodology.md``.

A factor's premium in a regime is estimated from the days in that regime. Two
estimators are offered (the UI lets the user pick):

* ``"mean"``       — conditional sample mean (annualised). Transparent baseline;
  a data-sparse regime falls back to the regime ERP scalar.
* ``"shrinkage"``  — the conditional mean shrunk toward a per-factor structural
  prior ``β_{f,MKT} · ERP_regime`` with weight ``w = n / (n + n0)``. A sparse
  regime collapses to the prior, which is *per-factor* (a high-MKT-beta factor
  gets a larger bear drawdown than a defensive one) rather than a flat scalar.
  This is the recommended estimator given short samples (drift estimation is
  notoriously noisy — Merton, 1980).

Output: DataFrame indexed by regime, one column per factor. These are forward
*scenario assumptions* for the factor-stress path simulator, not return
forecasts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Regime taxonomy
# ──────────────────────────────────────────────────────────────────────────

# Internal regime keys (used as CSV row index), ordered bear → bull.
REGIMES: tuple[str, ...] = ("bear", "flat", "bull")

# Trailing-12mo MKT return thresholds. A date is in regime R if its
# trailing-12mo MKT log-return r satisfies ``THRESHOLDS[R][0] <= r < [1]``.
REGIME_THRESHOLDS: dict[str, tuple[float, float]] = {
    "bear": (-np.inf, -0.05),
    "flat": (-0.05, +0.10),
    "bull": (+0.10, +np.inf),
}

# Assumed market equity-risk-premium per regime (annualised). Serves two roles:
# the shrinkage prior anchor (prior_f = β_{f,MKT} · ERP_regime) and the
# fall-back scalar for the "mean" estimator when a regime is data-sparse.
REGIME_ERP: dict[str, float] = {
    "bear": -0.10,
    "flat":  0.00,
    "bull": +0.12,
}

# Sparse-regime threshold for the "mean" estimator: below this the regime row
# falls back to the ERP scalar.
MIN_OBS_PER_REGIME: int = 60  # ~3 trading months

# Shrinkage prior strength n0 (in observations): the data weight is
# w = n / (n + n0). Set to ~1 year because an *annualised drift* estimated from
# fewer than ~a year of daily observations is dominated by noise (mean × 252
# amplifies it), so thin regimes should lean on the structural prior.
SHRINKAGE_PRIOR_STRENGTH: int = 252

# Length of history used to estimate the premiums (calendar years).
ESTIMATION_LOOKBACK_YEARS: int = 5

# Trailing-window length for the MKT trend classifier (in trading days).
TRAILING_WINDOW_DAYS: int = 252  # 1 year

# Vol-aware stress override. Realized MKT volatility above the threshold marks a
# stress regime (→ bear) regardless of trend, and bull requires calm (vol below
# it). The window is ~1 trading month; the threshold (annualised) is a common
# stress marker (~VIX 25). The classifier accepts an external vol signal too, so
# a forward-looking gauge (e.g. VIX) can be substituted for realized vol.
VOL_WINDOW_DAYS: int = 21
STRESS_VOL_THRESHOLD: float = 0.25  # annualised volatility

# Clip annualised drifts to a sane band — short concentrated moves can produce
# implausible annualised means.
DRIFT_CLIP_BAND: tuple[float, float] = (-0.25, +0.25)

PREMIUM_METHODS: tuple[str, ...] = ("mean", "shrinkage")


# ──────────────────────────────────────────────────────────────────────────
# Compute
# ──────────────────────────────────────────────────────────────────────────

def classify_regimes(
    mkt_returns: pd.Series, vol_signal: pd.Series | None = None,
) -> pd.Series:
    """Tag each date with its regime, vol-aware.

    Two signals combine:

    * **Trend** — the trailing-12mo MKT log-return places the day on the
      bear / flat / bull return axis via :data:`REGIME_THRESHOLDS`.
    * **Stress** — a *down-stress* day (volatility elevated, ``vol_signal``
      above :data:`STRESS_VOL_THRESHOLD`, **and** the recent return negative) is
      forced to ``bear`` regardless of trend; and ``bull`` requires calm (vol not
      elevated). The negative-return condition matters: volatility is
      direction-agnostic, and high-vol *rebound* days must not be labelled bear
      or they pollute the bear-regime mean with positive returns.

    ``vol_signal`` defaults to ~1-month annualised realized MKT volatility; pass
    a forward-looking gauge (e.g. VIX, as a fraction) to override it.

    Dates without a full trailing window (the first ~year) are NaN and dropped
    by callers.
    """
    trailing = mkt_returns.rolling(TRAILING_WINDOW_DAYS).sum()
    short_ret = mkt_returns.rolling(VOL_WINDOW_DAYS).sum()
    if vol_signal is None:
        vol_signal = mkt_returns.rolling(VOL_WINDOW_DAYS).std() * np.sqrt(252.0)

    vol_elevated = (vol_signal.reindex(mkt_returns.index) > STRESS_VOL_THRESHOLD).fillna(False)
    down_stress = vol_elevated & (short_ret < 0)

    bear_hi = REGIME_THRESHOLDS["bear"][1]    # upper bound of the bear return band
    bull_lo = REGIME_THRESHOLDS["bull"][0]    # lower bound of the bull return band

    valid = trailing.notna()
    out = pd.Series(index=mkt_returns.index, dtype=object)
    out[valid] = "flat"
    # Bull = calm uptrend; a high-vol melt-up is not a clean bull → stays flat.
    out[valid & (trailing >= bull_lo) & ~vol_elevated] = "bull"
    # Stress wins: a falling, high-vol day is bear even if the slow trend isn't.
    out[valid & ((trailing < bear_hi) | down_stress)] = "bear"
    return out


def factor_mkt_betas(returns: pd.DataFrame) -> dict[str, float]:
    """Univariate β of each factor on MKT over the full sample (MKT β ≡ 1).

    ``β_f = Cov(r_f, r_MKT) / Var(r_MKT)``. Used to build the per-factor
    shrinkage prior.
    """
    mkt = returns["MKT"]
    var = float(mkt.var())
    if var <= 0:
        return {c: (1.0 if c == "MKT" else 0.0) for c in returns.columns}
    return {
        c: (1.0 if c == "MKT" else float(returns[c].cov(mkt) / var))
        for c in returns.columns
    }


def compute_factor_premiums(
    factor_engine,
    lookback_years: Optional[int] = ESTIMATION_LOOKBACK_YEARS,
    method: str = "mean",
) -> pd.DataFrame:
    """Per-regime, per-factor annualised drifts.

    ``method`` ∈ :data:`PREMIUM_METHODS`:

    * ``"mean"``      — conditional sample mean × 252, clipped; a regime with
      fewer than :data:`MIN_OBS_PER_REGIME` days falls back to the regime ERP
      scalar across all factors.
    * ``"shrinkage"`` — ``w · sample_mean + (1 − w) · (β_f · ERP_regime)`` with
      ``w = n / (n + MIN_OBS_PER_REGIME)``, then clipped.

    Returns a DataFrame indexed by :data:`REGIMES` with one column per factor.
    """
    if method not in PREMIUM_METHODS:
        raise ValueError(f"Unknown method {method!r}; use one of {PREMIUM_METHODS}.")

    returns = factor_engine.build_returns(years=lookback_years)
    if "MKT" not in returns.columns:
        raise RuntimeError("Factor returns missing MKT column; cannot classify regimes.")

    labels = classify_regimes(returns["MKT"])
    factor_codes = list(returns.columns)
    lo, hi = DRIFT_CLIP_BAND
    betas = factor_mkt_betas(returns) if method == "shrinkage" else None

    rows: dict[str, dict[str, float]] = {}
    for regime in REGIMES:
        mask = labels == regime
        n = int(mask.sum())
        erp = REGIME_ERP[regime]

        if method == "mean":
            if n < MIN_OBS_PER_REGIME:
                log.warning(
                    "Regime %r has only %d observations (< %d); 'mean' method "
                    "falls back to the ERP scalar %.4f for all factors.",
                    regime, n, MIN_OBS_PER_REGIME, erp,
                )
                rows[regime] = {c: erp for c in factor_codes}
            else:
                ann = (returns.loc[mask].mean() * 252.0).clip(lo, hi)
                rows[regime] = ann.to_dict()
        else:  # shrinkage
            prior = {c: betas[c] * erp for c in factor_codes}
            if n == 0:
                rows[regime] = {c: float(np.clip(prior[c], lo, hi)) for c in factor_codes}
            else:
                sample = returns.loc[mask].mean() * 252.0
                w = n / (n + SHRINKAGE_PRIOR_STRENGTH)
                rows[regime] = {
                    c: float(np.clip(w * float(sample[c]) + (1.0 - w) * prior[c], lo, hi))
                    for c in factor_codes
                }

    df = pd.DataFrame.from_dict(rows, orient="index", columns=factor_codes)
    df.index.name = "regime"
    return df.loc[list(REGIMES)]   # stable row order


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CSV_PATH = Path("data") / "factor_premiums.csv"


def csv_path_for_method(method: str, csv_path: Path = DEFAULT_CSV_PATH) -> Path:
    """Per-method cache path: ``mean`` uses the default file, others suffix it."""
    csv_path = Path(csv_path)
    if method == "mean":
        return csv_path
    return csv_path.with_name(f"{csv_path.stem}_{method}{csv_path.suffix}")


def save_premiums(df: pd.DataFrame, csv_path: Path = DEFAULT_CSV_PATH) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)


def load_premiums(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col="regime")


def load_or_compute_premiums(
    factor_engine=None,
    csv_path: Path = DEFAULT_CSV_PATH,
    recompute: bool = False,
    method: str = "mean",
) -> pd.DataFrame:
    """Read the cached premiums for ``method``; compute + cache if missing or
    ``recompute``. Each method has its own cache file (see
    :func:`csv_path_for_method`)."""
    path = csv_path_for_method(method, csv_path)
    if not recompute and path.exists():
        return load_premiums(path)
    if factor_engine is None:
        raise RuntimeError(
            f"Premium cache {path} missing and no factor_engine provided to recompute."
        )
    df = compute_factor_premiums(factor_engine, method=method)
    save_premiums(df, path)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Public lookup used by scenario_archetypes
# ──────────────────────────────────────────────────────────────────────────

def get_factor_drift(
    regime: str,
    factor_codes: Iterable[str],
    premiums: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Per-factor annualised drift for a regime.

    If ``premiums`` is not supplied, loads the default (``mean``) cache. If that
    is missing, falls back to the regime ERP scalar broadcast across factors —
    so the function is always safe to call.
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime!r}")

    if premiums is None:
        try:
            premiums = load_premiums(DEFAULT_CSV_PATH)
        except FileNotFoundError:
            return {c: REGIME_ERP[regime] for c in factor_codes}

    row = premiums.loc[regime]
    return {
        c: float(row[c]) if c in row.index else REGIME_ERP[regime]
        for c in factor_codes
    }
