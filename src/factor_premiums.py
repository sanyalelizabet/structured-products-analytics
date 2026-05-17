"""Per-factor historical premium drifts, conditional on equity-market regime.

Replaces the old scalar "+7 %/y" treatment of initial market state with a
**per-factor** drift dict, derived from historical data.

Conceptual model
----------------
We classify every historical trading day by the **trailing 12-month MKT
return**:

    MKT 12-mo return         regime
    > +15 %                  strong_bull
    +5 % … +15 %             moderate_bull
    −5 % … +5 %              stable
    < −5 %                   bear

Within each regime, we take the **annualised mean daily log-return** of
each factor as that factor's premium in that regime.  The result is a
DataFrame with one row per regime and one column per factor.

Why MKT-conditional and not per-factor regimes?  Because the UI exposes a
single "initial market state" picker — a global ambient assumption — not
one regime per factor.  Conditioning on MKT gives us the realistic
cross-asset behaviour during equity regimes (e.g. how FX/Energy actually
co-move when broad equities are in a bull run).

Data scarcity handling
----------------------
With limited price history some regimes will be under-sampled.  If a
regime has fewer than :data:`MIN_OBS_PER_REGIME` observations, we fall
back to a hardcoded scalar default (the legacy "+7 %/y" pattern, applied
uniformly to all factors).  This keeps the engine producing sensible
drifts on a fresh checkout while preserving room for richer behaviour as
the price DB grows.

Storage
-------
Computed once and cached to ``data/factor_premiums.csv``.  Recompute on
demand by calling :func:`compute_factor_premiums` directly, or pass
``recompute=True`` to :func:`load_or_compute_premiums`.
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

# Internal regime keys (used as CSV row index).
REGIMES: tuple[str, ...] = ("bear", "stable", "moderate_bull", "strong_bull")

# Trailing-12mo MKT return thresholds defining each regime.  Tunable in
# one place.  Convention: a date falls in regime R if its trailing-12mo
# MKT log-return r satisfies ``REGIME_THRESHOLDS[R][0] <= r < [1]``.
REGIME_THRESHOLDS: dict[str, tuple[float, float]] = {
    "bear":          (-np.inf,  -0.05),
    "stable":        (-0.05,    +0.05),
    "moderate_bull": (+0.05,    +0.15),
    "strong_bull":   (+0.15,    +np.inf),
}

# Legacy scalar fallbacks — used when a regime is data-sparse.  These are
# the original scalar drifts from the previous design, broadcast uniformly
# across factors.  They give a sensible "neutral" answer when history is
# insufficient to estimate.
LEGACY_SCALAR_DRIFTS: dict[str, float] = {
    "bear":          -0.07,
    "stable":         0.00,
    "moderate_bull": +0.07,
    "strong_bull":   +0.15,
}

# A regime needs at least this many observations to trust the computed
# per-factor mean; otherwise fall back to the scalar default.
MIN_OBS_PER_REGIME: int = 60  # ~3 trading months

# Trailing-window length for the MKT classifier (in trading days).
TRAILING_WINDOW_DAYS: int = 252  # 1 year

# Sanity clip applied to every computed drift before it's cached.  With
# only a few years of history, short concentrated rallies (e.g. a 6-month
# oil spike falling entirely inside the moderate_bull bucket) can produce
# annualised means well outside any realistic long-run drift.  Clipping
# to ±25%/yr keeps the regime drifts useable as scenario-segment
# assumptions without distorting them when more history is available.
DRIFT_CLIP_BAND: tuple[float, float] = (-0.25, +0.25)


# ──────────────────────────────────────────────────────────────────────────
# Compute
# ──────────────────────────────────────────────────────────────────────────

def classify_regimes(mkt_returns: pd.Series) -> pd.Series:
    """Tag each date with its regime based on trailing-12mo MKT log-return.

    Dates without a full trailing window (the first ~year of history) are
    labelled NaN and dropped by callers.
    """
    trailing = mkt_returns.rolling(TRAILING_WINDOW_DAYS).sum()
    out = pd.Series(index=mkt_returns.index, dtype=object)
    for regime, (lo, hi) in REGIME_THRESHOLDS.items():
        mask = (trailing >= lo) & (trailing < hi)
        out.loc[mask] = regime
    return out


def compute_factor_premiums(
    factor_engine,
    lookback_years: Optional[int] = None,
) -> pd.DataFrame:
    """Per-regime, per-factor annualised drifts from historical returns.

    Returns a DataFrame indexed by regime (``REGIMES``) with one column per
    factor code.  Sparse regimes (< ``MIN_OBS_PER_REGIME`` observations)
    are populated with the legacy scalar default for that regime.
    """
    returns = factor_engine.build_returns(years=lookback_years)
    if "MKT" not in returns.columns:
        raise RuntimeError(
            "Factor returns missing MKT column; cannot classify regimes."
        )

    regime_labels = classify_regimes(returns["MKT"])
    factor_codes = list(returns.columns)

    rows = {}
    for regime in REGIMES:
        mask = regime_labels == regime
        n = int(mask.sum())
        if n < MIN_OBS_PER_REGIME:
            log.warning(
                "Regime %r has only %d observations (< %d); falling back "
                "to scalar default %.4f for all factors.",
                regime, n, MIN_OBS_PER_REGIME, LEGACY_SCALAR_DRIFTS[regime],
            )
            scalar = LEGACY_SCALAR_DRIFTS[regime]
            rows[regime] = {c: scalar for c in factor_codes}
            continue
        # Annualise: mean daily log-return × 252, then clip to a sane
        # band to keep short concentrated rallies from polluting the
        # multi-year drift.
        ann_means = returns.loc[mask].mean() * 252.0
        lo, hi = DRIFT_CLIP_BAND
        ann_means = ann_means.clip(lower=lo, upper=hi)
        rows[regime] = ann_means.to_dict()

    df = pd.DataFrame.from_dict(rows, orient="index", columns=factor_codes)
    df.index.name = "regime"
    return df.loc[list(REGIMES)]   # stable row order


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CSV_PATH = Path("data") / "factor_premiums.csv"


def save_premiums(df: pd.DataFrame, csv_path: Path = DEFAULT_CSV_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)


def load_premiums(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col="regime")


def load_or_compute_premiums(
    factor_engine=None,
    csv_path: Path = DEFAULT_CSV_PATH,
    recompute: bool = False,
) -> pd.DataFrame:
    """Read cached premiums; compute + cache if missing or ``recompute``."""
    csv_path = Path(csv_path)
    if not recompute and csv_path.exists():
        return load_premiums(csv_path)
    if factor_engine is None:
        raise RuntimeError(
            f"Premium cache {csv_path} missing and no factor_engine "
            "provided to recompute."
        )
    df = compute_factor_premiums(factor_engine)
    save_premiums(df, csv_path)
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

    If ``premiums`` is not supplied, attempts to load from the default
    CSV cache.  If the cache is missing, falls back to the legacy scalar
    drift broadcast uniformly across factors — so the function is always
    safe to call.
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime!r}")

    if premiums is None:
        try:
            premiums = load_premiums()
        except FileNotFoundError:
            scalar = LEGACY_SCALAR_DRIFTS[regime]
            return {c: scalar for c in factor_codes}

    row = premiums.loc[regime]
    scalar = LEGACY_SCALAR_DRIFTS[regime]
    return {
        c: float(row[c]) if c in row.index else scalar
        for c in factor_codes
    }
