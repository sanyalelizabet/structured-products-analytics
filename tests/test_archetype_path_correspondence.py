"""Each recovery archetype must produce the factor path it prescribes.

Volatility is set to zero so the factor path is fully deterministic; the only
dynamics are the prescribed drift and the discrete shock. After a shock of s the
factor (base 100) should reach, once the recovery horizon has elapsed and drift
reverts to the flat initial state:

    terminal = 100 · (1 + s)^(1 + sign)

i.e. baseline 100 for any recovery (sign −1), 100·(1+s) for stable (sign 0),
and 100·(1+s)² for continued bear (sign +1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.factor_engine import FACTORS
from src.factor_premiums import REGIMES
from src.factor_scenario_engine import FactorScenarioEngine
from src.scenario_archetypes import EVENT_RECOVERY_ARCHETYPES
from tests.conftest import make_brc_row

FIXED_TODAY = pd.Timestamp("2024-01-02")
FACTOR_CODES = list(FACTORS.keys())
SHOCK_PCT = -25.0


def _zero_premiums():
    return pd.DataFrame(
        0.0, index=pd.Index(list(REGIMES), name="regime"), columns=FACTOR_CODES,
    )


def _engine():
    """Deterministic engine: zero factor vol, zero idio, no mean reversion,
    flat (zero) initial drift."""
    row = make_brc_row(
        current_spot=100.0, strike=100.0, initial_level=100.0,
        maturity_date="2028-01-03",   # ~4y out — longer than any recovery horizon
    )
    portfolio = pd.DataFrame([row])
    isin = row["underlying_isins"][0]
    loadings = {
        isin: {
            "betas": {c: (1.0 if c == "MKT" else 0.0) for c in FACTOR_CODES},
            "alpha": 0.0, "idio_vol": 0.0, "r_squared": 0.9, "n_obs": 750,
        }
    }
    fe = MagicMock()
    fe.factor_vol.return_value = pd.Series(0.0, index=FACTOR_CODES)   # σ=0 → deterministic
    fe.factor_corr.return_value = pd.DataFrame(
        np.eye(len(FACTOR_CODES)), index=FACTOR_CODES, columns=FACTOR_CODES,
    )
    return FactorScenarioEngine(
        portfolio=portfolio, loadings=loadings, factor_engine=fe,
        n_paths=1, idio_intensity=0.0, mean_reversion_kappa=0.0,
        premiums=_zero_premiums(),
    )


def _run_mkt(archetype):
    eng = _engine()
    ui = {
        "initial_market_state": "Flat",
        "events": [{"day": 30, "factor_shock": {"MKT": SHOCK_PCT},
                    "recovery": archetype}],
    }
    res = eng.run_path_scenario(ui, today=FIXED_TODAY)
    return res["factor_paths"]["MKT"]["median"].to_numpy()


@pytest.mark.parametrize("archetype,sign,horizon",
                         [(a, s, h) for a, (s, h) in EVENT_RECOVERY_ARCHETYPES.items()])
def test_terminal_matches_archetype(archetype, sign, horizon):
    mkt = _run_mkt(archetype)
    expected = 100.0 * (1.0 + SHOCK_PCT / 100.0) ** (1 + sign)
    assert mkt[-1] == pytest.approx(expected, abs=1.0), (
        f"{archetype}: terminal {mkt[-1]:.2f} != expected {expected:.2f}"
    )


def test_baseline_before_shock_is_100():
    mkt = _run_mkt("Stable (no drift)")
    # Day 0 (before the day-30 shock) sits at the base level.
    assert mkt[0] == pytest.approx(100.0, abs=1e-6)


def test_shock_drops_factor_to_seventy_five():
    # Immediately after the shock, before any recovery accrues, the factor is
    # at 100·(1+s) = 75 for every archetype.
    for archetype in EVENT_RECOVERY_ARCHETYPES:
        mkt = _run_mkt(archetype)
        assert mkt.min() <= 75.0 + 1.0, f"{archetype}: never dipped to the shock level"


def test_recovery_rises_continuation_falls_after_shock():
    """Direction check: a recovery archetype ends above the shock level; a
    continuation ends below it; stable stays at it."""
    shock_level = 100.0 * (1.0 + SHOCK_PCT / 100.0)   # 75
    assert _run_mkt("Fast recovery (~6mo)")[-1] > shock_level + 1.0
    assert _run_mkt("Continued bear")[-1] < shock_level - 1.0
    assert _run_mkt("Stable (no drift)")[-1] == pytest.approx(shock_level, abs=1.0)
