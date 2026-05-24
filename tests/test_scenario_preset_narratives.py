"""Each preset scenario's factor path must match its narrative.

Run deterministically (zero factor volatility, no mean reversion, zero ambient
drift) so the MKT factor path reflects only the events and their recovery
archetypes. The presets fall into three narrative classes:

* FULL recovery   — the market falls then returns to (or above) its pre-shock
                    level by the horizon (V-shape / deal-driven rally).
* PARTIAL         — the market recovers from its trough but ends materially
                    below baseline (slow grind, no V-shape).
* NO recovery     — the market falls and stays down (continuation).

A path that fell and never recovered in *every* scenario — the symptom that
motivated this test — would violate the FULL and PARTIAL assertions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
from src.factor_engine import FACTORS
from src.factor_premiums import REGIMES
from src.factor_scenario_engine import FactorScenarioEngine
from tests.conftest import make_brc_row

FIXED_TODAY = pd.Timestamp("2024-01-02")
FACTOR_CODES = list(FACTORS.keys())
BASE = 100.0

FULL_RECOVERY = {"V-Shape Sell-off", "COVID March 2020",
                 "Tariffs / Trade War", "Oil Spike"}
PARTIAL_RECOVERY = {"Inflation 2022", "Tech Wreck"}
NO_RECOVERY = {"Slow Bleed"}


def _deterministic_engine():
    """Zero factor vol, zero idio, no mean reversion, zero ambient drift — so
    the factor path is driven solely by the events. Maturity is ~4y so any
    recovery horizon (≤ 2y) completes."""
    row = make_brc_row(
        current_spot=100.0, strike=100.0, initial_level=100.0,
        maturity_date="2028-01-03",
    )
    isin = row["underlying_isins"][0]
    loadings = {
        isin: {
            "betas": {c: (1.0 if c == "MKT" else 0.0) for c in FACTOR_CODES},
            "alpha": 0.0, "idio_vol": 0.0, "r_squared": 0.9, "n_obs": 750,
        }
    }
    fe = MagicMock()
    fe.factor_vol.return_value = pd.Series(0.0, index=FACTOR_CODES)
    fe.factor_corr.return_value = pd.DataFrame(
        np.eye(len(FACTOR_CODES)), index=FACTOR_CODES, columns=FACTOR_CODES,
    )
    zero_premiums = pd.DataFrame(
        0.0, index=pd.Index(list(REGIMES), name="regime"), columns=FACTOR_CODES,
    )
    return FactorScenarioEngine(
        portfolio=pd.DataFrame([row]), loadings=loadings, factor_engine=fe,
        n_paths=1, idio_intensity=0.0, mean_reversion_kappa=0.0,
        premiums=zero_premiums,
    )


def _mkt_path(preset_name):
    res = _deterministic_engine().run_path_scenario(
        FACTOR_SCENARIO_PRESETS[preset_name], today=FIXED_TODAY,
    )
    return res["factor_paths"]["MKT"]["median"].to_numpy()


@pytest.mark.parametrize("name", sorted(FULL_RECOVERY))
def test_full_recovery_returns_to_baseline(name):
    mkt = _mkt_path(name)
    assert mkt.min() <= 96.0, f"{name}: never fell (trough {mkt.min():.1f})"
    assert mkt[-1] >= 96.0, (
        f"{name}: fell to {mkt.min():.1f} but only recovered to {mkt[-1]:.1f}"
    )


@pytest.mark.parametrize("name", sorted(PARTIAL_RECOVERY))
def test_partial_recovery_climbs_from_trough_but_stays_below_baseline(name):
    mkt = _mkt_path(name)
    trough, end = float(mkt.min()), float(mkt[-1])
    assert trough <= 90.0, f"{name}: did not fall enough (trough {trough:.1f})"
    assert end >= trough + 2.0, (
        f"{name}: no recovery from trough ({end:.1f} vs trough {trough:.1f})"
    )
    assert end <= 95.0, f"{name}: recovered fully ({end:.1f}); expected partial"


@pytest.mark.parametrize("name", sorted(NO_RECOVERY))
def test_no_recovery_stays_down(name):
    mkt = _mkt_path(name)
    trough, end = float(mkt.min()), float(mkt[-1])
    assert end <= 82.0, f"{name}: ended too high ({end:.1f}) for a no-recovery bear"
    assert end <= trough + 2.0, (
        f"{name}: recovered ({end:.1f}) but should stay near the trough ({trough:.1f})"
    )


def test_custom_preset_is_flat():
    # No events → no shock, zero drift → the factor stays at baseline.
    mkt = _mkt_path("Custom")
    assert np.allclose(mkt, BASE, atol=1e-6)


def test_every_named_preset_is_classified():
    # Guards against a new preset silently escaping the narrative checks.
    named = set(FACTOR_SCENARIO_PRESETS) - {"Custom"}
    classified = FULL_RECOVERY | PARTIAL_RECOVERY | NO_RECOVERY
    assert named == classified, f"unclassified presets: {named ^ classified}"
