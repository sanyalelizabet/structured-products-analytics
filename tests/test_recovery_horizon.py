"""Tests for the recovery-horizon mechanic in both scenario engines.

After a shock, the recovery drift is supposed to run for a *bounded*
window (the archetype's horizon — 6 mo for Fast, 1 mo for Very Fast,
2 y for Slow, 1 y for Continued bear).  Past that window, drift
reverts to the *initial market state* so a steep recovery drift
doesn't run forever and produce extreme overshoots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.risk.scenario_engine        import ScenarioEngine
from src.risk.factor_scenario_engine import FactorScenarioEngine
from src.risk.factor_engine          import FACTORS, FactorEngine
from src.risk.factor_premiums        import REGIMES
from src.market_data.market_data_engine     import MarketDataEngine
from src.numerics.noise_sampler          import NoiseSampler


def _neutral_premiums():
    """Controlled regime×factor premiums (flat = 0, modest bull/bear) so the
    recovery-horizon mechanic is tested independently of the live premium CSV."""
    vals = {"bear": -0.08, "flat": 0.0, "bull": 0.10}
    return pd.DataFrame(
        {c: [vals[r] for r in REGIMES] for c in FACTORS},
        index=pd.Index(list(REGIMES), name="regime"),
    )
from src.risk.scenario_archetypes    import (
    EVENT_RECOVERY_ARCHETYPES,
    event_drift_for_factor,
    initial_drift_dict,
)
from tests.conftest import make_brc_row, make_mbrc_row


# ──────────────────────────────────────────────────────────────────────────
# Single-factor ScenarioEngine — recovery_horizon_years
# ──────────────────────────────────────────────────────────────────────────

class TestSingleFactorRecoveryHorizon:
    """``ScenarioEngine`` should switch from the recovery drift back to
    ``post_recovery_drift_pa`` once ``recovery_horizon_years`` elapses
    after the last shock."""

    def _engine(self, n_paths=1):
        portfolio = pd.DataFrame([
            make_brc_row(maturity_date="2030-01-01", current_spot=100.0),
        ])
        return ScenarioEngine(
            portfolio=portfolio,
            beta_map={"CH0012221716": 1.0},
            vol_map={"CH0012221716": 0.001},   # near-zero vol → near-deterministic
            n_paths=n_paths,
            mean_reversion_kappa=0.0,           # turn off OU pull for clean test
        )

    def _scenario(self, market_shock=-25, horizon=None,
                  pre=0.0, post_recovery=0.0):
        # Recovery drift ≈ −log(0.75)/0.5 ≈ +0.575/y for Fast (~6mo).
        recovery_drift = event_drift_for_factor(market_shock, "Fast recovery (~6mo)")
        s = {
            "market_shock":        market_shock,
            "n_shocks":            1,
            "shock_in_days":       30,
            "shock_spacing_days":  0,
            "pre_shock_drift_pa":  pre,
            "post_shock_drift_pa": recovery_drift,
        }
        if horizon is not None:
            s["recovery_horizon_years"] = horizon
            s["post_recovery_drift_pa"] = post_recovery
        return s

    def test_with_horizon_path_does_not_overshoot(self):
        """With a 0.5y horizon and -25% shock + Fast recovery, terminal
        spot should not be far above baseline — past 6 months drift = 0."""
        eng = self._engine()
        # Sampler shared so CRN keeps noise constant across runs.
        today = pd.Timestamp.today().normalize()
        n_days = len(pd.bdate_range(start=today, end=pd.Timestamp("2030-01-01")))
        sampler = NoiseSampler(n_paths=1, n_days=n_days, factor_codes=[],
                               isins=["CH0012221716"], seed=1)
        eng.noise_sampler = sampler

        with_horizon = eng.run_path_scenario(
            self._scenario(market_shock=-25, horizon=0.5, post_recovery=0.0)
        )
        terminal = float(with_horizon["asset_paths"]["CH0012221716"]["median"].iloc[-1])
        # With horizon: at ~3y horizon, only 6mo of recovery drift, then flat.
        # Terminal should sit close to baseline (within ±15 % under near-zero vol).
        assert 85.0 < terminal < 115.0, (
            f"With recovery horizon, terminal should stay near 100; got {terminal:.1f}"
        )

    def test_without_horizon_path_overshoots(self):
        """Sanity check: omitting the horizon reproduces legacy behaviour
        — drift runs forever, terminal blows past baseline."""
        eng = self._engine()
        today = pd.Timestamp.today().normalize()
        n_days = len(pd.bdate_range(start=today, end=pd.Timestamp("2030-01-01")))
        sampler = NoiseSampler(n_paths=1, n_days=n_days, factor_codes=[],
                               isins=["CH0012221716"], seed=1)
        eng.noise_sampler = sampler

        no_horizon = eng.run_path_scenario(self._scenario(market_shock=-25))
        terminal = float(no_horizon["asset_paths"]["CH0012221716"]["median"].iloc[-1])
        # Drift ≈ +0.575/y over ~3y of post-shock window → e^(1.7) ≈ 550 % of
        # post-shock level.  Even with mean-reversion off, well above 200.
        assert terminal > 200.0, (
            f"Without horizon, terminal should overshoot; got {terminal:.1f}"
        )

    def test_horizon_terminal_lower_than_no_horizon(self):
        """With identical inputs except the horizon, the bounded run
        terminal must be below the unbounded one."""
        eng = self._engine()
        today = pd.Timestamp.today().normalize()
        n_days = len(pd.bdate_range(start=today, end=pd.Timestamp("2030-01-01")))
        sampler = NoiseSampler(n_paths=1, n_days=n_days, factor_codes=[],
                               isins=["CH0012221716"], seed=42)
        eng.noise_sampler = sampler

        bounded   = eng.run_path_scenario(
            self._scenario(market_shock=-25, horizon=0.5, post_recovery=0.0)
        )
        eng.noise_sampler.regenerate(seed=42)   # same seed → same noise
        unbounded = eng.run_path_scenario(self._scenario(market_shock=-25))

        a = float(bounded["asset_paths"]["CH0012221716"]["median"].iloc[-1])
        b = float(unbounded["asset_paths"]["CH0012221716"]["median"].iloc[-1])
        assert a < b


# ──────────────────────────────────────────────────────────────────────────
# Factor engine — per-event horizon
# ──────────────────────────────────────────────────────────────────────────

FACTOR_CODES = list(FACTORS.keys())


def _seed_factor_db(tmp_path, n_days=900, seed=29):
    rng   = np.random.default_rng(seed)
    end   = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)
    sigmas = {"MKT": 0.001, "TECH": 0.001, "HC": 0.001,
              "FIN": 0.001, "ENERGY": 0.001, "FX": 0.001}   # near-zero vol
    rows = []
    for code in FACTOR_CODES:
        ticker, key, _ = FACTORS[code]
        rets = rng.normal(0, sigmas[code] / np.sqrt(252), n_days)
        prices = 100.0 * np.exp(np.cumsum(rets))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "isin": key, "ticker": ticker, "price": p})
    pd.DataFrame(rows).to_csv(tmp_path / "prices.csv", index=False)


def _factor_engine(tmp_path, n_paths=1):
    _seed_factor_db(tmp_path)
    mock_client = MagicMock()
    mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
    mde.fetch_daily_prices = MagicMock(return_value=None)
    fe = FactorEngine(mde)

    row = make_mbrc_row(
        initial_levels=[100.0, 100.0],
        strikes=[100.0, 100.0],
        current_spots=[100.0, 100.0],
        maturity_date="2030-01-01",
    )
    row["underlyings"]      = ["X", "Y"]
    row["underlying_isins"] = ["TEST_X", "TEST_Y"]
    portfolio = pd.DataFrame([row])

    loadings = {
        "TEST_X": {"betas": {"MKT": 1.0, "TECH": 0.0, "HC": 0.0,
                              "FIN": 0.0, "ENERGY": 0.0, "FX": 0.0},
                    "alpha": 0.0, "idio_vol": 0.001,
                    "r_squared": 0.9, "n_obs": 750},
        "TEST_Y": {"betas": {"MKT": 1.0, "TECH": 0.0, "HC": 0.0,
                              "FIN": 0.0, "ENERGY": 0.0, "FX": 0.0},
                    "alpha": 0.0, "idio_vol": 0.001,
                    "r_squared": 0.9, "n_obs": 750},
    }

    return FactorScenarioEngine(
        portfolio=portfolio, loadings=loadings, factor_engine=fe,
        n_paths=n_paths, idio_intensity=0.0, mean_reversion_kappa=0.0,
        premiums=_neutral_premiums(),
    )


class TestFactorEngineRecoveryHorizon:

    def _ui_scenario(self, archetype, market_shock=-25, day=30,
                     initial="Flat"):
        return {
            "initial_market_state": initial,
            "events": [
                {"day": day,
                 "factor_shock": {"MKT": market_shock},
                 "recovery": archetype},
            ],
        }

    def test_fast_recovery_does_not_overshoot(self, tmp_path):
        """Fast recovery (~6mo) horizon = 0.5y.  After that, drift reverts
        to initial = 0 → no upward overshoot until maturity."""
        eng = _factor_engine(tmp_path)
        res = eng.run_path_scenario(
            self._ui_scenario("Fast recovery (~6mo)", market_shock=-25)
        )
        mkt = res["factor_paths"]["MKT"]["median"].to_numpy()
        terminal = float(mkt[-1])
        # With horizon, terminal MKT should stay near baseline (100).
        assert 85.0 < terminal < 115.0, (
            f"Fast recovery overshooting beyond bounded window: {terminal:.1f}"
        )

    def test_very_fast_recovery_does_not_overshoot(self, tmp_path):
        eng = _factor_engine(tmp_path)
        res = eng.run_path_scenario(
            self._ui_scenario("Very fast recovery (~1mo)", market_shock=-25)
        )
        mkt = res["factor_paths"]["MKT"]["median"].to_numpy()
        terminal = float(mkt[-1])
        assert 85.0 < terminal < 115.0, (
            f"Very fast recovery overshooting: {terminal:.1f}"
        )

    def test_continued_bear_reverts_after_one_year(self, tmp_path):
        """Continued bear horizon = 1y: drift continues bear for a year,
        then back to initial.  Terminal shouldn't keep falling forever."""
        eng = _factor_engine(tmp_path)
        res = eng.run_path_scenario(
            self._ui_scenario("Continued bear", market_shock=-15)
        )
        mkt = res["factor_paths"]["MKT"]["median"].to_numpy()
        # log change after 1y of bear at log(0.85) drift = log(0.85) ≈ −0.163
        # Terminal stays around 0.85 × (post-shock 0.85 × 100) ≈ 72 — not
        # decaying further with 3+ years of horizon left.
        terminal = float(mkt[-1])
        # If drift hadn't reverted, it would compound to e^(−0.163 × 3) × 85 ≈ 52.
        # With reversion at 1y, terminal sits well above that.
        assert terminal > 60.0, (
            f"Continued bear should revert after 1y; got {terminal:.1f}"
        )

    def test_recovery_horizon_consistent_across_archetypes(self, tmp_path):
        """For each archetype, the engine schema event must include a
        ``next_drift_horizon_years`` matching the archetype table."""
        eng = _factor_engine(tmp_path)
        for archetype, (_sign, horizon) in EVENT_RECOVERY_ARCHETYPES.items():
            ui = self._ui_scenario(archetype, market_shock=-10)
            engine_form = eng._normalise_scenario(ui)
            ev = engine_form["events"][0]
            assert ev["next_drift_horizon_years"] == pytest.approx(horizon)

    def test_initial_drift_resumes_after_horizon(self, tmp_path):
        """If initial market state = Bull and recovery is Fast,
        after the 6-month recovery window the path drifts upward at the
        bull rate — not at the recovery rate, and not flat.

        Bounds are derived from the live MKT drift rather than hard-
        coded (drifts are now per-factor from historical premiums).
        """
        import numpy as np

        eng = _factor_engine(tmp_path)
        res = eng.run_path_scenario(self._ui_scenario(
            "Fast recovery (~6mo)",
            market_shock=-15,
            initial="Bull",
        ))
        mkt = res["factor_paths"]["MKT"]["median"].to_numpy()
        terminal = float(mkt[-1])

        # After ~6mo at recovery drift returning us near 100, remaining
        # ~3y compounds at the bull-regime MKT drift.  Allow ±25 around
        # the deterministic expectation to absorb path-median noise.  The
        # bull drift comes from the same controlled table the engine uses.
        bull_mkt_drift = _neutral_premiums().loc["bull", "MKT"]
        expected_terminal = 100.0 * float(np.exp(bull_mkt_drift * 3.0))
        lo, hi = expected_terminal - 25.0, expected_terminal + 25.0
        assert lo < terminal < hi, (
            f"Bull-state resumption after recovery expected near "
            f"{expected_terminal:.0f} (±25); got {terminal:.1f}"
        )
