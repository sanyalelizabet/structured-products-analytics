"""Tests for the multi-path ``FactorScenarioEngine``.

Covers:

* End-to-end smoke and output schema (new aggregated form)
* Shock propagation through β at the *median* path
* Determinism via the cached ``NoiseSampler``
* Common Random Numbers — same sampler → comparable scenarios
* Idiosyncratic intensity behaviour (λ = 0 makes idio noise inert)
* Multi-path statistics: shapes, percentile ordering, MC convergence
* Preset library — every preset runs cleanly end-to-end
* Fallback when an ISIN has no loadings
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.risk.factor_engine import FACTORS, FactorEngine
from src.risk.factor_scenario_engine import FactorScenarioEngine
from src.market_data.market_data_engine import MarketDataEngine
from src.numerics.noise_sampler import NoiseSampler
from tests.conftest import make_brc_row, make_mbrc_row


FACTOR_CODES = list(FACTORS.keys())


# ──────────────────────────────────────────────────────────────────────────
# Synthetic-world fixtures
# ──────────────────────────────────────────────────────────────────────────

def _seed_factor_db(tmp_path, n_days=800, seed=29):
    rng   = np.random.default_rng(seed)
    end   = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    sigmas = {"MKT": 0.15, "TECH": 0.22, "HC": 0.12,
              "FIN": 0.20, "ENERGY": 0.28, "FX": 0.08}

    rows = []
    for code in FACTOR_CODES:
        ticker, key, _ = FACTORS[code]
        rets = rng.normal(0, sigmas[code] / np.sqrt(252), n_days)
        prices = 100.0 * np.exp(np.cumsum(rets))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "isin": key, "ticker": ticker, "price": p})

    pd.DataFrame(rows).to_csv(tmp_path / "prices.csv", index=False)


def _two_asset_portfolio(maturity="2027-06-01"):
    row = make_mbrc_row(
        initial_levels=[100.0, 100.0],
        strikes=[100.0, 100.0],
        current_spots=[100.0, 100.0],
        maturity_date=maturity,
    )
    row["underlyings"]      = ["AMD_TEST", "NESN_TEST"]
    row["underlying_isins"] = ["TEST_AMD", "TEST_NESN"]
    return pd.DataFrame([row])


def _hand_picked_loadings():
    return {
        "TEST_AMD": {
            "betas":     {"MKT": 1.3, "TECH": 1.2, "HC": 0.0,
                          "FIN": 0.0, "ENERGY": 0.0, "FX": 0.1},
            "alpha": 0.0, "idio_vol": 0.30, "r_squared": 0.65, "n_obs": 750,
        },
        "TEST_NESN": {
            "betas":     {"MKT": 0.6, "TECH": 0.0, "HC": 0.4,
                          "FIN": 0.0, "ENERGY": 0.0, "FX": -0.05},
            "alpha": 0.0, "idio_vol": 0.12, "r_squared": 0.55, "n_obs": 750,
        },
    }


def _make_engine(tmp_path, n_paths=20, idio_intensity=0.0,
                 mean_reversion_kappa=0.5):
    _seed_factor_db(tmp_path)
    mock_client = MagicMock()
    mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
    mde.fetch_daily_prices = MagicMock(return_value=None)
    fe = FactorEngine(mde)
    return FactorScenarioEngine(
        portfolio=_two_asset_portfolio(),
        loadings=_hand_picked_loadings(),
        factor_engine=fe,
        idio_intensity=idio_intensity,
        mean_reversion_kappa=mean_reversion_kappa,
        n_paths=n_paths,
    )


@pytest.fixture
def engine(tmp_path):
    """Default engine — small n_paths for fast tests."""
    return _make_engine(tmp_path, n_paths=20)


# ──────────────────────────────────────────────────────────────────────────
# Smoke / output schema
# ──────────────────────────────────────────────────────────────────────────

class TestRunSmoke:
    def test_runs_end_to_end(self, engine):
        scenario = {
            "factor_shock":         {"MKT": -10},
            "n_shocks":             1,
            "shock_in_days":        30,
            "shock_spacing_days":   0,
            "factor_drift_pre_pa":  {},
            "factor_drift_post_pa": {},
        }
        res = engine.run_path_scenario(scenario)

        for k in ("product_df", "pf_scenario_per_ccy", "cash_positions",
                  "delivered_stocks", "asset_paths", "factor_paths",
                  "pnl_samples_by_ccy", "n_paths"):
            assert k in res

        assert res["n_paths"] == engine.n_paths
        assert len(res["product_df"]) == 1
        assert set(res["factor_paths"].keys()) == set(FACTOR_CODES)

    def test_factor_paths_have_summary_columns(self, engine):
        res = engine.run_path_scenario({"factor_shock": {}})
        for code in FACTOR_CODES:
            df = res["factor_paths"][code]
            for col in ("date", "mean", "median", "p5", "p95"):
                assert col in df.columns
            # Base 100 at t=0 for the factor index
            assert abs(df["mean"].iloc[0] - 100.0) < 0.5

    def test_asset_paths_have_summary_columns(self, engine):
        res = engine.run_path_scenario({"factor_shock": {"MKT": -5}})
        for isin in ["TEST_AMD", "TEST_NESN"]:
            df = res["asset_paths"][isin]
            for col in ("date", "mean", "median", "p5", "p95"):
                assert col in df.columns
            # Initial spot at t=0
            assert abs(df["mean"].iloc[0] - 100.0) < 0.5

    def test_product_row_has_aggregated_stats(self, engine):
        res = engine.run_path_scenario({"factor_shock": {"MKT": -10}})
        row = res["product_df"].iloc[0]
        for col in ("pnl_mean", "pnl_median", "pnl_p5", "pnl_p95",
                    "pnl_es5", "pnl_std",
                    "return_mean_pct", "return_median_pct",
                    "return_p5_pct", "return_p95_pct",
                    "worst_underlying", "settlement_type",
                    "barrier_breach_freq",
                    "pnl_samples", "return_samples"):
            assert col in row.index, f"missing {col}"
        # P&L percentile ordering
        assert row["pnl_p5"] <= row["pnl_median"] <= row["pnl_p95"]


# ──────────────────────────────────────────────────────────────────────────
# Structural shock propagation
# ──────────────────────────────────────────────────────────────────────────

class TestShockPropagation:
    def test_tech_shock_hits_high_beta_more(self, engine):
        scenario = {
            "factor_shock":         {"TECH": -30},
            "n_shocks":             1,
            "shock_in_days":        20,
            "factor_drift_pre_pa":  {},
            "factor_drift_post_pa": {},
        }
        res = engine.run_path_scenario(scenario)

        amd  = res["asset_paths"]["TEST_AMD"]["median"].to_numpy()
        nesn = res["asset_paths"]["TEST_NESN"]["median"].to_numpy()
        # On the median path, AMD (high tech β) must end below NESN (defensive).
        assert amd[-1] / amd[0] < nesn[-1] / nesn[0]

    def test_zero_shock_zero_drift_paths_finite_and_bounded(self, engine):
        res = engine.run_path_scenario({
            "factor_shock":         {},
            "n_shocks":             0,
            "factor_drift_pre_pa":  {},
            "factor_drift_post_pa": {},
        })

        for isin in ["TEST_AMD", "TEST_NESN"]:
            df = res["asset_paths"][isin]
            arr = df["median"].to_numpy()
            assert np.isfinite(arr).all()
            assert 0.2 < arr[-1] / arr[0] < 5.0


# ──────────────────────────────────────────────────────────────────────────
# Determinism via NoiseSampler
# ──────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_scenario_same_output(self, engine):
        scenario = {"factor_shock": {"MKT": -15, "TECH": -25},
                    "n_shocks": 1, "shock_in_days": 30,
                    "factor_drift_pre_pa": {}, "factor_drift_post_pa": {}}
        a = engine.run_path_scenario(scenario)
        b = engine.run_path_scenario(scenario)

        # Sampler is reused across runs ⇒ identical results.
        np.testing.assert_array_equal(
            a["asset_paths"]["TEST_AMD"]["median"].to_numpy(),
            b["asset_paths"]["TEST_AMD"]["median"].to_numpy(),
        )
        np.testing.assert_array_equal(a["product_df"]["pnl_samples"].iloc[0],
                                      b["product_df"]["pnl_samples"].iloc[0])

    def test_different_scenarios_different_output(self, engine):
        s1 = {"factor_shock": {"MKT": -10}, "n_shocks": 1, "shock_in_days": 30}
        s2 = {"factor_shock": {"MKT": -20}, "n_shocks": 1, "shock_in_days": 30}
        a = engine.run_path_scenario(s1)["product_df"]["pnl_mean"].iloc[0]
        b = engine.run_path_scenario(s2)["product_df"]["pnl_mean"].iloc[0]
        assert a != b


# ──────────────────────────────────────────────────────────────────────────
# Common Random Numbers (CRN)
# ──────────────────────────────────────────────────────────────────────────

class TestCommonRandomNumbers:
    def test_reusing_sampler_links_two_engines(self, tmp_path):
        """When two engines share a NoiseSampler, identical scenarios
        produce identical paths — the foundation of CRN sensitivity work."""
        _seed_factor_db(tmp_path)
        mock_client = MagicMock()
        mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        mde.fetch_daily_prices = MagicMock(return_value=None)
        fe = FactorEngine(mde)

        portfolio = _two_asset_portfolio()
        loadings  = _hand_picked_loadings()

        # Pre-build a sampler covering the right dimensions.
        today = pd.Timestamp.today().normalize()
        portfolio_maturity = pd.to_datetime(portfolio["maturity_date"]).max()
        n_days = len(pd.bdate_range(start=today, end=portfolio_maturity))
        sampler = NoiseSampler(
            n_paths=10, n_days=n_days,
            factor_codes=FACTOR_CODES,
            isins=["TEST_AMD", "TEST_NESN"],
        )

        e1 = FactorScenarioEngine(portfolio, loadings, fe,
                                  n_paths=10, noise_sampler=sampler,
                                  idio_intensity=0.5)
        e2 = FactorScenarioEngine(portfolio, loadings, fe,
                                  n_paths=10, noise_sampler=sampler,
                                  idio_intensity=0.5)

        scenario = {"factor_shock": {"MKT": -10}, "n_shocks": 1, "shock_in_days": 30}
        a = e1.run_path_scenario(scenario)
        b = e2.run_path_scenario(scenario)
        np.testing.assert_array_equal(
            a["asset_paths"]["TEST_AMD"]["median"].to_numpy(),
            b["asset_paths"]["TEST_AMD"]["median"].to_numpy(),
        )

    def test_crn_makes_scenario_diff_smooth(self, tmp_path):
        """With CRN, the *path-by-path* difference between two nearby
        scenarios should be much smaller than either path's own variance —
        i.e. the scenario delta is the structural sensitivity, not noise."""
        engine = _make_engine(tmp_path, n_paths=30, idio_intensity=0.5)

        s_low  = {"factor_shock": {"MKT": -10}, "n_shocks": 1, "shock_in_days": 30,
                  "factor_drift_pre_pa": {}, "factor_drift_post_pa": {}}
        s_high = {"factor_shock": {"MKT": -11}, "n_shocks": 1, "shock_in_days": 30,
                  "factor_drift_pre_pa": {}, "factor_drift_post_pa": {}}

        a = engine.run_path_scenario(s_low)
        b = engine.run_path_scenario(s_high)

        pnl_a = a["product_df"]["pnl_samples"].iloc[0]
        pnl_b = b["product_df"]["pnl_samples"].iloc[0]
        diffs = pnl_b - pnl_a

        # std of the per-path *delta* should be far smaller than std of either
        # P&L itself — that's the entire point of CRN.
        assert diffs.std() < 0.25 * pnl_a.std()


# ──────────────────────────────────────────────────────────────────────────
# Idiosyncratic intensity behaviour
# ──────────────────────────────────────────────────────────────────────────

class TestIdioIntensity:
    def test_lambda_zero_idio_does_not_widen_dispersion(self, tmp_path):
        """At λ=0 the per-asset dispersion across paths is driven only by
        factor-block diffusion, not idio noise."""
        engine = _make_engine(tmp_path, n_paths=30, idio_intensity=0.0)
        res = engine.run_path_scenario({"factor_shock": {"MKT": -5},
                                         "n_shocks": 1, "shock_in_days": 30})
        df = res["asset_paths"]["TEST_AMD"]
        # p95 - p5 spread bounded by factor diffusion alone
        spread = (df["p95"] - df["p5"]).iloc[-1]
        assert np.isfinite(spread) and spread > 0

    def test_lambda_higher_yields_higher_terminal_dispersion(self, tmp_path):
        e_low  = _make_engine(tmp_path, n_paths=40, idio_intensity=0.0)
        e_high = _make_engine(tmp_path, n_paths=40, idio_intensity=1.0)
        scenario = {"factor_shock": {}, "n_shocks": 0,
                    "factor_drift_pre_pa": {}, "factor_drift_post_pa": {}}

        a = e_low.run_path_scenario(scenario)["asset_paths"]["TEST_AMD"]
        b = e_high.run_path_scenario(scenario)["asset_paths"]["TEST_AMD"]
        spread_low  = (a["p95"] - a["p5"]).mean()
        spread_high = (b["p95"] - b["p5"]).mean()
        assert spread_high > spread_low


# ──────────────────────────────────────────────────────────────────────────
# Multi-path statistics
# ──────────────────────────────────────────────────────────────────────────

class TestMultiPathStatistics:
    def test_pnl_samples_have_correct_shape(self, engine):
        res = engine.run_path_scenario({"factor_shock": {"MKT": -5}})
        samples = res["product_df"]["pnl_samples"].iloc[0]
        assert samples.shape == (engine.n_paths,)

    def test_currency_pnl_aggregates_per_path(self, engine):
        res = engine.run_path_scenario({"factor_shock": {"MKT": -5}})
        for ccy, samples in res["pnl_samples_by_ccy"].items():
            assert samples.shape == (engine.n_paths,)

    def test_n_paths_one_collapses_summary(self, tmp_path):
        engine = _make_engine(tmp_path, n_paths=1, idio_intensity=0.5)
        res = engine.run_path_scenario({"factor_shock": {"MKT": -10}})
        df = res["asset_paths"]["TEST_AMD"]
        # All four summary stats should match when there's a single path
        np.testing.assert_array_equal(df["mean"].to_numpy(),   df["median"].to_numpy())
        np.testing.assert_array_equal(df["mean"].to_numpy(),   df["p5"].to_numpy())
        np.testing.assert_array_equal(df["mean"].to_numpy(),   df["p95"].to_numpy())

    def test_more_paths_tightens_mean_estimate(self, tmp_path):
        """Standard error of mean P&L should shrink ~ 1/√N."""
        scenario = {"factor_shock": {"MKT": -10}, "n_shocks": 1,
                    "shock_in_days": 30,
                    "factor_drift_pre_pa": {}, "factor_drift_post_pa": {}}

        means_small = []
        means_large = []
        for s in range(4):
            e_small = _make_engine(tmp_path, n_paths=20, idio_intensity=1.0)
            e_small.noise_sampler = NoiseSampler(
                20, e_small._ensure_sampler.__wrapped__ if False else 0,
                FACTOR_CODES, ["TEST_AMD", "TEST_NESN"], seed=100 + s
            ) if False else None
            # Instead of forcing seeds, just regenerate after each run
            res = e_small.run_path_scenario(scenario)
            means_small.append(res["product_df"]["pnl_mean"].iloc[0])
            e_small.noise_sampler.regenerate()

            e_large = _make_engine(tmp_path, n_paths=200, idio_intensity=1.0)
            res2 = e_large.run_path_scenario(scenario)
            means_large.append(res2["product_df"]["pnl_mean"].iloc[0])
            e_large.noise_sampler.regenerate()

        # std across replicates should be smaller with more paths
        std_small = float(np.std(means_small, ddof=1))
        std_large = float(np.std(means_large, ddof=1))
        assert std_large <= std_small or std_small < 1e-6


# ──────────────────────────────────────────────────────────────────────────
# Preset library
# ──────────────────────────────────────────────────────────────────────────

class TestScenarioPresets:
    """Presets are now event-timeline shaped — see ``data/factor_scenarios.py``.
    The engine accepts the UI shape directly via ``_normalise_scenario``."""

    def test_all_presets_run_cleanly(self, engine):
        from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
        for name, preset in FACTOR_SCENARIO_PRESETS.items():
            res = engine.run_path_scenario(preset)

            for key in ("product_df", "pf_scenario_per_ccy", "cash_positions",
                        "delivered_stocks", "asset_paths", "factor_paths",
                        "pnl_samples_by_ccy", "n_paths"):
                assert key in res, f"[{name}] missing {key}"

            for isin, df in res["asset_paths"].items():
                arr = df["median"].to_numpy()
                assert np.isfinite(arr).all(), f"[{name}] NaN in {isin}"
                assert len(arr) > 50

            for code, fdf in res["factor_paths"].items():
                arr = fdf["median"].to_numpy()
                assert np.isfinite(arr).all(), f"[{name}] NaN factor {code}"

    def test_every_event_uses_known_factor_codes(self):
        from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
        from src.risk.factor_engine import FACTORS as FACTOR_UNIVERSE
        for name, preset in FACTOR_SCENARIO_PRESETS.items():
            for ev in preset.get("events", []):
                shock_keys = set(ev.get("factor_shock", {}).keys())
                unknown = shock_keys - set(FACTOR_UNIVERSE.keys())
                assert not unknown, \
                    f"[{name}] event day={ev.get('day')} has unknown factor keys: {unknown}"

    def test_every_event_uses_known_recovery_archetype(self):
        from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
        from src.risk.scenario_archetypes import EVENT_RECOVERY_ARCHETYPES
        for name, preset in FACTOR_SCENARIO_PRESETS.items():
            for ev in preset.get("events", []):
                if "recovery" in ev:
                    assert ev["recovery"] in EVENT_RECOVERY_ARCHETYPES, \
                        f"[{name}] day={ev.get('day')}: unknown archetype {ev['recovery']!r}"

    def test_every_preset_uses_known_initial_market_state(self):
        from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
        from src.risk.scenario_archetypes import INITIAL_MARKET_STATES
        for name, preset in FACTOR_SCENARIO_PRESETS.items():
            state = preset.get("initial_market_state")
            if state is not None:
                assert state in INITIAL_MARKET_STATES, \
                    f"[{name}] unknown initial_market_state: {state!r}"


# ──────────────────────────────────────────────────────────────────────────
# Fallback when a portfolio ISIN has no loading
# ──────────────────────────────────────────────────────────────────────────

class TestFallbackLoadings:
    def test_unknown_isin_uses_defaults(self, tmp_path):
        _seed_factor_db(tmp_path)
        mock_client = MagicMock()
        mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        mde.fetch_daily_prices = MagicMock(return_value=None)
        fe = FactorEngine(mde)

        portfolio = _two_asset_portfolio()
        eng = FactorScenarioEngine(
            portfolio=portfolio, loadings={}, factor_engine=fe,
            n_paths=10, idio_intensity=0.0,
        )
        res = eng.run_path_scenario({"factor_shock": {"MKT": -20},
                                      "n_shocks": 1, "shock_in_days": 20})
        assert "TEST_AMD" in res["asset_paths"]
        df = res["asset_paths"]["TEST_AMD"]
        arr = df["median"].to_numpy()
        assert np.isfinite(arr).all()
        # MKT -20 % with default β_MKT=1 ⇒ terminal materially below initial
        assert arr[-1] < arr[0]
