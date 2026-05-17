"""Tests for ``vectorised_autocallable_rc_summary``.

Covers the autocall mechanic plus the engine dispatch in both
``ScenarioEngine`` and ``FactorScenarioEngine``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.autocallable_reverse_convertible import (
    vectorised_autocallable_rc_summary,
)
from src.reverse_convertible import vectorised_european_rc_summary
from tests.conftest import make_brc_row, make_mbrc_row


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _make_ac_row(
    coupon=0.08,
    strike=100.0,
    notional=100_000,
    initial_fixing_date="2024-01-01",
    maturity_date="2028-06-01",
    obs_dates=("2024-04-01", "2024-07-01", "2024-10-01",
                "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"),
    trigger=1.00,
):
    """Single-underlying autocallable BRC row."""
    return pd.Series({
        "product_id":              "AC_BRC_001",
        "product_type":            "AC_BRC",
        "type_style":              "european",
        "currency":                "CHF",
        "notional":                notional,
        "position_units":          1,
        "cost_price":              1.0,
        "coupon":                  coupon,
        "barrier_pct":             0.60,
        "underlyings":             ["NESN"],
        "underlying_isins":        ["CH0012221716"],
        "initial_levels":          [100.0],
        "strike":                  [strike],
        "current_spots":           [95.0],
        "initial_fixing_date":     initial_fixing_date,
        "maturity_date":           maturity_date,
        "autocall_obs_dates":      list(obs_dates),
        "autocall_trigger_pct":    trigger,
        "autocall_coupon_memory":  False,
    })


def _make_ac_mbrc_row(strikes=(100.0, 80.0), trigger=1.0,
                      initial_fixing_date="2024-01-01",
                      maturity_date="2026-01-01"):
    """Two-underlying autocallable BRC row."""
    return pd.Series({
        "product_id":              "AC_MBRC_001",
        "product_type":            "AC_BRC",
        "type_style":              "european",
        "currency":                "CHF",
        "notional":                100_000,
        "position_units":          1,
        "cost_price":              1.0,
        "coupon":                  0.08,
        "barrier_pct":             0.60,
        "underlyings":             ["NESN", "NOVN"],
        "underlying_isins":        ["CH0012221716", "CH0012221717"],
        "initial_levels":          [100.0, 80.0],
        "strike":                  list(strikes),
        "current_spots":           [95.0, 72.0],
        "initial_fixing_date":     initial_fixing_date,
        "maturity_date":           maturity_date,
        "autocall_obs_dates":      ["2024-04-01", "2024-07-01", "2024-10-01",
                                     "2025-01-01", "2025-04-01", "2025-07-01"],
        "autocall_trigger_pct":    trigger,
        "autocall_coupon_memory":  False,
    })


def _grid(start="2024-01-02", end="2028-06-01"):
    return pd.bdate_range(start=start, end=end)


def _flat_paths(grid, n_assets, levels, n_paths=1):
    """Constant-level price tensor (no diffusion)."""
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    arr = np.broadcast_to(levels, (n_paths, len(grid), n_assets)).copy()
    return arr


# ──────────────────────────────────────────────────────────────────────────
# 1. Always-above-trigger → called at the FIRST observation
# ──────────────────────────────────────────────────────────────────────────

class TestEarliestCall:

    def test_above_trigger_throughout_calls_at_first_obs(self):
        row  = _make_ac_row()
        grid = _grid()
        # Spot pinned at 110 (10 % above the strike) for every business day.
        paths = _flat_paths(grid, n_assets=1, levels=[110.0])

        v = vectorised_autocallable_rc_summary(row, paths, grid)

        assert v["autocalled"].all()
        assert v["call_date"][0] == pd.Timestamp("2024-04-01")
        assert v["settlement_type"][0] == "autocalled"
        assert not v["barrier_breached"].any()

    def test_call_payoff_is_par_plus_prorata_coupon(self):
        row   = _make_ac_row(coupon=0.08, notional=100_000,
                              initial_fixing_date="2024-01-01")
        grid  = _grid()
        paths = _flat_paths(grid, n_assets=1, levels=[110.0])

        v = vectorised_autocallable_rc_summary(row, paths, grid)

        # First obs is 2024-04-01 → 91 days from initial fixing → T ≈ 91/360
        T_call = (pd.Timestamp("2024-04-01") - pd.Timestamp("2024-01-01")).days / 360
        expected_payoff = 100_000 * (1.0 + 0.08 * T_call)
        assert v["total_payoff"][0] == pytest.approx(expected_payoff, rel=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 2. Drop below then recover → call only on the recovery obs (no retro)
# ──────────────────────────────────────────────────────────────────────────

class TestPathDependentCall:

    def test_drop_then_recover_calls_after_recovery_only(self):
        row  = _make_ac_row()
        grid = _grid()
        # Below trigger at obs 1, 2, 3; above at obs 4 (2025-01-01).
        levels = np.full(len(grid), 110.0)
        below_mask = (grid <= pd.Timestamp("2024-12-15"))
        levels[below_mask] = 80.0
        paths = levels.reshape(1, len(grid), 1)

        v = vectorised_autocallable_rc_summary(row, paths, grid)

        assert v["autocalled"][0]
        # Snapped to the obs-date 2025-01-01 (Wed = bday).
        assert v["call_date"][0] == pd.Timestamp("2025-01-01")


# ──────────────────────────────────────────────────────────────────────────
# 3. Always-below-trigger, no barrier breach → standard BRC at maturity
# ──────────────────────────────────────────────────────────────────────────

class TestNeverCalled:

    def test_never_calls_pays_full_coupon(self):
        """Uncalled paths receive the *full-life* coupon at maturity, the
        same as a vanilla BRC would.  (The breach side of the payoff is
        covered separately — see ``test_never_calls_with_breach_matches_brc``.)
        """
        row  = _make_ac_row(coupon=0.08, notional=100_000)
        grid = _grid()
        # Spot stays at 90 — below strike (100) so trigger never fires.
        paths = _flat_paths(grid, n_assets=1, levels=[90.0])

        v = vectorised_autocallable_rc_summary(row, paths, grid)
        brc = vectorised_european_rc_summary(row, paths[:, -1, :])

        assert not v["autocalled"].any()
        # Total payoff matches the standard BRC at-maturity payoff exactly.
        np.testing.assert_array_equal(v["total_payoff"], brc["total_payoff"])

    def test_never_calls_with_breach_matches_brc(self):
        row   = _make_ac_row()
        grid  = _grid()
        # Spot 50 — never above strike 100, breaches barrier 60.
        paths = _flat_paths(grid, n_assets=1, levels=[50.0])

        v = vectorised_autocallable_rc_summary(row, paths, grid)

        # Compare to running the European BRC helper directly on the
        # same terminal slice — should produce identical core fields.
        terminal = paths[:, -1, :]
        brc      = vectorised_european_rc_summary(row, terminal)

        assert not v["autocalled"][0]
        assert v["barrier_breached"][0] == brc["barrier_breached"][0]
        np.testing.assert_array_equal(v["pnl"], brc["pnl"])
        np.testing.assert_array_equal(v["total_payoff"], brc["total_payoff"])


# ──────────────────────────────────────────────────────────────────────────
# 4. Worst-of: a single weak underlying blocks the call
# ──────────────────────────────────────────────────────────────────────────

class TestWorstOfBlocksCall:

    def test_one_weak_asset_blocks_call(self):
        row  = _make_ac_mbrc_row(strikes=[100.0, 80.0])
        grid = _grid()
        # Asset 1 above its strike (110), asset 2 below (70 < 80).
        # Trigger 1.0 on worst-of strike → worst is 70/80 = 0.875 < 1 → no call.
        levels = np.broadcast_to(
            np.array([110.0, 70.0]), (1, len(grid), 2)
        ).copy()

        v = vectorised_autocallable_rc_summary(row, levels, grid)

        assert not v["autocalled"][0]


# ──────────────────────────────────────────────────────────────────────────
# 5. Trigger sensitivity
# ──────────────────────────────────────────────────────────────────────────

class TestTriggerSensitivity:

    def test_higher_trigger_calls_less_often(self):
        # Spot 104 — above strike 100 (≥ trigger 1.0) but below 1.05 × strike.
        row_lo   = _make_ac_row(trigger=1.00)
        row_hi   = _make_ac_row(trigger=1.05)
        grid     = _grid()
        paths    = _flat_paths(grid, n_assets=1, levels=[104.0])

        v_lo = vectorised_autocallable_rc_summary(row_lo, paths, grid)
        v_hi = vectorised_autocallable_rc_summary(row_hi, paths, grid)

        assert v_lo["autocalled"][0]
        assert not v_hi["autocalled"][0]

    def test_sub_strike_trigger_calls_even_when_below_strike(self):
        # Spot 95 — below strike, but above 0.9 × strike.
        row   = _make_ac_row(trigger=0.90)
        grid  = _grid()
        paths = _flat_paths(grid, n_assets=1, levels=[95.0])

        v = vectorised_autocallable_rc_summary(row, paths, grid)
        assert v["autocalled"][0]


# ──────────────────────────────────────────────────────────────────────────
# 6. No observation dates → falls through to BRC
# ──────────────────────────────────────────────────────────────────────────

class TestNoObservationDates:

    def test_empty_obs_list_matches_brc(self):
        row = _make_ac_row(obs_dates=())
        grid = _grid()
        paths = _flat_paths(grid, n_assets=1, levels=[105.0])

        v   = vectorised_autocallable_rc_summary(row, paths, grid)
        brc = vectorised_european_rc_summary(row, paths[:, -1, :])

        assert not v["autocalled"].any()
        np.testing.assert_array_equal(v["pnl"], brc["pnl"])
        np.testing.assert_array_equal(v["barrier_breached"], brc["barrier_breached"])


# ──────────────────────────────────────────────────────────────────────────
# 7. Mixed cohort of paths
# ──────────────────────────────────────────────────────────────────────────

class TestMixedCohort:

    def test_some_called_some_not(self):
        row  = _make_ac_row()
        grid = _grid()
        # Path 0: high → calls at first obs; path 1: low → never calls.
        n_days = len(grid)
        paths = np.empty((2, n_days, 1))
        paths[0, :, 0] = 110.0
        paths[1, :, 0] = 90.0

        v = vectorised_autocallable_rc_summary(row, paths, grid)

        assert v["autocalled"][0]
        assert not v["autocalled"][1]
        assert v["settlement_type"][0] == "autocalled"
        assert v["settlement_type"][1] in {"cash", "physical"}

    def test_all_called_yields_positive_returns(self):
        row  = _make_ac_row(coupon=0.08, notional=100_000)
        grid = _grid()
        paths = _flat_paths(grid, n_assets=1, levels=[120.0], n_paths=10)

        v = vectorised_autocallable_rc_summary(row, paths, grid)
        assert v["autocalled"].all()
        # Coupon is positive, cost = par at par → return > 0.
        assert (v["return_pct"] > 0).all()


# ──────────────────────────────────────────────────────────────────────────
# 8. Engine dispatch — both engines pick autocallable when product_type is set
# ──────────────────────────────────────────────────────────────────────────

class TestEngineDispatch:

    def _portfolio_with_autocallable(self):
        ac  = _make_ac_row()
        brc = make_brc_row(maturity_date="2026-01-01")
        return pd.DataFrame([ac, brc])

    def test_single_factor_engine_runs_with_autocallable(self):
        """``ScenarioEngine`` should run end-to-end when one product is
        autocallable — confirming the dispatch path is wired."""
        from src.scenario_engine import ScenarioEngine
        from src.noise_sampler   import NoiseSampler

        portfolio = self._portfolio_with_autocallable()
        beta_map  = {"CH0012221716": 1.0}
        vol_map   = {"CH0012221716": 0.2}

        eng = ScenarioEngine(
            portfolio=portfolio, beta_map=beta_map, vol_map=vol_map,
            n_paths=20, mean_reversion_kappa=0.0,
        )
        today = pd.Timestamp.today().normalize()
        n_days = len(pd.bdate_range(start=today, end=pd.Timestamp("2027-06-01")))
        eng.noise_sampler = NoiseSampler(
            n_paths=20, n_days=n_days, factor_codes=[],
            isins=["CH0012221716"], seed=1,
        )

        scenario = {"market_shock": 0, "n_shocks": 1, "shock_in_days": 30,
                     "pre_shock_drift_pa": 0.05,
                     "post_shock_drift_pa": 0.05}
        res = eng.run_path_scenario(scenario)

        assert "AC_BRC_001" in set(res["product_df"]["product_id"])
        ac_row = res["product_df"][
            res["product_df"]["product_id"] == "AC_BRC_001"
        ].iloc[0]
        assert np.isfinite(ac_row["pnl_mean"])

    def test_factor_engine_runs_with_autocallable(self, tmp_path):
        """Same dispatch test for ``FactorScenarioEngine``."""
        from src.factor_engine          import FACTORS, FactorEngine
        from src.factor_scenario_engine import FactorScenarioEngine
        from src.market_data_engine     import MarketDataEngine

        # Seed a tiny factor DB so the engine boots.
        rng   = np.random.default_rng(0)
        end   = pd.Timestamp.today().normalize()
        dates = pd.bdate_range(end=end, periods=900)
        rows  = []
        for code, (ticker, key, _) in FACTORS.items():
            rets   = rng.normal(0, 0.001, len(dates))
            prices = 100.0 * np.exp(np.cumsum(rets))
            for d, p in zip(dates, prices):
                rows.append({"date": d, "isin": key, "ticker": ticker, "price": p})
        pd.DataFrame(rows).to_csv(tmp_path / "prices.csv", index=False)

        mock_client = MagicMock()
        mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
        mde.fetch_daily_prices = MagicMock(return_value=None)
        fe = FactorEngine(mde)

        portfolio = self._portfolio_with_autocallable()

        # Loadings — synthetic, a single-factor projection on MKT.
        loadings = {
            "CH0012221716": {
                "betas":     {c: (1.0 if c == "MKT" else 0.0) for c in FACTORS},
                "alpha":     0.0, "idio_vol": 0.05,
                "r_squared": 0.6, "n_obs": 750,
            },
        }
        eng = FactorScenarioEngine(
            portfolio=portfolio, loadings=loadings, factor_engine=fe,
            n_paths=10, idio_intensity=0.0, mean_reversion_kappa=0.0,
        )
        scenario = {
            "initial_market_state": "Stable",
            "events": [],
        }
        res = eng.run_path_scenario(scenario)
        assert "AC_BRC_001" in set(res["product_df"]["product_id"])
