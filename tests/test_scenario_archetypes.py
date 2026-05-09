"""Tests for ``src.scenario_archetypes`` — the UI archetype → numerical
drift translator that couples post-event drift to shock magnitude.
"""
from __future__ import annotations

import math

import pytest

from src.scenario_archetypes import (
    DEFAULT_INITIAL_MARKET_STATE,
    DEFAULT_RECOVERY_ARCHETYPE,
    EVENT_RECOVERY_ARCHETYPES,
    INITIAL_MARKET_STATES,
    event_drift_for_factor,
    event_next_drift_dict,
    initial_drift_dict,
    translate_ui_scenario,
)


FACTORS = ["MKT", "TECH", "HC", "FIN", "ENERGY", "FX"]


# ──────────────────────────────────────────────────────────────────────────
# event_drift_for_factor — the core coupling formula
# ──────────────────────────────────────────────────────────────────────────

class TestEventDriftFormula:

    def test_stable_returns_zero_for_any_shock(self):
        for shock in [-50, -10, 0, 10, 35]:
            assert event_drift_for_factor(shock, "Stable (no drift)") == 0.0

    def test_continued_bear_drift_has_same_sign_as_shock(self):
        assert event_drift_for_factor(-25, "Continued bear") < 0   # crash → drift down
        assert event_drift_for_factor(+30, "Continued bear") > 0   # spike → drift up
        assert event_drift_for_factor(  0, "Continued bear") == 0.0

    def test_recovery_drift_has_opposite_sign_to_shock(self):
        # "Recovery" means drift reverses the shock toward pre-shock level.
        assert event_drift_for_factor(-25, "Slow recovery (~2y)")  > 0
        assert event_drift_for_factor(-25, "Fast recovery (~6mo)") > 0
        assert event_drift_for_factor(+30, "Slow recovery (~2y)")  < 0
        assert event_drift_for_factor(+30, "Fast recovery (~6mo)") < 0

    def test_fast_recovery_is_steeper_than_slow_recovery(self):
        for shock in [-25, -10, +35]:
            slow = event_drift_for_factor(shock, "Slow recovery (~2y)")
            fast = event_drift_for_factor(shock, "Fast recovery (~6mo)")
            # Fast recovery has 4× the magnitude of slow (0.5y vs 2y horizon).
            assert abs(fast) > abs(slow)
            assert abs(fast) == pytest.approx(4 * abs(slow), rel=1e-9)

    def test_very_fast_recovery_is_steeper_than_fast(self):
        """1-month recovery is 6× the magnitude of the 6-month recovery."""
        for shock in [-25, -10, +35]:
            fast      = event_drift_for_factor(shock, "Fast recovery (~6mo)")
            very_fast = event_drift_for_factor(shock, "Very fast recovery (~1mo)")
            assert abs(very_fast) > abs(fast)
            # 0.5y / (1/12)y = 6
            assert abs(very_fast) == pytest.approx(6 * abs(fast), rel=1e-9)

    def test_very_fast_recovery_brings_back_in_one_month(self):
        """A −15% shock + Very fast recovery should snap back in 1 month."""
        shock = -15.0
        drift = event_drift_for_factor(shock, "Very fast recovery (~1mo)")
        log_start = math.log(1 + shock / 100.0)
        log_end   = log_start + drift * (1.0 / 12.0)
        assert log_end == pytest.approx(0.0, abs=1e-9)

    def test_continued_bear_horizon_one_year(self):
        # Continued bear over 1 year: drift = log(1+s/100) / 1
        for shock in [-25, -10, +20]:
            expected = math.log(1 + shock / 100.0) / 1.0
            assert event_drift_for_factor(shock, "Continued bear") == pytest.approx(expected)

    def test_v_shape_recovery_brings_back_to_baseline(self):
        """A −25 % shock + Fast recovery (~6mo) should produce a drift such
        that, after 0.5 years of pure drift, log-level returns to zero."""
        shock = -25.0
        drift = event_drift_for_factor(shock, "Fast recovery (~6mo)")
        # Starting at log(0.75) (= -0.2877), after 0.5y of drift we should
        # be back near log(1) = 0.
        log_start = math.log(1 + shock / 100.0)   # negative
        log_end   = log_start + drift * 0.5
        assert log_end == pytest.approx(0.0, abs=1e-9)

    def test_slow_recovery_brings_back_in_two_years(self):
        shock = -10.0
        drift = event_drift_for_factor(shock, "Slow recovery (~2y)")
        log_start = math.log(1 + shock / 100.0)
        log_end   = log_start + drift * 2.0
        assert log_end == pytest.approx(0.0, abs=1e-9)

    def test_zero_shock_zero_recovery_drift(self):
        """If the shock is zero there is nothing to recover from, so drift = 0
        for every archetype."""
        for arch in EVENT_RECOVERY_ARCHETYPES:
            assert event_drift_for_factor(0.0, arch) == 0.0

    def test_unknown_archetype_raises(self):
        with pytest.raises(ValueError, match="Unknown recovery archetype"):
            event_drift_for_factor(-10, "Sideways for 17 years")

    def test_extreme_negative_shock_is_clamped_to_finite(self):
        # Even a -120 % shock (impossible in reality, guard against UI bug)
        # must not blow up to -infinity.
        d = event_drift_for_factor(-120.0, "Continued bear")
        assert math.isfinite(d)


# ──────────────────────────────────────────────────────────────────────────
# event_next_drift_dict — per-factor application
# ──────────────────────────────────────────────────────────────────────────

class TestEventNextDriftDict:

    def test_keys_match_requested_factors(self):
        d = event_next_drift_dict({"MKT": -10}, "Slow recovery (~2y)", FACTORS)
        assert set(d.keys()) == set(FACTORS)

    def test_factor_with_no_shock_gets_zero_drift(self):
        # Shock dict only mentions MKT; other factors default to 0 shock.
        d = event_next_drift_dict({"MKT": -25}, "Fast recovery (~6mo)", FACTORS)
        assert d["MKT"] != 0
        for c in ["TECH", "HC", "FIN", "ENERGY", "FX"]:
            assert d[c] == 0.0

    def test_per_factor_drifts_match_individual_calls(self):
        shock = {"MKT": -25, "TECH": -30, "ENERGY": +35, "FX": +5}
        archetype = "Slow recovery (~2y)"
        d = event_next_drift_dict(shock, archetype, FACTORS)
        for c in FACTORS:
            expected = event_drift_for_factor(shock.get(c, 0.0), archetype)
            assert d[c] == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────────────
# initial_drift_dict
# ──────────────────────────────────────────────────────────────────────────

class TestInitialDriftDict:

    def test_uniform_drift_across_all_factors(self):
        d = initial_drift_dict("Bull market (+7 %/y)", FACTORS)
        assert all(v == 0.07 for v in d.values())
        assert set(d) == set(FACTORS)

    def test_stable_is_exactly_zero(self):
        d = initial_drift_dict("Stable (0 %)", FACTORS)
        assert all(v == 0.0 for v in d.values())

    def test_unknown_state_raises(self):
        with pytest.raises(ValueError, match="Unknown initial market state"):
            initial_drift_dict("Hopeful but cautious", FACTORS)

    def test_all_listed_states_resolve(self):
        for state in INITIAL_MARKET_STATES:
            d = initial_drift_dict(state, FACTORS)
            assert len(d) == len(FACTORS)


# ──────────────────────────────────────────────────────────────────────────
# translate_ui_scenario — the user-facing entry point
# ──────────────────────────────────────────────────────────────────────────

class TestTranslateUiScenario:

    def test_returns_engine_schema_keys(self):
        ui = {
            "initial_market_state": "Stable (0 %)",
            "events": [
                {"day": 30, "factor_shock": {"MKT": -10}, "recovery": "Slow recovery (~2y)"},
            ],
        }
        out = translate_ui_scenario(ui, FACTORS)
        assert "initial_drift_pa" in out
        assert "events" in out
        for ev in out["events"]:
            assert "day" in ev
            assert "factor_shock" in ev
            assert "next_drift_pa" in ev

    def test_passthrough_keys_preserved(self):
        ui = {
            "label":                "X",
            "description":          "Y",
            "idio_intensity":       0.5,
            "mean_reversion_kappa": 0.7,
            "initial_market_state": "Stable (0 %)",
            "events": [],
        }
        out = translate_ui_scenario(ui, FACTORS)
        assert out["label"] == "X"
        assert out["description"] == "Y"
        assert out["idio_intensity"]       == 0.5
        assert out["mean_reversion_kappa"] == 0.7

    def test_default_archetypes_used_when_missing(self):
        # No initial_market_state, no recovery on the event
        ui = {"events": [{"day": 30, "factor_shock": {"MKT": -10}}]}
        out = translate_ui_scenario(ui, FACTORS)
        # initial drift = default ("Stable") = 0
        assert all(v == 0.0 for v in out["initial_drift_pa"].values())
        # event drift = default ("Stable (no drift)") = 0
        assert all(v == 0.0 for v in out["events"][0]["next_drift_pa"].values())

    def test_event_day_coerced_to_int(self):
        ui = {"events": [{"day": 30.7, "factor_shock": {"MKT": -10}, "recovery": "Stable (no drift)"}]}
        out = translate_ui_scenario(ui, FACTORS)
        assert isinstance(out["events"][0]["day"], int)

    def test_recovery_label_round_tripped(self):
        ui = {"events": [{"day": 30, "factor_shock": {"MKT": -10},
                           "recovery": "Fast recovery (~6mo)"}]}
        out = translate_ui_scenario(ui, FACTORS)
        # The label is preserved alongside the numerical drift for UI display.
        assert out["events"][0]["recovery"] == "Fast recovery (~6mo)"


# ──────────────────────────────────────────────────────────────────────────
# Defaults declared in module
# ──────────────────────────────────────────────────────────────────────────

class TestDefaults:

    def test_default_initial_market_state_is_known(self):
        assert DEFAULT_INITIAL_MARKET_STATE in INITIAL_MARKET_STATES

    def test_default_recovery_archetype_is_known(self):
        assert DEFAULT_RECOVERY_ARCHETYPE in EVENT_RECOVERY_ARCHETYPES
