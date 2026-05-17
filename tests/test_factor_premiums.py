"""Tests for the per-factor historical premium module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factor_premiums import (
    LEGACY_SCALAR_DRIFTS,
    MIN_OBS_PER_REGIME,
    REGIMES,
    REGIME_THRESHOLDS,
    TRAILING_WINDOW_DAYS,
    classify_regimes,
    get_factor_drift,
    load_or_compute_premiums,
    load_premiums,
    save_premiums,
)


# ──────────────────────────────────────────────────────────────────────────
# classify_regimes — pure function on a series
# ──────────────────────────────────────────────────────────────────────────

class TestClassifyRegimes:

    def _series(self, daily_log_returns):
        idx = pd.bdate_range("2020-01-01", periods=len(daily_log_returns))
        return pd.Series(daily_log_returns, index=idx)

    def test_warmup_window_is_nan(self):
        s = self._series([0.0] * (TRAILING_WINDOW_DAYS + 10))
        out = classify_regimes(s)
        # First (window − 1) entries cannot have a full trailing sum.
        assert out.iloc[: TRAILING_WINDOW_DAYS - 1].isna().all()

    def test_constant_zero_trailing_is_stable(self):
        s = self._series([0.0] * (TRAILING_WINDOW_DAYS + 5))
        out = classify_regimes(s)
        post = out.iloc[TRAILING_WINDOW_DAYS - 1 :]
        assert (post == "stable").all()

    def test_strong_uptrend_is_strong_bull(self):
        # +20%/yr split evenly across the window → trailing sum > +15%.
        daily = 0.20 / TRAILING_WINDOW_DAYS
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        out = classify_regimes(s).dropna()
        assert (out == "strong_bull").all()

    def test_moderate_uptrend_is_moderate_bull(self):
        daily = 0.10 / TRAILING_WINDOW_DAYS  # 10%/yr → in [+5%, +15%)
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        out = classify_regimes(s).dropna()
        assert (out == "moderate_bull").all()

    def test_downtrend_is_bear(self):
        daily = -0.20 / TRAILING_WINDOW_DAYS
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        out = classify_regimes(s).dropna()
        assert (out == "bear").all()


# ──────────────────────────────────────────────────────────────────────────
# Persistence round-trip
# ──────────────────────────────────────────────────────────────────────────

class TestPersistence:

    def _toy_premiums(self):
        return pd.DataFrame(
            {"MKT": [-0.10, 0.0, 0.08, 0.15], "FX": [0.02, 0.0, -0.01, -0.03]},
            index=pd.Index(list(REGIMES), name="regime"),
        )

    def test_save_then_load_round_trip(self, tmp_path):
        path = tmp_path / "factor_premiums.csv"
        df = self._toy_premiums()
        save_premiums(df, path)
        loaded = load_premiums(path)
        pd.testing.assert_frame_equal(df, loaded, check_dtype=False)

    def test_load_or_compute_reads_cache_when_present(self, tmp_path):
        path = tmp_path / "factor_premiums.csv"
        save_premiums(self._toy_premiums(), path)
        # No factor_engine needed when the cache is present.
        out = load_or_compute_premiums(factor_engine=None, csv_path=path)
        pd.testing.assert_frame_equal(out, self._toy_premiums(), check_dtype=False)

    def test_load_or_compute_raises_without_engine_when_cache_missing(self, tmp_path):
        path = tmp_path / "does_not_exist.csv"
        with pytest.raises(RuntimeError, match="missing"):
            load_or_compute_premiums(factor_engine=None, csv_path=path)


# ──────────────────────────────────────────────────────────────────────────
# get_factor_drift — the public lookup
# ──────────────────────────────────────────────────────────────────────────

class TestGetFactorDrift:

    def test_known_regime_returns_per_factor_dict(self):
        premiums = pd.DataFrame(
            {"MKT": [-0.05, 0.01, 0.10, 0.20], "FX": [0.02, 0.0, 0.0, -0.05]},
            index=pd.Index(list(REGIMES), name="regime"),
        )
        d = get_factor_drift("strong_bull", ["MKT", "FX"], premiums=premiums)
        assert d == {"MKT": 0.20, "FX": -0.05}

    def test_unknown_factor_falls_back_to_legacy_scalar(self):
        premiums = pd.DataFrame(
            {"MKT": [-0.05, 0.01, 0.10, 0.20]},
            index=pd.Index(list(REGIMES), name="regime"),
        )
        d = get_factor_drift("strong_bull", ["MKT", "GOLD"], premiums=premiums)
        assert d["MKT"] == 0.20
        assert d["GOLD"] == LEGACY_SCALAR_DRIFTS["strong_bull"]

    def test_unknown_regime_raises(self):
        with pytest.raises(ValueError, match="Unknown regime"):
            get_factor_drift("euphoria", ["MKT"], premiums=pd.DataFrame())

    def test_missing_cache_falls_back_to_legacy_scalar(self, tmp_path, monkeypatch):
        # Point the default CSV path to a nonexistent file so load fails.
        from src import factor_premiums as fp
        monkeypatch.setattr(fp, "DEFAULT_CSV_PATH", tmp_path / "missing.csv")
        d = get_factor_drift("bear", ["MKT", "FX"])
        assert d == {"MKT": LEGACY_SCALAR_DRIFTS["bear"],
                     "FX":  LEGACY_SCALAR_DRIFTS["bear"]}


# ──────────────────────────────────────────────────────────────────────────
# Integration with scenario_archetypes — public contract still holds
# ──────────────────────────────────────────────────────────────────────────

class TestIntegrationWithScenarioArchetypes:

    def test_initial_drift_dict_returns_per_factor_values(self):
        from src.factor_engine import FACTORS
        from src.scenario_archetypes import initial_drift_dict
        d = initial_drift_dict("Strong bull", FACTORS)
        assert set(d) == set(FACTORS)
        assert all(isinstance(v, float) for v in d.values())

    def test_initial_drift_dict_is_not_uniform_when_cache_present(self):
        """If the cached CSV exists with real data, different factors
        should generally have different drifts within a regime — the whole
        point of the refactor."""
        from pathlib import Path
        from src.factor_engine import FACTORS
        from src.scenario_archetypes import initial_drift_dict
        if not Path("data/factor_premiums.csv").exists():
            pytest.skip("No cached premiums available")
        d = initial_drift_dict("Strong bull", FACTORS)
        values = list(d.values())
        # Allow exact equality only for the data-sparse fallback case.
        assert len(set(values)) > 1, (
            "All factors got identical drifts — either the cache is the "
            "scalar fallback or the regime has all-equal historical means."
        )

    def test_regime_thresholds_partition_real_line(self):
        """Thresholds should cover all of (-inf, +inf) with no gaps."""
        sorted_ranges = sorted(REGIME_THRESHOLDS.values())
        assert sorted_ranges[0][0] == -np.inf
        assert sorted_ranges[-1][1] == +np.inf
        for (a, b), (c, d) in zip(sorted_ranges, sorted_ranges[1:]):
            assert b == c, f"gap or overlap between {b} and {c}"
