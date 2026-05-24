"""Tests for the per-factor premium module (3 regimes, mean & shrinkage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factor_premiums import (
    DRIFT_CLIP_BAND,
    MIN_OBS_PER_REGIME,
    REGIME_ERP,
    REGIMES,
    REGIME_THRESHOLDS,
    TRAILING_WINDOW_DAYS,
    classify_regimes,
    compute_factor_premiums,
    csv_path_for_method,
    factor_mkt_betas,
    get_factor_drift,
    load_or_compute_premiums,
    load_premiums,
    save_premiums,
)


class _FakeFactorEngine:
    """Minimal stand-in exposing ``build_returns`` like ``FactorEngine``."""
    def __init__(self, returns: pd.DataFrame):
        self._returns = returns

    def build_returns(self, years=None):
        return self._returns


# ──────────────────────────────────────────────────────────────────────────
# classify_regimes
# ──────────────────────────────────────────────────────────────────────────

class TestClassifyRegimes:

    def _series(self, daily_log_returns):
        idx = pd.bdate_range("2020-01-01", periods=len(daily_log_returns))
        return pd.Series(daily_log_returns, index=idx)

    def test_warmup_window_is_nan(self):
        s = self._series([0.0] * (TRAILING_WINDOW_DAYS + 10))
        out = classify_regimes(s)
        assert out.iloc[: TRAILING_WINDOW_DAYS - 1].isna().all()

    def test_constant_zero_trailing_is_flat(self):
        s = self._series([0.0] * (TRAILING_WINDOW_DAYS + 5))
        out = classify_regimes(s).iloc[TRAILING_WINDOW_DAYS - 1:]
        assert (out == "flat").all()

    def test_strong_uptrend_is_bull(self):
        daily = 0.20 / TRAILING_WINDOW_DAYS  # +20%/yr → trailing > +10%
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        assert (classify_regimes(s).dropna() == "bull").all()

    def test_mild_uptrend_is_flat(self):
        daily = 0.05 / TRAILING_WINDOW_DAYS  # +5%/yr → within [-5%, +10%)
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        assert (classify_regimes(s).dropna() == "flat").all()

    def test_downtrend_is_bear(self):
        daily = -0.20 / TRAILING_WINDOW_DAYS
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        assert (classify_regimes(s).dropna() == "bear").all()

    def test_high_vol_uptrend_is_not_bull(self):
        # Uptrend but elevated vol → not a clean bull; falls to flat (no down move).
        daily = 0.20 / TRAILING_WINDOW_DAYS
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        hi_vol = pd.Series(0.40, index=s.index)
        out = classify_regimes(s, vol_signal=hi_vol).dropna()
        assert (out == "flat").all()          # high-vol melt-up ≠ bull, ≠ bear

    def test_down_stress_overrides_to_bear(self):
        # Trend in the flat band, but a recent high-vol selloff → bear.
        n = TRAILING_WINDOW_DAYS + 30
        vals = [0.0] * n
        for i in range(n - 25, n):            # recent dip → short_ret < 0
            vals[i] = -0.001
        s = self._series(vals)
        hi_vol = pd.Series(0.40, index=s.index)
        out = classify_regimes(s, vol_signal=hi_vol)
        assert out.iloc[-1] == "bear"         # falling + high vol → bear
        assert out.iloc[TRAILING_WINDOW_DAYS] == "flat"  # calm-by-return pre-dip day

    def test_bull_requires_calm(self):
        # Uptrend with calm vol → bull (trend wins when not stressed).
        daily = 0.20 / TRAILING_WINDOW_DAYS
        s = self._series([daily] * (TRAILING_WINDOW_DAYS + 5))
        low_vol = pd.Series(0.10, index=s.index)
        out = classify_regimes(s, vol_signal=low_vol).dropna()
        assert (out == "bull").all()


# ──────────────────────────────────────────────────────────────────────────
# factor_mkt_betas
# ──────────────────────────────────────────────────────────────────────────

class TestFactorMktBetas:

    def test_beta_is_covariance_ratio(self):
        rng = np.random.default_rng(0)
        mkt = rng.normal(0.0, 0.01, 500)
        df = pd.DataFrame({"MKT": mkt, "F2": 3.0 * mkt})  # exact β=3
        betas = factor_mkt_betas(df)
        assert abs(betas["MKT"] - 1.0) < 1e-12
        assert abs(betas["F2"] - 3.0) < 1e-9

    def test_zero_variance_mkt_is_safe(self):
        df = pd.DataFrame({"MKT": [0.0] * 10, "F2": [0.0] * 10})
        betas = factor_mkt_betas(df)
        assert betas == {"MKT": 1.0, "F2": 0.0}


# ──────────────────────────────────────────────────────────────────────────
# compute_factor_premiums — mean vs shrinkage
# ──────────────────────────────────────────────────────────────────────────

class TestComputePremiums:

    def _uptrend_returns(self, n=400, seed=0):
        """All-bull sample: bear & flat regimes end up empty (n=0)."""
        rng = np.random.default_rng(seed)
        drift = 0.20 / TRAILING_WINDOW_DAYS
        mkt = rng.normal(drift, drift * 0.05, n)   # tiny noise, stays bull
        idx = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame({"MKT": mkt, "F2": 3.0 * mkt}, index=idx)

    def test_mean_sparse_regime_is_flat_erp_scalar(self):
        fe = _FakeFactorEngine(self._uptrend_returns())
        df = compute_factor_premiums(fe, method="mean")
        # bear has no observations → flat ERP scalar across all factors.
        assert df.loc["bear", "MKT"] == REGIME_ERP["bear"]
        assert df.loc["bear", "F2"] == REGIME_ERP["bear"]

    def test_shrinkage_empty_regime_is_beta_times_erp(self):
        fe = _FakeFactorEngine(self._uptrend_returns())
        df = compute_factor_premiums(fe, method="shrinkage")
        lo, hi = DRIFT_CLIP_BAND
        # bear empty → prior β_f · ERP_bear (clipped). β_MKT=1, β_F2≈3.
        assert abs(df.loc["bear", "MKT"] - REGIME_ERP["bear"]) < 1e-6      # 1·(−0.10)
        assert df.loc["bear", "F2"] == pytest.approx(lo)                   # 3·(−0.10)→clip
        # ...and unlike 'mean', the two factors differ (per-factor prior).
        assert df.loc["bear", "MKT"] != df.loc["bear", "F2"]

    def test_index_is_regimes_and_columns_are_factors(self):
        fe = _FakeFactorEngine(self._uptrend_returns())
        df = compute_factor_premiums(fe, method="shrinkage")
        assert list(df.index) == list(REGIMES)
        assert set(df.columns) == {"MKT", "F2"}

    def test_unknown_method_raises(self):
        fe = _FakeFactorEngine(self._uptrend_returns())
        with pytest.raises(ValueError, match="Unknown method"):
            compute_factor_premiums(fe, method="bogus")


# ──────────────────────────────────────────────────────────────────────────
# Persistence round-trip
# ──────────────────────────────────────────────────────────────────────────

class TestPersistence:

    def _toy_premiums(self):
        return pd.DataFrame(
            {"MKT": [-0.10, 0.0, 0.12], "FX": [0.02, 0.0, -0.03]},
            index=pd.Index(list(REGIMES), name="regime"),
        )

    def test_save_then_load_round_trip(self, tmp_path):
        path = tmp_path / "factor_premiums.csv"
        df = self._toy_premiums()
        save_premiums(df, path)
        pd.testing.assert_frame_equal(df, load_premiums(path), check_dtype=False)

    def test_load_or_compute_reads_cache_when_present(self, tmp_path):
        path = tmp_path / "factor_premiums.csv"
        save_premiums(self._toy_premiums(), path)
        out = load_or_compute_premiums(factor_engine=None, csv_path=path)
        pd.testing.assert_frame_equal(out, self._toy_premiums(), check_dtype=False)

    def test_load_or_compute_raises_without_engine_when_cache_missing(self, tmp_path):
        path = tmp_path / "does_not_exist.csv"
        with pytest.raises(RuntimeError, match="missing"):
            load_or_compute_premiums(factor_engine=None, csv_path=path)

    def test_method_has_separate_cache_path(self, tmp_path):
        base = tmp_path / "factor_premiums.csv"
        assert csv_path_for_method("mean", base) == base
        assert csv_path_for_method("shrinkage", base).name == "factor_premiums_shrinkage.csv"


# ──────────────────────────────────────────────────────────────────────────
# get_factor_drift
# ──────────────────────────────────────────────────────────────────────────

class TestGetFactorDrift:

    def _premiums(self):
        return pd.DataFrame(
            {"MKT": [-0.05, 0.01, 0.20], "FX": [0.02, 0.0, -0.05]},
            index=pd.Index(list(REGIMES), name="regime"),
        )

    def test_known_regime_returns_per_factor_dict(self):
        d = get_factor_drift("bull", ["MKT", "FX"], premiums=self._premiums())
        assert d == {"MKT": 0.20, "FX": -0.05}

    def test_unknown_factor_falls_back_to_regime_erp(self):
        prem = self._premiums()[["MKT"]]
        d = get_factor_drift("bull", ["MKT", "GOLD"], premiums=prem)
        assert d["MKT"] == 0.20
        assert d["GOLD"] == REGIME_ERP["bull"]

    def test_unknown_regime_raises(self):
        with pytest.raises(ValueError, match="Unknown regime"):
            get_factor_drift("euphoria", ["MKT"], premiums=pd.DataFrame())

    def test_missing_cache_falls_back_to_regime_erp(self, tmp_path, monkeypatch):
        from src import factor_premiums as fp
        monkeypatch.setattr(fp, "DEFAULT_CSV_PATH", tmp_path / "missing.csv")
        d = get_factor_drift("bear", ["MKT", "FX"])
        assert d == {"MKT": REGIME_ERP["bear"], "FX": REGIME_ERP["bear"]}


# ──────────────────────────────────────────────────────────────────────────
# Integration with scenario_archetypes
# ──────────────────────────────────────────────────────────────────────────

class TestIntegrationWithScenarioArchetypes:

    def test_initial_drift_dict_returns_per_factor_values(self):
        from src.factor_engine import FACTORS
        from src.scenario_archetypes import initial_drift_dict
        prem = pd.DataFrame(
            {c: [-0.05, 0.0, 0.1] for c in FACTORS},
            index=pd.Index(list(REGIMES), name="regime"),
        )
        d = initial_drift_dict("Bull", FACTORS, premiums=prem)
        assert set(d) == set(FACTORS)
        assert all(isinstance(v, float) for v in d.values())

    def test_regime_thresholds_partition_real_line(self):
        sorted_ranges = sorted(REGIME_THRESHOLDS.values())
        assert sorted_ranges[0][0] == -np.inf
        assert sorted_ranges[-1][1] == +np.inf
        for (a, b), (c, d) in zip(sorted_ranges, sorted_ranges[1:]):
            assert b == c, f"gap or overlap between {b} and {c}"
