"""Tests for ``FactorEngine`` — the data-plumbing layer that fetches and
exposes factor returns via ``MarketDataEngine``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.market_data_engine import MarketDataEngine
from src.factor_engine import FACTORS, FactorEngine


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _seed_factor_prices(tmp_path, factor_codes, n_days=800, seed=7):
    """Write a prices.csv populated with synthetic factor rows under the
    storage keys used by FACTORS.  Returns the path."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    rows = []
    for code in factor_codes:
        ticker, key, _label = FACTORS[code]
        # GBM-ish path with code-specific volatility so cross-section differs
        sigma = {"MKT": 0.15, "TECH": 0.22, "HC": 0.12,
                 "FIN": 0.20, "ENERGY": 0.28, "FX": 0.08}.get(code, 0.15)
        rets = rng.normal(0.0, sigma / np.sqrt(252), n_days)
        prices = 100.0 * np.exp(np.cumsum(rets))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "isin": key, "ticker": ticker, "price": p})

    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "prices.csv", index=False)
    return tmp_path / "prices.csv"


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mde(mock_client, tmp_path):
    return MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))


@pytest.fixture
def fe(mde):
    return FactorEngine(mde)


# ──────────────────────────────────────────────────────────────────────────
# fetch_factor_prices
# ──────────────────────────────────────────────────────────────────────────

class TestFetchFactorPrices:
    def test_delegates_to_market_data_engine(self, fe, mde):
        """FactorEngine should call ``mde.fetch_daily_prices`` with a
        storage-key→ticker map matching FACTORS."""
        mde.fetch_daily_prices = MagicMock(return_value=None)

        fe.fetch_factor_prices(years=3)

        mde.fetch_daily_prices.assert_called_once()
        args, kwargs = mde.fetch_daily_prices.call_args
        sent_map = args[0] if args else kwargs.get("isin_ticker_map") or kwargs.get("key_ticker_map")
        # All six factor storage keys are passed
        for code in FACTORS:
            ticker, key, _ = FACTORS[code]
            assert sent_map[key] == ticker

    def test_returns_factor_slice_only(self, fe, mde, tmp_path):
        # Seed only MKT and TECH rows plus an irrelevant ISIN row
        _seed_factor_prices(tmp_path, ["MKT", "TECH"], n_days=300)
        # Add a non-factor ISIN row that should be filtered out
        db = mde.load_db()
        extra = pd.DataFrame([{
            "date": pd.Timestamp.today().normalize(),
            "isin": "CH0012005267", "ticker": "NOVN.SW", "price": 90.0,
        }])
        mde.save_db(pd.concat([db, extra], ignore_index=True))

        # Patch fetch so test stays offline; the seeded CSV is enough
        mde.fetch_daily_prices = MagicMock(return_value=None)
        out = fe.fetch_factor_prices(factors=["MKT", "TECH"], years=3)

        keys = {FACTORS["MKT"][1], FACTORS["TECH"][1]}
        assert set(out["isin"].unique()) == keys


# ──────────────────────────────────────────────────────────────────────────
# build_returns
# ──────────────────────────────────────────────────────────────────────────

class TestBuildReturns:
    def test_returns_wide_dataframe_keyed_by_factor_code(self, fe, mde, tmp_path):
        _seed_factor_prices(tmp_path, list(FACTORS.keys()), n_days=600)
        mde.fetch_daily_prices = MagicMock(return_value=None)

        r = fe.build_returns(years=2)

        assert set(r.columns) == set(FACTORS.keys())
        # log returns ⇒ no NaN, finite values
        assert r.notna().all().all()
        assert np.isfinite(r.values).all()

    def test_subset_of_factors(self, fe, mde, tmp_path):
        _seed_factor_prices(tmp_path, ["MKT", "TECH", "FX"], n_days=400)
        mde.fetch_daily_prices = MagicMock(return_value=None)

        r = fe.build_returns(factors=["MKT", "FX"], years=1)

        assert set(r.columns) == {"MKT", "FX"}

    def test_empty_db_raises(self, fe):
        with pytest.raises(RuntimeError, match="Factor prices not in DB"):
            fe.build_returns()


# ──────────────────────────────────────────────────────────────────────────
# cov / corr / vol
# ──────────────────────────────────────────────────────────────────────────

class TestCovCorrVol:
    @pytest.fixture(autouse=True)
    def _seed(self, tmp_path, mde):
        _seed_factor_prices(tmp_path, list(FACTORS.keys()), n_days=800)
        mde.fetch_daily_prices = MagicMock(return_value=None)

    def test_factor_cov_is_annualised(self, fe):
        r = fe.build_returns(years=3)
        cov = fe.factor_cov(years=3)
        # cov should equal daily cov × 252
        np.testing.assert_allclose(cov.values, (r.cov() * 252).values, rtol=1e-9)

    def test_factor_corr_bounded(self, fe):
        c = fe.factor_corr(years=3)
        assert ((c.values >= -1 - 1e-9) & (c.values <= 1 + 1e-9)).all()
        np.testing.assert_allclose(np.diag(c.values), 1.0, atol=1e-9)

    def test_factor_vol_matches_sqrt_diag_cov(self, fe):
        cov = fe.factor_cov(years=3)
        vol = fe.factor_vol(years=3)
        np.testing.assert_allclose(
            vol.values, np.sqrt(np.diag(cov.values)), rtol=1e-9
        )

    def test_factor_vol_in_plausible_range(self, fe):
        """Synthetic factors were drawn at known annualised σ.  Recovered
        vols should be in the right order of magnitude."""
        vol = fe.factor_vol(years=3)
        # With 800-day samples, sample vol should be within ±25 % of target
        targets = {"MKT": 0.15, "TECH": 0.22, "HC": 0.12,
                   "FIN": 0.20, "ENERGY": 0.28, "FX": 0.08}
        for code, target in targets.items():
            assert 0.75 * target <= vol[code] <= 1.25 * target, (
                f"{code}: realised vol {vol[code]:.3f} outside ±25 % of {target}"
            )
