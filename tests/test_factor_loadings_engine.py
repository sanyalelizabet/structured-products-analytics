"""Tests for ``FactorLoadingsEngine`` — the multivariate-OLS layer that
fits asset returns onto factor returns.

We seed ``prices.csv`` with synthetic factor and asset prices whose
return-generating process is known, and check that OLS recovers the
designed-in betas, alphas, R², and idiosyncratic vol within tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.market_data_engine import MarketDataEngine
from src.factor_engine import FACTORS, FactorEngine
from src.factor_loadings_engine import FactorLoadingsEngine


# ──────────────────────────────────────────────────────────────────────────
# Helpers — synthetic data with KNOWN beta structure
# ──────────────────────────────────────────────────────────────────────────

def _make_synthetic_world(
    tmp_path,
    asset_specs: dict[str, dict],
    n_days: int = 800,
    seed: int = 11,
):
    """Write a prices.csv where:

    - Factor return processes are independent N(0, σ_k²/252).
    - Each asset's daily return is α + Σ β_k F_k + N(0, σ_eps²/252).

    Parameters
    ----------
    asset_specs : dict
        ``{isin: {"alpha": float, "betas": {factor: β}, "idio_vol": float}}``
        Annualised α and idio σ.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    factor_codes = list(FACTORS.keys())
    factor_sigmas = {"MKT": 0.15, "TECH": 0.22, "HC": 0.12,
                     "FIN": 0.20, "ENERGY": 0.28, "FX": 0.08}

    # Generate independent factor return series
    factor_returns = {
        code: rng.normal(0.0, factor_sigmas[code] / np.sqrt(252), n_days)
        for code in factor_codes
    }

    rows = []

    # Factor rows — under their storage keys
    for code in factor_codes:
        ticker, key, _ = FACTORS[code]
        prices = 100.0 * np.exp(np.cumsum(factor_returns[code]))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "isin": key, "ticker": ticker, "price": p})

    # Asset rows — built from spec
    for isin, spec in asset_specs.items():
        alpha = spec.get("alpha", 0.0)
        betas = spec.get("betas", {})
        idio_sigma = spec.get("idio_vol", 0.10)

        # Daily return = α/252 + Σ β_k F_k + idio noise
        daily_alpha = alpha / 252.0
        idio_daily = rng.normal(0.0, idio_sigma / np.sqrt(252), n_days)
        asset_returns = np.full(n_days, daily_alpha)
        for code, beta in betas.items():
            asset_returns += beta * factor_returns[code]
        asset_returns += idio_daily

        prices = 50.0 * np.exp(np.cumsum(asset_returns))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "isin": isin, "ticker": f"{isin[:4]}.X", "price": p})

    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "prices.csv", index=False)
    return tmp_path / "prices.csv"


@pytest.fixture
def fle(tmp_path):
    """Engine wired against an isolated tmp prices.csv with no network."""
    mock_client = MagicMock()
    mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
    fe = FactorEngine(mde)
    fle = FactorLoadingsEngine(mde, fe)
    # Patch fetches to no-op — synthetic data is pre-seeded in CSV
    mde.fetch_daily_prices = MagicMock(return_value=None)
    return fle, mde, tmp_path


# ──────────────────────────────────────────────────────────────────────────
# Single-factor recovery
# ──────────────────────────────────────────────────────────────────────────

class TestSingleFactor:
    def test_recovers_market_beta(self, fle):
        engine, mde, tmp_path = fle
        _make_synthetic_world(tmp_path, {
            "TEST_HIGHBETA": {"alpha": 0.0, "betas": {"MKT": 1.5}, "idio_vol": 0.05},
            "TEST_LOWBETA":  {"alpha": 0.0, "betas": {"MKT": 0.4}, "idio_vol": 0.05},
        })

        loadings = engine.build_loadings(
            {"TEST_HIGHBETA": "T1", "TEST_LOWBETA": "T2"},
            factors=["MKT"], years=3, force_refresh=False,
        )

        assert abs(loadings["TEST_HIGHBETA"]["betas"]["MKT"] - 1.5) < 0.05
        assert abs(loadings["TEST_LOWBETA"]["betas"]["MKT"] - 0.4) < 0.05

    def test_high_r_squared_when_noise_small(self, fle):
        engine, mde, tmp_path = fle
        _make_synthetic_world(tmp_path, {
            "CLEAN": {"alpha": 0.0, "betas": {"MKT": 1.0}, "idio_vol": 0.01},
        })

        loadings = engine.build_loadings(
            {"CLEAN": "T1"}, factors=["MKT"], years=3, force_refresh=False,
        )
        assert loadings["CLEAN"]["r_squared"] > 0.95

    def test_low_r_squared_when_noise_large(self, fle):
        engine, mde, tmp_path = fle
        _make_synthetic_world(tmp_path, {
            "NOISY": {"alpha": 0.0, "betas": {"MKT": 0.5}, "idio_vol": 0.50},
        })

        loadings = engine.build_loadings(
            {"NOISY": "T1"}, factors=["MKT"], years=3, force_refresh=False,
        )
        assert loadings["NOISY"]["r_squared"] < 0.20


# ──────────────────────────────────────────────────────────────────────────
# Multi-factor recovery
# ──────────────────────────────────────────────────────────────────────────

class TestMultiFactor:
    def test_recovers_all_betas(self, fle):
        engine, mde, tmp_path = fle
        true_betas = {"MKT": 0.9, "TECH": 1.3, "HC": 0.2,
                      "FIN": -0.1, "ENERGY": 0.05, "FX": 0.15}
        _make_synthetic_world(tmp_path, {
            "MULTI": {"alpha": 0.0, "betas": true_betas, "idio_vol": 0.05},
        })

        loadings = engine.build_loadings(
            {"MULTI": "T1"},
            factors=list(FACTORS.keys()),
            years=3,
            force_refresh=False,
        )

        recovered = loadings["MULTI"]["betas"]
        for f, true_b in true_betas.items():
            assert abs(recovered[f] - true_b) < 0.05, (
                f"{f}: recovered {recovered[f]:.3f} vs true {true_b:.3f}"
            )

    def test_idio_vol_recovered(self, fle):
        engine, mde, tmp_path = fle
        _make_synthetic_world(tmp_path, {
            "X": {"alpha": 0.0,
                  "betas": {"MKT": 1.0, "TECH": 0.5},
                  "idio_vol": 0.20},
        })
        loadings = engine.build_loadings(
            {"X": "T"},
            factors=["MKT", "TECH"],
            years=3,
            force_refresh=False,
        )
        # tolerance ±20% (sample std at ~750 obs is noisy)
        assert 0.16 <= loadings["X"]["idio_vol"] <= 0.24


# ──────────────────────────────────────────────────────────────────────────
# Fallbacks & validation
# ──────────────────────────────────────────────────────────────────────────

class TestFallbacks:
    def test_unknown_factor_raises(self, fle):
        engine, mde, tmp_path = fle
        _make_synthetic_world(tmp_path, {})
        with pytest.raises(ValueError, match="Unknown factor code"):
            engine.build_loadings({"X": "T"}, factors=["NOTAFACTOR"])

    def test_isin_with_no_data_defaults(self, fle):
        engine, mde, tmp_path = fle
        # Seed factor data but NOT the asset
        _make_synthetic_world(tmp_path, {})
        loadings = engine.build_loadings(
            {"GHOST_ISIN": "T"}, factors=["MKT"], years=3, force_refresh=False,
        )
        assert loadings["GHOST_ISIN"]["betas"]["MKT"] == 1.0
        assert loadings["GHOST_ISIN"]["r_squared"] == 0.0
        assert loadings["GHOST_ISIN"]["n_obs"] == 0

    def test_insufficient_overlap_defaults(self, fle):
        engine, mde, tmp_path = fle
        # Only 50 days of asset data — below default min_obs=252
        _make_synthetic_world(tmp_path, {
            "SHORT": {"alpha": 0.0, "betas": {"MKT": 1.0}, "idio_vol": 0.05},
        }, n_days=800)
        # Trim asset rows to last 50 days
        df = pd.read_csv(tmp_path / "prices.csv", parse_dates=["date"])
        keep = df["isin"] != "SHORT"
        short_rows = df[df["isin"] == "SHORT"].sort_values("date").tail(50)
        df = pd.concat([df[keep], short_rows], ignore_index=True)
        df.to_csv(tmp_path / "prices.csv", index=False)

        loadings = engine.build_loadings(
            {"SHORT": "T"}, factors=["MKT"], years=3, force_refresh=False,
        )
        assert loadings["SHORT"]["betas"]["MKT"] == 1.0   # default
        assert loadings["SHORT"]["n_obs"] == 0            # not fitted


# ──────────────────────────────────────────────────────────────────────────
# loadings_to_dataframe
# ──────────────────────────────────────────────────────────────────────────

class TestLoadingsToDataFrame:
    def test_columns_and_shape(self, fle):
        engine, mde, tmp_path = fle
        _make_synthetic_world(tmp_path, {
            "A": {"alpha": 0.0, "betas": {"MKT": 1.0, "TECH": 0.5}, "idio_vol": 0.10},
            "B": {"alpha": 0.0, "betas": {"MKT": 0.7, "TECH": -0.2}, "idio_vol": 0.10},
        })
        loadings = engine.build_loadings(
            {"A": "T1", "B": "T2"},
            factors=["MKT", "TECH"], years=3, force_refresh=False,
        )
        df = engine.loadings_to_dataframe(loadings)
        assert len(df) == 2
        for col in ["isin", "alpha", "idio_vol", "r_squared", "n_obs",
                    "β_MKT", "β_TECH"]:
            assert col in df.columns
