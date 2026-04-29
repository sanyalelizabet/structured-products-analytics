"""Tests for ``BetaEngine`` — now a thin shim over ``FactorLoadingsEngine``.

These tests verify (1) that the public API ``build_beta_map`` still returns
a flat ``{isin: float}``, (2) that the numerical result matches what
FactorLoadingsEngine produces with ``factors=["MKT"]``, and (3) that
fallback behaviour on missing data gives β = 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.market_data_engine import MarketDataEngine
from src.factor_engine import FACTORS, FactorEngine
from src.factor_loadings_engine import FactorLoadingsEngine
from src.beta_engine import BetaEngine, BENCHMARK_KEY, BENCHMARK_TICKER


def _seed_with_known_betas(tmp_path, isin_betas: dict[str, float],
                           n_days: int = 800, seed: int = 13):
    """Write a prices.csv where each ISIN's daily return is exactly
    β × MKT-return + small noise."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    mkt_rets = rng.normal(0.0, 0.15 / np.sqrt(252), n_days)
    mkt_prices = 100.0 * np.exp(np.cumsum(mkt_rets))

    rows = []
    # MKT factor row under the benchmark key
    for d, p in zip(dates, mkt_prices):
        rows.append({"date": d, "isin": BENCHMARK_KEY,
                     "ticker": BENCHMARK_TICKER, "price": p})

    # Asset rows
    for isin, beta in isin_betas.items():
        idio = rng.normal(0.0, 0.05 / np.sqrt(252), n_days)
        rets = beta * mkt_rets + idio
        prices = 50.0 * np.exp(np.cumsum(rets))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "isin": isin,
                         "ticker": f"{isin[:4]}.X", "price": p})

    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "prices.csv", index=False)


@pytest.fixture
def beta_engine(tmp_path):
    mock_client = MagicMock()
    mde = MarketDataEngine(client=mock_client, db_path=str(tmp_path / "prices.csv"))
    mde.fetch_daily_prices = MagicMock(return_value=None)
    return BetaEngine(mde), mde, tmp_path


# ──────────────────────────────────────────────────────────────────────────

class TestPublicAPI:
    def test_returns_flat_isin_to_float_map(self, beta_engine):
        eng, mde, tmp_path = beta_engine
        _seed_with_known_betas(tmp_path, {"AAA": 1.2, "BBB": 0.5})

        result = eng.build_beta_map({"AAA": "T1", "BBB": "T2"},
                                    years=3, force_refresh=False)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"AAA", "BBB"}
        for v in result.values():
            assert isinstance(v, float)


class TestNumericalRecovery:
    def test_recovers_known_betas(self, beta_engine):
        eng, mde, tmp_path = beta_engine
        _seed_with_known_betas(tmp_path, {"HIGH": 1.5, "LOW": 0.3})

        result = eng.build_beta_map({"HIGH": "T1", "LOW": "T2"},
                                    years=3, force_refresh=False)

        assert abs(result["HIGH"] - 1.5) < 0.05
        assert abs(result["LOW"]  - 0.3) < 0.05

    def test_matches_factor_loadings_engine(self, beta_engine):
        """BetaEngine must return numerically identical β to a
        ``FactorLoadingsEngine`` call with ``factors=["MKT"]``."""
        eng, mde, tmp_path = beta_engine
        _seed_with_known_betas(tmp_path, {"X": 0.9})

        beta_via_shim = eng.build_beta_map(
            {"X": "T"}, years=3, force_refresh=False,
        )["X"]

        fle = FactorLoadingsEngine(mde)
        beta_via_loadings = fle.build_loadings(
            {"X": "T"}, factors=["MKT"], years=3, force_refresh=False,
        )["X"]["betas"]["MKT"]

        assert beta_via_shim == pytest.approx(beta_via_loadings)


class TestFallback:
    def test_isin_with_no_data_defaults_to_one(self, beta_engine):
        eng, mde, tmp_path = beta_engine
        # Seed only the benchmark, no asset rows
        _seed_with_known_betas(tmp_path, {})

        result = eng.build_beta_map({"GHOST": "T"}, years=3, force_refresh=False)
        assert result["GHOST"] == 1.0
