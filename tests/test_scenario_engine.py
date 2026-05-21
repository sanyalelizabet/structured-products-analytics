"""Tests for the multi-path ``ScenarioEngine`` (single-factor MC + CRN).

Covers:

* Getters (β, σ defaults)
* Correlation submatrix extraction
* ``build_shock_paths`` — vectorised path tensor shape, finite values,
  shock direction, drift effect
* ``run_path_scenario`` — output schema, currency aggregation,
  determinism via cached ``NoiseSampler``, multi-path statistics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scenario_engine import ScenarioEngine
from src.noise_sampler import NoiseSampler
from tests.conftest import (
    make_portfolio, make_brc_row, make_mbrc_row,
    BETA_MAP, VOL_MAP, SCENARIOS,
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def make_engine(
    portfolio=None, beta_map=None, vol_map=None,
    n_paths=20, mean_reversion_kappa=0.5,
):
    return ScenarioEngine(
        portfolio=portfolio if portfolio is not None else make_portfolio(),
        beta_map=beta_map or BETA_MAP,
        vol_map=vol_map or VOL_MAP,
        n_paths=n_paths,
        mean_reversion_kappa=mean_reversion_kappa,
    )


def flat_scenario(**kw):
    base = {
        "market_shock": 0,
        "n_shocks": 1,
        "shock_in_days": 0,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.0,
        "post_shock_drift_pa": 0.0,
    }
    base.update(kw)
    return base


def _build_sampler_for(engine):
    """Construct a fresh sampler that matches the engine's portfolio."""
    today              = pd.Timestamp.today().normalize()
    portfolio_maturity = pd.to_datetime(engine.portfolio["maturity_date"]).max()
    n_days             = len(pd.bdate_range(start=today, end=portfolio_maturity))
    isins = sorted({isin for _, r in engine.portfolio.iterrows() for isin in r["underlying_isins"]})
    return NoiseSampler(
        n_paths=engine.n_paths, n_days=n_days,
        factor_codes=[], isins=isins,
    )


# ─────────────────────────────────────────
# Getters
# ─────────────────────────────────────────

class TestGetters:
    def test_known_beta(self):
        e = make_engine()
        assert e.get_beta("CH0012221716") == 1.0

    def test_unknown_beta_defaults_to_one(self):
        e = make_engine()
        assert e.get_beta("UNKNOWN") == 1.0

    def test_known_vol(self):
        e = make_engine()
        assert e.get_vol("CH0012221716") == 0.20

    def test_unknown_vol_defaults_to_015(self):
        e = make_engine()
        assert e.get_vol("UNKNOWN") == 0.15


# ─────────────────────────────────────────
# Correlation submatrix
# ─────────────────────────────────────────

class TestGetCorrSubset:
    @pytest.fixture
    def engine(self):
        return make_engine()

    @pytest.fixture
    def corr_df(self):
        return pd.DataFrame(
            {"CH0012221716": [1.0, 0.5, 0.2],
             "CH0012221717": [0.5, 1.0, 0.4],
             "CH0099999999": [0.2, 0.4, 1.0]},
            index=["CH0012221716", "CH0012221717", "CH0099999999"],
        )

    def test_none_returns_identity(self, engine):
        row = make_mbrc_row()
        m = engine.get_corr_subset(row, None)
        np.testing.assert_array_equal(m, np.eye(2))

    def test_extracts_correct_subset(self, engine, corr_df):
        row = make_mbrc_row()
        m = engine.get_corr_subset(row, corr_df)
        assert m.shape == (2, 2)
        assert m[0, 1] == pytest.approx(0.5)

    def test_preserves_isin_order(self, engine, corr_df):
        row = make_mbrc_row()
        # underlying_isins = ["CH0012221716", "CH0012221717"]
        m = engine.get_corr_subset(row, corr_df)
        assert m[0, 0] == 1.0 and m[1, 1] == 1.0
        assert m[0, 1] == m[1, 0] == pytest.approx(0.5)

    def test_missing_isin_falls_back_to_identity(self, engine):
        """A manually-entered product whose underlying ISIN isn't in the
        correlation matrix should not crash the analytics pipeline.  The
        engine returns identity (same as ``corr_df is None``); the
        Streamlit pre-flight surfaces a market-data coverage warning so
        the user knows their cross-asset estimates are simplified.
        """
        row = make_mbrc_row()  # MBRC has 2 underlyings
        bad = pd.DataFrame(
            {"CH0099999999": [1.0]},
            index=["CH0099999999"],
        )
        m = engine.get_corr_subset(row, bad)
        np.testing.assert_array_equal(m, np.eye(2))


# ─────────────────────────────────────────
# build_shock_paths — vectorised tensor
# ─────────────────────────────────────────

class TestBuildShockPaths:
    @pytest.fixture
    def engine(self):
        return make_engine(n_paths=20)

    @pytest.fixture
    def sampler(self, engine):
        return _build_sampler_for(engine)

    def test_returns_tensor_and_grid(self, engine, sampler):
        row = make_mbrc_row()
        price_paths, date_range, path_summary = engine.build_shock_paths(
            row, flat_scenario(), sampler,
        )
        N, n_days, n_assets = price_paths.shape
        assert N == engine.n_paths
        assert n_assets == 2                      # MBRC has two underlyings
        assert len(date_range) == n_days
        assert isinstance(path_summary, dict)

    def test_brc_single_asset_shape(self, engine, sampler):
        row = make_brc_row()
        price_paths, _, _ = engine.build_shock_paths(row, flat_scenario(), sampler)
        assert price_paths.shape[2] == 1          # BRC has one underlying

    def test_path_prices_positive_and_finite(self, engine, sampler):
        price_paths, _, _ = engine.build_shock_paths(
            make_mbrc_row(), flat_scenario(market_shock=-30), sampler,
        )
        assert np.isfinite(price_paths).all()
        assert (price_paths > 0).all()

    def test_negative_shock_lowers_terminal(self, engine, sampler):
        row = make_mbrc_row()
        no_shock = engine.build_shock_paths(row, flat_scenario(),                     sampler)[0]
        sampler.regenerate(seed=42)
        big_shock = engine.build_shock_paths(row, flat_scenario(market_shock=-40), sampler)[0]
        # Re-seeded — but on the *same* sampler the noise is fixed; we restore it
        # before comparing so the only difference is the shock magnitude.
        # Median terminal of shocked must be below median terminal of unshocked.
        assert np.median(big_shock[:, -1, :]) < np.median(no_shock[:, -1, :])

    def test_recovery_drift_raises_terminal(self, engine, sampler):
        row = make_mbrc_row()
        a = engine.build_shock_paths(
            row,
            flat_scenario(market_shock=-20, post_shock_drift_pa=0.0,
                          pre_shock_drift_pa=0.0),
            sampler,
        )[0]
        sampler.regenerate(seed=42)
        b = engine.build_shock_paths(
            row,
            flat_scenario(market_shock=-20, post_shock_drift_pa=0.30,
                          pre_shock_drift_pa=0.0),
            sampler,
        )[0]
        # Higher post-shock drift → higher median terminal price.
        assert np.median(b[:, -1, :]) > np.median(a[:, -1, :])

    def test_path_summary_required_keys(self, engine, sampler):
        _, _, summary = engine.build_shock_paths(
            make_mbrc_row(), flat_scenario(), sampler,
        )
        for k in ("maturity_date", "T_remaining_years", "T_first_shock_years",
                  "T_post_shock_years", "effective_n_shocks", "market_shock_pct",
                  "pre_shock_drift_pa", "post_shock_drift_pa", "correlation_used"):
            assert k in summary


# ─────────────────────────────────────────
# run_path_scenario — portfolio run
# ─────────────────────────────────────────

class TestRunPathScenario:
    @pytest.fixture
    def engine(self):
        return make_engine(n_paths=20)

    def test_required_keys(self, engine):
        res = engine.run_path_scenario(SCENARIOS["down_10"])
        for k in ("product_df", "pf_scenario_per_ccy", "cash_positions",
                  "delivered_stocks", "asset_paths", "pnl_samples_by_ccy",
                  "n_paths"):
            assert k in res
        assert res["n_paths"] == engine.n_paths

    def test_product_df_one_row_per_product(self, engine):
        res = engine.run_path_scenario(SCENARIOS["down_10"])
        assert len(res["product_df"]) == len(make_portfolio())

    def test_asset_paths_have_summary_columns(self, engine):
        res = engine.run_path_scenario(SCENARIOS["down_10"])
        for isin, df in res["asset_paths"].items():
            for col in ("date", "mean", "median", "p5", "p95"):
                assert col in df.columns
            assert np.isfinite(df["median"].to_numpy()).all()

    def test_pnl_percentiles_ordered(self, engine):
        res = engine.run_path_scenario(SCENARIOS["down_30"])
        for _, row in res["product_df"].iterrows():
            assert row["pnl_p5"] <= row["pnl_median"] <= row["pnl_p95"]

    def test_currency_pnl_aggregates_per_path(self, engine):
        res = engine.run_path_scenario(SCENARIOS["down_10"])
        for ccy, samples in res["pnl_samples_by_ccy"].items():
            assert samples.shape == (engine.n_paths,)

    def test_portfolio_return_columns_present(self, engine):
        res = engine.run_path_scenario(SCENARIOS["down_10"])
        cols = res["pf_scenario_per_ccy"].columns
        for c in ("portfolio_return_mean_pct", "portfolio_return_p5_pct",
                  "portfolio_return_p95_pct"):
            assert c in cols


# ─────────────────────────────────────────
# Determinism via cached NoiseSampler
# ─────────────────────────────────────────

class TestDeterminism:
    def test_same_engine_same_results(self):
        e = make_engine(n_paths=15)
        a = e.run_path_scenario(SCENARIOS["down_10"])
        b = e.run_path_scenario(SCENARIOS["down_10"])
        # Same sampler is reused → identical samples.
        np.testing.assert_array_equal(
            a["product_df"]["pnl_samples"].iloc[0],
            b["product_df"]["pnl_samples"].iloc[0],
        )

    def test_shared_sampler_links_two_engines(self):
        portfolio = make_portfolio()
        sampler = NoiseSampler(
            n_paths=15,
            n_days=len(pd.bdate_range(
                start=pd.Timestamp.today().normalize(),
                end=pd.to_datetime(portfolio["maturity_date"]).max(),
            )),
            factor_codes=[],
            isins=sorted({isin for _, r in portfolio.iterrows() for isin in r["underlying_isins"]}),
        )
        e1 = ScenarioEngine(portfolio=portfolio, beta_map=BETA_MAP, vol_map=VOL_MAP,
                            n_paths=15, noise_sampler=sampler)
        e2 = ScenarioEngine(portfolio=portfolio, beta_map=BETA_MAP, vol_map=VOL_MAP,
                            n_paths=15, noise_sampler=sampler)
        a = e1.run_path_scenario(SCENARIOS["down_10"])
        b = e2.run_path_scenario(SCENARIOS["down_10"])
        np.testing.assert_array_equal(
            a["product_df"]["pnl_samples"].iloc[0],
            b["product_df"]["pnl_samples"].iloc[0],
        )


# ─────────────────────────────────────────
# Multi-path statistics
# ─────────────────────────────────────────

class TestMultiPathStatistics:
    def test_pnl_samples_shape(self):
        e = make_engine(n_paths=25)
        res = e.run_path_scenario(SCENARIOS["down_10"])
        for samples in res["product_df"]["pnl_samples"]:
            assert samples.shape == (e.n_paths,)

    def test_n_paths_one_collapses_summary(self):
        e = make_engine(n_paths=1)
        res = e.run_path_scenario(SCENARIOS["down_10"])
        for _, df in res["asset_paths"].items():
            np.testing.assert_array_equal(df["median"].to_numpy(), df["mean"].to_numpy())
            np.testing.assert_array_equal(df["median"].to_numpy(), df["p5"].to_numpy())
            np.testing.assert_array_equal(df["median"].to_numpy(), df["p95"].to_numpy())

    def test_more_paths_finite(self):
        e = make_engine(n_paths=80)
        res = e.run_path_scenario(SCENARIOS["down_10"])
        for samples in res["product_df"]["pnl_samples"]:
            assert np.isfinite(samples).all()
