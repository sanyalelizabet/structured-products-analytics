"""
Tests for PortfolioAnalytics — FX API is mocked.
"""
import math
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from datetime import datetime
from src.portfolio_analytics import PortfolioAnalytics
from tests.conftest import make_portfolio, make_brc_row, make_mbrc_row


# ─────────────────────────────────────────
# FX mock: 1 CHF = 1 EUR (for simplicity)
# ─────────────────────────────────────────

FAKE_FX_RESPONSE = {
    "rates": {
        "EUR": 1.0,
        "USD": 1.1,
    }
}


def make_pa(portfolio=None, reference_currency="CHF"):
    if portfolio is None:
        portfolio = make_portfolio()
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = FAKE_FX_RESPONSE
        return PortfolioAnalytics(portfolio, reference_currency=reference_currency)


# ─────────────────────────────────────────
# FX conversion
# ─────────────────────────────────────────

class TestFxConversion:
    def test_same_currency_no_conversion(self):
        pa = make_pa()
        assert pa.convert_to_reference(100.0, "CHF") == 100.0

    def test_known_rate_conversion(self):
        pa = make_pa()
        # EUR rate is 1.0 in fake response → CHF/EUR = 1/1.0 = 1.0
        result = pa.convert_to_reference(100.0, "EUR")
        assert abs(result - 100.0) < 1e-9

    def test_unknown_rate_raises(self):
        pa = make_pa()
        with pytest.raises(ValueError, match="Missing FX rate"):
            pa.convert_to_reference(100.0, "GBP")


# ─────────────────────────────────────────
# build_product_analytics
# ─────────────────────────────────────────

class TestBuildProductAnalytics:
    def test_returns_dataframe(self):
        pa = make_pa()
        df = pa.build_product_analytics()
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_product(self):
        portfolio = make_portfolio()
        pa = make_pa(portfolio)
        df = pa.build_product_analytics()
        assert len(df) == len(portfolio)

    def test_required_columns_present(self):
        pa = make_pa()
        df = pa.build_product_analytics()
        for col in ["product_id", "pnl", "total_payoff", "total_cost",
                    "distance_to_barrier", "barrier_breached"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_sets_product_df_attribute(self):
        pa = make_pa()
        pa.build_product_analytics()
        assert pa.product_df is not None

    def test_require_product_df_raises_before_build(self):
        pa = make_pa()
        with pytest.raises(ValueError, match="build_product_analytics"):
            pa._require_product_df()


# ─────────────────────────────────────────
# portfolio_summary
# ─────────────────────────────────────────

class TestPortfolioSummary:
    def setup_method(self):
        self.pa = make_pa()
        self.pa.build_product_analytics()

    def test_returns_single_row_dataframe(self):
        summary = self.pa.portfolio_summary()
        assert len(summary) == 1

    def test_n_products_correct(self):
        summary = self.pa.portfolio_summary()
        assert summary.iloc[0]["n_products"] == 2

    def test_n_brc_and_n_mbrc(self):
        summary = self.pa.portfolio_summary()
        row = summary.iloc[0]
        assert row["n_brc"] == 1
        assert row["n_mbrc"] == 1

    def test_total_cost_positive(self):
        summary = self.pa.portfolio_summary()
        assert summary.iloc[0]["total_cost"] > 0

    def test_reference_currency_in_output(self):
        summary = self.pa.portfolio_summary()
        assert summary.iloc[0]["reference_currency"] == "CHF"


# ─────────────────────────────────────────
# underlying_lookthrough
# ─────────────────────────────────────────

class TestUnderlyingLookthrough:
    def setup_method(self):
        self.pa = make_pa()

    def test_returns_dataframe(self):
        df = self.pa.underlying_lookthrough()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = self.pa.underlying_lookthrough()
        for col in ["underlying", "isin", "n_products", "allocated_cost_ref",
                    "weight", "min_distance_to_barrier"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_weights_sum_to_one(self):
        df = self.pa.underlying_lookthrough()
        assert abs(df["weight"].sum() - 1.0) < 1e-6

    def test_brc_underlying_present(self):
        df = self.pa.underlying_lookthrough()
        assert "NESN" in df["underlying"].values

    def test_mbrc_underlyings_present(self):
        df = self.pa.underlying_lookthrough()
        assert "NOVN" in df["underlying"].values


# ─────────────────────────────────────────
# barrier_watchlist
# ─────────────────────────────────────────

class TestBarrierWatchlist:
    def setup_method(self):
        self.pa = make_pa()
        self.pa.build_product_analytics()

    def test_returns_dataframe(self):
        df = self.pa.barrier_watchlist()
        assert isinstance(df, pd.DataFrame)

    def test_near_barrier_flag_column_exists(self):
        df = self.pa.barrier_watchlist()
        assert "near_barrier_flag" in df.columns

    def test_custom_threshold(self):
        df_strict = self.pa.barrier_watchlist(threshold=0.01)
        df_loose = self.pa.barrier_watchlist(threshold=0.99)
        assert df_loose["near_barrier_flag"].sum() >= df_strict["near_barrier_flag"].sum()


# ─────────────────────────────────────────
# maturity_profile
# ─────────────────────────────────────────

class TestMaturityProfile:
    def setup_method(self):
        self.pa = make_pa()
        self.pa.build_product_analytics()

    def test_returns_all_buckets(self):
        df = self.pa.maturity_profile(today=datetime(2024, 6, 1))
        expected_buckets = ["<=1M", "1-3M", "3-6M", "6-12M", "1-2Y", ">2Y"]
        assert list(df["maturity_bucket"].astype(str)) == expected_buckets

    def test_total_products_across_buckets(self):
        df = self.pa.maturity_profile(today=datetime(2024, 6, 1))
        assert df["n_products"].sum() == 2


# ─────────────────────────────────────────
# scenario_attribution
# ─────────────────────────────────────────

class TestScenarioAttribution:
    def setup_method(self):
        self.pa = make_pa()
        self.pa.build_product_analytics()

    def test_returns_dataframe(self):
        df = self.pa.scenario_attribution()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = self.pa.scenario_attribution()
        for col in ["product_id", "pnl", "pnl_contribution_pct", "total_cost"]:
            assert col in df.columns

    @pytest.mark.xfail(reason="BUG: cost_weight is computed in scenario_attribution() but not included in the returned DataFrame's column selection", strict=True)
    def test_cost_weight_column_returned(self):
        """cost_weight is calculated but silently dropped — column missing from output."""
        df = self.pa.scenario_attribution()
        assert "cost_weight" in df.columns


# ─────────────────────────────────────────
# total_portfolio_metrics
# ─────────────────────────────────────────

class TestTotalPortfolioMetrics:
    def setup_method(self):
        self.pa = make_pa()
        self.pa.build_product_analytics()

    def test_returns_dict_with_required_keys(self):
        metrics = self.pa.total_portfolio_metrics()
        for key in ["total_notional", "total_cost", "total_payoff",
                    "total_pnl", "portfolio_return_pct"]:
            assert key in metrics

    def test_pnl_equals_payoff_minus_cost(self):
        m = self.pa.total_portfolio_metrics()
        assert abs(m["total_pnl"] - (m["total_payoff"] - m["total_cost"])) < 0.01

    def test_total_portfolio_table_wraps_dict(self):
        df = self.pa.total_portfolio_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
