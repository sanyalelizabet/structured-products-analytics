"""
Tests for ScenarioEngine (correlated GBM path-based engine).
"""
import numpy as np
import pandas as pd
import pytest
from src.scenario_engine import ScenarioEngine
from tests.conftest import (
    make_portfolio, make_brc_row, make_mbrc_row,
    BETA_MAP, VOL_MAP, SCENARIOS,
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def make_engine(portfolio=None, beta_map=None, vol_map=None, scenarios=None):
    return ScenarioEngine(
        portfolio=portfolio if portfolio is not None else make_portfolio(),
        beta_map=beta_map or BETA_MAP,
        vol_map=vol_map or VOL_MAP,
        scenarios=scenarios,
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


# ─────────────────────────────────────────
# get_beta / get_vol
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
# _get_isin_rng_map  ← same ISIN = same path
# ─────────────────────────────────────────

class TestIsinRngMap:
    def test_each_isin_has_own_rng(self):
        e = make_engine()
        rng_map = e._get_isin_rng_map(["CH0012221716", "CH0012221717"])
        assert set(rng_map.keys()) == {"CH0012221716", "CH0012221717"}

    def test_same_isin_produces_same_sequence(self):
        """Same ISIN → same MD5 seed → identical draws every time."""
        e = make_engine()
        rng_a = e._get_isin_rng_map(["CH0012221716"])["CH0012221716"]
        rng_b = e._get_isin_rng_map(["CH0012221716"])["CH0012221716"]
        draws_a = rng_a.standard_normal(100)
        draws_b = rng_b.standard_normal(100)
        np.testing.assert_array_equal(draws_a, draws_b)

    def test_different_isins_produce_different_sequences(self):
        e = make_engine()
        rng_map = e._get_isin_rng_map(["CH0012221716", "CH0012221717"])
        draws_nesn = rng_map["CH0012221716"].standard_normal(100)
        draws_novn = rng_map["CH0012221717"].standard_normal(100)
        assert not np.array_equal(draws_nesn, draws_novn)

    def test_portfolio_shared_isin_gets_same_path(self):
        """
        Two products sharing an ISIN must receive the same simulated path.
        This is the core portfolio consistency guarantee.
        """
        # BRC and MBRC both contain NESN (CH0012221716)
        e = make_engine()
        brc = make_brc_row()
        mbrc = make_mbrc_row()

        scenario = flat_scenario()
        _, final_brc, _, _, paths_brc   = e.build_shock_paths(brc,  scenario)
        _, final_mbrc, _, _, paths_mbrc = e.build_shock_paths(mbrc, scenario)

        nesn_isin = "CH0012221716"
        path_from_brc  = paths_brc[nesn_isin]["price"].values
        path_from_mbrc = paths_mbrc[nesn_isin]["price"].values

        np.testing.assert_array_almost_equal(
            path_from_brc, path_from_mbrc,
            err_msg="NESN path must be identical across both products"
        )


# ─────────────────────────────────────────
# build_shock_paths
# ─────────────────────────────────────────

class TestBuildShockPaths:
    def test_returns_five_values(self):
        e = make_engine()
        result = e.build_shock_paths(make_brc_row(), flat_scenario())
        assert len(result) == 5  # shocks, final_spots, T_remaining, path_summary, paths

    def test_shocks_length_matches_underlyings(self):
        e = make_engine()
        shocks, _, _, _, _ = e.build_shock_paths(make_brc_row(), flat_scenario())
        assert len(shocks) == 1

    def test_mbrc_shocks_length(self):
        e = make_engine()
        shocks, _, _, _, _ = e.build_shock_paths(make_mbrc_row(), flat_scenario())
        assert len(shocks) == 2

    def test_final_spots_length_matches_underlyings(self):
        e = make_engine()
        _, final_spots, _, _, _ = e.build_shock_paths(make_brc_row(), flat_scenario())
        assert len(final_spots) == 1

    def test_t_remaining_positive(self):
        e = make_engine()
        _, _, T_rem, _, _ = e.build_shock_paths(make_brc_row(), flat_scenario())
        assert T_rem > 0

    def test_paths_dict_keyed_by_isin(self):
        e = make_engine()
        _, _, _, _, paths = e.build_shock_paths(make_brc_row(), flat_scenario())
        assert "CH0012221716" in paths

    def test_path_df_has_date_and_price(self):
        e = make_engine()
        _, _, _, _, paths = e.build_shock_paths(make_brc_row(), flat_scenario())
        df = paths["CH0012221716"]
        assert "date" in df.columns
        assert "price" in df.columns

    def test_path_prices_positive(self):
        """GBM prices must stay positive throughout the path."""
        e = make_engine()
        _, _, _, _, paths = e.build_shock_paths(make_brc_row(), flat_scenario())
        assert (paths["CH0012221716"]["price"] > 0).all()

    def test_negative_shock_lowers_final_spot(self):
        e = make_engine()
        brc = make_brc_row(current_spot=100.0)
        _, final_no_shock, _, _, _ = e.build_shock_paths(brc, flat_scenario(market_shock=0))
        _, final_with_shock, _, _, _ = e.build_shock_paths(brc, flat_scenario(market_shock=-30))
        assert final_with_shock[0] < final_no_shock[0]

    def test_recovery_drift_raises_final_above_permanent(self):
        e = make_engine()
        brc = make_brc_row()
        perm    = flat_scenario(market_shock=-20, post_shock_drift_pa=0.0)
        recover = flat_scenario(market_shock=-20, post_shock_drift_pa=0.10)
        _, spots_perm,    _, _, _ = e.build_shock_paths(brc, perm)
        _, spots_recover, _, _, _ = e.build_shock_paths(brc, recover)
        assert spots_recover[0] > spots_perm[0]

    def test_path_summary_contains_required_keys(self):
        e = make_engine()
        _, _, _, summary, _ = e.build_shock_paths(make_brc_row(), flat_scenario())
        for key in ["maturity_date", "T_remaining_years", "market_shock_pct"]:
            assert key in summary

    def test_identity_corr_matrix_accepted(self):
        e = make_engine()
        corr = np.eye(2)
        shocks, _, _, _, _ = e.build_shock_paths(make_mbrc_row(), flat_scenario(), corr_matrix=corr)
        assert len(shocks) == 2

    def test_correlated_matrix_accepted(self):
        e = make_engine()
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        shocks, _, _, _, _ = e.build_shock_paths(make_mbrc_row(), flat_scenario(), corr_matrix=corr)
        assert len(shocks) == 2


# ─────────────────────────────────────────
# get_corr_subset
# ─────────────────────────────────────────

class TestGetCorrSubset:
    def _corr_df(self):
        isins = ["CH0012221716", "CH0012221717", "CH0012221718"]
        data = np.array([
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.6],
            [0.5, 0.6, 1.0],
        ])
        return pd.DataFrame(data, index=isins, columns=isins)

    def test_none_returns_identity(self):
        e = make_engine()
        row = make_brc_row()
        result = e.get_corr_subset(row, None)
        np.testing.assert_array_equal(result, np.eye(1))

    def test_extracts_correct_subset(self):
        e = make_engine()
        row = make_mbrc_row()
        corr_df = self._corr_df()
        result = e.get_corr_subset(row, corr_df)
        assert result.shape == (2, 2)
        assert abs(result[0, 1] - 0.7) < 1e-9

    def test_preserves_isin_order(self):
        """Subset must follow the order in row["underlying_isins"], not corr_df order."""
        e = make_engine()
        row = make_mbrc_row()
        corr_df = self._corr_df()
        result = e.get_corr_subset(row, corr_df)
        # row isins: [CH0012221716, CH0012221717] — off-diagonal should be 0.7
        assert abs(result[0, 1] - 0.7) < 1e-9
        assert abs(result[1, 0] - 0.7) < 1e-9

    def test_missing_isin_raises(self):
        e = make_engine()
        row = make_brc_row()
        # corr_df missing CH0012221716
        corr_df = pd.DataFrame(
            [[1.0]], index=["OTHER_ISIN"], columns=["OTHER_ISIN"]
        )
        with pytest.raises(KeyError, match="Missing ISIN"):
            e.get_corr_subset(row, corr_df)


# ─────────────────────────────────────────
# run_product_path_scenario
# ─────────────────────────────────────────

class TestRunProductPathScenario:
    def test_returns_required_keys(self):
        e = make_engine()
        result = e.run_product_path_scenario(make_brc_row(), flat_scenario())
        for key in ["product_id", "total_payoff", "pnl", "settlement_type",
                    "barrier_breached", "worst_underlying", "final_spots"]:
            assert key in result

    def test_cash_settlement_above_strike(self):
        """No shock → spot above strike → cash settlement."""
        e = make_engine()
        # With zero shock and current spot already above the scenario floor,
        # outcome depends on GBM path — use a positive shock to force cash
        row = make_brc_row(current_spot=100.0, strike=50.0)
        result = e.run_product_path_scenario(row, flat_scenario(market_shock=0))
        assert result["settlement_type"] == "cash"
        assert result["delivered_shares"] == 0

    def test_physical_settlement_below_strike(self):
        """Large negative shock → final below strike → physical delivery."""
        e = make_engine()
        row = make_brc_row(current_spot=100.0, strike=200.0)  # strike >> spot
        result = e.run_product_path_scenario(row, flat_scenario(market_shock=-50))
        assert result["settlement_type"] == "physical"
        assert result["delivered_shares"] > 0

    def test_product_id_preserved(self):
        e = make_engine()
        result = e.run_product_path_scenario(make_brc_row(), flat_scenario())
        assert result["product_id"] == "BRC001"

    def test_final_spots_count_matches_underlyings(self):
        e = make_engine()
        result = e.run_product_path_scenario(make_mbrc_row(), flat_scenario())
        assert len(result["final_spots"]) == 2


# ─────────────────────────────────────────
# run_path_scenario
# ─────────────────────────────────────────

class TestRunPathScenario:
    def test_returns_required_keys(self):
        e = make_engine()
        result = e.run_path_scenario(flat_scenario())
        for key in ["product_df", "pf_scenario_per_ccy", "cash_positions",
                    "delivered_stocks", "paths"]:
            assert key in result

    def test_product_df_one_row_per_product(self):
        e = make_engine()
        result = e.run_path_scenario(flat_scenario())
        assert len(result["product_df"]) == len(make_portfolio())

    def test_paths_dict_contains_all_unique_isins(self):
        """all_paths must include every unique ISIN across the portfolio."""
        e = make_engine()
        result = e.run_path_scenario(flat_scenario())
        # Portfolio has NESN (BRC) and NESN+NOVN (MBRC)
        assert "CH0012221716" in result["paths"]
        assert "CH0012221717" in result["paths"]

    def test_portfolio_run_is_deterministic(self):
        """
        Calling run_path_scenario twice with the same scenario must produce
        bit-identical paths — the scenario seed is derived from the scenario dict.
        """
        e = make_engine()
        scenario = flat_scenario()
        result_1 = e.run_path_scenario(scenario)
        result_2 = e.run_path_scenario(scenario)

        np.testing.assert_array_equal(
            result_1["paths"]["CH0012221716"]["price"].values,
            result_2["paths"]["CH0012221716"]["price"].values,
        )

    def test_cash_positions_has_currency_and_total_cash(self):
        e = make_engine()
        result = e.run_path_scenario(flat_scenario())
        cash = result["cash_positions"]
        assert "currency" in cash.columns
        assert "total_cash" in cash.columns

    def test_portfolio_return_pct_column_present(self):
        e = make_engine()
        result = e.run_path_scenario(flat_scenario())
        assert "portfolio_return_pct" in result["pf_scenario_per_ccy"].columns
