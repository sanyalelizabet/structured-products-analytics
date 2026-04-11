"""
Behavioural tests:

1.  Path duration  — paths span the full portfolio lifespan (max maturity)
2.  Scenario directional consistency — positive/negative shock and drift
    move the terminal spot in the correct direction
3.  Portfolio ISIN consistency — same underlying gets the same path
    across all products in run_path_scenario
4.  Correlation matrix — valid matrix, perfect/anti correlation cases
5.  Product return / annual-return consistency — exact numerical cases
    for BRC (no-breach, barrier breach) and MBRC (worst-of breach)
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.reverse_convertible import ReverseConvertible
from src.scenario_engine import ScenarioEngine
from src.market_data_engine import MarketDataEngine
from tests.conftest import (
    make_brc_row, make_mbrc_row, make_portfolio,
    BETA_MAP, VOL_MAP,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_engine(portfolio=None):
    return ScenarioEngine(
        portfolio=portfolio if portfolio is not None else make_portfolio(),
        beta_map=BETA_MAP,
        vol_map=VOL_MAP,
    )


def base_scenario(**overrides):
    s = {
        "market_shock": 0,
        "n_shocks": 1,
        "shock_in_days": 0,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.0,
        "post_shock_drift_pa": 0.0,
    }
    s.update(overrides)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATH DURATION
# ══════════════════════════════════════════════════════════════════════════════

class TestPathDuration:
    """
    build_shock_paths uses the maximum maturity across the whole portfolio as the
    endpoint for the business-day grid, so every product's path — regardless of
    its own maturity — runs all the way to the portfolio horizon.
    """

    def test_path_ends_at_portfolio_max_maturity(self):
        """
        Last date in paths dict must be the last business day on or before the
        portfolio's maximum maturity.  (If max_maturity falls on a weekend,
        bdate_range ends at the preceding Friday.)
        """
        brc  = make_brc_row(maturity_date="2027-06-01")
        mbrc = make_mbrc_row(maturity_date="2028-01-01")
        portfolio = pd.DataFrame([brc, mbrc])
        e = make_engine(portfolio)

        _, _, _, _, paths = e.build_shock_paths(brc, base_scenario())

        last_date    = pd.Timestamp(paths["CH0012221716"]["date"].iloc[-1])
        max_maturity = pd.Timestamp("2028-01-01")

        # last business day on or before max_maturity
        expected_last_bday = pd.bdate_range(end=max_maturity, periods=1)[0]

        assert last_date == expected_last_bday

    def test_path_length_equals_business_days_to_max_maturity(self):
        """Number of path rows == number of business days from today to max maturity."""
        brc  = make_brc_row(maturity_date="2027-06-01")
        mbrc = make_mbrc_row(maturity_date="2028-01-01")
        portfolio = pd.DataFrame([brc, mbrc])
        e = make_engine(portfolio)

        _, _, _, _, paths = e.build_shock_paths(brc, base_scenario())
        path_len = len(paths["CH0012221716"])

        today = pd.Timestamp.today().normalize()
        expected_len = len(pd.bdate_range(start=today, end=pd.Timestamp("2028-01-01")))

        assert path_len == expected_len

    def test_shorter_product_final_spot_captured_at_own_maturity(self):
        """
        final_spots for the BRC (maturity 2027-06-01) must be read from the
        path at that date, NOT at the portfolio horizon (2028-01-01).
        """
        brc  = make_brc_row(maturity_date="2027-06-01", current_spot=100.0)
        mbrc = make_mbrc_row(maturity_date="2028-01-01")
        portfolio = pd.DataFrame([brc, mbrc])
        e = make_engine(portfolio)

        _, final_spots, _, _, paths = e.build_shock_paths(brc, base_scenario())

        # Locate the price on the first business day >= 2027-06-01
        path_df = paths["CH0012221716"]
        maturity = pd.Timestamp("2027-06-01")
        price_at_maturity = path_df[path_df["date"] >= maturity].iloc[0]["price"]

        assert abs(final_spots[0] - round(float(price_at_maturity), 4)) < 1e-3

    def test_all_isins_have_same_length_path(self):
        """Every ISIN in a multi-underlying product gets the same-length path."""
        e = make_engine()
        _, _, _, _, paths = e.build_shock_paths(make_mbrc_row(), base_scenario())

        lengths = [len(df) for df in paths.values()]
        assert len(set(lengths)) == 1, f"Path lengths differ: {lengths}"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SCENARIO DIRECTIONAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioDirectionality:
    """
    Because _get_isin_rng_map seeds each ISIN deterministically (MD5 hash),
    the GBM noise is identical across scenario runs.  The only difference
    between scenarios is the discrete shock and the drift phase.  This makes
    directional comparisons exact, not statistical.
    """

    def _final(self, engine, row, **scenario_kw):
        _, final_spots, _, _, _ = engine.build_shock_paths(row, base_scenario(**scenario_kw))
        return final_spots[0]

    # ── Shock direction ──────────────────────────────────────────────────────

    def test_large_positive_shock_raises_final(self):
        e = make_engine()
        row = make_brc_row(current_spot=100.0)
        f_base = self._final(e, row, market_shock=0)
        f_up   = self._final(e, row, market_shock=50)
        assert f_up > f_base

    def test_large_negative_shock_lowers_final(self):
        e = make_engine()
        row = make_brc_row(current_spot=100.0)
        f_base = self._final(e, row, market_shock=0)
        f_down = self._final(e, row, market_shock=-50)
        assert f_down < f_base

    def test_ordering_across_three_shocks(self):
        """down_30 < base < up_30 — strict monotonicity."""
        e = make_engine()
        row = make_brc_row(current_spot=100.0)
        f_down = self._final(e, row, market_shock=-30)
        f_base = self._final(e, row, market_shock=0)
        f_up   = self._final(e, row, market_shock=30)
        assert f_down < f_base < f_up

    # ── Drift direction ──────────────────────────────────────────────────────

    def test_positive_post_shock_drift_raises_final(self):
        """Recovery drift after shock → higher final than permanent shock."""
        e = make_engine()
        row = make_brc_row(current_spot=100.0)
        f_perm   = self._final(e, row, market_shock=-20, post_shock_drift_pa=0.0)
        f_recover = self._final(e, row, market_shock=-20, post_shock_drift_pa=0.10)
        assert f_recover > f_perm

    def test_negative_post_shock_drift_lowers_final(self):
        """Continued deterioration drift → lower final than permanent shock."""
        e = make_engine()
        row = make_brc_row(current_spot=100.0)
        f_perm  = self._final(e, row, market_shock=-20, post_shock_drift_pa=0.0)
        f_bear  = self._final(e, row, market_shock=-20, post_shock_drift_pa=-0.15)
        assert f_bear < f_perm

    def test_positive_pre_shock_drift_raises_spot_at_shock_time(self):
        """
        Higher pre-shock drift → spot has drifted up more before the shock hits.
        shock_in_days=180 gives six months of pre-shock drift window to take effect.
        """
        e = make_engine()
        row = make_brc_row(current_spot=100.0)
        # shock delayed 180 days so pre-shock drift has 6 months to compound
        f_no_drift   = self._final(e, row, market_shock=-20, shock_in_days=180, pre_shock_drift_pa=0.0)
        f_with_drift = self._final(e, row, market_shock=-20, shock_in_days=180, pre_shock_drift_pa=0.20)
        assert f_with_drift > f_no_drift

    # ── MBRC: both underlyings move with their own beta ──────────────────────

    def test_mbrc_higher_beta_underlying_moves_more(self):
        """
        NOVN has beta=1.2, NESN has beta=1.0.
        Under the same negative market shock the drift phase hits NOVN harder,
        so its path should diverge more from the no-drift baseline.
        """
        e = make_engine()
        row = make_mbrc_row(current_spots=[100.0, 100.0])

        _, finals_drift, _, _, _   = e.build_shock_paths(
            row, base_scenario(market_shock=-20, post_shock_drift_pa=-0.10)
        )
        _, finals_nodrift, _, _, _ = e.build_shock_paths(
            row, base_scenario(market_shock=-20, post_shock_drift_pa=0.0)
        )

        # Both should be lower with bear drift, but NOVN (idx=1, beta=1.2) drops more
        drop_nesn = finals_nodrift[0] - finals_drift[0]
        drop_novn = finals_nodrift[1] - finals_drift[1]
        assert drop_novn > drop_nesn


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PORTFOLIO ISIN CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioIsinConsistency:
    """
    The same ISIN must travel the exact same price path regardless of which
    product it is evaluated through — both in build_shock_paths and in the
    full run_path_scenario pass.
    """

    def test_same_isin_same_path_across_two_products(self):
        """BRC and MBRC share NESN — their NESN paths must be bit-identical."""
        e = make_engine()
        s = base_scenario()
        _, _, _, _, paths_brc  = e.build_shock_paths(make_brc_row(),  s)
        _, _, _, _, paths_mbrc = e.build_shock_paths(make_mbrc_row(), s)

        np.testing.assert_array_equal(
            paths_brc["CH0012221716"]["price"].values,
            paths_mbrc["CH0012221716"]["price"].values,
            err_msg="NESN path diverged between BRC and MBRC evaluations"
        )

    def test_portfolio_run_is_deterministic(self):
        """
        Same scenario → same paths on every run (scenario seed is derived from
        the scenario dict, so identical inputs always produce identical outputs).
        """
        e = make_engine()
        s = base_scenario()
        result_1 = e.run_path_scenario(s)
        result_2 = e.run_path_scenario(s)

        np.testing.assert_array_equal(
            result_1["paths"]["CH0012221716"]["price"].values,
            result_2["paths"]["CH0012221716"]["price"].values,
        )

    def test_novn_path_only_in_mbrc_not_brc(self):
        """NOVN only appears in MBRC — it must not show up in a BRC-only path dict."""
        e = make_engine()
        _, _, _, _, paths_brc = e.build_shock_paths(make_brc_row(), base_scenario())
        assert "CH0012221717" not in paths_brc

    def test_portfolio_paths_contains_all_unique_isins(self):
        e = make_engine()
        result = e.run_path_scenario(base_scenario())
        assert "CH0012221716" in result["paths"]  # NESN
        assert "CH0012221717" in result["paths"]  # NOVN

    def test_different_scenarios_produce_different_path_shapes(self):
        """
        Two scenarios that differ only in market_shock must now produce visually
        distinct path shapes (not just a scaled version of the same line),
        because the scenario_seed changes with the scenario parameters.
        """
        e = make_engine()
        result_down = e.run_path_scenario(base_scenario(market_shock=-20))
        result_up   = e.run_path_scenario(base_scenario(market_shock=20))

        path_down = result_down["paths"]["CH0012221716"]["price"].values[1:]  # skip day 0
        path_up   = result_up["paths"]["CH0012221716"]["price"].values[1:]

        # If paths were just scaled copies, the ratio would be constant across all days
        ratios = path_down / path_up
        assert ratios.std() > 0.001, "Paths are identical in shape (just scaled) — scenario_seed not working"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CORRELATION MATRIX VALIDITY
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationMatrix:
    """
    Tests for MarketDataEngine.build_corr_matrix and ScenarioEngine.get_corr_subset.
    """

    # ── get_corr_subset ───────────────────────────────────────────────────────

    def _full_corr_df(self):
        isins = ["CH0012221716", "CH0012221717", "CH0012221718"]
        data  = np.array([[1.0, 0.7, 0.4],
                          [0.7, 1.0, 0.6],
                          [0.4, 0.6, 1.0]])
        return pd.DataFrame(data, index=isins, columns=isins)

    def test_subset_is_symmetric(self):
        e = make_engine()
        result = e.get_corr_subset(make_mbrc_row(), self._full_corr_df())
        np.testing.assert_array_almost_equal(result, result.T)

    def test_subset_diagonal_is_one(self):
        e = make_engine()
        result = e.get_corr_subset(make_mbrc_row(), self._full_corr_df())
        np.testing.assert_array_almost_equal(np.diag(result), [1.0, 1.0])

    def test_subset_off_diagonal_matches_source(self):
        e = make_engine()
        result = e.get_corr_subset(make_mbrc_row(), self._full_corr_df())
        # NESN-NOVN correlation in source is 0.7
        assert abs(result[0, 1] - 0.7) < 1e-9
        assert abs(result[1, 0] - 0.7) < 1e-9

    def test_subset_order_follows_isin_list(self):
        """If ISIN order in the row is [NOVN, NESN], subset must reflect that order."""
        e = make_engine()
        row = make_mbrc_row()
        # Swap the ISIN order in the row
        swapped_row = row.copy()
        swapped_row["underlying_isins"] = ["CH0012221717", "CH0012221716"]  # NOVN first

        result   = e.get_corr_subset(swapped_row,  self._full_corr_df())
        original = e.get_corr_subset(row, self._full_corr_df())

        # Swapped result should equal a row/col permutation of original
        # original[0,1] = NESN-NOVN = 0.7, swapped[0,1] = NOVN-NESN = 0.7 (same value)
        assert abs(result[0, 1] - original[0, 1]) < 1e-9

    # ── build_corr_matrix via MarketDataEngine ────────────────────────────────

    def _make_db_with_monthly_prices(self, engine, isin_prices: dict, n_months=48):
        """
        Write synthetic monthly prices to the engine's DB.
        isin_prices: {isin: list of monthly price multipliers (len n_months)}
        """
        rows = []
        base_date = pd.Timestamp("2020-01-31")
        for isin, multipliers in isin_prices.items():
            price = 100.0
            for m, mult in enumerate(multipliers):
                price *= mult
                rows.append({
                    "date"  : base_date + pd.DateOffset(months=m),
                    "isin"  : isin,
                    "ticker": isin + ".SW",
                    "price" : round(price, 4),
                })
        engine.save_db(pd.DataFrame(rows))

    def test_matrix_is_symmetric(self, tmp_path):
        client = type("C", (), {"get_monthly_prices": lambda *a, **kw: []})()
        engine = MarketDataEngine(client=client, db_path=str(tmp_path / "p.csv"))

        # Same growth factor each month → perfect correlation
        mults = [1.01] * 48
        self._make_db_with_monthly_prices(engine, {"CH001": mults, "CH002": mults})

        corr = engine.build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        np.testing.assert_array_almost_equal(corr.values, corr.values.T)

    def test_diagonal_is_one(self, tmp_path):
        client = type("C", (), {"get_monthly_prices": lambda *a, **kw: []})()
        engine = MarketDataEngine(client=client, db_path=str(tmp_path / "p.csv"))

        mults = [1.01] * 48
        self._make_db_with_monthly_prices(engine, {"CH001": mults, "CH002": mults})

        corr = engine.build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        np.testing.assert_array_almost_equal(np.diag(corr.values), [1.0, 1.0])

    def test_identical_series_gives_correlation_one(self, tmp_path):
        """Two price series with identical returns → correlation = +1."""
        client = type("C", (), {"get_monthly_prices": lambda *a, **kw: []})()
        engine = MarketDataEngine(client=client, db_path=str(tmp_path / "p.csv"))

        mults = [1.01, 0.99, 1.02, 1.00, 1.01, 0.98] * 8  # 48 months
        self._make_db_with_monthly_prices(engine, {"CH001": mults, "CH002": mults})

        corr = engine.build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        assert abs(corr.loc["CH001", "CH002"] - 1.0) < 1e-6

    def test_opposite_series_gives_correlation_minus_one(self, tmp_path):
        """Series A goes up when B goes down → correlation ≈ -1."""
        client = type("C", (), {"get_monthly_prices": lambda *a, **kw: []})()
        engine = MarketDataEngine(client=client, db_path=str(tmp_path / "p.csv"))

        up   = [1.02, 0.98] * 24   # alternates up/down
        down = [0.98, 1.02] * 24   # exact opposite
        self._make_db_with_monthly_prices(engine, {"CH001": up, "CH002": down})

        corr = engine.build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        assert corr.loc["CH001", "CH002"] < -0.99

    def test_off_diagonal_between_minus_one_and_one(self, tmp_path):
        client = type("C", (), {"get_monthly_prices": lambda *a, **kw: []})()
        engine = MarketDataEngine(client=client, db_path=str(tmp_path / "p.csv"))

        import random
        random.seed(0)
        mults_a = [1.0 + random.uniform(-0.03, 0.03) for _ in range(48)]
        mults_b = [1.0 + random.uniform(-0.03, 0.03) for _ in range(48)]
        self._make_db_with_monthly_prices(engine, {"CH001": mults_a, "CH002": mults_b})

        corr = engine.build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})
        val = corr.loc["CH001", "CH002"]
        assert -1.0 <= val <= 1.0

    def test_insufficient_data_raises(self, tmp_path):
        client = type("C", (), {"get_monthly_prices": lambda *a, **kw: []})()
        engine = MarketDataEngine(client=client, db_path=str(tmp_path / "p.csv"))

        mults = [1.01] * 5   # only 5 months — far below the 12-month minimum
        self._make_db_with_monthly_prices(engine, {"CH001": mults, "CH002": mults})

        with pytest.raises(ValueError, match="common monthly observations"):
            engine.build_corr_matrix({"CH001": "A.SW", "CH002": "B.SW"})


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PRODUCT RETURN & ANNUAL-RETURN CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

class TestProductReturnConsistency:
    """
    Exact numerical cases.  All use 360-day convention as in the source.

    Key identities:
      total_cost    = notional * cost_price
      coupon_pmt    = notional * coupon * (days / 360)
      T             = days / 360
      return_pa     = return_pct * (360 / days)
                    = pnl / total_cost * (360 / days)

    For a no-breach product bought at par (cost_price=1):
      return_pa == coupon   (exactly — the T factors cancel)
    """

    # ── Case 1: BRC, no breach, bought at par ────────────────────────────────

    def test_brc_no_breach_at_par_return_pa_equals_coupon(self):
        """
        BRC, no barrier breach, purchased at par.
        return_pa must equal the coupon rate exactly.
        """
        COUPON = 0.08
        DAYS   = 366   # 2024 is a leap year

        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=COUPON,
            current_spot=110.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        # final_levels=[10.0] → final = 110 * 1.10 = 121 > strike=100 → no breach
        rc = ReverseConvertible(row, final_levels=[10.0])

        assert rc.barrier_breached() is False
        assert rc.redemption() == 100_000
        assert abs(rc.return_pa() - COUPON) < 1e-9

    def test_brc_no_breach_at_par_exact_payoff(self):
        COUPON = 0.08
        DAYS   = 366

        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=COUPON,
            current_spot=110.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        rc = ReverseConvertible(row, final_levels=[10.0])

        expected_coupon_pmt = 100_000 * COUPON * (DAYS / 360)
        expected_payoff     = 100_000 + expected_coupon_pmt

        assert abs(rc.coupon_payment() - expected_coupon_pmt) < 1e-6
        assert abs(rc.total_payoff()   - expected_payoff)     < 1e-6

    # ── Case 2: BRC, barrier breach, bought at par ───────────────────────────

    def test_brc_barrier_breach_redemption_is_performance_times_notional(self):
        """
        30% downside shock → final=70, strike=100 → breach.
        Redemption = notional * performance = 100,000 * 0.70 = 70,000.
        """
        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            current_spot=100.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        rc = ReverseConvertible(row, final_levels=[-30.0])
        # final = 70, strike = 100 → breach
        assert rc.barrier_breached() is True
        assert abs(rc.redemption() - 70_000) < 1e-6

    def test_brc_barrier_breach_pnl_is_negative(self):
        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            current_spot=100.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        rc = ReverseConvertible(row, final_levels=[-30.0])
        # redemption=70,000 + coupon_pmt — this is less than cost=100,000
        assert rc.pnl() < 0

    def test_brc_barrier_breach_exact_payoff(self):
        DAYS = 366
        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            current_spot=100.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        rc = ReverseConvertible(row, final_levels=[-30.0])

        expected_redemption  = 100_000 * 0.70
        expected_coupon_pmt  = 100_000 * 0.08 * (DAYS / 360)
        expected_payoff      = expected_redemption + expected_coupon_pmt

        assert abs(rc.total_payoff() - expected_payoff) < 1e-6

    # ── Case 3: BRC bought at discount ───────────────────────────────────────

    def test_brc_discount_purchase_higher_return(self):
        """Buying at 95 instead of 100 → higher return_pct for same payoff."""
        row_par      = make_brc_row(notional=100_000, cost_price=1.00,
                                    current_spot=110.0, strike=100.0)
        row_discount = make_brc_row(notional=100_000, cost_price=0.95,
                                    current_spot=110.0, strike=100.0)

        rc_par      = ReverseConvertible(row_par,      final_levels=[10.0])
        rc_discount = ReverseConvertible(row_discount, final_levels=[10.0])

        assert rc_discount.return_pct() > rc_par.return_pct()

    def test_brc_discount_exact_cost(self):
        row = make_brc_row(notional=100_000, cost_price=0.95)
        rc  = ReverseConvertible(row)
        assert abs(rc.total_cost() - 95_000) < 1e-9

    # ── Case 4: MBRC worst-of breach ─────────────────────────────────────────

    def test_mbrc_worst_of_determines_payoff(self):
        """
        NESN +10% (no breach), NOVN -25% (breach).
        Worst-of performance = NOVN = 0.75.
        Redemption = 100,000 * 0.75 = 75,000.
        """
        row = make_mbrc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        # NESN shock=+10, NOVN shock=-25
        rc = ReverseConvertible(row, final_levels=[10.0, -25.0])

        assert rc.worst_underlying() == "NOVN"
        assert rc.barrier_breached() is True
        assert abs(rc.redemption() - 75_000) < 1e-6

    def test_mbrc_no_breach_if_all_above_strike(self):
        """Both underlyings above strike → no breach → full redemption."""
        row = make_mbrc_row(
            notional=100_000, cost_price=1.0,
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
        )
        rc = ReverseConvertible(row, final_levels=[5.0, 3.0])
        assert rc.barrier_breached() is False
        assert rc.redemption() == 100_000

    def test_mbrc_one_breach_sufficient(self):
        """Even if NESN is fine, a NOVN breach alone triggers loss of principal."""
        row = make_mbrc_row(
            notional=100_000, cost_price=1.0,
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
        )
        # NESN above strike, NOVN below strike
        rc = ReverseConvertible(row, final_levels=[10.0, -15.0])
        assert rc.barrier_breached() is True
        assert rc.redemption() < 100_000

    # ── Return annualisation identity ─────────────────────────────────────────

    def test_return_pa_equals_return_pct_times_360_over_days(self):
        """return_pa = return_pct * 360 / days — checked for several durations."""
        for start, end, days in [
            ("2024-01-01", "2024-07-01", 182),
            ("2024-01-01", "2025-01-01", 366),
            ("2024-01-01", "2026-01-01", 731),
        ]:
            row = make_brc_row(
                current_spot=110.0, strike=100.0,
                initial_fixing_date=start, maturity_date=end,
            )
            rc = ReverseConvertible(row, final_levels=[10.0])
            expected_pa = rc.return_pct() * 360 / days
            assert abs(rc.return_pa() - expected_pa) < 1e-9, \
                f"return_pa mismatch for {start}→{end}"

    def test_zero_cost_return_pct_is_nan(self):
        row = make_brc_row(notional=0, cost_price=1.0)
        rc  = ReverseConvertible(row, final_levels=[10.0])
        assert math.isnan(rc.return_pct())
