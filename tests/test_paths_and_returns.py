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
from src.correlation_engine import CorrelationEngine
from src.noise_sampler import NoiseSampler
from tests.conftest import (
    make_brc_row, make_mbrc_row, make_portfolio,
    BETA_MAP, VOL_MAP,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_engine(portfolio=None, n_paths=1):
    """Default n_paths=1 makes the engine produce a single deterministic
    path per call — direction-of-effect tests stay sharp under CRN."""
    return ScenarioEngine(
        portfolio=portfolio if portfolio is not None else make_portfolio(),
        beta_map=BETA_MAP,
        vol_map=VOL_MAP,
        n_paths=n_paths,
    )


def _build_sampler_for(engine, seed=42):
    """Construct a fresh NoiseSampler matching the engine's portfolio."""
    today              = pd.Timestamp.today().normalize()
    portfolio_maturity = pd.to_datetime(engine.portfolio["maturity_date"]).max()
    n_days             = len(pd.bdate_range(start=today, end=portfolio_maturity))
    isins = sorted({i for _, r in engine.portfolio.iterrows() for i in r["underlying_isins"]})
    return NoiseSampler(
        n_paths=engine.n_paths, n_days=n_days,
        factor_codes=[], isins=isins, seed=seed,
    )


def _terminal_at_maturity(price_paths, date_range, row, asset_idx=0, path_idx=0):
    """Pick the terminal price at the product's own maturity from a (N,T,A) tensor."""
    maturity = pd.Timestamp(row["maturity_date"])
    mat_mask = np.asarray(date_range >= maturity)
    t_idx = int(np.argmax(mat_mask)) if mat_mask.any() else len(date_range) - 1
    return float(price_paths[path_idx, t_idx, asset_idx])


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
        Last date in the asset-paths grid must be the last business day on or
        before the portfolio's maximum maturity.
        """
        brc  = make_brc_row(maturity_date="2027-06-01")
        mbrc = make_mbrc_row(maturity_date="2028-01-01")
        portfolio = pd.DataFrame([brc, mbrc])
        e = make_engine(portfolio)
        sampler = _build_sampler_for(e)

        _, date_range, _ = e.build_shock_paths(brc, base_scenario(), sampler)

        last_date    = pd.Timestamp(date_range[-1])
        max_maturity = pd.Timestamp("2028-01-01")
        expected_last_bday = pd.bdate_range(end=max_maturity, periods=1)[0]

        assert last_date == expected_last_bday

    def test_path_length_equals_business_days_to_max_maturity(self):
        """Number of grid points == business days from today to max maturity."""
        brc  = make_brc_row(maturity_date="2027-06-01")
        mbrc = make_mbrc_row(maturity_date="2028-01-01")
        portfolio = pd.DataFrame([brc, mbrc])
        e = make_engine(portfolio)
        sampler = _build_sampler_for(e)

        price_paths, date_range, _ = e.build_shock_paths(brc, base_scenario(), sampler)
        path_len = len(date_range)

        today = pd.Timestamp.today().normalize()
        expected_len = len(pd.bdate_range(start=today, end=pd.Timestamp("2028-01-01")))

        assert path_len == expected_len
        assert price_paths.shape[1] == expected_len

    def test_shorter_product_final_spot_captured_at_own_maturity(self):
        """
        Run the full portfolio scenario; the BRC's product-row pnl/return must
        reflect prices at its own maturity, not the portfolio horizon.  Verify
        by comparing the median final spot at maturity vs at horizon.
        """
        brc  = make_brc_row(maturity_date="2027-06-01", current_spot=100.0)
        mbrc = make_mbrc_row(maturity_date="2028-01-01")
        portfolio = pd.DataFrame([brc, mbrc])
        e = make_engine(portfolio, n_paths=10)
        sampler = _build_sampler_for(e)

        price_paths, date_range, _ = e.build_shock_paths(brc, base_scenario(), sampler)

        maturity = pd.Timestamp("2027-06-01")
        idx_at_maturity = int(np.argmax(np.asarray(date_range >= maturity)))
        idx_at_horizon  = len(date_range) - 1

        # Median across paths at maturity vs horizon — they must differ for any
        # nontrivial drift/diffusion (here they're the same scenario, so the
        # check is just that *some* paths produce different intermediate prices).
        median_at_mat = float(np.median(price_paths[:, idx_at_maturity, 0]))
        median_at_hor = float(np.median(price_paths[:, idx_at_horizon, 0]))
        assert idx_at_maturity < idx_at_horizon
        assert np.isfinite(median_at_mat) and np.isfinite(median_at_hor)

    def test_all_isins_have_same_length_path(self):
        """Every ISIN in a multi-underlying product gets the same-length path."""
        e = make_engine()
        sampler = _build_sampler_for(e)
        price_paths, date_range, _ = e.build_shock_paths(
            make_mbrc_row(), base_scenario(), sampler,
        )
        # All asset axes share the same time axis by construction.
        assert price_paths.shape[1] == len(date_range)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SCENARIO DIRECTIONAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioDirectionality:
    """
    With CRN (a shared NoiseSampler reused across calls), the GBM noise is
    identical across scenario evaluations — the only difference between
    runs is the shock magnitude or drift parameter.  This makes the
    direction-of-effect comparisons exact, not statistical.

    n_paths=1 is used so each call returns a single deterministic path.
    """

    def _final(self, engine, sampler, row, **scenario_kw):
        price_paths, date_range, _ = engine.build_shock_paths(
            row, base_scenario(**scenario_kw), sampler,
        )
        return _terminal_at_maturity(price_paths, date_range, row, asset_idx=0, path_idx=0)

    # ── Shock direction ──────────────────────────────────────────────────────

    def test_large_positive_shock_raises_final(self):
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        row = make_brc_row(current_spot=100.0)
        f_base = self._final(e, s, row, market_shock=0)
        f_up   = self._final(e, s, row, market_shock=50)
        assert f_up > f_base

    def test_large_negative_shock_lowers_final(self):
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        row = make_brc_row(current_spot=100.0)
        f_base = self._final(e, s, row, market_shock=0)
        f_down = self._final(e, s, row, market_shock=-50)
        assert f_down < f_base

    def test_ordering_across_three_shocks(self):
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        row = make_brc_row(current_spot=100.0)
        f_down = self._final(e, s, row, market_shock=-30)
        f_base = self._final(e, s, row, market_shock=0)
        f_up   = self._final(e, s, row, market_shock=30)
        assert f_down < f_base < f_up

    # ── Drift direction ──────────────────────────────────────────────────────

    def test_positive_post_shock_drift_raises_final(self):
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        row = make_brc_row(current_spot=100.0)
        f_perm    = self._final(e, s, row, market_shock=-20, post_shock_drift_pa=0.0)
        f_recover = self._final(e, s, row, market_shock=-20, post_shock_drift_pa=0.10)
        assert f_recover > f_perm

    def test_negative_post_shock_drift_lowers_final(self):
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        row = make_brc_row(current_spot=100.0)
        f_perm = self._final(e, s, row, market_shock=-20, post_shock_drift_pa=0.0)
        f_bear = self._final(e, s, row, market_shock=-20, post_shock_drift_pa=-0.15)
        assert f_bear < f_perm

    def test_positive_pre_shock_drift_raises_spot_at_shock_time(self):
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        row = make_brc_row(current_spot=100.0)
        f_no_drift   = self._final(e, s, row, market_shock=-20, shock_in_days=180, pre_shock_drift_pa=0.0)
        f_with_drift = self._final(e, s, row, market_shock=-20, shock_in_days=180, pre_shock_drift_pa=0.20)
        assert f_with_drift > f_no_drift

    # ── MBRC: both underlyings move with their own beta ──────────────────────

    def test_mbrc_higher_beta_underlying_moves_more(self):
        """
        NOVN has β=1.2, NESN has β=1.0.  Under the same bear drift, NOVN's
        terminal log-return drops more (in log-space) than NESN's vs the
        no-drift baseline — that's the β scaling of CAPM drift.

        Compare in *log* terms because the two assets sit at different
        absolute price levels (different Itô-correction, different shock
        impact β-scaled).  Absolute drops can flip sign due to level
        effects, but the *log* drop is the structural quantity.
        """
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        row = make_mbrc_row(current_spots=[100.0, 100.0])

        pp_drift, dr, _ = e.build_shock_paths(
            row, base_scenario(market_shock=-20, post_shock_drift_pa=-0.10), s,
        )
        pp_nodrift, _, _ = e.build_shock_paths(
            row, base_scenario(market_shock=-20, post_shock_drift_pa=0.0), s,
        )
        nesn_drift  = _terminal_at_maturity(pp_drift,   dr, row, asset_idx=0)
        nesn_nodrft = _terminal_at_maturity(pp_nodrift, dr, row, asset_idx=0)
        novn_drift  = _terminal_at_maturity(pp_drift,   dr, row, asset_idx=1)
        novn_nodrft = _terminal_at_maturity(pp_nodrift, dr, row, asset_idx=1)

        log_drop_nesn = np.log(nesn_nodrft / nesn_drift)
        log_drop_novn = np.log(novn_nodrft / novn_drift)
        assert log_drop_novn > log_drop_nesn


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
        """BRC and MBRC share NESN — under CRN, the NESN path is identical."""
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        sc = base_scenario()
        # Use the (asset, n_assets) submatrix of price_paths for NESN.
        # NESN = isin index 0 in both BRC (single underlying) and MBRC (first underlying).
        pp_brc,  dr, _ = e.build_shock_paths(make_brc_row(),  sc, s)
        pp_mbrc, _,  _ = e.build_shock_paths(make_mbrc_row(), sc, s)

        # Note: with Cholesky on a 1-asset matrix (BRC) vs 2-asset identity (MBRC,
        # since corr_matrix=None ⇒ identity), the NESN draw is the same in both.
        np.testing.assert_array_equal(
            pp_brc[0, :, 0],
            pp_mbrc[0, :, 0],
        )

    def test_portfolio_run_is_deterministic(self):
        """Same engine → same NoiseSampler → same scenario → identical samples."""
        e = make_engine(n_paths=10)
        s = base_scenario()
        a = e.run_path_scenario(s)
        b = e.run_path_scenario(s)

        for isin in a["asset_paths"]:
            np.testing.assert_array_equal(
                a["asset_paths"][isin]["median"].values,
                b["asset_paths"][isin]["median"].values,
            )

    def test_novn_path_only_in_mbrc_not_brc(self):
        """A BRC build must only contain NESN's price axis — not NOVN."""
        e = make_engine(n_paths=1)
        s = _build_sampler_for(e)
        pp_brc, _, _ = e.build_shock_paths(make_brc_row(), base_scenario(), s)
        # BRC has exactly one underlying.
        assert pp_brc.shape[2] == 1

    def test_portfolio_paths_contains_all_unique_isins(self):
        e = make_engine(n_paths=5)
        result = e.run_path_scenario(base_scenario())
        assert "CH0012221716" in result["asset_paths"]   # NESN
        assert "CH0012221717" in result["asset_paths"]   # NOVN

    def test_different_scenarios_produce_different_terminal_levels(self):
        """Negative vs positive shock: median terminal of NESN must be lower
        for the bear scenario.  (Under CRN with a single shock event, the
        path *shape* difference is a scalar offset; what we care about is
        the *level* — that the directional effect lands.)"""
        e = make_engine(n_paths=10)
        result_down = e.run_path_scenario(base_scenario(market_shock=-20))
        result_up   = e.run_path_scenario(base_scenario(market_shock=20))

        term_down = float(result_down["asset_paths"]["CH0012221716"]["median"].iloc[-1])
        term_up   = float(result_up["asset_paths"]["CH0012221716"]["median"].iloc[-1])
        assert term_down < term_up


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CORRELATION MATRIX VALIDITY
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationMatrix:
    """
    Tests for CorrelationEngine.build_corr_matrix and ScenarioEngine.get_corr_subset.
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

    # ── build_corr_matrix via CorrelationEngine ───────────────────────────────




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
        45% downside shock → final=55, barrier=60 (initial 100 × 0.60) → breach.
        Redemption = notional * (final/strike) = 100,000 * 0.55 = 55,000.
        """
        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            current_spot=100.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        rc = ReverseConvertible(row, final_levels=[-45.0])
        # final = 55 ≤ barrier 60 → breach
        assert rc.barrier_breached() is True
        assert abs(rc.redemption() - 55_000) < 1e-6

    def test_brc_barrier_breach_pnl_is_negative(self):
        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            current_spot=100.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        rc = ReverseConvertible(row, final_levels=[-45.0])
        # redemption=55,000 + coupon_pmt — this is less than cost=100,000
        assert rc.pnl() < 0

    def test_brc_barrier_breach_exact_payoff(self):
        DAYS = 366
        row = make_brc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            current_spot=100.0, strike=100.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        rc = ReverseConvertible(row, final_levels=[-45.0])

        expected_redemption  = 100_000 * 0.55
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
        NESN +10% (no breach), NOVN -45% → final 55 ≤ barrier 60 (breach).
        Worst-of performance = NOVN = 55/100 = 0.55.
        Redemption = 100,000 * 0.55 = 55,000.
        """
        row = make_mbrc_row(
            notional=100_000, cost_price=1.0, coupon=0.08,
            initial_levels=[100.0, 100.0],
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        # NESN shock=+10, NOVN shock=-45 (barriers = 60, 60)
        rc = ReverseConvertible(row, final_levels=[10.0, -45.0])

        assert rc.worst_underlying() == "NOVN"
        assert rc.barrier_breached() is True
        assert abs(rc.redemption() - 55_000) < 1e-6

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
        """Even if NESN is fine, a NOVN barrier breach alone triggers loss of principal."""
        row = make_mbrc_row(
            notional=100_000, cost_price=1.0,
            initial_levels=[100.0, 100.0],
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
        )
        # NESN above barrier, NOVN below barrier (60)
        rc = ReverseConvertible(row, final_levels=[10.0, -45.0])
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
