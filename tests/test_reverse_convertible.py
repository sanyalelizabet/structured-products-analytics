"""
Tests for ReverseConvertible.
"""
import math
import pytest
import pandas as pd
from src.pricing.products.reverse_convertible import ReverseConvertible
from tests.conftest import make_brc_row, make_mbrc_row


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def brc(final_pct=0.0, **kw):
    """Construct a BRC with a single final-level shock (% from current spot)."""
    row = make_brc_row(**kw)
    return ReverseConvertible(row, final_levels=[final_pct])


def mbrc(final_pcts=None, **kw):
    row = make_mbrc_row(**kw)
    final_pcts = final_pcts or [0.0, 0.0]
    return ReverseConvertible(row, final_levels=final_pcts)


# ─────────────────────────────────────────
# __init__ / validation
# ─────────────────────────────────────────

class TestInit:
    def test_mismatched_final_levels_raises(self):
        row = make_brc_row()
        with pytest.raises(ValueError, match="Scenario length"):
            ReverseConvertible(row, final_levels=[0.0, 0.0])

    def test_default_final_levels_equal_current_spots(self):
        row = make_brc_row(current_spot=95.0)
        rc = ReverseConvertible(row)
        # 0% shock → final == current spot
        assert rc.final_levels == [95.0]

    def test_positive_shock_increases_final(self):
        row = make_brc_row(current_spot=100.0)
        rc = ReverseConvertible(row, final_levels=[10.0])
        assert abs(rc.final_levels[0] - 110.0) < 1e-9

    def test_negative_shock_decreases_final(self):
        row = make_brc_row(current_spot=100.0)
        rc = ReverseConvertible(row, final_levels=[-20.0])
        assert abs(rc.final_levels[0] - 80.0) < 1e-9


# ─────────────────────────────────────────
# is_multi
# ─────────────────────────────────────────

class TestIsMulti:
    def test_brc_is_not_multi(self):
        assert not brc().is_multi()

    def test_mbrc_is_multi(self):
        assert mbrc().is_multi()


# ─────────────────────────────────────────
# performances / current_performances
# ─────────────────────────────────────────

class TestPerformances:
    def test_performances_at_par(self):
        # final == strike → performance == 1.0
        rc = brc(final_pct=0.0, current_spot=100.0, strike=100.0)
        assert abs(rc.delivery_performances()[0] - 1.0) < 1e-9

    def test_performances_above_strike(self):
        rc = brc(final_pct=10.0, current_spot=100.0, strike=100.0)
        assert abs(rc.delivery_performances()[0] - 1.10) < 1e-9

    def test_performances_below_strike(self):
        rc = brc(final_pct=-20.0, current_spot=100.0, strike=100.0)
        assert abs(rc.delivery_performances()[0] - 0.80) < 1e-9

    def test_current_performances_uses_spot_not_final(self):
        rc = brc(current_spot=90.0, strike=100.0)
        assert abs(rc.current_performances()[0] - 0.90) < 1e-9

    def test_mbrc_performances_length(self):
        rc = mbrc()
        assert len(rc.delivery_performances()) == 2

    def test_performance_brc_is_single(self):
        rc = brc(final_pct=-10.0, current_spot=100.0, strike=100.0)
        assert abs(rc.delivery_performance() - rc.delivery_performances()[0]) < 1e-9

    def test_performance_mbrc_is_worst_of(self):
        rc = mbrc(
            final_pcts=[-10.0, -30.0],
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
        )
        assert abs(rc.delivery_performance() - min(rc.delivery_performances())) < 1e-9


# ─────────────────────────────────────────
# Barrier logic  ← BUG DOCUMENTED HERE
# ─────────────────────────────────────────

class TestBarrier:
    """
    barrier_breaches_final() compares the final fixing against the barrier,
    where ``barrier = initial_level × barrier_pct`` (NOT the strike). With the
    test defaults (initial_level=100, barrier_pct=0.60) the barrier is 60.
    """

    def test_no_breach_above_strike(self):
        """Final above strike (and barrier) → no breach."""
        rc = brc(final_pct=5.0, current_spot=100.0, strike=100.0)
        # final=105, barrier=60 → no breach
        assert rc.barrier_breaches_final() == [False]

    def test_no_breach_between_barrier_and_strike(self):
        """Final below strike but above barrier → no breach (the whole point
        of a barrier: par is preserved unless the barrier is breached)."""
        rc = brc(final_pct=-20.0, current_spot=100.0, strike=100.0)
        # final=80, barrier=60 → no breach
        assert rc.barrier_breaches_final() == [False]

    def test_breach_below_barrier(self):
        """Final below barrier → breach."""
        rc = brc(final_pct=-45.0, current_spot=100.0, strike=100.0)
        # final=55, barrier=60 → breach
        assert rc.barrier_breaches_final() == [True]

    def test_breach_at_barrier(self):
        """Final exactly at barrier → breach (<=)."""
        rc = brc(final_pct=-40.0, current_spot=100.0, strike=100.0)
        # final=60, barrier=60 → breach
        assert rc.barrier_breaches_final() == [True]

    def test_mbrc_one_underlying_breaches(self):
        """MBRC: one underlying below its barrier → overall breach."""
        rc = mbrc(
            final_pcts=[5.0, -50.0],
            initial_levels=[100.0, 100.0],
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
        )
        # barriers=[60,60]; finals=[105,50] → second breaches
        assert rc.barrier_breaches_final() == [False, True]
        assert rc.barrier_breached() is True

    def test_mbrc_no_breach_when_all_above_barrier(self):
        """MBRC: all underlyings above their barrier → no breach."""
        rc = mbrc(
            final_pcts=[5.0, 3.0],
            initial_levels=[100.0, 100.0],
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
        )
        assert rc.barrier_breaches_final() == [False, False]
        assert rc.barrier_breached() is False

    def test_american_matches_european_on_flat_projection(self):
        # On this deterministic analytics path (flat spot to maturity) there is
        # no intermediate price path, so continuous (American) observation
        # reduces to the European terminal check — it must not raise, and must
        # agree with the European result.
        row_eur = make_brc_row(); row_eur["type_style"] = "european"
        row_am  = make_brc_row(); row_am["type_style"]  = "american"
        eur = ReverseConvertible(row_eur).barrier_breached()
        am  = ReverseConvertible(row_am).barrier_breached()
        assert am == eur

    def test_barrier_breached_raises_for_unknown_style(self):
        row = make_brc_row()
        row["type_style"] = "bermudan"
        rc = ReverseConvertible(row)
        with pytest.raises(NotImplementedError):
            rc.barrier_breached()


# ─────────────────────────────────────────
# Redemption / payoff / pnl
# ─────────────────────────────────────────

class TestPayoff:
    def test_full_redemption_above_strike(self):
        """No barrier breach at maturity → full notional returned."""
        rc = brc(final_pct=5.0, current_spot=100.0, strike=100.0, notional=100_000)
        assert rc.redemption() == 100_000

    def test_coupon_payment_one_year(self):
        rc = brc(
            coupon=0.08,
            notional=100_000,
            initial_fixing_date="2024-01-01",
            maturity_date="2025-01-01",
        )
        T = rc.total_product_time()
        expected = 100_000 * 0.08 * T
        assert abs(rc.coupon_payment() - expected) < 0.01

    def test_payoff_equals_redemption_plus_coupon(self):
        rc = brc(final_pct=5.0, current_spot=100.0, strike=100.0)
        assert abs(rc.payoff() - (rc.redemption() + rc.coupon_payment())) < 1e-9

    def test_total_cost(self):
        rc = brc(notional=100_000, cost_price=0.98)
        assert abs(rc.total_cost() - 98_000) < 1e-9

    def test_pnl_is_payoff_minus_cost(self):
        rc = brc(final_pct=5.0, current_spot=100.0, strike=100.0)
        assert abs(rc.pnl() - (rc.payoff() - rc.total_cost())) < 1e-9

    def test_return_pct_zero_cost_is_nan(self):
        row = make_brc_row(notional=0, cost_price=1.0)
        rc = ReverseConvertible(row)
        assert math.isnan(rc.return_pct())

    def test_return_pct_positive_when_profitable(self):
        rc = brc(final_pct=5.0, current_spot=100.0, strike=100.0,
                 notional=100_000, cost_price=1.0, coupon=0.08)
        assert rc.return_pct() > 0


# ─────────────────────────────────────────
# Barrier distances
# ─────────────────────────────────────────

class TestBarrierDistance:
    # Barrier = initial_level × barrier_pct. With initial_level=100 and
    # barrier_pct=0.70 the barrier sits at 70, independent of the strike.

    def test_distance_formula(self):
        # spot=90, barrier=100×0.70=70 → (90-70)/90 = 0.2222...
        rc = brc(current_spot=90.0, initial_level=100.0, barrier_pct=0.70)
        distances = rc.current_barrier_distances()
        assert abs(distances[0] - (20 / 90)) < 1e-9

    def test_distance_zero_at_barrier(self):
        # spot exactly at barrier (70) → distance = 0
        rc = brc(current_spot=70.0, initial_level=100.0, barrier_pct=0.70)
        assert abs(rc.current_barrier_distances()[0]) < 1e-9

    def test_distance_to_barrier_brc_single(self):
        rc = brc(current_spot=90.0, initial_level=100.0, barrier_pct=0.70)
        assert abs(rc.distance_to_barrier() - rc.current_barrier_distances()[0]) < 1e-9

    def test_distance_to_barrier_mbrc_is_minimum(self):
        rc = mbrc(
            initial_levels=[100.0, 100.0],
            current_spots=[90.0, 70.0],
            strikes=[70.0, 60.0],
            barrier_pct=0.70,
        )
        expected = min(rc.current_barrier_distances())
        assert abs(rc.distance_to_barrier() - expected) < 1e-9

    def test_negative_distance_means_breached_intraday(self):
        # spot below barrier (70) → distance negative
        rc = brc(current_spot=55.0, initial_level=100.0, barrier_pct=0.70)
        assert rc.current_barrier_distances()[0] < 0


# ─────────────────────────────────────────
# Break-even / worst underlying
# ─────────────────────────────────────────

class TestBreakEven:
    def test_break_even_less_than_one(self):
        rc = brc(coupon=0.08, initial_fixing_date="2024-01-01", maturity_date="2025-01-01")
        assert rc.break_even() < 1.0

    def test_break_even_decreases_with_higher_coupon(self):
        rc_low = brc(coupon=0.05)
        rc_high = brc(coupon=0.10)
        assert rc_high.break_even() < rc_low.break_even()


class TestStrikeDifferentFromInitial:
    """When `strike ≠ initial`, the spec splits the two references:
    the barrier is referenced to the initial fixing level, and the
    redemption on breach is referenced to the strike.
    """

    def test_breach_uses_initial_redemption_uses_strike(self):
        # initial = 100, barrier_pct = 0.60 → barrier = 60.
        # strike = 80 (off-ATM, different from initial).
        # final = current_spot × (1 + final_pct/100) = 100 × 0.55 = 55.
        # 55 ≤ 60 → breach. Redemption = N × 55 / 80 = 68_750.
        rc = brc(
            notional=100_000,
            initial_level=100.0, strike=80.0, barrier_pct=0.60,
            current_spot=100.0, final_pct=-45.0,
        )
        assert rc.barrier_breached() is True
        assert abs(rc.redemption() - 100_000 * 55 / 80) < 1e-9

    def test_no_breach_with_off_atm_strike_redeems_at_par(self):
        # Same fixture but final = 70 > barrier 60 → no breach. Par
        # redemption, independent of the strike.
        rc = brc(
            notional=100_000,
            initial_level=100.0, strike=80.0, barrier_pct=0.60,
            current_spot=100.0, final_pct=-30.0,
        )
        assert rc.barrier_breached() is False
        assert rc.redemption() == 100_000


class TestWorstUnderlying:
    def test_brc_worst_is_only_underlying(self):
        rc = brc()
        assert rc.worst_underlying() == "NESN"

    def test_mbrc_worst_is_lower_performer(self):
        # NOVN (index 1) drops 30%, NESN (index 0) drops 10%
        rc = mbrc(
            final_pcts=[-10.0, -30.0],
            current_spots=[100.0, 100.0],
            strikes=[100.0, 100.0],
        )
        assert rc.worst_underlying() == "NOVN"


# ─────────────────────────────────────────
# total_product_time
# ─────────────────────────────────────────

class TestProductTime:
    def test_one_year_product(self):
        rc = brc(initial_fixing_date="2024-01-01", maturity_date="2025-01-01")
        # 366 days (2024 leap year) / 360 ≈ 1.0167
        assert abs(rc.total_product_time() - 366 / 360) < 1e-6

    def test_six_month_product(self):
        rc = brc(initial_fixing_date="2024-01-01", maturity_date="2024-07-01")
        assert rc.total_product_time() < 1.0


# ─────────────────────────────────────────
# Previous spots / delta helpers
# ─────────────────────────────────────────

class TestPreviousSpots:
    def test_no_price_db_returns_current_spots(self):
        row = make_brc_row(current_spot=95.0)
        rc = ReverseConvertible(row, price_db=None)
        assert rc._get_previous_spots() == rc.current_spots

    def test_with_single_price_row_falls_back(self):
        import pandas as pd
        row = make_brc_row(current_spot=95.0)
        db = pd.DataFrame([{
            "isin": "CH0012221716", "date": pd.Timestamp("2024-12-31"), "price": 95.0
        }])
        rc = ReverseConvertible(row, price_db=db)
        # only one row in db → falls back to current spot
        assert rc._get_previous_spots() == [95.0]

    def test_with_two_price_rows_returns_second_most_recent(self):
        import pandas as pd
        row = make_brc_row(current_spot=95.0)
        db = pd.DataFrame([
            {"isin": "CH0012221716", "date": pd.Timestamp("2024-12-30"), "price": 93.0},
            {"isin": "CH0012221716", "date": pd.Timestamp("2024-12-31"), "price": 95.0},
        ])
        rc = ReverseConvertible(row, price_db=db)
        assert rc._get_previous_spots() == [93.0]

    def test_summary_pnl_delta_none_without_history(self):
        """Without a price DB, prev_spots == current_spots so the
        pnl/distance deltas are reported as None."""
        row = make_brc_row(current_spot=95.0)
        rc = ReverseConvertible(row, price_db=None)
        s = rc.summary()
        assert s["pnl_delta"]      is None
        assert s["distance_delta"] is None
