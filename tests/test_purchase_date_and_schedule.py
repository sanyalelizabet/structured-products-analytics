"""
Tests for the buy-date / coupon-schedule extension to ReverseConvertible:
  - accrued_at_purchase
  - dirty-price total_cost
  - cash_flows
  - IRR-based ytm
  - explicit coupon_dates / day_count input via row
"""
import math
import pytest
import pandas as pd
from src.pricing.products.reverse_convertible import ReverseConvertible
from tests.conftest import make_brc_row


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def row_with(**overrides):
    """make_brc_row with arbitrary fields set/overridden via Series mutation."""
    row = make_brc_row(
        notional=100_000, cost_price=1.0, coupon=0.08,
        current_spot=110.0, strike=100.0,
        initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
    )
    for k, v in overrides.items():
        row[k] = v
    return row


# ─────────────────────────────────────────
# Schedule defaults vs explicit input
# ─────────────────────────────────────────

class TestScheduleConstruction:
    def test_default_schedule_is_single_bullet_at_maturity(self):
        rc = ReverseConvertible(row_with())
        assert len(rc.schedule.payment_dates) == 1
        assert rc.schedule.payment_dates[0] == pd.Timestamp("2025-01-01")

    def test_default_day_count_is_act_360(self):
        rc = ReverseConvertible(row_with())
        assert rc.schedule.day_count == "ACT/360"

    def test_explicit_coupon_dates_used(self):
        row = row_with(
            initial_fixing_date="2024-01-01",
            maturity_date="2026-01-01",
            coupon_dates=["2025-01-01", "2026-01-01"],
        )
        rc = ReverseConvertible(row)
        assert len(rc.schedule.payment_dates) == 2

    def test_explicit_day_count_used(self):
        row = row_with(day_count="30/360")
        rc = ReverseConvertible(row)
        assert rc.schedule.day_count == "30/360"

    def test_empty_coupon_dates_falls_back_to_single_bullet(self):
        row = row_with(coupon_dates=[])
        rc = ReverseConvertible(row)
        assert len(rc.schedule.payment_dates) == 1


# ─────────────────────────────────────────
# Accrued at purchase
# ─────────────────────────────────────────

class TestAccrued:
    def test_accrued_zero_when_purchase_equals_initial_fixing(self):
        rc = ReverseConvertible(row_with(purchase_date="2024-01-01"))
        assert rc.accrued_at_purchase() == 0.0

    def test_accrued_zero_when_purchase_date_missing(self):
        # No purchase_date set at all → falls back to initial_fixing_date
        rc = ReverseConvertible(row_with())
        assert rc.accrued_at_purchase() == 0.0

    def test_accrued_halfway_is_half_coupon_30_360(self):
        # 30/360: 2024-01-01 → 2024-07-01 is exactly 0.5 of one year
        row = row_with(
            day_count="30/360",
            purchase_date="2024-07-01",
        )
        rc = ReverseConvertible(row)
        # full coupon = 100_000 × 0.08 × 1.0 = 8_000 → half = 4_000
        assert abs(rc.accrued_at_purchase() - 4_000) < 1e-9

    def test_accrued_resets_after_coupon_payment(self):
        # 2-year product with annual coupons.
        # Purchase one day after the first coupon → accrued near zero.
        row = row_with(
            initial_fixing_date="2024-01-01",
            maturity_date="2026-01-01",
            coupon_dates=["2025-01-01", "2026-01-01"],
            day_count="30/360",
            purchase_date="2025-01-02",
        )
        rc = ReverseConvertible(row)
        assert rc.accrued_at_purchase() < 100  # tiny — well under one full coupon


# ─────────────────────────────────────────
# Dirty-price total_cost
# ─────────────────────────────────────────

class TestTotalCost:
    def test_total_cost_at_issuance_is_clean_only(self):
        rc = ReverseConvertible(row_with(cost_price=0.98, purchase_date="2024-01-01"))
        assert abs(rc.total_cost() - 98_000) < 1e-9

    def test_total_cost_includes_accrued_when_bought_mid_period(self):
        row = row_with(
            cost_price=0.99,
            day_count="30/360",
            purchase_date="2024-07-01",  # halfway through year
        )
        rc = ReverseConvertible(row)
        # 99_000 clean + 4_000 accrued
        assert abs(rc.total_cost() - 103_000) < 1e-9

    def test_pnl_identity_holds_with_dirty_cost(self):
        rc = ReverseConvertible(
            row_with(purchase_date="2024-07-01", day_count="30/360"),
            final_levels=[10.0],
        )
        assert abs(rc.pnl() - (rc.total_payoff() - rc.total_cost())) < 1e-9


# ─────────────────────────────────────────
# Cash flows
# ─────────────────────────────────────────

class TestCashFlows:
    def test_single_bullet_has_three_flows(self):
        # outflow at purchase, coupon at maturity, redemption at maturity
        rc = ReverseConvertible(row_with(), final_levels=[10.0])
        flows = rc.cash_flows()
        assert len(flows) == 3
        assert flows[0][1] < 0  # outflow first

    def test_multi_coupon_has_n_plus_two_flows(self):
        # 2 coupons → outflow + 2 coupons + redemption = 4 entries
        row = row_with(
            initial_fixing_date="2024-01-01",
            maturity_date="2026-01-01",
            coupon_dates=["2025-01-01", "2026-01-01"],
        )
        rc = ReverseConvertible(row, final_levels=[10.0])
        flows = rc.cash_flows()
        assert len(flows) == 4

    def test_outflow_equals_negative_total_cost(self):
        rc = ReverseConvertible(row_with(), final_levels=[10.0])
        flows = rc.cash_flows()
        assert abs(flows[0][1] + rc.total_cost()) < 1e-9

    def test_inflows_sum_to_total_payoff(self):
        rc = ReverseConvertible(row_with(), final_levels=[10.0])
        flows = rc.cash_flows()
        inflows = sum(amt for _, amt in flows if amt > 0)
        assert abs(inflows - rc.total_payoff()) < 1e-9

    def test_past_coupons_excluded_when_bought_after_them(self):
        # 2-year product, buy after the first coupon — should see only the second
        row = row_with(
            initial_fixing_date="2024-01-01",
            maturity_date="2026-01-01",
            coupon_dates=["2025-01-01", "2026-01-01"],
            purchase_date="2025-06-01",
        )
        rc = ReverseConvertible(row, final_levels=[10.0])
        flows = rc.cash_flows()
        # outflow + 1 future coupon + redemption = 3
        assert len(flows) == 3


# ─────────────────────────────────────────
# YTM as true IRR
# ─────────────────────────────────────────

class TestYTM:
    def test_ytm_zero_when_payoff_equals_cost(self):
        # cost_price=1.0, no coupon, no breach → payoff == notional == cost
        row = row_with(coupon=0.0)
        rc = ReverseConvertible(row, final_levels=[10.0])
        assert abs(rc.ytm()) < 1e-6

    def test_ytm_positive_when_profitable(self):
        rc = ReverseConvertible(row_with(), final_levels=[10.0])
        assert rc.ytm() > 0

    def test_ytm_negative_on_breach_with_loss(self):
        # current_spot=110, barrier=60 (initial 100 × 0.60); a 50% shock takes
        # the final to 55 ≤ barrier → breach; the small coupon won't cover the loss.
        row = row_with(coupon=0.02)
        rc = ReverseConvertible(row, final_levels=[-50.0])
        assert rc.ytm() < 0

    def test_ytm_single_bullet_matches_compound_formula(self):
        # Single bullet → IRR satisfies (1+r)^T = payoff/cost where T = days/360
        row = row_with()  # 366 days, coupon 8%, no breach
        rc = ReverseConvertible(row, final_levels=[10.0])
        ratio = rc.total_payoff() / rc.total_cost()
        T_years = 366 / 360  # _xirr uses ACT/360 basis
        expected = ratio ** (1 / T_years) - 1
        assert abs(rc.ytm() - expected) < 1e-6

    def test_ytm_differs_from_simple_return_pa_for_multi_coupon(self):
        # For a multi-period product, true IRR ≠ simple annualization
        row = row_with(
            coupon=0.20,  # large intermediate coupon → clear compounding divergence
            initial_fixing_date="2024-01-01",
            maturity_date="2026-01-01",
            coupon_dates=["2025-01-01", "2026-01-01"],
        )
        rc = ReverseConvertible(row, final_levels=[10.0])
        # IRR includes compounding effect of intermediate coupon
        # while return_pa is pure simple annualization
        assert abs(rc.ytm() - rc.return_pa()) > 1e-4

    def test_ytm_today_nan_after_maturity(self):
        # Maturity is in the past → no future flows
        row = row_with(
            initial_fixing_date="2020-01-01",
            maturity_date="2020-12-31",
        )
        rc = ReverseConvertible(row, final_levels=[10.0])
        assert math.isnan(rc.ytm_today())


# ─────────────────────────────────────────
# coupon_payment derived from schedule
# ─────────────────────────────────────────

class TestCouponPaymentFromSchedule:
    def test_total_equals_sum_over_periods(self):
        row = row_with(
            initial_fixing_date="2024-01-01",
            maturity_date="2026-01-01",
            coupon_dates=["2025-01-01", "2026-01-01"],
            day_count="30/360",
        )
        rc = ReverseConvertible(row)
        # Two periods × notional × coupon × 1.0 each
        assert abs(rc.coupon_payment() - 2 * 100_000 * 0.08) < 1e-9

    def test_break_even_uses_total_coupon(self):
        row = row_with(
            initial_fixing_date="2024-01-01",
            maturity_date="2026-01-01",
            coupon_dates=["2025-01-01", "2026-01-01"],
            day_count="30/360",
        )
        rc = ReverseConvertible(row)
        # break_even = 1 - total_coupon / notional = 1 - 0.16 = 0.84
        assert abs(rc.break_even() - 0.84) < 1e-9
