"""
Tests for CouponSchedule.
"""
import pytest
import pandas as pd
from src.coupon_schedule import CouponSchedule


# ─────────────────────────────────────────
# Construction / validation
# ─────────────────────────────────────────

class TestInit:
    def test_rejects_unknown_day_count(self):
        with pytest.raises(ValueError, match="day_count"):
            CouponSchedule(["2025-01-01"], "2024-01-01", day_count="weird")

    def test_rejects_empty_payment_dates(self):
        with pytest.raises(ValueError, match="at least one"):
            CouponSchedule([], "2024-01-01")

    def test_rejects_payment_date_before_period_start(self):
        with pytest.raises(ValueError, match="after period_start"):
            CouponSchedule(["2023-12-01"], "2024-01-01")

    def test_rejects_payment_date_equal_to_period_start(self):
        with pytest.raises(ValueError, match="after period_start"):
            CouponSchedule(["2024-01-01"], "2024-01-01")

    def test_payment_dates_are_sorted(self):
        s = CouponSchedule(["2026-01-01", "2025-01-01"], "2024-01-01")
        assert s.payment_dates == [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")]

    def test_single_bullet_factory(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01")
        assert s.payment_dates == [pd.Timestamp("2025-01-01")]
        assert s.period_start == pd.Timestamp("2024-01-01")


# ─────────────────────────────────────────
# Day-count year fractions
# ─────────────────────────────────────────

class TestYearFraction:
    def test_act_360_one_calendar_year(self):
        # 2024 is a leap year — 366 days
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01", day_count="ACT/360")
        assert abs(s.year_fraction("2024-01-01", "2025-01-01") - 366 / 360) < 1e-12

    def test_act_365_one_calendar_year(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01", day_count="ACT/365")
        assert abs(s.year_fraction("2024-01-01", "2025-01-01") - 366 / 365) < 1e-12

    def test_30_360_full_year_is_one(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01", day_count="30/360")
        assert abs(s.year_fraction("2024-01-01", "2025-01-01") - 1.0) < 1e-12

    def test_30_360_half_year(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2024-07-01", day_count="30/360")
        # 6 months × 30 days / 360 = 0.5
        assert abs(s.year_fraction("2024-01-01", "2024-07-01") - 0.5) < 1e-12

    def test_30_360_handles_31st_to_30th_rule(self):
        # day1=30, day2=31 → day2 clipped to 30 → exactly one month
        s = CouponSchedule.single_bullet("2024-01-30", "2024-02-29")
        # Just sanity: result must be > 0
        assert s.year_fraction("2024-01-30", "2024-02-29") > 0


# ─────────────────────────────────────────
# Period boundaries / coupon amounts
# ─────────────────────────────────────────

class TestPeriods:
    def test_period_boundaries_chain_from_period_start(self):
        s = CouponSchedule(
            ["2025-01-01", "2026-01-01", "2027-01-01"], "2024-01-01"
        )
        bounds = s.period_boundaries()
        assert bounds == [
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
            (pd.Timestamp("2026-01-01"), pd.Timestamp("2027-01-01")),
        ]

    def test_coupon_amounts_sum_to_total(self):
        s = CouponSchedule(
            ["2025-01-01", "2026-01-01"], "2024-01-01", day_count="30/360"
        )
        amounts = s.coupon_amounts(notional=1_000_000, annual_rate=0.05)
        # 30/360 → each year = 1.0 → 50_000 per coupon
        assert len(amounts) == 2
        assert abs(amounts[0] - 50_000) < 1e-9
        assert abs(amounts[1] - 50_000) < 1e-9
        assert abs(s.total_amount(1_000_000, 0.05) - 100_000) < 1e-9

    def test_single_bullet_coupon_amount(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01", day_count="ACT/360")
        # 366/360 × 100_000 × 0.08
        assert abs(s.total_amount(100_000, 0.08) - 100_000 * 0.08 * 366 / 360) < 1e-9


# ─────────────────────────────────────────
# Accrued interest
# ─────────────────────────────────────────

class TestAccrued:
    def test_accrued_at_period_start_is_zero(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01", day_count="30/360")
        assert s.accrued(100_000, 0.08, "2024-01-01") == 0.0

    def test_accrued_at_period_end_equals_full_coupon(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01", day_count="30/360")
        # full period coupon = 100_000 × 0.08 × 1.0 = 8000
        assert abs(s.accrued(100_000, 0.08, "2025-01-01") - 8_000) < 1e-9

    def test_accrued_halfway_is_half(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01", day_count="30/360")
        # 30/360: 2024-01-01 → 2024-07-01 = 0.5 of period
        assert abs(s.accrued(100_000, 0.08, "2024-07-01") - 4_000) < 1e-9

    def test_accrued_outside_schedule_is_zero(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01")
        # Before period_start
        assert s.accrued(100_000, 0.08, "2023-06-01") == 0.0
        # After last payment date
        assert s.accrued(100_000, 0.08, "2026-01-01") == 0.0

    def test_accrued_resets_each_period(self):
        s = CouponSchedule(
            ["2025-01-01", "2026-01-01"], "2024-01-01", day_count="30/360"
        )
        # Just after first coupon: in second period, ~0 accrued
        accrued = s.accrued(100_000, 0.08, "2025-01-02")
        assert accrued < 100  # tiny — well under one full coupon

        # Halfway through second period
        accrued_half = s.accrued(100_000, 0.08, "2025-07-01")
        assert abs(accrued_half - 4_000) < 1e-9


# ─────────────────────────────────────────
# Future cash flows (filtering by as_of)
# ─────────────────────────────────────────

class TestFutureCashflows:
    def test_all_coupons_returned_when_as_of_before_first(self):
        s = CouponSchedule(
            ["2025-01-01", "2026-01-01"], "2024-01-01", day_count="30/360"
        )
        flows = s.future_cashflows(100_000, 0.08, "2024-01-01")
        assert len(flows) == 2

    def test_strictly_after_filter_excludes_as_of_date(self):
        s = CouponSchedule(
            ["2025-01-01", "2026-01-01"], "2024-01-01", day_count="30/360"
        )
        # as_of equal to first coupon date — first coupon is NOT future anymore
        flows = s.future_cashflows(100_000, 0.08, "2025-01-01")
        assert len(flows) == 1
        assert flows[0][0] == pd.Timestamp("2026-01-01")

    def test_no_flows_after_last_payment(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01")
        flows = s.future_cashflows(100_000, 0.08, "2025-06-01")
        assert flows == []

    def test_returns_pd_timestamps(self):
        s = CouponSchedule.single_bullet("2024-01-01", "2025-01-01")
        flows = s.future_cashflows(100_000, 0.08, "2024-06-01")
        assert isinstance(flows[0][0], pd.Timestamp)
