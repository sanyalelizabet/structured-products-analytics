import pandas as pd


class CouponSchedule:
    """
    Coupon payment schedule for structured products.

    Holds the contractual coupon dates and the day-count convention used to
    compute period accruals. All amounts are derived from `(notional, rate)`
    so the same schedule can be reused across products with the same calendar.
    """

    DAY_COUNTS = ("30/360", "ACT/360", "ACT/365")

    def __init__(self, payment_dates, period_start, day_count="ACT/360"):
        if day_count not in self.DAY_COUNTS:
            raise ValueError(
                f"Unsupported day_count {day_count!r}. "
                f"Use one of {self.DAY_COUNTS}."
            )
        if not payment_dates:
            raise ValueError("payment_dates must contain at least one date.")

        self.period_start = pd.Timestamp(period_start)
        self.payment_dates = sorted(pd.Timestamp(d) for d in payment_dates)
        self.day_count = day_count

        if self.payment_dates[0] <= self.period_start:
            raise ValueError("All payment_dates must be after period_start.")

    @classmethod
    def single_bullet(cls, period_start, maturity, day_count="ACT/360"):
        """One coupon paid at maturity over the full life of the product."""
        return cls([maturity], period_start, day_count=day_count)

    def period_boundaries(self):
        """List of (start, end) tuples for each accrual period."""
        starts = [self.period_start] + self.payment_dates[:-1]
        return list(zip(starts, self.payment_dates))

    def year_fraction(self, d1, d2):
        d1 = pd.Timestamp(d1)
        d2 = pd.Timestamp(d2)
        if self.day_count == "30/360":
            y1, m1, day1 = d1.year, d1.month, min(d1.day, 30)
            y2, m2, day2 = d2.year, d2.month, d2.day
            if day1 == 30 and day2 == 31:
                day2 = 30
            return ((y2 - y1) * 360 + (m2 - m1) * 30 + (day2 - day1)) / 360.0
        if self.day_count == "ACT/360":
            return (d2 - d1).days / 360.0
        return (d2 - d1).days / 365.0  # ACT/365

    def coupon_amounts(self, notional, annual_rate):
        """Per-period coupon amounts, aligned with `payment_dates`."""
        return [
            notional * annual_rate * self.year_fraction(s, e)
            for s, e in self.period_boundaries()
        ]

    def total_amount(self, notional, annual_rate):
        return sum(self.coupon_amounts(notional, annual_rate))

    def current_period(self, as_of):
        """Returns the (start, end) of the accrual period containing `as_of`."""
        as_of = pd.Timestamp(as_of)
        for s, e in self.period_boundaries():
            if s <= as_of <= e:
                return s, e
        return None

    def accrued(self, notional, annual_rate, as_of):
        """
        Coupon accrued from the start of the current period up to `as_of`.
        Returns 0 if `as_of` is outside the schedule.
        """
        period = self.current_period(as_of)
        if period is None:
            return 0.0
        s, e = period
        full_period_yf = self.year_fraction(s, e)
        if full_period_yf <= 0:
            return 0.0
        elapsed_yf = self.year_fraction(s, as_of)
        full_amount = notional * annual_rate * full_period_yf
        return full_amount * (elapsed_yf / full_period_yf)

    def future_cashflows(self, notional, annual_rate, as_of):
        """
        Coupon payments strictly after `as_of`.
        Returns list of (pd.Timestamp, amount).
        """
        as_of = pd.Timestamp(as_of)
        return [
            (date, amount)
            for date, amount in zip(
                self.payment_dates, self.coupon_amounts(notional, annual_rate)
            )
            if date > as_of
        ]
