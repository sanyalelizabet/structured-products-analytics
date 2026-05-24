from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.coupon_schedule import CouponSchedule


def barrier_levels(initial_levels, barrier_pct) -> np.ndarray:
    """Absolute downside barrier per underlying.

    The barrier is a fraction of the **initial fixing level**, not the
    strike::

        barrier_i = initial_level_i * barrier_pct

    A missing / ``None`` ``barrier_pct`` is treated as ``1.0`` — i.e. no
    separate downside barrier, with the breach observed at the initial
    level (plain reverse-convertible behaviour).
    """
    bp = 1.0 if barrier_pct is None else float(barrier_pct)
    return np.asarray([float(x) for x in initial_levels], dtype=float) * bp


def vectorised_european_rc_summary(row, final_prices: np.ndarray) -> dict:
    """Vectorised per-path summary for European reverse convertibles.

    Pure-numpy equivalent of looping ``ReverseConvertible(...).summary()`` over
    a ``(N, n_assets)`` terminal-price tensor.

    Parameters
    ----------
    row : pd.Series
        Portfolio row. Requires ``notional``, ``strike``, ``initial_levels``,
        ``barrier_pct``, ``underlyings``, plus the fields
        :class:`ReverseConvertible` needs for ``total_cost`` / ``coupon_payment``.
    final_prices : np.ndarray
        Terminal underlying prices, shape ``(n_paths, n_assets)``.

    Returns
    -------
    dict
        Per-path arrays/scalars matching the keys of ``ReverseConvertible.summary()``.
    """
    final_prices = np.asarray(final_prices, dtype=float)
    N, n_assets = final_prices.shape

    notional    = float(row["notional"])
    strikes     = np.array([float(k) for k in row["strike"]])  # (n_assets,)
    barriers    = barrier_levels(row["initial_levels"], row.get("barrier_pct"))  # (n_assets,)
    underlyings = list(row["underlyings"])

    # Constants — computed once via a prototype RC instance.
    rc_proto       = ReverseConvertible(row, final_levels=[0.0] * n_assets)
    coupon_payment = rc_proto.coupon_payment()
    total_cost     = rc_proto.total_cost()

    perfs       = final_prices / strikes[None, :]                     # (N, n_assets)
    worst_idx   = np.argmin(perfs, axis=1)                            # (N,)
    worst_perf  = perfs[np.arange(N), worst_idx]                      # (N,)

    # European barrier observation at final fixing: breach if ANY underlying's
    # terminal level is at or below its barrier (= initial_level × barrier_pct).
    breached    = (final_prices <= barriers[None, :]).any(axis=1)     # (N,)

    # Redemption: par if the barrier held, else worst-of conversion at strike.
    redemption    = np.where(breached, notional * worst_perf, notional)
    total_payoff  = redemption + coupon_payment
    pnl           = total_payoff - total_cost
    return_pct    = (pnl / total_cost) if total_cost else np.full(N, np.nan)

    worst_underlying = np.array([underlyings[i] for i in worst_idx], dtype=object)

    strike_used     = strikes[worst_idx]                              # (N,)
    final_spot_used = final_prices[np.arange(N), worst_idx]           # (N,)
    # Physical delivery occurs exactly when the barrier is breached — the
    # worst-of is converted into the underlying; otherwise cash at par.
    physical        = breached

    total_shares     = notional / strike_used                         # (N,)
    delivered_shares = np.where(physical, np.floor(total_shares), 0).astype(int)
    fractional_cash  = np.where(
        physical, (total_shares - delivered_shares) * final_spot_used, 0.0,
    )
    cash_redemption  = np.where(physical, fractional_cash, total_payoff)
    settlement_type  = np.where(physical, "physical", "cash").astype(object)
    delivered_under  = np.where(physical, worst_underlying, None)

    return {
        "pnl":                  pnl,
        "return_pct":           return_pct,
        "total_payoff":         total_payoff,
        "barrier_breached":     breached,
        "worst_idx":            worst_idx,
        "worst_underlying":     worst_underlying,
        "strike_used":          strike_used,
        "final_spot_used":      final_spot_used,
        "settlement_type":      settlement_type,
        "delivered_underlying": delivered_under,
        "delivered_shares":     delivered_shares,
        "fractional_cash":      fractional_cash,
        "cash_redemption":      cash_redemption,
        "total_cost":           float(total_cost),
    }


class ReverseConvertible:

    def __init__(self, row, final_levels=None, price_db=None):
        self.row = row
        self.price_db = price_db

        # Basic inputs
        self.position_units = int(row["position_units"])
        self.cost_price = row["cost_price"]
        self.coupon = row["coupon"]
        self.barrier_pct = row["barrier_pct"]
        self.product_type = row["product_type"]
        self.type_style = row["type_style"]

        # Notional: per-piece denomination + total position notional.
        # `denomination` is the contractual nominal of one certificate;
        # `total_notional` = denomination × position_units. `notional` is
        # kept as an alias for backward compatibility.
        self._init_notional(row)

        # Issuer info (optional — see _init_issuer for which fields are read)
        self._init_issuer(row)

        # Underlyings
        self.underlyings = row["underlyings"]
        self.initial_levels = row["initial_levels"]
        self.strike_levels = row["strike"]
        self.current_spots = row["current_spots"]

        # Coupon schedule (defaults to single bullet at maturity for back-compat)
        self.schedule = self._build_schedule(row)

        if final_levels is None:
            final_levels = [0.0] * len(self.current_spots)

        if len(final_levels) != len(self.current_spots):
            raise ValueError("Scenario length must match number of underlyings.")

        # final_levels here are scenario shocks in %
        self.final_levels = [
            spot * (1 + level / 100)
            for spot, level in zip(self.current_spots, final_levels)
        ]

    def _init_notional(self, row):
        """
        Resolve `denomination` (per-piece) and `total_notional` (position).
        Prefers explicit `denomination`; otherwise falls back to legacy
        `notional` field, treating it as the total position notional and
        deriving denomination as `notional / position_units`.
        """
        denom = self._read_optional(row, "denomination", numeric=True)
        if denom is not None:
            self.denomination = float(denom)
            self.total_notional = self.denomination * self.position_units
        else:
            if "notional" not in row.index or row["notional"] is None:
                raise ValueError(
                    "Row must provide either `denomination` or legacy `notional`."
                )
            self.total_notional = float(row["notional"])
            self.denomination = (
                self.total_notional / self.position_units
                if self.position_units else self.total_notional
            )
        # Backward-compat alias used internally and by older callers.
        self.notional = self.total_notional

    def _init_issuer(self, row):
        """Reads optional issuer fields. All default to None when absent."""
        self.issuer = self._read_optional(row, "issuer")
        self.issuer_rating = self._read_optional(row, "issuer_rating")
        self.issuer_country = self._read_optional(row, "issuer_country")

    @staticmethod
    def _read_optional(row, field, numeric=False):
        """Reads a row field, treating missing / NaN / empty-string as None."""
        if field not in row.index:
            return None
        val = row[field]
        if val is None:
            return None
        if isinstance(val, float) and pd.isna(val):
            return None
        if not numeric and isinstance(val, str) and not val.strip():
            return None
        return val

    @staticmethod
    def _build_schedule(row):
        day_count = "ACT/360"
        if "day_count" in row.index:
            dc = row["day_count"]
            if isinstance(dc, str) and dc:
                day_count = dc

        coupon_dates = None
        if "coupon_dates" in row.index:
            val = row["coupon_dates"]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                coupon_dates = list(val)

        if coupon_dates is None:
            return CouponSchedule.single_bullet(
                row["initial_fixing_date"], row["maturity_date"], day_count=day_count
            )
        return CouponSchedule(
            coupon_dates, row["initial_fixing_date"], day_count=day_count
        )
    
    # =========================
    # PREVIOUS SPOTS FROM DB
    # =========================
    def _get_previous_spots(self):
        """
        Look up the second most recent price per ISIN from price_db.
        Falls back to current_spots if DB not available or insufficient history.
        """
        if self.price_db is None:
            return self.current_spots

        prev_spots = []

        for isin, current_spot in zip(self.row["underlying_isins"], self.current_spots):
            prices = (
                self.price_db[self.price_db["isin"] == isin]
                .sort_values("date")
                .reset_index(drop=True)
            )

            if len(prices) >= 2:
                prev_spots.append(float(prices.iloc[-2]["price"]))
            else:
                prev_spots.append(current_spot)  # no history — no delta

        return prev_spots

    def barrier_levels(self):
        """Per-underlying absolute barrier = initial fixing level × barrier_pct.

        See the module-level :func:`barrier_levels` for the convention.
        """
        return barrier_levels(self.initial_levels, self.barrier_pct)

    def _redemption_for(self, spots) -> float:
        """Closed-form redemption when ``spots`` are treated as terminal levels.

        Mirrors :meth:`redemption` exactly but takes spots as an argument
        instead of using ``self.final_levels``.  Used by :meth:`summary`
        to compute yesterday's hypothetical PnL without instantiating a
        whole second :class:`ReverseConvertible` (which is expensive —
        rebuilds the coupon schedule, parses dates, etc.).
        """
        barriers = self.barrier_levels()
        breached = any(s <= b for s, b in zip(spots, barriers))
        if not breached:
            return self.notional
        worst_perf = min(s / k for s, k in zip(spots, self.strike_levels))
        return self.notional * worst_perf

    def _pnl_for(self, spots) -> float:
        """PnL if ``spots`` were the terminal levels — total_cost / coupon
        are constants of the product."""
        return (
            self._redemption_for(spots)
            + self.coupon_payment()
            - self.total_cost()
        )

    def _min_distance_for(self, spots) -> float:
        """Distance-to-barrier for a hypothetical spot vector.

        Distance is measured to the barrier (= initial × barrier_pct), as a
        fraction of the hypothetical spot.
        """
        barriers = self.barrier_levels()
        distances = [(s - b) / s for s, b in zip(spots, barriers)]
        return min(distances) if self.is_multi() else distances[0]


    def is_multi(self):
        return len(self.initial_levels) > 1
    
    def performances(self):
        """
        Final performance vs strike. for each underlying in a given MRC
        Better for payoff/redemption logic.
        """
        return [
            final / strike
            for final, strike in zip(self.final_levels, self.strike_levels)
        ]
    
    def current_performances(self):
        """
        Current spot performance vs strike.
        """
        return [
            spot / strike
            for spot, strike in zip(self.current_spots, self.strike_levels)
        ]
    
    def performance(self):
        """
        Relevant payoff performance:
        - BRC: single underlying performance
        - MBRC: worst-of performance
        """
        if self.is_multi():
            return min(self.performances())
        return self.performances()[0]
    
    def barrier_breaches_final(self):
        """
        European barrier observation at final fixing.
        Barrier level = initial fixing level × barrier_pct (NOT the strike).
        """
        return [
            final <= barrier
            for final, barrier in zip(self.final_levels, self.barrier_levels())
        ]
    
    def barrier_breached(self):
        if self.type_style.lower() != "european":
            raise NotImplementedError("Only European style implemented.")
        return any(self.barrier_breaches_final())
    
    def redemption(self):
        """
        If no barrier event: nominal redemption.
        If barrier event: economic value of delivery.
        """
        if not self.barrier_breached():
            return self.notional
        return self.notional * self.performance()
    
    def total_product_time(self):
        start = datetime.strptime(self.row["initial_fixing_date"], "%Y-%m-%d")
        maturity = datetime.strptime(self.row["maturity_date"], "%Y-%m-%d")
        return (maturity - start).days / 360

    def coupon_payment(self):
        """Total contractual coupon over the life of the product."""
        return self.schedule.total_amount(self.notional, self.coupon)

    def accrued_at_purchase(self):
        """
        Coupon accrued from the start of the current accrual period up to
        purchase_date. Buyer reimburses the seller for this — adds to cost.
        Zero when the product is bought at issuance.
        """
        return self.schedule.accrued(
            self.notional, self.coupon, self._purchase_date()
        )

    def payoff(self):
        return self.redemption() + self.coupon_payment()

    def total_payoff(self):
        return self.payoff()

    def total_cost(self):
        """Dirty price: clean cost + accrued interest at purchase."""
        return self.notional * self.cost_price + self.accrued_at_purchase()

    def pnl(self):
        return self.total_payoff() - self.total_cost()
    
    def return_pct(self):
        total_cost = self.total_cost()
        if total_cost == 0:
            return np.nan
        return self.pnl() / total_cost

    def _purchase_date(self):
        """Returns purchase_date if set in row, else falls back to initial_fixing_date."""
        val = None
        if "purchase_date" in self.row.index:
            val = self.row["purchase_date"]
        if val is None or (hasattr(val, '__class__') and str(val) in ("nan", "NaT", "None")):
            val = self.row["initial_fixing_date"]
        return datetime.strptime(str(val), "%Y-%m-%d")

    def cash_flows(self, as_of=None):
        """
        Dated cash flows from `as_of` (default: purchase_date) to maturity.
        Used as the basis for the IRR-based yield to maturity.

        Convention:
          - Outflow: -total_cost (dirty price) at `as_of`
          - Inflow:  each scheduled coupon strictly after `as_of`
          - Inflow:  redemption at maturity (separate from the maturity coupon)
        """
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(self._purchase_date())
        flows = [(as_of, -self.total_cost())]
        flows.extend(
            self.schedule.future_cashflows(self.notional, self.coupon, as_of)
        )
        flows.append((pd.Timestamp(self.row["maturity_date"]), self.redemption()))
        return flows

    @staticmethod
    def _xirr(cash_flows):
        """Internal XIRR — annual rate r where NPV of all cash flows is zero."""
        if not cash_flows:
            return np.nan
        dates, amounts = zip(*sorted(cash_flows, key=lambda x: x[0]))
        t0 = pd.Timestamp(dates[0])
        years = [(pd.Timestamp(d) - t0).days / 360.0 for d in dates]  # ACT/360

        def npv(r):
            return sum(cf / (1 + r) ** t for cf, t in zip(amounts, years))

        try:
            return brentq(npv, -0.9999, 100.0, maxiter=1000)
        except (ValueError, RuntimeError):
            return np.nan

    def ytm(self):
        """
        Yield to Maturity — true IRR over scheduled cash flows from purchase
        date to maturity. Accounts for the timing of every coupon.
        """
        return self._xirr(self.cash_flows())

    def ytm_today(self):
        """
        Forward YTM from today — IRR over remaining cash flows, with the
        outflow today equal to current dirty price (notional × cost_price +
        accrued from previous coupon to today).
        """
        today = pd.Timestamp(datetime.today().date())
        if today >= pd.Timestamp(self.row["maturity_date"]):
            return np.nan
        accrued_today = self.schedule.accrued(self.notional, self.coupon, today)
        cost_today = self.notional * self.cost_price + accrued_today
        flows = [(today, -cost_today)]
        flows.extend(
            self.schedule.future_cashflows(self.notional, self.coupon, today)
        )
        flows.append((pd.Timestamp(self.row["maturity_date"]), self.redemption()))
        return self._xirr(flows)

    def return_pa(self):
        """
        Simple annualized return: return_pct × 360 / days_held.
        Money-yield convention — does not compound. Use `ytm()` for the
        professional IRR-based yield.
        """
        days = (datetime.strptime(self.row["maturity_date"], "%Y-%m-%d")
                - self._purchase_date()).days
        if days <= 0:
            return np.nan
        return self.return_pct() * 360 / days
    
    def current_barrier_distances(self):
        """
        How far spot can fall from current price before hitting the barrier.
        Expressed as a fraction of current spot:
            (spot - barrier) / spot  where barrier = initial_level × barrier_pct
        Positive = above barrier, 0 = at barrier, negative = below barrier.
        """
        return [
            (spot - barrier) / spot
            for spot, barrier in zip(self.current_spots, self.barrier_levels())
        ]
    
    def distance_to_barrier(self):
        distances = self.current_barrier_distances()
        if self.is_multi():
            return min(distances)
        return distances[0]
    
    def break_even(self):
        """
        Performance level (as a fraction of strike) below which PnL turns
        negative. Total coupon income offsets capital loss.
        """
        if self.notional == 0:
            return 1.0
        return 1 - self.coupon_payment() / self.notional
    
    def worst_underlying(self):
        if self.is_multi():
            idx = np.argmin(self.performances())
            return self.underlyings[idx]
        return self.underlyings[0]
    def summary(self):
        maturity = datetime.strptime(self.row["maturity_date"], "%Y-%m-%d")
        today = datetime.today()
        days_to_expiry = (maturity - today).days

        # Yesterday's hypothetical PnL / distance — computed in closed
        # form against prev spots instead of building a second RC
        # (avoids re-parsing dates / re-building the coupon schedule).
        prev_spots = self._get_previous_spots()
        if prev_spots == self.current_spots:
            pnl_delta      = None
            distance_delta = None
        else:
            pnl_delta      = self.pnl() - self._pnl_for(prev_spots)
            distance_delta = self.distance_to_barrier() - self._min_distance_for(prev_spots)

        return {
            "product_type": self.product_type,
            "is_multi": self.is_multi(),
            "issuer": self.issuer,
            "issuer_rating": self.issuer_rating,
            "issuer_country": self.issuer_country,
            "denomination": self.denomination,
            "position_units": self.position_units,
            "total_notional": self.total_notional,
            "maturity_date": self.row["maturity_date"],
            "days_to_expiry": days_to_expiry,
            "performance": self.performance() - 1,
            "barrier_breached": self.barrier_breached(),
            "worst_underlying": self.worst_underlying(),
            "payoff_per_unit": self.payoff() / self.position_units,
            "total_payoff": self.total_payoff(),
            "total_cost": self.total_cost(),
            "pnl": self.pnl(),
            "return_pct": self.return_pct(),
            "ytm": self.ytm(),
            "ytm_today": self.ytm_today(),
            "return_pa": self.return_pa(),
            "distance_to_barrier": self.distance_to_barrier(),
            "break_even": self.break_even(),

            # Deltas — None if no history
            "pnl_delta": pnl_delta,
            "distance_delta": distance_delta,
        }
    