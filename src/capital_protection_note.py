"""Capital Protection Note (single underlying, vanilla uncapped).

Payoff at maturity (per unit of notional N):

    Payoff(S_T) = N * [pi + p * max(S_T/K - 1, 0)]

where
    pi = protection_pct      (e.g. 1.00 for 100% capital protection)
    p  = participation_pct   (e.g. 0.80 for 80% upside participation)
    K  = strike              (single underlying)

Analytic price decomposition (zero-coupon bond + scaled BS call):

    PV = N * pi * exp(-r*T)           # protection leg (ZCB)
       + (N * p / K) * Call_BS         # participation leg
       + PV(coupons)                   # discounted coupon stream (optional)

The participation leg multiplier ``N * p / K`` comes from rewriting the
payoff as ``N*pi + (N*p/K) * max(S_T - K, 0)`` — i.e. ``N*p/K`` units
of a standard European call struck at K.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from scipy.optimize import brentq

from src.coupon_schedule import CouponSchedule
from src.pricing.black_scholes import BlackScholes


# ──────────────────────────────────────────────────────────────────────────
# Vectorised payoff for scenario engines
# ──────────────────────────────────────────────────────────────────────────

def vectorised_cpn_summary(row, final_prices: np.ndarray) -> dict:
    """Vectorised per-path summary for Capital Protection Notes.

    Same dict keys as :func:`vectorised_european_rc_summary`. CPNs are
    cash-settled with no barrier, so settlement/delivery fields are constant.

    Parameters
    ----------
    row : pd.Series
        Portfolio row. Requires ``notional``, ``strike`` (length-1),
        ``underlyings`` (length-1), ``protection_pct``, ``participation_pct``,
        plus the schedule/cost fields used by :class:`CapitalProtectionNote`.
    final_prices : np.ndarray
        Terminal underlying prices, shape ``(n_paths, 1)``.
    """
    final_prices = np.asarray(final_prices, dtype=float)
    if final_prices.ndim != 2 or final_prices.shape[1] != 1:
        raise ValueError(
            "CPN payoff requires single-underlying terminal prices "
            f"of shape (n_paths, 1); got {final_prices.shape}."
        )
    N = final_prices.shape[0]

    notional      = float(row["notional"])
    strike        = float(row["strike"][0])
    underlying    = str(row["underlyings"][0])
    protection    = float(row["protection_pct"])
    participation = float(row["participation_pct"])

    # Constants — computed once via a prototype CPN instance.
    proto          = CapitalProtectionNote(row, final_level=0.0)
    coupon_payment = proto.coupon_payment()
    total_cost     = proto.total_cost()

    spots         = final_prices[:, 0]                              # (N,)
    upside        = np.maximum(spots / strike - 1.0, 0.0)            # (N,)
    redemption    = notional * (protection + participation * upside) # (N,)
    total_payoff  = redemption + coupon_payment
    pnl           = total_payoff - total_cost
    return_pct    = (pnl / total_cost) if total_cost else np.full(N, np.nan)

    worst_underlying = np.full(N, underlying, dtype=object)
    settlement_type  = np.full(N, "cash", dtype=object)
    delivered_under  = np.full(N, None, dtype=object)
    delivered_shares = np.zeros(N, dtype=int)
    fractional_cash  = np.zeros(N, dtype=float)
    barrier_breached = np.zeros(N, dtype=bool)
    strike_used      = np.full(N, strike, dtype=float)

    return {
        "pnl":                  pnl,
        "return_pct":           return_pct,
        "total_payoff":         total_payoff,
        "barrier_breached":     barrier_breached,
        "worst_idx":            np.zeros(N, dtype=int),
        "worst_underlying":     worst_underlying,
        "strike_used":          strike_used,
        "final_spot_used":      spots,
        "settlement_type":      settlement_type,
        "delivered_underlying": delivered_under,
        "delivered_shares":     delivered_shares,
        "fractional_cash":      fractional_cash,
        "cash_redemption":      total_payoff,
        "total_cost":           float(total_cost),
    }


# ──────────────────────────────────────────────────────────────────────────
# Analytic pricing  (ZCB + scaled BS call + PV of coupons)
# ──────────────────────────────────────────────────────────────────────────

def analytic_cpn_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    notional: float,
    protection_pct: float,
    participation_pct: float,
    q: float = 0.0,
    coupon_pv: float = 0.0,
) -> float:
    """Closed-form fair value of a vanilla CPN.

    ``coupon_pv`` is the present value of the contractual coupon stream;
    pass 0 for zero-coupon notes.
    """
    if T <= 0:
        intrinsic = notional * (protection_pct
                                + participation_pct * max(S / K - 1.0, 0.0))
        return intrinsic + coupon_pv
    bs = BlackScholes(S, K, T, r, sigma, q)
    zcb_leg  = notional * protection_pct * np.exp(-r * T)
    part_leg = (notional * participation_pct / K) * bs.call()
    return float(zcb_leg + part_leg + coupon_pv)


def analytic_cpn_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    notional: float,
    protection_pct: float,
    participation_pct: float,
    q: float = 0.0,
) -> dict:
    """Closed-form Greeks of the (coupon-free) CPN.

    Coupons are deterministic and have no Greek exposure beyond a
    rho contribution — left to the caller if needed.
    """
    bs = BlackScholes(S, K, T, r, sigma, q)
    scale = notional * participation_pct / K
    return {
        "delta": scale * bs.delta("call"),
        "gamma": scale * bs.gamma(),
        "vega":  scale * bs.vega(),
        "theta": scale * bs.theta("call"),
        # ZCB rho is per 1% rate move, matching BS convention.
        "rho":   scale * bs.rho("call")
                 - notional * protection_pct * T * np.exp(-r * T) * 0.01,
    }


# ──────────────────────────────────────────────────────────────────────────
# Product class
# ──────────────────────────────────────────────────────────────────────────

class CapitalProtectionNote:
    """Single-underlying vanilla (uncapped) capital protection note."""

    def __init__(self, row, final_level: Optional[float] = None, price_db=None):
        self.row = row
        self.price_db = price_db

        self.product_type = row.get("product_type", "CPN")
        self.position_units = int(row["position_units"])
        self.cost_price = float(row["cost_price"])
        self.coupon = float(row.get("coupon", 0.0) or 0.0)

        self.protection_pct    = float(row["protection_pct"])
        self.participation_pct = float(row["participation_pct"])

        if len(row["underlyings"]) != 1 or len(row["strike"]) != 1:
            raise ValueError("CPN is single-underlying; expected length-1 lists.")

        self.underlying    = str(row["underlyings"][0])
        self.initial_level = float(row["initial_levels"][0])
        self.strike        = float(row["strike"][0])
        self.current_spot  = float(row["current_spots"][0])

        # Notional bookkeeping (per-piece + total). Mirrors RC convention.
        self._init_notional(row)

        # Optional issuer chain (issuer, guarantor, keep-well counterparty).
        self.issuer              = self._read_optional(row, "issuer")
        self.issuer_rating       = self._read_optional(row, "issuer_rating")
        self.issuer_country      = self._read_optional(row, "issuer_country")
        self.guarantor           = self._read_optional(row, "guarantor")
        self.guarantor_rating    = self._read_optional(row, "guarantor_rating")
        self.keep_well_agreement = self._read_optional(row, "keep_well_agreement")

        # Optional term-sheet fields commonly disclosed in CPN prospectuses.
        # All are informational / round-trip-validation; the core payoff math
        # uses ``protection_pct``, ``participation_pct``, ``denomination`` and
        # ``strike`` exclusively.
        self.issue_price              = self._read_optional(row, "issue_price",  numeric=True)
        self.capital_protection_amount = self._read_optional(row, "capital_protection_amount", numeric=True)
        self.number_of_underlyings    = self._read_optional(row, "number_of_underlyings", numeric=True)
        self.spot_reference_price     = self._read_optional(row, "spot_reference_price", numeric=True)
        self.issue_size               = self._read_optional(row, "issue_size",  numeric=True)
        self.last_trading_day         = self._read_optional(row, "last_trading_day")
        self.payment_date             = self._read_optional(row, "payment_date")
        self.final_fixing_date        = self._read_optional(row, "final_fixing_date")
        self.bond_npv_at_issue        = self._read_optional(row, "bond_npv_at_issue", numeric=True)
        self.implied_irr_at_issue     = self._read_optional(row, "implied_irr_at_issue", numeric=True)

        self._validate_term_sheet_consistency()

        self.schedule = self._build_schedule(row)

        # Scenario shock in % applied to current_spot; None → current_spot.
        if final_level is None:
            self.final_level = self.current_spot
        else:
            self.final_level = self.current_spot * (1 + float(final_level) / 100.0)

    # ----- init helpers (kept lightweight; CPN-local copies of RC patterns) ---

    def _init_notional(self, row):
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
        self.notional = self.total_notional

    def _validate_term_sheet_consistency(self):
        """Cross-check redundant term-sheet fields against derived values.

        Real prospectuses disclose ``capital_protection_amount`` (USD) and
        ``number_of_underlyings`` (B) as headline terms even though they
        are mechanically derived from ``denomination``, ``protection_pct``,
        ``participation_pct`` and ``strike``.  We accept the row but raise
        if the disclosure disagrees with the derived value by more than a
        small rounding tolerance — that catches data-entry errors at load
        time instead of at risk-display time.
        """
        # capital_protection_amount ≟ issue_price * protection_pct
        # (fall back to denomination when issue_price isn't supplied)
        if self.capital_protection_amount is not None:
            base = self.issue_price if self.issue_price is not None else self.denomination
            derived = base * self.protection_pct
            if not np.isclose(self.capital_protection_amount, derived, atol=0.01):
                raise ValueError(
                    f"capital_protection_amount={self.capital_protection_amount} "
                    f"disagrees with issue_price * protection_pct = {derived:.4f}."
                )

        # number_of_underlyings (B) ≟ denomination / strike
        if self.number_of_underlyings is not None:
            base = self.issue_price if self.issue_price is not None else self.denomination
            derived_b = base / self.strike
            if not np.isclose(self.number_of_underlyings, derived_b, rtol=1e-4):
                raise ValueError(
                    f"number_of_underlyings={self.number_of_underlyings} "
                    f"disagrees with denomination / strike = {derived_b:.5f}."
                )

    @staticmethod
    def _read_optional(row, field, numeric=False):
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

    # ----- product mechanics -------------------------------------------------

    def performance(self) -> float:
        """Final-spot performance vs strike (1.0 = at-strike)."""
        return self.final_level / self.strike

    def upside(self) -> float:
        return max(self.performance() - 1.0, 0.0)

    def redemption(self) -> float:
        return self.notional * (self.protection_pct
                                + self.participation_pct * self.upside())

    def coupon_payment(self) -> float:
        return self.schedule.total_amount(self.notional, self.coupon)

    def payoff(self) -> float:
        return self.redemption() + self.coupon_payment()

    def total_payoff(self) -> float:
        return self.payoff()

    def _purchase_date(self):
        val = None
        if "purchase_date" in self.row.index:
            val = self.row["purchase_date"]
        if val is None or str(val) in ("nan", "NaT", "None"):
            val = self.row["initial_fixing_date"]
        return datetime.strptime(str(val), "%Y-%m-%d")

    def accrued_at_purchase(self) -> float:
        return self.schedule.accrued(
            self.notional, self.coupon, self._purchase_date()
        )

    def total_cost(self) -> float:
        return self.notional * self.cost_price + self.accrued_at_purchase()

    def pnl(self) -> float:
        return self.total_payoff() - self.total_cost()

    def return_pct(self) -> float:
        tc = self.total_cost()
        return self.pnl() / tc if tc else np.nan

    def break_even(self) -> float:
        """Performance level (S_T / K) at which PnL = 0.

        Solving ``N*(pi + p*max(perf-1,0)) + coupons - cost = 0``:

          * If protection + coupons already cover cost → break-even = 0
            (note is profitable even at S_T = 0).
          * Otherwise the loss has to be made up by upside participation,
            which requires ``perf > 1`` and the call leg to pay enough.
        """
        cost = self.total_cost()
        floor = self.notional * self.protection_pct + self.coupon_payment()
        if floor >= cost:
            return 0.0
        if self.participation_pct <= 0:
            return np.inf
        return 1.0 + (cost - floor) / (self.notional * self.participation_pct)

    def is_multi(self) -> bool:
        return False

    def worst_underlying(self) -> str:
        return self.underlying

    # ----- RC-compatible interface for shared analytics ---------------------
    # Vanilla CPNs have no barrier, so distance metrics are not meaningful;
    # we return NaN so the same product_df schema can carry both products.

    def current_barrier_distances(self) -> list:
        return [float("nan")]

    def distance_to_barrier(self) -> float:
        return float("nan")

    def cash_flows(self, as_of=None):
        """Dated cash flows from ``as_of`` (default purchase_date) to maturity.

        Convention matches :meth:`ReverseConvertible.cash_flows` so the
        same XIRR routine can be reused.
        """
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(self._purchase_date())
        flows = [(as_of, -self.total_cost())]
        flows.extend(
            self.schedule.future_cashflows(self.notional, self.coupon, as_of)
        )
        # Redemption at maturity uses the same payoff formula as redemption(),
        # evaluated against current_spot (consistent with RC's flat-spot proj.).
        flows.append((pd.Timestamp(self.row["maturity_date"]), self.redemption()))
        return flows

    @staticmethod
    def _xirr(cash_flows):
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

    def ytm(self) -> float:
        return self._xirr(self.cash_flows())

    def ytm_today(self) -> float:
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

    def return_pa(self) -> float:
        days = (datetime.strptime(self.row["maturity_date"], "%Y-%m-%d")
                - self._purchase_date()).days
        if days <= 0:
            return np.nan
        return self.return_pct() * 360 / days

    # ----- analytic pricing (delegates to module-level helpers) --------------

    def analytic_price(self, sigma: float, r: float, q: float = 0.0,
                       as_of: Optional[pd.Timestamp] = None) -> float:
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
        maturity = pd.Timestamp(self.row["maturity_date"])
        T = max((maturity - as_of).days / 360.0, 0.0)  # ACT/360

        # PV of remaining coupons (assume flat rate r, continuous discounting).
        coupon_pv = 0.0
        for date, amount in self.schedule.future_cashflows(
            self.notional, self.coupon, as_of
        ):
            t = max((date - as_of).days / 360.0, 0.0)  # ACT/360
            coupon_pv += amount * np.exp(-r * t)

        return analytic_cpn_price(
            S=self.current_spot, K=self.strike, T=T, r=r, sigma=sigma,
            notional=self.notional,
            protection_pct=self.protection_pct,
            participation_pct=self.participation_pct,
            q=q, coupon_pv=coupon_pv,
        )

    def analytic_greeks(self, sigma: float, r: float, q: float = 0.0,
                        as_of: Optional[pd.Timestamp] = None) -> dict:
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
        maturity = pd.Timestamp(self.row["maturity_date"])
        T = max((maturity - as_of).days / 360.0, 1e-9)  # ACT/360
        return analytic_cpn_greeks(
            S=self.current_spot, K=self.strike, T=T, r=r, sigma=sigma,
            notional=self.notional,
            protection_pct=self.protection_pct,
            participation_pct=self.participation_pct,
            q=q,
        )

    # ----- summary (mirrors RC.summary keys where meaningful) ----------------

    def summary(self) -> dict:
        maturity = datetime.strptime(self.row["maturity_date"], "%Y-%m-%d")
        days_to_expiry = (maturity - datetime.today()).days
        return {
            "product_type":        self.product_type,
            "is_multi":            False,
            "issuer":              self.issuer,
            "issuer_rating":       self.issuer_rating,
            "issuer_country":      self.issuer_country,
            "denomination":        self.denomination,
            "position_units":      self.position_units,
            "total_notional":      self.total_notional,
            "maturity_date":       self.row["maturity_date"],
            "days_to_expiry":      days_to_expiry,
            "performance":         self.performance() - 1.0,
            "protection_pct":      self.protection_pct,
            "participation_pct":   self.participation_pct,
            "barrier_breached":    False,
            "worst_underlying":    self.underlying,
            "payoff_per_unit":     self.payoff() / self.position_units,
            "total_payoff":        self.total_payoff(),
            "total_cost":          self.total_cost(),
            "pnl":                 self.pnl(),
            "return_pct":          self.return_pct(),
            "ytm":                 self.ytm(),
            "ytm_today":           self.ytm_today(),
            "return_pa":           self.return_pa(),
            "distance_to_barrier": float("nan"),
            "break_even":          self.break_even(),
            # RC tracks yesterday-vs-today deltas via price_db.  CPN's only
            # spot-sensitive piece is the optional upside leg; for now we
            # leave the deltas None (consistent with RC when no history).
            "pnl_delta":           None,
            "distance_delta":      None,

            # ----- Term-sheet disclosures (optional) -----
            "guarantor":                  self.guarantor,
            "guarantor_rating":           self.guarantor_rating,
            "keep_well_agreement":        self.keep_well_agreement,
            "issue_price":                self.issue_price,
            "capital_protection_amount":  self.capital_protection_amount,
            "number_of_underlyings":      self.number_of_underlyings,
            "spot_reference_price":       self.spot_reference_price,
            "issue_size":                 self.issue_size,
            "last_trading_day":           self.last_trading_day,
            "payment_date":               self.payment_date,
            "final_fixing_date":          self.final_fixing_date,
            "bond_npv_at_issue":          self.bond_npv_at_issue,
            "implied_irr_at_issue":       self.implied_irr_at_issue,
        }
