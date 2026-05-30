"""Issuer-Callable Barrier Reverse Convertible — vectorised, optimal-exercise.

An ``IC_BRC`` is a worst-of reverse convertible (coupons + a down-and-in barrier
on the worst-performing share) whose *issuer* holds the right — but not the
obligation — to redeem the note at par on a set of optional redemption dates.

The issuer exercises that right to *minimise the holder's value*: it calls when
the note's continuation value (what it is worth if left alive) exceeds the call
payoff (par + the coupon accrued to the call date).  This is an optimal-stopping
problem, valued here by **Longstaff–Schwartz**: at each call date the
continuation value is estimated by regressing the realised, discounted
continuation payoff on a low-order polynomial of the worst-of level, and the
issuer calls where that estimate exceeds the call payoff.

This differs fundamentally from the autocallable (``AC_BRC``): the autocall is a
mechanical redemption triggered by the underlyings being *high* (it helps the
holder), whereas the issuer call is an *optimal* redemption that *caps* the
holder's value (it reduces it).

The barrier knock-in is supplied by the caller as ``breach_mask`` (a definite
per-path indicator) — continuously monitored for American observation, terminal
for European — because the optimal-exercise regression operates on realised
per-path cashflows.

Returns the same dict keys as :func:`vectorised_european_rc_summary` plus
``issuer_called`` and ``call_date``.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.reverse_convertible import (
    ReverseConvertible,
    barrier_levels,
    vectorised_european_rc_summary,
)


def vectorised_issuer_callable_rc_summary(
    row,
    price_paths: np.ndarray,
    date_range: pd.DatetimeIndex,
    risk_free_rate: float,
    breach_mask: np.ndarray | None = None,
) -> dict:
    """Vectorised payoff for an issuer-callable barrier reverse convertible.

    Parameters
    ----------
    row : pd.Series
        Portfolio row.  Standard worst-of BRC fields plus
        ``issuer_call_dates`` — the optional redemption dates (ISO strings).
    price_paths : np.ndarray
        Asset price paths, shape ``(n_paths, n_days, n_assets)``.
    date_range : pd.DatetimeIndex
        Business-day grid for the ``n_days`` axis of ``price_paths``.
    risk_free_rate : float
        Annualised rate used to discount continuation values in the optimal
        exercise (the issuer's economic decision is a risk-neutral comparison).
    breach_mask : np.ndarray of bool, optional
        Per-path knock-in indicator.  When ``None`` the barrier is observed at
        final fixing (European); for continuous (American) observation the
        caller supplies a sampled mask.

    Returns
    -------
    dict
        Same keys as :func:`vectorised_european_rc_summary` plus
        ``issuer_called`` (bool array) and ``call_date`` (Timestamp or None).
    """
    price_paths = np.asarray(price_paths, dtype=float)
    N, n_days, n_assets = price_paths.shape

    notional = float(row["notional"])
    coupon   = float(row["coupon"])
    strikes  = np.array([float(k) for k in row["strike"]])

    initial_fixing = datetime.strptime(row["initial_fixing_date"], "%Y-%m-%d")
    maturity       = pd.Timestamp(row["maturity_date"])

    rc_proto   = ReverseConvertible(row, final_levels=[0.0] * n_assets)
    total_cost = rc_proto.total_cost()

    # ── Maturity (uncalled) settlement, via the worst-of European summary with
    #    the supplied knock-in mask.  This seeds the backward induction. ──────
    mat_mask    = np.asarray(date_range >= maturity)
    t_idx_term  = int(np.argmax(mat_mask)) if mat_mask.any() else n_days - 1
    terminal    = price_paths[:, t_idx_term, :]
    mat_summary = vectorised_european_rc_summary(row, terminal, breach_mask=breach_mask)

    # Per-path running termination state (starts at maturity for every path).
    payoff_term = np.asarray(mat_summary["total_payoff"], dtype=float).copy()
    term_date   = np.full(N, np.datetime64(maturity.to_datetime64(), "D"))

    def _accrued_coupon(t: pd.Timestamp) -> float:
        years = max((pd.Timestamp(t).to_pydatetime() - initial_fixing).days / 360.0, 0.0)
        return notional * coupon * years

    # ── Resolve call dates onto the grid (drop any outside the horizon). ─────
    raw_calls = list(row.get("issuer_call_dates", []) or [])
    call_dates = pd.to_datetime(raw_calls) if raw_calls else pd.DatetimeIndex([])
    call_idx = []
    for cd in call_dates:
        if cd < date_range[0] or cd > date_range[-1]:
            continue
        call_idx.append(int(np.argmin(np.abs(date_range - cd))))
    call_idx = sorted(set(call_idx))

    # ── Backward induction over call dates (latest → earliest). ──────────────
    for idx in reversed(call_idx):
        t_k   = date_range[idx]
        t_k64 = np.datetime64(pd.Timestamp(t_k).to_datetime64(), "D")

        # Discount each path's current termination payoff back to t_k.
        days  = (term_date - t_k64) / np.timedelta64(1, "D")
        tau   = np.maximum(days.astype(float) / 360.0, 0.0)
        pv_cont = payoff_term * np.exp(-risk_free_rate * tau)

        # Continuation value estimate: regress pv_cont on a low-order polynomial
        # of the worst-of performance at t_k (Longstaff–Schwartz).
        w = (price_paths[:, idx, :] / strikes[None, :]).min(axis=1)
        X = np.column_stack([np.ones_like(w), w, w ** 2])
        coeffs, *_ = np.linalg.lstsq(X, pv_cont, rcond=None)
        cont_hat = X @ coeffs

        # Issuer calls when continuing is dearer than calling at par + accrued.
        v_call = notional + _accrued_coupon(t_k)
        call   = cont_hat > v_call

        payoff_term[call] = v_call
        term_date[call]   = t_k64

    # ── Assemble outputs. ────────────────────────────────────────────────────
    maturity64    = np.datetime64(maturity.to_datetime64(), "D")
    issuer_called = term_date < maturity64
    call_date     = np.empty(N, dtype=object)
    call_date.fill(None)

    pnl              = np.zeros(N)
    return_pct       = np.zeros(N)
    total_payoff     = payoff_term.copy()
    cash_redemption  = np.zeros(N)
    worst_underlying = np.array(mat_summary["worst_underlying"], dtype=object).copy()
    settlement_type  = np.array(mat_summary["settlement_type"], dtype=object).copy()
    delivered_under  = np.array(mat_summary["delivered_underlying"], dtype=object).copy()
    delivered_shares = np.asarray(mat_summary["delivered_shares"]).copy()
    fractional_cash  = np.asarray(mat_summary["fractional_cash"], dtype=float).copy()
    barrier_breached = np.asarray(mat_summary["barrier_breached"], dtype=bool).copy()
    strike_used      = np.asarray(mat_summary["strike_used"], dtype=float).copy()
    final_spot_used  = np.asarray(mat_summary["final_spot_used"], dtype=float).copy()

    # Called paths: cash settlement at par + accrued coupon; no delivery, no breach.
    for p in np.flatnonzero(issuer_called):
        cd = pd.Timestamp(term_date[p])
        call_date[p]        = cd
        cash_redemption[p]  = total_payoff[p]
        settlement_type[p]  = "issuer_called"
        # A called path redeems at par in cash — there is no delivery and no
        # breach.  ``worst_underlying`` is left at the terminal-worst label
        # (never None) so downstream aggregations that sort underlying names
        # are not broken by a missing value.
        delivered_under[p]  = None
        delivered_shares[p] = 0
        fractional_cash[p]  = 0.0
        barrier_breached[p] = False

    # Uncalled paths keep the maturity settlement's cash redemption.
    uncalled = ~issuer_called
    cash_redemption[uncalled] = np.asarray(mat_summary["cash_redemption"], dtype=float)[uncalled]

    pnl        = total_payoff - total_cost
    return_pct = (pnl / total_cost) if total_cost else np.full(N, np.nan)

    return {
        "pnl":                  pnl,
        "return_pct":           return_pct,
        "total_payoff":         total_payoff,
        "barrier_breached":     barrier_breached,
        "worst_underlying":     worst_underlying,
        "strike_used":          strike_used,
        "final_spot_used":      final_spot_used,
        "settlement_type":      settlement_type,
        "delivered_underlying": delivered_under,
        "delivered_shares":     delivered_shares,
        "fractional_cash":      fractional_cash,
        "cash_redemption":      cash_redemption,
        "total_cost":           float(total_cost),
        # Issuer-callable extras
        "issuer_called":        issuer_called,
        "call_date":            call_date,
    }
