"""Autocallable Barrier Reverse Convertible — vectorised path-dependent payoff.

Phoenix-without-memory autocallable BRC:

* At each observation date, if the worst-of ratio to strike is at or above
  the autocall trigger, the product is called early at par + prorata coupon.
* Otherwise the path settles at maturity via the European barrier BRC payoff
  (:func:`vectorised_european_rc_summary`).

Returns the same dict keys as the BRC helper, plus ``autocalled`` and
``call_date``.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.pricing.products.reverse_convertible import (
    ReverseConvertible,
    vectorised_european_rc_summary,
)


def vectorised_autocallable_rc_summary(
    row,
    price_paths: np.ndarray,
    date_range: pd.DatetimeIndex,
    uncalled_breach_mask: np.ndarray | None = None,
) -> dict:
    """Vectorised payoff for an autocallable barrier reverse convertible.

    Parameters
    ----------
    row : pd.Series
        Portfolio row.  Must carry the standard BRC fields plus:

        * ``autocall_obs_dates``      — list of ISO-format dates
        * ``autocall_trigger_pct``    — float, ratio against **strike**
          (1.00 = call when worst-of ≥ strike).  Optional, default 1.0.
        * ``autocall_coupon_memory``  — bool, default False.  Reserved
          for future Phoenix-with-memory implementation; not used here.

    price_paths : np.ndarray
        Asset price paths, shape ``(n_paths, n_days, n_assets)``.

    date_range : pd.DatetimeIndex
        Business-day grid corresponding to the ``n_days`` axis of
        ``price_paths``.

    uncalled_breach_mask : np.ndarray of bool, optional
        Per-path knock-in indicator, shape ``(n_paths,)``, for **American
        (continuous)** barrier observation.  An autocalled path redeems at par
        regardless of the barrier, so the mask only affects paths that run to
        maturity: when supplied, those paths settle with this continuously
        determined breach instead of the European final-fixing check.  When
        ``None`` the uncalled paths use European observation.

    Returns
    -------
    dict
        Same keys as :func:`vectorised_european_rc_summary` plus:

        * ``autocalled``     — boolean array (n_paths,)
        * ``call_date``      — object array of ``pd.Timestamp`` or ``None``
        * ``settlement_type`` is ``"autocalled"`` for called paths
    """
    price_paths = np.asarray(price_paths, dtype=float)
    N, n_days, n_assets = price_paths.shape

    notional    = float(row["notional"])
    strikes     = np.array([float(k) for k in row["strike"]])  # (n_assets,)
    underlyings = list(row["underlyings"])
    coupon      = float(row["coupon"])

    trigger     = float(row.get("autocall_trigger_pct", 1.0))
    raw_obs     = list(row.get("autocall_obs_dates", []) or [])
    obs_dates   = pd.to_datetime(raw_obs) if raw_obs else pd.DatetimeIndex([])

    # Constant — total dirty cost is identical for every path, so compute it
    # once via a prototype RC instance. Per-path coupon income is computed
    # inline below: pro-rata to the call date for autocalled paths, and via
    # `vectorised_european_rc_summary` for uncalled paths.
    rc_proto       = ReverseConvertible(row, final_levels=[0.0] * n_assets)
    total_cost     = rc_proto.total_cost()

    initial_fixing = datetime.strptime(row["initial_fixing_date"], "%Y-%m-%d")

    # ── Walk observation dates, find the first call per path ────────────
    called_at_idx  = np.full(N, -1, dtype=np.int64)            # index in date_range
    call_date      = np.empty(N, dtype=object)
    call_date.fill(None)

    for obs_d in obs_dates:
        # Snap to nearest business day inside the simulation grid; drop
        # any observation past the grid (post-maturity / outside horizon).
        if obs_d > date_range[-1] or obs_d < date_range[0]:
            continue
        idx = int(np.argmin(np.abs(date_range - obs_d)))

        spots_at_obs = price_paths[:, idx, :]                  # (N, n_assets)
        worst_perf   = (spots_at_obs / strikes[None, :]).min(axis=1)

        not_yet_called = called_at_idx < 0
        triggered      = (worst_perf >= trigger) & not_yet_called

        called_at_idx[triggered] = idx
        for p in np.flatnonzero(triggered):
            call_date[p] = pd.Timestamp(obs_d)

    autocalled = called_at_idx >= 0

    # ── Called paths: par + prorata coupon to the call date ─────────────
    pnl              = np.zeros(N)
    return_pct       = np.zeros(N)
    total_payoff     = np.zeros(N)
    cash_redemption  = np.zeros(N)
    worst_underlying = np.empty(N, dtype=object)
    settlement_type  = np.empty(N, dtype=object)
    delivered_under  = np.empty(N, dtype=object)
    delivered_shares = np.zeros(N, dtype=int)
    fractional_cash  = np.zeros(N)
    barrier_breached = np.zeros(N, dtype=bool)
    strike_used      = np.zeros(N)
    final_spot_used  = np.zeros(N)

    if autocalled.any():
        for p in np.flatnonzero(autocalled):
            t_call_years = rc_proto.schedule.year_fraction(initial_fixing, call_date[p])
            accrued = notional * coupon * max(t_call_years, 0.0)
            payoff_p = notional + accrued

            spots_at_call = price_paths[p, called_at_idx[p], :]
            perfs_at_call = spots_at_call / strikes
            worst_idx_p   = int(np.argmin(perfs_at_call))

            total_payoff[p]     = payoff_p
            pnl[p]              = payoff_p - total_cost
            return_pct[p]       = pnl[p] / total_cost if total_cost else 0.0
            cash_redemption[p]  = payoff_p
            settlement_type[p]  = "autocalled"
            worst_underlying[p] = underlyings[worst_idx_p]
            strike_used[p]      = strikes[worst_idx_p]
            final_spot_used[p]  = spots_at_call[worst_idx_p]
            # Autocalled paths never breach barrier (worst-of ≥ strike,
            # which is by construction ≥ barrier).
            barrier_breached[p] = False
            # Autocalled = cash settlement at par; no physical delivery.
            delivered_under[p]  = None
            delivered_shares[p] = 0
            fractional_cash[p]  = 0.0

    # ── Uncalled paths fall through to the existing European-barrier
    #     payoff at maturity. ────────────────────────────────────────────
    if (~autocalled).any():
        maturity = pd.Timestamp(row["maturity_date"])
        mat_mask = np.asarray(date_range >= maturity)
        if mat_mask.any():
            t_idx_terminal = int(np.argmax(mat_mask))
        else:
            t_idx_terminal = n_days - 1

        # Run the BRC helper on the terminal slice of *uncalled* paths.
        uncalled_idx = np.flatnonzero(~autocalled)
        terminal_uncalled = price_paths[uncalled_idx, t_idx_terminal, :]

        # American observation: the uncalled paths carry a continuously
        # determined knock-in (computed upstream over the full path); European
        # observation falls back to the terminal-fixing check.
        if uncalled_breach_mask is not None:
            um = np.asarray(uncalled_breach_mask, dtype=bool)[uncalled_idx]
            v = vectorised_european_rc_summary(row, terminal_uncalled, breach_mask=um)
        else:
            v = vectorised_european_rc_summary(row, terminal_uncalled)

        pnl[uncalled_idx]              = v["pnl"]
        return_pct[uncalled_idx]       = v["return_pct"]
        total_payoff[uncalled_idx]     = v["total_payoff"]
        cash_redemption[uncalled_idx]  = v["cash_redemption"]
        worst_underlying[uncalled_idx] = v["worst_underlying"]
        settlement_type[uncalled_idx]  = v["settlement_type"]
        delivered_under[uncalled_idx]  = v["delivered_underlying"]
        delivered_shares[uncalled_idx] = v["delivered_shares"]
        fractional_cash[uncalled_idx]  = v["fractional_cash"]
        barrier_breached[uncalled_idx] = v["barrier_breached"]
        strike_used[uncalled_idx]      = v["strike_used"]
        final_spot_used[uncalled_idx]  = v["final_spot_used"]

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
        # Autocallable-specific extras
        "autocalled":           autocalled,
        "call_date":            call_date,
    }
