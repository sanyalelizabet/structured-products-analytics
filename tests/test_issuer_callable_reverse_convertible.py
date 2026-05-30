"""Tests for the issuer-callable reverse convertible (IC_BRC).

The defining property: the issuer redeems optimally to *minimise* the holder's
value, so an IC_BRC is worth no more than the identical note without the call,
and strictly less when the call is ever in the issuer's interest.  With no call
dates it must coincide with the plain worst-of note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pricing.monte_carlo import MonteCarloPricer, european_brc_payoff
from src.issuer_callable_reverse_convertible import (
    vectorised_issuer_callable_rc_summary,
)


_T = pd.Timestamp.today().normalize()
_VOLS = {"GB00BRXH2664": 0.35, "CA06849F1080": 0.40, "US6516391066": 0.32}


def _row(*, product_type, call_dates=None, style="American"):
    r = {
        "product_id":          "XS3338647111",
        "product_type":        product_type,
        "type_style":          style,
        "currency":            "USD",
        "notional":            1000.0,
        "position_units":      1,
        "cost_price":          1.0,
        "underlyings":         ["AngloGold", "Barrick", "Newmont"],
        "underlying_isins":    ["GB00BRXH2664", "CA06849F1080", "US6516391066"],
        "initial_levels":      [93.47, 40.60, 109.85],
        "current_spots":       [93.47, 40.60, 109.85],
        "strike":              [93.47, 40.60, 109.85],
        "barrier_pct":         0.52,
        "coupon":              0.155,
        "initial_fixing_date": "2026-02-27",
        "maturity_date":       (_T + pd.Timedelta(days=360)).strftime("%Y-%m-%d"),
        "barrier_breached":    False,
    }
    if call_dates is not None:
        r["issuer_call_dates"] = call_dates
    return pd.Series(r)


def _fv(row):
    return MonteCarloPricer(n_paths=4000, seed=3).price(
        row, european_brc_payoff, _VOLS, 0.03,
    )["fair_value"]


class TestIssuerCallableFairValue:
    def test_issuer_call_reduces_value_vs_no_call(self):
        calls = [(_T + pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
                 (_T + pd.Timedelta(days=240)).strftime("%Y-%m-%d")]
        ic = _fv(_row(product_type="IC_BRC", call_dates=calls))
        mb = _fv(_row(product_type="MBRC"))           # identical, no call
        assert np.isfinite(ic)
        assert ic <= mb + 1e-6
        assert ic < mb * 0.999                        # materially cheaper

    def test_no_call_dates_matches_worst_of(self):
        # With no call dates the issuer optionality vanishes, so the value must
        # match the plain worst-of note up to Monte-Carlo error.  (It is not bit
        # identical: the IC_BRC samples its American knock-in per path, as the
        # optimal-exercise regression needs realised cashflows, whereas the plain
        # MBRC uses the lower-variance survival-probability expectation.)
        ic = _fv(_row(product_type="IC_BRC", call_dates=[]))
        mb = _fv(_row(product_type="MBRC"))
        assert ic == pytest.approx(mb, rel=2e-3)

    def test_reproducible(self):
        calls = [(_T + pd.Timedelta(days=180)).strftime("%Y-%m-%d")]
        a = _fv(_row(product_type="IC_BRC", call_dates=calls))
        b = _fv(_row(product_type="IC_BRC", call_dates=calls))
        assert a == b


class TestIssuerCallableSummary:
    def test_called_paths_redeem_at_par_plus_accrued(self):
        # Deep paths above strike everywhere → issuer should call (note is dear);
        # called payoff is par + accrued coupon, settlement flagged issuer_called.
        row = _row(product_type="IC_BRC",
                   call_dates=[(_T + pd.Timedelta(days=120)).strftime("%Y-%m-%d")])
        dates = pd.bdate_range(_T, pd.Timestamp(row["maturity_date"]))
        n_days = len(dates)
        # Flat paths well above strike (never knock in, high continuation value).
        spots = np.array(row["current_spots"]) * 1.5
        paths = np.tile(spots, (200, n_days, 1))
        breach = np.zeros(200, dtype=bool)
        out = vectorised_issuer_callable_rc_summary(row, paths, dates, 0.03,
                                                    breach_mask=breach)
        assert out["issuer_called"].any()
        called = out["issuer_called"]
        assert (out["settlement_type"][called] == "issuer_called").all()
        # Par (1000) ≤ called payoff ≤ par + full coupon
        assert np.all(out["total_payoff"][called] >= 1000.0 - 1e-6)
        assert np.all(out["total_payoff"][called] <= 1000.0 * (1 + 0.155) + 1e-6)
