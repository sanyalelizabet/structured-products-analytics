"""American (continuous) barrier monitoring in the fair-value pricer.

A continuously-monitored down-barrier can only knock in *more* often than one
checked at maturity, so for a barrier reverse convertible with downside risk the
American fair value must sit at or below the European one — and the two must
coincide when the barrier is effectively unreachable.  Combining continuous
monitoring with an autocallable is not implemented and must fail loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pricing.monte_carlo import MonteCarloPricer, european_brc_payoff


_TODAY = pd.Timestamp.today().normalize()
_VOLS  = {"CH0012221716": 0.30}
_R     = 0.0


def _brc_row(*, style: str, barrier_pct: float, product_type: str = "BRC"):
    return pd.Series({
        "product_id":          "BRC_TEST",
        "product_type":        product_type,
        "type_style":          style,                 # "european" | "american"
        "currency":            "CHF",
        "notional":            100_000.0,
        "position_units":      1,
        "cost_price":          1.0,
        "coupon":              0.05,
        "barrier_pct":         barrier_pct,
        "underlyings":         ["NESN"],
        "underlying_isins":    ["CH0012221716"],
        "initial_levels":      [100.0],
        "strike":              [100.0],
        "current_spots":       [100.0],
        "initial_fixing_date": (_TODAY - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "maturity_date":       (_TODAY + pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
        # Autocall fields (only read for AC_BRC):
        "autocall_obs_dates":  [(_TODAY + pd.Timedelta(days=120)).strftime("%Y-%m-%d")],
        "autocall_trigger_pct": 1.0,
        "autocall_coupon_memory": False,
    })


def _fv(row):
    pricer = MonteCarloPricer(n_paths=6_000, seed=11)
    return pricer.price(row, european_brc_payoff, _VOLS, _R)["fair_value"]


class TestAmericanBarrierFairValue:
    def test_american_not_more_valuable_than_european(self):
        """Continuous monitoring finds at least as many breaches → fair value
        cannot exceed the European note, and is strictly lower when the barrier
        is genuinely at risk."""
        bp = 0.70   # barrier at 70 with spot/strike 100 and 30% vol → real risk
        eur = _fv(_brc_row(style="european", barrier_pct=bp))
        am  = _fv(_brc_row(style="american", barrier_pct=bp))
        assert am <= eur + 1e-6
        assert am < eur * 0.999          # materially lower, not a rounding tie

    def test_unreachable_barrier_matches_european(self):
        """A barrier the path cannot reach is never knocked in under either
        convention, so the two valuations coincide."""
        bp = 0.01   # barrier at 1.0 — unreachable
        eur = _fv(_brc_row(style="european", barrier_pct=bp))
        am  = _fv(_brc_row(style="american", barrier_pct=bp))
        assert am == pytest.approx(eur, rel=1e-3)

    def test_american_autocallable_prices_and_is_conservative(self):
        """Continuous barrier + autocall is now supported: the uncalled paths
        observe the barrier continuously, so the American autocallable prices
        without error and is worth no more than its European-observed twin."""
        am = _brc_row(style="american", barrier_pct=0.70, product_type="AC_BRC")
        eu = _brc_row(style="european", barrier_pct=0.70, product_type="AC_BRC")
        fv_am, fv_eu = _fv(am), _fv(eu)
        assert np.isfinite(fv_am)
        assert fv_am <= fv_eu + 1e-6
