"""Fair-value Monte Carlo must price autocallables path-dependently.

Regression guard for a production gap: the fair-value pricer used to route
``AC_BRC`` through the plain European BRC payoff, ignoring the autocall feature
entirely. These tests pin the corrected behaviour — early redemption is modelled
and each path is discounted to its own cashflow date — and check the two
limiting cases against an equivalent European note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pricing.monte_carlo import MonteCarloPricer, european_brc_payoff


_TODAY = pd.Timestamp.today().normalize()


def _ac_row(*, trigger: float):
    """Single-underlying autocallable with observation dates in the future."""
    fixing   = _TODAY - pd.Timedelta(days=60)
    obs1     = (_TODAY + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    obs2     = (_TODAY + pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    maturity = (_TODAY + pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    return pd.Series({
        "product_id":           "AC_TEST",
        "product_type":         "AC_BRC",
        "type_style":           "european",
        "currency":             "CHF",
        "notional":             100_000.0,
        "position_units":       1,
        "cost_price":           1.0,
        "coupon":               0.08,
        "barrier_pct":          0.60,
        "underlyings":          ["NESN"],
        "underlying_isins":     ["CH0012221716"],
        "initial_levels":       [100.0],
        "strike":               [100.0],
        "current_spots":        [100.0],
        "initial_fixing_date":  fixing.strftime("%Y-%m-%d"),
        "maturity_date":        maturity,
        "autocall_obs_dates":   [obs1, obs2],
        "autocall_trigger_pct": trigger,
        "autocall_coupon_memory": False,
    })


_VOLS  = {"CH0012221716": 0.10}
_RATES = {"CHF": 0.0}          # r = 0 keeps the assertions about coupon crisp


def _price(row):
    pricer = MonteCarloPricer(n_paths=4_000, seed=7)
    # payoff_fn is ignored for AC_BRC, but the API requires one.
    return pricer.price(row, european_brc_payoff, _VOLS, _RATES["CHF"])


class TestAutocallableFairValue:
    def test_near_certain_call_truncates_coupon(self):
        """A deep-in-the-money trigger calls almost immediately, so the note pays
        roughly par plus a little accrued coupon — strictly less than the
        full-life European payoff of the same note."""
        ac_row = _ac_row(trigger=0.50)          # worst-of ≥ 50% of strike → called
        ac_fv  = _price(ac_row)["fair_value"]

        eur_row = ac_row.copy()
        eur_row["product_type"] = "BRC"         # same terms, no autocall
        eur_fv  = _price(eur_row)["fair_value"]

        notional = float(ac_row["notional"])
        # Called at the first observation (~45d): par + ~0.105y of 8% coupon.
        assert notional <= ac_fv < notional * 1.05
        # Autocall truncates the coupon stream → cheaper than the European note.
        assert ac_fv < eur_fv

    def test_unreachable_trigger_matches_european(self):
        """If the trigger can never be hit, no path is called and the
        autocallable must settle exactly like the European note it embeds."""
        ac_row = _ac_row(trigger=5.0)           # worst-of ≥ 500% → never called
        ac_fv  = _price(ac_row)["fair_value"]

        eur_row = ac_row.copy()
        eur_row["product_type"] = "BRC"
        eur_fv  = _price(eur_row)["fair_value"]

        assert ac_fv == pytest.approx(eur_fv, rel=1e-6)

    def test_fair_value_is_finite_and_priced(self):
        result = _price(_ac_row(trigger=1.0))
        assert np.isfinite(result["fair_value"])
        assert np.isfinite(result["fair_value_pct"])
        assert result["fair_value"] > 0
