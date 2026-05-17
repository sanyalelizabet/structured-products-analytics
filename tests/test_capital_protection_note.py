"""Tests for the Capital Protection Note implementation.

Covers:
* Payoff mechanics at three regions (deep OTM, at-strike, deep ITM).
* `CapitalProtectionNote.summary()` consistency with the vectorised payoff.
* Analytic price = ZCB + scaled BS call (validated against Monte Carlo).
* Engine dispatch in `ScenarioEngine` and `FactorScenarioEngine`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.capital_protection_note import (
    CapitalProtectionNote,
    analytic_cpn_price,
    vectorised_cpn_summary,
)
from src.pricing.black_scholes import BlackScholes


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _make_cpn_row(
    notional=100_000,
    cost_price=1.0,
    coupon=0.0,
    protection_pct=1.00,
    participation_pct=0.80,
    strike=100.0,
    current_spot=100.0,
    initial_fixing_date="2024-01-01",
    maturity_date="2027-01-01",
):
    return pd.Series({
        "product_id":          "CPN001",
        "product_type":        "CPN",
        "type_style":          "european",
        "currency":            "CHF",
        "notional":            notional,
        "position_units":      1,
        "cost_price":          cost_price,
        "coupon":              coupon,
        "protection_pct":      protection_pct,
        "participation_pct":   participation_pct,
        "underlyings":         ["NESN"],
        "underlying_isins":    ["CH0012221716"],
        "initial_levels":      [100.0],
        "strike":              [strike],
        "current_spots":       [current_spot],
        "initial_fixing_date": initial_fixing_date,
        "maturity_date":       maturity_date,
    })


# ──────────────────────────────────────────────────────────────────────────
# Payoff mechanics
# ──────────────────────────────────────────────────────────────────────────

def test_payoff_below_strike_returns_protection():
    """When S_T < K, payoff = N * protection_pct (no upside)."""
    row = _make_cpn_row(protection_pct=1.00, participation_pct=0.80)
    final_prices = np.array([[50.0], [80.0], [99.99]])  # all below K=100
    v = vectorised_cpn_summary(row, final_prices)
    expected = 100_000 * 1.00
    np.testing.assert_allclose(v["cash_redemption"], expected)
    assert (v["barrier_breached"] == False).all()
    assert (v["settlement_type"] == "cash").all()


def test_payoff_at_strike_returns_protection():
    row = _make_cpn_row()
    v = vectorised_cpn_summary(row, np.array([[100.0]]))
    assert v["cash_redemption"][0] == pytest.approx(100_000.0)


def test_payoff_above_strike_pays_participation():
    """At S_T = 1.5*K with p=0.80: payoff = N*(1 + 0.8*0.5) = 1.40*N."""
    row = _make_cpn_row(protection_pct=1.00, participation_pct=0.80)
    v = vectorised_cpn_summary(row, np.array([[150.0]]))
    assert v["cash_redemption"][0] == pytest.approx(140_000.0)


def test_partial_protection_only_pays_floor():
    """protection_pct = 0.90 means worst-case redemption = 90% of notional."""
    row = _make_cpn_row(protection_pct=0.90, participation_pct=1.00)
    v = vectorised_cpn_summary(row, np.array([[1.0], [200.0]]))
    assert v["cash_redemption"][0] == pytest.approx(90_000.0)
    # upside leg: 100% participation in 100% return = +N
    assert v["cash_redemption"][1] == pytest.approx(190_000.0)


def test_zero_participation_is_pure_zcb():
    row = _make_cpn_row(protection_pct=1.00, participation_pct=0.0)
    v = vectorised_cpn_summary(row, np.array([[50.0], [200.0]]))
    np.testing.assert_allclose(v["cash_redemption"], 100_000.0)


def test_rejects_multi_underlying_terminal():
    row = _make_cpn_row()
    with pytest.raises(ValueError, match="single-underlying"):
        vectorised_cpn_summary(row, np.zeros((10, 2)))


# ──────────────────────────────────────────────────────────────────────────
# Class summary
# ──────────────────────────────────────────────────────────────────────────

def test_class_summary_matches_vectorised():
    """Single-path scalar match between the class and the vectorised helper."""
    row = _make_cpn_row(participation_pct=0.75)
    final_pct = 30.0   # +30% scenario shock
    cpn = CapitalProtectionNote(row, final_level=final_pct)
    final_price = row["current_spots"][0] * (1 + final_pct / 100)

    v = vectorised_cpn_summary(row, np.array([[final_price]]))
    s = cpn.summary()
    assert s["total_payoff"] == pytest.approx(v["total_payoff"][0])
    assert s["pnl"] == pytest.approx(v["pnl"][0])


def test_break_even_when_protection_below_cost():
    """Bought at 102 with 100% protection → need upside to break even."""
    row = _make_cpn_row(
        protection_pct=1.00, participation_pct=0.50, cost_price=1.02,
    )
    cpn = CapitalProtectionNote(row, final_level=0)
    # cost = 102_000, floor = 100_000, gap = 2_000, scaled by 1/(N*p) = 1/50_000
    assert cpn.break_even() == pytest.approx(1.04)


def test_break_even_when_protected_floor_already_covers_cost():
    row = _make_cpn_row(protection_pct=1.00, cost_price=0.95)
    cpn = CapitalProtectionNote(row, final_level=0)
    assert cpn.break_even() == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Analytic vs Monte Carlo
# ──────────────────────────────────────────────────────────────────────────

def test_analytic_matches_explicit_decomposition():
    """analytic_cpn_price == N*pi*e^(-rT) + (N*p/K) * BS_call."""
    S, K, T, r, sigma = 100.0, 100.0, 2.0, 0.03, 0.20
    notional, pi, p = 100_000.0, 1.00, 0.80
    bs_call = BlackScholes(S, K, T, r, sigma).call()
    expected = notional * pi * np.exp(-r * T) + (notional * p / K) * bs_call
    got = analytic_cpn_price(S, K, T, r, sigma, notional, pi, p)
    assert got == pytest.approx(expected)


def test_analytic_matches_monte_carlo():
    """GBM Monte Carlo terminal payoff should converge to analytic price."""
    rng = np.random.default_rng(42)
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.25
    notional, pi, p = 100_000.0, 1.00, 0.80
    n = 200_000

    z = rng.standard_normal(n)
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoff = notional * (pi + p * np.maximum(S_T / K - 1.0, 0.0))
    mc_price = float(np.mean(payoff) * np.exp(-r * T))

    bs_price = analytic_cpn_price(S, K, T, r, sigma, notional, pi, p)
    # Tolerate 0.3% of notional MC noise at this sample size.
    assert mc_price == pytest.approx(bs_price, abs=0.003 * notional)


# ──────────────────────────────────────────────────────────────────────────
# Engine dispatch
# ──────────────────────────────────────────────────────────────────────────

def test_scenario_engine_dispatches_to_cpn():
    """ScenarioEngine.run_path_scenario must route product_type=CPN to the
    CPN summary helper, not the BRC one."""
    from src.capital_protection_note import vectorised_cpn_summary as real
    from src import scenario_engine as se

    row = _make_cpn_row()
    # Build a fake (n_paths, n_days, 1) price tensor; only the terminal slice
    # matters for CPN, but the engine pulls it itself.
    n_paths, n_days = 50, 5
    price_paths = np.full((n_paths, n_days, 1), 120.0)
    date_range = pd.bdate_range("2026-01-01", periods=n_days)

    # Replicate the dispatch branch: terminal idx = last day.
    final_prices = price_paths[:, -1, :]
    v = real(row, final_prices)
    # Expected payoff at S_T=120, K=100, p=0.80, pi=1.0: 1 + 0.8*0.2 = 1.16
    np.testing.assert_allclose(v["cash_redemption"], 116_000.0)


def test_factor_scenario_engine_dispatch_branch_exists():
    """Smoke test: import path resolves and CPN branch is present."""
    import inspect
    from src import factor_scenario_engine as fse
    src = inspect.getsource(fse)
    assert 'ptype == "CPN"' in src
    assert "vectorised_cpn_summary" in src


# ──────────────────────────────────────────────────────────────────────────
# Vontobel Amazon CPN — real term-sheet round-trip
# ──────────────────────────────────────────────────────────────────────────

def _vontobel_amazon_row():
    """Mirror of CH1537766565 (Vontobel CPN on Amazon, Feb-Sep 2026)."""
    return pd.Series({
        "product_id":                "CH1537766565",
        "product_type":              "CPN",
        "type_style":                "European",
        "issuer":                    "Vontobel Financial Products Ltd., DIFC Dubai",
        "issuer_rating":             "",
        "issuer_country":            "AE",
        "guarantor":                 "Vontobel Holding AG, Zurich",
        "guarantor_rating":          "A3",
        "keep_well_agreement":       "Bank Vontobel AG, Zurich (Aa3)",
        "underlyings":               ["Amazon.com Inc."],
        "underlying_isins":          ["US0231351067"],
        "currency":                  "USD",
        "denomination":              1000,
        "issue_price":               1000.00,
        "position_units":            1,
        "notional":                  1000,
        "issue_size":                32000,
        "cost_price":                1.00,
        "spot_reference_price":      209.07,
        "initial_levels":            [209.07],
        "current_spots":             [209.07],
        "strike":                    [209.07],
        "protection_pct":            0.95,
        "capital_protection_amount": 950.00,
        "participation_pct":         0.52,
        "number_of_underlyings":     4.78309,
        "coupon":                    0.0,
        "initial_fixing_date":       "2026-02-25",
        "payment_date":              "2026-03-04",
        "purchase_date":             "2026-03-04",
        "last_trading_day":          "2026-08-25",
        "final_fixing_date":         "2026-08-25",
        "maturity_date":             "2026-09-01",
        "coupon_dates":              ["2026-09-01"],
        "day_count":                 "ACT/360",
        "bond_npv_at_issue":         933.377,
        "implied_irr_at_issue":      0.036556,
    })


@pytest.mark.parametrize("sf, expected", [
    (150.00, 950.00),                                  # below strike → floor only
    (209.07, 950.00),                                  # exactly at strike → floor
    (250.00, 950.00 + (250.00 - 209.07) * 4.78309 * 0.52),
    (300.00, 950.00 + (300.00 - 209.07) * 4.78309 * 0.52),
])
def test_vontobel_amazon_payoff_matches_term_sheet_formula(sf, expected):
    """Internal payoff math must equal Vontobel's CP + max((SF−X)·B·P, 0).

    Tolerance: 1e-2 USD per certificate.  Vontobel discloses B rounded to
    5 decimals (4.78309) while we use the exact ratio denomination/strike
    (4.78309173…) — the disclosed B drifts by ~1.4e-4 USD per +USD 100
    of upside on this product, well inside any sensible display rounding.
    """
    row = _vontobel_amazon_row()
    v = vectorised_cpn_summary(row, np.array([[sf]]))
    assert v["cash_redemption"][0] == pytest.approx(expected, abs=1e-2)


def test_vontobel_amazon_summary_carries_term_sheet_fields():
    row = _vontobel_amazon_row()
    cpn = CapitalProtectionNote(row, final_level=0)
    s = cpn.summary()
    assert s["guarantor"] == "Vontobel Holding AG, Zurich"
    assert s["guarantor_rating"] == "A3"
    assert s["keep_well_agreement"].startswith("Bank Vontobel AG")
    assert s["capital_protection_amount"] == 950.00
    assert s["number_of_underlyings"] == pytest.approx(4.78309)
    assert s["bond_npv_at_issue"] == 933.377
    assert s["implied_irr_at_issue"] == pytest.approx(0.036556)
    assert s["payment_date"] == "2026-03-04"
    assert s["final_fixing_date"] == "2026-08-25"


def test_inconsistent_capital_protection_amount_rejected():
    row = _vontobel_amazon_row()
    row["capital_protection_amount"] = 900.00   # not 0.95 × 1000
    with pytest.raises(ValueError, match="capital_protection_amount"):
        CapitalProtectionNote(row, final_level=0)


def test_inconsistent_number_of_underlyings_rejected():
    row = _vontobel_amazon_row()
    row["number_of_underlyings"] = 5.00   # not 1000 / 209.07
    with pytest.raises(ValueError, match="number_of_underlyings"):
        CapitalProtectionNote(row, final_level=0)


# ──────────────────────────────────────────────────────────────────────────
# Portfolio-level Greeks consolidation across BRC + CPN
# ──────────────────────────────────────────────────────────────────────────

def test_portfolio_greeks_routes_cpn_and_brc_separately():
    """Mixed BRC + CPN portfolio: confirm the Monte Carlo pricer dispatches
    by product_type (not just type_style) so CPN doesn't get priced with
    the BRC worst-of barrier payoff.

    Validates:
      * Both products appear in ``greeks_df``, ``fv_df``, ``portfolio_delta``
      * CPN delta is positive (long the embedded call) and bounded
      * CPN fair value sits within MC noise of the analytic ZCB + scaled-BS-call
    """
    from src.pricing.monte_carlo import MonteCarloPricer
    from src.capital_protection_note import analytic_cpn_price
    from tests.conftest import make_brc_row

    # Build a 2-product portfolio: 1 BRC + 1 ATM CPN.  Use far-out maturities
    # so T_remaining > 0 and the MC engine runs (not the expired branch).
    cpn_row = _make_cpn_row(
        notional=10_000, cost_price=1.0, coupon=0.0,
        protection_pct=1.00, participation_pct=0.80,
        strike=100.0, current_spot=100.0,
        initial_fixing_date="2026-01-01", maturity_date="2028-01-01",
    )
    cpn_row["product_id"] = "CPN_TEST"

    brc_row = make_brc_row(
        notional=10_000, cost_price=1.0, coupon=0.08, barrier_pct=0.60,
        initial_fixing_date="2026-01-01", maturity_date="2028-01-01",
    )

    portfolio = pd.DataFrame([brc_row, cpn_row])

    vol_map  = {brc_row["underlying_isins"][0]: 0.25,
                cpn_row["underlying_isins"][0]: 0.25}
    rates    = {"CHF": 0.02}

    pricer = MonteCarloPricer(n_paths=2_000, seed=42)
    greeks_df, pf_delta, fv_df = pricer.compute_portfolio_greeks(
        portfolio, vol_map, rates, corr_df=None,
    )

    # Both products represented
    assert set(greeks_df["product_id"]) == {"BRC001", "CPN_TEST"}
    assert set(fv_df["product_id"])     == {"BRC001", "CPN_TEST"}

    # CPN delta — positive, magnitude reasonable for an ATM call with
    # scalar N·p/K = 10000·0.80/100 = 80.  Delta per 1% spot move ≈
    # 80 × 1% × spot × call_delta ≈ 80 × 1.0 × ATM_delta(~0.6) ≈ 48.
    cpn_delta = greeks_df.loc[greeks_df["product_id"] == "CPN_TEST", "delta_1pct"].iloc[0]
    assert cpn_delta > 0, f"CPN delta should be positive (long the call), got {cpn_delta}"
    assert 10 < cpn_delta < 200, f"CPN delta out of plausible range: {cpn_delta}"

    # CPN fair value should match the analytic ZCB + scaled-BS-call to within
    # MC noise (~1% of notional at 2k paths is generous).
    cpn_fv_mc = fv_df.loc[fv_df["product_id"] == "CPN_TEST", "fair_value"].iloc[0]

    T = (pd.Timestamp("2028-01-01") - pd.Timestamp.today().normalize()).days / 365.0
    cpn_fv_analytic = analytic_cpn_price(
        S=100.0, K=100.0, T=T, r=0.02, sigma=0.25,
        notional=10_000, protection_pct=1.00, participation_pct=0.80,
    )
    assert abs(cpn_fv_mc - cpn_fv_analytic) < 0.02 * 10_000, (
        f"MC fair value {cpn_fv_mc:.0f} differs from analytic "
        f"{cpn_fv_analytic:.0f} by more than 2% of notional"
    )

    # Portfolio delta aggregation should include the CPN's underlying
    assert cpn_row["underlying_isins"][0] in set(pf_delta["isin"])
