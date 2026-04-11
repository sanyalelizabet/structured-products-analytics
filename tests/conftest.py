"""
Shared fixtures for all tests.
"""
import pandas as pd
import pytest


def make_brc_row(
    notional=100_000,
    cost_price=1.0,
    coupon=0.08,
    barrier_pct=0.60,
    initial_level=100.0,
    strike=100.0,
    current_spot=95.0,
    initial_fixing_date="2024-01-01",
    maturity_date="2027-06-01",
):
    return pd.Series({
        "product_id": "BRC001",
        "product_type": "BRC",
        "type_style": "european",
        "currency": "CHF",
        "notional": notional,
        "position_units": 1,
        "cost_price": cost_price,
        "coupon": coupon,
        "barrier_pct": barrier_pct,
        "underlyings": ["NESN"],
        "underlying_isins": ["CH0012221716"],
        "initial_levels": [initial_level],
        "strike": [strike],
        "current_spots": [current_spot],
        "initial_fixing_date": initial_fixing_date,
        "maturity_date": maturity_date,
    })


def make_mbrc_row(
    notional=100_000,
    cost_price=1.0,
    coupon=0.08,
    barrier_pct=0.60,
    initial_levels=None,
    strikes=None,
    current_spots=None,
    initial_fixing_date="2024-01-01",
    maturity_date="2027-06-01",
):
    initial_levels = initial_levels or [100.0, 80.0]
    strikes = strikes or [100.0, 80.0]
    current_spots = current_spots or [95.0, 72.0]
    return pd.Series({
        "product_id": "MBRC001",
        "product_type": "MBRC",
        "type_style": "european",
        "currency": "CHF",
        "notional": notional,
        "position_units": 1,
        "cost_price": cost_price,
        "coupon": coupon,
        "barrier_pct": barrier_pct,
        "underlyings": ["NESN", "NOVN"],
        "underlying_isins": ["CH0012221716", "CH0012221717"],
        "initial_levels": initial_levels,
        "strike": strikes,
        "current_spots": current_spots,
        "initial_fixing_date": initial_fixing_date,
        "maturity_date": maturity_date,
    })


def make_portfolio():
    rows = [make_brc_row(), make_mbrc_row()]
    return pd.DataFrame(rows)


BETA_MAP = {
    "CH0012221716": 1.0,   # NESN
    "CH0012221717": 1.2,   # NOVN
}

VOL_MAP = {
    "CH0012221716": 0.20,
    "CH0012221717": 0.25,
}

SCENARIOS = {
    "base": {"market_shock": 0},
    "down_10": {"market_shock": -10},
    "down_30": {"market_shock": -30},
}


@pytest.fixture
def brc_row():
    return make_brc_row()


@pytest.fixture
def mbrc_row():
    return make_mbrc_row()


@pytest.fixture
def portfolio():
    return make_portfolio()
