import pandas as pd

# =========================
# Portfolio Input
# =========================
# This will be replaced by a CSV loader once all required fields are finalised.

p1 = {
    "product_id": "CH1483491150",
    "product_type": "BRC",
    "type_style": "European",
    "underlyings": ["ALCON"],
    "underlying_isins": ["CH0432492467"],
    "tickers": ["ALC.SW"],
    "currency": "CHF",
    "position_units": 1,
    "notional": 1000,
    "cost_price": 1.00,
    "initial_levels": [59.72],
    "current_spots": [58.76],
    "strike": [59.72],
    "barrier_pct": 0.70,
    "coupon": 0.04,
    "initial_fixing_date": "2025-11-10",
    "maturity_date": "2026-11-17",
    "barrier_breached": False,
}

p2 = {
    "product_id": "CH1449111066",
    "product_type": "MBRC",
    "type_style": "European",
    "underlyings": ["ABB", "HOLCIM", "NOVARTIS", "ROCHE"],
    "underlying_isins": [
        "CH0012221716",
        "CH0012214059",
        "CH0012005267",
        "CH0012032048",
    ],
    "tickers": ["ABBN.SW", "HOLN.SW", "NOVN.SW", "ROG.SW"],
    "currency": "CHF",
    "position_units": 1,
    "notional": 1000,
    "cost_price": 0.98,
    "initial_levels": [35.00, 70.00, 90.00, 250.00],
    "current_spots": [34.00, 68.00, 92.00, 245.00],
    "strike": [35.00, 70.00, 90.00, 250.00],
    "barrier_pct": 0.70,
    "coupon": 0.0675,
    "initial_fixing_date": "2025-12-30",
    "maturity_date": "2026-12-28",
    "barrier_breached": True,
}

p3 = {
    "product_id": "CH1461018793",
    "product_type": "MBRC",
    "type_style": "European",
    "underlyings": ["ABB", "LONZA", "NESTLE"],
    "underlying_isins": ["CH0012221716", "CH0013841017", "CH0038863350"],
    "tickers": ["ABBN.SW", "LONN.SW", "NESN.SW"],
    "currency": "CHF",
    "position_units": 10,
    "notional": 10000,
    "cost_price": 1.00,
    "initial_levels": [53.94, 555.20, 72.49],
    "current_spots": [53.94, 555.20, 72.49],
    "strike": [53.94, 555.20, 72.49],
    "barrier_pct": 0.70,
    "coupon": 0.0866,
    "initial_fixing_date": "2025-08-19",
    "maturity_date": "2026-08-19",
    "barrier_breached": False,
}

p4 = {
    "product_id": "CH1483484015",
    "product_type": "BRC",
    "type_style": "European",
    "underlyings": ["Airbnb Inc."],
    "underlying_isins": ["US0090661010"],
    "tickers": ["ABNB"],
    "currency": "USD",
    "position_units": 1,
    "notional": 5000,
    "cost_price": 0.98,
    "initial_levels": [120.46],
    "current_spots": [120.46],
    "strike": [120.46],
    "barrier_pct": 0.65,
    "coupon": 0.100556,
    "initial_fixing_date": "2025-10-02",
    "maturity_date": "2026-10-09",
    "barrier_breached": False,
}

portfolio = pd.DataFrame([p1, p2, p3, p4])
