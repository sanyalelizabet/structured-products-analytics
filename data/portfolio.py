import pandas as pd

# =========================
# Portfolio Input
# =========================
# This will be replaced by a CSV loader once all required fields are finalised.
#
# Schema notes:
#   denomination     – contractual nominal of one certificate (per-piece)
#   position_units   – number of certificates held
#   total_notional   – derived = denomination * position_units (do not set here)
#   issuer / issuer_rating / issuer_country – populate from the term sheet

p1 = {
    "product_id": "CH1483491150",
    "product_type": "BRC",
    "type_style": "European",
    "issuer": "Bank Vontobel AG",
    "issuer_rating": "",
    "issuer_country": "CH",
    "underlyings": ["ALCON"],
    "underlying_isins": ["CH0432492467"],
    "currency": "CHF",
    "denomination": 1000,
    "position_units": 1,
    "notional": 1000,
    "cost_price": 1.00,
    "initial_levels": [59.72],
    "current_spots": [58.76],
    "current_spot_dates": [None],
    "strike": [59.72],
    "barrier_pct": 0.70,
    "coupon": 0.04,
    "initial_fixing_date": "2025-11-10",
    "purchase_date": "2025-11-10",
    "maturity_date": "2026-11-17",
    "coupon_dates": ["2026-11-17"],
    "day_count": "30/360",
    "barrier_breached": False,
}

p2 = {
    "product_id": "CH1449111066",
    "product_type": "MBRC",
    "type_style": "European",
    "issuer": "Bank Vontobel AG",
    "issuer_rating": "",
    "issuer_country": "CH",
    "underlyings": ["ABB", "HOLCIM", "NOVARTIS", "ROCHE"],
    "underlying_isins": [
        "CH0012221716",
        "CH0012214059",
        "CH0012005267",
        "CH0012032048",
    ],
    "currency": "CHF",
    "denomination": 1000,
    "position_units": 1,
    "notional": 1000,
    "cost_price": 0.98,
    "initial_levels": [35.00, 70.00, 90.00, 250.00],
    "current_spots": [34.00, 68.00, 92.00, 245.00],
    "current_spot_dates": [None, None, None, None],
    "strike": [35.00, 70.00, 90.00, 250.00],
    "barrier_pct": 0.70,
    "coupon": 0.0675,
    "initial_fixing_date": "2025-12-30",
    "purchase_date": "2025-12-30",
    "maturity_date": "2026-12-28",
    "coupon_dates": ["2026-12-28"],
    "day_count": "30/360",
    "barrier_breached": True,
}

p3 = {
    "product_id": "CH1461018793",
    "product_type": "MBRC",
    "type_style": "European",
    "issuer": "Bank Vontobel AG",
    "issuer_rating": "",
    "issuer_country": "CH",
    "underlyings": ["ABB", "LONZA", "NESTLE"],
    "underlying_isins": ["CH0012221716", "CH0013841017", "CH0038863350"],
    "currency": "CHF",
    "denomination": 1000,
    "position_units": 10,
    "notional": 10000,
    "cost_price": 1.00,
    "initial_levels": [53.94, 555.20, 72.49],
    "current_spots": [53.94, 555.20, 72.49],
    "current_spot_dates": [None, None, None],
    "strike": [53.94, 555.20, 72.49],
    "barrier_pct": 0.70,
    "coupon": 0.0866,
    "initial_fixing_date": "2025-08-19",
    "purchase_date": "2025-08-19",
    "maturity_date": "2026-08-19",
    "coupon_dates": ["2026-08-19"],
    "day_count": "30/360",
    "barrier_breached": False,
}

p4 = {
    "product_id": "CH1483484015",
    "product_type": "BRC",
    "type_style": "European",
    "issuer": "Bank Vontobel AG",
    "issuer_rating": "",
    "issuer_country": "CH",
    "underlyings": ["Airbnb Inc."],
    "underlying_isins": ["US0090661010"],
    "currency": "USD",
    "denomination": 5000,
    "position_units": 1,
    "notional": 5000,
    "cost_price": 0.98,
    "initial_levels": [120.46],
    "current_spots": [120.46],
    "current_spot_dates": [None],
    "strike": [120.46],
    "barrier_pct": 0.65,
    "coupon": 0.100556,
    "initial_fixing_date": "2025-10-02",
    "purchase_date": "2025-10-02",
    "maturity_date": "2026-10-09",
    "coupon_dates": ["2026-10-09"],
    "day_count": "ACT/360",
    "barrier_breached": False,
}

p5 = {
    "product_id": "XS3317100249",
    "product_type": "MBRC",
    "type_style": "European",
    "issuer": "BBVA Global Markets B.V.",
    "issuer_rating": "",
    "issuer_country": "ES",
    "underlyings": ["Advanced Micro Devices, Inc.", "Broadcom Inc", "NVIDIA Corp"],
    "underlying_isins": [
        "US0079031078",
        "US11135F1012",
        "US67066G1040",
    ],
    "currency": "USD",
    "denomination": 1000,
    "position_units": 10,
    "notional": 10000,
    "cost_price": 1.00,
    "initial_levels": [255.07, 380.78, 196.51],
    "current_spots": [255.07, 380.78, 196.51],
    "current_spot_dates": [None, None, None],
    "strike": [255.07, 380.78, 196.51],
    "barrier_pct": 0.526,
    "coupon": 0.14,
    "initial_fixing_date": "2026-04-14",
    "purchase_date": "2026-04-14",
    "maturity_date": "2027-04-27",
    "coupon_dates": ["2027-04-27", "2028-04-27"],
    "day_count": "ACT/360",
    "barrier_breached": False,
}

# =========================================================================
# p6 — Autocallable BRC on Coca-Cola (Opus / Vontobel)
# =========================================================================
# Term sheet: 5.00 % p.a. Autocallable Reverse Convertible on Coca-Cola
# Company, ISIN CH1532240566, USD 1'000 denomination.  Coupon paid quarterly
# (1.25 % per period).  Two early-redemption monitoring dates: 31 Aug 2026
# and 01 Dec 2026; if KO closes ≥ Autocall Level (USD 81.56 = 100 % of spot
# reference) on either date, the product is called early on the corresponding
# Early Payment Date at par + the relevant period coupon.
#
# Strike Price = 85 % of Spot Reference (USD 69.33).  No separate barrier —
# the strike acts as the protection level.  Hence ``barrier_pct = 1.00``
# (barrier = strike × 1.0).
#
# Autocall trigger expressed as a ratio against the strike (per our schema
# convention): ``autocall_trigger_pct = 81.56 / 69.33 ≈ 1.1764``.

p6 = {
    "product_id":              "CH1532240566",
    "product_type":            "AC_BRC",
    "type_style":              "European",
    "issuer":                  "Opus (Public) Chartered Issuance SA",
    "issuer_rating":           "",
    "issuer_country":          "LU",
    "underlyings":             ["Coca-Cola"],
    "underlying_isins":        ["US1912161007"],
    "currency":                "USD",
    "denomination":            1000,
    "position_units":          1,
    "notional":                1000,
    "cost_price":              1.00,
    "initial_levels":          [81.56],   # Spot Reference Price
    "current_spots":           [81.56],   # placeholder; live feed will refresh
    "current_spot_dates":      [None],
    "strike":                  [69.33],   # Strike Price = 85 % × spot ref
    "barrier_pct":             1.00,      # strike acts as barrier
    "coupon":                  0.05,      # 5.00 % p.a., paid quarterly
    "initial_fixing_date":     "2026-02-27",
    "purchase_date":           "2026-03-06",   # Payment Date
    "maturity_date":           "2027-03-08",   # Repayment Date
    "coupon_dates":            ["2026-06-08", "2026-09-08",
                                 "2026-12-08", "2027-03-08"],
    "day_count":               "ACT/360",
    "barrier_breached":        False,
    # Autocallable-specific fields (consumed by AC_BRC dispatch)
    "autocall_obs_dates":      ["2026-08-31", "2026-12-01"],
    "autocall_trigger_pct":    81.56 / 69.33,   # ≈ 1.17643
    "autocall_coupon_memory":  False,
}


# =========================================================================
# p7 — Capital Protection Note with Participation on Amazon (Vontobel)
# =========================================================================
# Term sheet: ISIN CH1537766565.  6-month CPN on Amazon.com Inc, USD 1'000
# denomination.  Capital protection 95% (USD 950 per certificate),
# participation 52% in upside above strike (USD 209.07 = 100% of spot ref).
# Number of Underlyings B = 1000 / 209.07 = 4.78309 ≈ N·p/K·(1/p).
#
# Payoff per certificate (Vontobel formula):
#     CP + max((SF − X) · B · P, 0)
# which is algebraically identical to
#     N · [protection_pct + participation_pct · max(SF/K − 1, 0)]
# used internally by `CapitalProtectionNote`.
#
# Disclosed at issue: bond-leg NPV USD 933.377, implied IRR 3.6556%.
p7 = {
    # Identifiers
    "product_id":                 "CH1537766565",
    "product_type":               "CPN",
    "type_style":                 "European",

    # Issuer chain (Vontobel: DIFC issuer, Swiss guarantor, keep-well from bank)
    "issuer":                     "Vontobel Financial Products Ltd., DIFC Dubai",
    "issuer_rating":              "",
    "issuer_country":             "AE",
    "guarantor":                  "Vontobel Holding AG, Zurich",
    "guarantor_rating":           "A3",            # Moody's LT Issuer Rating
    "keep_well_agreement":        "Bank Vontobel AG, Zurich (Aa3)",

    # Underlying
    "underlyings":                ["Amazon.com Inc."],
    "underlying_isins":           ["US0231351067"],
    "currency":                   "USD",

    # Sizing
    "denomination":               1000,
    "issue_price":                1000.00,
    "position_units":             10,
    "notional":                   10000,
    "issue_size":                 32000,           # total certificates outstanding
    "cost_price":                 1.00,

    # Spot & strike
    "spot_reference_price":       209.07,
    "initial_levels":             [209.07],
    "current_spots":              [209.07],
    "current_spot_dates":         [None],
    "strike":                     [209.07],
    "strike_pct_of_spot":         1.00,            # 100% of spot reference

    # Product terms
    "protection_pct":             0.95,            # 95% capital protection
    "capital_protection_amount":  950.00,          # USD per certificate
    "participation_pct":          0.52,            # 52% upside participation
    "number_of_underlyings":      4.78309,         # B = denomination / strike
    "coupon":                     0.0,             # zero-coupon

    # Dates
    "initial_fixing_date":        "2026-02-25",
    "payment_date":               "2026-03-04",
    "purchase_date":              "2026-03-04",
    "last_trading_day":           "2026-08-25",
    "final_fixing_date":          "2026-08-25",
    "maturity_date":              "2026-09-01",   # repayment date
    "coupon_dates":               ["2026-09-01"], # bullet at maturity, rate = 0
    "day_count":                  "ACT/360",

    # Issuer-disclosed bond-leg pricing at issue (informational)
    "bond_npv_at_issue":          933.377,
    "implied_irr_at_issue":       0.036556,

    # Not applicable for CPN
    "barrier_pct":                None,
    "barrier_breached":           False,
}


portfolio = pd.DataFrame([p1, p2, p3, p4, p5, p6, p7])
