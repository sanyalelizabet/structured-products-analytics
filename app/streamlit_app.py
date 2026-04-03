import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# make src importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reverse_convertible import ReverseConvertible

st.set_page_config(page_title="Structured Products Dashboard", layout="wide")
st.title("Structured Products Analytics")

# =========================
# Portfolio Input
# =========================

p1 = {
    "product_id": "CH1483491150",
    "product_type": "BRC",
    "type_style": "European",
    "underlyings": ["ALCON"],
    "underlying_isins": ["CH0432492467"],
    "currency": "CHF",
    "position_units": 10,
    "notional": 1000,
    "cost_price": 1.00,
    "initial_levels": [59.72],
    "current_spots": [58.76],
    "strike": [59.72],
    "barrier_pct": 0.70,
    "coupon": 0.04,
    "initial_fixing_date": "2025-11-10",
    "maturity_date": "2026-11-17",
    "barrier_breached": False
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
        "CH0012032048"
    ],
    "currency": "CHF",
    "position_units": 5,
    "notional": 1000,
    "cost_price": 0.98,
    "initial_levels": [35.00, 70.00, 90.00, 250.00],
    "current_spots": [34.00, 68.00, 92.00, 245.00],
    "strike": [35.00, 70.00, 90.00, 250.00],
    "barrier_pct": 0.70,
    "coupon": 0.0675,
    "initial_fixing_date": "2025-12-30",
    "maturity_date": "2026-12-28",
    "barrier_breached": True
}

p3 = {
    "product_id": "CH1461018793",
    "product_type": "MBRC",
    "type_style": "European",
    "underlyings": ["ABB", "LONZA", "NESTLE"],
    "underlying_isins": ["CH0012221716", "CH0013841017", "CH0038863350"],
    "currency": "CHF",
    "position_units": 1,
    "notional": 10000,
    "cost_price": 1.00,
    "initial_levels": [53.94, 555.20, 72.49],
    "current_spots": [53.94, 555.20, 72.49],
    "strike": [53.94, 555.20, 72.49],
    "barrier_pct": 0.70,
    "coupon": 0.0866,
    "initial_fixing_date": "2025-08-19",
    "maturity_date": "2026-08-19",
    "barrier_breached": False
}

portfolio = pd.DataFrame([p1, p2, p3])

# =========================
# Build Analytics
# =========================

results = []

for _, row in portfolio.iterrows():
    rc = ReverseConvertible(row)
    res = rc.summary()
    res["product_id"] = row["product_id"]
    results.append(res)

df = pd.DataFrame(results)

# format %
df["return_pa"] *= 100
df["distance_to_barrier"] *= 100

# reduce columns


# =========================
# Portfolio Overview
# =========================

overview_cols = [
    "product_id",
    "product_type",
    "underlyings",
    "notional",
    "currency",
    "coupon",
    "maturity_date"
]

st.subheader("Portfolio Overview")
st.dataframe(portfolio[overview_cols], use_container_width=True)

# =========================
# Product Selection
# =========================

selected_product = st.selectbox(
    "Select product",
    portfolio["product_id"].tolist()
)

row = df[df["product_id"] == selected_product].iloc[0]


df_display = df[
    [
        "product_id",
        "maturity_date",
        "days_to_expiry",
        "product_type",
        "total_payoff",
        "total_cost",
        "break_even",
        "return_pa",
        "return_pct",
        "distance_to_barrier"
    ]
]

df_display = df_display.rename(columns={
    "product_id": "Product ID",
    "maturity_date": "Maturity Date",
    "days_to_expiry": "Days to Expiry",
    "product_type": "Product Type",
    "total_payoff": "Total Payoff",
    "total_cost": "Total Cost",
    "break_even": "Break-even",
    "return_pa": "Return p.a. (%)",
    "return_pct": "Return (%)",
    "distance_to_barrier": "Distance to Barrier (%)"
})
row_selected = df_display[df_display["Product ID"] == selected_product].iloc[0]



st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("P&L", f"{row['pnl']:.2f}")
col2.metric("Total Payoff", f"{row['total_payoff']:.2f}")
col3.metric("Days to Expiry", f"{int(row['days_to_expiry'])}")

col4, col5, col6 = st.columns(3)
col4.metric("Return p.a. (%)", f"{row['return_pa']:.2f}")
col5.metric("Distance to Barrier (%)", f"{row['distance_to_barrier']:.2f}")
col6.metric("Worst Underlying", row["worst_underlying"])

# =========================
# Detailed Table
# =========================

st.subheader("Product Detail")

detail = pd.DataFrame({
    "Metric": row_selected.index,
    "Value": row_selected.values
})

detail["Value"] = detail["Value"].apply(
    lambda x: round(float(x), 2) if isinstance(x, (int, float, np.floating)) else x
)

st.table(detail)