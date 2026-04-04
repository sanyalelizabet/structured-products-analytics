import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# make src importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reverse_convertible import ReverseConvertible
from src.portfolio_analytics import PortfolioAnalytics

st.set_page_config(page_title="Structured Products Dashboard", layout="wide")
st.title("Structured Products Analytics")

if "view" not in st.session_state:
    st.session_state.view = "product"
col1, col2 = st.columns(2)

with col1:
    if st.button("Product View"):
        st.session_state.view = "product"

with col2:
    if st.button("Portfolio View"):
        st.session_state.view = "portfolio"
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
    "barrier_breached": True
}

p3 = {
    "product_id": "CH1461018793",
    "product_type": "MBRC",
    "type_style": "European",
    "underlyings": ["ABB", "LONZA", "NESTLE"],
    "underlying_isins": ["CH0012221716", "CH0013841017", "CH0038863350"],
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

if st.session_state.view == "product":
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
elif st.session_state.view == "portfolio":

    st.subheader("Portfolio Analytics")
    st.caption(
    "Returns assume current market levels remain unchanged until maturity."
    )

    analytics = PortfolioAnalytics(portfolio, reference_currency="CHF")
    analytics.build_product_analytics()

    st.write("### Portfolio Overview")
    portfolio_overview = analytics.total_portfolio_metrics()
    ref_ccy = portfolio_overview["reference_currency"]


    col1, col2, col3, col4 = st.columns(4)
    
    pnl = portfolio_overview["total_pnl"]
    notional = portfolio_overview["total_notional"]
    
    col1.metric("Total PnL", f"{ref_ccy} {pnl:,.2f}")
    col2.metric("Return (%)", f"{portfolio_overview['portfolio_return_pct']*100:.2f}")
    col3.metric("Total Notional", f"{ref_ccy} {notional:,.2f}")
    col4.metric("# Products", portfolio_overview["total_products"])

    
    st.write("### Product Analytics")

    ref_ccy = analytics.reference_currency
    
    product_table = analytics.product_df.copy()
    
    product_table["cost_ref"] = product_table.apply(
        lambda row: analytics.convert_to_reference(row["total_cost"], row["currency"]),
        axis=1
    )
    
    product_table["payoff_ref"] = product_table.apply(
        lambda row: analytics.convert_to_reference(row["total_payoff"], row["currency"]),
        axis=1
    )
    
    product_table["pnl_ref"] = product_table.apply(
        lambda row: analytics.convert_to_reference(row["pnl"], row["currency"]),
        axis=1
    )
    
    product_table = product_table[
        [
            "product_id",
            "currency",
            "notional",
            "maturity_date",
            "product_type",
            "underlyings",
            "cost_ref",
            "payoff_ref",
            "pnl_ref",
            "return_pct",
            "distance_to_barrier"
        ]
    ].copy()
    
    product_table = product_table.round(2)
    
    st.dataframe(
        product_table,
        width=1400,
        hide_index=True,
        column_config={
            "product_id": "Product ID",
            "currency": "Original CCY",
            "notional": st.column_config.NumberColumn("Notional", format="%.2f"),
            "maturity_date": "Maturity Date",
            "product_type": "Product Type",
            "underlyings": "Underlyings",
            "cost_ref": st.column_config.NumberColumn(f"Cost ({ref_ccy})", format="%.2f"),
            "payoff_ref": st.column_config.NumberColumn(f"Payoff ({ref_ccy})", format="%.2f"),
            "pnl_ref": st.column_config.NumberColumn(f"PnL ({ref_ccy})", format="%.2f"),
            "return_pct": st.column_config.NumberColumn("Return (%)", format="%.2f"),
            "distance_to_barrier": st.column_config.NumberColumn("Distance to Barrier (%)", format="%.2f"),
        }
    )
    
    st.write("### Underlying Exposure")


    underlying_table = analytics.underlying_lookthrough().copy()
    underlying_table = underlying_table.round(2)
    
    st.dataframe(
        underlying_table,
        width=1400,
        hide_index=True,
        column_config={
            "underlying": "Underlying",
            "isin": "ISIN",
            "price_ccy": "Price CCY",
    
            "n_products": "Number of Products",
    
            "allocated_cost_ref": st.column_config.NumberColumn(
                f"Allocated Cost ({analytics.reference_currency})",
                format="%.2f"
            ),
    
            "avg_current_spot": st.column_config.NumberColumn(
                "Current Price",
                format="%.2f"
            ),
    
            "min_distance_to_barrier": st.column_config.NumberColumn(
                "Min Distance to Barrier (%)",
                format="%.2f"
            ),
    
            "avg_distance_to_barrier": st.column_config.NumberColumn(
                "Avg Distance to Barrier (%)",
                format="%.2f"
            ),
    
            "worst_of_count": "Worst-Of Count",
    
            "weight": st.column_config.NumberColumn(
                "Portfolio Weight",
                format="%.2f"
            ),
        }
    )
    st.write("### Maturity Profile")

    maturity_table = analytics.maturity_profile().copy()
    maturity_table = maturity_table.round(2)
    
    st.dataframe(
        maturity_table,
        width=800,
        hide_index=True,
        column_config={
            "maturity_bucket": "Maturity Bucket",
            "n_products": "Number of Products",
            "total_cost": st.column_config.NumberColumn("Total Cost", format="%.2f"),
            "total_payoff": st.column_config.NumberColumn("Total Payoff", format="%.2f"),
            "total_pnl": st.column_config.NumberColumn("Total PnL", format="%.2f"),
        }
    )
