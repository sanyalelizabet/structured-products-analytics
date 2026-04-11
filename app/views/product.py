import streamlit as st
import pandas as pd
import numpy as np


def render(portfolio, df, analytics, valuation_date, vol_map, beta_map):

    overview_cols = ["product_id", "product_type", "underlyings", "notional", "currency", "coupon", "maturity_date"]

    st.subheader("Portfolio Overview")
    st.caption(
        "Projected at maturity assuming current spot prices remain unchanged. "
        "Coupons accrued over full product life."
    )

    overview_display = portfolio[overview_cols].copy()
    overview_display["coupon"] = overview_display["coupon"] * 100

    st.dataframe(
        overview_display,
        width="stretch",
        hide_index=True,
        column_config={
            "product_id":    st.column_config.TextColumn("Product ID",   width="medium"),
            "product_type":  st.column_config.TextColumn("Type",         width="small"),
            "underlyings":   st.column_config.ListColumn("Underlyings",  width="medium"),
            "notional":      st.column_config.NumberColumn("Notional",   width="small", format="%.0f"),
            "currency":      st.column_config.TextColumn("CCY",          width="small"),
            "coupon":        st.column_config.NumberColumn("Coupon (%)", width="small", format="%.2f"),
            "maturity_date": st.column_config.TextColumn("Maturity",     width="small"),
        }
    )

    # =========================
    # Product Selection
    # =========================
    selected_product = st.selectbox("Select product", portfolio["product_id"].tolist())
    row = df[df["product_id"] == selected_product].iloc[0]

    df_display = df[[
        "product_id", "maturity_date", "days_to_expiry", "product_type",
        "total_payoff", "total_cost", "break_even", "return_pa", "return_pct", "distance_to_barrier"
    ]].rename(columns={
        "product_id":         "Product ID",
        "maturity_date":      "Maturity Date",
        "days_to_expiry":     "Days to Expiry",
        "product_type":       "Product Type",
        "total_payoff":       "Total Payoff",
        "total_cost":         "Total Cost",
        "break_even":         "Break-even",
        "return_pa":          "Return p.a. (%)",
        "return_pct":         "Return (%)",
        "distance_to_barrier":"Downside to Barrier (%)",
    })
    row_selected = df_display[df_display["Product ID"] == selected_product].iloc[0]

    # =========================
    # Key Metrics
    # =========================
    st.subheader("Key Metrics")
    vd_label = valuation_date.strftime('%d.%m.%Y') if valuation_date is not None else "Using default values"
    st.caption(f"Valuation Date: {vd_label}")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Projected P&L (flat spot)",
        f"{row['pnl']:,.2f}",
        delta=f"{row['pnl_delta']:+,.2f} vs yesterday" if row["pnl_delta"] is not None else None
    )
    col2.metric("Total Payoff", f"{row['total_payoff']:,.2f}")
    col3.metric("Days to Expiry", f"{int(row['days_to_expiry'])}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Return p.a. (%)", f"{row['return_pa']:.2f}")
    col5.metric(
        "Downside to Barrier (%)",
        f"{row['distance_to_barrier']:.2f}%",
        delta=f"{row['distance_delta']*100:+.2f}pp vs yesterday" if row["distance_delta"] is not None else None,
        delta_color="normal"
    )
    col6.metric("Worst Underlying", row["worst_underlying"])

    # =========================
    # Product Detail + Underlying Metrics
    # =========================
    st.subheader("Product Detail")

    prod_row    = portfolio[portfolio["product_id"] == selected_product].iloc[0]
    und_isins   = prod_row["underlying_isins"]
    und_names   = prod_row["underlyings"]
    und_spots   = prod_row["current_spots"]
    und_strikes = prod_row["strike"]

    und_rows = []
    for name, isin, spot, strike in zip(und_names, und_isins, und_spots, und_strikes):
        ytd = analytics.calc_ytd(isin, spot)
        und_rows.append({
            "Underlying": name,
            "Spot":    round(spot, 2),
            "Strike":  round(strike, 2),
            "Vol (%)": round(vol_map.get(isin, float("nan")) * 100, 1),
            "Beta":    beta_map.get(isin, float("nan")),
            "YTD (%)": round(ytd, 2) if ytd is not None else "n/a",
        })
    und_df = pd.DataFrame(und_rows)

    col_detail, col_und = st.columns([1, 2])

    with col_detail:
        detail = pd.DataFrame({"Metric": row_selected.index, "Value": row_selected.values})
        detail["Value"] = detail["Value"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else str(x)
        )
        st.dataframe(detail, width="stretch", hide_index=True)

    with col_und:
        st.caption("Underlying Metrics")
        st.dataframe(
            und_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Spot":    st.column_config.NumberColumn("Spot",    format="%.2f"),
                "Strike":  st.column_config.NumberColumn("Strike",  format="%.2f"),
                "Vol (%)": st.column_config.NumberColumn("Vol (%)", format="%.1f"),
                "Beta":    st.column_config.NumberColumn("Beta",    format="%.2f"),
                "YTD (%)": st.column_config.NumberColumn("YTD (%)", format="%.2f"),
            }
        )
