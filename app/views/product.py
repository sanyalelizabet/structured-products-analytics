import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Palette — kept in sync with ``app/views/portfolio.py`` so the app speaks
# one visual language.
_LINE_PALETTE = [
    "#4E79A7", "#76A65A", "#9C755F", "#C9A961", "#2A3F5F", "#7A8797",
    "#3FA29E", "#8E7CC3", "#A35F4A", "#5E8A6E", "#7E91AB",
]


def render(portfolio, df, analytics, valuation_date, vol_map, beta_map):

    overview_cols = ["product_id", "product_type", "underlyings", "notional", "currency", "coupon", "maturity_date"]

    st.subheader("Portfolio Overview")
    st.caption(
        "Projected at maturity assuming current spot prices remain unchanged. "
        "Coupons accrued over full product life."
    )

    # Merge fair value into overview
    fv_cols = df[["product_id", "fair_value", "fair_value_pct"]].copy()
    fv_cols["fair_value_pct"] = fv_cols["fair_value_pct"] * 100

    overview_display = portfolio[overview_cols].copy()
    overview_display["coupon"] = overview_display["coupon"] * 100
    overview_display = overview_display.merge(fv_cols, on="product_id", how="left")

    st.dataframe(
        overview_display,
        width="stretch",
        hide_index=True,
        column_config={
            "product_id":     st.column_config.TextColumn("Product ID",      width="medium"),
            "product_type":   st.column_config.TextColumn("Type",            width="small"),
            "underlyings":    st.column_config.ListColumn("Underlyings",     width="medium"),
            "notional":       st.column_config.NumberColumn("Notional",      width="small", format="%.0f"),
            "currency":       st.column_config.TextColumn("CCY",             width="small"),
            "coupon":         st.column_config.NumberColumn("Coupon (%)",    width="small", format="%.2f"),
            "maturity_date":  st.column_config.TextColumn("Maturity",        width="small"),
            "fair_value":     st.column_config.NumberColumn("Fair Value",    width="small", format="%.2f"),
            "fair_value_pct": st.column_config.NumberColumn("Fair Value (%)", width="small", format="%.2f"),
        }
    )

    # =========================
    # Product Selection
    # =========================
    selected_product = st.selectbox("Select product", portfolio["product_id"].tolist())
    row = df[df["product_id"] == selected_product].iloc[0]

    df_display = df[[
        "product_id", "maturity_date", "days_to_expiry", "product_type",
        "total_payoff", "total_cost", "break_even", "ytm", "ytm_today", "return_pct", "distance_to_barrier"
    ]].rename(columns={
        "product_id":         "Product ID",
        "maturity_date":      "Maturity Date",
        "days_to_expiry":     "Days to Expiry",
        "product_type":       "Product Type",
        "total_payoff":       "Total Payoff",
        "total_cost":         "Total Cost",
        "break_even":         "Break-even",
        "ytm":                "YTM from Purchase (%)",
        "ytm_today":          "YTM from Today (%)",
        "return_pct":         "Total Return (%)",
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
    col4.metric("YTM from Today (%)", f"{row['ytm_today']:.2f}")
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

    col_detail, col_chart = st.columns([1, 2])

    with col_detail:
        st.markdown("**Product Detail**")
        st.caption("Key metrics for the selected product.")
        detail = pd.DataFrame({"Metric": row_selected.index, "Value": row_selected.values})
        detail["Value"] = detail["Value"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else str(x)
        )
        st.dataframe(detail, width="stretch", hide_index=True)

    with col_chart:
        st.markdown("**Underlying Price History**")
        st.caption(
            "Last ~2 years, normalised to base 100. Dashed lines mark each strike."
        )
        _render_underlying_prices(analytics, prod_row)

    # =========================
    # Underlying Metrics (full width below)
    # =========================
    st.markdown("**Underlying Metrics**")
    st.caption("Spot, strike, vol, β and YTD per underlying for this product.")
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


def _render_underlying_prices(analytics, prod_row):
    """Plot historical prices of each underlying for the selected product.

    Lines are normalised to base 100 at the first available date so
    multiple price scales (e.g. CHF Nestlé vs USD Coca-Cola) can share
    one chart.  Strike levels are drawn as dashed horizontal lines in
    the same colour as their underlying, also normalised to base 100.
    """
    db = getattr(analytics, "price_db", None)
    if db is None or db.empty:
        return

    isins   = list(prod_row["underlying_isins"])
    names   = list(prod_row["underlyings"])
    strikes = list(prod_row["strike"])
    spots   = list(prod_row["current_spots"])

    sub = db[db["isin"].isin(isins)].copy()
    if sub.empty:
        return
    sub["date"] = pd.to_datetime(sub["date"])
    # Trim to the most recent ~2 years so the chart isn't dominated by old data.
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=2)
    sub = sub[sub["date"] >= cutoff]
    if sub.empty:
        return

    fig = go.Figure()
    for idx, (isin, name) in enumerate(zip(isins, names)):
        series = (
            sub[sub["isin"] == isin]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
        )
        if series.empty:
            continue
        color = _LINE_PALETTE[idx % len(_LINE_PALETTE)]
        spot0 = float(series["price"].iloc[0])
        norm  = series["price"] / spot0 * 100.0

        fig.add_trace(go.Scatter(
            x=series["date"], y=norm,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            customdata=series["price"].values,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "%{x|%d %b %Y}<br>"
                "Price: %{customdata:.2f}<br>"
                "Normalised: %{y:.1f}<extra></extra>"
            ),
        ))

        # Strike line — normalised to the same base
        try:
            strike = float(strikes[idx])
            fig.add_hline(
                y=strike / spot0 * 100.0,
                line=dict(color=color, dash="dash", width=1),
                opacity=0.45,
                annotation_text=f"{name} strike",
                annotation_position="right",
                annotation_font=dict(color=color, size=9),
            )
        except (IndexError, TypeError, ValueError):
            pass

    fig.add_hline(y=100, line=dict(color="grey", dash="dot", width=1), opacity=0.4)
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=30, b=30, l=40, r=110),
        xaxis_title="Date",
        yaxis_title="Normalised Price (base 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
