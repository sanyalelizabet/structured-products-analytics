import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(analytics, df, greeks_df, pf_delta, valuation_date):

    st.subheader("Portfolio Analytics")
    st.caption(
        "Projected at maturity assuming current spot prices remain unchanged. "
        "Coupons accrued over full product life."
    )
    vd_label = valuation_date.strftime('%d.%m.%Y') if valuation_date is not None else "Using default prices"
    st.caption(f"Valuation Date: {vd_label}")

    # =========================
    # Portfolio Overview
    # =========================
    st.write("### Portfolio Overview")
    portfolio_overview = analytics.total_portfolio_metrics()
    ref_ccy  = portfolio_overview["reference_currency"]
    pnl      = portfolio_overview["total_pnl"]
    notional = portfolio_overview["total_notional"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total PnL",      f"{ref_ccy} {pnl:,.2f}")
    col2.metric("Return (%)",     f"{portfolio_overview['portfolio_return_pct']*100:.2f}")
    col3.metric("Total Notional", f"{ref_ccy} {notional:,.2f}")
    col4.metric("# Products",     portfolio_overview["total_products"])

    # =========================
    # Product Analytics
    # =========================
    st.write("### Product Analytics")
    ref_ccy = analytics.reference_currency

    product_table = analytics.product_df.copy()
    product_table["cost_ref"]   = product_table.apply(lambda r: analytics.convert_to_reference(r["total_cost"],   r["currency"]), axis=1)
    product_table["payoff_ref"] = product_table.apply(lambda r: analytics.convert_to_reference(r["total_payoff"], r["currency"]), axis=1)
    product_table["pnl_ref"]    = product_table.apply(lambda r: analytics.convert_to_reference(r["pnl"],          r["currency"]), axis=1)

    # Merge fair value from the enriched df passed in from the app
    fv_cols = df[["product_id", "fair_value", "fair_value_pct"]].copy()
    fv_cols["fair_value_pct"] = fv_cols["fair_value_pct"] * 100
    product_table = product_table.merge(fv_cols, on="product_id", how="left")

    product_table = product_table[[
        "product_id", "currency", "notional", "maturity_date", "product_type",
        "underlyings", "cost_ref", "payoff_ref", "pnl_ref", "return_pct",
        "distance_to_barrier", "fair_value", "fair_value_pct"
    ]].copy()
    product_table["return_pct"]          *= 100
    product_table["distance_to_barrier"] *= 100
    product_table = product_table.round(2)

    st.dataframe(
        product_table,
        width="stretch",
        hide_index=True,
        column_config={
            "product_id":          "Product ID",
            "currency":            "Original CCY",
            "notional":            st.column_config.NumberColumn("Notional",              format="%.2f"),
            "maturity_date":       "Maturity Date",
            "product_type":        "Product Type",
            "underlyings":         "Underlyings",
            "cost_ref":            st.column_config.NumberColumn(f"Cost ({ref_ccy})",     format="%.2f"),
            "payoff_ref":          st.column_config.NumberColumn(f"Payoff ({ref_ccy})",   format="%.2f"),
            "pnl_ref":             st.column_config.NumberColumn(f"PnL ({ref_ccy})",      format="%.2f"),
            "return_pct":          st.column_config.NumberColumn("Return (%)",            format="%.2f"),
            "distance_to_barrier": st.column_config.NumberColumn("Downside to Barrier (%)", format="%.2f"),
            "fair_value":          st.column_config.NumberColumn("Fair Value",            format="%.2f"),
            "fair_value_pct":      st.column_config.NumberColumn("Fair Value (%)",        format="%.2f"),
        }
    )

    # =========================
    # Underlying Exposure
    # =========================
    st.write("### Underlying Exposure")
    underlying_table = analytics.underlying_lookthrough().copy().round(2)

    col_exp, col_tree = st.columns([2, 1])

    with col_tree:
        fig_treemap = px.treemap(
            underlying_table,
            path=["underlying"],
            values="allocated_cost_ref",
            color="min_distance_to_barrier",
            color_continuous_scale=["#2A2F38", "#4A5563", "#7A8797"],
            hover_data={"weight": True, "n_products": True, "min_distance_to_barrier": True},
            labels={
                "allocated_cost_ref":      f"Cost ({analytics.reference_currency})",
                "min_distance_to_barrier": "Min Downside to Barrier (%)"
            }
        )
        fig_treemap.update_layout(height=350, margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig_treemap, width="stretch")

    with col_exp:
        st.dataframe(
            underlying_table,
            width="stretch",
            hide_index=True,
            column_config={
                "underlying":              "Underlying",
                "isin":                    "ISIN",
                "price_ccy":               "Price CCY",
                "n_products":              "Number of Products",
                "allocated_cost_ref":      st.column_config.NumberColumn(f"Allocated Cost ({analytics.reference_currency})", format="%.2f"),
                "avg_current_spot":        st.column_config.NumberColumn("Current Price",             format="%.2f"),
                "min_distance_to_barrier": st.column_config.NumberColumn("Min Downside to Barrier (%)", format="%.2f"),
                "avg_distance_to_barrier": st.column_config.NumberColumn("Avg Downside to Barrier (%)", format="%.2f"),
                "worst_of_count":          "Worst-Of Count",
                "weight":                  st.column_config.NumberColumn("Portfolio Weight", format="%.2f"),
            }
        )

    # =========================
    # Maturity Profile
    # =========================
    st.write("### Maturity Profile")
    maturity_table = analytics.maturity_profile().copy().round(2)

    col_table, col_chart = st.columns(2)

    with col_chart:
        bar_df = analytics.product_df[["maturity_date", "product_type", "notional", "currency", "underlyings"]].copy()
        bar_df["maturity_date"] = pd.to_datetime(bar_df["maturity_date"]).dt.strftime("%Y-%m-%d")
        bar_df["label"] = bar_df["product_type"] + " | " + bar_df["underlyings"]

        fig_maturity = px.bar(
            bar_df,
            x="maturity_date", y="notional", color="product_type", text="label",
            hover_data={"currency": True, "underlyings": True, "label": False},
            labels={"maturity_date": "Maturity Date", "notional": "Notional", "product_type": "Type"},
            color_discrete_sequence=["#2A2F38", "#4A5563", "#7A8797"]
        )
        fig_maturity.update_traces(textposition="inside", textangle=0)
        fig_maturity.update_layout(
            height=350, margin=dict(t=20, b=10, l=10, r=10),
            xaxis_type="category", xaxis_title=None, barmode="stack"
        )
        st.plotly_chart(fig_maturity, width="stretch")

    with col_table:
        st.dataframe(
            maturity_table,
            width="stretch",
            hide_index=True,
            column_config={
                "maturity_bucket": "Maturity Bucket",
                "n_products":      "Number of Products",
                "total_cost":      st.column_config.NumberColumn(f"Total Cost ({analytics.reference_currency})",   format="%.2f"),
                "total_payoff":    st.column_config.NumberColumn(f"Total Payoff ({analytics.reference_currency})", format="%.2f"),
                "total_pnl":       st.column_config.NumberColumn(f"Total PnL ({analytics.reference_currency})",    format="%.2f"),
            }
        )

    # =========================
    # Greeks
    # =========================
    st.write("### Greeks")
    st.caption(
        "Computed via Monte Carlo bump-and-reprice. "
        "Delta = FV change for a 1% spot move. "
        "Vega = FV change for a 1pp vol move. "
        "Theta = FV change per calendar day. "
        "Corr = FV change per 1pp uniform correlation shift (MBRC only)."
    )

    col_greek_table, col_delta_chart = st.columns([2, 1])

    with col_greek_table:
        st.write("#### Per-Product Greeks")
        st.dataframe(
            greeks_df.rename(columns={
                "product_id":  "Product ID",
                "currency":    "CCY",
                "isin":        "ISIN",
                "underlying":  "Underlying",
                "delta_1pct":  "Delta (1% spot)",
                "vega_1pp":    "Vega (1pp vol)",
                "theta":       "Theta (daily)",
                "rho":         "Rho (1bp rate)",
                "corr_sens":   "Corr Sens (1pp)",
            }),
            width="stretch",
            hide_index=True,
            column_config={
                "Delta (1% spot)":  st.column_config.NumberColumn(format="%.2f"),
                "Vega (1pp vol)":   st.column_config.NumberColumn(format="%.2f"),
                "Theta (daily)":    st.column_config.NumberColumn(format="%.2f"),
                "Rho (1bp rate)":   st.column_config.NumberColumn(format="%.2f"),
                "Corr Sens (1pp)":  st.column_config.NumberColumn(format="%.2f"),
            }
        )

        st.write("#### Portfolio Delta by Underlying")
        st.caption("Sum of delta across all products per underlying — shows net directional exposure.")
        st.dataframe(
            pf_delta.rename(columns={
                "isin":             "ISIN",
                "underlying":       "Underlying",
                "currency":         "CCY",
                "total_delta_1pct": "Total Delta (1% spot)",
            }),
            width="stretch",
            hide_index=True,
            column_config={
                "Total Delta (1% spot)": st.column_config.NumberColumn(format="%.2f"),
            }
        )

    with col_delta_chart:
        st.write("#### Portfolio Delta")
        colors = [
            "#4C9BE8" if d >= 0 else "#E84C4C"
            for d in pf_delta["total_delta_1pct"]
        ]
        fig_delta = go.Figure(go.Bar(
            x=pf_delta["total_delta_1pct"],
            y=pf_delta["underlying"],
            orientation="h",
            marker_color=colors,
            text=pf_delta["total_delta_1pct"].apply(lambda x: f"{x:+.0f}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Total Delta: %{x:+.2f}<extra></extra>"
            )
        ))
        fig_delta.update_layout(
            template="plotly_dark",
            height=max(250, len(pf_delta) * 55),
            margin=dict(t=20, b=20, l=10, r=80),
            xaxis_title="FV change per 1% spot move",
            yaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(fig_delta, width="stretch")
