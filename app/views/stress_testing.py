import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.scenario_engine import ScenarioEngine
from app.config import SCENARIO_PRESETS, SCENARIO_CUSTOM_DEFAULT


def render(portfolio, corr_df, beta_map, vol_map):

    st.subheader("Stress Testing (Path-Based)")
    st.caption("Scenario defined as a time path: drift → shocks → drift to maturity.")

    # =========================
    # Scenario Builder
    # =========================
    selected_preset = st.selectbox("Scenario Preset", list(SCENARIO_PRESETS.keys()))
    preset  = SCENARIO_PRESETS[selected_preset]
    default = preset if preset is not None else SCENARIO_CUSTOM_DEFAULT

    col1, col2 = st.columns(2)

    with col1:
        market_shock = st.slider(
            "Market Shock per Event (%)", min_value=-25, max_value=2,
            value=int(default["market_shock"])
        )
        n_shocks = st.number_input(
            "Number of Shock Events", min_value=1, max_value=3,
            value=int(default["n_shocks"])
        )
        shock_in_days = st.number_input(
            "Days to First Shock", min_value=1, max_value=300,
            value=int(default["shock_in_days"])
        )

    with col2:
        shock_spacing_days = st.number_input(
            "Days Between Shocks", min_value=0, max_value=365,
            value=int(default["shock_spacing_days"])
        )
        pre_shock_drift = st.slider(
            "Pre-Shock Drift (% p.a.)", min_value=-20.0, max_value=20.0,
            value=float(default["pre_shock_drift_pa"] * 100)
        ) / 100
        post_shock_drift = st.slider(
            "Post-Shock Drift (% p.a.)", min_value=-20.0, max_value=20.0,
            value=float(default["post_shock_drift_pa"] * 100)
        ) / 100

    st.write(f"Selected market shock: {market_shock}%")

    scenario = {
        "market_shock":       market_shock,
        "n_shocks":           int(n_shocks),
        "shock_in_days":      int(shock_in_days),
        "shock_spacing_days": int(shock_spacing_days),
        "pre_shock_drift_pa": float(pre_shock_drift),
        "post_shock_drift_pa": float(post_shock_drift),
    }

    # =========================
    # Run engine
    # =========================
    engine = ScenarioEngine(portfolio=portfolio, beta_map=beta_map, vol_map=vol_map)
    res = engine.run_path_scenario(scenario, corr_df=corr_df)

    paths = res["paths"]
    path_rows = []
    for isin, df_path in paths.items():
        temp = df_path.copy()
        temp["isin"] = isin
        path_rows.append(temp)
    path_plot_df = pd.concat(path_rows, ignore_index=True)

    isin_to_name = {}
    for _, row in portfolio.iterrows():
        for isin, name in zip(row["underlying_isins"], row["underlyings"]):
            isin_to_name[isin] = name
    path_plot_df["name"] = path_plot_df["isin"].map(isin_to_name)

    # =========================
    # Format results
    # =========================
    product_df   = res["product_df"].copy()
    pf_df        = res["pf_scenario_per_ccy"].copy()
    cash_df      = res["cash_positions"].copy()
    delivered_df = res["delivered_stocks"]

    product_df["return_pct"]             *= 100
    pf_df["portfolio_return_pct"]        *= 100
    product_df  = product_df.round(2)
    pf_df       = pf_df.round(2)
    cash_df     = cash_df.round(2)

    if len(delivered_df) > 0:
        delivered_df = delivered_df.copy()
        delivered_df["return_pct"] *= 100
        delivered_df = delivered_df.round(2)

    # =========================
    # Portfolio Stress Summary
    # =========================
    st.write("### Portfolio Stress Summary")
    st.dataframe(
        pf_df[["currency", "n_products", "underlyings", "total_cost", "total_payoff", "total_pnl", "portfolio_return_pct"]].rename(columns={
            "currency":             "Currency",
            "n_products":           "Number of Products",
            "underlyings":          "Underlyings",
            "total_cost":           "Total Cost",
            "total_payoff":         "Total Payoff",
            "total_pnl":            "Total PnL",
            "portfolio_return_pct": "Portfolio Return (%)"
        }),
        width="stretch", hide_index=True
    )

    st.write("### Stress Testing Results")
    left_col, right_col = st.columns([2, 3])

    with left_col:
        st.write("### Product-Level Stress Results")
        st.dataframe(
            product_df[["product_id", "currency", "worst_underlying", "settlement_type", "total_payoff", "pnl", "return_pct"]].rename(columns={
                "product_id":       "Product ID",
                "currency":         "Currency",
                "worst_underlying": "Worst Underlying",
                "settlement_type":  "Settlement Type",
                "total_payoff":     "Total Payoff",
                "pnl":              "PnL",
                "return_pct":       "Return (%)"
            }),
            width="stretch", hide_index=True
        )

        st.write("### Delivered Stocks (Physical Settlement)")
        if len(delivered_df) > 0:
            st.dataframe(
                delivered_df.rename(columns={
                    "delivered_underlying": "Delivered Underlying",
                    "delivered_shares":     "Delivered Shares",
                    "strike":               "Strike",
                    "price":                "Final Spot",
                    "currency":             "Currency",
                    "fractional_cash":      "Fractional Cash",
                    "cash_redemption":      "Cash Redemption",
                    "pnl":                  "PnL",
                    "return_pct":           "Return (%)"
                }),
                width="stretch", hide_index=True
            )
        else:
            st.info("No physical delivery in this scenario.")

        st.write("### Cash Positions")
        st.dataframe(
            cash_df.rename(columns={"currency": "Currency", "total_cash": "Total Cash Redemption"}),
            width="stretch", hide_index=True
        )

    with right_col:
        st.write("### Underlying Scenario Paths")

        isin_to_strike = {}
        for _, prow in portfolio.iterrows():
            for isin, strike in zip(prow["underlying_isins"], prow["strike"]):
                isin_to_strike[isin] = strike

        today = pd.Timestamp.today().normalize()
        shock_dates = [
            today + pd.Timedelta(days=shock_in_days + i * shock_spacing_days)
            for i in range(int(n_shocks))
        ]

        colors = ["#4C9BE8", "#E8844C", "#4CE87A", "#E84C4C",
                  "#C44CE8", "#E8D94C", "#4CE8D9", "#E84C99"]

        fig_paths = go.Figure()

        for idx, (isin, grp) in enumerate(path_plot_df.groupby("isin")):
            name   = isin_to_name.get(isin, isin)
            strike = isin_to_strike.get(isin)
            color  = colors[idx % len(colors)]
            grp    = grp.sort_values("date")
            norm   = grp["price"] / grp["price"].iloc[0] * 100

            fig_paths.add_trace(go.Scatter(
                x=grp["date"], y=norm,
                mode="lines", name=name,
                line=dict(color=color, width=2),
                customdata=grp["price"].values,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Price: %{customdata:.2f}<br>"
                    "Normalised: %{y:.1f}<extra></extra>"
                )
            ))

            if strike is not None:
                norm_strike = strike / grp["price"].iloc[0] * 100
                fig_paths.add_hline(
                    y=norm_strike,
                    line=dict(color=color, dash="dash", width=1),
                    opacity=0.45,
                    annotation_text=f"{name} strike",
                    annotation_position="right",
                    annotation_font=dict(color=color, size=10),
                )

        fig_paths.add_hline(y=100, line=dict(color="grey", dash="dot", width=1), opacity=0.4)

        for i, sd in enumerate(shock_dates):
            fig_paths.add_vline(
                x=sd.timestamp() * 1000,
                line=dict(color="red", dash="dash", width=1.5),
                opacity=0.8,
                annotation_text=f"Shock {i+1}: {market_shock:+}%",
                annotation_position="top right" if i % 2 == 0 else "top left",
                annotation_font=dict(color="red", size=10),
            )

        fig_paths.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(t=40, b=40, l=40, r=120),
            xaxis_title="Date",
            yaxis_title="Normalised Price (base 100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )

        st.plotly_chart(fig_paths, width="stretch")
