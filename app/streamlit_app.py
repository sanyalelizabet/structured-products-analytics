import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px



# make src importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
from src.portfolio_analytics import PortfolioAnalytics
from src.scenario_engine import ScenarioEngine
from src.eod_client import EODClient
from src.market_data_engine import MarketDataEngine
from data.reference_data import isin_ticker_map, beta_map, vol_map
from data.portfolio import portfolio


@st.cache_resource
def get_market_engine():
    api_key = st.secrets["EOD_API_KEY"]
    client = EODClient(api_key)
    return MarketDataEngine(client)

@st.cache_data(ttl=3600)
def build_product_analytics(_portfolio, _db):
    pa = PortfolioAnalytics(_portfolio, reference_currency="CHF", price_db=_db)
    df = pa.build_product_analytics()
    df["return_pa"] *= 100
    df["distance_to_barrier"] *= 100
    return pa, df

@st.cache_data(ttl=3600)
def build_corr_matrix():
    return get_market_engine().build_corr_matrix(isin_ticker_map, years=4)

@st.cache_data(ttl=3600)
def fetch_market_data(_portfolio):
    engine = get_market_engine()
    try:
        engine.fetch_latest_prices(_portfolio)
        updated_portfolio = engine.update_spots(_portfolio)
        db = engine.load_db()
        valuation_date = db["date"].max() if not db.empty else None
        return updated_portfolio, db, valuation_date, None
    except Exception as e:
        db = engine.load_db()
        valuation_date = db["date"].max() if not db.empty else None
        return _portfolio, db, valuation_date, str(e)


st.set_page_config(page_title="Structured Products Dashboard", layout="wide")
st.title("Structured Products Analytics")
st.sidebar.image(str(logo_path), width=160)
view = st.sidebar.radio("View", ["Product", "Portfolio", "Stress Testing"])



portfolio, db, valuation_date, fetch_error = fetch_market_data(portfolio)
if fetch_error:
    st.warning(f"Could not refresh market prices. Using portfolio default spots. {fetch_error}")
    

# =========================
# Build Analytics
# =========================

analytics, df = build_product_analytics(portfolio, db)
corr_df = build_corr_matrix()


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

if view == "Product":
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
            "product_id":   st.column_config.TextColumn("Product ID",   width="medium"),
            "product_type": st.column_config.TextColumn("Type",         width="small"),
            "underlyings":  st.column_config.ListColumn("Underlyings",  width="medium"),
            "notional":     st.column_config.NumberColumn("Notional",   width="small", format="%.0f"),
            "currency":     st.column_config.TextColumn("CCY",          width="small"),
            "coupon":       st.column_config.NumberColumn("Coupon (%)", width="small", format="%.2f"),
            "maturity_date":st.column_config.TextColumn("Maturity",     width="small"),
        }
    )

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
        "distance_to_barrier": "Downside to Barrier (%)"
    })
    row_selected = df_display[df_display["Product ID"] == selected_product].iloc[0]
    
    



    

    st.subheader("Key Metrics")
    vd_label = valuation_date.strftime('%d.%m.%Y') if valuation_date is not None else "Using default values"
    st.caption(f"Valuation Date: {vd_label}")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Projected P&L (flat spot)",
        f"{row['pnl']:,.2f}",
        delta=f"{row['pnl_delta']:+,.2f} vs yesterday" if row["pnl_delta"] is not None else None
    )
    
    col2.metric(
        "Total Payoff",
        f"{row['total_payoff']:,.2f}"
    )
    
    col3.metric(
        "Days to Expiry",
        f"{int(row['days_to_expiry'])}"
    )
    
    col4, col5, col6 = st.columns(3)
    
    col4.metric(
        "Return p.a. (%)",
        f"{row['return_pa']:.2f}"
    )
    
    col5.metric(
        "Downside to Barrier (%)",
        f"{row['distance_to_barrier']:.2f}%",
        delta=f"{row['distance_delta']*100:+.2f}pp vs yesterday" if row["distance_delta"] is not None else None,
        delta_color="normal"  # positive = further from barrier = green = good
    )
    
    col6.metric(
        "Worst Underlying",
        row["worst_underlying"]
    )

# =========================
# Detailed Table
# =========================

    st.subheader("Product Detail")

    # --- Underlying metrics (spot, vol, beta, YTD) ---
    prod_row = portfolio[portfolio["product_id"] == selected_product].iloc[0]
    und_isins   = prod_row["underlying_isins"]
    und_names   = prod_row["underlyings"]
    und_spots   = prod_row["current_spots"]
    und_strikes = prod_row["strike"]

    def calc_ytd(isin, current_price):
        if db.empty:
            return None
        isin_db = db[db["isin"] == isin].copy()
        if isin_db.empty:
            return None
        year_start = pd.Timestamp(pd.Timestamp.today().year, 1, 1)
        historical = isin_db[isin_db["date"] < year_start].sort_values("date")
        if historical.empty:
            return None
        soy_price = historical.iloc[-1]["price"]
        return (current_price / soy_price - 1) * 100

    und_rows = []
    for name, isin, spot, strike in zip(und_names, und_isins, und_spots, und_strikes):
        ytd = calc_ytd(isin, spot)
        und_rows.append({
            "Underlying": name,
            "Spot":       round(spot, 2),
            "Strike":     round(strike, 2),
            "Vol (%)":    round(vol_map.get(isin, float("nan")) * 100, 1),
            "Beta":       beta_map.get(isin, float("nan")),
            "YTD (%)":    round(ytd, 2) if ytd is not None else "n/a",
        })
    und_df = pd.DataFrame(und_rows)

    col_detail, col_und = st.columns([1, 2])

    with col_detail:
        detail = pd.DataFrame({
            "Metric": row_selected.index,
            "Value": row_selected.values
        })
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
                "Spot":    st.column_config.NumberColumn("Spot", format="%.2f"),
                "Strike":  st.column_config.NumberColumn("Strike", format="%.2f"),
                "Vol (%)": st.column_config.NumberColumn("Vol (%)", format="%.1f"),
                "Beta":    st.column_config.NumberColumn("Beta", format="%.2f"),
                "YTD (%)": st.column_config.NumberColumn("YTD (%)", format="%.2f"),
            }
        )
elif view == "Portfolio":

    st.subheader("Portfolio Analytics")
    st.caption(
    "Projected at maturity assuming current spot prices remain unchanged. "
        "Coupons accrued over full product life."
    )
    vd_label = valuation_date.strftime('%d.%m.%Y') if valuation_date is not None else "Using default prices"
    st.caption(f"Valuation Date: {vd_label}")

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
    
    product_table["return_pct"] *= 100
    product_table["distance_to_barrier"] *= 100
    
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
            "distance_to_barrier": st.column_config.NumberColumn("Downside to Barrier (%)", format="%.2f"),
        }
    )    
    st.write("### Underlying Exposure")

    underlying_table = analytics.underlying_lookthrough().copy()
    underlying_table = underlying_table.round(2)

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
                "allocated_cost_ref": f"Cost ({analytics.reference_currency})",
                "min_distance_to_barrier": "Min Downside to Barrier (%)"
            }
        )
        fig_treemap.update_layout(height=350, margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig_treemap, width='stretch')

    with col_exp:
        st.dataframe(
            underlying_table,
            width='stretch',
            hide_index=True,
            column_config={
                "underlying": "Underlying",
                "isin": "ISIN",
                "price_ccy": "Price CCY",
                "n_products": "Number of Products",
                "allocated_cost_ref": st.column_config.NumberColumn(
                    f"Allocated Cost ({analytics.reference_currency})", format="%.2f"
                ),
                "avg_current_spot": st.column_config.NumberColumn("Current Price", format="%.2f"),
                "min_distance_to_barrier": st.column_config.NumberColumn("Min Downside to Barrier (%)", format="%.2f"),
                "avg_distance_to_barrier": st.column_config.NumberColumn("Avg Downside to Barrier (%)", format="%.2f"),
                "worst_of_count": "Worst-Of Count",
                "weight": st.column_config.NumberColumn("Portfolio Weight", format="%.2f"),
            }
        )
    st.write("### Maturity Profile")    
    
    maturity_table = analytics.maturity_profile().copy()
    maturity_table = maturity_table.round(2)

    col_table, col_chart = st.columns(2)

    with col_chart:
        bar_df = analytics.product_df[
            ["maturity_date", "product_type", "notional", "currency", "underlyings"]
        ].copy()
        bar_df["maturity_date"] = pd.to_datetime(bar_df["maturity_date"]).dt.strftime("%Y-%m-%d")
        bar_df["label"] = bar_df["product_type"] + " | " + bar_df["underlyings"]

        fig_maturity = px.bar(
            bar_df,
            x="maturity_date",
            y="notional",
            color="product_type",
            text="label",
            hover_data={"currency": True, "underlyings": True, "label": False},
            labels={"maturity_date": "Maturity Date", "notional": "Notional", "product_type": "Type"},
            color_discrete_sequence=["#2A2F38", "#4A5563", "#7A8797"]
        )
        fig_maturity.update_traces(textposition="inside", textangle=0)
        fig_maturity.update_layout(
            height=350, margin=dict(t=20, b=10, l=10, r=10),
            xaxis_type="category", xaxis_title=None, barmode="stack"
        )
        st.plotly_chart(fig_maturity,  width="stretch")

    with col_table:
        st.dataframe(
            maturity_table,
            width="stretch",
            hide_index=True,
            column_config={
                "maturity_bucket": "Maturity Bucket",
                "n_products": "Number of Products",
                "total_cost": st.column_config.NumberColumn(f"Total Cost ({analytics.reference_currency})", format="%.2f"),
                "total_payoff": st.column_config.NumberColumn(f"Total Payoff ({analytics.reference_currency})", format="%.2f"),
                "total_pnl": st.column_config.NumberColumn(f"Total PnL ({analytics.reference_currency})", format="%.2f"),
            }
        )

    
elif view == "Stress Testing":    
    
   
    st.subheader("Stress Testing (Path-Based)")
    st.caption(
        "Scenario defined as a time path: drift → shocks → drift to maturity."
        )


    # =========================
    # Scenario Builder
    # =========================
    
   # =========================
    # Presets
    # =========================
    scenario_presets = {
        "Custom": None,
        "Current": {
            "market_shock": 0,
            "n_shocks": 1,
            "shock_in_days": 2,
            "shock_spacing_days": 0,
            "pre_shock_drift_pa": 0.0,
            "post_shock_drift_pa": 0.0,
        },
        "Down 5%": {
            "market_shock": -5,
            "n_shocks": 1,
            "shock_in_days": 2,
            "shock_spacing_days": 0,
            "pre_shock_drift_pa": 0.05,
            "post_shock_drift_pa": 0.05,
        },
        "Down 10%": {
            "market_shock": -10,
            "n_shocks": 1,
            "shock_in_days": 2,
            "shock_spacing_days": 0,
            "pre_shock_drift_pa": 0.05,
            "post_shock_drift_pa":0.05,
        },
        "Crash (-20%)": {
            "market_shock": -20,
            "n_shocks": 1,
            "shock_in_days": 0,
            "shock_spacing_days": 0,
            "pre_shock_drift_pa": 0.05,
            "post_shock_drift_pa": 0.05,
        },
        "Recovery (+10% + ERP)": {
            "market_shock": -20,
            "n_shocks": 1,
            "shock_in_days": 1,
            "shock_spacing_days": 0,
            "pre_shock_drift_pa": 0.05,
            "post_shock_drift_pa": 0.05,
        },
    }

    selected_preset = st.selectbox(
        "Scenario Preset",
        list(scenario_presets.keys())
    )

    preset = scenario_presets[selected_preset]

    # =========================
    # Default values (from preset or fallback)
    # =========================
    default = preset if preset is not None else {
        "market_shock": -10,
        "n_shocks": 1,
        "shock_in_days": 15,
        "shock_spacing_days": 0,
        "pre_shock_drift_pa": 0.05,
        "post_shock_drift_pa": 0.0,
    }

    # =========================
    # Manual Controls
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        market_shock = st.slider(
            "Market Shock per Event (%)",
            min_value=-25,
            max_value=2,
            value=int(default["market_shock"])
        )

        n_shocks = st.number_input(
            "Number of Shock Events",
            min_value=1,
            max_value=3,
            value=int(default["n_shocks"])
        )

        shock_in_days = st.number_input(
            "Days to First Shock",
            min_value=1,
            max_value=300,
            value=int(default["shock_in_days"])
        )

    with col2:
        shock_spacing_days = st.number_input(
            "Days Between Shocks",
            min_value=0,
            max_value=365,
            value=int(default["shock_spacing_days"])
        )

        pre_shock_drift = st.slider(
            "Pre-Shock Drift (% p.a.)",
            min_value=-20.0,
            max_value=20.0,
            value=float(default["pre_shock_drift_pa"]*100)
        ) / 100

        post_shock_drift = st.slider(
            "Post-Shock Drift (% p.a.)",
            min_value=-20.0,
            max_value=20.0,
            value=float(default["post_shock_drift_pa"]* 100)
        ) / 100


    st.write(f"Selected market shock: {market_shock}%")

    # =========================
    # Build scenario
    # =========================
    scenario = {
        "market_shock": market_shock,
        "n_shocks": int(n_shocks),
        "shock_in_days": int(shock_in_days),
        "shock_spacing_days": int(shock_spacing_days),
        "pre_shock_drift_pa": float(pre_shock_drift),
        "post_shock_drift_pa": float(post_shock_drift),
    }
    
    
    # =========================
    # Run engine
    # =========================
    engine = ScenarioEngine(
        portfolio=portfolio,
        beta_map=beta_map,
        vol_map=vol_map
    )

    res = engine.run_path_scenario(
        scenario,
        corr_df=corr_df
    )
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
    # Extract results
    # =========================
    product_df = res["product_df"]
    pf_df = res["pf_scenario_per_ccy"]
    cash_df = res["cash_positions"]
    delivered_df = res["delivered_stocks"]
    
    # =========================
    # Format
    # =========================
    product_df = product_df.copy()
    pf_df = pf_df.copy()
    cash_df = cash_df.copy()
    
    product_df["return_pct"] = product_df["return_pct"] * 100
    pf_df["portfolio_return_pct"] = pf_df["portfolio_return_pct"] * 100
    
    product_df = product_df.round(2)
    pf_df = pf_df.round(2)
    cash_df = cash_df.round(2)
    
    if len(delivered_df) > 0:
        delivered_df = delivered_df.copy()
        delivered_df["return_pct"] = delivered_df["return_pct"] * 100
        delivered_df = delivered_df.round(2)
    
    # =========================
    # Portfolio Stress Summary
    # =========================
    st.write("### Portfolio Stress Summary")
    
    st.dataframe(
        pf_df[
            [
                "currency",
                "n_products",
                "underlyings",
                "total_cost",
                "total_payoff",
                "total_pnl",
                "portfolio_return_pct"
            ]
        ].rename(columns={
            "currency": "Currency",
            "n_products": "Number of Products",
            "underlyings": "Underlyings",
            "total_cost": "Total Cost",
            "total_payoff": "Total Payoff",
            "total_pnl": "Total PnL",
            "portfolio_return_pct": "Portfolio Return (%)"
        }),
        width=1200,
        hide_index=True
    )
    st.write("### Stress Testing Results")

    left_col, right_col = st.columns([2, 3])
    
    with left_col:
    
        # =========================
        # Product-Level Results
        # =========================
        st.write("### Product-Level Stress Results")
        
        st.dataframe(
            product_df[
                [
                    "product_id",
                    "currency",
                    "worst_underlying",
                    "settlement_type",
                    "total_payoff",
                    "pnl",
                    "return_pct"
                ]
            ].rename(columns={
                "product_id": "Product ID",
                "currency": "Currency",
                "worst_underlying": "Worst Underlying",
                "settlement_type": "Settlement Type",
                "total_payoff": "Total Payoff",
                "pnl": "PnL",
                "return_pct": "Return (%)"
            }),
            width=1200,
            hide_index=True
        )
        st.write("### Delivered Stocks (Physical Settlement)")
        
        if len(delivered_df) > 0:
            st.dataframe(
                delivered_df.rename(columns={
                    "delivered_underlying": "Delivered Underlying",
                    "delivered_shares": "Delivered Shares",
                    "strike": "Strike",
                    "price": "Final Spot",
                    "currency": "Currency",
                    "fractional_cash": "Fractional Cash",
                    "cash_redemption": "Cash Redemption",
                    "pnl": "PnL",
                    "return_pct": "Return (%)"
                }),
                width=1200,
                hide_index=True
            )
        else:
            st.info("No physical delivery in this scenario.")
        
        # =========================
        # Cash Positions
        # =========================
        st.write("### Cash Positions")
        
        st.dataframe(
            cash_df.rename(columns={
                "currency": "Currency",
                "total_cash": "Total Cash Redemption"
            }),
            width=1200,
            hide_index=True
        )
                
                
        
        
        
    
        
    with right_col:
        st.write("### Underlying Scenario Paths")
        
        fig_paths = px.line(
            path_plot_df,
            x="date",
            y="price",
            color="name",
            line_group="isin",
            color_discrete_sequence=["#2A2F38", "#4A5563", "#7A8797", "#A7B0BC", "#C4CBD4"],
            labels={
                "date": "Date",
                "spot": "Simulated Price",
                "name": "Underlying"
            }
        )
        fig_paths.update_layout(
            height=650,                  
            width=800,                   
            margin=dict(t=20, b=10, l=10, r=10),
            template="plotly_dark"
        )
        st.plotly_chart(fig_paths, width='content')
        
    
        
    