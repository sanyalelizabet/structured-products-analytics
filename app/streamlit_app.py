import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px



# make src importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
from src.reverse_convertible import ReverseConvertible
from src.portfolio_analytics import PortfolioAnalytics
from src.scenario_engine import ScenarioEngine
from src.eod_client import EODClient
from src.market_data_engine import MarketDataEngine


@st.cache_resource
def get_market_engine():
    api_key = st.secrets["EOD_API_KEY"]
    client = EODClient(api_key)
    return MarketDataEngine(client)

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

market_engine = get_market_engine()





st.set_page_config(page_title="Structured Products Dashboard", layout="wide")
st.title("Structured Products Analytics")
st.sidebar.image(str(logo_path), width=160)
view = st.sidebar.radio("View", ["Product", "Portfolio", "Stress Testing"])


# =========================
# Portfolio Input
# =========================

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
    "barrier_breached": True
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
    "barrier_breached": False
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
    "barrier_breached": False
}
isin_ticker_map = {
    "CH0432492467": "ALC.SW",
    "CH0012221716": "ABBN.SW",
    "CH0012214059": "HOLN.SW",
    "CH0012005267": "NOVN.SW",
    "CH0012032048": "ROG.SW",
    "CH0013841017": "LONN.SW",
    "CH0038863350": "NESN.SW",
    "US0090661010": "ABNB",
}


beta_map = {
    "CH0432492467": 0.75,   # ALCON
    "CH0012221716": 0.95,   # ABB
    "CH0012214059": 0.70,   # HOLCIM
    "CH0012005267": 0.55,   # NOVARTIS
    "CH0012032048": 0.25,   # ROCHE
    "CH0013841017": 0.20,   # LONZA
    "CH0038863350": 0.50,   # NESTLE
    "US0090661010": 1.15    # Airbnb
}
vol_map = {
    "CH0432492467": 0.24,   # ALCON
    "CH0012221716": 0.22,   # ABB
    "CH0012214059": 0.18,   # HOLCIM
    "CH0012005267": 0.16,   # NOVARTIS
    "CH0012032048": 0.14,   # ROCHE
    "CH0013841017": 0.28,   # LONZA
    "CH0038863350": 0.12,   # NESTLE
    "US0090661010": 0.35    # Airbnb
}
scenarios = {
    "Current": 0,
    "Down 5%": -5,
    "Down 10%": -10,
    "Crash (-20%)": -20,
    "Up 10%": 10
}

portfolio = pd.DataFrame([p1, p2, p3, p4])

valuation_date  = None
try:
    market_engine.fetch_latest_prices(portfolio)
    portfolio = market_engine.update_spots(portfolio)
    db = market_engine.load_db()
    if not db.empty:
        valuation_date = db["date"].max()
except Exception as e:
    st.warning(f"Could not refresh market prices. Using portfolio default spots. {e}")
    

# =========================
# Build Analytics
# =========================

results = []

for _, row in portfolio.iterrows():
    rc = ReverseConvertible(row, price_db=db)
    res = rc.summary()
    res["product_id"] = row["product_id"]
    results.append(res)

df = pd.DataFrame(results)

# format %
df["return_pa"] *= 100
df["distance_to_barrier"] *= 100

corr_df = market_engine.build_corr_matrix(isin_ticker_map, years=4)


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
        
    st.dataframe(portfolio[overview_cols], width="stretch")

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
        st.dataframe(detail, use_container_width=True, hide_index=True)

    with col_und:
        st.caption("Underlying Metrics")
        st.dataframe(
            und_df,
            use_container_width=True,
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

    analytics = PortfolioAnalytics(portfolio, reference_currency="CHF", price_db=db)
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
            "distance_to_barrier": st.column_config.NumberColumn("Distance to Barrier (%)", format="%.2f"),
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
                "min_distance_to_barrier": "Min Distance to Barrier (%)"
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
                "min_distance_to_barrier": st.column_config.NumberColumn("Min Distance to Barrier (%)", format="%.2f"),
                "avg_distance_to_barrier": st.column_config.NumberColumn("Avg Distance to Barrier (%)", format="%.2f"),
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
        
    
        
    