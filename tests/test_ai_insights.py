import json

import numpy as np
import pandas as pd

from app.ai_insights import (
    build_factor_stress_payload,
    build_portfolio_page_payload,
    build_stress_testing_payload,
    payload_hash,
)
from src.portfolio_analytics import PortfolioAnalytics
from src.scenario_engine import ScenarioEngine
from tests.conftest import BETA_MAP, VOL_MAP, SCENARIOS, make_portfolio


def _portfolio_holdings_table(analytics, product_df):
    ref = analytics.reference_currency
    t = analytics.product_df.copy()
    t["cost_ref"] = t.apply(
        lambda r: analytics.convert_to_reference(r["total_cost"], r["currency"]),
        axis=1,
    )
    t["payoff_ref"] = t.apply(
        lambda r: analytics.convert_to_reference(r["total_payoff"], r["currency"]),
        axis=1,
    )
    t["pnl_ref"] = t.apply(
        lambda r: analytics.convert_to_reference(r["pnl"], r["currency"]),
        axis=1,
    )
    fv = product_df[["product_id"]].copy()
    fv["fair_value"] = 100.0
    fv["fair_value_pct"] = 100.0
    t = t.merge(fv, on="product_id", how="left")
    cols = [
        "product_id", "product_type", "underlyings", "currency",
        "maturity_date", "total_notional", "cost_ref", "weight_pct",
        "payoff_ref", "pnl_ref", "return_pct", "distance_to_barrier",
        "fair_value", "fair_value_pct",
    ]
    t = t[cols].copy()
    t["return_pct"] *= 100
    t["maturity_date"] = pd.to_datetime(t["maturity_date"]).dt.strftime("%d %b %Y")
    return t, ref


def test_portfolio_page_payload_is_stable_json_with_no_value_errors():
    portfolio = make_portfolio()
    analytics = PortfolioAnalytics(portfolio)
    product_df = analytics.build_product_analytics()
    holdings, ref = _portfolio_holdings_table(analytics, product_df)
    greeks_df = pd.DataFrame({
        "product_id": ["BRC001"],
        "currency": ["CHF"],
        "isin": ["CH0012221716"],
        "underlying": ["NESN"],
        "delta_1pct": [1.23456789],
    })
    pf_delta = pd.DataFrame({
        "isin": ["CH0012221716"],
        "underlying": ["NESN"],
        "total_delta_1pct": [1.23456789],
    })

    payload_1 = build_portfolio_page_payload(
        holdings_table=holdings,
        reference_currency=ref,
        analytics=analytics,
        product_df=product_df,
        greeks_df=greeks_df,
        pf_delta=pf_delta,
        valuation_date=pd.Timestamp("2026-05-25"),
    )
    payload_2 = build_portfolio_page_payload(
        holdings_table=holdings,
        reference_currency=ref,
        analytics=analytics,
        product_df=product_df,
        greeks_df=greeks_df,
        pf_delta=pf_delta,
        valuation_date=pd.Timestamp("2026-05-25"),
    )

    assert payload_1 == payload_2
    parsed = json.loads(payload_1)
    assert parsed["reference_currency"] == "CHF"
    assert payload_hash(payload_1) == payload_hash(payload_2)
    # Barrier observation style is exposed per holding (and explained in units)
    # so the insight can describe American (continuous) vs European barriers.
    assert "barrier_observation" in parsed["units"]
    assert all("barrier_observation" in h for h in parsed["holdings"])


def test_portfolio_payload_reports_american_barrier_observation():
    portfolio = make_portfolio()
    portfolio.loc[0, "type_style"] = "American"
    analytics = PortfolioAnalytics(portfolio)
    product_df = analytics.build_product_analytics()
    holdings, ref = _portfolio_holdings_table(analytics, product_df)
    greeks_df = pd.DataFrame({"product_id": [], "currency": [], "isin": [],
                              "underlying": [], "delta_1pct": []})
    pf_delta = pd.DataFrame({"isin": [], "underlying": [], "total_delta_1pct": []})

    payload = build_portfolio_page_payload(
        holdings_table=holdings, reference_currency=ref, analytics=analytics,
        product_df=product_df, greeks_df=greeks_df, pf_delta=pf_delta,
        valuation_date=pd.Timestamp("2026-05-25"),
    )
    obs = {h["product_id"]: h["barrier_observation"]
           for h in json.loads(payload)["holdings"]}
    assert obs.get("BRC001") == "American"


def test_stress_testing_payload_handles_array_like_values_without_value_error():
    portfolio = make_portfolio()
    engine = ScenarioEngine(
        portfolio=portfolio,
        beta_map=BETA_MAP,
        vol_map=VOL_MAP,
        risk_free_rates={"CHF": 0.01},
        fx_rates={("CHF", "CHF"): 1.0},
        reference_currency="CHF",
        n_paths=20,
    )
    res = engine.run_path_scenario(SCENARIOS["down_10"])
    # Regression guard: array-like objects inside row dicts used to make
    # pd.isna(...) return an array and crash stable JSON generation.
    res["product_df"]["debug_array"] = None
    res["product_df"].at[0, "debug_array"] = np.array([1.0, np.nan, 3.0])

    payload = build_stress_testing_payload(
        res=res,
        scenario=SCENARIOS["down_10"],
        portfolio=portfolio,
        beta_map=BETA_MAP,
        vol_map=VOL_MAP,
        risk_free_rates={"CHF": 0.01},
        corr_df=pd.DataFrame(),
        selected_preset="Down 10%",
        initial_state="Flat",
        recovery="Slow recovery (~2y)",
    )

    parsed = json.loads(payload)
    assert parsed["simulation"]["n_paths"] == 20
    assert parsed["whole_portfolio_reference_summary"]
    assert payload_hash(payload)


def test_factor_stress_payload_handles_full_factor_context_without_value_error():
    portfolio = make_portfolio()
    product_df = pd.DataFrame({
        "product_id": ["BRC001", "MBRC001"],
        "currency": ["CHF", "CHF"],
        "total_cost": [100_000.0, 100_000.0],
        "worst_underlying": ["NESN", "NOVN"],
        "settlement_type": ["cash", "physical"],
        "barrier_breach_freq": [0.05, 0.35],
        "pnl_mean": [2_500.0, -7_500.0],
        "pnl_median": [3_000.0, -6_000.0],
        "pnl_p5": [-8_000.0, -25_000.0],
        "pnl_p95": [10_000.0, 5_000.0],
        "return_mean_pct": [2.5, -7.5],
        "return_p5_pct": [-8.0, -25.0],
        "mean_cash_redemption": [102_500.0, 85_000.0],
        "pnl_samples": [
            np.array([-8_000.0, 3_000.0, 10_000.0]),
            np.array([-25_000.0, -6_000.0, 5_000.0]),
        ],
    })
    product_df["debug_array"] = None
    product_df.at[0, "debug_array"] = np.array([1.0, np.nan, 3.0])
    dates = pd.date_range("2026-05-25", periods=3)
    factor_paths = {
        "MKT": pd.DataFrame({
            "date": dates, "mean": [100, 95, 92], "median": [100, 94, 90],
            "p5": [100, 88, 80], "p95": [100, 101, 103],
            "lower_1sd": [100, 90, 84], "upper_1sd": [100, 98, 96],
        }),
        "TECH": pd.DataFrame({
            "date": dates, "mean": [100, 90, 82], "median": [100, 88, 80],
            "p5": [100, 75, 62], "p95": [100, 99, 96],
            "lower_1sd": [100, 80, 70], "upper_1sd": [100, 96, 90],
        }),
    }
    asset_paths = {
        "CH0012221716": pd.DataFrame({
            "date": dates, "mean": [100, 97, 95], "median": [100, 96, 94],
            "p5": [100, 88, 84], "p95": [100, 103, 106],
            "lower_1sd": [100, 91, 88], "upper_1sd": [100, 101, 100],
        }),
        "CH0012221717": pd.DataFrame({
            "date": dates, "mean": [100, 85, 78], "median": [100, 84, 76],
            "p5": [100, 70, 60], "p95": [100, 98, 94],
            "lower_1sd": [100, 77, 68], "upper_1sd": [100, 92, 84],
        }),
    }
    res = {
        "product_df": product_df,
        "pf_scenario_per_ccy": pd.DataFrame({
            "currency": ["CHF"],
            "n_products": [2],
            "underlyings": [["NESN", "NOVN"]],
            "total_cost": [200_000.0],
            "pnl_mean": [-5_000.0],
            "pnl_median": [-3_000.0],
            "pnl_p5": [-33_000.0],
            "pnl_p95": [15_000.0],
            "pnl_es5": [-33_000.0],
            "portfolio_return_mean_pct": [-2.5],
            "portfolio_return_p5_pct": [-16.5],
        }),
        "cash_positions": pd.DataFrame({
            "currency": ["CHF"],
            "total_cash": [187_500.0],
        }),
        "delivered_stocks": pd.DataFrame({
            "delivered_underlying": ["NOVN"],
            "total_shares": [100.0],
            "strike": [100.0],
            "price": [76.0],
            "currency": ["CHF"],
            "market_value": [7_600.0],
            "total_fractional_cash": [0.0],
            "total_value_incl_cash": [7_600.0],
            "cost": [10_000.0],
            "pnl": [-2_400.0],
            "return_pct": [-0.24],
        }),
        "asset_paths": asset_paths,
        "factor_paths": factor_paths,
        "pnl_samples_by_ccy": {
            "CHF": np.array([-33_000.0, -3_000.0, 15_000.0]),
        },
        "n_paths": 3,
    }
    loadings = {
        "CH0012221716": {
            "betas": {"MKT": 0.6, "TECH": 0.1},
            "alpha": 0.0,
            "idio_vol": 0.12,
            "r_squared": 0.7,
            "n_obs": 100,
        },
        "CH0012221717": {
            "betas": {"MKT": 0.9, "TECH": 1.4},
            "alpha": 0.0,
            "idio_vol": 0.2,
            "r_squared": 0.8,
            "n_obs": 100,
        },
    }

    payload = build_factor_stress_payload(
        res=res,
        portfolio=portfolio,
        loadings=loadings,
        ui_scenario={
            "initial_market_state": "Bear",
            "events": [{
                "day": 1,
                "factor_shock": {"MKT": -8.0, "TECH": -18.0},
                "recovery": "Slow recovery (~2y)",
            }],
            "idio_intensity": 0.3,
            "mean_reversion_kappa": 0.5,
        },
        preset_name="Tech selloff",
        preset={"label": "Tech selloff", "description": "Technology factor stress"},
        fx_rates={("CHF", "CHF"): 1.0},
        reference_currency="CHF",
        premiums_by_method={
            "mean": pd.DataFrame({"MKT": [-0.01], "TECH": [-0.03]}, index=["Bear"]),
        },
        premium_method="mean",
    )

    parsed = json.loads(payload)
    assert parsed["simulation"]["n_paths"] == 3
    assert parsed["whole_portfolio_reference_summary"][0]["pnl_mean"] == -7000.0
    assert parsed["largest_factor_shocks"][0]["factor"] == "TECH"
    assert parsed["median_path_factor_decomposition"]
    assert payload_hash(payload)


def test_stress_insight_closed_does_not_build_payload(monkeypatch):
    import app.views.stress_testing as stress_view

    monkeypatch.setitem(stress_view.st.session_state, "stress_ai_open", False)

    def boom(**_kwargs):
        raise AssertionError("payload should not be built while insight is closed")

    monkeypatch.setattr(stress_view, "build_stress_testing_payload", boom)
    stress_view._render_stress_ai_insight()


def test_factor_stress_insight_closed_does_not_build_payload(monkeypatch):
    import app.views.factor_stress as factor_view

    monkeypatch.setitem(factor_view.st.session_state, "factor_stress_ai_open", False)

    def boom(**_kwargs):
        raise AssertionError("payload should not be built while insight is closed")

    monkeypatch.setattr(factor_view, "build_factor_stress_payload", boom)
    factor_view._render_factor_stress_ai_insight()


def test_portfolio_insight_closed_does_not_build_payload(monkeypatch):
    import app.views.portfolio as portfolio_view

    monkeypatch.setitem(portfolio_view.st.session_state, "holdings_ai_open", False)

    def boom(**_kwargs):
        raise AssertionError("payload should not be built while insight is closed")

    monkeypatch.setattr(portfolio_view, "_build_holdings_table", boom)
    portfolio_view._render_portfolio_page_insights(
        analytics=None,
        df=pd.DataFrame(),
        greeks_df=pd.DataFrame(),
        pf_delta=pd.DataFrame(),
        valuation_date=None,
    )
