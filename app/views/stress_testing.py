"""Single-factor Stress Testing view — multi-path Monte Carlo with CRN.

The single-factor engine (CAPM drift + per-asset OU) now runs the same
multi-path simulation as the factor model, with a session-level
:class:`NoiseSampler` for Common Random Numbers.

Layout
------
* Top:    preset selector · n_paths slider · Regenerate button
* Body:   scenario controls (left) · stacked output panes (right)
* Bottom: portfolio summary, product detail, delivered stocks, cash
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import SCENARIO_PRESETS, SCENARIO_CUSTOM_DEFAULT
from src.noise_sampler import NoiseSampler
from src.scenario_engine import ScenarioEngine


_ASSET_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#86BCB6",
]
_SHOCK_COLOR = "#C0504D"

_SAMPLER_KEY = "single_factor_noise_sampler"


def render(portfolio, corr_df, beta_map, vol_map_implied, vol_map_realised,
           risk_free_rates, fx_rates=None, reference_currency=None):
    st.subheader("Stress Testing — Single-Factor Path Engine (multi-path MC)")
    st.caption(
        "Path defined as drift → shock(s) → drift to maturity, vectorised "
        "across paths.  A session-cached NoiseSampler keeps the underlying "
        "noise stable across slider nudges (Common Random Numbers)."
    )

    # ── Top control row ──────────────────────────────────────────────────
    col_preset, col_paths, col_regen = st.columns([3, 2, 1])

    with col_preset:
        selected_preset = st.selectbox("Scenario Preset", list(SCENARIO_PRESETS.keys()))
    preset  = SCENARIO_PRESETS[selected_preset]
    default = preset if preset is not None else SCENARIO_CUSTOM_DEFAULT

    # Always use realised (historical) vol — toggle removed.
    vol_map = vol_map_realised

    with col_paths:
        n_paths = st.slider(
            "Paths", min_value=1, max_value=500,
            value=100, step=10,
            help="Monte Carlo paths. Higher = tighter mean estimate, slower run.",
        )

    with col_regen:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        regen_clicked = st.button("Regenerate", help="Draw a fresh noise sample.")

    # ── Scenario controls ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        market_shock = st.slider(
            "Market Shock per Event (%)", min_value=-25, max_value=2,
            value=int(default["market_shock"]),
        )
        n_shocks = st.number_input(
            "Number of Shock Events", min_value=1, max_value=3,
            value=int(default["n_shocks"]),
        )
        shock_in_days = st.number_input(
            "Days to First Shock", min_value=1, max_value=300,
            value=int(default["shock_in_days"]),
        )

    with col2:
        shock_spacing_days = st.number_input(
            "Days Between Shocks", min_value=0, max_value=365,
            value=int(default["shock_spacing_days"]),
        )
        pre_shock_drift = st.slider(
            "Pre-Shock Drift (% p.a.)", min_value=-20.0, max_value=20.0,
            value=float(default["pre_shock_drift_pa"] * 100),
        ) / 100
        post_shock_drift = st.slider(
            "Post-Shock Drift (% p.a.)", min_value=-20.0, max_value=20.0,
            value=float(default["post_shock_drift_pa"] * 100),
        ) / 100

    scenario = {
        "market_shock":       market_shock,
        "n_shocks":           int(n_shocks),
        "shock_in_days":      int(shock_in_days),
        "shock_spacing_days": int(shock_spacing_days),
        "pre_shock_drift_pa": float(pre_shock_drift),
        "post_shock_drift_pa": float(post_shock_drift),
    }


    # ── Session-level NoiseSampler (CRN) ─────────────────────────────────
    sampler = _get_or_make_sampler(portfolio, n_paths, regen_clicked)

    # ── Run engine ───────────────────────────────────────────────────────
    engine = ScenarioEngine(
        portfolio=portfolio, beta_map=beta_map, vol_map=vol_map,
        risk_free_rates=risk_free_rates,
        n_paths=n_paths, noise_sampler=sampler,
        fx_rates=fx_rates, reference_currency=reference_currency,
    )
    res = engine.run_path_scenario(scenario, corr_df=corr_df)
    st.session_state[_SAMPLER_KEY] = engine.noise_sampler

    # ── Output ───────────────────────────────────────────────────────────
    st.markdown("---")
    _render_portfolio_summary(res)
    _render_portfolio_summary_ref(res)

    left_col, right_col = st.columns([2, 3])

    with left_col:
        _render_product_detail(res)
        _render_delivered_stocks(res)
        _render_cash_positions(res)

    with right_col:
        _render_asset_paths_fan(
            asset_paths=res["asset_paths"],
            portfolio=portfolio,
            scenario=scenario,
        )
        _render_engine_inputs_table(
            portfolio=portfolio,
            beta_map=beta_map,
            vol_map=vol_map,
            risk_free_rates=risk_free_rates,
            market_shock=market_shock,
        )
        _render_pnl_distribution(res["pnl_samples_by_ccy"])
        _render_pnl_distribution_ref(res)


# ──────────────────────────────────────────────────────────────────────────
# NoiseSampler in session state
# ──────────────────────────────────────────────────────────────────────────

def _get_or_make_sampler(portfolio, n_paths, regen_clicked):
    today              = pd.Timestamp.today().normalize()
    portfolio_maturity = pd.to_datetime(portfolio["maturity_date"]).max()
    n_days             = len(pd.bdate_range(start=today, end=portfolio_maturity))
    isins = sorted({i for _, r in portfolio.iterrows() for i in r["underlying_isins"]})

    sampler: NoiseSampler | None = st.session_state.get(_SAMPLER_KEY)

    # Single-factor engine uses no factor block → empty factor universe in sampler.
    if sampler is None or not sampler.matches(n_paths, n_days, [], isins):
        sampler = NoiseSampler(n_paths=n_paths, n_days=n_days,
                               factor_codes=[], isins=isins)
    elif regen_clicked:
        sampler.regenerate()

    st.session_state[_SAMPLER_KEY] = sampler
    return sampler


# ──────────────────────────────────────────────────────────────────────────
# Plot / table helpers
# ──────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _fan_band_traces(df, color, name):
    """Two transparent traces forming a median ± 1σ band, both grouped
    with the asset's median line so legend clicks toggle the whole asset."""
    return [
        go.Scatter(
            x=df["date"], y=df["upper_1sd"],
            mode="lines", line=dict(width=0),
            legendgroup=name, showlegend=False, hoverinfo="skip",
        ),
        go.Scatter(
            x=df["date"], y=df["lower_1sd"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_hex_to_rgba(color, 0.18),
            legendgroup=name, showlegend=False, hoverinfo="skip",
        ),
    ]


def _fan_median_trace(df, color, name):
    return go.Scatter(
        x=df["date"], y=df["median"],
        mode="lines", name=name,
        line=dict(color=color, width=2),
        legendgroup=name, showlegend=True,
        hovertemplate=(
            f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>"
            "Median: %{y:.1f}<extra></extra>"
        ),
    )


def _render_asset_paths_fan(asset_paths: dict, portfolio, scenario):
    st.markdown("### Underlying Scenario Paths (median ± 1σ)")

    isin_to_name   = {}
    isin_to_strike = {}
    for _, prow in portfolio.iterrows():
        for isin, name in zip(prow["underlying_isins"], prow["underlyings"]):
            isin_to_name[isin] = name
        for isin, strike in zip(prow["underlying_isins"], prow["strike"]):
            isin_to_strike[isin] = strike

    today = pd.Timestamp.today().normalize()
    shock_dates = [
        today + pd.Timedelta(days=int(scenario["shock_in_days"])
                                  + i * int(scenario["shock_spacing_days"]))
        for i in range(int(scenario["n_shocks"]))
    ]

    # Pre-normalise once per asset
    assets = []
    for idx, (isin, df) in enumerate(asset_paths.items()):
        name  = isin_to_name.get(isin, isin)
        color = _ASSET_PALETTE[idx % len(_ASSET_PALETTE)]
        spot0 = float(df["mean"].iloc[0])
        norm = pd.DataFrame({
            "date":      df["date"],
            "median":    df["median"]    / spot0 * 100,
            "p5":        df["p5"]        / spot0 * 100,
            "p95":       df["p95"]       / spot0 * 100,
            "lower_1sd": df["lower_1sd"] / spot0 * 100,
            "upper_1sd": df["upper_1sd"] / spot0 * 100,
        })
        assets.append((isin, name, color, norm, spot0))

    fig = go.Figure()
    # Pass 1: all bands
    for _, name, color, norm, _ in assets:
        for tr in _fan_band_traces(norm, color, name):
            fig.add_trace(tr)
    # Pass 2: medians on top
    for _, name, color, norm, _ in assets:
        fig.add_trace(_fan_median_trace(norm, color, name))

    # Strike lines
    for isin, name, color, _, spot0 in assets:
        strike = isin_to_strike.get(isin)
        if strike is not None:
            norm_strike = strike / spot0 * 100
            fig.add_hline(
                y=norm_strike,
                line=dict(color=color, dash="dash", width=1),
                opacity=0.45,
                annotation_text=f"{name} strike",
                annotation_position="right",
                annotation_font=dict(color=color, size=10),
            )

    fig.add_hline(y=100, line=dict(color="grey", dash="dot", width=1), opacity=0.4)
    for i, sd in enumerate(shock_dates):
        fig.add_vline(
            x=sd.timestamp() * 1000,
            line=dict(color=_SHOCK_COLOR, dash="dash", width=1.5),
            opacity=0.8,
            annotation_text=f"Shock {i+1}: {scenario['market_shock']:+}%",
            annotation_position="top right" if i % 2 == 0 else "top left",
            annotation_font=dict(color=_SHOCK_COLOR, size=10),
        )

    fig.update_layout(
        template="plotly_dark", height=560,
        margin=dict(t=40, b=40, l=40, r=120),
        xaxis_title="Date",
        yaxis_title="Normalised Price (base 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def _render_engine_inputs_table(portfolio, beta_map, vol_map, risk_free_rates,
                                 market_shock):
    """Show the β / σ / r_f and the implied β-scaled shock per underlying.

    This is the diagnostic that explains "why is asset X moving like that"
    — if a β is unexpectedly small or has the wrong sign, the per-event
    shock factor here will surface it immediately.
    """
    st.markdown("### Engine Inputs per Underlying (β · σ · shock %)")
    rows = []
    seen = set()
    for _, prow in portfolio.iterrows():
        rf = float(risk_free_rates.get(prow["currency"], 0.0))
        for isin, name in zip(prow["underlying_isins"], prow["underlyings"]):
            if isin in seen:
                continue
            seen.add(isin)
            beta = float(beta_map.get(isin, 1.0))
            vol  = float(vol_map.get(isin, 0.15))
            # β-scaled shock per event = market_shock × β  (in %)
            scaled_shock = float(market_shock) * beta
            rows.append({
                "Underlying": name,
                "ISIN":       isin,
                "Currency":   prow["currency"],
                "β":          round(beta, 3),
                "σ (annual)": round(vol,  3),
                "r_f":        round(rf,   4),
                "Per-event shock × β (%)": round(scaled_shock, 2),
            })
    df = pd.DataFrame(rows)

    # Highlight any β that's suspicious (negative, near-zero, or > 2)
    def _flag(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v < 0:
            return "color: #C0504D; font-weight: 600;"      # negative β — flips shock sign!
        if v < 0.3:
            return "color: #C9A961;"                          # very low β — small response
        if v > 2.0:
            return "color: #C0504D; font-weight: 600;"      # implausibly high
        return ""

    styled = df.style.map(_flag, subset=["β"]).format({
        "β":           "{:+.3f}",
        "σ (annual)":  "{:.2%}",
        "r_f":         "{:.2%}",
        "Per-event shock × β (%)": "{:+.2f}",
    })
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption(
        "If a β is negative or near zero, a *negative* market shock can map to "
        "a *positive* (or near-zero) effect on that underlying — the OU drift / "
        "post-shock recovery then dominates. Red = β < 0 or > 2. Gold = |β| < 0.3."
    )


def _render_pnl_distribution(samples_by_ccy: dict):
    if not samples_by_ccy:
        return

    st.markdown("### Portfolio P&L Distribution")
    for ccy, samples in samples_by_ccy.items():
        mean   = float(samples.mean())
        median = float(np.median(samples))
        p5     = float(np.percentile(samples, 5))
        p95    = float(np.percentile(samples, 95))
        es5    = float(samples[samples <= p5].mean()) if len(samples) >= 20 else float(samples.min())

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples, nbinsx=30, marker_color="#4E79A7", opacity=0.85,
        ))
        for label, value, color in [
            ("Mean",   mean,   "#EDC948"),
            ("Median", median, "#76B7B2"),
            ("5%",     p5,     _SHOCK_COLOR),
            ("95%",    p95,    "#59A14F"),
        ]:
            fig.add_vline(
                x=value,
                line=dict(color=color, width=1.5, dash="dash"),
                annotation_text=f"{label}: {value:,.0f}",
                annotation_position="top",
                annotation_font=dict(color=color, size=10),
            )
        fig.update_layout(
            template="plotly_dark", height=320,
            margin=dict(t=50, b=40, l=40, r=20),
            xaxis_title=f"Portfolio P&L ({ccy})",
            yaxis_title="Number of paths",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean P&L",   f"{mean:,.0f}")
        c2.metric("Median P&L", f"{median:,.0f}")
        c3.metric("5% P&L",     f"{p5:,.0f}")
        c4.metric("95% P&L",    f"{p95:,.0f}")
        c5.metric("ES (worst 5%)", f"{es5:,.0f}")


# ──────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────

def _render_portfolio_summary_ref(res):
    """Reference-currency portfolio summary (item 4).  Skipped silently
    when no FX context was passed to the engine."""
    pf_ref = res.get("pf_scenario_ref")
    if pf_ref is None or len(pf_ref) == 0:
        return
    ccy = res.get("reference_currency", "")
    st.markdown(f"### Portfolio Stress Summary — {ccy} (reference currency)")
    st.dataframe(
        pf_ref.round(2).rename(columns={
            "reference_currency":           "Reference CCY",
            "n_currencies":                 "# Currencies",
            "total_cost_ref":               f"Total Cost ({ccy})",
            "pnl_mean":                     "PnL Mean",
            "pnl_median":                   "PnL Median",
            "pnl_p5":                       "PnL 5%",
            "pnl_p95":                      "PnL 95%",
            "pnl_es5":                      "ES (worst 5%)",
            "portfolio_return_mean_pct":    "Return Mean (%)",
            "portfolio_return_p5_pct":      "Return 5% (%)",
        }),
        width="stretch", hide_index=True,
    )


def _render_pnl_distribution_ref(res):
    """Histogram + metrics for the per-path total P&L in the reference
    currency (item 4)."""
    samples = res.get("pnl_samples_ref")
    if samples is None or len(samples) == 0:
        return
    ccy = res.get("reference_currency", "")

    st.markdown(f"### Portfolio P&L Distribution — {ccy} (reference currency)")
    mean   = float(samples.mean())
    median = float(np.median(samples))
    p5     = float(np.percentile(samples, 5))
    p95    = float(np.percentile(samples, 95))
    es5    = (float(samples[samples <= p5].mean())
              if len(samples) >= 20 else float(samples.min()))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=samples, nbinsx=30, marker_color="#76B7B2", opacity=0.85))
    for label, value, color in [
        ("Mean",   mean,   "#EDC948"),
        ("Median", median, "#4E79A7"),
        ("5%",     p5,     _SHOCK_COLOR),
        ("95%",    p95,    "#59A14F"),
    ]:
        fig.add_vline(
            x=value,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=f"{label}: {value:,.0f}",
            annotation_position="top",
            annotation_font=dict(color=color, size=10),
        )
    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(t=50, b=40, l=40, r=20),
        xaxis_title=f"Portfolio P&L ({ccy})",
        yaxis_title="Number of paths",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean P&L",       f"{mean:,.0f}")
    c2.metric("Median P&L",     f"{median:,.0f}")
    c3.metric("5% P&L",         f"{p5:,.0f}")
    c4.metric("95% P&L",        f"{p95:,.0f}")
    c5.metric("ES (worst 5%)",  f"{es5:,.0f}")


def _render_portfolio_summary(res):
    st.markdown("### Portfolio Stress Summary")
    pf = res["pf_scenario_per_ccy"].copy().round(2)
    st.dataframe(
        pf[[
            "currency", "n_products", "underlyings", "total_cost",
            "pnl_mean", "pnl_median", "pnl_p5", "pnl_p95", "pnl_es5",
            "portfolio_return_mean_pct", "portfolio_return_p5_pct",
        ]].rename(columns={
            "currency":                    "Currency",
            "n_products":                  "Products",
            "underlyings":                 "Worst Underlyings",
            "total_cost":                  "Total Cost",
            "pnl_mean":                    "PnL Mean",
            "pnl_median":                  "PnL Median",
            "pnl_p5":                      "PnL 5%",
            "pnl_p95":                     "PnL 95%",
            "pnl_es5":                     "ES (worst 5%)",
            "portfolio_return_mean_pct":   "Return Mean (%)",
            "portfolio_return_p5_pct":     "Return 5% (%)",
        }),
        width="stretch", hide_index=True,
    )


def _render_product_detail(res):
    st.markdown("### Product-Level Stress Results")
    df = res["product_df"].copy().round(2)
    st.dataframe(
        df[[
            "product_id", "currency", "worst_underlying", "settlement_type",
            "barrier_breach_freq",
            "pnl_mean", "pnl_median", "pnl_p5", "pnl_p95",
            "return_mean_pct", "return_p5_pct",
        ]].rename(columns={
            "product_id":          "Product ID",
            "currency":            "Currency",
            "worst_underlying":    "Worst (mode)",
            "settlement_type":     "Settlement (mode)",
            "barrier_breach_freq": "Barrier Breach Freq",
            "pnl_mean":            "PnL Mean",
            "pnl_median":          "PnL Median",
            "pnl_p5":              "PnL 5%",
            "pnl_p95":             "PnL 95%",
            "return_mean_pct":     "Return Mean (%)",
            "return_p5_pct":       "Return 5% (%)",
        }),
        width="stretch", hide_index=True,
    )


def _render_delivered_stocks(res):
    st.markdown("### Delivered Stocks (mean across paths)")
    delivered = res["delivered_stocks"]
    if delivered is None or len(delivered) == 0:
        st.info("No physical delivery in this scenario.")
        return
    df = delivered.copy()
    df["return_pct"] = df["return_pct"] * 100
    # Format delivery date for readability
    if "final_delivery_date" in df.columns:
        df["final_delivery_date"] = pd.to_datetime(
            df["final_delivery_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    df = df.round(2)
    st.dataframe(
        df.rename(columns={
            "delivered_underlying":  "Delivered Underlying",
            "total_shares":          "Total Shares (mean)",
            "strike":                "Strike (mean)",
            "price":                 "Final Spot (mean)",
            "currency":              "Currency",
            "market_value":          "Market Value",
            "total_fractional_cash": "Fractional Cash",
            "total_value_incl_cash": "Total Value",
            "cost":                  "Cost",
            "pnl":                   "PnL",
            "return_pct":            "Return (%)",
            "final_delivery_date":   "Final Delivery Date",
        }),
        width="stretch", hide_index=True,
    )


def _render_cash_positions(res):
    st.markdown("### Cash Positions (mean across paths)")
    cash = res["cash_positions"].copy().round(2)
    st.dataframe(
        cash.rename(columns={
            "currency":   "Currency",
            "total_cash": "Mean Cash Redemption",
        }),
        width="stretch", hide_index=True,
    )
