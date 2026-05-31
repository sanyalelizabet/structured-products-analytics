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

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import SCENARIO_PRESETS, SCENARIO_CUSTOM_DEFAULT
from app.formatting import chf
from app.views._layout import fit_height as _fit_height
from app.ai_insights import (
    build_stress_testing_payload,
    generate_stress_testing_insight,
    payload_hash,
)
from src.numerics.noise_sampler import NoiseSampler
from src.risk.scenario_archetypes import (
    DEFAULT_INITIAL_MARKET_STATE,
    DEFAULT_RECOVERY_ARCHETYPE,
    EVENT_RECOVERY_ARCHETYPES,
    INITIAL_MARKET_STATES,
    event_drift_for_factor,
    initial_drift_dict,
)
from src.risk.scenario_engine import ScenarioEngine


# Palette — kept in sync with ``app/views/portfolio.py`` so the whole app
# speaks one visual language.
_PRIMARY      = "#4E79A7"
_POSITIVE     = "#76A65A"
_NEGATIVE     = "#C0504D"
_NEUTRAL_GREY = "#7A8797"
_WARNING      = "#C9A961"
_ACCENT_DEEP  = "#2A3F5F"

_ASSET_PALETTE = [
    _PRIMARY, _POSITIVE, "#9C755F", _WARNING, _ACCENT_DEEP, _NEUTRAL_GREY,
    "#3FA29E", "#8E7CC3", "#A35F4A", "#5E8A6E", "#7E91AB",
]
_SHOCK_COLOR = _NEGATIVE

_SAMPLER_KEY = "single_factor_noise_sampler"

log = logging.getLogger(__name__)


def render(portfolio, corr_df, beta_map, vol_map_implied, vol_map_realised,
           risk_free_rates, fx_rates=None, reference_currency=None):
    """``fx_rates`` and ``reference_currency`` are accepted for API
    compatibility with the streamlit_app routing layer."""
    st.subheader("Stress Testing")

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
            help="Magnitude of each discrete shock applied to the market index.",
        )
        n_shocks = st.number_input(
            "Number of Shock Events", min_value=1, max_value=3,
            value=int(default["n_shocks"]),
        )
        shock_in_days = st.number_input(
            "Days to First Shock", min_value=1, max_value=300,
            value=int(default["shock_in_days"]),
        )
        shock_spacing_days = st.number_input(
            "Days Between Shocks", min_value=0, max_value=365,
            value=int(default["shock_spacing_days"]),
        )

    with col2:
        # Initial market state — same vocabulary as Factor Stress.
        init_options = list(INITIAL_MARKET_STATES.keys())
        init_default = default.get("initial_market_state", DEFAULT_INITIAL_MARKET_STATE)
        initial_state = st.selectbox(
            "Initial market state",
            options=init_options,
            index=init_options.index(init_default) if init_default in init_options
                  else init_options.index(DEFAULT_INITIAL_MARKET_STATE),
            help="Market behaviour BEFORE the first shock.",
        )

        # Recovery archetype — coupled to the market shock magnitude.
        rec_options = list(EVENT_RECOVERY_ARCHETYPES.keys())
        rec_default = default.get("recovery", DEFAULT_RECOVERY_ARCHETYPE)
        recovery = st.selectbox(
            "Recovery (after final shock)",
            options=rec_options,
            index=rec_options.index(rec_default) if rec_default in rec_options
                  else rec_options.index(DEFAULT_RECOVERY_ARCHETYPE),
            help=("How the market behaves AFTER the final shock.  Speed is "
                  "*coupled* to the shock magnitude — a bigger shock + Fast "
                  "recovery produces a steeper rebound (V-shape)."),
        )

    # ── Translate archetypes → numerical drifts the engine expects ────────
    pre_shock_drift   = initial_drift_dict(initial_state, ["MKT"])["MKT"]
    post_shock_drift  = event_drift_for_factor(market_shock, recovery)
    # Recovery archetype carries its own horizon (years).  After that, the
    # market reverts to the initial market state — preventing the recovery
    # drift from running forever and producing extreme overshoots.
    _, recovery_horizon_years = EVENT_RECOVERY_ARCHETYPES[recovery]
    post_recovery_drift = pre_shock_drift   # back to "normal"

    # Show the user the implied numerical drifts (transparency).
    st.caption(
        f"→ Implied drift: pre-shock **{pre_shock_drift*100:+.1f} %/y**, "
        f"recovery **{post_shock_drift*100:+.1f} %/y** for "
        f"~{recovery_horizon_years:.2f} y, then back to "
        f"**{post_recovery_drift*100:+.1f} %/y**."
    )

    scenario = {
        "market_shock":           market_shock,
        "n_shocks":               int(n_shocks),
        "shock_in_days":          int(shock_in_days),
        "shock_spacing_days":     int(shock_spacing_days),
        "pre_shock_drift_pa":     float(pre_shock_drift),
        "post_shock_drift_pa":    float(post_shock_drift),
        "recovery_horizon_years": float(recovery_horizon_years),
        "post_recovery_drift_pa": float(post_recovery_drift),
    }

    # ── Session-level NoiseSampler (CRN) ─────────────────────────────────
    sampler = _get_or_make_sampler(portfolio, n_paths, regen_clicked)

    # ── Run engine ───────────────────────────────────────────────────────
    engine = ScenarioEngine(
        portfolio=portfolio, beta_map=beta_map, vol_map=vol_map,
        risk_free_rates=risk_free_rates,
        fx_rates=fx_rates,
        reference_currency=reference_currency,
        n_paths=n_paths, noise_sampler=sampler,
    )
    res = engine.run_path_scenario(scenario, corr_df=corr_df)
    st.session_state[_SAMPLER_KEY] = engine.noise_sampler

    # ── Output ───────────────────────────────────────────────────────────
    st.markdown("---")
    _render_stress_ai_insight(
        res=res,
        scenario=scenario,
        portfolio=portfolio,
        beta_map=beta_map,
        vol_map=vol_map,
        risk_free_rates=risk_free_rates,
        corr_df=corr_df,
        selected_preset=selected_preset,
        initial_state=initial_state,
        recovery=recovery,
    )
    _render_portfolio_summary(res)

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
    """Two transparent traces forming a ±1σ band, both grouped
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
    st.dataframe(styled, width="stretch", hide_index=True,
                 height=_fit_height(len(df)))
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
                annotation_text=f"{label}: {chf(value, 2)}",
                annotation_position="top",
                annotation_font=dict(color=color, size=10),
            )
        fig.update_layout(
            template="plotly_dark", height=320,
            margin=dict(t=50, b=40, l=40, r=20),
            xaxis_title=f"Portfolio P&L ({ccy})",
            yaxis_title="Number of paths",
            showlegend=False,
            separators=".'",
        )
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean P&L",   chf(mean, 2))
        c2.metric("Median P&L", chf(median, 2))
        c3.metric("5% P&L",     chf(p5, 2))
        c4.metric("95% P&L",    chf(p95, 2))
        c5.metric("ES (worst 5%)", chf(es5, 2))


# ──────────────────────────────────────────────────────────────────────────
# Gemini insight display
# ──────────────────────────────────────────────────────────────────────────

def _render_stress_ai_insight(**kwargs):
    open_key = "stress_ai_open"
    is_open = bool(st.session_state.get(open_key, False))
    label = "Hide Gemini AI insight" if is_open else "Show Gemini AI insight"
    if st.button(label, key="stress_ai_toggle"):
        st.session_state[open_key] = not is_open
        st.rerun()

    if not st.session_state.get(open_key, False):
        return

    insight_payload = build_stress_testing_payload(**kwargs)
    hash_key = payload_hash(insight_payload)

    with st.container(border=True):
        st.caption("Gemini AI insight")
        if st.session_state.get("stress_ai_hash") != hash_key:
            with st.spinner("Generating Gemini AI insight..."):
                try:
                    st.session_state["stress_ai_insight"] = generate_stress_testing_insight(
                        insight_payload,
                    )
                    st.session_state["stress_ai_hash"] = hash_key
                except Exception:  # noqa: BLE001
                    log.exception("Stress Testing Gemini insight generation failed")
                    st.error("Gemini AI insight is temporarily unavailable.")
                    return

        insight = st.session_state.get("stress_ai_insight")
        if insight:
            st.markdown(insight)


def _render_portfolio_summary(res):
    st.markdown("### Portfolio Stress Summary")
    ref = res.get("pf_scenario_ref")
    if ref is not None and not ref.empty:
        ref_df = ref.copy().round(2)
        ref_df = ref_df[[
            "reference_currency", "n_currencies", "total_cost_ref",
            "pnl_mean", "pnl_median", "pnl_p5", "pnl_p95", "pnl_es5",
            "portfolio_return_mean_pct", "portfolio_return_p5_pct",
        ]].rename(columns={
            "reference_currency":          "Reference",
            "n_currencies":                "Currencies",
            "total_cost_ref":              "Total Cost",
            "pnl_mean":                    "PnL Mean",
            "pnl_median":                  "PnL Median",
            "pnl_p5":                      "PnL 5%",
            "pnl_p95":                     "PnL 95%",
            "pnl_es5":                     "ES (worst 5%)",
            "portfolio_return_mean_pct":   "Return Mean (%)",
            "portfolio_return_p5_pct":     "Return 5% (%)",
        })
        st.markdown("**Whole portfolio**")
        money = {"Total Cost": False, "PnL Mean": True, "PnL Median": True,
                 "PnL 5%": True, "PnL 95%": True, "ES (worst 5%)": True}
        styled_ref = ref_df.style.format(
            {col: (lambda v, s=signed: chf(v, 2, signed=s))
             for col, signed in money.items() if col in ref_df.columns}
        )
        st.dataframe(
            styled_ref,
            width="stretch", hide_index=True,
            height=_fit_height(len(ref_df)),
        )

    st.markdown("**By currency**")
    pf = res["pf_scenario_per_ccy"].copy().round(2)
    pf = pf[[
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
    })
    money = {"Total Cost": False, "PnL Mean": True, "PnL Median": True,
             "PnL 5%": True, "PnL 95%": True, "ES (worst 5%)": True}
    styled = pf.style.format(
        {col: (lambda v, s=signed: chf(v, 2, signed=s))
         for col, signed in money.items() if col in pf.columns}
    )
    st.dataframe(
        styled,
        width="stretch", hide_index=True,
        height=_fit_height(len(pf)),
    )


def _render_product_detail(res):
    st.markdown("### Product-Level Stress Results")
    df = res["product_df"].copy().round(2)
    df = df[[
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
    })
    styled = df.style.format(
        {col: (lambda v: chf(v, 2, signed=True))
         for col in ["PnL Mean", "PnL Median", "PnL 5%", "PnL 95%"]
         if col in df.columns}
    )
    st.dataframe(
        styled,
        width="stretch", hide_index=True,
        height=_fit_height(len(df)),
    )


def _render_delivered_stocks(res):
    st.markdown("### Delivered Stocks (mean across paths)")
    delivered = res["delivered_stocks"]
    if delivered is None or len(delivered) == 0:
        st.info("No physical delivery in this scenario.")
        return
    df = delivered.copy()
    df["return_pct"] = df["return_pct"] * 100
    df = df.round(2)
    df = df.rename(columns={
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
    })
    money = {"Market Value": False, "Fractional Cash": False,
             "Total Value": False, "Cost": False, "PnL": True}
    styled = df.style.format(
        {col: (lambda v, s=signed: chf(v, 2, signed=s))
         for col, signed in money.items() if col in df.columns}
    )
    st.dataframe(
        styled,
        width="stretch", hide_index=True,
        height=_fit_height(len(df)),
    )


def _render_cash_positions(res):
    st.markdown("### Cash Positions (mean across paths)")
    cash = res["cash_positions"].copy().round(2)
    cash = cash.rename(columns={
        "currency":   "Currency",
        "total_cash": "Mean Cash Redemption",
    })
    styled = cash.style.format(
        {"Mean Cash Redemption": lambda v: chf(v, 2)}
    )
    st.dataframe(
        styled,
        width="stretch", hide_index=True,
        height=_fit_height(len(cash)),
    )
