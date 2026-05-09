"""Factor Stress Testing view — multi-factor structural Monte Carlo engine.

The user thinks in **scenarios as event timelines**: an initial market
state, plus a series of dated events.  Each event has per-factor shocks
and a *Recovery* archetype that, coupled with the shock magnitude,
defines the drift afterwards.  The engine translates this into numerical
drifts and runs a vectorised multi-path simulation with Common Random
Numbers (cached :class:`NoiseSampler` in session state).

Layout
------
* Top:    preset · λ · paths · Regenerate
* Left:   Initial market state + Events table + κ
* Right:  Factor paths · Asset paths · P&L distribution · Decomposition · Loadings
* Bottom: portfolio summary · product detail · delivered · cash
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
from src.factor_engine import FACTORS
from src.factor_scenario_engine import FactorScenarioEngine
from src.noise_sampler import NoiseSampler
from src.scenario_archetypes import (
    DEFAULT_INITIAL_MARKET_STATE,
    DEFAULT_RECOVERY_ARCHETYPE,
    EVENT_RECOVERY_ARCHETYPES,
    INITIAL_MARKET_STATES,
)


# ──────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────

# Palette — kept in sync with ``app/views/portfolio.py`` so the whole app
# speaks one visual language.  The factor colours pull from the shared
# constants where the meaning matches (MKT = primary, HC = positive/sage,
# FX = warning/gold) and use distinct accents for TECH / FIN / ENERGY.
_PRIMARY      = "#4E79A7"
_POSITIVE     = "#76A65A"
_NEGATIVE     = "#C0504D"
_NEUTRAL_GREY = "#7A8797"
_WARNING      = "#C9A961"
_ACCENT_DEEP  = "#2A3F5F"

_FACTOR_COLORS = {
    "MKT":    _PRIMARY,    # steel blue — the market reference
    "TECH":   "#3FA29E",   # muted teal
    "HC":     _POSITIVE,   # sage — defensive
    "FIN":    "#8E7CC3",   # slate-lavender
    "ENERGY": "#9C755F",   # warm brown — commodity
    "FX":     _WARNING,    # muted gold
}

_ASSET_PALETTE = [
    _PRIMARY, _POSITIVE, "#9C755F", _WARNING, _ACCENT_DEEP, _NEUTRAL_GREY,
    "#3FA29E", "#8E7CC3", "#A35F4A", "#5E8A6E", "#7E91AB",
]

_SHOCK_COLOR = _NEGATIVE

_SAMPLER_KEY = "factor_noise_sampler"


# ──────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────

def render(portfolio, loadings, factor_engine, risk_free_rates,
           fx_rates=None, reference_currency=None):
    """Render the Factor Stress tab.

    ``fx_rates`` and ``reference_currency`` are accepted for API
    compatibility with the streamlit_app routing layer (which forwards
    them from PortfolioAnalytics).  Currently used for any future
    cross-currency consolidation of the bottom tables.
    """
    _ = fx_rates, reference_currency   # reserved for cross-ccy consolidation
    st.subheader("Factor Stress Testing")

    # ── Top control row ──────────────────────────────────────────────────
    col_preset, col_lambda, col_paths, col_regen = st.columns([3, 2, 2, 1])

    with col_preset:
        preset_name = st.selectbox(
            "Scenario Preset", list(FACTOR_SCENARIO_PRESETS.keys()),
        )
    preset = FACTOR_SCENARIO_PRESETS[preset_name]

    with col_lambda:
        idio_intensity = st.slider(
            "Idio intensity λ",
            min_value=0.0, max_value=1.0,
            value=float(preset.get("idio_intensity", 0.3)),
            step=0.05,
            help=("0 = deterministic factor projection · "
                  "0.3 = recommended damped Monte Carlo · "
                  "1.0 = full historical idio noise"),
        )

    with col_paths:
        n_paths = st.slider(
            "Paths", min_value=1, max_value=500, value=100, step=10,
            help="Number of Monte Carlo paths.",
        )

    with col_regen:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        regen_clicked = st.button("Regenerate", help="Draw a fresh noise sample.")

    st.info(f"**{preset['label']}** — {preset['description']}")

    # ── Two-column control / output layout ───────────────────────────────
    col_left, col_right = st.columns([2, 3])

    with col_left:
        ui_scenario = _render_scenario_builder(preset, preset_name, idio_intensity)

    # ── NoiseSampler (CRN) in session state ──────────────────────────────
    sampler = _get_or_make_sampler(
        portfolio=portfolio,
        n_paths=n_paths,
        regen_clicked=regen_clicked,
    )

    # ── Run engine ───────────────────────────────────────────────────────
    engine = FactorScenarioEngine(
        portfolio=portfolio,
        loadings=loadings,
        factor_engine=factor_engine,
        risk_free_rates=risk_free_rates,
        idio_intensity=ui_scenario["idio_intensity"],
        mean_reversion_kappa=ui_scenario["mean_reversion_kappa"],
        n_paths=n_paths,
        noise_sampler=sampler,
    )
    res = engine.run_path_scenario(ui_scenario)
    st.session_state[_SAMPLER_KEY] = engine.noise_sampler

    with col_right:
        _render_output_tabs(res, portfolio, loadings, ui_scenario)

    # ── Portfolio-level tables ───────────────────────────────────────────
    st.markdown("---")
    _render_portfolio_summary(res)
    _render_product_detail(res)

    col_d, col_c = st.columns(2)
    with col_d:
        _render_delivered_stocks(res)
    with col_c:
        _render_cash_positions(res)


# ──────────────────────────────────────────────────────────────────────────
# Scenario Builder (left column)
# ──────────────────────────────────────────────────────────────────────────

def _render_scenario_builder(preset: dict, preset_name: str,
                             idio_intensity: float) -> dict:
    """Render the scenario-builder UI and return a UI-format scenario dict
    (which the engine accepts directly via ``_normalise_scenario``)."""
    st.markdown("#### Initial market state")
    init_options = list(INITIAL_MARKET_STATES.keys())
    init_default = preset.get("initial_market_state", DEFAULT_INITIAL_MARKET_STATE)
    initial_state = st.selectbox(
        "Market behaviour before any event",
        options=init_options,
        index=init_options.index(init_default) if init_default in init_options
              else init_options.index(DEFAULT_INITIAL_MARKET_STATE),
        label_visibility="collapsed",
    )

    st.markdown("#### Events")
    st.caption(
        "Each row is one event.  Choose **when** it happens (days from today), "
        "**what shocks** each factor, and the **recovery type** that follows. "
        "The recovery is *coupled* to the shock — bigger shocks produce "
        "proportionally steeper recoveries."
    )

    df_initial = _events_to_df(preset.get("events", []))
    edited = st.data_editor(
        df_initial,
        num_rows="dynamic",
        width="stretch",
        column_config=_events_column_config(),
        key=f"events_table::{preset_name}",
    )
    events = _df_to_events(edited)

    st.markdown("#### Mean reversion")
    kappa = st.slider(
        "κ (Schwartz mean-reversion speed)",
        min_value=0.0, max_value=2.0,
        value=float(preset.get("mean_reversion_kappa", 0.5)),
        step=0.05,
        label_visibility="collapsed",
        help="Higher κ → tighter pull-back to fair-value trajectory. 0 = pure GBM.",
    )

    return {
        "initial_market_state": initial_state,
        "events":               events,
        "idio_intensity":       float(idio_intensity),
        "mean_reversion_kappa": float(kappa),
    }


def _events_column_config():
    """Streamlit column config for the events ``st.data_editor`` table."""
    cfg = {
        "Day": st.column_config.NumberColumn(
            "Day",
            help="Days from today when the event hits.",
            min_value=1, max_value=2500, step=1,
            required=True, format="%d",
        ),
    }
    for code in FACTORS:
        ticker, _key, label = FACTORS[code]
        cfg[f"Δ {code}"] = st.column_config.NumberColumn(
            f"Δ {code}",
            help=f"Shock to {code} ({label}) at this event, in %.",
            min_value=-60.0, max_value=80.0, step=0.5, format="%+.1f",
        )
    cfg["Recovery"] = st.column_config.SelectboxColumn(
        "Recovery",
        help=("What happens after this event:\n"
              "• Continued bear — shock continues for ~1 year\n"
              "• Stable — flat at the new level\n"
              "• Slow recovery — reverses shock over ~2 years\n"
              "• Fast recovery — V-shape, full snap-back in ~6 months"),
        options=list(EVENT_RECOVERY_ARCHETYPES.keys()),
        required=True,
    )
    return cfg


def _events_to_df(events: list[dict]) -> pd.DataFrame:
    """Convert preset's events list → DataFrame for the editor."""
    if not events:
        # Provide one empty starter row so the user has something to edit.
        empty = {"Day": None}
        for code in FACTORS:
            empty[f"Δ {code}"] = 0.0
        empty["Recovery"] = DEFAULT_RECOVERY_ARCHETYPE
        return pd.DataFrame([empty])

    rows = []
    for ev in events:
        shock = ev.get("factor_shock", {}) or {}
        row = {"Day": int(ev["day"])}
        for code in FACTORS:
            row[f"Δ {code}"] = float(shock.get(code, 0.0))
        row["Recovery"] = ev.get("recovery", DEFAULT_RECOVERY_ARCHETYPE)
        rows.append(row)
    return pd.DataFrame(rows)


def _df_to_events(df: pd.DataFrame) -> list[dict]:
    """Convert edited DataFrame → events list consumable by the engine."""
    events = []
    for _, row in df.iterrows():
        day = row.get("Day")
        if day is None or pd.isna(day):
            continue
        shock = {}
        for code in FACTORS:
            v = row.get(f"Δ {code}")
            shock[code] = float(v) if v is not None and not pd.isna(v) else 0.0
        recovery = row.get("Recovery")
        if recovery is None or (isinstance(recovery, float) and pd.isna(recovery)):
            recovery = DEFAULT_RECOVERY_ARCHETYPE
        events.append({
            "day":          int(day),
            "factor_shock": shock,
            "recovery":     str(recovery),
        })
    return events


# ──────────────────────────────────────────────────────────────────────────
# Session-level NoiseSampler
# ──────────────────────────────────────────────────────────────────────────

def _get_or_make_sampler(portfolio, n_paths, regen_clicked):
    today              = pd.Timestamp.today().normalize()
    portfolio_maturity = pd.to_datetime(portfolio["maturity_date"]).max()
    n_days             = len(pd.bdate_range(start=today, end=portfolio_maturity))
    isins        = sorted({isin for _, r in portfolio.iterrows() for isin in r["underlying_isins"]})
    factor_codes = list(FACTORS.keys())

    sampler: NoiseSampler | None = st.session_state.get(_SAMPLER_KEY)

    if sampler is None or not sampler.matches(n_paths, n_days, factor_codes, isins):
        sampler = NoiseSampler(n_paths=n_paths, n_days=n_days,
                               factor_codes=factor_codes, isins=isins)
    elif regen_clicked:
        sampler.regenerate()

    st.session_state[_SAMPLER_KEY] = sampler
    return sampler


# ──────────────────────────────────────────────────────────────────────────
# Output tabs (right column)
# ──────────────────────────────────────────────────────────────────────────

def _render_output_tabs(res, portfolio, loadings, ui_scenario):
    today = pd.Timestamp.today().normalize()
    shock_dates = [
        today + pd.Timedelta(days=int(ev["day"]))
        for ev in ui_scenario.get("events", [])
    ]

    tabs = st.tabs([
        "Factor Paths", "Asset Paths", "P&L Distribution",
        "P&L Decomposition", "Loadings",
    ])

    with tabs[0]:
        _plot_factor_paths(res["factor_paths"], shock_dates)
    with tabs[1]:
        _plot_asset_paths(res["asset_paths"], portfolio, shock_dates)
    with tabs[2]:
        _render_pnl_distribution(res)
    with tabs[3]:
        _render_pl_decomposition(res, portfolio, loadings)
    with tabs[4]:
        _render_loadings_table(portfolio, loadings)


# ──────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _add_fan_band(fig: go.Figure, df: pd.DataFrame, color: str,
                  legendgroup: str):
    """Add only the ±1σ shaded band (no median line).  Bands are drawn
    *first* across every series, then median lines on top — so one
    series's band never paints over another series's median line."""
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["upper_1sd"],
        mode="lines", line=dict(width=0),
        legendgroup=legendgroup, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["lower_1sd"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_hex_to_rgba(color, 0.18),
        legendgroup=legendgroup, showlegend=False, hoverinfo="skip",
    ))


def _add_fan_line(fig: go.Figure, df: pd.DataFrame, color: str, name: str,
                  hover_unit: str, legendgroup: str):
    """Add only the median line (drawn on top of all bands)."""
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["median"],
        mode="lines", name=name,
        line=dict(color=color, width=2),
        legendgroup=legendgroup,
        hovertemplate=(
            f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>"
            f"Median {hover_unit}: %{{y:.2f}}<extra></extra>"
        ),
    ))


def _add_shock_markers(fig: go.Figure, shock_dates):
    fig.add_hline(y=100, line=dict(color="grey", dash="dot", width=1), opacity=0.4)
    for i, sd in enumerate(shock_dates):
        fig.add_vline(
            x=sd.timestamp() * 1000,
            line=dict(color=_SHOCK_COLOR, dash="dash", width=1.5),
            opacity=0.7,
            annotation_text=f"Event {i+1}",
            annotation_position="top right" if i % 2 == 0 else "top left",
            annotation_font=dict(color=_SHOCK_COLOR, size=10),
        )


def _plot_factor_paths(factor_paths: dict, shock_dates):
    fig = go.Figure()
    # Pass 1 — shaded bands first (so they never paint over neighbouring lines).
    for code, df in factor_paths.items():
        _add_fan_band(fig, df,
                      color=_FACTOR_COLORS.get(code, "#aaaaaa"),
                      legendgroup=code)
    # Pass 2 — median lines on top.
    for code, df in factor_paths.items():
        _add_fan_line(fig, df,
                      color=_FACTOR_COLORS.get(code, "#aaaaaa"),
                      name=code, hover_unit="Index", legendgroup=code)
    _add_shock_markers(fig, shock_dates)
    fig.update_layout(
        template="plotly_dark", height=480,
        margin=dict(t=40, b=40, l=40, r=20),
        xaxis_title="Date",
        yaxis_title="Factor Index (base 100, median ± 1σ)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def _plot_asset_paths(asset_paths: dict, portfolio: pd.DataFrame, shock_dates):
    isin_to_name   = {}
    isin_to_strike = {}
    for _, prow in portfolio.iterrows():
        for isin, name in zip(prow["underlying_isins"], prow["underlyings"]):
            isin_to_name[isin] = name
        for isin, strike in zip(prow["underlying_isins"], prow["strike"]):
            isin_to_strike[isin] = strike

    fig = go.Figure()
    # Pre-compute normalised frames so we can iterate twice.
    norms: list[tuple[str, str, str, pd.DataFrame, float | None]] = []
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
        strike = isin_to_strike.get(isin)
        norm_strike = strike / spot0 * 100 if strike is not None else None
        norms.append((isin, name, color, norm, norm_strike))

    # Pass 1 — shaded bands first
    for isin, name, color, norm, _strike in norms:
        _add_fan_band(fig, norm, color=color, legendgroup=isin)

    # Pass 2 — median lines + strike lines on top
    for isin, name, color, norm, norm_strike in norms:
        _add_fan_line(fig, norm, color=color, name=name,
                      hover_unit="Normalised", legendgroup=isin)
        if norm_strike is not None:
            fig.add_hline(
                y=norm_strike,
                line=dict(color=color, dash="dash", width=1),
                opacity=0.40,
                annotation_text=f"{name} strike",
                annotation_position="right",
                annotation_font=dict(color=color, size=9),
            )

    _add_shock_markers(fig, shock_dates)
    fig.update_layout(
        template="plotly_dark", height=520,
        margin=dict(t=40, b=40, l=40, r=120),
        xaxis_title="Date",
        yaxis_title="Normalised Price (base 100, median ± 1σ)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────────────────────
# P&L distribution
# ──────────────────────────────────────────────────────────────────────────

def _render_pnl_distribution(res):
    samples_by_ccy: dict = res["pnl_samples_by_ccy"]
    if not samples_by_ccy:
        st.info("No P&L samples to display.")
        return

    for ccy, samples in samples_by_ccy.items():
        st.markdown(f"**{ccy} portfolio P&L distribution**  (n = {len(samples)} paths)")

        mean   = float(samples.mean())
        median = float(np.median(samples))
        p5     = float(np.percentile(samples, 5))
        p95    = float(np.percentile(samples, 95))
        es5    = (float(samples[samples <= p5].mean())
                  if len(samples) >= 20 else float(samples.min()))

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
        st.markdown("&nbsp;")


# ──────────────────────────────────────────────────────────────────────────
# P&L decomposition (median path)
# ──────────────────────────────────────────────────────────────────────────

def _render_pl_decomposition(res, portfolio, loadings):
    factor_dlogs = {
        code: float(np.log(df["median"].iloc[-1] / df["median"].iloc[0]))
        for code, df in res["factor_paths"].items()
    }

    rows = []
    for isin, path_df in res["asset_paths"].items():
        name    = _isin_label(portfolio, isin)
        loading = loadings.get(isin, {})
        betas   = loading.get("betas", {})
        contributions = {
            code: betas.get(code, 1.0 if code == "MKT" else 0.0)
                  * factor_dlogs.get(code, 0.0)
            for code in FACTORS
        }
        systematic_total = sum(contributions.values())
        terminal_dlog    = float(np.log(path_df["median"].iloc[-1] /
                                        path_df["median"].iloc[0]))
        idio = terminal_dlog - systematic_total
        row = {"Underlying": name, "ISIN": isin}
        for code in FACTORS:
            row[code] = contributions[code] * 100
        row["Idio"]  = idio * 100
        row["Total"] = terminal_dlog * 100
        rows.append(row)

    df = pd.DataFrame(rows)

    st.markdown("**Median-path return decomposition by factor (%)**")
    st.caption(
        "Each cell is β · Δlog(factor) on the median simulated path.  "
        "*Idio* captures the residual."
    )

    fmt_cols = list(FACTORS) + ["Idio", "Total"]
    styled = (
        df.style
        .format({c: "{:+.2f}" for c in fmt_cols})
        .background_gradient(cmap="RdYlGn", subset=fmt_cols, vmin=-50, vmax=50)
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    st.markdown("**Visual decomposition**")
    fig = go.Figure()
    for code in FACTORS:
        fig.add_trace(go.Bar(
            name=code, x=df["Underlying"], y=df[code],
            marker_color=_FACTOR_COLORS[code],
        ))
    fig.add_trace(go.Bar(
        name="Idio", x=df["Underlying"], y=df["Idio"],
        marker_color="#888888",
    ))
    fig.update_layout(
        template="plotly_dark", barmode="relative",
        height=420, margin=dict(t=20, b=40, l=40, r=20),
        yaxis_title="Contribution to Median Terminal Log-Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")


def _render_loadings_table(portfolio: pd.DataFrame, loadings: dict):
    isins = sorted({isin for _, r in portfolio.iterrows() for isin in r["underlying_isins"]})
    rows = []
    for isin in isins:
        name = _isin_label(portfolio, isin)
        loading = loadings.get(isin)
        if not loading:
            rows.append({"Underlying": name, "ISIN": isin,
                         **{f"β_{c}": np.nan for c in FACTORS},
                         "α": np.nan, "Idio σ": np.nan,
                         "R²": np.nan, "n_obs": 0})
            continue
        row = {"Underlying": name, "ISIN": isin}
        for c in FACTORS:
            row[f"β_{c}"] = loading["betas"].get(c, np.nan)
        row["α"]      = loading["alpha"]
        row["Idio σ"] = loading["idio_vol"]
        row["R²"]     = loading["r_squared"]
        row["n_obs"]  = loading["n_obs"]
        rows.append(row)

    df = pd.DataFrame(rows)
    beta_cols = [f"β_{c}" for c in FACTORS]
    fmt = {c: "{:+.2f}" for c in beta_cols}
    fmt.update({"α": "{:+.4f}", "Idio σ": "{:.2%}", "R²": "{:.2f}"})

    styled = (
        df.style
        .format(fmt)
        .background_gradient(cmap="RdYlGn", subset=beta_cols, vmin=-1.5, vmax=1.5)
        .background_gradient(cmap="Greens", subset=["R²"], vmin=0, vmax=1)
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def _isin_label(portfolio: pd.DataFrame, isin: str) -> str:
    for _, prow in portfolio.iterrows():
        for i, name in zip(prow["underlying_isins"], prow["underlyings"]):
            if i == isin:
                return name
    return isin


# ──────────────────────────────────────────────────────────────────────────
# Bottom tables
# ──────────────────────────────────────────────────────────────────────────

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
