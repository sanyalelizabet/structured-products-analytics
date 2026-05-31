"""Factor Stress Testing view — multi-factor structural Monte Carlo engine.

Scenarios are event timelines: an initial market state plus a series of dated
events. Each event carries per-factor shocks and a Recovery archetype that,
coupled with the shock magnitude, sets the post-event drift. The engine maps
this to numerical drifts and runs a vectorised multi-path simulation with
Common Random Numbers (cached :class:`NoiseSampler` in session state).

Layout
------
* Top:    preset · λ · paths · Regenerate
* Left:   Initial market state + Events table + κ
* Right:  Factor paths · Asset paths · P&L distribution · Decomposition · Loadings
* Bottom: portfolio summary · product detail · delivered · cash
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.formatting import chf
from app.views._layout import fit_height as _fit_height
from data.factor_scenarios import FACTOR_SCENARIO_PRESETS
from app.ai_insights import (
    build_factor_stress_payload,
    generate_factor_stress_insight,
    payload_hash,
)
from src.risk.factor_engine import FACTORS
from src.risk.factor_premiums import (
    ESTIMATION_LOOKBACK_YEARS, PREMIUM_METHODS, compute_factor_premiums,
)
from src.risk.factor_scenario_engine import FactorScenarioEngine
from src.numerics.noise_sampler import NoiseSampler
from src.risk.scenario_archetypes import (
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

log = logging.getLogger(__name__)


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
    st.subheader("Factor Stress Testing")

    # ── How-to-use panel — collapsed by default so it doesn't clutter, but
    #    one click away.  Explains every input the user has to fill in.
    with st.expander("How to use this view", expanded=False):
        st.markdown(
            """
Test the portfolio against a market scenario you design, and see the range of
possible P&L.

1. **Pick a preset** (COVID, energy crisis…) to fill in the scenario, then edit
   it if you like.
2. **Set the sampling.** *Paths* = number of simulations (100 is fine, 500 if
   the result looks noisy). *Idio intensity* = extra random noise (0.3
   recommended). *Regenerate* draws a new random sample.
3. **Build the scenario** (left). *Initial market state* = Bull / Flat / Bear
   assumed between shocks. *Events* = one row per shock: when (days from today),
   how much each factor moves (%), and what happens afterwards (recovery type).
   *κ* = how fast factors revert to trend.
4. **Read the output** (right). *Factor / Asset Paths* show the median and a
   spread band, with a line at each shock date. *P&L Distribution* shows the
   range of portfolio outcomes.
"""
        )

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

    # ── Factor-premium estimator selector ────────────────────────────────
    premium_method = st.radio(
        "Factor premium estimate",
        options=list(PREMIUM_METHODS),
        format_func=lambda m: {
            "mean":      "Historical average",
            "shrinkage": "Stabilised (recommended)",
        }.get(m, m),
        horizontal=True,
        help=("How each regime's factor drifts are estimated. "
              "'Historical average' uses the plain in-regime mean (and a fixed "
              "fallback when a regime has little data). 'Stabilised' blends that "
              "average with a model-based estimate, which is steadier and gives "
              "sensible per-factor values even when a regime is thinly observed."),
    )
    premiums_by_method, chosen_premiums = _compute_premiums(factor_engine, premium_method)

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
        premiums=chosen_premiums,
    )
    res = engine.run_path_scenario(ui_scenario)
    st.session_state[_SAMPLER_KEY] = engine.noise_sampler

    with col_right:
        _render_output_tabs(
            res, portfolio, loadings, ui_scenario,
            fx_rates=fx_rates, reference_currency=reference_currency,
            premiums_by_method=premiums_by_method, premium_method=premium_method,
        )

    # ── Portfolio-level tables ───────────────────────────────────────────
    st.markdown("---")
    _render_factor_stress_ai_insight(
        res=res,
        portfolio=portfolio,
        loadings=loadings,
        ui_scenario=ui_scenario,
        preset_name=preset_name,
        preset=preset,
        fx_rates=fx_rates,
        reference_currency=reference_currency,
        premiums_by_method=premiums_by_method,
        premium_method=premium_method,
    )
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

def _compute_premiums(factor_engine, chosen_method):
    """Compute the regime×factor premium tables for every estimator.

    Returns ``({method: DataFrame}, chosen_DataFrame)``. On failure (e.g. no
    factor history yet) returns ``({}, None)`` so the engine falls back to the
    cached default premiums.
    """
    try:
        # Ensure the estimation window of factor history is present (skip-safe:
        # re-fetches only when the stored history is shorter than requested).
        try:
            factor_engine.fetch_factor_prices(
                years=ESTIMATION_LOOKBACK_YEARS, force_refresh=False,
            )
        except Exception as e:  # network/data issues → use whatever is stored
            st.warning(f"Could not extend factor history to "
                       f"{ESTIMATION_LOOKBACK_YEARS}y ({e}); using stored data.")
        by_method = {
            m: compute_factor_premiums(factor_engine, method=m)
            for m in PREMIUM_METHODS
        }
    except Exception as e:  # never block the view on a premium-compute failure
        st.warning(f"Could not compute factor premiums ({e}); using cached defaults.")
        return {}, None
    return by_method, by_method.get(chosen_method)


def _render_output_tabs(res, portfolio, loadings, ui_scenario,
                        fx_rates=None, reference_currency=None,
                        premiums_by_method=None, premium_method="mean"):
    today = pd.Timestamp.today().normalize()
    shock_dates = [
        today + pd.Timedelta(days=int(ev["day"]))
        for ev in ui_scenario.get("events", [])
    ]

    tabs = st.tabs([
        "Factor Paths", "Asset Paths", "P&L Distribution",
        "P&L Decomposition", "Loadings", "Premiums",
    ])

    with tabs[0]:
        _plot_factor_paths(res["factor_paths"], shock_dates)
    with tabs[1]:
        _plot_asset_paths(res["asset_paths"], portfolio, shock_dates)
    with tabs[2]:
        _render_pnl_distribution(res, fx_rates=fx_rates,
                                  reference_currency=reference_currency)
    with tabs[3]:
        _render_pl_decomposition(res, portfolio, loadings)
    with tabs[4]:
        _render_loadings_table(portfolio, loadings)
    with tabs[5]:
        _render_factor_premiums(premiums_by_method, premium_method, ui_scenario)


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

def _render_pnl_distribution(res, fx_rates=None, reference_currency=None):
    """Render P&L distribution with a default aggregate (reference-currency)
    view and an opt-in per-currency breakdown.

    Aggregation rationale: each currency's ``pnl_samples`` array comes from
    the same Monte Carlo run, so path indices are aligned across currencies.
    Path-wise FX-converted sum is a valid portfolio P&L sample.
    """
    samples_by_ccy: dict = res["pnl_samples_by_ccy"]
    if not samples_by_ccy:
        st.info("No P&L samples to display.")
        return

    # ── View toggle ───────────────────────────────────────────────────────
    has_multi_ccy = len(samples_by_ccy) > 1
    can_aggregate = (
        reference_currency is not None
        and (fx_rates is not None or not has_multi_ccy)
    )
    default = "Reference currency (aggregate)" if can_aggregate else "Per currency"
    options = (
        ["Reference currency (aggregate)", "Per currency"]
        if can_aggregate else ["Per currency"]
    )
    view_mode = st.radio(
        "P&L view", options, index=options.index(default),
        horizontal=True,
        help=("Aggregate = sum across currencies after converting each path "
              "to the reference currency (path-wise; FX rates from PortfolioAnalytics).  "
              "Per currency = one distribution per currency, unconverted."),
    )

    if view_mode == "Reference currency (aggregate)":
        _render_aggregate_pnl(samples_by_ccy, fx_rates, reference_currency)
    else:
        _render_per_ccy_pnl(samples_by_ccy)


def _aggregate_samples(samples_by_ccy: dict, fx_rates: dict | None,
                        reference_currency: str) -> np.ndarray:
    """Path-wise sum of FX-converted samples → portfolio P&L array."""
    arrays = []
    for ccy, samples in samples_by_ccy.items():
        if ccy == reference_currency:
            rate = 1.0
        else:
            key = (ccy, reference_currency)
            rate = fx_rates.get(key) if fx_rates else None
            if rate is None:
                st.warning(
                    f"Missing FX rate for {ccy} → {reference_currency}; "
                    "this currency excluded from the aggregate."
                )
                continue
        arrays.append(np.asarray(samples) * float(rate))
    if not arrays:
        return np.array([])
    # All currencies share path indices (same MC run), so element-wise sum
    # is the correct aggregation.
    return np.sum(np.stack(arrays, axis=0), axis=0)


def _plot_pnl_histogram(samples: np.ndarray, x_title: str):
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
            annotation_text=f"{label}: {chf(value, 2)}",
            annotation_position="top",
            annotation_font=dict(color=color, size=10),
        )
    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(t=50, b=40, l=40, r=20),
        xaxis_title=x_title,
        yaxis_title="Number of paths",
        showlegend=False,
        separators=".'",
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean P&L",      chf(mean, 2))
    c2.metric("Median P&L",    chf(median, 2))
    c3.metric("5% P&L",        chf(p5, 2))
    c4.metric("95% P&L",       chf(p95, 2))
    c5.metric("ES (worst 5%)", chf(es5, 2))


def _render_aggregate_pnl(samples_by_ccy, fx_rates, reference_currency):
    samples = _aggregate_samples(samples_by_ccy, fx_rates, reference_currency)
    if samples.size == 0:
        st.info("No samples available after FX conversion.")
        return
    st.markdown(
        f"**Portfolio P&L distribution — aggregate ({reference_currency})**  "
        f"(n = {len(samples)} paths)"
    )
    _plot_pnl_histogram(samples, x_title=f"Portfolio P&L ({reference_currency})")


def _render_per_ccy_pnl(samples_by_ccy):
    for ccy, samples in samples_by_ccy.items():
        samples = np.asarray(samples)
        st.markdown(f"**{ccy} portfolio P&L distribution**  (n = {len(samples)} paths)")
        _plot_pnl_histogram(samples, x_title=f"Portfolio P&L ({ccy})")
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
    st.dataframe(styled, width="stretch", hide_index=True,
                 height=_fit_height(len(df)))

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
    st.dataframe(styled, width="stretch", hide_index=True,
                 height=_fit_height(len(df)))


def _render_factor_premiums(premiums_by_method: dict, premium_method: str,
                            ui_scenario: dict):
    """Show the per-regime, per-factor premium drifts (%/yr) for both
    estimators, with the active regime row highlighted.

    These are the values that feed the pre-shock (initial-market-state) drift.
    The estimator selected above is marked as **in use**; the other is shown
    for comparison.
    """
    from src.risk.factor_premiums import REGIMES

    st.markdown("#### Factor premiums by regime")
    st.caption(
        "Each number is the assumed **yearly drift (%)** for a risk factor while "
        "the market sits in a given regime (Bear / Flat / Bull). The regime is "
        "judged from the market's recent trend and how turbulent it has been. "
        "Your **Initial market state** above selects which row the simulation uses."
    )

    if not premiums_by_method:
        st.warning("No factor premiums available (factor history not loaded yet).")
        return

    init_state    = ui_scenario.get("initial_market_state", DEFAULT_INITIAL_MARKET_STATE)
    active_regime = INITIAL_MARKET_STATES.get(init_state)

    labels = {
        "mean":      "Historical average",
        "shrinkage": "Stabilised (recommended)",
    }

    def _highlight_active(row):
        hit = row.name == active_regime
        return ["background-color: #2A3F5F; color: white; font-weight: 600"
                if hit else "" for _ in row]

    for method, prem in premiums_by_method.items():
        factor_cols = [c for c in FACTORS if c in prem.columns]
        disp = prem.reindex(REGIMES)[factor_cols] * 100.0
        in_use = " — in use" if method == premium_method else ""
        st.markdown(f"**{labels.get(method, method)}**{in_use}")
        styled = (
            disp.style
            .format("{:+.2f}")
            .apply(_highlight_active, axis=1)
            .background_gradient(cmap="RdYlGn", vmin=-25, vmax=25)
        )
        st.dataframe(styled, width="stretch")

    st.caption(
        f"Showing the **{init_state}** regime (highlighted row). "
        "**Stabilised** is the recommended table — it stays reliable even for "
        "regimes that rarely occur; **Historical average** can be patchy there."
    )


def _isin_label(portfolio: pd.DataFrame, isin: str) -> str:
    for _, prow in portfolio.iterrows():
        for i, name in zip(prow["underlying_isins"], prow["underlyings"]):
            if i == isin:
                return name
    return isin


# ──────────────────────────────────────────────────────────────────────────
# Bottom tables
# ──────────────────────────────────────────────────────────────────────────

def _render_factor_stress_ai_insight(**kwargs):
    open_key = "factor_stress_ai_open"
    is_open = bool(st.session_state.get(open_key, False))
    label = "Hide Gemini AI insight" if is_open else "Show Gemini AI insight"
    if st.button(label, key="factor_stress_ai_toggle"):
        st.session_state[open_key] = not is_open
        st.rerun()

    if not st.session_state.get(open_key, False):
        return

    insight_payload = build_factor_stress_payload(**kwargs)
    hash_key = payload_hash(insight_payload)

    with st.container(border=True):
        st.caption("Gemini AI insight")
        if st.session_state.get("factor_stress_ai_hash") != hash_key:
            with st.spinner("Generating Gemini AI insight..."):
                try:
                    st.session_state["factor_stress_ai_insight"] = (
                        generate_factor_stress_insight(insight_payload)
                    )
                    st.session_state["factor_stress_ai_hash"] = hash_key
                except Exception:  # noqa: BLE001
                    log.exception("Factor Stress Gemini insight generation failed")
                    st.error("Gemini AI insight is temporarily unavailable.")
                    return

        insight = st.session_state.get("factor_stress_ai_insight")
        if insight:
            st.markdown(insight)


def _render_portfolio_summary(res):
    st.markdown("### Portfolio Stress Summary")
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
