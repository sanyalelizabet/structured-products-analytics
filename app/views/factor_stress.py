"""Factor Stress Testing view — multi-factor structural Monte Carlo engine.

Built on top of:

* :class:`FactorScenarioEngine`     (vectorised multi-path simulation)
* :class:`NoiseSampler`             (Common Random Numbers cache)
* :data:`FACTOR_SCENARIO_PRESETS`   (named coherent scenarios)
* :class:`FactorLoadingsEngine`     (OLS β-vectors per ISIN)

The :class:`NoiseSampler` lives in ``st.session_state`` so two consecutive
scenarios share the same noise realisation — which is what makes slider
nudges produce *smooth*, interpretable changes (CRN sensitivity).

Layout
------
* Top:    preset selector · λ slider · n_paths slider · Regenerate button
* Left:   factor-shock sliders, path timing, advanced drift expander, κ
* Right:  five tabs — Factor Paths · Asset Paths · P&L Distribution ·
                       P&L Decomposition · Loadings
* Bottom: portfolio summary · product detail · delivered stocks · cash
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


# ──────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────

# Professional palette — desaturated, distinguishable on dark backgrounds.
_FACTOR_COLORS = {
    "MKT":    "#5B7C99",   # steel blue
    "TECH":   "#3FA29E",   # muted teal
    "HC":     "#7FA670",   # sage green
    "FIN":    "#8E7CC3",   # slate / lavender
    "ENERGY": "#B5651D",   # burnt sienna (oil)
    "FX":     "#C9A961",   # muted gold
}

# Tableau 10 — finance-dashboard standard.
_ASSET_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#86BCB6",
]

_SHOCK_COLOR = "#C0504D"

_SAMPLER_KEY = "factor_noise_sampler"


# ──────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────

def render(portfolio, loadings, factor_engine, risk_free_rates,
           fx_rates=None, reference_currency=None):
    st.subheader("Factor Stress Testing — Multi-Factor Structural Engine")
    st.caption(
        "Scenarios are *coherent vectors* across MKT · TECH · HC · FIN · ENERGY · FX. "
        "Asset paths are projected from simulated factor paths via OLS-estimated loadings — "
        "dispersion comes from exposure, not from random noise. "
        "Common Random Numbers ensure that scenario-to-scenario differences reflect "
        "the scenario change, not seed jitter."
    )

    # ── Top control row ──────────────────────────────────────────────────
    col_preset, col_lambda, col_paths, col_regen = st.columns([3, 2, 2, 1])

    with col_preset:
        preset_name = st.selectbox(
            "Scenario Preset",
            list(FACTOR_SCENARIO_PRESETS.keys()),
        )
    preset = FACTOR_SCENARIO_PRESETS[preset_name]

    with col_lambda:
        idio_intensity = st.slider(
            "Idio intensity λ",
            min_value=0.0, max_value=1.0,
            value=float(preset["idio_intensity"]), step=0.05,
            help=("0 = deterministic factor projection · "
                  "0.3 = recommended damped Monte Carlo · "
                  "1.0 = full historical idio noise"),
        )

    with col_paths:
        n_paths = st.slider(
            "Paths", min_value=1, max_value=500,
            value=100, step=10,
            help="Number of Monte Carlo paths. Higher = tighter mean estimate, slower run.",
        )

    with col_regen:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        regen_clicked = st.button("Regenerate", help="Draw a fresh noise sample.")

    st.info(f"**{preset['label']}** — {preset['description']}")

    # ── Two-column control / output layout ───────────────────────────────
    col_left, col_right = st.columns([2, 3])

    with col_left:
        scenario = _render_scenario_controls(preset, idio_intensity)

    # ── Build / reuse session-level NoiseSampler ─────────────────────────
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
        idio_intensity=scenario["idio_intensity"],
        mean_reversion_kappa=scenario["mean_reversion_kappa"],
        n_paths=n_paths,
        noise_sampler=sampler,
        fx_rates=fx_rates,
        reference_currency=reference_currency,
    )
    res = engine.run_path_scenario(scenario)
    # Persist the (possibly grown / re-sized) sampler back into session state.
    st.session_state[_SAMPLER_KEY] = engine.noise_sampler

    with col_right:
        _render_output_tabs(res, portfolio, loadings, scenario)

    # ── Portfolio-level tables (full width) ──────────────────────────────
    st.markdown("---")
    _render_portfolio_summary(res)
    _render_portfolio_summary_ref(res)
    _render_pnl_distribution_ref(res)
    _render_product_detail(res)

    col_d, col_c = st.columns(2)
    with col_d:
        _render_delivered_stocks(res)
    with col_c:
        _render_cash_positions(res)


# ──────────────────────────────────────────────────────────────────────────
# Session-level NoiseSampler
# ──────────────────────────────────────────────────────────────────────────

def _get_or_make_sampler(portfolio, n_paths, regen_clicked):
    """Return a sampler that matches the request, creating a new one only
    when dimensions change.  ``regen_clicked`` triggers a fresh draw with
    the same dimensions (smooth UX: same fan widths, different realisation)."""
    today              = pd.Timestamp.today().normalize()
    portfolio_maturity = pd.to_datetime(portfolio["maturity_date"]).max()
    n_days             = len(pd.bdate_range(start=today, end=portfolio_maturity))
    isins = sorted({isin for _, r in portfolio.iterrows() for isin in r["underlying_isins"]})
    factor_codes = list(FACTORS.keys())

    sampler: NoiseSampler | None = st.session_state.get(_SAMPLER_KEY)

    if sampler is None or not sampler.matches(n_paths, n_days, factor_codes, isins):
        sampler = NoiseSampler(
            n_paths=n_paths, n_days=n_days,
            factor_codes=factor_codes, isins=isins,
        )
    elif regen_clicked:
        sampler.regenerate()

    st.session_state[_SAMPLER_KEY] = sampler
    return sampler


# ──────────────────────────────────────────────────────────────────────────
# Left column — scenario controls
# ──────────────────────────────────────────────────────────────────────────

def _render_scenario_controls(preset: dict, idio_intensity: float) -> dict:
    st.markdown("#### Factor Shocks (% per event)")
    factor_shocks = {}
    for code in FACTORS:
        _ticker, _key, label = FACTORS[code]
        factor_shocks[code] = st.slider(
            f"{code} — {label}",
            min_value=-60.0, max_value=80.0,
            value=float(preset["factor_shock"].get(code, 0.0)),
            step=0.5, key=f"shock_{code}",
        )

    st.markdown("#### Path Timing")
    n_shocks = st.number_input(
        "Number of shock events", min_value=0, max_value=5,
        value=int(preset["n_shocks"]), key="n_shocks",
    )
    shock_in_days = st.number_input(
        "Days to first shock", min_value=1, max_value=720,
        value=int(preset["shock_in_days"]), key="shock_in_days",
    )
    shock_spacing_days = st.number_input(
        "Days between shocks", min_value=0, max_value=720,
        value=int(preset["shock_spacing_days"]), key="shock_spacing_days",
    )

    kappa = st.slider(
        "Mean reversion κ", min_value=0.0, max_value=2.0,
        value=float(preset["mean_reversion_kappa"]), step=0.05,
        help="Higher κ → tighter pull-back to fair-value trajectory. 0 = pure GBM.",
    )

    with st.expander("Advanced — factor drifts (annualised, %)"):
        st.caption("Pre-shock drift")
        drift_pre = {}
        for code in FACTORS:
            drift_pre[code] = st.slider(
                f"μ_pre {code}", -30.0, 30.0,
                value=float(preset["factor_drift_pre_pa"].get(code, 0.0)) * 100,
                step=1.0, key=f"drift_pre_{code}",
            ) / 100.0

        st.caption("Post-shock drift")
        drift_post = {}
        for code in FACTORS:
            drift_post[code] = st.slider(
                f"μ_post {code}", -30.0, 30.0,
                value=float(preset["factor_drift_post_pa"].get(code, 0.0)) * 100,
                step=1.0, key=f"drift_post_{code}",
            ) / 100.0

    return {
        "factor_shock":         factor_shocks,
        "n_shocks":             int(n_shocks),
        "shock_in_days":        int(shock_in_days),
        "shock_spacing_days":   int(shock_spacing_days),
        "factor_drift_pre_pa":  drift_pre,
        "factor_drift_post_pa": drift_post,
        "idio_intensity":       float(idio_intensity),
        "mean_reversion_kappa": float(kappa),
    }


# ──────────────────────────────────────────────────────────────────────────
# Right column — output tabs
# ──────────────────────────────────────────────────────────────────────────

def _render_output_tabs(res, portfolio, loadings, scenario):
    today = pd.Timestamp.today().normalize()
    shock_dates = _scenario_shock_dates(today, scenario)

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

def _scenario_shock_dates(today, scenario):
    n_shocks      = int(scenario["n_shocks"])
    shock_in_days = int(scenario["shock_in_days"])
    spacing       = int(scenario["shock_spacing_days"])
    return [today + pd.Timedelta(days=shock_in_days + i * spacing)
            for i in range(n_shocks)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _fan_band_traces(df: pd.DataFrame, color: str, name: str):
    """Two transparent traces forming a median ± 1σ band.  Both share
    ``legendgroup=name`` with the median trace so legend clicks toggle the
    whole asset (median + band)."""
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


def _fan_median_trace(df: pd.DataFrame, color: str, name: str,
                      hover_unit: str = "Index"):
    return go.Scatter(
        x=df["date"], y=df["median"],
        mode="lines", name=name,
        line=dict(color=color, width=2),
        legendgroup=name, showlegend=True,
        hovertemplate=(
            f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>"
            f"Median {hover_unit}: %{{y:.2f}}<extra></extra>"
        ),
    )


def _add_shock_markers(fig: go.Figure, shock_dates):
    fig.add_hline(y=100, line=dict(color="grey", dash="dot", width=1), opacity=0.4)
    for i, sd in enumerate(shock_dates):
        fig.add_vline(
            x=sd.timestamp() * 1000,
            line=dict(color=_SHOCK_COLOR, dash="dash", width=1.5),
            opacity=0.7,
            annotation_text=f"Shock {i+1}",
            annotation_position="top right" if i % 2 == 0 else "top left",
            annotation_font=dict(color=_SHOCK_COLOR, size=10),
        )


def _plot_factor_paths(factor_paths: dict, shock_dates):
    """Median ± 1σ fan for each factor index.  Two-pass draw: all bands
    first, then all median lines on top, so no asset's median gets
    visually buried under another asset's band."""
    fig = go.Figure()
    items = list(factor_paths.items())
    # Pass 1: bands
    for code, df in items:
        for tr in _fan_band_traces(df, _FACTOR_COLORS.get(code, "#aaaaaa"), code):
            fig.add_trace(tr)
    # Pass 2: medians on top
    for code, df in items:
        fig.add_trace(_fan_median_trace(
            df, _FACTOR_COLORS.get(code, "#aaaaaa"), code, hover_unit="Index",
        ))
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
    """Per-asset median + percentile fan, normalised to 100 at t=0."""
    isin_to_name   = {}
    isin_to_strike = {}
    for _, prow in portfolio.iterrows():
        for isin, name in zip(prow["underlying_isins"], prow["underlyings"]):
            isin_to_name[isin] = name
        for isin, strike in zip(prow["underlying_isins"], prow["strike"]):
            isin_to_strike[isin] = strike

    fig = go.Figure()

    # Pre-normalise once per asset
    norm_assets = []
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
        norm_assets.append((isin, name, color, norm, spot0))

    # Pass 1: all bands
    for _, name, color, norm, _ in norm_assets:
        for tr in _fan_band_traces(norm, color, name):
            fig.add_trace(tr)
    # Pass 2: all median lines (drawn on top)
    for _, name, color, norm, _ in norm_assets:
        fig.add_trace(_fan_median_trace(norm, color, name, hover_unit="Normalised"))

    # Strike lines
    for isin, name, color, _, spot0 in norm_assets:
        strike = isin_to_strike.get(isin)
        if strike is not None:
            norm_strike = strike / spot0 * 100
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
# P&L distribution histogram
# ──────────────────────────────────────────────────────────────────────────

def _render_pnl_distribution(res):
    """Histogram of portfolio-level P&L draws per currency with mean / p5 lines."""
    samples_by_ccy: dict = res["pnl_samples_by_ccy"]
    if not samples_by_ccy:
        st.info("No P&L samples to display.")
        return

    for ccy, samples in samples_by_ccy.items():
        st.markdown(f"**{ccy} portfolio P&L distribution** "
                    f"(n = {len(samples)} paths)")

        mean   = float(samples.mean())
        median = float(np.median(samples))
        p5     = float(np.percentile(samples, 5))
        p95    = float(np.percentile(samples, 95))
        es5    = float(samples[samples <= p5].mean()) if len(samples) >= 20 else float(samples.min())

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=samples,
            nbinsx=30,
            marker_color="#4E79A7",
            opacity=0.85,
            name=ccy,
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
            template="plotly_dark",
            height=320,
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
# P&L decomposition (factor contributions, computed at the median path)
# ──────────────────────────────────────────────────────────────────────────

def _render_pl_decomposition(res, portfolio, loadings):
    """Per-underlying terminal log-return decomposed into factor contributions
    on the *median* path:

        Δ log S_i (median)   ≈   Σ_k  β_{i,k} · Δ log F_k (median)   +   ε_i
    """
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
        "Each cell is β · Δlog(factor) on the median simulated path — the "
        "contribution of that factor to the underlying's median terminal "
        "log-return.  *Idio* captures the residual."
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
            name=code,
            x=df["Underlying"], y=df[code],
            marker_color=_FACTOR_COLORS[code],
        ))
    fig.add_trace(go.Bar(
        name="Idio", x=df["Underlying"], y=df["Idio"],
        marker_color="#888888",
    ))
    fig.update_layout(
        template="plotly_dark",
        barmode="relative",
        height=420,
        margin=dict(t=20, b=40, l=40, r=20),
        yaxis_title="Contribution to Median Terminal Log-Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")


def _render_loadings_table(portfolio: pd.DataFrame, loadings: dict):
    """Display the OLS loadings used by the engine — full transparency."""
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
# Bottom — portfolio / product / delivered / cash tables
# ──────────────────────────────────────────────────────────────────────────

def _render_portfolio_summary_ref(res):
    """Reference-currency portfolio summary (item 4)."""
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
            "pnl_p95":                     "PnL 95%",
            "pnl_es5":                      "ES (worst 5%)",
            "portfolio_return_mean_pct":    "Return Mean (%)",
            "portfolio_return_p5_pct":      "Return 5% (%)",
        }),
        width="stretch", hide_index=True,
    )


def _render_pnl_distribution_ref(res):
    """Histogram of total per-path P&L in reference currency (item 4)."""
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
    c1.metric("Mean P&L",      f"{mean:,.0f}")
    c2.metric("Median P&L",    f"{median:,.0f}")
    c3.metric("5% P&L",        f"{p5:,.0f}")
    c4.metric("95% P&L",       f"{p95:,.0f}")
    c5.metric("ES (worst 5%)", f"{es5:,.0f}")


def _render_portfolio_summary(res):
    st.markdown("### Portfolio Stress Summary")
    pf = res["pf_scenario_per_ccy"].copy()
    pf = pf.round(2)
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
            "worst_underlying":    "Worst Underlying (mode)",
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
