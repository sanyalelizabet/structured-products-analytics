"""Portfolio view — buy-side risk dashboard.

Layout
------
1. Page header — valuation date + FX banner.
2. Top KPI strip — two rows of 4 cards each, signed metrics for at-a-glance read.
3. Quick stats row — # products, # underlyings, avg DTM, min distance to barrier.
4. Holdings — product table with conditional gradients on Return %,
   Distance-to-Barrier and Fair Value spread.
5. Underlying concentration — sorted horizontal bar + treemap + concentration KPIs.
6. Maturity profile — bucket table + stacked bar by maturity.
7. Risk (Greeks) — per-product table + horizontal bars for net Delta, Vega, Theta.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from app.formatting import chf
from app.ai_insights import (
    build_portfolio_page_payload,
    generate_portfolio_insight,
    payload_hash,
)


# ──────────────────────────────────────────────────────────────────────────
# Palette for the whole view
# ──────────────────────────────────────────────────────────────────────────


PRIMARY      = "#4E79A7"   # steel blue
ACCENT_DEEP  = "#2A3F5F"   # navy
POSITIVE     = "#76A65A"   # sage
NEGATIVE     = "#C0504D"   # brick
NEUTRAL_GREY = "#7A8797"   # slate
WARNING      = "#C9A961"   # muted gold


PRODUCT_TYPE_PALETTE = [
    PRIMARY,        # steel blue
    POSITIVE,       # sage
    "#9C755F",      # warm brown
    WARNING,        # muted gold
    ACCENT_DEEP,    # navy
    NEUTRAL_GREY,
]


DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "buyside_diverging",
    [
        (0.00, NEGATIVE),
        (0.50, NEUTRAL_GREY),
        (1.00, POSITIVE),
    ],
    N=256,
)


PLOTLY_DIVERGING_SCALE = [
    (0.00, NEGATIVE),
    (0.50, NEUTRAL_GREY),
    (1.00, POSITIVE),
]


def render(analytics, df, greeks_df, pf_delta, valuation_date, corr_df=None):
    _render_header(analytics, valuation_date)
    _render_kpi_strip(analytics, df)
    _render_quick_stats(analytics, df)
    st.markdown("---")
    _render_portfolio_page_insights(analytics, df, greeks_df, pf_delta, valuation_date)
    _render_maturity(analytics)
    st.markdown("---")
    _render_holdings(analytics, df, greeks_df, pf_delta, valuation_date)
    st.markdown("---")
    _render_concentration(analytics, corr_df)
    st.markdown("---")
    _render_risk(greeks_df, pf_delta)


from app.views._layout import fit_height as _fit_height

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

def _render_header(analytics, valuation_date):
    st.markdown("## Portfolio Analytics")
    vd = (valuation_date.strftime("%d %b %Y")
          if valuation_date is not None else "default spots")
    ref_ccy = analytics.reference_currency

    used_ccys = sorted(set(analytics.product_df["currency"]))
    fx_strings = []
    for ccy in used_ccys:
        if ccy != ref_ccy:
            r = analytics.fx_rates.get((ccy, ref_ccy))
            if r is not None:
                fx_strings.append(f"{ccy}→{ref_ccy} {r:.4f}")
    fx_caption = "  ·  ".join(fx_strings) if fx_strings else "—"

    st.markdown(
        f"<div style='color:{NEUTRAL_GREY}; font-size:0.85em; margin-top:-8px;'>"
        f"Valuation date · <b>{vd}</b>  |  Reference · <b>{ref_ccy}</b>  "
        f"|  FX · {fx_caption}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# KPI strip
# ──────────────────────────────────────────────────────────────────────────

def _signed(value, currency=None, fmt="{:,.0f}"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    sign = "+" if value > 0 else ("−" if value < 0 else "")
    body = fmt.format(abs(value)).replace(",", "'")
    if currency:
        return f"{sign}{body} {currency}"
    return f"{sign}{body}"


def _render_kpi_strip(analytics, df):
    pf  = analytics.total_portfolio_metrics()
    ref = analytics.reference_currency

    # MTM aggregation in reference currency
    fv = df[df["fair_value"].notna()].copy()
    fv["fv_ref"]   = fv.apply(lambda r: analytics.convert_to_reference(r["fair_value"],  r["currency"]), axis=1)
    fv["cost_ref"] = fv.apply(lambda r: analytics.convert_to_reference(r["total_cost"], r["currency"]), axis=1)
    mtm_value  = fv["fv_ref"].sum()   if not fv.empty else 0.0
    mtm_cost   = fv["cost_ref"].sum() if not fv.empty else 0.0
    mtm_pnl    = mtm_value - mtm_cost
    mtm_return = (mtm_pnl / mtm_cost * 100) if mtm_cost else None

    proj_pnl    = pf["total_pnl"]
    proj_payoff = pf["total_payoff"]
    proj_return = pf["portfolio_return_pct"] * 100
    notional    = pf["total_notional"]
    mwr         = pf["portfolio_mwr"] * 100

    # Row 1 — Mark-to-market block (today's snapshot)
    st.markdown(
        f"<div style='color:{NEUTRAL_GREY}; font-size:0.7em; "
        f"letter-spacing:1.5px; margin-bottom:-6px;'>"
        f"MARK-TO-MARKET  ·  TODAY"
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Notional", f"{chf(notional)} {ref}")
    c2.metric("MTM Value",      f"{chf(mtm_value)} {ref}")
    c3.metric("MTM PnL",        _signed(mtm_pnl, ref),
              delta=_signed(mtm_pnl, ref))
    c4.metric("MTM Return",
              f"{mtm_return:+.2f} %" if mtm_return is not None else "—",
              delta=f"{mtm_return:+.2f} %" if mtm_return is not None else None)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Row 2 — Projection block (at maturity)
    st.markdown(
        f"<div style='color:{NEUTRAL_GREY}; font-size:0.7em; "
        f"letter-spacing:1.5px; margin-bottom:-6px;'>"
        f"PROJECTED AT MATURITY  ·  CURRENT SPOTS HELD"
        f"</div>",
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Proj. Payoff",   f"{chf(proj_payoff)} {ref}")
    d2.metric("Proj. PnL",      _signed(proj_pnl, ref),
              delta=_signed(proj_pnl, ref))
    d3.metric("Proj. Return",   f"{proj_return:+.2f} %",
              delta=f"{proj_return:+.2f} %")
    d4.metric("MWR (p.a.)",     f"{mwr:+.2f} %",
              help="Money-Weighted Return — annualised IRR accounting for "
                   "deployment timing and notional size.")


def _render_quick_stats(analytics, df):
    pf = analytics.product_df
    n_products    = len(pf)
    n_underlyings = pf["underlyings"].nunique() if "underlyings" in pf.columns else 0
    # average days-to-maturity, weighted by notional
    today = pd.Timestamp.today().normalize()
    pf2 = pf.copy()
    pf2["dtm"] = (pd.to_datetime(pf2["maturity_date"]) - today).dt.days
    avg_dtm = (pf2["dtm"] * pf2["total_notional"]).sum() / pf2["total_notional"].sum()
    # min distance-to-barrier — already in % at this layer
    # (build_product_analytics multiplies by 100 before this view sees it).
    if "distance_to_barrier" in df.columns:
        min_dtb = df["distance_to_barrier"].min()
    else:
        min_dtb = None
    n_currencies = pf["currency"].nunique() if "currency" in pf.columns else 0

    st.markdown("&nbsp;", unsafe_allow_html=True)
    parts = [
        f"<b>{n_products}</b> products",
        f"<b>{n_underlyings}</b> underlyings",
        f"<b>{n_currencies}</b> currencies",
        f"<b>{avg_dtm:.0f}</b> avg days to maturity",
    ]
    if min_dtb is not None:
        color = NEGATIVE if min_dtb < 10 else (WARNING if min_dtb < 25 else POSITIVE)
        parts.append(
            f"min downside to barrier <span style='color:{color}'>"
            f"<b>{min_dtb:+.1f} %</b></span>"
        )

    st.markdown(
        f"<div style='color:{NEUTRAL_GREY}; font-size:0.9em;'>"
        + "  ·  ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# Holdings table — gradient-coloured for at-a-glance risk read
# ──────────────────────────────────────────────────────────────────────────

def _build_holdings_table(analytics, df) -> tuple[pd.DataFrame, str]:
    ref = analytics.reference_currency
    t   = analytics.product_df.copy()
    t["cost_ref"]   = t.apply(lambda r: analytics.convert_to_reference(r["total_cost"],   r["currency"]), axis=1)
    t["payoff_ref"] = t.apply(lambda r: analytics.convert_to_reference(r["total_payoff"], r["currency"]), axis=1)
    t["pnl_ref"]    = t.apply(lambda r: analytics.convert_to_reference(r["pnl"],          r["currency"]), axis=1)

    # Bring fair-value columns from the enriched df
    fv = df[["product_id", "fair_value", "fair_value_pct"]].copy()
    fv["fair_value_pct"] = fv["fair_value_pct"] * 100
    t = t.merge(fv, on="product_id", how="left")

    cols = [
        "product_id", "product_type", "underlyings", "currency",
        "maturity_date", "total_notional",
        "cost_ref", "weight_pct", "payoff_ref", "pnl_ref",
        "return_pct", "distance_to_barrier",
        "fair_value", "fair_value_pct",
    ]
    t = t[cols].copy()
    # ``return_pct`` and ``distance_to_barrier`` are already in % at this layer —
    # the streamlit app mutates ``analytics.product_df[...] *= 100`` as a side
    # effect of ``build_product_analytics``.  Don't scale them again.
    if "maturity_date" in t.columns:
        t["maturity_date"] = pd.to_datetime(t["maturity_date"]).dt.strftime("%d %b %Y")
    return t, ref


def _render_holdings(analytics, df, greeks_df, pf_delta, valuation_date):
    t, ref = _build_holdings_table(analytics, df)

    st.markdown("### Holdings")

    rename = {
        "product_id":          "Product",
        "product_type":        "Type",
        "underlyings":         "Underlyings",
        "currency":            "CCY",
        "maturity_date":       "Maturity",
        "total_notional":      "Notional",
        "cost_ref":            f"Cost {ref}",
        "weight_pct":          "Weight %",
        "payoff_ref":          f"Payoff {ref}",
        "pnl_ref":             f"PnL {ref}",
        "return_pct":          "Return %",
        "distance_to_barrier": "Dist. to Barrier %",
        "fair_value":          "Fair Value",
        "fair_value_pct":      "Fair Value %",
    }
    t = t.rename(columns=rename)

    fmt = {
        "Notional":         lambda v: chf(v, 2),
        f"Cost {ref}":      lambda v: chf(v, 2),
        "Weight %":         "{:+.2f}",
        f"Payoff {ref}":    lambda v: chf(v, 2),
        f"PnL {ref}":       lambda v: chf(v, 2, signed=True),
        "Return %":         "{:+.2f}",
        "Dist. to Barrier %": "{:+.1f}",
        "Fair Value":       lambda v: chf(v, 2),
        "Fair Value %":     "{:+.2f}",
    }

    styled = (
        t.style
        .format(fmt, na_rep="—")
        .background_gradient(cmap=DIVERGING_CMAP,   subset=["Return %"],          vmin=-30, vmax=30)
        .background_gradient(cmap=DIVERGING_CMAP,   subset=["Dist. to Barrier %"], vmin=-25, vmax=50)
        .background_gradient(cmap=DIVERGING_CMAP,   subset=[f"PnL {ref}"],
                              vmin=-(t[f"PnL {ref}"].abs().max() or 1),
                              vmax= (t[f"PnL {ref}"].abs().max() or 1))
    )
    st.dataframe(styled, width="stretch", hide_index=True,
                 height=_fit_height(len(t)))


# ──────────────────────────────────────────────────────────────────────────
# Gemini insight display
# ──────────────────────────────────────────────────────────────────────────

def _render_portfolio_page_insights(
    analytics,
    df: pd.DataFrame,
    greeks_df: pd.DataFrame,
    pf_delta: pd.DataFrame,
    valuation_date,
):
    open_key = "holdings_ai_open"
    is_open = bool(st.session_state.get(open_key, False))
    label = "Hide Gemini AI insight" if is_open else "Show Gemini AI insight"
    if st.button(label, key="holdings_ai_toggle"):
        st.session_state[open_key] = not is_open
        st.rerun()

    if not st.session_state.get(open_key, False):
        return

    t, ref = _build_holdings_table(analytics, df)
    insight_payload = build_portfolio_page_payload(
        holdings_table=t,
        reference_currency=ref,
        analytics=analytics,
        product_df=df,
        greeks_df=greeks_df,
        pf_delta=pf_delta,
        valuation_date=valuation_date,
    )
    hash_key = payload_hash(insight_payload)

    with st.container(border=True):
        st.caption("Gemini AI insight")
        if st.session_state.get("holdings_ai_hash") != hash_key:
            with st.spinner("Generating Gemini AI insight..."):
                try:
                    st.session_state["holdings_ai_insight"] = generate_portfolio_insight(
                        insight_payload,
                    )
                    st.session_state["holdings_ai_hash"] = hash_key
                except Exception:  # noqa: BLE001
                    log.exception("Portfolio Gemini insight generation failed")
                    st.error("Gemini AI insight is temporarily unavailable.")
                    return

        insight = st.session_state.get("holdings_ai_insight")
        if insight:
            st.markdown(insight)


def _render_concentration(analytics, corr_df=None):
    st.markdown("### Underlying Concentration")

    u = analytics.underlying_lookthrough().copy()
    ref = analytics.reference_currency
    u = u.sort_values("allocated_cost_ref", ascending=False).reset_index(drop=True)

    # underlying_lookthrough returns barrier distances as *fractions* (e.g. 0.20
    # for 20 %).  The bar-colour thresholds and the treemap range below speak
    # in %, so convert here once.
    for c in ("min_distance_to_barrier", "avg_distance_to_barrier"):
        if c in u.columns:
            u[c] = u[c] * 100

    total_cost = u["allocated_cost_ref"].sum()
    top1_share = (u["allocated_cost_ref"].iloc[0] / total_cost * 100) if not u.empty else 0
    top3_share = (u["allocated_cost_ref"].head(3).sum() / total_cost * 100) if not u.empty else 0
    top5_share = (u["allocated_cost_ref"].head(5).sum() / total_cost * 100) if not u.empty else 0

    # Concentration KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("# Underlyings", f"{len(u)}")
    k2.metric("Top 1 share",   f"{top1_share:.1f} %")
    k3.metric("Top 3 share",   f"{top3_share:.1f} %")
    k4.metric("Top 5 share",   f"{top5_share:.1f} %")

    st.markdown("&nbsp;", unsafe_allow_html=True)
    col_bar, col_tree = st.columns([3, 2])

    # Horizontal bar chart — sorted descending exposure
    with col_bar:
        st.markdown("**Allocated Cost by Underlying**")
        st.caption(
            "Bar length = capital invested in that underlying (allocated cost). "
            "Sorted from largest to smallest exposure."
        )
        bar_df = u.copy()
        # Colour bars by min distance-to-barrier (riskier = warmer)
        if "min_distance_to_barrier" in bar_df.columns:
            colors = [_dtb_color(v) for v in bar_df["min_distance_to_barrier"]]
        else:
            colors = [PRIMARY] * len(bar_df)
        fig_bar = go.Figure(go.Bar(
            x=bar_df["allocated_cost_ref"],
            y=bar_df["underlying"],
            orientation="h",
            marker_color=colors,
            text=[chf(v, 2) for v in bar_df["allocated_cost_ref"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Allocated cost: %{{x:,.0f}} {ref}<br>"
                "<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            height=max(280, 36 * len(bar_df)),
            margin=dict(t=10, b=10, l=10, r=70),
            xaxis_title=f"Cost ({ref})",
            yaxis=dict(autorange="reversed"),
            showlegend=False,
            separators=".'",
        )
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(
            f"Bar colour: <span style='color:{NEGATIVE}'>red</span> = "
            f"close to barrier, <span style='color:{WARNING}'>amber</span> "
            f"= moderate, <span style='color:{POSITIVE}'>green</span> = "
            f"comfortable cushion.",
            unsafe_allow_html=True,
        )

    # Treemap for visual sense of relative size
    with col_tree:
        st.markdown("**Cost Treemap**")
        st.caption(
            "Tile size = relative cost allocation; tile colour = the underlying's "
            "minimum distance to barrier (red = close to breach, green = cushion)."
        )
        fig_tree = px.treemap(
            u, path=["underlying"], values="allocated_cost_ref",
            color="min_distance_to_barrier",
            color_continuous_scale=PLOTLY_DIVERGING_SCALE,
            range_color=[-25, 50],
            hover_data={"weight": ":.2%", "n_products": True,
                         "min_distance_to_barrier": ":+.1f"},
        )
        fig_tree.update_layout(
            height=max(280, 36 * len(u)),
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_colorbar=dict(title="Dist. to<br>Barrier %", thickness=10),
        )
        st.plotly_chart(fig_tree, width="stretch")

    with st.expander("Underlying detail"):
        ut = u.copy()
        ut["weight"] = ut["weight"] * 100
        rename = {
            "underlying":              "Underlying",
            "isin":                    "ISIN",
            "price_ccy":               "CCY",
            "n_products":              "# Products",
            "allocated_cost_ref":      f"Allocated Cost {ref}",
            "avg_current_spot":        "Spot",
            "min_distance_to_barrier": "Min Dist. to Barrier %",
            "avg_distance_to_barrier": "Avg Dist. to Barrier %",
            "worst_of_count":          "# Worst-Of",
            "weight":                  "Weight %",
        }
        ut = ut.rename(columns=rename).round(2)
        styled = (
            ut.style
            .format({f"Allocated Cost {ref}": lambda v: chf(v, 2),
                      "Spot":                  lambda v: chf(v, 2),
                      "Weight %":              "{:.2f}",
                      "Min Dist. to Barrier %":"{:+.1f}",
                      "Avg Dist. to Barrier %":"{:+.1f}"})
            .background_gradient(cmap=DIVERGING_CMAP,
                                 subset=["Min Dist. to Barrier %"],
                                 vmin=-25, vmax=50)
        )
        st.dataframe(styled, width="stretch", hide_index=True,
                     height=_fit_height(len(ut)))

    _render_correlation_matrix(analytics, corr_df, u)


def _render_correlation_matrix(analytics, corr_df, underlying_df):
    with st.expander("Correlation matrix"):

        if corr_df is None or corr_df.empty:
            st.info(
                "Correlation data is unavailable. Multi-underlying analytics "
                "will fall back to identity correlation where required."
            )
            return

        corr = corr_df.copy()
        corr.index = corr.index.astype(str)
        corr.columns = corr.columns.astype(str)

        lookup = (
            underlying_df[["isin", "underlying"]]
            .dropna(subset=["isin"])
            .drop_duplicates("isin")
            .assign(isin=lambda x: x["isin"].astype(str))
        )
        label_map = {
            row["isin"]: _correlation_label(row["underlying"], row["isin"])
            for _, row in lookup.iterrows()
        }

        isins = [
            isin for isin in lookup["isin"].tolist()
            if isin in corr.index and isin in corr.columns
        ]
        if len(isins) < 2:
            st.info("At least two covered underlyings are needed for a matrix.")
            return

        sub = corr.loc[isins, isins].astype(float).clip(-1.0, 1.0).fillna(0.0)
        np.fill_diagonal(sub.values, 1.0)
        labels = [label_map.get(isin, isin) for isin in isins]

        off_diag = sub.to_numpy(copy=True)
        mask = ~np.eye(len(sub), dtype=bool)
        if np.allclose(off_diag[mask], 0.0):
            st.warning(
                "This matrix is effectively identity. That usually means "
                "correlations were defaulted or there was insufficient overlap."
            )

        numeric = sub.copy()
        numeric.index = labels
        numeric.columns = labels
        styled_matrix = (
            numeric.style
            .format("{:.2f}")
            .background_gradient(cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
        )
        st.dataframe(
            styled_matrix,
            width="stretch",
            height=_fit_height(len(numeric)),
        )


def _correlation_label(underlying, isin):
    name = str(underlying).strip() if pd.notna(underlying) else str(isin)
    if len(name) > 24:
        name = name[:21] + "..."
    return f"{name} ({str(isin)[-4:]})"


def _dtb_color(value):
    """Map distance-to-barrier (%) → palette colour."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NEUTRAL_GREY
    if value < 10:
        return NEGATIVE
    if value < 25:
        return WARNING
    return POSITIVE


# ──────────────────────────────────────────────────────────────────────────
# Maturity profile
# ──────────────────────────────────────────────────────────────────────────

def _render_maturity(analytics):
    st.markdown("### Maturity Profile")

    ref = analytics.reference_currency

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        bar_df = analytics.product_df[
            ["maturity_date", "product_type", "total_notional",
             "currency", "underlyings"]
        ].copy()
        bar_df["maturity_date"] = pd.to_datetime(bar_df["maturity_date"])
        bar_df = bar_df.sort_values("maturity_date")
        bar_df["mat_label"] = bar_df["maturity_date"].dt.strftime("%b %Y")
        # Two-line in-bar label: product type on the first row, the underlying(s)
        # on the second, so long basket names stay readable inside the bar.
        bar_df["product_label"] = (
            bar_df["product_type"] + "<br>" + bar_df["underlyings"]
        )

        fig = px.bar(
            bar_df,
            x="mat_label", y="total_notional", color="product_type",
            text="product_label",
            color_discrete_sequence=PRODUCT_TYPE_PALETTE,
            hover_data={"currency": True, "underlyings": True,
                         "product_label": False, "mat_label": False},
            labels={"mat_label": "Maturity", "total_notional": "Notional"},
        )
        fig.update_traces(textposition="inside", textangle=0,
                           insidetextanchor="middle",
                           textfont=dict(size=11, color="white"))
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(t=10, b=10, l=10, r=10),
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                         xanchor="left", x=0),
            xaxis_title=None,
            yaxis_title=f"Notional ({ref})",
        )
        st.plotly_chart(fig, width="stretch")

    with col_table:
        m = analytics.maturity_profile().copy()
        rename = {
            "maturity_bucket": "Bucket",
            "n_products":      "# Products",
            "total_cost":      f"Cost {ref}",
            "total_payoff":    f"Payoff {ref}",
            "total_pnl":       f"PnL {ref}",
        }
        m = m.rename(columns=rename).round(2)
        styled = (
            m.style
            .format({f"Cost {ref}": lambda v: chf(v, 2),
                      f"Payoff {ref}": lambda v: chf(v, 2),
                      f"PnL {ref}": lambda v: chf(v, 2, signed=True)})
            .background_gradient(cmap=DIVERGING_CMAP, subset=[f"PnL {ref}"],
                                 vmin=-(m[f"PnL {ref}"].abs().max() or 1),
                                 vmax= (m[f"PnL {ref}"].abs().max() or 1))
        )
        st.dataframe(styled, width="stretch", hide_index=True,
                     height=_fit_height(len(m)))


# ──────────────────────────────────────────────────────────────────────────
# Greeks / Risk
# ──────────────────────────────────────────────────────────────────────────

def _render_risk(greeks_df, pf_delta):
    st.markdown("### Risk · Greeks")
    st.markdown(
        f"<div style='color:{NEUTRAL_GREY}; font-size:0.85em;'>"
        f"Bump-and-reprice via Monte Carlo.  Δ per 1 % spot · ν per 1 pp vol · "
        f"θ per calendar day · ρ per 1 bp rate · corr per 1 pp uniform shift."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;", unsafe_allow_html=True)

    col_table, col_charts = st.columns([3, 2])

    with col_table:
        st.markdown("**Per-Product Greeks**")
        st.caption(
            "One row per (product, underlying) — a multi-underlying product "
            "appears multiple times.  Each Δ shows the FV change in the "
            "product currency for a 1 % spot move in that single underlying, "
            "holding all others fixed."
        )
        g = greeks_df.copy().rename(columns={
            "product_id":  "Product",
            "currency":    "CCY",
            "isin":        "ISIN",
            "underlying":  "Underlying",
            "delta_1pct":  "Δ (1% spot)",
            "vega_1pp":    "ν (1pp vol)",
            "theta":       "θ (daily)",
            "rho":         "ρ (1bp rate)",
            "corr_sens":   "corr (1pp)",
        })
        # ranges for symmetric gradients
        def _sym(col):
            try:
                m = g[col].abs().max()
                return -m or -1, m or 1
            except Exception:
                return -1, 1

        styled = g.style.format(
            {c: (lambda v: chf(v, 2, signed=True))
             for c in ["Δ (1% spot)", "ν (1pp vol)",
                       "θ (daily)", "ρ (1bp rate)", "corr (1pp)"]
             if c in g.columns},
            na_rep="—",
        )
        for col in ["Δ (1% spot)", "ν (1pp vol)", "θ (daily)"]:
            if col in g.columns:
                lo, hi = _sym(col)
                styled = styled.background_gradient(
                    cmap=DIVERGING_CMAP, subset=[col], vmin=lo, vmax=hi,
                )
        st.dataframe(styled, width="stretch", hide_index=True,
                     height=_fit_height(len(g)))

    with col_charts:
        st.markdown("**Portfolio Delta by Underlying**")
        st.caption(
            "One row per *unique underlying*.  The per-product Δ rows above "
            "are summed: this is your net directional exposure to each name. "
            "Positive = long the underlying; negative = short."
        )
        _delta_bar(pf_delta)


def _delta_bar(pf_delta):
    pf = pf_delta.sort_values("total_delta_1pct").copy()
    colors = [POSITIVE if d >= 0 else NEGATIVE
              for d in pf["total_delta_1pct"]]
    fig = go.Figure(go.Bar(
        x=pf["total_delta_1pct"],
        y=pf["underlying"],
        orientation="h",
        marker_color=colors,
        text=[chf(v, 2, signed=True) for v in pf["total_delta_1pct"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Net Δ: %{x:+,.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        template="plotly_dark",
        separators=".'",
        height=max(260, 38 * len(pf)),
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis_title="FV change per 1 % spot move",
        yaxis_title=None,
        showlegend=False,
    )
    fig.add_vline(x=0, line=dict(color=NEUTRAL_GREY, width=1, dash="dot"))
    st.plotly_chart(fig, width="stretch")
