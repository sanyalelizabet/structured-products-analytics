import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.formatting import chf


# Palette — kept in sync with ``app/views/portfolio.py``.
_LINE_PALETTE = [
    "#4E79A7", "#76A65A", "#9C755F", "#C9A961", "#2A3F5F", "#7A8797",
    "#3FA29E", "#8E7CC3", "#A35F4A", "#5E8A6E", "#7E91AB",
]


def render(portfolio, df, analytics, valuation_date, vol_map, beta_map):

    overview_cols = ["product_id", "product_type", "underlyings", "notional", "currency", "coupon", "maturity_date"]

    st.subheader("Portfolio Overview")
    st.caption(
        "Projected at maturity assuming current spot prices remain unchanged. "
        "Coupons accrued over full product life."
    )

    # Merge fair value into overview
    fv_cols = df[["product_id", "fair_value", "fair_value_pct"]].copy()
    fv_cols["fair_value_pct"] = fv_cols["fair_value_pct"] * 100

    overview_display = portfolio[overview_cols].copy()
    overview_display["coupon"] = overview_display["coupon"] * 100
    overview_display = overview_display.merge(fv_cols, on="product_id", how="left")
    overview_display["notional"]   = overview_display["notional"].map(lambda v: chf(v, 2))
    overview_display["fair_value"] = overview_display["fair_value"].map(lambda v: chf(v, 2))

    st.dataframe(
        overview_display,
        width="stretch",
        hide_index=True,
        column_config={
            "product_id":     st.column_config.TextColumn("Product ID",      width="medium"),
            "product_type":   st.column_config.TextColumn("Type",            width="small"),
            "underlyings":    st.column_config.ListColumn("Underlyings",     width="medium"),
            "notional":       st.column_config.TextColumn("Notional",        width="small"),
            "currency":       st.column_config.TextColumn("CCY",             width="small"),
            "coupon":         st.column_config.NumberColumn("Coupon (%)",    width="small", format="%.2f"),
            "maturity_date":  st.column_config.TextColumn("Maturity",        width="small"),
            "fair_value":     st.column_config.TextColumn("Fair Value",      width="small"),
            "fair_value_pct": st.column_config.NumberColumn("Fair Value (%)", width="small", format="%.2f"),
        }
    )

    # =========================
    # Product Selection
    # =========================
    selected_product = st.selectbox("Select product", portfolio["product_id"].tolist())
    row = df[df["product_id"] == selected_product].iloc[0]

    df_display = df[[
        "product_id", "maturity_date", "days_to_expiry", "product_type",
        "total_payoff", "total_cost", "break_even", "ytm", "ytm_today", "return_pct", "distance_to_barrier"
    ]].rename(columns={
        "product_id":         "Product ID",
        "maturity_date":      "Maturity Date",
        "days_to_expiry":     "Days to Expiry",
        "product_type":       "Product Type",
        "total_payoff":       "Total Payoff",
        "total_cost":         "Total Cost",
        "break_even":         "Break-even",
        "ytm":                "YTM from Purchase (%)",
        "ytm_today":          "YTM from Today (%)",
        "return_pct":         "Total Return (%)",
        "distance_to_barrier":"Downside to Barrier (%)",
    })
    row_selected = df_display[df_display["Product ID"] == selected_product].iloc[0]

    # =========================
    # Key Metrics
    # =========================
    st.subheader("Key Metrics")
    vd_label = valuation_date.strftime('%d.%m.%Y') if valuation_date is not None else "Using default values"
    st.caption(f"Valuation Date: {vd_label}")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Projected P&L (flat spot)",
        chf(row['pnl'], 2),
        delta=f"{chf(row['pnl_delta'], 2, signed=True)} vs yesterday" if row["pnl_delta"] is not None else None
    )
    col2.metric("Total Payoff", chf(row['total_payoff'], 2))
    col3.metric("Days to Expiry", f"{int(row['days_to_expiry'])}")

    is_cpn = str(row["product_type"]).upper() == "CPN"

    col4, col5, col6 = st.columns(3)
    col4.metric("YTM from Today (%)", f"{row['ytm_today']:.2f}")
    if is_cpn:
        # No barrier on a vanilla CPN — show protection floor instead.
        prod_row_full = portfolio[portfolio["product_id"] == selected_product].iloc[0]
        protection = float(prod_row_full.get("protection_pct", float("nan")))
        col5.metric("Capital Protection", f"{protection * 100:.0f}%")
    else:
        col5.metric(
            "Downside to Barrier (%)",
            f"{row['distance_to_barrier']:.2f}%",
            delta=f"{row['distance_delta']*100:+.2f}pp vs yesterday" if row["distance_delta"] is not None else None,
            delta_color="normal"
        )
    col6.metric("Worst Underlying", row["worst_underlying"])

    if is_cpn:
        prod_row_full = portfolio[portfolio["product_id"] == selected_product].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Capital Protection", f"{float(prod_row_full['protection_pct']) * 100:.0f}%")
        c2.metric("Participation", f"{float(prod_row_full['participation_pct']) * 100:.0f}%")
        c3.metric(
            "Break-even Spot Performance",
            "0.00%" if row["break_even"] == 0.0
            else f"{(row['break_even'] - 1.0) * 100:+.2f}%",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Product description (after Key Metrics — full name, underlyings,
    # maturity, coupon if any). Same format for every product type so the
    # reader gets a consistent one-line "what is this".
    # ─────────────────────────────────────────────────────────────────────
    prod_row_full = portfolio[portfolio["product_id"] == selected_product].iloc[0]
    _render_product_description(prod_row_full, row)

    # =========================
    # Term Sheet (uniform across product types)
    # =========================
    _render_term_sheet(prod_row_full)

    # =========================
    # Product Detail + Underlying Metrics
    # =========================
    st.subheader("Product Detail")

    prod_row    = portfolio[portfolio["product_id"] == selected_product].iloc[0]
    und_isins   = prod_row["underlying_isins"]
    und_names   = prod_row["underlyings"]
    und_spots   = prod_row["current_spots"]
    und_strikes = prod_row["strike"]

    und_rows = []
    for name, isin, spot, strike in zip(und_names, und_isins, und_spots, und_strikes):
        ytd = analytics.calc_ytd(isin, spot)
        und_rows.append({
            "Underlying": name,
            "Spot":    round(spot, 2),
            "Strike":  round(strike, 2),
            "Vol (%)": round(vol_map.get(isin, float("nan")) * 100, 1),
            "Beta":    beta_map.get(isin, float("nan")),
            "YTD (%)": round(ytd, 2) if ytd is not None else "n/a",
        })
    und_df = pd.DataFrame(und_rows)

    col_detail, col_chart = st.columns([1, 2])

    with col_detail:
        st.markdown("**Product Detail**")
        st.caption("Key metrics for the selected product.")
        detail = pd.DataFrame({"Metric": row_selected.index, "Value": row_selected.values})

        def _fmt(x):
            if isinstance(x, (int, float, np.floating)):
                if pd.isna(x):
                    return "n/a"
                return chf(x, 2)
            return str(x)

        detail["Value"] = detail["Value"].apply(_fmt)
        st.dataframe(detail, width="stretch", hide_index=True)

    with col_chart:
        st.markdown("**Underlying Price History**")
        st.caption(
            "Last ~2 years, normalised to base 100. Dashed lines mark each strike."
        )
        _render_underlying_prices(analytics, prod_row)

    # =========================
    # Underlying Metrics (full width below)
    # =========================
    st.markdown("**Underlying Metrics**")
    st.caption("Spot, strike, vol, β and YTD per underlying for this product.")
    st.dataframe(
        und_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Spot":    st.column_config.NumberColumn("Spot",    format="%.2f"),
            "Strike":  st.column_config.NumberColumn("Strike",  format="%.2f"),
            "Vol (%)": st.column_config.NumberColumn("Vol (%)", format="%.1f"),
            "Beta":    st.column_config.NumberColumn("Beta",    format="%.2f"),
            "YTD (%)": st.column_config.NumberColumn("YTD (%)", format="%.2f"),
        }
    )


# Imported from the schema module so the dropdown in the entry form, the
# product-detail header, and any future surface all read the same labels.
from src.portfolio_entry import PRODUCT_TYPE_FULL_NAME  # noqa: E402


def _row_get(prod_row, field):
    """Return ``prod_row[field]`` or None for missing/NaN/empty values."""
    if field not in prod_row.index:
        return None
    val = prod_row[field]
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str) and not val.strip():
        return None
    return val


def _render_product_description(prod_row, analytics_row):
    """One-line product description directly below Key Metrics.

    Format: **Full product type (ABBR)** · Underlying(s) · Maturity DD.MM.YYYY · Coupon X.XX%

    Coupon segment is omitted when the row has no coupon (zero-coupon
    notes, CPNs without periodic coupons).
    """
    ptype       = str(prod_row.get("product_type", "")).upper()
    full_name   = PRODUCT_TYPE_FULL_NAME.get(ptype, ptype or "Structured Product")
    underlyings = ", ".join(prod_row.get("underlyings", []) or [])
    maturity    = prod_row.get("maturity_date", "")
    coupon      = prod_row.get("coupon", 0.0) or 0.0

    try:
        maturity_fmt = pd.Timestamp(maturity).strftime("%d.%m.%Y")
    except (ValueError, TypeError):
        maturity_fmt = str(maturity)

    # full_name already includes the abbreviation, e.g. "Barrier Reverse
    # Convertible (BRC)" — don't append ({ptype}) again or it doubles up.
    parts = [
        f"<strong>{full_name}</strong>",
        f"Underlying: {underlyings}",
        f"Maturity: {maturity_fmt}",
    ]
    if coupon and coupon > 0:
        parts.append(f"Coupon: {coupon * 100:.2f}% p.a.")

    # Rendered as a title-sized line so the product identity stands out
    # between the Key Metrics block and the Term Sheet panel.
    st.markdown(
        f"<div style='font-size: 1.4rem; line-height: 1.5; "
        f"margin: 0.5rem 0 0.75rem 0;'>{' · '.join(parts)}</div>",
        unsafe_allow_html=True,
    )


def _render_term_sheet(prod_row):
    """Unified term-sheet panel for any product type.

    Three columns:
      * Issuer chain      — issuer / rating / country / guarantor / keep-well
      * Product terms     — denomination, position, notional, dates, coupon
                            cadence, plus product-specific contractual fields
                            (barrier, autocall, capital protection)
      * Disclosures       — issue size, bond NPV at issue, implied IRR,
                            issue / payment / final-fixing / last-trading dates

    Missing fields are silently skipped, so partially-populated rows still
    render cleanly.  This replaces the CPN-only panel — every product now
    shows the same shape of term-sheet info.
    """
    ptype = str(prod_row.get("product_type", "")).upper()

    issuer_chain = [
        ("Issuer",            _row_get(prod_row, "issuer")),
        ("Issuer rating",     _row_get(prod_row, "issuer_rating")),
        ("Issuer country",    _row_get(prod_row, "issuer_country")),
        ("Guarantor",         _row_get(prod_row, "guarantor")),
        ("Guarantor rating",  _row_get(prod_row, "guarantor_rating")),
        ("Keep-well",         _row_get(prod_row, "keep_well_agreement")),
    ]

    # ── Product terms — fields that drive the payoff ─────────────────────
    denom    = _row_get(prod_row, "denomination")
    units    = _row_get(prod_row, "position_units")
    notional = _row_get(prod_row, "notional")
    coupon   = _row_get(prod_row, "coupon")
    coupon_dates = _row_get(prod_row, "coupon_dates")
    day_count = _row_get(prod_row, "day_count")
    init_fix = _row_get(prod_row, "initial_fixing_date")
    purchase = _row_get(prod_row, "purchase_date")
    maturity = _row_get(prod_row, "maturity_date")

    terms = [
        ("Denomination",       chf(denom, 2) if isinstance(denom, (int, float)) else denom),
        ("Position units",     units),
        ("Total notional",     chf(notional, 2) if isinstance(notional, (int, float)) else notional),
        ("Day count",          day_count),
        ("Initial fixing",     init_fix),
        ("Purchase date",      purchase),
        ("Maturity",           maturity),
    ]
    if coupon is not None and coupon > 0:
        terms.append(("Coupon (p.a.)", f"{coupon * 100:.4f}%"))
    if coupon_dates:
        try:
            n_periods = len(coupon_dates)
            terms.append(("Coupon periods", n_periods))
        except TypeError:
            pass

    # Product-specific contractual fields
    if ptype in ("BRC", "MBRC", "AC_BRC"):
        bp = _row_get(prod_row, "barrier_pct")
        if bp is not None:
            terms.append(("Barrier (% of strike)", f"{bp * 100:.2f}%"))
    if ptype == "AC_BRC":
        obs = _row_get(prod_row, "autocall_obs_dates")
        if obs:
            terms.append(("Autocall observation dates", ", ".join(map(str, obs))))
        trig = _row_get(prod_row, "autocall_trigger_pct")
        if trig is not None:
            terms.append(("Autocall trigger (× strike)", f"{trig:.4f}"))
    if ptype == "CPN":
        ip = _row_get(prod_row, "issue_price")
        if ip is not None:
            terms.append(("Issue price", chf(ip, 2)))
        cp_amt = _row_get(prod_row, "capital_protection_amount")
        if cp_amt is not None:
            terms.append(("Capital protection", chf(cp_amt, 2)))
        pp = _row_get(prod_row, "protection_pct")
        if pp is not None:
            terms.append(("Protection level", f"{pp * 100:.2f}%"))
        part = _row_get(prod_row, "participation_pct")
        if part is not None:
            terms.append(("Participation", f"{part * 100:.2f}%"))
        b = _row_get(prod_row, "number_of_underlyings")
        if b is not None:
            terms.append(("Number of Underlyings (B)", f"{b:.5f}"))

    # ── Disclosures (issuer-disclosed, mostly CPN today) ─────────────────
    disclosures = [
        ("Issue size",        _row_get(prod_row, "issue_size")),
        ("Bond NPV at issue", _row_get(prod_row, "bond_npv_at_issue")),
        ("Implied IRR at issue",
         f"{_row_get(prod_row, 'implied_irr_at_issue') * 100:.4f}%"
         if _row_get(prod_row, "implied_irr_at_issue") is not None else None),
        ("Payment date",      _row_get(prod_row, "payment_date")),
        ("Final fixing",      _row_get(prod_row, "final_fixing_date")),
        ("Last trading day",  _row_get(prod_row, "last_trading_day")),
    ]

    def _tidy(pairs):
        # Render every value as a string so the resulting column is uniformly
        # typed.  A mixed object column (e.g. the day-count "30/360" string next
        # to numeric fields) makes Streamlit's Arrow conversion infer int64 and
        # raise ArrowInvalid.  Dates are shown date-only for readability.
        def _as_text(v):
            if hasattr(v, "strftime"):           # date / datetime / Timestamp
                return pd.Timestamp(v).strftime("%Y-%m-%d")
            return str(v)
        return [{"Field": k, "Value": _as_text(v)} for k, v in pairs if v is not None]

    issuer_rows = _tidy(issuer_chain)
    term_rows   = _tidy(terms)
    disc_rows   = _tidy(disclosures)

    if not (issuer_rows or term_rows or disc_rows):
        return

    with st.expander("Term sheet", expanded=True):
        # Layout: 2 cols if no disclosures, 3 cols otherwise — keeps the
        # display tight for products that don't have issuer-disclosure data.
        if disc_rows:
            col_a, col_b, col_c = st.columns(3)
        else:
            col_a, col_b = st.columns(2)
            col_c = None

        if term_rows:
            with col_a:
                st.markdown("**Product terms**")
                st.dataframe(pd.DataFrame(term_rows), width="stretch", hide_index=True)
        if issuer_rows:
            with col_b:
                st.markdown("**Issuer chain**")
                st.dataframe(pd.DataFrame(issuer_rows), width="stretch", hide_index=True)
        if disc_rows and col_c is not None:
            with col_c:
                st.markdown("**Disclosures**")
                st.dataframe(pd.DataFrame(disc_rows), width="stretch", hide_index=True)


def _render_underlying_prices(analytics, prod_row):
    """Plot historical prices of each underlying for the selected product.

    Lines are normalised to base 100 at the first available date so
    multiple price scales (e.g. CHF Nestlé vs USD Coca-Cola) can share
    one chart.  Strike levels are drawn as dashed horizontal lines in
    the same colour as their underlying, also normalised to base 100.
    """
    db = getattr(analytics, "price_db", None)
    if db is None or db.empty:
        return

    isins   = list(prod_row["underlying_isins"])
    names   = list(prod_row["underlyings"])
    strikes = list(prod_row["strike"])
    spots   = list(prod_row["current_spots"])

    sub = db[db["isin"].isin(isins)].copy()
    if sub.empty:
        return
    sub["date"] = pd.to_datetime(sub["date"])
    # Trim to the most recent ~2 years so the chart isn't dominated by old data.
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=2)
    sub = sub[sub["date"] >= cutoff]
    if sub.empty:
        return

    fig = go.Figure()
    for idx, (isin, name) in enumerate(zip(isins, names)):
        series = (
            sub[sub["isin"] == isin]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
        )
        if series.empty:
            continue
        color = _LINE_PALETTE[idx % len(_LINE_PALETTE)]
        spot0 = float(series["price"].iloc[0])
        norm  = series["price"] / spot0 * 100.0

        fig.add_trace(go.Scatter(
            x=series["date"], y=norm,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            customdata=series["price"].values,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "%{x|%d %b %Y}<br>"
                "Price: %{customdata:.2f}<br>"
                "Normalised: %{y:.1f}<extra></extra>"
            ),
        ))

        # Strike line — normalised to the same base
        try:
            strike = float(strikes[idx])
            fig.add_hline(
                y=strike / spot0 * 100.0,
                line=dict(color=color, dash="dash", width=1),
                opacity=0.45,
                annotation_text=f"{name} strike",
                annotation_position="right",
                annotation_font=dict(color=color, size=9),
            )
        except (IndexError, TypeError, ValueError):
            pass

    fig.add_hline(y=100, line=dict(color="grey", dash="dot", width=1), opacity=0.4)
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=30, b=30, l=40, r=110),
        xaxis_title="Date",
        yaxis_title="Normalised Price (base 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
