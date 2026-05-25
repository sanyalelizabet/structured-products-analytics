"""Gemini insight payloads and prompts for analytics views."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


GEMINI_MODEL = "gemini-2.5-flash"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_MAX_ENTRIES = 128


def payload_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _safe_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict]:
    out = frame.copy()
    if limit is not None:
        out = out.head(limit)
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _round_obj(obj):
    if isinstance(obj, dict):
        return {str(k): _round_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [_round_obj(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_round_obj(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [_round_obj(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        if pd.isna(obj):
            return None
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None:
        return None
    if not isinstance(obj, (str, bytes)):
        try:
            missing = pd.isna(obj)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    return obj


def _stable_json(payload: dict) -> str:
    return json.dumps(
        _round_obj(payload),
        default=str,
        ensure_ascii=False,
        sort_keys=True,
    )


def _gemini_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception as exc:
        raise RuntimeError("Missing GEMINI_API_KEY in Streamlit secrets.") from exc


def _call_gemini(prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=_gemini_api_key())
    try:
        from google.genai import types

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
    except ImportError:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def build_portfolio_page_payload(
    *,
    holdings_table: pd.DataFrame,
    reference_currency: str,
    analytics: Any,
    product_df: pd.DataFrame,
    greeks_df: pd.DataFrame,
    pf_delta: pd.DataFrame,
    valuation_date,
) -> str:
    cols = [
        "product_id", "product_type", "underlyings", "currency",
        "maturity_date", "total_notional", "cost_ref", "weight_pct",
        "payoff_ref", "pnl_ref", "return_pct", "distance_to_barrier",
        "fair_value", "fair_value_pct",
    ]
    payload = holdings_table[cols].copy()
    payload = payload.sort_values("cost_ref", ascending=False)
    payload = payload.round({
        "total_notional": 2,
        "cost_ref": 2,
        "weight_pct": 4,
        "payoff_ref": 2,
        "pnl_ref": 2,
        "return_pct": 4,
        "distance_to_barrier": 2,
        "fair_value": 2,
        "fair_value_pct": 4,
    })
    payload = payload.where(pd.notna(payload), None)

    numeric = holdings_table.copy()
    numeric["maturity_dt"] = pd.to_datetime(numeric["maturity_date"], errors="coerce")
    barrier_rows = numeric.dropna(subset=["distance_to_barrier"]).sort_values(
        "distance_to_barrier",
    )
    summary = {
        "n_holdings": int(len(numeric)),
        "total_cost_ref": round(float(numeric["cost_ref"].sum()), 2),
        "top_holding": (
            numeric.sort_values("weight_pct", ascending=False)
            [["product_id", "underlyings", "weight_pct"]]
            .head(1)
            .to_dict(orient="records")
        ),
        "top_3_weight_pct": round(
            float(numeric["weight_pct"].nlargest(min(3, len(numeric))).sum()), 2
        ),
        "return_range_pct": {
            "min": round(float(numeric["return_pct"].min()), 2),
            "max": round(float(numeric["return_pct"].max()), 2),
        },
        "closest_barriers": (
            barrier_rows[[
                "product_id", "underlyings", "distance_to_barrier",
                "weight_pct", "return_pct",
            ]]
            .head(3)
            .round(2)
            .to_dict(orient="records")
        ),
        "barrier_watch_count_below_20pct": int(
            (numeric["distance_to_barrier"] < 20).sum()
        ),
        "maturity_window": {
            "earliest": numeric["maturity_dt"].min().strftime("%Y-%m-%d"),
            "latest": numeric["maturity_dt"].max().strftime("%Y-%m-%d"),
        },
        "fair_value_below_100pct": (
            numeric[numeric["fair_value_pct"] < 100]
            [["product_id", "underlyings", "fair_value_pct"]]
            .round(2)
            .to_dict(orient="records")
        ),
    }

    pf_metrics = analytics.total_portfolio_metrics()
    quick_stats = {
        "n_products": int(len(analytics.product_df)),
        "n_underlyings": int(
            analytics.product_df["underlyings"].nunique()
            if "underlyings" in analytics.product_df.columns else 0
        ),
        "n_currencies": int(
            analytics.product_df["currency"].nunique()
            if "currency" in analytics.product_df.columns else 0
        ),
        "min_distance_to_barrier_pct": (
            round(float(product_df["distance_to_barrier"].min()), 2)
            if "distance_to_barrier" in product_df.columns else None
        ),
    }
    maturity = analytics.maturity_profile().round(2)
    concentration = analytics.underlying_lookthrough().copy()
    if "min_distance_to_barrier" in concentration.columns:
        concentration["min_distance_to_barrier"] *= 100
    if "avg_distance_to_barrier" in concentration.columns:
        concentration["avg_distance_to_barrier"] *= 100
    concentration = concentration.sort_values(
        "allocated_cost_ref", ascending=False,
    ).round(4)
    greeks = greeks_df.copy().round(4)
    delta = pf_delta.copy().sort_values(
        "total_delta_1pct",
        key=lambda s: s.abs(),
        ascending=False,
    ).round(4)

    return _stable_json(
        {
            "reference_currency": reference_currency,
            "valuation_date": (
                valuation_date.strftime("%Y-%m-%d")
                if valuation_date is not None else None
            ),
            "units": {
                "weight_pct": "percent of portfolio cost",
                "return_pct": "percent",
                "distance_to_barrier": "percent",
                "fair_value_pct": "percent of notional",
                "greeks": "fair-value change in product currency for stated bump",
            },
            "derived_summary": summary,
            "portfolio_metrics": pf_metrics,
            "quick_stats": quick_stats,
            "holdings": payload.to_dict(orient="records"),
            "maturity_profile": _safe_records(maturity),
            "underlying_concentration": _safe_records(concentration),
            "product_greeks": _safe_records(greeks),
            "portfolio_delta_by_underlying": _safe_records(delta),
        },
    )


@st.cache_data(
    ttl=_CACHE_TTL_SECONDS,
    max_entries=_CACHE_MAX_ENTRIES,
    show_spinner=False,
)
def generate_portfolio_insight(payload_json: str) -> str:
    prompt = f"""
You are a structured-products desk specialist explaining a client portfolio in
plain language.

These holdings are not ordinary stocks or funds. Their outcome depends on
product terms such as barriers, coupons, maturity dates, linked underlyings,
projected payoff, and today's fair value. A product can look profitable and
still carry meaningful risk if the linked shares fall together, move closer to
their barriers, or if today's fair value is below the expected maturity payoff.

Use the Portfolio page JSON and derived_summary to write exactly 4 sentences:
1. Give the overall portfolio read: healthy, mixed, or under pressure, and why.
2. Explain the main structured-product risk: barrier distance, concentration in
   a few products or underlyings, short maturity clustering, or issuer/credit
   exposure if visible.
3. Explain dependency between holdings: products linked to shares that may move
   together can lose diversification benefit; discuss correlation qualitatively
   unless exact correlation numbers are supplied.
4. End with what should be monitored next, without giving buy, sell, or hold
   advice.

Rules:
- Use simple client language with desk-level judgment.
- Do not use headings, bullet points, questions, greetings, or markdown.
- Do not recommend buying, selling, holding, switching, or changing exposure.
- Do not invent facts, market views, client objectives, taxes, fees, or exact
  correlations that are not supplied.
- Do not say "the JSON", "the table", or "the data".
- Explain technical terms briefly when used: barrier, fair value, projected
  payoff, concentration, maturity, and correlation.
- Keep each sentence short enough that a non-finance client can follow it.
- Mention exact numbers only when they clarify the key risk.

Portfolio page JSON:
{payload_json}
""".strip()
    return _call_gemini(prompt)


def _engine_inputs_records(portfolio, beta_map, vol_map, risk_free_rates,
                           market_shock) -> list[dict]:
    rows = []
    seen = set()
    for _, prow in portfolio.iterrows():
        rf = float(risk_free_rates.get(prow["currency"], 0.0))
        for isin, name in zip(prow["underlying_isins"], prow["underlyings"]):
            if isin in seen:
                continue
            seen.add(isin)
            beta = float(beta_map.get(isin, 1.0))
            vol = float(vol_map.get(isin, 0.15))
            rows.append({
                "underlying": name,
                "isin": isin,
                "currency": prow["currency"],
                "beta": round(beta, 4),
                "annual_volatility": round(vol, 4),
                "risk_free_rate": round(rf, 4),
                "per_event_shock_beta_scaled_pct": round(float(market_shock) * beta, 2),
            })
    return rows


def _pnl_distribution_records(samples_by_ccy: dict) -> list[dict]:
    rows = []
    for ccy, samples in samples_by_ccy.items():
        arr = np.asarray(samples, dtype=float)
        p5 = float(np.percentile(arr, 5))
        rows.append({
            "currency": ccy,
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "p5": round(p5, 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "expected_shortfall_5": round(
                float(arr[arr <= p5].mean()) if len(arr) >= 20 else float(arr.min()),
                2,
            ),
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
            "n_paths": int(len(arr)),
        })
    return rows


def _asset_path_records(asset_paths: dict, portfolio) -> list[dict]:
    isin_name = {
        isin: name
        for _, row in portfolio.iterrows()
        for isin, name in zip(row["underlying_isins"], row["underlyings"])
    }
    rows = []
    for isin, path_df in asset_paths.items():
        df = path_df.copy()
        terminal = df.iloc[-1]
        min_median = df.loc[df["median"].idxmin()]
        max_median = df.loc[df["median"].idxmax()]
        rows.append({
            "isin": isin,
            "underlying": isin_name.get(isin, isin),
            "start_median": round(float(df["median"].iloc[0]), 2),
            "terminal_median": round(float(terminal["median"]), 2),
            "terminal_p5": round(float(terminal["lower_1sd"]), 2)
            if "lower_1sd" in df.columns else None,
            "terminal_p95": round(float(terminal["upper_1sd"]), 2)
            if "upper_1sd" in df.columns else None,
            "lowest_median": round(float(min_median["median"]), 2),
            "lowest_median_date": str(pd.to_datetime(min_median["date"]).date()),
            "highest_median": round(float(max_median["median"]), 2),
            "highest_median_date": str(pd.to_datetime(max_median["date"]).date()),
        })
    return rows


def _path_summary_records(paths: dict, *, labels: dict | None = None) -> list[dict]:
    rows = []
    labels = labels or {}
    for code, path_df in paths.items():
        df = path_df.copy()
        if df.empty or "median" not in df.columns:
            continue
        terminal = df.iloc[-1]
        min_median = df.loc[df["median"].idxmin()]
        max_median = df.loc[df["median"].idxmax()]
        start = float(df["median"].iloc[0])
        terminal_median = float(terminal["median"])
        rows.append({
            "code": code,
            "label": labels.get(code, code),
            "start_median": round(start, 2),
            "terminal_median": round(terminal_median, 2),
            "terminal_move_pct": (
                round((terminal_median / start - 1.0) * 100, 2)
                if start else None
            ),
            "terminal_p5": round(float(terminal["p5"]), 2)
            if "p5" in df.columns else None,
            "terminal_p95": round(float(terminal["p95"]), 2)
            if "p95" in df.columns else None,
            "lowest_median": round(float(min_median["median"]), 2),
            "lowest_median_date": str(pd.to_datetime(min_median["date"]).date()),
            "highest_median": round(float(max_median["median"]), 2),
            "highest_median_date": str(pd.to_datetime(max_median["date"]).date()),
        })
    return rows


def _reference_pnl_summary(
    samples_by_ccy: dict,
    cost_by_ccy: dict,
    fx_rates: dict | None,
    reference_currency: str | None,
) -> tuple[list[dict], np.ndarray | None]:
    if not samples_by_ccy or reference_currency is None:
        return [], None

    total = None
    total_cost = 0.0
    for ccy, samples in samples_by_ccy.items():
        if ccy == reference_currency:
            rate = 1.0
        else:
            rate = fx_rates.get((ccy, reference_currency)) if fx_rates else None
        if rate is None:
            continue

        arr = np.asarray(samples, dtype=float) * float(rate)
        total = arr.copy() if total is None else total + arr
        total_cost += float(cost_by_ccy.get(ccy, 0.0)) * float(rate)

    if total is None:
        return [], None

    p5 = float(np.percentile(total, 5))
    p95 = float(np.percentile(total, 95))
    es5 = (
        float(total[total <= p5].mean())
        if len(total) >= 20 else float(total.min())
    )
    row = {
        "reference_currency": reference_currency,
        "n_currencies": len(samples_by_ccy),
        "total_cost_ref": round(total_cost, 2),
        "pnl_mean": round(float(total.mean()), 2),
        "pnl_median": round(float(np.median(total)), 2),
        "pnl_p5": round(p5, 2),
        "pnl_p95": round(p95, 2),
        "pnl_es5": round(es5, 2),
        "portfolio_return_mean_pct": (
            round(float(total.mean() / total_cost * 100), 2)
            if total_cost else 0.0
        ),
        "portfolio_return_p5_pct": (
            round(float(p5 / total_cost * 100), 2) if total_cost else 0.0
        ),
    }
    return [row], total


def _factor_decomposition_records(res, portfolio, loadings: dict) -> list[dict]:
    factor_paths = res.get("factor_paths", {})
    asset_paths = res.get("asset_paths", {})
    factor_dlogs = {}
    for code, df in factor_paths.items():
        if df.empty or float(df["median"].iloc[0]) <= 0:
            continue
        factor_dlogs[code] = float(np.log(df["median"].iloc[-1] / df["median"].iloc[0]))

    rows = []
    for isin, path_df in asset_paths.items():
        if path_df.empty or float(path_df["median"].iloc[0]) <= 0:
            continue
        loading = loadings.get(isin, {}) or {}
        betas = loading.get("betas", {}) or {}
        contributions = {
            code: float(betas.get(code, 1.0 if code == "MKT" else 0.0))
            * factor_dlogs.get(code, 0.0)
            for code in factor_dlogs
        }
        systematic_total = sum(contributions.values())
        terminal_dlog = float(np.log(
            path_df["median"].iloc[-1] / path_df["median"].iloc[0],
        ))
        row = {
            "isin": isin,
            "underlying": _isin_name(portfolio, isin),
            "total_median_return_log_pct": round(terminal_dlog * 100, 2),
            "systematic_factor_contribution_pct": round(systematic_total * 100, 2),
            "idiosyncratic_residual_pct": round(
                (terminal_dlog - systematic_total) * 100,
                2,
            ),
        }
        for code, value in contributions.items():
            row[f"{code}_contribution_pct"] = round(value * 100, 2)
        rows.append(row)
    rows.sort(key=lambda r: r["total_median_return_log_pct"])
    return rows


def _isin_name(portfolio, isin: str) -> str:
    for _, row in portfolio.iterrows():
        for code, name in zip(row["underlying_isins"], row["underlyings"]):
            if code == isin:
                return name
    return isin


def _loadings_records(portfolio, loadings: dict) -> list[dict]:
    isins = sorted({
        isin
        for _, row in portfolio.iterrows()
        for isin in row["underlying_isins"]
    })
    rows = []
    factor_codes = sorted({
        code
        for loading in loadings.values()
        for code in (loading.get("betas", {}) or {})
    })
    for isin in isins:
        loading = loadings.get(isin, {}) or {}
        betas = loading.get("betas", {}) or {}
        row = {
            "isin": isin,
            "underlying": _isin_name(portfolio, isin),
            "alpha": loading.get("alpha"),
            "idio_vol": loading.get("idio_vol"),
            "r_squared": loading.get("r_squared"),
            "n_obs": loading.get("n_obs"),
        }
        for code in factor_codes:
            row[f"beta_{code}"] = betas.get(code)
        rows.append(row)
    return rows


def _factor_premium_records(premiums_by_method: dict | None) -> list[dict]:
    if not premiums_by_method:
        return []
    rows = []
    for method, frame in premiums_by_method.items():
        if frame is None or frame.empty:
            continue
        out = frame.copy()
        out = out.reset_index(names="regime")
        out.insert(0, "method", method)
        rows.extend(_safe_records(out.round(6)))
    return rows


def _correlation_records(corr_df: pd.DataFrame, limit: int = 12) -> list[dict]:
    if corr_df is None or corr_df.empty:
        return []
    rows = []
    cols = list(corr_df.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            try:
                value = float(corr_df.loc[left, right])
            except Exception:
                continue
            rows.append({
                "left": str(left),
                "right": str(right),
                "correlation": round(value, 4),
            })
    rows.sort(key=lambda r: abs(r["correlation"]), reverse=True)
    return rows[:limit]


def build_stress_testing_payload(
    *,
    res,
    scenario,
    portfolio,
    beta_map,
    vol_map,
    risk_free_rates,
    corr_df,
    selected_preset,
    initial_state,
    recovery,
) -> str:
    product_df = res["product_df"].copy()
    product_summary = product_df.drop(columns=["pnl_samples"], errors="ignore").round(4)
    top_loss_cols = [
        "product_id", "currency", "worst_underlying", "settlement_type",
        "barrier_breach_freq", "pnl_mean", "pnl_p5", "return_p5_pct",
    ]
    top_loss_products = (
        product_df[[c for c in top_loss_cols if c in product_df.columns]]
        .sort_values("pnl_p5", ascending=True)
        .head(5)
        .round(4)
    )
    top_breach_products = (
        product_df[[c for c in top_loss_cols if c in product_df.columns]]
        .sort_values("barrier_breach_freq", ascending=False)
        .head(5)
        .round(4)
        if "barrier_breach_freq" in product_df.columns else pd.DataFrame()
    )
    product_samples = []
    if "pnl_samples" in product_df.columns:
        for _, row in product_df.iterrows():
            samples = np.asarray(row["pnl_samples"], dtype=float)
            p5 = float(np.percentile(samples, 5))
            product_samples.append({
                "product_id": row.get("product_id"),
                "currency": row.get("currency"),
                "pnl_mean": round(float(samples.mean()), 2),
                "pnl_median": round(float(np.median(samples)), 2),
                "pnl_p5": round(p5, 2),
                "pnl_p95": round(float(np.percentile(samples, 95)), 2),
                "expected_shortfall_5": round(
                    float(samples[samples <= p5].mean())
                    if len(samples) >= 20 else float(samples.min()),
                    2,
                ),
            })

    engine_inputs = _engine_inputs_records(
        portfolio, beta_map, vol_map, risk_free_rates,
        scenario["market_shock"],
    )
    asset_paths = _asset_path_records(res["asset_paths"], portfolio)
    delivered = (
        res["delivered_stocks"].copy()
        if res["delivered_stocks"] is not None else pd.DataFrame()
    )
    if not delivered.empty and "return_pct" in delivered.columns:
        delivered["return_pct_percent"] = delivered["return_pct"] * 100
    physical_products = pd.DataFrame()
    if "settlement_type" in product_df.columns:
        physical_products = product_df[
            product_df["settlement_type"].astype(str).str.contains(
                "physical", case=False, na=False,
            )
        ].drop(columns=["pnl_samples"], errors="ignore").round(4)

    payload = {
        "scenario_controls": {
            "preset": selected_preset,
            "initial_market_state": initial_state,
            "recovery": recovery,
            **scenario,
        },
        "simulation": {
            "n_paths": int(res.get("n_paths", 0)),
            "method": "single-factor path Monte Carlo with common random numbers",
        },
        "units": {
            "market_shock": "percent per shock event",
            "drifts": "annual rates",
            "pnl": "product currency",
            "returns": "percent",
            "asset_paths": "normalised price, initial level = 100",
            "correlation": "historical realised correlation used by the path engine",
        },
        "portfolio_stress_summary": _safe_records(res["pf_scenario_per_ccy"].round(4)),
        "whole_portfolio_reference_summary": _safe_records(res["pf_scenario_ref"].round(4))
        if "pf_scenario_ref" in res and not res["pf_scenario_ref"].empty else [],
        "product_stress_results": _safe_records(product_summary),
        "top_loss_products_by_5pct_outcome": _safe_records(top_loss_products),
        "top_barrier_breach_products": _safe_records(top_breach_products),
        "product_pnl_sample_stats": product_samples,
        "portfolio_pnl_distribution": _pnl_distribution_records(res["pnl_samples_by_ccy"]),
        "delivered_stocks": _safe_records(delivered.round(4)),
        "physical_delivery_products": _safe_records(physical_products),
        "cash_positions": _safe_records(res["cash_positions"].round(4)),
        "largest_beta_scaled_shocks": sorted(
            engine_inputs,
            key=lambda r: abs(r["per_event_shock_beta_scaled_pct"]),
            reverse=True,
        )[:5],
        "engine_inputs_by_underlying": engine_inputs,
        "worst_terminal_underlying_paths": sorted(
            asset_paths,
            key=lambda r: r["terminal_median"],
        )[:5],
        "asset_path_summaries": asset_paths,
        "largest_correlations": _correlation_records(corr_df),
    }
    return _stable_json(payload)


@st.cache_data(
    ttl=_CACHE_TTL_SECONDS,
    max_entries=_CACHE_MAX_ENTRIES,
    show_spinner=False,
)
def generate_stress_testing_insight(payload_json: str) -> str:
    prompt = f"""
You are a senior structured-products advisor speaking to a private client after
running a stress-test simulation.

Write like a person in a client review: calm, specific, and easy to understand.
The client wants to know what this simulation actually showed, not a textbook
explanation. Use the named products, underlyings, currencies, delivered stocks,
breach frequencies, P&L figures, weak-case losses, asset path summaries,
beta-scaled shocks, and correlations supplied below.

Write one polished paragraph of exactly 6 sentences:
1. Open with how many paths were simulated and the whole-portfolio
   reference-currency profit or loss and return from
   whole_portfolio_reference_summary; use per-currency only if the
   whole-portfolio summary is missing.
2. Explain the main reason for the result by naming the product or underlying
   that mattered most.
3. Explain the weak-case result in client language, using the 5% P&L or expected
   shortfall only if it adds real meaning.
4. Say what underlying was delivered on average, what its delivered-stock
   performance was, and why it was delivered; if delivered_stocks is empty, say
   no share delivery appeared in this simulation.
5. Explain whether the stress is concentrated or linked through correlation,
   naming the concrete product, underlying, currency, or correlation pair.
6. Close with what would have improved the simulated performance, phrased as
   scenario drivers such as smaller underlying losses, fewer barrier breaches,
   stronger recovery, lower correlation, or no physical delivery; do not phrase
   this as advice.

Style rules:
- Sound like an expert advisor, not a report template.
- Use simple, natural language; avoid quant jargon.
- Always state clearly whether the result is a profit or a loss; do not use
  neutral wording like "P&L of" without saying profit or loss.
- For weak-case outcomes, also say whether it is a profit or loss.
- Avoid broad teaching sentences such as "this demonstrates how concentrated
  exposures can create downside risk"; instead name the exact product,
  underlying, currency, delivery outcome, or weak-case loss that proved it here.
- Prefer whole-portfolio reference-currency return/P&L over per-currency output
  when whole_portfolio_reference_summary is available.
- The final sentence must say "The simulated performance would have improved
  if..." and then name the concrete driver from this run.
- At least 5 of the 6 sentences must contain a concrete supplied item: product
  ID, underlying name, currency, P&L number, breach frequency, delivery amount,
  correlation pair, or weak-case result.
- Use numbers sparingly, but do use the few numbers that explain the result.
- Do not explain general structured-product theory unless it is tied to a
  specific product or underlying from the simulation.
- Do not use headings, bullets, questions, greetings, or markdown.
- Do not recommend buying, selling, holding, hedging, switching, or changing
  exposure.
- Do not say "monitor", "watch next", "keep an eye on", or similar follow-up
  language.
- Do not invent facts, market views, exact probabilities, or correlations that
  are not supplied.
- Do not say "the JSON", "the table", "the data", "Monte Carlo", or "path
  dispersion".
- If a value is missing, do not mention it.

Stress Testing JSON:
{payload_json}
""".strip()
    return _call_gemini(prompt)


def build_factor_stress_payload(
    *,
    res,
    portfolio,
    loadings,
    ui_scenario,
    preset_name,
    preset,
    fx_rates=None,
    reference_currency=None,
    premiums_by_method=None,
    premium_method="mean",
) -> str:
    product_df = res["product_df"].copy()
    product_summary = product_df.drop(columns=["pnl_samples"], errors="ignore").round(4)
    top_loss_cols = [
        "product_id", "currency", "worst_underlying", "settlement_type",
        "barrier_breach_freq", "pnl_mean", "pnl_p5", "return_p5_pct",
    ]
    top_loss_products = (
        product_df[[c for c in top_loss_cols if c in product_df.columns]]
        .sort_values("pnl_p5", ascending=True)
        .head(5)
        .round(4)
        if "pnl_p5" in product_df.columns else pd.DataFrame()
    )
    top_breach_products = (
        product_df[[c for c in top_loss_cols if c in product_df.columns]]
        .sort_values("barrier_breach_freq", ascending=False)
        .head(5)
        .round(4)
        if "barrier_breach_freq" in product_df.columns else pd.DataFrame()
    )

    cost_by_ccy = {}
    if "currency" in product_df.columns and "total_cost" in product_df.columns:
        cost_by_ccy = {
            ccy: float(sub["total_cost"].sum())
            for ccy, sub in product_df.groupby("currency")
        }
    reference_summary, reference_samples = _reference_pnl_summary(
        res.get("pnl_samples_by_ccy", {}),
        cost_by_ccy,
        fx_rates,
        reference_currency,
    )
    portfolio_distribution = _pnl_distribution_records(res["pnl_samples_by_ccy"])
    if reference_samples is not None and reference_currency is not None:
        portfolio_distribution.insert(0, {
            **_pnl_distribution_records({reference_currency: reference_samples})[0],
            "currency": f"{reference_currency} reference aggregate",
        })

    delivered = (
        res["delivered_stocks"].copy()
        if res.get("delivered_stocks") is not None else pd.DataFrame()
    )
    if not delivered.empty and "return_pct" in delivered.columns:
        delivered["return_pct_percent"] = delivered["return_pct"] * 100

    product_samples = []
    if "pnl_samples" in product_df.columns:
        for _, row in product_df.iterrows():
            samples = np.asarray(row["pnl_samples"], dtype=float)
            p5 = float(np.percentile(samples, 5))
            product_samples.append({
                "product_id": row.get("product_id"),
                "currency": row.get("currency"),
                "worst_underlying": row.get("worst_underlying"),
                "settlement_type": row.get("settlement_type"),
                "pnl_mean": round(float(samples.mean()), 2),
                "pnl_median": round(float(np.median(samples)), 2),
                "pnl_p5": round(p5, 2),
                "pnl_p95": round(float(np.percentile(samples, 95)), 2),
                "expected_shortfall_5": round(
                    float(samples[samples <= p5].mean())
                    if len(samples) >= 20 else float(samples.min()),
                    2,
                ),
            })

    factor_path_summaries = _path_summary_records(res.get("factor_paths", {}))
    asset_path_summaries = _asset_path_records(res.get("asset_paths", {}), portfolio)
    decomposition = _factor_decomposition_records(res, portfolio, loadings)

    events = ui_scenario.get("events", []) or []
    largest_factor_shocks = []
    for event in events:
        for code, shock in (event.get("factor_shock", {}) or {}).items():
            largest_factor_shocks.append({
                "day": event.get("day"),
                "factor": code,
                "shock_pct": shock,
                "recovery": event.get("recovery"),
            })
    largest_factor_shocks.sort(key=lambda r: abs(float(r["shock_pct"])), reverse=True)

    payload = {
        "scenario_controls": {
            "preset_name": preset_name,
            "preset_label": preset.get("label") if isinstance(preset, dict) else None,
            "preset_description": (
                preset.get("description") if isinstance(preset, dict) else None
            ),
            "active_premium_method": premium_method,
            **ui_scenario,
        },
        "simulation": {
            "n_paths": int(res.get("n_paths", 0)),
            "method": "multi-factor structured-products path simulation with common random numbers",
        },
        "units": {
            "factor_shocks": "percent move in factor level at event date",
            "factor_paths": "normalised factor level, initial level = 100",
            "asset_paths": "normalised underlying price, initial level = 100",
            "pnl": "product currency unless marked as reference aggregate",
            "returns": "percent",
            "loadings": "estimated sensitivity of each underlying to each factor",
            "decomposition": "median-path log-return contribution in percent",
        },
        "portfolio_stress_summary": _safe_records(res["pf_scenario_per_ccy"].round(4)),
        "whole_portfolio_reference_summary": reference_summary,
        "portfolio_pnl_distribution": portfolio_distribution,
        "product_stress_results": _safe_records(product_summary),
        "top_loss_products_by_5pct_outcome": _safe_records(top_loss_products),
        "top_barrier_breach_products": _safe_records(top_breach_products),
        "product_pnl_sample_stats": product_samples,
        "delivered_stocks": _safe_records(delivered.round(4)),
        "cash_positions": _safe_records(res["cash_positions"].round(4)),
        "largest_factor_shocks": largest_factor_shocks[:8],
        "factor_path_summaries": factor_path_summaries,
        "largest_factor_moves": sorted(
            factor_path_summaries,
            key=lambda r: abs(r["terminal_move_pct"] or 0),
            reverse=True,
        )[:6],
        "worst_terminal_underlying_paths": sorted(
            asset_path_summaries,
            key=lambda r: r["terminal_median"],
        )[:6],
        "asset_path_summaries": asset_path_summaries,
        "median_path_factor_decomposition": decomposition,
        "largest_negative_decomposition_items": sorted(
            decomposition,
            key=lambda r: r["total_median_return_log_pct"],
        )[:6],
        "loadings_by_underlying": _loadings_records(portfolio, loadings),
        "factor_premiums": _factor_premium_records(premiums_by_method),
    }
    return _stable_json(payload)


@st.cache_data(
    ttl=_CACHE_TTL_SECONDS,
    max_entries=_CACHE_MAX_ENTRIES,
    show_spinner=False,
)
def generate_factor_stress_insight(payload_json: str) -> str:
    prompt = f"""
You are a structured-products desk specialist explaining a multi-factor stress
simulation to a private client in plain language.

This is not a generic market comment. Use only the supplied Factor Stress JSON:
the scenario events, factor shocks, number of paths, portfolio P&L, product
P&L, barrier breaches, delivered stocks, factor paths, underlying paths,
loadings, and median-path factor decomposition.

Write one polished paragraph of exactly 6 sentences:
1. Start with the number of simulated paths, the scenario or preset name, and
   the whole-portfolio reference-currency profit or loss and return; use
   per-currency results only if the reference summary is missing.
2. Name the factor shock or factor path that mattered most, and connect it to
   the product or underlying that moved the result.
3. Explain the weakest product or weak-case outcome using the 5% P&L,
   expected shortfall, or barrier breach frequency when supplied.
4. Say whether shares were delivered; if yes, name the delivered underlying,
   its profit or loss, return, and why delivery appeared in this scenario.
5. Explain the factor link in client language by naming one concrete loading or
   decomposition result, such as TECH, MKT, HC, FIN, ENERGY, or FX driving a
   specific underlying.
6. Close with "The simulated performance would have improved if..." and name
   the concrete scenario driver from this run, such as a smaller factor shock,
   less negative underlying path, fewer barrier breaches, lower linked-factor
   sensitivity, or no physical delivery.

Style rules:
- Sound like a senior advisor in a client review, not a model report.
- Use simple, natural language; avoid quant jargon where a normal phrase works.
- Always state clearly whether the result is a profit or a loss.
- For weak-case outcomes, also say whether it is a profit or loss.
- At least 5 of the 6 sentences must contain a concrete supplied item: product
  ID, underlying name, currency, factor code, P&L number, return number,
  breach frequency, delivered amount, loading, or decomposition value.
- Use numbers only when they explain the result.
- Do not use headings, bullets, questions, greetings, or markdown.
- Do not recommend buying, selling, holding, hedging, switching, or changing
  exposure.
- Do not say "monitor", "watch next", "keep an eye on", or similar follow-up
  language.
- Do not invent facts, market views, exact probabilities, or relationships
  that are not supplied.
- Do not say "the JSON", "the table", "the data", "Monte Carlo", or "path
  dispersion".
- If a value is missing, do not mention it.

Factor Stress JSON:
{payload_json}
""".strip()
    return _call_gemini(prompt)
