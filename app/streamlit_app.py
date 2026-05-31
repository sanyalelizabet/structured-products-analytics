import streamlit as st
import sys
from pathlib import Path

# make src and data importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"

from src.logging_config import configure_logging
configure_logging()

from src.portfolio.portfolio_analytics import PortfolioAnalytics, scale_display_units
from src.market_data.eod_client import EODClient
from src.market_data.market_data_engine import MarketDataEngine
from src.risk.correlation_engine import CorrelationEngine
from src.risk.beta_engine import BetaEngine
from src.risk.factor_engine import FactorEngine
from src.risk.factor_loadings_engine import FactorLoadingsEngine
from data.reference_data import beta_map as beta_map_static
from data.reference_data import risk_free_rates as risk_free_rates_static
from src.market_data.yahoo_client import YahooClient
from src.pricing.monte_carlo import MonteCarloPricer
from app import portfolio_source, portfolio_storage
from app.views import (
    product, portfolio as portfolio_view, stress_testing, factor_stress,
    portfolio_entry, onboarding,
)


@st.cache_resource
def get_market_engine():
    api_key = st.secrets["EOD_API_KEY"]
    client = EODClient(api_key)
    return MarketDataEngine(client)

@st.cache_resource
def get_yahoo_client():
    return YahooClient()

# Cache TTLs — daily-frequency data lives 24h, intraday spots/rates 1h.
_TTL_INTRADAY = 60 * 60          # 1h
_TTL_DAILY    = 24 * 60 * 60     # 24h

# History window for historical estimations (factor loadings, single-factor
# betas, correlations). Matches the factor-premium estimation window so all
# estimates share the same horizon.
_ESTIMATION_YEARS = 5


# ---------------------------------------------------------------------------
# Cache-key derivation
# ---------------------------------------------------------------------------
# Every cached function below takes a ``portfolio_key`` string in addition
# to the underscored DataFrame.  Streamlit ignores underscored args when
# computing the cache key, so without an explicit key the cache would
# return the *same result for every portfolio* — which is exactly the bug
# that made switching from demo to a user portfolio appear to "stick" on
# demo numbers.  The key is a deterministic fingerprint of the portfolio
# content (product IDs + count); it changes when the portfolio changes,
# correctly invalidating the cache.
def _portfolio_cache_key(portfolio) -> str:
    if portfolio is None or len(portfolio) == 0:
        return "empty"
    pids = sorted(str(x) for x in portfolio.get("product_id", []))
    return f"{len(pids)}::{'|'.join(pids)}"


@st.cache_data(ttl=_TTL_INTRADAY)
def compute_pricing_and_greeks(
    _portfolio, _corr_df, vol_map, risk_free_rates,
    portfolio_key: str,
    _vol_surfaces=None,
    valuation_date=None,
):
    """One-shot Monte-Carlo: greeks, portfolio delta, and fair values.

    The base fair value used for finite-difference Greeks is identical to
    the standalone fair-value Monte Carlo (same seed, same paths), so we
    return all three from a single pass instead of pricing the portfolio
    twice.

    The ``_vol_surfaces`` argument, when supplied, enables the
    surface-aware barrier-strike volatility input introduced in Stage 3
    Substage A: each product is priced with a per-underlying volatility
    drawn from the calibrated surface at the strike of its downside
    barrier and at its residual maturity, rather than at the
    at-the-money volatility used by the legacy pricer. The argument is
    prefixed with an underscore so that Streamlit excludes it from the
    cache key; the cache invalidation continues to be driven by
    ``portfolio_key``, which captures the active-portfolio identity.
    """
    _ = portfolio_key   # consumed only as a cache key
    # 5,000 paths is plenty for finite-difference Greeks under common
    # random numbers — the bias from halving paths is well below the 1 %
    # bump precision.  Cuts runtime roughly in half.
    pricer = MonteCarloPricer(n_paths=5_000, seed=42)
    return pricer.compute_portfolio_greeks(
        _portfolio, vol_map, risk_free_rates, corr_df=_corr_df,
        vol_surfaces=_vol_surfaces, valuation_date=valuation_date,
    )


@st.cache_data(ttl=_TTL_INTRADAY)
def fetch_market_data(_portfolio, portfolio_key: str):
    _ = portfolio_key
    engine = get_market_engine()
    try:
        all_isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
        engine.fetch_securities_master(all_isins)
        engine.fetch_latest_prices(_portfolio)
        updated_portfolio = engine.update_spots(_portfolio)
        db = engine.load_db()
        valuation_date = db["date"].max() if not db.empty else None
        return updated_portfolio, db, valuation_date, None
    except Exception as e:
        db = engine.load_db()
        valuation_date = db["date"].max() if not db.empty else None
        return _portfolio, db, valuation_date, str(e)

@st.cache_data(ttl=_TTL_DAILY)
def fetch_implied_vols(_portfolio, portfolio_key: str):
    _ = portfolio_key
    engine = get_market_engine()
    yahoo  = get_yahoo_client()
    isins  = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    try:
        engine.fetch_options_chain(isins, yahoo_client=yahoo)
        return engine.build_atm_vol_map(_portfolio)
    except Exception as e:
        st.warning(f"Could not build implied vol map, falling back to static vols. {e}")
        from data.reference_data import vol_map as vol_map_static
        return vol_map_static

@st.cache_data(ttl=_TTL_INTRADAY)
def fetch_fx_rates(reference_currency: str):
    """Refresh the FX store and return ``(fx_map, as_of_str)`` for the reference.

    Appends today's snapshot to ``data/fx_rates.csv`` via the market-data
    engine (skipped if already present). On failure the stored snapshot remains
    the fallback. ``as_of_str`` feeds the analytics cache key so conversions
    re-run when the FX snapshot changes.
    """
    engine = get_market_engine()
    try:
        engine.fetch_latest_fx(reference_currency)
    except Exception as e:  # defensive: never block analytics on an FX refresh
        st.warning(f"Could not refresh FX rates. Using stored snapshot. {e}")
    fx_map, as_of = engine.build_fx_rate_map(reference_currency)
    return fx_map, str(as_of)


@st.cache_data(ttl=_TTL_INTRADAY)
def fetch_risk_free_rates(_portfolio, portfolio_key: str,
                          currencies=("CHF", "USD", "EUR", "GBP"),
                          tenor: str = "3M"):
    """Refresh GBOND yields and return a {currency: rate} map.

    The rates DB on disk is the authoritative source: any successful API
    pull appends a fresh row, and ``build_risk_free_rate_map`` always
    reads the most recent stored value per currency.  When the API call
    fails we silently fall back to the previous CSV row.  As a final
    safety net (e.g. first run with no DB and no network), the static
    dict in ``data/reference_data.py`` is layered underneath.
    """
    _ = portfolio_key
    engine = get_market_engine()
    try:
        engine.fetch_latest_rates(list(currencies), tenor=tenor)
    except Exception as e:
        st.warning(f"Could not refresh risk-free rates from EOD. {e}")

    dynamic = engine.build_risk_free_rate_map(list(currencies), tenor=tenor)
    # Layer onto the static dict for any currency missing from the DB.
    rates = {**risk_free_rates_static, **dynamic}
    return rates


@st.cache_data(ttl=_TTL_DAILY)
def fetch_realised_vols(_portfolio, portfolio_key: str):
    _ = portfolio_key
    engine = get_market_engine()
    try:
        return engine.build_realised_vol_map(_portfolio, window=252)
    except Exception as e:
        st.warning(f"Could not build realised vol map, falling back to static vols. {e}")
        from data.reference_data import vol_map as vol_map_static
        return vol_map_static


@st.cache_data(ttl=_TTL_DAILY)
def fetch_vol_surfaces(_portfolio, portfolio_key: str):
    """Per-ISIN term-structure implied-volatility surfaces.

    Returns one :class:`VolSurface` per active-portfolio underlying.
    Each surface internally composes the per-(ISIN, expiry) SVI slices
    built in Stage 1 into a term-structure-consistent object via the
    Stage 2 linear-in-total-variance recipe. The surface exposes
    :meth:`sigma(K, T)` at arbitrary strike and tenor, with a status
    taxonomy (interpolated, extrapolated, single_slice, fallback) that
    the user interface can badge.

    The function is currently additive: no view or pricer consumes it
    yet. Stage 3 will replace the constant-vol input to the barrier
    product pricers with surface evaluations at the barrier strike,
    which is the substance of the pricer bug-fix.

    The cache key follows the same ``portfolio_key`` convention as the
    sibling ``fetch_implied_vols`` and ``fetch_realised_vols``.

    Returns
    -------
    dict
        ``{ isin: VolSurface }``. Empty if no options data is available.
        ISINs without any chain coverage are absent rather than mapped
        to a fallback surface.
    """
    _ = portfolio_key
    engine = get_market_engine()
    try:
        return engine.build_vol_surfaces(_portfolio)
    except Exception as e:
        st.warning(f"Could not build vol surfaces; downstream surface-aware "
                   f"analytics will fall back to constant vol. {e}")
        return {}


@st.cache_data(ttl=_TTL_INTRADAY)
def build_product_analytics(_portfolio, _db, reference_currency: str,
                            portfolio_key: str, fx_as_of: str = "", _fx_map=None):
    # ``reference_currency`` + ``portfolio_key`` + ``fx_as_of`` form the cache
    # key so the analytics pipeline re-runs when the roll-up currency, the
    # portfolio composition, OR the FX snapshot changes. ``_fx_map`` is the
    # engine-built rate map (leading underscore → not hashed; fx_as_of keys it).
    _ = portfolio_key
    pa = PortfolioAnalytics(_portfolio, reference_currency=reference_currency,
                            price_db=_db, fx_rates=_fx_map, fx_as_of=fx_as_of)
    df = pa.build_product_analytics()
    # Percent-ify the fractional rate columns once, here at the source, so every
    # downstream view renders them as percentages without re-scaling locally.
    scale_display_units(df)
    return pa, df

@st.cache_data(ttl=_TTL_DAILY)
def build_corr_matrix(_portfolio, portfolio_key: str):
    _ = portfolio_key
    engine = get_market_engine()
    isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    isin_ticker_map = engine.build_isin_ticker_map(isins)
    return CorrelationEngine(engine).build_corr_matrix(isin_ticker_map, years=_ESTIMATION_YEARS)


@st.cache_data(ttl=_TTL_DAILY)
def build_display_corr_matrix(_portfolio, portfolio_key: str, target_ccy: str = "USD"):
    """Investor-view correlation: prices translated to ``target_ccy`` first.

    Used by the Portfolio view's correlation panel only — the stress and factor
    engines keep using the native-return correlation produced by
    :func:`build_corr_matrix`.  Cross-currency baskets show the joint risk a
    single-currency investor actually experiences, with FX co-movement baked in.
    """
    _ = portfolio_key
    engine = get_market_engine()
    isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    isin_ticker_map   = engine.build_isin_ticker_map(isins)
    isin_currency_map = engine.build_isin_currency_map(isins)
    return CorrelationEngine(engine).build_translated_corr_matrix(
        isin_ticker_map, isin_currency_map,
        target_ccy=target_ccy, years=_ESTIMATION_YEARS,
    )

@st.cache_data(ttl=_TTL_DAILY)
def build_beta_map(_portfolio, portfolio_key: str):
    _ = portfolio_key
    engine = get_market_engine()
    isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    isin_ticker_map = engine.build_isin_ticker_map(isins)
    try:
        return BetaEngine(engine).build_beta_map(isin_ticker_map, years=_ESTIMATION_YEARS)
    except Exception as e:
        st.warning(f"Could not compute dynamic betas, falling back to static. {e}")
        return beta_map_static

@st.cache_resource
def get_factor_engine():
    return FactorEngine(get_market_engine())

@st.cache_data(ttl=_TTL_DAILY)
def build_factor_loadings(_portfolio, portfolio_key: str):
    """Multivariate OLS loadings against the full factor universe.

    Returns ``{isin: {betas, alpha, idio_vol, r_squared, n_obs}}``.
    Falls back to defaults per ISIN if data is unavailable.
    """
    _ = portfolio_key
    mde = get_market_engine()
    fe  = get_factor_engine()
    fle = FactorLoadingsEngine(mde, fe)

    isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    isin_ticker_map = mde.build_isin_ticker_map(isins)
    try:
        return fle.build_loadings(isin_ticker_map, years=_ESTIMATION_YEARS)
    except Exception as e:
        st.warning(f"Factor loadings unavailable, using safe defaults. {e}")
        from src.risk.factor_engine import FACTORS
        return {
            isin: {
                "betas":     {f: (1.0 if f == "MKT" else 0.0) for f in FACTORS},
                "alpha":     0.0,
                "idio_vol":  0.15,
                "r_squared": 0.0,
                "n_obs":     0,
            }
            for isin in isins
        }



# =========================
# Page setup
# =========================
st.set_page_config(page_title="Structured Products Dashboard", layout="wide")

# ───────────────────────────────────────────────────────────────────────
# Onboarding gate — runs before everything else.
# ───────────────────────────────────────────────────────────────────────
# A fresh session has no portfolio mode set.  The splash asks the user
# whether to Load existing (demo / JSON upload) or Create new, then
# stores the choice in session state.  Until that's resolved we render
# only the splash — no sidebar, no analytics pipeline, no main app.
if onboarding.is_active():
    onboarding.render()
    st.stop()

# ───────────────────────────────────────────────────────────────────────
# Sidebar — two visually-distinct sections, both rendered as button groups
# for a consistent professional look.
#
#   1. PORTFOLIO  — what data is loaded and how to manage it (badge,
#                   Select Portfolio, Add / Edit products in user mode).
#   2. ANALYTICS  — which analytical view to render.  Buttons (not a
#                   radio) so the styling matches the Portfolio section.
# ───────────────────────────────────────────────────────────────────────
mode = portfolio_source.get_mode()
active_portfolio = portfolio_source.get_active_portfolio()
n_products = len(active_portfolio)
portfolio_name = portfolio_source.get_name()
portfolio_currency = portfolio_source.get_reference_currency()

# --- Main title — always shows which portfolio is being analysed --------
st.title(f"Structured Products Analytics")
st.markdown(
    f"<div style='color:#aaa; margin-top:-0.4rem; margin-bottom:0.5rem;'>"
    f"Portfolio: <b>{portfolio_name}</b>  ·  "
    f"Reference currency: <b>{portfolio_currency}</b>"
    f"</div>",
    unsafe_allow_html=True,
)

# --- Per-view orientation — a brief account of what the active view -----
# presents, so the reader knows what is being shown before engaging with
# the figures.  Phrased as standing descriptions of each analytical lens.
VIEW_INTROS = {
    "Product": (
        "Examine a single product in detail — its payoff, its current "
        "valuation, and the status of its barrier and coupon levels."
    ),
    "Stress Testing": (
        "Assess how the portfolio's value would respond to adverse market "
        "moves."
    ),
    "Factor Stress": (
        "Identify the portfolio's exposure to the main market drivers."
    ),
}
_active_view = st.session_state.get("active_view", "Portfolio")
_view_intro = VIEW_INTROS.get(_active_view)
if _view_intro:
    st.markdown(
        f"<div style='color:#888; margin-bottom:1.2rem; max-width:60rem; "
        f"line-height:1.5;'>{_view_intro}</div>",
        unsafe_allow_html=True,
    )

st.sidebar.image(str(logo_path), width=160)

# --- Section 1: Portfolio management ----------------------------------
st.sidebar.markdown("### Portfolio")
n_word = "product" if n_products == 1 else "products"
if mode == "demo":
    st.sidebar.info(
        f"**{portfolio_name}**  \n"
        f"_{n_products} sample {n_word}_ • read-only  \n"
        f"_{portfolio_currency} ref._"
    )
else:
    st.sidebar.info(
        f"**{portfolio_name}**  \n"
        f"_{n_products} {n_word}_ • _{portfolio_currency} ref._"
    )
if st.sidebar.button("Select Portfolio", width="stretch",
                     key="sidebar_switch"):
    portfolio_source.clear_mode()
    st.rerun()
# Manage products (add / edit / delete) — user mode only, demo is read-only.
if mode == "user":
    add_active = (st.session_state.get("active_view") == "Add Product")
    if st.sidebar.button(
        "Add / Edit products",
        type="primary" if add_active else "secondary",
        width="stretch", key="sidebar_add",
    ):
        st.session_state["active_view"] = "Add Product"
        st.rerun()

st.sidebar.markdown("---")

# --- Section 2: Analytics views — buttons (matches Portfolio section) ----
st.sidebar.markdown("### Analytics")
analytics_views = ["Portfolio", "Product", "Stress Testing", "Factor Stress"]

if "active_view" not in st.session_state:
    st.session_state["active_view"] = analytics_views[0]

for v in analytics_views:
    is_active = (st.session_state["active_view"] == v)
    if st.sidebar.button(
        v,
        type="primary" if is_active else "secondary",
        width="stretch",
        key=f"sidebar_view_{v}",
    ):
        st.session_state["active_view"] = v
        st.rerun()

view = st.session_state["active_view"]

# ───────────────────────────────────────────────────────────────────────
# Route standalone views before the heavy data fetches.
# ───────────────────────────────────────────────────────────────────────
# Views that don't need the analytics pipeline (market data, vols, greeks)
# are routed here so the user doesn't wait for fetches they won't use.
if view == "Add Product":
    portfolio_entry.render()
    st.stop()

# ───────────────────────────────────────────────────────────────────────
# Empty-state guard for user mode with no products yet.
# ───────────────────────────────────────────────────────────────────────
# Analytics views all assume at least one product is present.  Render a
# clean empty state and point the user at the Add Product item in the
# sidebar rather than letting the pipeline crash on an empty DataFrame.
if mode == "user" and n_products == 0:
    st.info(
        "Your portfolio is empty.\n\n"
        "Use **Add Product** in the sidebar to enter your first product, "
        "or click **Switch portfolio** to load the demo."
    )
    st.stop()

# ───────────────────────────────────────────────────────────────────────
# Bad-row guard for user mode.
# ───────────────────────────────────────────────────────────────────────
# Analytics-pipeline failures cascade into a full-page stack trace that
# can trap the user on an unrecoverable view.  Pre-flight every row
# against the same hard-error contract the form uses, so a bad row
# surfaces here as a friendly banner with a recovery path (open Add
# Product to fix, or Switch portfolio to escape entirely).  The sidebar
# remains visible because we never reach the analytics pipeline.
if mode == "user":
    from src.portfolio.portfolio_entry import validate_row_errors as _row_errs
    bad_rows: list[tuple[int, str, list[str]]] = []
    for i, row in active_portfolio.reset_index(drop=True).iterrows():
        errs = _row_errs(row.to_dict())
        if errs:
            bad_rows.append((int(i), str(row.get("product_id", "?")), errs))
    if bad_rows:
        st.error(
            f"**{len(bad_rows)} product(s) in your portfolio can't be "
            "loaded for analytics**.  Open **Add Product** in the "
            "sidebar to edit or delete them, or click **Switch portfolio** "
            "to leave this portfolio."
        )
        with st.expander("What's wrong with each product?", expanded=True):
            for i, pid, errs in bad_rows:
                st.markdown(f"**Row {i + 1} · {pid}**")
                for e in errs:
                    st.markdown(f"- {e}")
        st.stop()

# =========================
# Shared data
# =========================
# Fingerprint of the currently-active portfolio.  Threaded into every
# cached function below so cache invalidation reflects portfolio changes
# (mode switches, JSON loads, product add/edit/delete).
pkey = _portfolio_cache_key(active_portfolio)

portfolio, db, valuation_date, fetch_error = fetch_market_data(
    active_portfolio, portfolio_key=pkey,
)
if fetch_error:
    st.warning(f"Could not refresh market prices. Using portfolio default spots. {fetch_error}")

# Refresh the FX store and build the rate map BEFORE analytics use it. The
# snapshot date feeds the analytics cache key so conversions re-run when the FX
# snapshot changes.
fx_map, fx_as_of = fetch_fx_rates(portfolio_currency)

# Re-fingerprint after fetch_market_data: ``update_spots`` may have set
# current_spots, but it does not change product IDs, so the key is the
# same — passing ``pkey`` here keeps caches consistent.
analytics, df        = build_product_analytics(
    portfolio, db, portfolio_currency, portfolio_key=pkey,
    fx_as_of=fx_as_of, _fx_map=fx_map,
)
corr_df              = build_corr_matrix(portfolio, portfolio_key=pkey)
beta_map             = build_beta_map(portfolio, portfolio_key=pkey)

vol_map_implied      = fetch_implied_vols(portfolio, portfolio_key=pkey)
vol_map_realised     = fetch_realised_vols(portfolio, portfolio_key=pkey)
# Stage 1: per-(isin, expiry) SVI slice surfaces. Populated but not yet
# consumed by the pricer or any view; the term-structure assembly (Stage 2)
# and surface-aware pricing (Stage 3) build on this object.
vol_surfaces         = fetch_vol_surfaces(portfolio, portfolio_key=pkey)

risk_free_rates      = fetch_risk_free_rates(portfolio, portfolio_key=pkey)

# ───────────────────────────────────────────────────────────────────────
# Market-data coverage + retry.
# ───────────────────────────────────────────────────────────────────────
# When a user-entered product references an ISIN our EOD API hasn't
# indexed, the master fetch returns an empty list silently (it logs
# "No listings found for <ISIN>" at info level, but nothing surfaces
# to the UI).  The downstream pipeline then has no ticker for that
# ISIN, no historical prices, no row in the correlation matrix, no
# beta — analytics for that product can only proceed with safe
# defaults.
#
# Make all of that explicit here:
#   1. Detect ISINs missing from securities_master_data.csv;
#   2. Show a clear, actionable warning with the affected ISIN(s);
#   3. Offer a "Retry" that force-refreshes the master fetch and
#      clears the analytics cache so the pipeline re-runs end-to-end.
_portfolio_isins = sorted({
    isin for _, row in portfolio.iterrows()
    for isin in (row.get("underlying_isins") or [])
})
try:
    _isin_ticker_map = get_market_engine().build_isin_ticker_map(_portfolio_isins)
except (ValueError, FileNotFoundError):
    _isin_ticker_map = {}
_uncovered = [isin for isin in _portfolio_isins if isin not in _isin_ticker_map]

if _uncovered and mode == "user":
    with st.container(border=True):
        st.warning(
            f"**Market data unavailable for {len(_uncovered)} ISIN(s)** — "
            f"{', '.join(_uncovered[:5])}"
            f"{'  …' if len(_uncovered) > 5 else ''}.\n\n"
            "Our market-data provider (EOD) returned no listings for "
            "these on the last fetch, so we have no historical prices, "
            "beta, or correlation data for them.\n\n"
            "**Likely causes**\n"
            "- The ISIN isn't in EOD's catalogue (some exotic / new / "
            "private issues aren't indexed).\n"
            "- A transient API error or rate-limit on the last fetch.\n"
            "- Your API key doesn't cover this exchange / region.\n\n"
            "**Until this is resolved**, analytics for affected products "
            "fall back to safe defaults: identity correlation, β = 1.0 on "
            "MKT, static implied vols.  Other products are unaffected."
        )
        if st.button("Retry market-data fetch",
                     key="retry_master_fetch",
                     help=(
                         "Force a fresh EOD lookup for the missing ISIN(s). "
                         "Use after fixing API access or if you suspect a "
                         "transient failure."
                     )):
            mde = get_market_engine()
            failed: list[str] = []
            with st.spinner(
                f"Re-fetching securities master for {len(_uncovered)} ISIN(s)…"
            ):
                try:
                    mde.fetch_securities_master(_uncovered, force_refresh=True)
                except Exception as exc:   # noqa: BLE001
                    failed.append(str(exc))
            # Wipe the analytics cache so the pipeline re-runs with the
            # freshly-fetched master data.
            st.cache_data.clear()
            if failed:
                st.error(
                    "Retry hit an error from the API:\n\n"
                    + "\n".join(f"- {e}" for e in failed)
                )
            else:
                st.toast("Re-fetch complete — refreshing analytics…",
                         icon="🔄")
            st.rerun()

vol_map              = vol_map_implied
greeks_df, pf_delta, fv_df = compute_pricing_and_greeks(
    portfolio, corr_df, vol_map, risk_free_rates,
    portfolio_key=pkey,
    _vol_surfaces=vol_surfaces,
    valuation_date=valuation_date,
)

if mode == "user" and portfolio_name != portfolio_source.UNSAVED_PORTFOLIO_NAME:
    try:
        portfolio_storage.save_valuation_snapshot(
            portfolio_name,
            fv_df,
            valuation_date=valuation_date,
            reference_currency=portfolio_currency,
        )
    except portfolio_storage.NotFoundError:
        pass
    except OSError as exc:
        st.warning(f"Could not save fair-value snapshot. {exc}")

# Merge fair value columns into product analytics df
df = df.merge(fv_df[["product_id", "fair_value", "fair_value_pct"]], on="product_id", how="left")

# =========================
# Route to view
# =========================
if view == "Product":
    product.render(portfolio, df, analytics, valuation_date, vol_map, beta_map)

elif view == "Portfolio":
    # The portfolio view's correlation panel is the investor view: prices
    # translated to a single currency (USD) before correlating, so the matrix
    # reflects the joint risk a single-currency investor actually experiences.
    # The stress and factor engines (below) keep using native ``corr_df``.
    display_corr_df = build_display_corr_matrix(
        portfolio, portfolio_key=pkey, target_ccy="USD",
    )
    portfolio_view.render(
        analytics, df, greeks_df, pf_delta, valuation_date,
        corr_df=display_corr_df,
        vol_surfaces=vol_surfaces,
        vol_map_realised=vol_map_realised,
        portfolio=portfolio,
    )

elif view == "Stress Testing":
    stress_testing.render(
        portfolio, corr_df, beta_map, vol_map_implied, vol_map_realised,
        risk_free_rates,
        fx_rates=analytics.fx_rates,
        reference_currency=analytics.reference_currency,
    )

elif view == "Factor Stress":
    loadings      = build_factor_loadings(portfolio, portfolio_key=pkey)
    factor_engine = get_factor_engine()
    factor_stress.render(
        portfolio, loadings, factor_engine, risk_free_rates,
        fx_rates=analytics.fx_rates,
        reference_currency=analytics.reference_currency,
    )
