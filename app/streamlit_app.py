import streamlit as st
import sys
from pathlib import Path

# make src and data importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"

from src.logging_config import configure_logging
configure_logging()

from src.portfolio_analytics import PortfolioAnalytics
from src.eod_client import EODClient
from src.market_data_engine import MarketDataEngine
from src.correlation_engine import CorrelationEngine
from src.beta_engine import BetaEngine
from src.factor_engine import FactorEngine
from src.factor_loadings_engine import FactorLoadingsEngine
from data.reference_data import beta_map as beta_map_static
from data.reference_data import risk_free_rates as risk_free_rates_static
from src.yahoo_client import YahooClient
from src.pricing.monte_carlo import MonteCarloPricer
from data.portfolio import portfolio
from app.views import product, portfolio as portfolio_view, stress_testing, factor_stress


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


@st.cache_data(ttl=_TTL_INTRADAY)
def compute_pricing_and_greeks(_portfolio, _corr_df, vol_map, risk_free_rates):
    """One-shot Monte-Carlo: greeks, portfolio delta, and fair values.

    The base fair value used for finite-difference Greeks is identical to
    the standalone fair-value Monte Carlo (same seed, same paths), so we
    return all three from a single pass instead of pricing the portfolio
    twice.
    """
    # 5,000 paths is plenty for finite-difference Greeks under common
    # random numbers — the bias from halving paths is well below the 1 %
    # bump precision.  Cuts runtime roughly in half.
    pricer = MonteCarloPricer(n_paths=5_000, seed=42)
    return pricer.compute_portfolio_greeks(
        _portfolio, vol_map, risk_free_rates, corr_df=_corr_df,
    )


@st.cache_data(ttl=_TTL_INTRADAY)
def fetch_market_data(_portfolio):
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
def fetch_implied_vols(_portfolio):
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
def fetch_risk_free_rates(_portfolio, currencies=("CHF", "USD", "EUR", "GBP"),
                          tenor: str = "3M"):
    """Refresh GBOND yields and return a {currency: rate} map.

    The rates DB on disk is the authoritative source: any successful API
    pull appends a fresh row, and ``build_risk_free_rate_map`` always
    reads the most recent stored value per currency.  When the API call
    fails we silently fall back to the previous CSV row.  As a final
    safety net (e.g. first run with no DB and no network), the static
    dict in ``data/reference_data.py`` is layered underneath.
    """
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
def fetch_realised_vols(_portfolio):
    engine = get_market_engine()
    try:
        return engine.build_realised_vol_map(_portfolio, window=252)
    except Exception as e:
        st.warning(f"Could not build realised vol map, falling back to static vols. {e}")
        from data.reference_data import vol_map as vol_map_static
        return vol_map_static

@st.cache_data(ttl=_TTL_INTRADAY)
def build_product_analytics(_portfolio, _db):
    pa = PortfolioAnalytics(_portfolio, reference_currency="CHF", price_db=_db)
    df = pa.build_product_analytics()
    df["return_pa"]  *= 100
    df["ytm"]        *= 100
    df["ytm_today"]  *= 100
    df["distance_to_barrier"] *= 100
    return pa, df

@st.cache_data(ttl=_TTL_DAILY)
def build_corr_matrix(_portfolio):
    engine = get_market_engine()
    isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    isin_ticker_map = engine.build_isin_ticker_map(isins)
    return CorrelationEngine(engine).build_corr_matrix(isin_ticker_map, years=3)

@st.cache_data(ttl=_TTL_DAILY)
def build_beta_map(_portfolio):
    engine = get_market_engine()
    isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    isin_ticker_map = engine.build_isin_ticker_map(isins)
    try:
        return BetaEngine(engine).build_beta_map(isin_ticker_map, years=3)
    except Exception as e:
        st.warning(f"Could not compute dynamic betas, falling back to static. {e}")
        return beta_map_static

@st.cache_resource
def get_factor_engine():
    return FactorEngine(get_market_engine())

@st.cache_data(ttl=_TTL_DAILY)
def build_factor_loadings(_portfolio):
    """Multivariate OLS loadings against the full factor universe.

    Returns ``{isin: {betas, alpha, idio_vol, r_squared, n_obs}}``.
    Falls back to defaults per ISIN if data is unavailable.
    """
    mde = get_market_engine()
    fe  = get_factor_engine()
    fle = FactorLoadingsEngine(mde, fe)

    isins = list({isin for _, row in _portfolio.iterrows() for isin in row["underlying_isins"]})
    isin_ticker_map = mde.build_isin_ticker_map(isins)
    try:
        return fle.build_loadings(isin_ticker_map, years=3)
    except Exception as e:
        st.warning(f"Factor loadings unavailable, using safe defaults. {e}")
        from src.factor_engine import FACTORS
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
st.title("Structured Products Analytics")
st.sidebar.image(str(logo_path), width=160)
view = st.sidebar.radio(
    "View",
    ["Product", "Portfolio", "Stress Testing", "Factor Stress"],
)

# =========================
# Shared data
# =========================
portfolio, db, valuation_date, fetch_error = fetch_market_data(portfolio)
if fetch_error:
    st.warning(f"Could not refresh market prices. Using portfolio default spots. {fetch_error}")

analytics, df        = build_product_analytics(portfolio, db)
corr_df              = build_corr_matrix(portfolio)
beta_map             = build_beta_map(portfolio)

vol_map_implied      = fetch_implied_vols(portfolio)
vol_map_realised     = fetch_realised_vols(portfolio)

risk_free_rates      = fetch_risk_free_rates(portfolio)

vol_map              = vol_map_implied
greeks_df, pf_delta, fv_df = compute_pricing_and_greeks(
    portfolio, corr_df, vol_map, risk_free_rates,
)

# Merge fair value columns into product analytics df
df = df.merge(fv_df[["product_id", "fair_value", "fair_value_pct"]], on="product_id", how="left")

# =========================
# Route to view
# =========================
if view == "Product":
    product.render(portfolio, df, analytics, valuation_date, vol_map, beta_map)

elif view == "Portfolio":
    portfolio_view.render(analytics, df, greeks_df, pf_delta, valuation_date)

elif view == "Stress Testing":
    stress_testing.render(
        portfolio, corr_df, beta_map, vol_map_implied, vol_map_realised,
        risk_free_rates,
        fx_rates=analytics.fx_rates,
        reference_currency=analytics.reference_currency,
    )

elif view == "Factor Stress":
    loadings      = build_factor_loadings(portfolio)
    factor_engine = get_factor_engine()
    factor_stress.render(
        portfolio, loadings, factor_engine, risk_free_rates,
        fx_rates=analytics.fx_rates,
        reference_currency=analytics.reference_currency,
    )
