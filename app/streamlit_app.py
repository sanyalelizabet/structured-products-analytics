import streamlit as st
import sys
from pathlib import Path

# make src and data importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"

from src.portfolio_analytics import PortfolioAnalytics
from src.eod_client import EODClient
from src.market_data_engine import MarketDataEngine
from src.correlation_engine import CorrelationEngine
from data.reference_data import isin_ticker_map, beta_map, vol_map, risk_free_rates
from src.pricing.monte_carlo import MonteCarloPricer
from data.portfolio import portfolio
from app.views import product, portfolio as portfolio_view, stress_testing


@st.cache_resource
def get_market_engine():
    api_key = st.secrets["EOD_API_KEY"]
    client = EODClient(api_key)
    return MarketDataEngine(client)

@st.cache_data(ttl=3600)
def fetch_market_data(_portfolio):
    engine = get_market_engine()
    try:
        engine.fetch_latest_prices(_portfolio)
        updated_portfolio = engine.update_spots(_portfolio)
        db = engine.load_db()
        valuation_date = db["date"].max() if not db.empty else None
        return updated_portfolio, db, valuation_date, None
    except Exception as e:
        db = engine.load_db()
        valuation_date = db["date"].max() if not db.empty else None
        return _portfolio, db, valuation_date, str(e)

@st.cache_data(ttl=3600)
def build_product_analytics(_portfolio, _db):
    pa = PortfolioAnalytics(_portfolio, reference_currency="CHF", price_db=_db)
    df = pa.build_product_analytics()
    df["return_pa"] *= 100
    df["distance_to_barrier"] *= 100
    return pa, df

@st.cache_data(ttl=3600)
def build_corr_matrix():
    return CorrelationEngine(get_market_engine()).build_corr_matrix(isin_ticker_map, years=4)

@st.cache_data(ttl=3600)
def compute_fair_values(_portfolio, _corr_df, vol_map, risk_free_rates):
    pricer = MonteCarloPricer(n_paths=10_000, seed=42)
    return pricer.price_portfolio(_portfolio, vol_map, risk_free_rates, corr_df=_corr_df)

@st.cache_data(ttl=3600)
def compute_greeks(_portfolio, _corr_df, vol_map, risk_free_rates):
    pricer = MonteCarloPricer(n_paths=10_000, seed=42)
    return pricer.compute_portfolio_greeks(_portfolio, vol_map, risk_free_rates, corr_df=_corr_df)


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Structured Products Dashboard", layout="wide")
st.title("Structured Products Analytics")
st.sidebar.image(str(logo_path), width=160)
view = st.sidebar.radio("View", ["Product", "Portfolio", "Stress Testing"])

# =========================
# Shared data
# =========================
portfolio, db, valuation_date, fetch_error = fetch_market_data(portfolio)
if fetch_error:
    st.warning(f"Could not refresh market prices. Using portfolio default spots. {fetch_error}")

analytics, df       = build_product_analytics(portfolio, db)
corr_df             = build_corr_matrix()
fv_df               = compute_fair_values(portfolio, corr_df, vol_map, risk_free_rates)
greeks_df, pf_delta = compute_greeks(portfolio, corr_df, vol_map, risk_free_rates)

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
    stress_testing.render(portfolio, corr_df, beta_map, vol_map)
