
import streamlit as st
import sys
from pathlib import Path

# make src and data importable


from src.portfolio_analytics import PortfolioAnalytics
from src.eod_client import EODClient
from src.market_data_engine import MarketDataEngine
from src.correlation_engine import CorrelationEngine
from data.reference_data import isin_ticker_map, beta_map, vol_map, risk_free_rates
from src.pricing.monte_carlo import MonteCarloPricer
from data.portfolio import portfolio


import yfinance as yf
import pandas as pd


from src.yahoo_client import YahooClient
from src.market_data_engine import MarketDataEngine

