# =========================
# Security reference data
# =========================
# Beta and volatility are static inputs used for stress testing and monitoring.
# isin_ticker_map drives market data fetching from the EOD API.
# All keyed by ISIN.

isin_ticker_map = {
    "CH0432492467": "ALC.SW",   # ALCON
    "CH0012221716": "ABBN.SW",  # ABB
    "CH0012214059": "HOLN.SW",  # HOLCIM
    "CH0012005267": "NOVN.SW",  # NOVARTIS
    "CH0012032048": "ROG.SW",   # ROCHE
    "CH0013841017": "LONN.SW",  # LONZA
    "CH0038863350": "NESN.SW",  # NESTLE
    "US0090661010": "ABNB",     # AIRBNB
}

# Equity beta vs broad market index
beta_map = {
    "CH0432492467": 0.75,   # ALCON
    "CH0012221716": 0.95,   # ABB
    "CH0012214059": 0.70,   # HOLCIM
    "CH0012005267": 0.55,   # NOVARTIS
    "CH0012032048": 0.25,   # ROCHE
    "CH0013841017": 0.20,   # LONZA
    "CH0038863350": 0.50,   # NESTLE
    "US0090661010": 1.15,   # AIRBNB
}


# Risk-free rates (annualised, continuously compounded, as a decimal).
# Used as the risk-neutral drift in Monte Carlo pricing.
# Sources: SNB, Fed, ECB policy rates (approximate, update periodically).
risk_free_rates = {
    "CHF": 0.00,    # SNB at lower bound after 2025 cuts
    "USD": 0.0365,   # Fed funds effective
    "EUR": 0.025,   # ECB deposit rate
    "GBP": 0.045,   # BoE Bank Rate
}

#  volatility (annualised, as a decimal)
vol_map = {
    "CH0432492467": 0.24,   # ALCON
    "CH0012221716": 0.22,   # ABB
    "CH0012214059": 0.18,   # HOLCIM
    "CH0012005267": 0.16,   # NOVARTIS
    "CH0012032048": 0.14,   # ROCHE
    "CH0013841017": 0.28,   # LONZA
    "CH0038863350": 0.12,   # NESTLE
    "US0090661010": 0.35,   # AIRBNB
}
