# Structured Products Analytics

A Python-based analytics framework for evaluating, monitoring, and stress-testing structured product portfolio (e.g. BRCs and MBRCs.

The project is designed from a **buy-side portfolio analytics perspective**: it focuses on portfolio monitoring, payoff transparency, scenario analysis, and risk reporting rather than sell-side issuance pricing.

---

## Dashboard

The project includes an interactive dashboard designed to explore structured products at both product and portfolio level.

---

## Live App

[Open the dashboard](https://struxq.streamlit.app/)

![Dashboard](figures/dashboard.png)

---

## Project Overview

Structured products can be difficult to monitor because their payoff depends on several interacting components:

- underlying price performance
- barrier distance and breach status
- coupon accrual
- worst-of logic for multi-asset products
- remaining time to maturity
- currency exposure
- scenario-dependent physical delivery risk

This project provides a modular analytics framework to make these risks more transparent at both **product level** and **portfolio level**.

---

## Key Features

### Product-Level Analytics

- Payoff decomposition into coupon and redemption components
- Barrier monitoring and distance-to-barrier calculation
- Worst-of logic for multi-asset reverse convertibles
- Current performance versus strike
- Projected payoff and return if held to maturity
- Product-level P&L and annualized return metrics

### Portfolio Analytics

- Aggregated portfolio P&L and return metrics
- Portfolio-level exposure by product, currency, and underlying
- Look-through analysis of underlying exposure
- Worst-of concentration analysis
- Maturity profile and cash-flow overview
- Multi-currency normalization into a reference currency

### Market Valuation (Mark-to-Market)
- Fair value integration via external pricing engine
- Comparison: projected vs mark-to-market valuation


### Multi-Factor Stress Engine

- Factor-based scenario simulation using systematic market drivers
- Factor loadings estimated through multivariate regression
- Correlated factor shocks
- Idiosyncratic asset-level noise
- Scenario comparison using Common Random Numbers
- Mean-reverting log-price dynamics for factor and asset paths


### Risk Sensitivities

- Delta
- Vega
- Theta
- Rho
- Correlation sensitivity for multi-asset products
- Portfolio aggregation of sensitivities

### Market Data Integration

- Historical and latest price fetching through EOD Historical Data
- Local market data storage and reuse
- Securities master data support
- FX conversion for multi-currency portfolio reporting
- Historical correlation estimation from stored price data
- Implied volatility map support from option-chain data

## Architecture

The framework is modular and separates product logic, portfolio analytics, market data, scenario simulation, and visualization.

### Core Components

- `ReverseConvertible`  
  Product-level payoff, barrier, redemption, and return logic for BRC and MBRC structures.

- `PortfolioAnalytics`  
  Portfolio aggregation, look-through exposure, currency conversion, maturity profile, and portfolio metrics.

- `ScenarioEngine`  
  Single-factor path-based scenario engine using correlated GBM-style simulations, beta-adjusted shocks, and product-level payoff aggregation.

- `FactorScenarioEngine`  
  Multi-factor stress engine that simulates correlated factor paths and projects them to asset returns through factor loadings.

- `MarketDataEngine`  
  Market data fetching, storage, refresh logic, securities master data, and historical price handling.

- `CorrelationEngine`  
  Realized correlation matrix estimation from historical price data.

- `FactorLoadingsEngine`  
  Multivariate regression engine for estimating asset exposure to systematic factors.

- `MonteCarloPricer`  
  Fair-value and Greeks calculation using Monte Carlo simulation.

- `Streamlit App`  
  Interactive dashboard for product analytics, portfolio views, stress testing, and visual reporting.

---

## Example Analytics

### Portfolio Overview

- Total cost
- Projected payoff
- Projected P&L
- Portfolio return
- Product weights
- Currency-level aggregation
- Reference-currency portfolio value

### Product View

- Product terms
- Underlying performance
- Worst underlying
- Barrier distance
- Days to expiry
- Projected payoff
- Return and annualized return

### Risk View

- Underlying look-through exposure
- Barrier proximity monitoring
- Worst-of exposure concentration
- Product-level and portfolio-level Greeks
- Correlation sensitivity for multi-asset products

### Scenario Analysis

- Simulated underlying price paths
- Mean path with dispersion bands
- Terminal payoff distribution
- Scenario P&L distribution
- Delivered-stock reporting under physical settlement

---

## Use Cases

- Monitoring structured product portfolios
- Understanding payoff behavior of barrier reverse convertibles
- Analyzing worst-of multi-asset structures
- Stress testing product and portfolio outcomes
- Estimating downside exposure and physical delivery risk
- Comparing projected hold-to-maturity value with fair value
- Building educational tools for structured products and derivatives
- Prototyping analytics logic for structured product portfolio management

---


## Installation

```bash
git clone https://github.com/sanyalelizabet/structured-products-analytics.git
cd structured-products-analytics
pip install -r requirements.txt