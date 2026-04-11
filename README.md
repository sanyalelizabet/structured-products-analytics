# Structured Products Analytics

A professional analytics framework for evaluating, pricing, and risk-managing structured products, with a focus on barrier reverse convertibles (BRC) and multi-asset worst-of structures.

Built to replicate real-world workflows used on trading, structuring, and portfolio management desks.

---

## Dashboard

The project includes an interactive dashboard designed to explore structured products at both product and portfolio level.

---

## Live App

[Open the dashboard](https://struxq.streamlit.app/)

![Dashboard](figures/dashboard.png)

---
## Key Features

### Product-Level Analytics
- Payoff decomposition (coupon + redemption)
- Barrier monitoring (distance to barrier, breach detection)
- Worst-of logic for multi-asset structures

### Portfolio Analytics
- Aggregated PnL and return metrics
- Look-through exposure by underlying
- Maturity ladder and cash flow profile

### Market Valuation (Mark-to-Market)
- Fair value integration via external pricing engine
- Comparison: projected vs mark-to-market valuation
- Multi-currency portfolio normalization

### Scenario Engine
- Correlated Monte Carlo path simulation
- Beta-adjusted shocks and stress scenarios
- Path-consistent multi-asset dynamics

### Risk Sensitivities (Greeks)
- Delta, Vega, Theta, Rho
- Correlation sensitivity (multi-asset products)
- Portfolio aggregation of sensitivities

### Market Data Integration
- Live price fetching (EOD API)
- Historical data storage and reuse
- FX conversion across currencies

---

## Architecture

The framework is modular and separates concerns across distinct layers:

- `ReverseConvertible`  
  Product-level payoff and risk logic (single-asset and multi-asset worst-of structures)

- `PortfolioAnalytics`  
  Aggregation, reporting, and portfolio-level metrics

- `ScenarioEngine`  
  Correlated path simulation and stress testing

- `MarketDataEngine`  
  Market data retrieval, storage, and preprocessing

- `Streamlit App`  
  Interactive visualization and analytics dashboard

This design allows independent extension of pricing models, data sources, and analytics layers.

---

## Example Analytics

### Portfolio Overview
- Projected Value (hold-to-maturity)
- Mark-to-Market Value (fair value)
- PnL and return comparison

### Risk View
- Portfolio delta by underlying
- Barrier proximity monitoring
- Worst-of exposure concentration

### Scenario Analysis
- Correlated equity shocks
- Path-dependent payoff outcomes
- Stress testing under different market regimes

---

## Use Cases

- Structured products portfolio monitoring
- Risk management for barrier products
- Scenario analysis and stress testing
- Educational tool for structured payoff understanding
- Prototyping structuring ideas and payoff profiles

---

## Installation

```bash
git clone https://github.com/sanyalelizabet/structured-products-analytics.git
cd structured-products-analytics
pip install -r requirements.txt