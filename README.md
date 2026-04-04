# Structured Products Analytics

This repository contains a growing set of tools and models for analyzing structured products, with a focus on reverse convertibles and multi-asset structures.

The project is designed to build a modular analytics framework combining product-level valuation, portfolio insights, and scenario-based risk analysis.

## Dashboard

The project includes an interactive dashboard designed to explore structured products at both product and portfolio level.
---
## Live App

[Open the dashboard](https://strukis.streamlit.app/)

## Dashboard

![Dashboard](figures/dashboard.png)


---


### Current Functionality

- Standardized representation of structured products (BRC, MBRC) in a unified data model  
- **Product-level analytics**, including:
  - Payoff and redemption logic (including barrier conditions)
  - Profit & Loss and return metrics
  - Time to maturity and annualized return
  - Worst-of underlying identification for multi-asset structures  
- **Portfolio-level aggregation**:
  - Total payoff, cost, and P&L
  - Aggregated return metrics  
- Interactive interface for analysis and scenario exploration  

---

### Approach

The system is built around a modular class-based architecture:

- **Product abstraction layer**  
  Each product is modeled independently (e.g. `ReverseConvertible`), encapsulating payoff logic and analytics  

- **Separation of concerns**  
  Clear distinction between:
  - product logic (pricing, payoff, scenarios)  
  - analytics layer (aggregation, metrics)  
  - presentation layer (dashboard)  

- **Scalability**  
  The framework naturally extends:
  - from single-product analysis → portfolio-level views  
  - from static analytics → scenario and stress testing  

---

### Design Philosophy

The goal is to replicate a **sell-side structuring / trading perspective**, where:

- products are treated as **objects with defined payoff rules**  
- portfolio behavior emerges from **aggregation of individual payoffs**  
- analytics remain **transparent and traceable**, not black-box  