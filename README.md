# Structured Products Analytics

This repository contains a growing set of tools and models for analyzing structured products, with a focus on reverse convertibles and multi-asset structures.

The project is designed to build a modular analytics framework combining product-level valuation, portfolio insights, and scenario-based risk analysis.

---

## Dashboard

![Dashboard](figures/dashboard.png)


---

## Current Functionality

- Representation of structured products (BRC, MBRC) in a consistent data format  
- Product-level analytics:
  - payoff and redemption logic  
  - P&L and return metrics  
  - time to maturity  
  - worst-of underlying identification  
- Basic portfolio aggregation  
- Interactive dashboard for exploration  

---

## Approach

Products are modeled through a dedicated class structure, enabling:

- separation of product logic from presentation  
- scalability from single-product to portfolio-level analysis  
- extension towards scenario and risk modeling  


---

## 📂 Project Structure

│
├── app/
├── src/
├── notebooks/
├── figures/
│   └── dashboard.png
├── README.md
