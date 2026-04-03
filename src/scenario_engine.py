class ScenarioEngine:

    def __init__(self, portfolio, beta_map, scenarios):
        self.portfolio = portfolio
        self.beta_map = beta_map
        self.scenarios = scenarios
        
    def get_beta(self, isin):
        return self.beta_map.get(isin, 1.0)
    
    def build_shocks(self, row, market_shock):
        shocks = []

        for isin in row["underlying_isins"]:
            beta = self.get_beta(isin)
            shock = beta * market_shock
            shocks.append(shock)
        return shocks
    
    def run_product(self, row, market_shock):
        shocks = self.build_shocks(row, market_shock)

        rc = ReverseConvertible(row, final_levels=shocks)
        s = rc.summary()

        return {
        "product_id": row["product_id"],
        "product_type": row["product_type"],
        "position_units": row["position_units"],
        "notional": row["notional"],
        "total_cost": s["total_cost"],
        "market_shock": market_shock,
        "underlyings": row["underlyings"],
        "shocks": shocks,
        "payoff_per_unit": s["payoff_per_unit"],
        "total_payoff": s["total_payoff"],
        "pnl": s["pnl"],
        "return_pct": s["return_pct"],
        "barrier_breached": s["barrier_breached"],
        "worst_underlying": s["worst_underlying"]
    }
    
    def run(self, market_shock):
        results = []

        for _, row in self.portfolio.iterrows():
            res = self.run_product(row, market_shock)
            results.append(res)

        product_df = pd.DataFrame(results)

        total_cost = product_df["total_cost"].sum()
        total_payoff = product_df["total_payoff"].sum()
        total_pnl = product_df["pnl"].sum()

        portfolio_return = total_pnl / total_cost if total_cost != 0 else np.nan

        portfolio_summary = pd.DataFrame([{
            "market_shock": market_shock,
            "n_products": len(product_df),
            "total_cost": total_cost,
            "total_payoff": total_payoff,
            "total_pnl": total_pnl,
            "portfolio_return_pct": portfolio_return
        }])

        return {
            "product_df": product_df,
            "portfolio_summary": portfolio_summary
        }
    
def run_all(self):
    product_results = []
    portfolio_results = []

    for shock in self.scenarios:
        res = self.run(shock)

        df = res["product_df"]
        summary = res["portfolio_summary"]

        df["market_shock"] = shock
        product_results.append(df)

        portfolio_results.append(summary)

    return {
        "product_scenarios": pd.concat(product_results, ignore_index=True),
        "portfolio_scenarios": pd.concat(portfolio_results, ignore_index=True)
    }
