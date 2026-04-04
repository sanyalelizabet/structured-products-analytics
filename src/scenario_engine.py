from src.reverse_convertible import ReverseConvertible
import pandas as pd


class ScenarioEngine:

    def __init__(self, portfolio, beta_map, scenarios=None):
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
    
        idx = row["underlyings"].index(s["worst_underlying"])
    
        strike = row["strike"][idx]
        notional = row["notional"]
    
        total_shares = (notional / strike) 
        price = row["current_spots"][idx] * (1 + shocks[idx] / 100)
    
        is_physical = (price < strike)
    
        if is_physical:
            delivered_shares = int(total_shares)
            fractional_shares = total_shares - delivered_shares
            fractional_cash = fractional_shares * price
            delivered_underlying = s["worst_underlying"]
            cash_redemption = fractional_cash
            settlement_type = "physical"
        else:
            delivered_shares = 0
            fractional_shares = 0
            fractional_cash = 0
            delivered_underlying = None
            cash_redemption = s["total_payoff"]
            settlement_type = "cash"
    
        return {
            "product_id": row["product_id"],
            "product_type": row["product_type"],
            "currency": row["currency"],
            "position_units": row["position_units"],
            "notional": row["notional"],
            "total_cost": s["total_cost"],
            "market_shock": market_shock,
            "underlyings": row["underlyings"],
            "underlying_isins": row["underlying_isins"],
            "shocks": shocks,
            "payoff_per_unit": s["payoff_per_unit"],
            "total_payoff": s["total_payoff"],
            "pnl": s["pnl"],
            "return_pct": s["return_pct"],
            "barrier_breached": s["barrier_breached"],
            "worst_underlying": s["worst_underlying"],
            "settlement_type": settlement_type,
            "delivered_underlying": delivered_underlying,
            "delivered_shares": delivered_shares,
            "strike": strike,
            "price": price,
            "fractional_shares": fractional_shares,
            "fractional_cash": fractional_cash,
            "cash_redemption": cash_redemption
        }
        
    def run(self, market_shock):
            """
            Runs one market scenario across the portfolio.
            """
            results = []
        
            for _, row in self.portfolio.iterrows():
                res = self.run_product(row, market_shock)
                results.append(res)
        
            product_df = pd.DataFrame(results)
        
            # =========================
            # Helper columns for weighting
            # =========================
            product_df["strike_x_shares"] = product_df["strike"] * product_df["delivered_shares"]
            product_df["price_x_shares"] = product_df["price"] * product_df["delivered_shares"]
        
            # =========================
            # Cash positions
            # =========================
            cash_positions = (
                product_df.groupby("currency", as_index=False)
                .agg(
                    total_cash=("cash_redemption", "sum")
                )
            )
        
            # =========================
            # Delivered stocks (weighted)
            # =========================
            
            delivered_df = product_df[
                product_df["delivered_underlying"].notna()
                ].copy()
            
            delivered_stocks = (
                delivered_df.groupby(["currency", "delivered_underlying"], as_index=False)
                .agg(
                    total_shares=("delivered_shares", "sum"),
                    total_fractional_cash=("fractional_cash", "sum"),
                    strike_x_shares=("strike_x_shares", "sum"),
                    price_x_shares=("price_x_shares", "sum")
                )
            )
        
            # Weighted strike & price
            delivered_stocks["strike"] = (
                delivered_stocks["strike_x_shares"] /
                delivered_stocks["total_shares"]
            )
        
            delivered_stocks["price"] = (
                delivered_stocks["price_x_shares"] /
                delivered_stocks["total_shares"]
            )
        
            # Valuation
            delivered_stocks["market_value"] = (
                delivered_stocks["total_shares"] *
                delivered_stocks["price"]
            )
        
            delivered_stocks["cost"] = (
                delivered_stocks["total_shares"] *
                delivered_stocks["strike"]
            )
        
            # Include fractional cash
            delivered_stocks["total_value_incl_cash"] = (
                delivered_stocks["market_value"] +
                delivered_stocks["total_fractional_cash"]
            )
        
            delivered_stocks["pnl"] = (
                delivered_stocks["total_value_incl_cash"] -
                delivered_stocks["cost"]
            )
        
            delivered_stocks["return_pct"] = (
                delivered_stocks["pnl"] / delivered_stocks["cost"]
            )
        
            # Clean output
            delivered_stocks = delivered_stocks[
                [
                    
                    "delivered_underlying",
                    "total_shares",
                    "strike",
                    "price",
                    "currency",
                    "market_value",
                    "total_fractional_cash",
                    "total_value_incl_cash",
                    "cost",
                    "pnl",
                    "return_pct"
                ]
            ]
        
            # =========================
            # Portfolio per currency
            # =========================
            pf_scenario_per_ccy = (
                product_df.groupby("currency", as_index=False)
                .agg(
                    n_products=("product_id", "count"),
                    total_cost=("total_cost", "sum"),
                    total_payoff=("total_payoff", "sum"),
                    total_pnl=("pnl", "sum"),
                    underlyings=("underlyings", lambda x: sorted(set(sum(x, []))))
                )
            )
        
            pf_scenario_per_ccy["market_shock"] = market_shock
        
            pf_scenario_per_ccy["portfolio_return_pct"] = (
                pf_scenario_per_ccy["total_pnl"] /
                pf_scenario_per_ccy["total_cost"]
            )
        
            pf_scenario_per_ccy = pf_scenario_per_ccy[
                [
                    "market_shock",
                    "currency",
                    "n_products",
                    "underlyings",
                    "total_cost",
                    "total_payoff",
                    "total_pnl",
                    "portfolio_return_pct"
                ]
            ]
        
            return {
                "product_df": product_df,
                "pf_scenario_per_ccy": pf_scenario_per_ccy,
                "cash_positions": cash_positions,
                "delivered_stocks": delivered_stocks
            }
    
    def stress_test(self):
        if self.scenarios is None:
            self.scenarios = {"base": 0}
        product_results = []
        portfolio_results = []
        cash_results = []
        delivered_results = []
    
        for scenario_name, market_shock in self.scenarios.items():
    
            res = self.run(market_shock)
    
            product_df = res["product_df"].copy()
            pf_scenario_per_ccy = res["pf_scenario_per_ccy"].copy()
            cash_positions = res["cash_positions"].copy()
            delivered_stocks = res["delivered_stocks"].copy()
    
            product_df["scenario_name"] = scenario_name
            pf_scenario_per_ccy["scenario_name"] = scenario_name
            cash_positions["scenario_name"] = scenario_name
            delivered_stocks["scenario_name"] = scenario_name
    
            product_results.append(product_df)
            portfolio_results.append(pf_scenario_per_ccy)
            cash_results.append(cash_positions)
            delivered_results.append(delivered_stocks)
    
        return {
            "product_scenarios": pd.concat(product_results, ignore_index=True),
            "portfolio_scenarios": pd.concat(portfolio_results, ignore_index=True),
            "cash_scenarios": pd.concat(cash_results, ignore_index=True),
            "delivered_stock_scenarios": pd.concat(delivered_results, ignore_index=True)
        }                                   