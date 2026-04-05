from src.reverse_convertible import ReverseConvertible
import pandas as pd
from datetime import datetime
import numpy as np

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

    def simulate_terminal_spots(self, row, n_paths=10000, vol_map=None, drift_pa=0.0):
        """
        Monte Carlo simulation of terminal spot per underlying.
        Returns distribution of payoffs across n_paths.
        """
        today = datetime.today()
        maturity = datetime.strptime(row["maturity_date"], "%Y-%m-%d")
        T = (maturity - today).days / 365
    
        results = []
    
        for isin, spot in zip(row["underlying_isins"], row["current_spots"]):
            vol = vol_map.get(isin, 0.25)  # annualised vol per underlying
    
            # GBM terminal distribution
            # S(T) = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
            Z = np.random.standard_normal(n_paths)
            S_T = spot * np.exp(
                (drift_pa - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z
            )
            results.append(S_T)
    
        return np.array(results)   


    def build_shock_paths(self, row, scenario):
        """
        Piecewise deterministic path to terminal spot at product maturity.
    
       
        -----------------------------------------
        S_final = S_current
                  * exp(pre_shock_drift  * T_first_shock)     # drift to shock
                  * (1 + beta * market_shock/100)^n_shocks    # compounded shocks
                  * exp(post_shock_drift * T_post_shock)      # drift to maturity
    
        Post-shock drift interpretation
        --------------------------------
        0.0   : permanent shock — most conservative stress assumption
        >0    : recovery — approximates equity risk premium (~5% for Swiss equities)
        <0    : continued deterioration — bear market / crisis scenario
    
        Parameters
        ----------
        scenario : dict
            market_shock        : float  — % index move per shock event (e.g. -20)
            n_shocks            : int    — number of shock events (default 1)
            shock_in_days       : int    — days from today to first shock (default 0)
            shock_spacing_days  : int    — days between shocks if n_shocks > 1
            pre_shock_drift_pa  : float  — annual drift before first shock (default 0.0)
            post_shock_drift_pa : float  — annual drift after last shock to maturity
                                           0.0 = permanent shock
                                           0.05 = ERP-based recovery
                                           -0.10 = bear market continuation
    
        Returns
        -------
        shocks : list of float
            % change from current spot to terminal spot per underlying.
            Passed directly into ReverseConvertible(final_levels=shocks).
    
        final_spots : list of float
            Actual terminal spot level per underlying at product maturity.
            Used for display and audit — not passed into ReverseConvertible.
    
        T_remaining : float
            Years from today to product maturity.
    
        path_summary : dict
            Human-readable path description for display in Streamlit.
        """
        today = datetime.today()
        maturity = datetime.strptime(row["maturity_date"], "%Y-%m-%d")
        T_remaining = (maturity - today).days / 365
    
        # Unpack scenario parameters
        market_shock        = scenario.get("market_shock", 0)
        n_shocks            = scenario.get("n_shocks", 1)
        shock_in_days       = scenario.get("shock_in_days", 0)
        shock_spacing_days  = scenario.get("shock_spacing_days", 0)
        pre_shock_drift     = scenario.get("pre_shock_drift_pa", 0.0)
        post_shock_drift    = scenario.get("post_shock_drift_pa", 0.0)
    
        # Timing — all capped at maturity
        T_first_shock = min(shock_in_days / 365, T_remaining)
        T_last_shock  = min(
            (shock_in_days + shock_spacing_days * (n_shocks - 1)) / 365,
            T_remaining
        )
        T_post_shock  = max(T_remaining - T_last_shock, 0.0)
    
        shocks      = []
        final_spots = []
    
        for isin, spot in zip(row["underlying_isins"], row["current_spots"]):
            beta = self.get_beta(isin)
    
            # ─────────────────────────────────────────
            # Phase 1: pre-shock drift to first shock
            # ─────────────────────────────────────────
            spot_at_shock = spot * np.exp(pre_shock_drift * T_first_shock)
    
            # ─────────────────────────────────────────
            # Phase 2: n compounded discrete shocks
            # beta scales each shock to the underlying
            # ─────────────────────────────────────────
            spot_after_shocks = spot_at_shock * (
                (1 + beta * market_shock / 100) ** n_shocks
            )
    
            # ─────────────────────────────────────────
            # Phase 3: post-shock drift to maturity
            # this is where ERP / recovery assumption lives
            # ─────────────────────────────────────────
            final_spot = spot_after_shocks * np.exp(post_shock_drift * T_post_shock)
    
            # % change from current spot — what ReverseConvertible expects
            pct_change = (final_spot / spot - 1) * 100
    
            shocks.append(pct_change)
            final_spots.append(round(final_spot, 4))
    
        # ─────────────────────────────────────────
        # Path summary — for display and audit
        # ─────────────────────────────────────────
        path_summary = {
            "maturity_date"     : row["maturity_date"],
            "T_remaining_years" : round(T_remaining, 3),
            "T_first_shock"     : round(T_first_shock, 3),
            "T_post_shock"      : round(T_post_shock, 3),
            "n_shocks"          : n_shocks,
            "market_shock_pct"  : market_shock,
            "pre_shock_drift_pa": pre_shock_drift,
            "post_shock_drift_pa": post_shock_drift,
        }
    
        return shocks, final_spots, T_remaining, path_summary

    def run_product_path_scenario(self, row, scenario):
        shocks, final_spots, T_remaining, path_summary = self.build_shock_paths(row, scenario)
    
        rc = ReverseConvertible(row, final_levels=shocks)
        s = rc.summary()
    
        idx = row["underlyings"].index(s["worst_underlying"])
        strike = row["strike"][idx]
        notional = row["notional"]
    
        total_shares = notional / strike
        price = final_spots[idx]  # ← final spot at maturity, not current * shock
    
        is_physical = price < strike
    
        if is_physical:
            delivered_shares     = int(total_shares)
            fractional_shares    = total_shares - delivered_shares
            fractional_cash      = fractional_shares * price
            delivered_underlying = s["worst_underlying"]
            cash_redemption      = fractional_cash
            settlement_type      = "physical"
        else:
            delivered_shares     = 0
            fractional_shares    = 0
            fractional_cash      = 0
            delivered_underlying = None
            cash_redemption      = s["total_payoff"]
            settlement_type      = "cash"
    
        return {
            # Product identifiers
            "product_id"          : row["product_id"],
            "product_type"        : row["product_type"],
            "currency"            : row["currency"],
            "position_units"      : row["position_units"],
            "notional"            : row["notional"],
            "maturity_date"       : row["maturity_date"],
    
            # Path
            "T_remaining_years"   : round(T_remaining, 2),
            "final_spots"         : final_spots,  # terminal levels at maturity
    
            # Scenario inputs — explicit for auditability
            "market_shock"        : scenario.get("market_shock", 0),
            "n_shocks"            : scenario.get("n_shocks", 1),
            "shock_in_days"       : scenario.get("shock_in_days", 0),
            "pre_shock_drift_pa"  : scenario.get("pre_shock_drift_pa", 0.0),
            "post_shock_drift_pa" : scenario.get("post_shock_drift_pa", 0.0),
    
            # Payoff
            "total_cost"          : s["total_cost"],
            "payoff_per_unit"     : s["payoff_per_unit"],
            "total_payoff"        : s["total_payoff"],
            "pnl"                 : s["pnl"],
            "return_pct"          : s["return_pct"],
            "barrier_breached"    : s["barrier_breached"],
            "worst_underlying"    : s["worst_underlying"],
    
            # Settlement
            "settlement_type"     : settlement_type,
            "delivered_underlying": delivered_underlying,
            "delivered_shares"    : delivered_shares,
            "strike"              : strike,
            "price"               : price,  # final spot at maturity of worst underlying
            "fractional_shares"   : fractional_shares,
            "fractional_cash"     : fractional_cash,
            "cash_redemption"     : cash_redemption,
    
            # Full path audit
            "path_summary"        : path_summary
        } 
    def run_path_scenario(self, scenario):
        """
        Runs path-based scenario across full portfolio
        (equivalent to run(), but using path logic)
        """
    
        results = []
    
        for _, row in self.portfolio.iterrows():
            res = self.run_product_path_scenario(row, scenario)
            results.append(res)
    
        product_df = pd.DataFrame(results)
    
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
        # Delivered stocks
        # =========================
        delivered_df = product_df[
            product_df["delivered_underlying"].notna()
        ].copy()
    
        if not delivered_df.empty:
    
            delivered_stocks = (
                delivered_df.groupby(["currency", "delivered_underlying"], as_index=False)
                .agg(
                    total_shares=("delivered_shares", "sum"),
                    total_fractional_cash=("fractional_cash", "sum"),
                    strike_x_shares=("strike", lambda x: (x * delivered_df.loc[x.index, "delivered_shares"]).sum()),
                    price_x_shares=("price", lambda x: (x * delivered_df.loc[x.index, "delivered_shares"]).sum()),
                )
            )
    
            delivered_stocks["strike"] = delivered_stocks["strike_x_shares"] / delivered_stocks["total_shares"]
            delivered_stocks["price"] = delivered_stocks["price_x_shares"] / delivered_stocks["total_shares"]
    
            delivered_stocks["market_value"] = delivered_stocks["total_shares"] * delivered_stocks["price"]
            delivered_stocks["cost"] = delivered_stocks["total_shares"] * delivered_stocks["strike"]
    
            delivered_stocks["total_value_incl_cash"] = (
                delivered_stocks["market_value"] +
                delivered_stocks["total_fractional_cash"]
            )
    
            delivered_stocks["pnl"] = (
                delivered_stocks["total_value_incl_cash"] -
                delivered_stocks["cost"]
            )
    
            delivered_stocks["return_pct"] = delivered_stocks["pnl"] / delivered_stocks["cost"]
    
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
    
        else:
            delivered_stocks = pd.DataFrame()
    
        # =========================
        # Portfolio aggregation
        # =========================
        pf_scenario_per_ccy = (
            product_df.groupby("currency", as_index=False)
            .agg(
                n_products=("product_id", "count"),
                total_cost=("total_cost", "sum"),
                total_payoff=("total_payoff", "sum"),
                total_pnl=("pnl", "sum"),
                underlyings=("worst_underlying", lambda x: sorted(set(x)))
            )
        )
    
        pf_scenario_per_ccy["portfolio_return_pct"] = (
            pf_scenario_per_ccy["total_pnl"] /
            pf_scenario_per_ccy["total_cost"]
        )
    
        return {
            "product_df": product_df,
            "pf_scenario_per_ccy": pf_scenario_per_ccy,
            "cash_positions": cash_positions,
            "delivered_stocks": delivered_stocks
        }                             