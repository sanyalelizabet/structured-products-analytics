from src.reverse_convertible import ReverseConvertible
import pandas as pd
from datetime import datetime
import numpy as np

class ScenarioEngine:

    def __init__(self, portfolio, beta_map, vol_map, scenarios=None):
        self.portfolio = portfolio
        self.beta_map = beta_map
        self.scenarios = scenarios
        self.vol = vol_map
        
    def get_beta(self, isin):
        return self.beta_map.get(isin, 1.0)
    
    def get_vol(self, isin):
        return self.vol.get(isin, 0.15)
    
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
        Day-by-day GBM simulation from today to maturity.

        Each business day:
          1. Identify drift phase (pre-shock / between shocks / post-shock)
          2. Apply GBM step: S *= exp((drift - 0.5*vol²)*dt + vol*sqrt(dt)*Z)
          3. Apply discrete shock on shock dates (beta-scaled)

        The final path price is the terminal spot — used for both the payoff
        calculation and the path chart. Single source of truth.

        Parameters
        ----------
        scenario : dict
            market_shock        : float — % index move per shock (e.g. -20)
            n_shocks            : int   — number of shock events (default 1)
            shock_in_days       : int   — calendar days from today to first shock
            shock_spacing_days  : int   — calendar days between consecutive shocks
            pre_shock_drift_pa  : float — annualised drift before first shock
            post_shock_drift_pa : float — annualised drift after last shock to maturity
                                          0.0  → permanent shock (conservative)
                                          0.05 → ERP recovery
                                         -0.10 → bear market continuation

        Volatility is sourced from self.vol (isin-keyed dict, annualised).
        Beta    is sourced from self.beta_map (isin-keyed dict).

        Returns
        -------
        shocks      : list[float]   % change current→terminal per underlying
        final_spots : list[float]   terminal spot per underlying at maturity
        T_remaining : float         years from today to maturity
        path_summary: dict          human-readable audit trail
        paths       : dict[isin → pd.DataFrame(date, price)]
        """
        today    = pd.Timestamp.today().normalize()
        maturity = pd.Timestamp(row["maturity_date"])
        T_remaining = (maturity - today).days / 360

        # Scenario parameters
        market_shock       = scenario.get("market_shock", 0)
        n_shocks           = scenario.get("n_shocks", 1)
        shock_in_days      = scenario.get("shock_in_days", 0)
        shock_spacing_days = scenario.get("shock_spacing_days", 0)
        pre_shock_drift    = scenario.get("pre_shock_drift_pa", 0.0)
        post_shock_drift   = scenario.get("post_shock_drift_pa", 0.0)

        # Business-day grid from today to maturity (inclusive)
        date_range = pd.bdate_range(start=today, end=maturity)

        # Shock offsets (calendar days from today), filtered to product life
        shock_day_offsets = sorted(
            shock_in_days + i * shock_spacing_days
            for i in range(n_shocks)
            if (shock_in_days + i * shock_spacing_days) / 360 <= T_remaining
        )

        # Snap each shock offset to the nearest business day in the grid
        shock_dates = set()
        for d in shock_day_offsets:
            target = today + pd.Timedelta(days=d)
            if len(date_range) > 0:
                nearest_idx =  np.argmin(np.abs(date_range - target))
                shock_dates.add(date_range[nearest_idx])

        # Phase boundary (years from today) — same for all underlyings
        T_first_shock = shock_day_offsets[0] / 365 if shock_day_offsets else T_remaining
        T_last_shock  = shock_day_offsets[-1] / 365 if shock_day_offsets else 0.0
        T_post_shock  = max(T_remaining - T_last_shock, 0.0)

        shocks      = []
        final_spots = []
        paths       = {}

        for isin, spot in zip(row["underlying_isins"], row["current_spots"]):
            beta = self.get_beta(isin)
            vol  = self.get_vol(isin)

            current_price = float(spot)
            price_path    = []
            prev_date     = today

            for t_idx, bday in enumerate(date_range):
                # ACT/360 day fraction (0 for the first observation)
                dt      = (bday - prev_date).days / 360 if t_idx > 0 else 0.0
                t_years = (bday - today).days / 360

                # Drift phase
                if t_years <= T_first_shock:
                    drift = pre_shock_drift * beta
                elif t_years > T_last_shock:
                    drift = post_shock_drift * beta
                else:
                    drift = 0.0

                # GBM step (drift + optional volatility noise)
                if dt > 0:
                    Z = np.random.standard_normal()
                    current_price *= np.exp(
                        (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z
                    )

                # Discrete beta-scaled shock on shock dates
                if bday in shock_dates:
                    current_price *= (1 + market_shock / 100)

                price_path.append(current_price)
                prev_date = bday

            final_spot = price_path[-1] if price_path else float(spot)
            pct_change = (final_spot / spot - 1) * 100

            shocks.append(pct_change)
            final_spots.append(round(final_spot, 4))
            paths[isin] = pd.DataFrame({"date": date_range, "price": price_path})

        path_summary = {
            "maturity_date"       : row["maturity_date"],
            "T_remaining_years"   : round(T_remaining, 3),
            "T_first_shock_years" : round(T_first_shock, 3),
            "T_post_shock_years"  : round(T_post_shock, 3),
            "effective_n_shocks"  : len(shock_day_offsets),
            "market_shock_pct"    : market_shock,
            "pre_shock_drift_pa"  : pre_shock_drift,
            "post_shock_drift_pa" : post_shock_drift,
        }

        return shocks, final_spots, T_remaining, path_summary, paths

    

    def run_product_path_scenario(self, row, scenario):
        shocks, final_spots, T_remaining, path_summary, paths = self.build_shock_paths(row, scenario)
    
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