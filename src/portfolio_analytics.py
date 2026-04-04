import pandas as pd
import numpy as np
import requests
from datetime import datetime
from src.reverse_convertible import ReverseConvertible


class PortfolioAnalytics:

    def __init__(self, portfolio: pd.DataFrame, reference_currency: str = "CHF"):
        self.portfolio = portfolio
        self.reference_currency = reference_currency
        self.product_df = None
        self.fx_rates = self._get_fx_rates()

    # =========================
    # 1. PRODUCT LEVEL
    # =========================
    def build_product_analytics(self) -> pd.DataFrame:
        rows = []

        for _, row in self.portfolio.iterrows():
            rc = ReverseConvertible(row)
            s = rc.summary()

            rows.append({
                "product_id": row["product_id"],
                "product_type": row["product_type"],
                "type_style": row["type_style"],
                "currency": row["currency"],
                "position_units": row["position_units"],
                "notional": row["notional"],
                "coupon": row["coupon"],
                "underlyings": ", ".join(row["underlyings"]),
                "n_underlyings": len(row["underlyings"]),
                "initial_fixing_date": row["initial_fixing_date"],
                "maturity_date": row["maturity_date"],

                "performance": s["performance"],
                "barrier_breached": s["barrier_breached"],
                "worst_underlying": s["worst_underlying"],
                "payoff_per_unit": s["payoff_per_unit"],
                "total_payoff": s["total_payoff"],
                "total_cost": s["total_cost"],
                "pnl": s["pnl"],
                "return_pct": s["return_pct"],
                "return_pa": s["return_pa"],
                "distance_to_barrier": s["distance_to_barrier"],
                "break_even": s["break_even"]
            })

        self.product_df = pd.DataFrame(rows)
        return self.product_df
    def _get_fx_rates(self):
        url = f"https://api.frankfurter.app/latest?from={self.reference_currency}"
        response = requests.get(url)
        data = response.json()
    
        rates = data["rates"]
    
        fx_dict = {}

        for ccy, rate in rates.items():
            # invert to get ccy → reference_currency
            fx_dict[(ccy, self.reference_currency)] = 1 / rate
    
        return fx_dict
    
    def convert_to_reference(self, value, from_currency):
        if from_currency == self.reference_currency:
            return value
    
        rate = self.fx_rates.get((from_currency, self.reference_currency))
    
        if rate is None:
            raise ValueError(f"Missing FX rate: {from_currency}")
    
        return value * rate

    # =========================
    # 2. PORTFOLIO SUMMARY
    # =========================
    def portfolio_summary(self) -> pd.DataFrame:
        df = self._require_product_df()

        total_cost = df["total_cost"].sum()
        total_pnl = df["pnl"].sum()

        return pd.DataFrame([{
            "n_products": len(df),
            "n_brc": (df["product_type"] == "BRC").sum(),
            "n_mbrc": (df["product_type"] == "MBRC").sum(),
            "total_cost": total_cost,
            "total_payoff": df["total_payoff"].sum(),
            "total_pnl": total_pnl,
            "portfolio_return_pct": total_pnl / total_cost if total_cost != 0 else np.nan,
            "avg_return_pct": df["return_pct"].mean(),
            "weighted_return_pct": np.average(
                df["return_pct"],
                weights=df["total_cost"]
            ) if total_cost != 0 else np.nan,
            "worst_product_pnl": df["pnl"].min(),
            "best_product_pnl": df["pnl"].max(),
            "barrier_breached_count": df["barrier_breached"].sum(),
            "near_barrier_count": (df["distance_to_barrier"] <= 0.05).sum()
        }])

    # =========================
    # 3. UNDERLYING LOOKTHROUGH
    # =========================
    def underlying_lookthrough(self) -> pd.DataFrame:
        """
    Underlying-level exposure view.

    - Prices are shown in their natural currency
    - Allocated values are converted into reference currency
    - Shows both worst-case and average barrier distance
    """

        rows = []
    
        for _, row in self.portfolio.iterrows():
            rc = ReverseConvertible(row)
    
            product_cost = rc.total_cost()
            product_ccy = row["currency"]
            barrier_distances = rc.current_barrier_distances()
    
            for i, underlying in enumerate(row["underlyings"]):
                rows.append({
                    "product_id": row["product_id"],
                    "underlying": underlying,
                    "isin": row["underlying_isins"][i],
    
                    # exposure converted to reference currency
                    "allocated_cost_ref": self.convert_to_reference(
                        product_cost, product_ccy
                    ),
    
                    # market data (native currency)
                    "current_spot": row["current_spots"][i],
                    "price_ccy": product_ccy,
    
                    # risk
                    "distance_to_barrier": barrier_distances[i],
                    "is_worst_underlying": underlying == rc.worst_underlying()
                })  
    
        df = pd.DataFrame(rows)
    
        summary = (
            df.groupby(["underlying", "isin", "price_ccy"], as_index=False)
            .agg(
                n_products=("product_id", "nunique"),
                allocated_cost_ref=("allocated_cost_ref", "sum"),
                avg_current_spot=("current_spot", "mean"),
    
                
                min_distance_to_barrier=("distance_to_barrier", "min"),
                avg_distance_to_barrier=("distance_to_barrier", "mean"),
    
                worst_of_count=("is_worst_underlying", "sum")
            )
            .sort_values("allocated_cost_ref", ascending=False)
            .reset_index(drop=True)
        )
    
        total_alloc = summary["allocated_cost_ref"].sum()
    
        summary["weight"] = (
            summary["allocated_cost_ref"] / total_alloc
            if total_alloc != 0 else np.nan
        )
    
        return summary
    
    def total_portfolio_metrics(self) -> dict:
        product_analytics = self._require_product_df()
    
        total_notional = 0
        total_cost = 0
        total_payoff = 0
        total_pnl = 0
    
        for _, row in product_analytics.iterrows():
            currency = row["currency"]
    
            total_notional += self.convert_to_reference(
                row["notional"],
                currency
            )
            total_cost += self.convert_to_reference(
                row["total_cost"],
                currency
            )
            total_payoff += self.convert_to_reference(
                row["total_payoff"],
                currency
            )
            total_pnl += self.convert_to_reference(
                row["pnl"],
                currency
            )
    
        portfolio_return = total_pnl / total_cost if total_cost != 0 else np.nan
    
        weighted_return_pa = np.average(
            product_analytics["return_pa"],
            weights=product_analytics["total_cost"]
        ) if total_cost != 0 else np.nan
    
        return {
            "reference_currency": self.reference_currency,
            "total_products": len(product_analytics),
            "total_notional": total_notional,
            "total_cost": total_cost,
            "total_payoff": total_payoff,
            "total_pnl": total_pnl,
            "portfolio_return_pct": portfolio_return,
            "portfolio_return_pa": weighted_return_pa
        }
    
    def total_portfolio_table(self) -> pd.DataFrame:
        return pd.DataFrame([self.total_portfolio_metrics()])
    
    

    # =========================
    # 4. BARRIER WATCHLIST
    # =========================
    def barrier_watchlist(self, threshold: float = 0.05) -> pd.DataFrame:
        df = self._require_product_df().copy()
        df["near_barrier_flag"] = df["distance_to_barrier"] <= threshold

        return df.sort_values(
            ["near_barrier_flag", "distance_to_barrier", "pnl"],
            ascending=[False, True, True]
        ).reset_index(drop=True)

    # =========================
    # 5. MATURITY PROFILE
    # =========================
    def maturity_profile(self, today=None) -> pd.DataFrame:
        df = self._require_product_df().copy()

        if today is None:
            today = datetime.today()

        df["maturity_date_dt"] = pd.to_datetime(df["maturity_date"])
        df["days_to_maturity"] = (df["maturity_date_dt"] - pd.Timestamp(today)).dt.days

        bins = [-np.inf, 30, 90, 180, 365, 730, np.inf]
        labels = ["<=1M", "1-3M", "3-6M", "6-12M", "1-2Y", ">2Y"]

        df["maturity_bucket"] = pd.cut(df["days_to_maturity"], bins=bins, labels=labels)

        return (
            df.groupby("maturity_bucket", as_index=False, observed=False)
            .agg(
                n_products=("product_id", "count"),
                total_cost=("total_cost", "sum"),
                total_payoff=("total_payoff", "sum"),
                total_pnl=("pnl", "sum")
            )
        )

    # =========================
    # 6. SCENARIO ATTRIBUTION
    # =========================
    def scenario_attribution(self) -> pd.DataFrame:
        df = self._require_product_df().copy()

        total_pnl = df["pnl"].sum()

        df["pnl_contribution_pct"] = (
            df["pnl"] / total_pnl if total_pnl != 0 else np.nan
        )

        df["cost_weight"] = (
            df["total_cost"] / df["total_cost"].sum()
            if df["total_cost"].sum() != 0 else np.nan
        )

        return df[
            [
                "product_id",
                "product_type",
                "underlyings",
                "worst_underlying",
                "barrier_breached",
                "total_cost",
                "pnl",
                "pnl_contribution_pct",
                "return_pct",
                "distance_to_barrier"
            ]
        ].sort_values("pnl", ascending=True).reset_index(drop=True)

    # =========================
    # 7. RUN ALL
    # =========================
    def run_all(self) -> dict:
        self.build_product_analytics()

        return {
            "product_analytics": self.product_df,
            "portfolio_summary": self.portfolio_summary(),
            "underlying_lookthrough": self.underlying_lookthrough(),
            "barrier_watchlist": self.barrier_watchlist(),
            "maturity_profile": self.maturity_profile(),
            "scenario_attribution": self.scenario_attribution()
        }

    # =========================
    # INTERNAL HELPER
    # =========================
    def _require_product_df(self):
        if self.product_df is None:
            raise ValueError("Run build_product_analytics() first.")
        return self.product_df
    
    