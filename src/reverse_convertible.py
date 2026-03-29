from datetime import datetime
import numpy as np


class ReverseConvertible:
    """
    Payoff analytics for single-name and multi-name reverse convertibles
    under a user-defined scenario of final underlying moves.

    Parameters
    ----------
    row : pandas.Series
        Product data row from the portfolio DataFrame.
    final_levels : list[float]
        Scenario shocks in percent for each underlying, applied to current spot.
        Example: [-10, 5, 0] means -10%, +5%, 0%.
    """

    def __init__(self, row, final_levels):
        self.row = row
        
        # Basic inputs
        self.notional = row["notional"]
        self.position_units = row["position_units"]
        self.cost_price = row["cost_price"]
        self.coupon = row["coupon"]
        self.barrier_pct = row["barrier_pct"]
        self.product_type = row["product_type"]
        self.type_style = row["type_style"]
        
        # Underlyings
        self.underlyings = row["underlyings"]
        self.initial_levels = row["initial_levels"]
        self.strike_levels = row["strike"]
        self.current_spots = row["current_spots"]

        if len(final_levels) != len(self.current_spots):
            raise ValueError("Scenario length must match number of underlyings.")

        # final_levels here are scenario shocks in %
        self.final_levels = [
            spot * (1 + level / 100)
            for spot, level in zip(self.current_spots, final_levels)
        ]
        
    def is_multi(self):
        return len(self.initial_levels) > 1
    
    def performances(self):
        """
        Final performance vs strike.
        Better for payoff/redemption logic.
        """
        return [
            final / strike
            for final, strike in zip(self.final_levels, self.strike_levels)
        ]
    
    def current_performances(self):
        """
        Current spot performance vs strike.
        """
        return [
            spot / strike
            for spot, strike in zip(self.current_spots, self.strike_levels)
        ]
    
    def performance(self):
        """
        Relevant payoff performance:
        - BRC: single underlying performance
        - MBRC: worst-of performance
        """
        if self.is_multi():
            return min(self.performances())
        return self.performances()[0]
    
    def barrier_breaches_final(self):
        """
        European barrier observation at final fixing.
        Barrier level = strike * barrier_pct
        """
        return [
            final <= strike
            for final, strike in zip(self.final_levels, self.strike_levels)
        ]
    
    def barrier_breached(self):
        if self.type_style.lower() != "european":
            raise NotImplementedError("Only European style implemented.")
        return any(self.barrier_breaches_final())
    
    def redemption(self):
        """
        If no barrier event: nominal redemption.
        If barrier event: economic value of delivery.
        """
        if not self.barrier_breached():
            return self.notional
        return self.notional * self.performance()
    
    def total_product_time(self):
        start = datetime.strptime(self.row["initial_fixing_date"], "%Y-%m-%d")
        maturity = datetime.strptime(self.row["maturity_date"], "%Y-%m-%d")
        return (maturity - start).days / 365

    def coupon_payment(self):
        T_total = self.total_product_time()
        return self.notional * self.coupon * T_total
    
    def payoff_per_unit(self):
        return self.redemption() + self.coupon_payment()
    
    def total_payoff(self):
        return self.payoff_per_unit() * self.position_units
    
    def total_cost(self):
        return self.position_units * self.notional * self.cost_price
    
    def pnl(self):
        return self.total_payoff() - self.total_cost()
    
    def return_pct(self):
        total_cost = self.total_cost()
        if total_cost == 0:
            return np.nan
        return self.pnl() / total_cost
    
    
    def return_pa(self):
        days = (datetime.strptime(self.row["maturity_date"], "%Y-%m-%d")
                - datetime.strptime(self.row["initial_fixing_date"], "%Y-%m-%d")).days
        if days <= 0:
            return np.nan
        return self.return_pct() * 360 / days
    
    def current_barrier_distances(self):
        """
        Current normalized distance to barrier:
        current/strike - barrier_pct
        """
        return [
            (spot / strike) - self.barrier_pct
            for spot, strike in zip(self.current_spots, self.strike_levels)
        ]
    
    def distance_to_barrier(self):
        distances = self.current_barrier_distances()
        if self.is_multi():
            return min(distances)
        return distances[0]
    
    def break_even(self):
        return 1 - self.coupon * self.total_product_time()
    
    def worst_underlying(self):
        if self.is_multi():
            idx = np.argmin(self.performances())
            return self.underlyings[idx]
        return self.underlyings[0]
    
    def summary(self):
        return {
            "product_type": self.product_type,
            "is_multi": self.is_multi(),
            "performance": self.performance()-1,
            "barrier_breached": self.barrier_breached(),
            "worst_underlying": self.worst_underlying(),
            "payoff_per_unit": self.payoff_per_unit(),
            "total_payoff": self.total_payoff(),
            "total_cost": self.total_cost(),
            "pnl": self.pnl(),
            "return_pct": self.return_pct(),
            "return_pa": self.return_pa(),
            "distance_to_barrier": self.distance_to_barrier(),
            "break_even": self.break_even()
        }