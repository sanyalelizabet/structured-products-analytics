import numpy as np
from scipy.stats import norm


class BlackScholes:
    """
    Black-Scholes pricer for European vanilla options.

    Parameters
    ----------
    S     : current spot price
    K     : strike price
    T     : time to maturity in years
    r     : risk-free rate (continuous, annualised)
    sigma : implied volatility (annualised)
    q     : continuous dividend yield (default 0)
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    # ------------------------------------------------------------------
    # Core d1 / d2
    # ------------------------------------------------------------------

    def d1(self) -> float:
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    def d2(self) -> float:
        return self.d1() - self.sigma * np.sqrt(self.T)

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def call(self) -> float:
        if self.T <= 0:
            return max(self.S - self.K, 0.0)
        d1, d2 = self.d1(), self.d2()
        return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def put(self) -> float:
        if self.T <= 0:
            return max(self.K - self.S, 0.0)
        d1, d2 = self.d1(), self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(self, option: str = "call") -> float:
        """dV/dS"""
        d1 = self.d1()
        if option == "call":
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        return np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)

    def gamma(self) -> float:
        """d²V/dS²  — same for call and put"""
        d1 = self.d1()
        return np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """dV/d(sigma) — same for call and put, expressed per 1% vol move"""
        d1 = self.d1()
        return self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T) * 0.01

    def theta(self, option: str = "call") -> float:
        """dV/dt per day on an ACT/360 basis (1 day = 1/360 year), T decreasing."""
        d1, d2 = self.d1(), self.d2()
        common = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option == "call":
            return (common
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                    + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)) / 360
        return (common
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)) / 360

    def rho(self, option: str = "call") -> float:
        """dV/dr per 1% rate move"""
        d2 = self.d2()
        if option == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) * 0.01
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) * 0.01

    # ------------------------------------------------------------------
    # Put-call parity check
    # ------------------------------------------------------------------

    def put_call_parity_check(self) -> float:
        """Should be ~0. C - P = S*e^(-qT) - K*e^(-rT)"""
        lhs = self.call() - self.put()
        rhs = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        return lhs - rhs

    # ------------------------------------------------------------------
    # Implied vol (Newton-Raphson)
    # ------------------------------------------------------------------

    def implied_vol(self, market_price: float, option: str = "call", tol: float = 1e-6, max_iter: int = 100) -> float:
        """
        Infer implied volatility from a market price using Newton-Raphson.
        Returns NaN if it does not converge.
        """
        sigma = 0.20  # initial guess
        for _ in range(max_iter):
            bs = BlackScholes(self.S, self.K, self.T, self.r, sigma, self.q)
            price = bs.call() if option == "call" else bs.put()
            vega = bs.vega() * 100  # vega() returns per 1%, multiply back
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            if vega < 1e-10:
                return np.nan
            sigma -= diff / vega
            if sigma <= 0:
                return np.nan
        return np.nan
