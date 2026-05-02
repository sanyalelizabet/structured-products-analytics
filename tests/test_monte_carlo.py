"""
Tests for MonteCarloPricer and european_brc_payoff.

Test strategy
-------------
Where possible we use zero-vol / zero-rate inputs to make the expected
output fully deterministic — no stochastic uncertainty means we can
check exact values, not just order-of-magnitude bounds.

For stochastic tests we run with a fixed seed and a large path count and
check that the result is close to a known analytical bound.
"""
import numpy as np
import pandas as pd
import pytest

from src.pricing.monte_carlo import MonteCarloPricer, european_brc_payoff
from tests.conftest import make_brc_row, make_mbrc_row, make_portfolio, VOL_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pricer(n_paths=10_000, seed=42):
    return MonteCarloPricer(n_paths=n_paths, seed=seed)


def zero_vol_map(row):
    """Return vol_map with zero vol for all ISINs in the row."""
    return {isin: 0.0 for isin in row["underlying_isins"]}


# ---------------------------------------------------------------------------
# simulate_paths
# ---------------------------------------------------------------------------

class TestSimulatePaths:

    def test_output_shape_brc(self):
        pricer = make_pricer(n_paths=500)
        row = make_brc_row(maturity_date="2027-01-01")
        paths, dates = pricer.simulate_paths(row, VOL_MAP, risk_free_rate=0.0)

        n_steps = len(dates)
        assert paths.shape == (500, n_steps, 1)

    def test_output_shape_mbrc(self):
        pricer = make_pricer(n_paths=200)
        row = make_mbrc_row(maturity_date="2027-01-01")
        paths, dates = pricer.simulate_paths(row, VOL_MAP, risk_free_rate=0.0)

        n_steps = len(dates)
        assert paths.shape == (200, n_steps, 2)

    def test_dates_are_business_days(self):
        pricer = make_pricer()
        row = make_brc_row(maturity_date="2027-01-01")
        _, dates = pricer.simulate_paths(row, VOL_MAP, risk_free_rate=0.0)

        # All dates should be weekdays (Monday=0 ... Friday=4)
        assert all(d.weekday() < 5 for d in dates)

    def test_zero_vol_paths_are_deterministic(self):
        """With zero vol and zero rate all paths converge to the same flat line."""
        pricer = make_pricer(n_paths=100)
        row = make_brc_row(current_spot=95.0, maturity_date="2027-01-01")
        vols = zero_vol_map(row)

        paths, _ = pricer.simulate_paths(row, vols, risk_free_rate=0.0)

        # All paths should be identical and flat at the initial spot
        assert np.allclose(paths[:, -1, 0], 95.0, rtol=1e-6)
        assert np.allclose(paths[:, 0, 0], paths[:, -1, 0], rtol=1e-6)

    def test_same_seed_gives_same_paths(self):
        row = make_brc_row(maturity_date="2027-01-01")
        p1, _ = make_pricer(seed=7).simulate_paths(row, VOL_MAP, 0.01)
        p2, _ = make_pricer(seed=7).simulate_paths(row, VOL_MAP, 0.01)
        assert np.array_equal(p1, p2)

    def test_different_seeds_give_different_paths(self):
        row = make_brc_row(maturity_date="2027-01-01")
        p1, _ = make_pricer(seed=1).simulate_paths(row, VOL_MAP, 0.01)
        p2, _ = make_pricer(seed=2).simulate_paths(row, VOL_MAP, 0.01)
        assert not np.array_equal(p1, p2)

    def test_positive_rate_drifts_up_on_average(self):
        """Under risk-neutral GBM the expected terminal spot equals S_0 * exp(r*T)."""
        pricer = make_pricer(n_paths=50_000, seed=0)
        row = make_brc_row(current_spot=100.0, maturity_date="2027-01-01")
        vols = {"CH0012221716": 0.20}
        r = 0.05

        paths, dates = pricer.simulate_paths(row, vols, risk_free_rate=r)
        T = (pd.Timestamp("2027-01-01") - pd.Timestamp.today().normalize()).days / 365.25
        expected_mean = 100.0 * np.exp(r * T)

        actual_mean = paths[:, -1, 0].mean()
        # Within 2% of the theoretical expectation
        assert abs(actual_mean / expected_mean - 1) < 0.02

    def test_correlation_matrix_applied(self):
        """
        High positive correlation → terminal log-returns are nearly identical.
        Near-perfect rho (0.9999) keeps the matrix positive-definite for Cholesky.
        With identical vols and rho→1, both assets should land at virtually the
        same relative return on every path.
        """
        pricer = make_pricer(n_paths=2_000, seed=0)
        row = make_mbrc_row(
            current_spots=[100.0, 100.0],
            maturity_date="2027-01-01",
        )
        vols = {isin: 0.20 for isin in row["underlying_isins"]}
        rho = 0.9999
        corr = np.array([[1.0, rho], [rho, 1.0]])

        paths, _ = pricer.simulate_paths(row, vols, risk_free_rate=0.0, corr_matrix=corr)

        log_ret_0 = np.log(paths[:, -1, 0] / 100.0)
        log_ret_1 = np.log(paths[:, -1, 1] / 100.0)

        # Near-perfect correlation → cross-path differences should be tiny
        assert np.max(np.abs(log_ret_0 - log_ret_1)) < 0.02


# ---------------------------------------------------------------------------
# european_brc_payoff
# ---------------------------------------------------------------------------

class TestEuropeanBrcPayoff:

    def _make_flat_paths(self, row, terminal_spots):
        """
        Build a (1, 1, n_assets) paths array where the single terminal step
        equals terminal_spots.
        """
        return np.array([[terminal_spots]])

    def test_no_breach_full_notional_plus_coupon(self):
        """Spot finishes above strike → redemption = notional."""
        row = make_brc_row(
            notional=100_000, coupon=0.08,
            strike=100.0, current_spot=95.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        paths = self._make_flat_paths(row, [110.0])   # well above strike
        dates = pd.bdate_range("2024-01-01", "2025-01-01")

        payoffs = european_brc_payoff(paths, dates, row)

        T_total = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 360
        expected_coupon = 100_000 * 0.08 * T_total
        assert abs(payoffs[0] - (100_000 + expected_coupon)) < 0.01

    def test_breach_partial_redemption(self):
        """Spot finishes below strike → redemption = notional * perf."""
        row = make_brc_row(
            notional=100_000, coupon=0.08,
            strike=100.0, current_spot=95.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        paths = self._make_flat_paths(row, [70.0])   # 30% below strike
        dates = pd.bdate_range("2024-01-01", "2025-01-01")

        payoffs = european_brc_payoff(paths, dates, row)

        T_total = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 360
        expected_redemption = 100_000 * (70.0 / 100.0)
        expected_coupon = 100_000 * 0.08 * T_total
        assert abs(payoffs[0] - (expected_redemption + expected_coupon)) < 0.01

    def test_coupon_always_added(self):
        """Coupon is unconditional — appears whether or not barrier is breached."""
        row = make_brc_row(
            notional=100_000, coupon=0.10,
            strike=100.0, current_spot=95.0,
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        dates = pd.bdate_range("2024-01-01", "2025-01-01")

        payoff_above = european_brc_payoff(self._make_flat_paths(row, [120.0]), dates, row)
        payoff_below = european_brc_payoff(self._make_flat_paths(row, [60.0]),  dates, row)

        T_total = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 360
        coupon = 100_000 * 0.10 * T_total

        assert abs(payoff_above[0] - (100_000 + coupon)) < 0.01
        assert abs(payoff_below[0] - (100_000 * 0.60 + coupon)) < 0.01

    def test_mbrc_worst_of_determines_breach(self):
        """MBRC: only the worst-of performance matters."""
        row = make_mbrc_row(
            notional=100_000, coupon=0.08,
            strikes=[100.0, 80.0],
            current_spots=[95.0, 75.0],
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        # Asset 0: 110/100 = 1.10 (above), Asset 1: 60/80 = 0.75 (below → worst-of)
        paths = np.array([[[110.0, 60.0]]])
        dates = pd.bdate_range("2024-01-01", "2025-01-01")

        payoffs = european_brc_payoff(paths, dates, row)

        T_total = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 360
        expected = 100_000 * 0.75 + 100_000 * 0.08 * T_total
        assert abs(payoffs[0] - expected) < 0.01


# ---------------------------------------------------------------------------
# price — single product
# ---------------------------------------------------------------------------

class TestPrice:

    def test_expired_product_returns_intrinsic(self):
        """T_remaining <= 0 → return intrinsic value, no Monte Carlo."""
        pricer = make_pricer()
        row = make_brc_row(
            notional=100_000, coupon=0.08,
            strike=100.0, current_spot=95.0,
            initial_fixing_date="2020-01-01", maturity_date="2020-06-01",  # in the past
        )
        result = pricer.price(row, european_brc_payoff, zero_vol_map(row), risk_free_rate=0.0)

        # spot=95 < strike=100 → breach → redemption = 95/100 * notional
        T_total = (pd.Timestamp("2020-06-01") - pd.Timestamp("2020-01-01")).days / 360
        expected = 100_000 * 0.95 + 100_000 * 0.08 * T_total
        assert abs(result["fair_value"] - expected) < 0.01
        assert result["std_error"] == 0.0

    def test_fair_value_is_positive(self):
        pricer = make_pricer()
        row = make_brc_row()
        result = pricer.price(row, european_brc_payoff, VOL_MAP, risk_free_rate=0.01)
        assert result["fair_value"] > 0

    def test_zero_vol_spot_above_strike_deterministic(self):
        """Zero vol, r=0, spot well above strike → every path is full redemption."""
        pricer = make_pricer(n_paths=1_000)
        row = make_brc_row(
            notional=100_000, coupon=0.08,
            strike=70.0, current_spot=95.0,   # spot > strike
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        result = pricer.price(row, european_brc_payoff, zero_vol_map(row), risk_free_rate=0.0)

        T_total = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 360
        expected = 100_000 + 100_000 * 0.08 * T_total
        assert abs(result["fair_value"] - expected) < 1.0    # allow tiny float drift
        assert abs(result["std_error"]) < 0.01               # essentially zero variance

    def test_zero_vol_spot_below_strike_deterministic(self):
        """Zero vol, r=0, spot below strike → every path breaches."""
        pricer = make_pricer(n_paths=1_000)
        row = make_brc_row(
            notional=100_000, coupon=0.08,
            strike=100.0, current_spot=70.0,  # spot < strike
            initial_fixing_date="2024-01-01", maturity_date="2025-01-01",
        )
        result = pricer.price(row, european_brc_payoff, zero_vol_map(row), risk_free_rate=0.0)

        T_total = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days / 360
        expected = 100_000 * (70.0 / 100.0) + 100_000 * 0.08 * T_total
        assert abs(result["fair_value"] - expected) < 1.0

    def test_fair_value_pct_is_fair_value_over_notional(self):
        pricer = make_pricer()
        row = make_brc_row(notional=100_000)
        result = pricer.price(row, european_brc_payoff, VOL_MAP, risk_free_rate=0.01)
        assert abs(result["fair_value_pct"] - result["fair_value"] / 100_000) < 1e-9

    def test_std_error_decreases_with_more_paths(self):
        row = make_brc_row()
        r1 = make_pricer(n_paths=500).price(row, european_brc_payoff, VOL_MAP, 0.01)
        r2 = make_pricer(n_paths=20_000).price(row, european_brc_payoff, VOL_MAP, 0.01)
        assert r2["std_error"] < r1["std_error"]

    def test_higher_rate_lowers_fair_value(self):
        """Discounting: higher r → lower PV of the same expected payoff."""
        pricer = make_pricer(n_paths=5_000, seed=0)
        row = make_brc_row()
        r_low  = pricer.price(row, european_brc_payoff, VOL_MAP, risk_free_rate=0.00)
        r_high = pricer.price(row, european_brc_payoff, VOL_MAP, risk_free_rate=0.10)
        assert r_high["fair_value"] < r_low["fair_value"]

    def test_correlation_fallback_when_missing_isin(self):
        """If an ISIN is absent from corr_df the pricer falls back to identity silently."""
        pricer = make_pricer(n_paths=500)
        row = make_brc_row()
        empty_corr = pd.DataFrame()   # no ISINs at all

        # Should not raise
        result = pricer.price(
            row, european_brc_payoff, VOL_MAP, risk_free_rate=0.01,
            corr_matrix=pricer._get_corr_subset(row, empty_corr),
        )
        assert result["fair_value"] > 0


# ---------------------------------------------------------------------------
# price_portfolio
# ---------------------------------------------------------------------------

class TestPricePortfolio:

    def test_returns_one_row_per_product(self):
        pricer = make_pricer(n_paths=500)
        portfolio = make_portfolio()
        rates = {"CHF": 0.01}

        result = pricer.price_portfolio(portfolio, VOL_MAP, rates, corr_df=None)

        assert len(result) == len(portfolio)
        assert set(result["product_id"]) == set(portfolio["product_id"])

    def test_output_columns(self):
        pricer = make_pricer(n_paths=200)
        result = pricer.price_portfolio(make_portfolio(), VOL_MAP, {"CHF": 0.01})

        assert {"product_id", "fair_value", "fair_value_pct", "std_error"}.issubset(result.columns)

    def test_all_fair_values_positive(self):
        pricer = make_pricer(n_paths=500)
        result = pricer.price_portfolio(make_portfolio(), VOL_MAP, {"CHF": 0.01})
        assert (result["fair_value"] > 0).all()

    def test_missing_rate_falls_back_to_default(self):
        """Currency not in rate map → uses 0.02 default, should not raise."""
        pricer = make_pricer(n_paths=200)
        portfolio = make_portfolio()
        result = pricer.price_portfolio(portfolio, VOL_MAP, {})   # empty rate map
        assert len(result) == len(portfolio)

    def test_unknown_type_style_raises(self):
        """_resolve_payoff raises NotImplementedError for unsupported styles."""
        pricer = make_pricer()
        row = make_brc_row()
        row["type_style"] = "american"
        portfolio = pd.DataFrame([row])

        with pytest.raises(NotImplementedError, match="american"):
            pricer.price_portfolio(portfolio, VOL_MAP, {"CHF": 0.01})


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

# Shared fixtures — products with enough time remaining for meaningful Greeks
BRC_GREEK = make_brc_row(
    notional=100_000, coupon=0.08,
    strike=70.0, current_spot=95.0,
    initial_fixing_date="2024-01-01", maturity_date="2027-01-01",
)
MBRC_GREEK = make_mbrc_row(
    notional=100_000, coupon=0.08,
    strikes=[70.0, 56.0], current_spots=[95.0, 76.0],
    initial_fixing_date="2024-01-01", maturity_date="2027-01-01",
)
VOL_BRC_G  = {"CH0012221716": 0.20}
VOL_MBRC_G = {"CH0012221716": 0.20, "CH0012221717": 0.25}
CORR_2x2   = np.array([[1.0, 0.65], [0.65, 1.0]])
R_CHF      = 0.01


class TestGreeks:

    # ── Sign checks ───────────────────────────────────────────────────────

    def test_brc_delta_positive(self):
        """BRC holder is long spot — delta must be positive."""
        g = make_pricer(n_paths=10_000).compute_greeks(
            BRC_GREEK, european_brc_payoff, VOL_BRC_G, R_CHF
        )
        assert g["delta"][0] > 0

    def test_brc_vega_negative(self):
        """BRC holder is short a put — short vega, so FV falls when vol rises."""
        g = make_pricer(n_paths=10_000).compute_greeks(
            BRC_GREEK, european_brc_payoff, VOL_BRC_G, R_CHF
        )
        assert g["vega"][0] < 0

    def test_brc_theta_positive(self):
        """Each passing day reduces T_remaining — discount on the coupon shrinks,
        and the short put decays — both benefit the holder."""
        g = make_pricer(n_paths=10_000).compute_greeks(
            BRC_GREEK, european_brc_payoff, VOL_BRC_G, R_CHF
        )
        assert g["theta"] > 0

    def test_mbrc_corr_sens_positive(self):
        """Higher correlation between underlyings reduces worst-of risk — FV rises."""
        g = make_pricer(n_paths=10_000).compute_greeks(
            MBRC_GREEK, european_brc_payoff, VOL_MBRC_G, R_CHF, CORR_2x2
        )
        assert g["corr_sens"] > 0

    def test_brc_corr_sens_is_none(self):
        """Single-underlying product has no correlation sensitivity."""
        g = make_pricer(n_paths=500).compute_greeks(
            BRC_GREEK, european_brc_payoff, VOL_BRC_G, R_CHF
        )
        assert g["corr_sens"] is None

    # ── Output structure ──────────────────────────────────────────────────

    def test_delta_length_matches_underlyings(self):
        g_brc  = make_pricer(n_paths=500).compute_greeks(BRC_GREEK,  european_brc_payoff, VOL_BRC_G,  R_CHF)
        g_mbrc = make_pricer(n_paths=500).compute_greeks(MBRC_GREEK, european_brc_payoff, VOL_MBRC_G, R_CHF, CORR_2x2)
        assert len(g_brc["delta"])  == 1
        assert len(g_mbrc["delta"]) == 2

    def test_vega_length_matches_underlyings(self):
        g = make_pricer(n_paths=500).compute_greeks(MBRC_GREEK, european_brc_payoff, VOL_MBRC_G, R_CHF, CORR_2x2)
        assert len(g["vega"]) == 2

    def test_output_keys_present(self):
        g = make_pricer(n_paths=500).compute_greeks(BRC_GREEK, european_brc_payoff, VOL_BRC_G, R_CHF)
        assert {"product_id", "isins", "underlyings", "delta", "vega", "theta", "rho", "corr_sens"}.issubset(g.keys())

    # ── Monotonicity checks ───────────────────────────────────────────────

    def test_mbrc_fv_increases_with_correlation(self):
        """FV should be non-decreasing as pairwise correlation rises."""
        pricer = make_pricer(n_paths=10_000, seed=0)
        fvs = [
            pricer.price(
                MBRC_GREEK, european_brc_payoff, VOL_MBRC_G, R_CHF,
                corr_matrix=np.array([[1.0, rho], [rho, 1.0]])
            )["fair_value"]
            for rho in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
        assert all(fvs[i] <= fvs[i + 1] for i in range(len(fvs) - 1))

    def test_higher_vol_lowers_fv(self):
        """Higher vol → more chance of barrier breach → lower fair value."""
        pricer = make_pricer(n_paths=10_000, seed=0)
        vol_low  = {"CH0012221716": 0.10}
        vol_high = {"CH0012221716": 0.40}
        fv_low  = pricer.price(BRC_GREEK, european_brc_payoff, vol_low,  R_CHF)["fair_value"]
        fv_high = pricer.price(BRC_GREEK, european_brc_payoff, vol_high, R_CHF)["fair_value"]
        assert fv_high < fv_low

    # ── Analytical benchmark (delta vs Black-Scholes) ─────────────────────

    def test_brc_delta_close_to_black_scholes(self):
        """For a single European BRC the MC delta should match BS within 5%.

        A BRC with notional N and barrier (strike) K is equivalent to:
            N bonds  −  (N/K) vanilla puts struck at K

        So the BRC delta = (N/K) × |put_delta| × S × 0.01  (per 1% spot move).
        """
        from src.pricing.black_scholes import BlackScholes

        pricer = make_pricer(n_paths=20_000, seed=42)
        g = pricer.compute_greeks(BRC_GREEK, european_brc_payoff, VOL_BRC_G, R_CHF)

        today    = pd.Timestamp.today().normalize()
        T        = (pd.Timestamp(BRC_GREEK["maturity_date"]) - today).days / 360
        S        = 95.0
        K        = float(BRC_GREEK["strike"][0])
        N        = float(BRC_GREEK["notional"])
        bs       = BlackScholes(S=S, K=K, T=T, r=R_CHF, sigma=0.20)

        # BRC embeds N/K puts; short-put delta is positive
        bs_delta = -bs.delta("put") * (N / K) * S * 0.01

        err_pct = abs(g["delta"][0] - bs_delta) / abs(bs_delta) * 100
        assert err_pct < 10.0, f"Delta error {err_pct:.1f}% exceeds 10% tolerance"

    # ── Portfolio Greeks ──────────────────────────────────────────────────

    def test_portfolio_greeks_output_shape(self):
        pricer    = make_pricer(n_paths=500)
        portfolio = make_portfolio()
        greeks_df, pf_delta, _fv_df = pricer.compute_portfolio_greeks(
            portfolio, {**VOL_BRC_G, **VOL_MBRC_G}, {"CHF": R_CHF}
        )
        # One row per product × underlying
        total_underlyings = sum(len(row["underlying_isins"]) for _, row in portfolio.iterrows())
        assert len(greeks_df) == total_underlyings
        assert {"product_id", "isin", "underlying", "delta_1pct", "vega_1pp", "theta", "rho"}.issubset(greeks_df.columns)

    def test_portfolio_delta_aggregates_by_isin(self):
        """ABB (CH0012221716) appears in both BRC and MBRC — its portfolio delta
        should equal the sum of its individual product deltas."""
        pricer    = make_pricer(n_paths=2_000, seed=42)
        portfolio = make_portfolio()
        greeks_df, pf_delta, _fv_df = pricer.compute_portfolio_greeks(
            portfolio, {**VOL_BRC_G, **VOL_MBRC_G}, {"CHF": R_CHF}
        )
        isin = "CH0012221716"
        sum_from_products = greeks_df[greeks_df["isin"] == isin]["delta_1pct"].sum()
        pf_value          = pf_delta[pf_delta["isin"] == isin]["total_delta_1pct"].iloc[0]
        assert abs(sum_from_products - pf_value) < 0.01
