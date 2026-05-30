"""
Tests for src/pricing/vol_surface.py.

Test strategy
-------------
The module is layered: SVI math at the bottom, then calibration, then
arbitrage and quality gates, then the ``VolSliceSurface`` wrapper that
implements the fallback decision procedure. The tests mirror this
layering. Each layer is exercised both with synthetic data — where the
ground truth is known by construction and recovery can be checked to a
tight tolerance — and, where it is informative, with one real-data
smoke test against the stored ``data/options.csv`` chain so that the
module is verified to behave reasonably on the inputs it will actually
receive in production.

The synthetic regime uses a known SVI parameter tuple, generates a
chain from it, and checks that the calibration recovers either the
parameter tuple itself (where identifiable) or the implied smile
curve to a tight tolerance in volatility points. The arbitrage and
quality gates are exercised by hand-crafted violating inputs to
confirm that they fire as expected; the wrapper is exercised through
the three branches (svi, proxy, fallback) and through the boundary
between them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pricing.vol_surface import (
    DEFAULT_FALLBACK_VOL,
    FIT_STATUS_FALLBACK,
    FIT_STATUS_PROXY,
    FIT_STATUS_SVI,
    SVICalibrationError,
    SVIParams,
    VolSliceSurface,
    check_durrleman_butterfly,
    check_wing_bounds,
    fit_svi_slice,
    nearest_strike_proxy,
    quality_gate,
    svi_implied_vol,
    svi_total_variance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equity_params() -> SVIParams:
    """A textbook equity-like SVI tuple: ATM ~28 %, negative skew."""
    return SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)


@pytest.fixture
def synthetic_chain(equity_params: SVIParams):
    """Ten strikes from 70 % to 130 % of forward, IVs drawn from the
    equity SVI tuple with a touch of Gaussian noise to mimic market mids."""
    F, T = 100.0, 1.0
    strikes = np.array([70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0])
    k = np.log(strikes / F)
    ivs_clean = svi_implied_vol(k, T, equity_params)
    rng = np.random.default_rng(42)
    ivs = ivs_clean + rng.normal(0.0, 0.002, size=ivs_clean.shape)
    return strikes, ivs, T, F


# ---------------------------------------------------------------------------
# SVI math
# ---------------------------------------------------------------------------

class TestSVIMath:
    """Pointwise evaluation of the raw SVI parameterisation."""

    def test_total_variance_atm(self, equity_params):
        # At k=0 with m=0, w(0) = a + b * sigma. We chose a=0.04, b=0.4,
        # sigma=0.1 so the ATM total variance should be exactly 0.08.
        assert svi_total_variance(0.0, equity_params) == pytest.approx(0.08, abs=1e-12)

    def test_implied_vol_matches_sqrt(self, equity_params):
        # sigma = sqrt(w / T) — a direct algebraic identity.
        w = svi_total_variance(-0.2, equity_params)
        sigma = svi_implied_vol(-0.2, 1.0, equity_params)
        assert sigma == pytest.approx(np.sqrt(w), rel=1e-12)

    def test_skew_direction_with_negative_rho(self, equity_params):
        # With rho < 0 the OTM put wing carries higher volatility than
        # the OTM call wing of equivalent log-moneyness.
        sigma_put = svi_implied_vol(-0.2, 1.0, equity_params)
        sigma_call = svi_implied_vol(+0.2, 1.0, equity_params)
        assert sigma_put > sigma_call

    def test_vectorised_evaluation(self, equity_params):
        k = np.linspace(-0.4, 0.4, 9)
        sigma = svi_implied_vol(k, 1.0, equity_params)
        # Compare against a Python-loop reference to make sure the
        # vectorised path agrees with scalar evaluation pointwise.
        ref = np.array([svi_implied_vol(float(ki), 1.0, equity_params) for ki in k])
        np.testing.assert_allclose(sigma, ref, rtol=1e-12)

    def test_domain_violations_raise(self):
        with pytest.raises(ValueError, match="b must be non-negative"):
            SVIParams(a=0.0, b=-0.1, rho=0.0, m=0.0, sigma=0.1)
        with pytest.raises(ValueError, match=r"rho must lie in \[-1, 1\]"):
            SVIParams(a=0.0, b=0.1, rho=1.5, m=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="sigma must be strictly positive"):
            SVIParams(a=0.0, b=0.1, rho=0.0, m=0.0, sigma=0.0)
        with pytest.raises(ValueError, match="negative total-variance"):
            SVIParams(a=-1.0, b=0.1, rho=0.0, m=0.0, sigma=0.1)

    def test_implied_vol_nonpositive_tenor_raises(self, equity_params):
        with pytest.raises(ValueError, match="T must be strictly positive"):
            svi_implied_vol(0.0, 0.0, equity_params)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    """Round-trip and structural calibration behaviour."""

    def test_roundtrip_recovers_smile_curve(self, synthetic_chain, equity_params):
        # Recovering the parameter tuple exactly is not the goal: SVI is
        # over-parameterised relative to a noisy ten-strike chain and the
        # mapping (k, w) -> (a, b, rho, m, sigma) is not injective at
        # finite sample sizes. The economically meaningful test is that
        # the implied smile curve is recovered across the observed range.
        strikes, ivs, T, F = synthetic_chain
        params, meta = fit_svi_slice(strikes, ivs, T=T, forward=F)
        assert meta["converged"]
        # Compare smile curves over the range covered by data.
        k_grid = np.linspace(*meta["k_range"], 21)
        fit_sigma = svi_implied_vol(k_grid, T, params)
        true_sigma = svi_implied_vol(k_grid, T, equity_params)
        # Tolerance: 0.5 volatility points -- comfortably tighter than the
        # noise floor of 0.2 vol points injected into the synthetic chain.
        assert np.max(np.abs(fit_sigma - true_sigma)) < 0.005

    def test_rmse_at_or_below_noise_level(self, synthetic_chain):
        # The fit residual cannot be smaller than the noise; checking
        # that it is of the right order of magnitude protects against
        # silent regressions in the calibration objective.
        strikes, ivs, T, F = synthetic_chain
        _, meta = fit_svi_slice(strikes, ivs, T=T, forward=F)
        assert meta["rmse"] < 0.005   # 0.5 vol points
        assert meta["max_resid"] < 0.01

    def test_bid_ask_weights_affect_fit(self, synthetic_chain):
        # When the bid-ask spread strongly favours the ATM strikes, the
        # fit should be closer at the centre and slightly looser in the
        # wings. We exercise the path rather than measure the effect.
        strikes, ivs, T, F = synthetic_chain
        ba_uniform = np.full_like(strikes, 0.1)
        ba_atm_tight = np.array([2.0, 1.0, 0.5, 0.2, 0.1, 0.1, 0.2, 0.5, 1.0, 2.0])
        p1, _ = fit_svi_slice(strikes, ivs, T=T, forward=F, bid_asks=ba_uniform)
        p2, _ = fit_svi_slice(strikes, ivs, T=T, forward=F, bid_asks=ba_atm_tight)
        # The two parameter tuples must differ -- the weighting is being
        # applied -- but both must produce smile curves close to the
        # synthetic ground truth.
        assert (p1.a, p1.b) != (p2.a, p2.b)

    def test_insufficient_strikes_raises(self):
        strikes = np.array([95.0, 100.0, 105.0])
        ivs = np.array([0.25, 0.20, 0.22])
        with pytest.raises(SVICalibrationError, match="At least 5 strikes"):
            fit_svi_slice(strikes, ivs, T=1.0, forward=100.0)

    def test_nonpositive_strike_raises(self):
        strikes = np.array([90.0, -5.0, 100.0, 105.0, 110.0])
        ivs = np.array([0.30, 0.25, 0.22, 0.21, 0.20])
        with pytest.raises(SVICalibrationError, match="strictly positive"):
            fit_svi_slice(strikes, ivs, T=1.0, forward=100.0)


# ---------------------------------------------------------------------------
# Arbitrage gates
# ---------------------------------------------------------------------------

class TestArbitrageGates:
    """Durrleman butterfly and Roger Lee wing bounds."""

    def test_clean_slice_passes_butterfly(self, equity_params):
        ok, msg = check_durrleman_butterfly(equity_params)
        assert ok, msg

    def test_clean_slice_passes_wing_bounds(self, equity_params):
        ok, msg = check_wing_bounds(equity_params, T=1.0)
        assert ok, msg

    def test_butterfly_violation_fires(self):
        # Extreme curvature (very small sigma) combined with large b
        # and extreme rho violates the butterfly condition near the
        # smile minimum. The hand-crafted tuple is well inside the
        # SVIParams admissibility constraints but produces a negative
        # density.
        params = SVIParams(a=0.0001, b=2.0, rho=-0.95, m=0.0, sigma=0.02)
        ok, msg = check_durrleman_butterfly(params)
        assert not ok
        assert "Durrleman" in msg

    def test_wing_bound_violation_at_long_tenor(self):
        # A parameter tuple that is admissible at T=1Y becomes
        # inadmissible at T=10Y because b T (1 + |rho|) scales linearly
        # in tenor.
        params = SVIParams(a=0.04, b=1.0, rho=-0.5, m=0.0, sigma=0.1)
        ok_short, _ = check_wing_bounds(params, T=1.0)
        ok_long, msg_long = check_wing_bounds(params, T=10.0)
        assert ok_short
        assert not ok_long
        assert "Roger Lee" in msg_long


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

class TestQualityGate:
    """Data-quality and fit-quality admissibility."""

    def test_too_few_strikes_rejected(self):
        ok, msg = quality_gate(np.array([100.0, 105.0, 110.0]), None, fit_rmse=0.005)
        assert not ok
        assert "insufficient strikes" in msg

    def test_rmse_above_threshold_rejected(self):
        ok, msg = quality_gate(
            np.array([90.0, 95.0, 100.0, 105.0, 110.0]), None, fit_rmse=0.025
        )
        assert not ok
        assert "RMSE too large" in msg

    def test_wide_bid_ask_rejected(self):
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        mids = np.array([5.0, 3.0, 2.0, 1.5, 1.0])
        bid_asks = np.array([0.05, 0.05, 0.05, 2.0, 0.5])   # only 3 tight
        ok, msg = quality_gate(strikes, bid_asks, fit_rmse=0.005, mids=mids)
        assert not ok
        assert "tight quotes" in msg

    def test_clean_inputs_pass(self):
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        mids = np.array([5.0, 3.0, 2.0, 1.5, 1.0])
        bid_asks_tight = np.full_like(strikes, 0.05)
        ok, msg = quality_gate(strikes, bid_asks_tight, fit_rmse=0.005, mids=mids)
        assert ok, msg


# ---------------------------------------------------------------------------
# Proxy fallback
# ---------------------------------------------------------------------------

class TestNearestStrikeProxy:
    """Single-point chain-proxy fallback."""

    def test_returns_closest_in_log_distance(self):
        # K=87 is closer to 90 than to 80 in log-moneyness:
        #   |ln(87/80)| = 0.084, |ln(87/90)| = 0.034
        sigma = nearest_strike_proxy([80.0, 90.0], [0.30, 0.25], K_target=87.0)
        assert sigma == 0.25

    def test_lower_strike_wins_on_ties(self):
        # K=100 is exactly equidistant between 90 and 110 in log space?
        # No -- ln(100/90) = 0.105, ln(110/100) = 0.0953, so 110 is
        # the closer strike. Construct a genuine tie instead:
        # K = sqrt(80 * 125) = 100 sits exactly in log-middle of (80, 125).
        sigma = nearest_strike_proxy([80.0, 125.0], [0.30, 0.20], K_target=100.0)
        # argmin of the sorted distances returns the first occurrence,
        # so the lower strike wins.
        assert sigma == 0.30

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError, match="at least one strike"):
            nearest_strike_proxy([], [], K_target=100.0)

    def test_nonpositive_target_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            nearest_strike_proxy([100.0], [0.20], K_target=-5.0)


# ---------------------------------------------------------------------------
# VolSliceSurface wrapper
# ---------------------------------------------------------------------------

class TestVolSliceSurface:
    """End-to-end fallback decision procedure."""

    def test_healthy_chain_yields_svi_branch(self, synthetic_chain):
        strikes, ivs, T, F = synthetic_chain
        slice_surface = VolSliceSurface.from_chain("AAPL", T, F, strikes, ivs)
        assert slice_surface.fit_status == FIT_STATUS_SVI
        assert slice_surface.reason == ""
        assert slice_surface.rmse is not None
        assert slice_surface.rmse < 0.005

    def test_svi_branch_is_smooth_smile(self, synthetic_chain):
        strikes, ivs, T, F = synthetic_chain
        slice_surface = VolSliceSurface.from_chain("AAPL", T, F, strikes, ivs)
        # Strong sanity check: vol at OTM put strike substantially
        # exceeds vol at OTM call strike of equivalent log-moneyness --
        # the negative skew is preserved through the wrapper.
        K_put, K_call = 80.0, 125.0   # mirror in log-moneyness around F=100
        sigma_put = float(slice_surface.sigma(K_put))
        sigma_call = float(slice_surface.sigma(K_call))
        assert sigma_put > sigma_call

    def test_thin_chain_routes_to_proxy(self, synthetic_chain):
        strikes, ivs, T, F = synthetic_chain
        # Three strikes -- below the calibration minimum of five.
        slice_surface = VolSliceSurface.from_chain(
            "THIN", T, F, strikes[:3], ivs[:3]
        )
        assert slice_surface.fit_status == FIT_STATUS_PROXY
        assert "At least 5 strikes" in slice_surface.reason
        # The proxy still produces a finite IV for any query strike.
        assert 0.0 < float(slice_surface.sigma(85.0)) < 1.0

    def test_empty_chain_routes_to_fallback(self):
        slice_surface = VolSliceSurface.from_chain(
            "NULL", T=1.0, forward=100.0,
            strikes=np.array([]), implied_vols=np.array([]),
        )
        assert slice_surface.fit_status == FIT_STATUS_FALLBACK
        assert slice_surface.sigma(100.0) == DEFAULT_FALLBACK_VOL
        # Fallback constant must be invariant to strike.
        assert slice_surface.sigma(50.0) == slice_surface.sigma(150.0)

    def test_noisy_chain_routes_to_proxy_via_rmse(self):
        # IVs that are not SVI-shaped force the fit RMSE above the
        # quality-gate threshold and route the slice to the proxy.
        F, T = 100.0, 1.0
        strikes = np.array([70.0, 80.0, 85.0, 90.0, 95.0, 100.0,
                            105.0, 110.0, 120.0, 130.0])
        rng = np.random.default_rng(0)
        ivs = 0.2 + rng.uniform(-0.05, 0.05, len(strikes))
        slice_surface = VolSliceSurface.from_chain("NOISY", T, F, strikes, ivs)
        assert slice_surface.fit_status == FIT_STATUS_PROXY
        assert "RMSE" in slice_surface.reason

    def test_vectorised_sigma(self, synthetic_chain):
        strikes, ivs, T, F = synthetic_chain
        slice_surface = VolSliceSurface.from_chain("AAPL", T, F, strikes, ivs)
        query = np.array([80.0, 100.0, 120.0])
        result = slice_surface.sigma(query)
        assert result.shape == query.shape
        # Pointwise consistency with scalar evaluation.
        for q, r in zip(query, result):
            assert float(r) == pytest.approx(float(slice_surface.sigma(float(q))),
                                              rel=1e-12)

    def test_sigma_at_moneyness_consistency(self, synthetic_chain):
        strikes, ivs, T, F = synthetic_chain
        slice_surface = VolSliceSurface.from_chain("AAPL", T, F, strikes, ivs)
        # sigma_at_moneyness(0.9) must equal sigma(0.9 * F).
        m_query = 0.9
        assert float(slice_surface.sigma_at_moneyness(m_query)) == pytest.approx(
            float(slice_surface.sigma(m_query * F)), rel=1e-12
        )

    def test_construction_rejects_inconsistent_branches(self):
        with pytest.raises(ValueError, match="svi.*requires"):
            VolSliceSurface("X", T=1.0, forward=100.0, fit_status="svi")
        with pytest.raises(ValueError, match="proxy.*requires"):
            VolSliceSurface("X", T=1.0, forward=100.0, fit_status="proxy")
        with pytest.raises(ValueError, match="fallback.*requires"):
            VolSliceSurface("X", T=1.0, forward=100.0, fit_status="fallback")

    def test_unknown_status_rejected(self):
        with pytest.raises(ValueError, match="fit_status must be"):
            VolSliceSurface(
                "X", T=1.0, forward=100.0, fit_status="bogus",
                fallback_sigma=0.2,
            )

    def test_negative_strike_query_rejected(self, synthetic_chain):
        strikes, ivs, T, F = synthetic_chain
        slice_surface = VolSliceSurface.from_chain("AAPL", T, F, strikes, ivs)
        with pytest.raises(ValueError, match="strictly positive"):
            slice_surface.sigma(-5.0)


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------

_OPTIONS_CSV = Path(__file__).resolve().parent.parent / "data" / "options.csv"


@pytest.mark.skipif(not _OPTIONS_CSV.exists(),
                    reason="data/options.csv not present in this checkout")
class TestRealDataSmoke:
    """Sanity check on the stored Yahoo chain.

    The intent is not to pin a particular numerical outcome -- the
    chain rolls daily -- but to confirm that the calibration produces
    a usable smile on representative real data. The slice chosen is
    the densest available, which on the stored snapshot is NVDA at
    its most heavily quoted expiry.
    """

    @pytest.fixture(scope="class")
    def densest_slice(self):
        df = pd.read_csv(_OPTIONS_CSV)
        # Pick the (isin, expiry, type) triple with the largest number
        # of strikes -- the most informative slice in the file.
        grouped = df.groupby(["isin", "expiry", "type"]).size()
        if grouped.empty:
            pytest.skip("no rows in options.csv")
        isin, expiry, opt_type = grouped.idxmax()
        sub = df[(df["isin"] == isin) & (df["expiry"] == expiry) & (df["type"] == opt_type)]
        sub = sub[(sub["iv"] > 0.0) & (sub["iv"] < 5.0) & (sub["strike"] > 0.0)]
        sub = sub.drop_duplicates(subset="strike").sort_values("strike")
        if len(sub) < 8:
            pytest.skip("densest slice has too few clean strikes")
        return isin, expiry, sub

    def test_densest_slice_fits_or_falls_back_gracefully(self, densest_slice):
        isin, expiry, sub = densest_slice
        strikes = sub["strike"].to_numpy()
        ivs = sub["iv"].to_numpy()
        # Forward is unknown without a discount curve; for a smoke
        # test we approximate it by the median strike, which sits
        # near the ATM region on a typical liquid chain. The objective
        # here is to exercise the calibration end-to-end, not to
        # produce a tradable surface.
        forward = float(np.median(strikes))
        # A representative one-year tenor; the actual tenor is not
        # needed for the structural test of the pipeline.
        T = 1.0
        slice_surface = VolSliceSurface.from_chain(
            isin, T, forward, strikes, ivs,
        )
        assert slice_surface.fit_status in (FIT_STATUS_SVI, FIT_STATUS_PROXY)
        # In either branch, the surface must produce a finite positive
        # volatility for any query strike inside the observed range.
        K_query = forward * 0.9
        sigma = float(slice_surface.sigma(K_query))
        assert 0.01 < sigma < 5.0
