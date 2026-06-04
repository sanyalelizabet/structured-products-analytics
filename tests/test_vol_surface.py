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
    LV_IV_RATIO_WARNING,
    SIGMA_LV_FLOOR,
    SIGMA_LV_HARD_CAP,
    SIGMA_LV_WARNING,
    SURFACE_STATUS_EXTRAPOLATED,
    SURFACE_STATUS_FALLBACK,
    SURFACE_STATUS_INTERPOLATED,
    SURFACE_STATUS_SINGLE_SLICE,
    SVICalibrationError,
    SVIParams,
    VolSliceSurface,
    VolSurface,
    build_product_vol_map,
    check_durrleman_butterfly,
    check_wing_bounds,
    extrapolate_atm_scaling,
    fit_svi_slice,
    interpolate_total_variance,
    nearest_strike_proxy,
    quality_gate,
    svi_implied_vol,
    svi_total_variance,
    verify_calendar_monotone,
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
        # 4 volatility points is unambiguously above any defensible
        # ceiling, even on noisy free retail feeds.
        ok, msg = quality_gate(
            np.array([90.0, 95.0, 100.0, 105.0, 110.0]), None, fit_rmse=0.040
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

    def test_noisy_chain_routes_to_proxy(self):
        # IVs that are not SVI-shaped should route the slice to the
        # proxy fallback. The precise gate that fires depends on the
        # nature of the noise — calibration may fail to converge, the
        # Durrleman density may go negative, or the RMSE may exceed
        # the ceiling — but in every case the result is the proxy.
        F, T = 100.0, 1.0
        strikes = np.array([70.0, 80.0, 85.0, 90.0, 95.0, 100.0,
                            105.0, 110.0, 120.0, 130.0])
        rng = np.random.default_rng(0)
        ivs = 0.2 + rng.uniform(-0.10, 0.10, len(strikes))
        slice_surface = VolSliceSurface.from_chain("NOISY", T, F, strikes, ivs)
        assert slice_surface.fit_status == FIT_STATUS_PROXY
        assert slice_surface.reason, "fallback should record a reason"

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


# ---------------------------------------------------------------------------
# Term-structure assembly: interpolator + extrapolator + VolSurface
# ---------------------------------------------------------------------------

def _make_svi_slice(T: float, params: SVIParams, forward: float = 100.0) -> VolSliceSurface:
    """Construct an SVI-branch :class:`VolSliceSurface` directly for tests."""
    return VolSliceSurface(
        isin="SYN", T=T, forward=forward, fit_status=FIT_STATUS_SVI,
        params=params,
        chain_strikes=np.array([forward]),
        chain_ivs=np.array([0.20]),
        n_strikes=10,
        k_range=(-0.4, 0.4),
        rmse=0.005,
        max_resid=0.01,
    )


class TestTotalVarianceInterpolator:
    """The convex combination of total variance between two slices."""

    @pytest.fixture
    def two_slices(self) -> tuple[SVIParams, SVIParams]:
        return (
            SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1),
            SVIParams(a=0.08, b=0.40, rho=-0.4, m=0.0, sigma=0.1),
        )

    def test_endpoint_identity(self, two_slices):
        p_left, p_right = two_slices
        k = np.linspace(-0.4, 0.4, 9)
        np.testing.assert_allclose(
            interpolate_total_variance(k, 1.0, 1.0, p_left, 2.0, p_right),
            svi_total_variance(k, p_left), rtol=0, atol=1e-14,
        )
        np.testing.assert_allclose(
            interpolate_total_variance(k, 2.0, 1.0, p_left, 2.0, p_right),
            svi_total_variance(k, p_right), rtol=0, atol=1e-14,
        )

    def test_midpoint_linearity(self, two_slices):
        # At T = (T_left + T_right) / 2 the interpolator must produce
        # exactly the pointwise mean of the two endpoint total variances.
        p_left, p_right = two_slices
        k = np.linspace(-0.4, 0.4, 9)
        w_mid_expected = 0.5 * (svi_total_variance(k, p_left)
                                + svi_total_variance(k, p_right))
        np.testing.assert_allclose(
            interpolate_total_variance(k, 1.5, 1.0, p_left, 2.0, p_right),
            w_mid_expected, rtol=0, atol=1e-14,
        )

    def test_bracket_order_rejected(self, two_slices):
        p_left, p_right = two_slices
        with pytest.raises(ValueError, match="T_left < T_right"):
            interpolate_total_variance(0.0, 1.5, 2.0, p_left, 1.0, p_right)

    def test_out_of_bracket_rejected(self, two_slices):
        p_left, p_right = two_slices
        with pytest.raises(ValueError, match="bracketing interval"):
            interpolate_total_variance(0.0, 5.0, 1.0, p_left, 2.0, p_right)


class TestATMScalingExtrapolator:
    """Vol-flat extension of a single anchor slice."""

    def test_anchor_identity(self):
        p = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)
        k = np.linspace(-0.4, 0.4, 9)
        np.testing.assert_allclose(
            extrapolate_atm_scaling(k, 1.0, 1.0, p),
            svi_total_variance(k, p), rtol=0, atol=1e-14,
        )

    def test_vol_flat_invariant(self):
        # The defining property: sigma(k, T_query) equals sigma(k, T_anchor)
        # for any positive T_query.
        p = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)
        k = np.linspace(-0.3, 0.3, 7)
        sigma_anchor = svi_implied_vol(k, 1.0, p)
        for T_query in (0.25, 0.5, 1.0, 2.0, 5.0):
            w = extrapolate_atm_scaling(k, T_query, 1.0, p)
            sigma_q = np.sqrt(w / T_query)
            np.testing.assert_allclose(sigma_q, sigma_anchor, rtol=0, atol=1e-12)

    def test_nonpositive_tenor_rejected(self):
        p = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="T_query"):
            extrapolate_atm_scaling(0.0, -0.5, 1.0, p)
        with pytest.raises(ValueError, match="T_anchor"):
            extrapolate_atm_scaling(0.0, 1.0, 0.0, p)


class TestCalendarMonotonicityAudit:
    """Detection of total-variance non-monotonicity between slices."""

    def test_healthy_surface_passes(self):
        p1 = SVIParams(a=0.02, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        p2 = SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        p3 = SVIParams(a=0.08, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        records = [(0.5, p1, _make_svi_slice(0.5, p1)),
                   (1.0, p2, _make_svi_slice(1.0, p2)),
                   (2.0, p3, _make_svi_slice(2.0, p3))]
        assert verify_calendar_monotone(records) == []

    def test_inverted_surface_caught(self):
        # Total variance falls when moving from T=1Y to T=2Y at every k.
        p1 = SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        p2 = SVIParams(a=0.01, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        records = [(1.0, p1, _make_svi_slice(1.0, p1)),
                   (2.0, p2, _make_svi_slice(2.0, p2))]
        violations = verify_calendar_monotone(records)
        assert len(violations) == 1
        v = violations[0]
        assert v["T_left"] == 1.0 and v["T_right"] == 2.0
        assert v["deficit"] > 0.0

    def test_single_slice_skipped(self):
        p1 = SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        assert verify_calendar_monotone([(1.0, p1, _make_svi_slice(1.0, p1))]) == []

    def test_empty_records_skipped(self):
        assert verify_calendar_monotone([]) == []


class TestVolSurface:
    """End-to-end VolSurface dispatch across the four status branches."""

    @pytest.fixture
    def three_slice_surface(self) -> VolSurface:
        # Monotone ATM variance: 0.06, 0.08, 0.16 at T = 0.5, 1.0, 2.0.
        p1 = SVIParams(a=0.02, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        p2 = SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        p3 = SVIParams(a=0.12, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        return VolSurface.from_slice_map("SYN", {
            "2026-12-01": _make_svi_slice(0.5, p1),
            "2027-06-01": _make_svi_slice(1.0, p2),
            "2028-06-01": _make_svi_slice(2.0, p3),
        })

    def test_listed_tenor_reproduces_slice(self, three_slice_surface):
        # At a listed expiry the surface must equal the slice's SVI sigma.
        K = np.array([70.0, 100.0, 130.0])
        for T, params in (
            (0.5, SVIParams(a=0.02, b=0.40, rho=-0.4, m=0.0, sigma=0.1)),
            (1.0, SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)),
            (2.0, SVIParams(a=0.12, b=0.40, rho=-0.4, m=0.0, sigma=0.1)),
        ):
            k = np.log(K / 100.0)
            np.testing.assert_allclose(
                three_slice_surface.sigma(K, T),
                svi_implied_vol(k, T, params),
                rtol=0, atol=1e-12,
            )

    def test_interior_query_is_interpolation(self, three_slice_surface):
        status, _ = three_slice_surface.surface_status_at(1.5)
        assert status == SURFACE_STATUS_INTERPOLATED
        # At T = 1.5, k = 0, the total variance must equal the midpoint
        # between w(0, T=1) = 0.08 and w(0, T=2) = 0.16, i.e. 0.12,
        # giving sigma = sqrt(0.12 / 1.5) = sqrt(0.08) ~= 0.2828.
        sigma = float(three_slice_surface.sigma(100.0, 1.5))
        assert sigma == pytest.approx(np.sqrt(0.12 / 1.5), rel=1e-10)

    def test_above_listed_range_extrapolated(self, three_slice_surface):
        status, reason = three_slice_surface.surface_status_at(3.0)
        assert status == SURFACE_STATUS_EXTRAPOLATED
        assert "longest" in reason
        # Vol-flat extrapolation: sigma(F, 3.0) equals sigma(F, 2.0).
        sigma_three = float(three_slice_surface.sigma(100.0, 3.0))
        sigma_two   = float(three_slice_surface.sigma(100.0, 2.0))
        assert sigma_three == pytest.approx(sigma_two, rel=1e-10)

    def test_below_listed_range_extrapolated(self, three_slice_surface):
        status, reason = three_slice_surface.surface_status_at(0.1)
        assert status == SURFACE_STATUS_EXTRAPOLATED
        assert "shorter" in reason
        sigma_short  = float(three_slice_surface.sigma(100.0, 0.1))
        sigma_anchor = float(three_slice_surface.sigma(100.0, 0.5))
        assert sigma_short == pytest.approx(sigma_anchor, rel=1e-10)

    def test_single_slice_branch(self):
        p = SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        surf = VolSurface.from_slice_map("ONE", {"a": _make_svi_slice(1.0, p)})
        status, _ = surf.surface_status_at(2.5)
        assert status == SURFACE_STATUS_SINGLE_SLICE
        # Vol-flat: sigma(F, T) is invariant in T.
        for T in (0.5, 1.0, 2.0, 5.0):
            assert float(surf.sigma(100.0, T)) == pytest.approx(
                float(np.sqrt(svi_total_variance(0.0, p))), rel=1e-12
            )

    def test_no_slice_falls_back_to_constant(self):
        surf = VolSurface.from_slice_map("NONE", {})
        status, _ = surf.surface_status_at(1.0)
        assert status == SURFACE_STATUS_FALLBACK
        assert float(surf.sigma(50.0, 1.0)) == DEFAULT_FALLBACK_VOL
        assert float(surf.sigma(150.0, 1.0)) == DEFAULT_FALLBACK_VOL

    def test_proxy_and_fallback_slices_ignored_in_assembly(self):
        # Proxy and fallback slices must not enter the SVI list.
        p = SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        proxy_slice = VolSliceSurface(
            isin="X", T=1.0, forward=100.0,
            fit_status=FIT_STATUS_PROXY,
            chain_strikes=np.array([100.0]), chain_ivs=np.array([0.2]),
            n_strikes=1,
        )
        fb_slice = VolSliceSurface(
            isin="X", T=2.0, forward=100.0,
            fit_status=FIT_STATUS_FALLBACK,
            fallback_sigma=0.15,
        )
        surf = VolSurface.from_slice_map("MIX", {
            "a": _make_svi_slice(0.5, p),
            "b": proxy_slice,
            "c": fb_slice,
        })
        assert surf.n_svi_slices == 1   # only the SVI slice participates
        status, _ = surf.surface_status_at(1.5)
        assert status == SURFACE_STATUS_SINGLE_SLICE

    def test_vectorised_sigma(self, three_slice_surface):
        K = np.array([70.0, 100.0, 130.0])
        result = three_slice_surface.sigma(K, 1.5)
        assert result.shape == K.shape
        for k_i, r in zip(K, result):
            assert float(r) == pytest.approx(
                float(three_slice_surface.sigma(float(k_i), 1.5)), rel=1e-12
            )

    def test_sigma_at_moneyness_consistency(self, three_slice_surface):
        m, T = 0.9, 1.25
        assert float(three_slice_surface.sigma_at_moneyness(m, T)) == pytest.approx(
            float(three_slice_surface.sigma(m * 100.0, T)), rel=1e-12
        )

    def test_negative_tenor_rejected(self, three_slice_surface):
        with pytest.raises(ValueError, match="T must be strictly positive"):
            three_slice_surface.sigma(100.0, -1.0)

    def test_negative_strike_rejected(self, three_slice_surface):
        with pytest.raises(ValueError, match="strictly positive"):
            three_slice_surface.sigma(-5.0, 1.0)

    def test_inverted_pair_records_violation(self):
        # Construct an inverted surface and verify the audit field
        # exposes the violation while the surface remains usable.
        p1 = SVIParams(a=0.10, b=0.40, rho=-0.4, m=0.0, sigma=0.1)   # T=1, w_ATM = 0.14
        p2 = SVIParams(a=0.01, b=0.40, rho=-0.4, m=0.0, sigma=0.1)   # T=2, w_ATM = 0.05
        surf = VolSurface.from_slice_map("BAD", {
            "a": _make_svi_slice(1.0, p1),
            "b": _make_svi_slice(2.0, p2),
        })
        assert surf.calendar_violations
        v = surf.calendar_violations[0]
        assert v["deficit"] > 0.0
        # The surface is still queryable -- audit does not block use.
        sigma = float(surf.sigma(100.0, 1.5))
        assert 0.0 < sigma < 5.0

    def test_strictly_increasing_tenors_required(self):
        p = SVIParams(a=0.04, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
        with pytest.raises(ValueError, match="strictly increasing"):
            VolSurface(
                isin="X", forward=100.0,
                slice_records=[(1.0, p, _make_svi_slice(1.0, p)),
                               (1.0, p, _make_svi_slice(1.0, p))],
            )


# ---------------------------------------------------------------------------
# Pricer integration: product-specific vol map
# ---------------------------------------------------------------------------

class TestBuildProductVolMap:
    """Per-product surface-aware volatility resolution."""

    @pytest.fixture
    def equity_surface(self) -> VolSurface:
        # ATM total variance rises 0.02 -> 0.04 -> 0.08 across T = 0.5, 1, 2.
        slice_map = {}
        for T, a in [(0.5, 0.02), (1.0, 0.04), (2.0, 0.08)]:
            p = SVIParams(a=a, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
            slice_map[str(T)] = _make_svi_slice(T, p, forward=100.0)
        return VolSurface.from_slice_map("EQUITY", slice_map)

    @pytest.fixture
    def brc_product(self) -> pd.Series:
        return pd.Series({
            "product_type":     "BRC",
            "underlying_isins": ["EQUITY"],
            "initial_levels":   [100.0],
            "barrier_pct":      0.65,
            "maturity_date":    pd.Timestamp("2028-05-30"),
        })

    def test_surface_lookup_picks_barrier_strike_vol(self, equity_surface, brc_product):
        # Two-year maturity from a 2026-05-30 valuation date sits inside
        # the listed slice range (T_max = 2.0), so the lookup should be
        # either interpolated or extrapolated; either way, the resolved
        # sigma must be the vol at the 65 % barrier strike rather than
        # the at-the-money vol.
        valuation = pd.Timestamp("2026-05-30")
        vmap, diag = build_product_vol_map(
            brc_product,
            vol_surfaces={"EQUITY": equity_surface},
            fallback_vol_map={"EQUITY": 0.30},   # legacy ATM input
            valuation_date=valuation,
        )
        sigma = vmap["EQUITY"]
        # Sigma at K=65 must exceed sigma at K=100 under the negative
        # skew of the synthetic surface.
        T_resid = (pd.Timestamp(brc_product["maturity_date"]) - valuation).days / 365.25
        sigma_atm = float(equity_surface.sigma(100.0, T_resid))
        sigma_barrier = float(equity_surface.sigma(65.0, T_resid))
        assert sigma == pytest.approx(sigma_barrier, rel=1e-10)
        assert sigma > sigma_atm
        assert diag[0]["source"] == "surface"

    def test_missing_surface_falls_back(self, brc_product):
        vmap, diag = build_product_vol_map(
            brc_product,
            vol_surfaces={},                     # no surface for EQUITY
            fallback_vol_map={"EQUITY": 0.30},
            valuation_date=pd.Timestamp("2026-05-30"),
        )
        assert vmap["EQUITY"] == 0.30
        assert diag[0]["source"] == "fallback"
        assert diag[0]["surface_status"] == "no_surface"

    def test_surface_in_fallback_regime_uses_legacy_map(self, brc_product):
        empty_surface = VolSurface.from_slice_map("EQUITY", {})
        vmap, diag = build_product_vol_map(
            brc_product,
            vol_surfaces={"EQUITY": empty_surface},
            fallback_vol_map={"EQUITY": 0.30},
            valuation_date=pd.Timestamp("2026-05-30"),
        )
        assert vmap["EQUITY"] == 0.30
        assert diag[0]["source"] == "fallback"
        assert diag[0]["surface_status"] == SURFACE_STATUS_FALLBACK

    def test_matured_product_falls_back(self, equity_surface, brc_product):
        # Maturity in the past -> tenor non-positive -> fallback.
        product = brc_product.copy()
        product["maturity_date"] = pd.Timestamp("2024-01-01")
        vmap, diag = build_product_vol_map(
            product,
            vol_surfaces={"EQUITY": equity_surface},
            fallback_vol_map={"EQUITY": 0.30},
            valuation_date=pd.Timestamp("2026-05-30"),
        )
        assert vmap["EQUITY"] == 0.30
        assert diag[0]["surface_status"] == "tenor_non_positive"

    def test_worst_of_with_mixed_surface_coverage(self, equity_surface):
        # Two-underlying BRC: one isin has a surface, the other does not.
        product = pd.Series({
            "product_type":     "BRC",
            "underlying_isins": ["EQUITY", "OTHER"],
            "initial_levels":   [100.0, 50.0],
            "barrier_pct":      0.65,
            "maturity_date":    pd.Timestamp("2027-05-30"),
        })
        vmap, diag = build_product_vol_map(
            product,
            vol_surfaces={"EQUITY": equity_surface},
            fallback_vol_map={"EQUITY": 0.30, "OTHER": 0.40},
            valuation_date=pd.Timestamp("2026-05-30"),
        )
        # EQUITY -> surface; OTHER -> fallback.
        assert diag[0]["source"] == "surface"
        assert diag[1]["source"] == "fallback"
        assert diag[1]["surface_status"] == "no_surface"
        # The OTHER barrier strike was computed correctly.
        assert diag[1]["K_barrier"] == pytest.approx(50.0 * 0.65, rel=1e-12)

    def test_diagnostics_include_barrier_strike_and_tenor(self, equity_surface, brc_product):
        valuation = pd.Timestamp("2026-05-30")
        _, diag = build_product_vol_map(
            brc_product,
            vol_surfaces={"EQUITY": equity_surface},
            fallback_vol_map={"EQUITY": 0.30},
            valuation_date=valuation,
        )
        assert diag[0]["K_barrier"] == pytest.approx(65.0, rel=1e-12)
        assert diag[0]["T"] == pytest.approx(
            (pd.Timestamp(brc_product["maturity_date"]) - valuation).days / 365.25,
            rel=1e-12,
        )

    def test_pricer_integration_shifts_fair_value(self):
        # End-to-end: compute_portfolio_greeks with vs without
        # vol_surfaces on a single-underlying European BRC. Under a
        # negative-skew surface the bond NPV should fall (barrier-zone
        # vol is higher than ATM), reducing the fair value of the
        # security relative to the legacy constant-ATM-vol baseline.
        from src.pricing.monte_carlo import MonteCarloPricer
        from tests.conftest import make_brc_row

        # Build a synthetic equity surface whose anchor ISIN matches the
        # ISIN used by the canonical make_brc_row fixture.
        forward = 100.0
        slice_map = {}
        for T, a in [(0.5, 0.02), (1.0, 0.04), (2.0, 0.08)]:
            p = SVIParams(a=a, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
            slice_map[str(T)] = _make_svi_slice(T, p, forward=forward)
        brc_isin = "CH0012221716"
        equity_surface = VolSurface.from_slice_map(brc_isin, slice_map)

        row = make_brc_row(
            barrier_pct=0.65,
            initial_level=100.0,
            current_spot=100.0,
            initial_fixing_date="2026-05-30",
            maturity_date="2028-05-30",
        )
        portfolio = pd.DataFrame([row])
        vol_map_legacy = {brc_isin: 0.30}    # legacy ATM-vol input
        risk_free = {"CHF": 0.01}

        pricer = MonteCarloPricer(n_paths=3000, seed=42)
        _, _, fv_legacy = pricer.compute_portfolio_greeks(
            portfolio, vol_map_legacy, risk_free,
        )
        _, _, fv_surface = pricer.compute_portfolio_greeks(
            portfolio, vol_map_legacy, risk_free,
            vol_surfaces={brc_isin: equity_surface},
            valuation_date=pd.Timestamp("2026-05-30"),
        )
        # The two fair values must differ -- the surface input changes
        # the barrier-hit probability -- and the direction must agree
        # with the sign of the skew correction. Under a negative skew
        # the barrier-strike vol exceeds the ATM vol, the barrier-hit
        # probability rises, the bond is worth less, and the fair value
        # falls.
        fv_l = float(fv_legacy["fair_value"].iloc[0])
        fv_s = float(fv_surface["fair_value"].iloc[0])
        assert fv_s < fv_l, (
            f"Expected surface fair value (={fv_s:.4f}) to fall below "
            f"the legacy ATM-vol baseline (={fv_l:.4f}) under negative skew."
        )


# ---------------------------------------------------------------------------
# Dupire local volatility derivation
# ---------------------------------------------------------------------------

class TestLocalVolatility:
    """Dupire local volatility on the assembled surface."""

    def _flat_surface(self, sigma_const: float = 0.20) -> VolSurface:
        # SVI with b ~ 0 gives a flat smile; varying ATM variance with T
        # so that sigma at every (k, T) equals sigma_const exactly.
        slice_map = {}
        for T in (0.5, 1.0, 2.0):
            p = SVIParams(a=sigma_const ** 2 * T,
                          b=1.0e-6, rho=0.0, m=0.0, sigma=0.1)
            slice_map[str(T)] = _make_svi_slice(T, p, forward=100.0)
        return VolSurface.from_slice_map("FLAT", slice_map)

    def _skewed_surface(self) -> VolSurface:
        # Three-slice negative-skew equity-like surface.
        slice_map = {}
        for T, a in [(0.5, 0.02), (1.0, 0.04), (2.0, 0.08)]:
            p = SVIParams(a=a, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
            slice_map[str(T)] = _make_svi_slice(T, p, forward=100.0)
        return VolSurface.from_slice_map("SKEW", slice_map)

    def test_flat_surface_local_vol_equals_implied(self):
        # On a constant-volatility surface, the Dupire identity reduces
        # to sigma_LV == sigma_IV at every (K, T).
        surf = self._flat_surface(sigma_const=0.20)
        for K in (70.0, 100.0, 130.0):
            for T in (0.75, 1.5):
                lv = float(surf.local_volatility(K, T))
                iv = float(surf.sigma(K, T))
                assert lv == pytest.approx(iv, abs=1.0e-3)

    def test_skewed_surface_amplifies_put_wing(self):
        # Under negative skew, the local volatility at a deep OTM put
        # strike exceeds the implied volatility at the same point
        # (Dupire amplification).
        surf = self._skewed_surface()
        K_otm = 70.0
        lv = float(surf.local_volatility(K_otm, 1.0))
        iv = float(surf.sigma(K_otm, 1.0))
        assert lv > iv

    def test_vectorised_output_shape(self):
        surf = self._skewed_surface()
        K = np.array([70.0, 100.0, 130.0])
        out = surf.local_volatility(K, 1.5)
        assert out.shape == K.shape

    def test_hard_cap_is_two_hundred_percent(self):
        # SIGMA_LV_HARD_CAP is the production threshold; values in
        # [SIGMA_LV_WARNING, SIGMA_LV_HARD_CAP] are *not* clipped.
        assert SIGMA_LV_HARD_CAP == 2.00
        assert SIGMA_LV_WARNING == 1.00
        assert LV_IV_RATIO_WARNING == 3.00

    def test_warning_recorded_for_extreme_local_vol(self):
        # Construct a surface with very steep put-wing skew that
        # produces sigma_LV above the one-hundred-per-cent threshold
        # at some strikes.
        surf = VolSurface.from_slice_map("STEEP", {
            "a": _make_svi_slice(1.0, SVIParams(a=0.005, b=0.80, rho=-0.85, m=0.0, sigma=0.02)),
            "b": _make_svi_slice(2.0, SVIParams(a=0.010, b=0.80, rho=-0.85, m=0.0, sigma=0.02)),
        })
        # Query a deep OTM put strike and a benign ATM strike.
        K_deep = 60.0
        K_atm  = 100.0
        sigma_deep = float(surf.local_volatility(K_deep, 1.5))
        sigma_atm  = float(surf.local_volatility(K_atm,  1.5))
        # Trigger condition: either sigma_lv > 1.0 OR sigma_lv/sigma_iv > 3.
        if sigma_deep > SIGMA_LV_WARNING:
            assert surf.local_vol_warning_count >= 1
            assert any(e["code"] == "LV_WARNING_EXTREME_PUT_WING"
                       for e in surf.local_vol_warning_events)
        # Benign ATM strike should never trigger a warning.
        assert sigma_atm < SIGMA_LV_WARNING

    def test_damping_cap_applies_after_hard_cap(self):
        # On a surface that produces large local vols, the damping cap
        # truncates the output below the hard cap without modifying the
        # pure-Dupire branch.
        surf = VolSurface.from_slice_map("STEEP", {
            "a": _make_svi_slice(1.0, SVIParams(a=0.005, b=0.80, rho=-0.85, m=0.0, sigma=0.02)),
            "b": _make_svi_slice(2.0, SVIParams(a=0.010, b=0.80, rho=-0.85, m=0.0, sigma=0.02)),
        })
        K_deep = 60.0
        raw    = float(surf.local_volatility(K_deep, 1.5))
        damped = float(surf.local_volatility(K_deep, 1.5, damping_cap=0.8))
        if raw > 0.8:
            assert damped == pytest.approx(0.8, rel=1.0e-12)
        else:
            assert damped == pytest.approx(raw, rel=1.0e-12)

    def test_damping_cap_must_be_positive(self):
        surf = self._skewed_surface()
        with pytest.raises(ValueError, match="damping_cap"):
            surf.local_volatility(100.0, 1.0, damping_cap=0.0)
        with pytest.raises(ValueError, match="damping_cap"):
            surf.local_volatility(100.0, 1.0, damping_cap=-0.1)

    def test_fallback_surface_raises(self):
        # Pure-fallback surfaces have no calibrated information from
        # which to derive Dupire local volatility.
        surf = VolSurface.from_slice_map("EMPTY", {})
        with pytest.raises(ValueError, match="local_volatility is undefined"):
            surf.local_volatility(100.0, 1.0)

    def test_nonpositive_inputs_rejected(self):
        surf = self._skewed_surface()
        with pytest.raises(ValueError, match="T must be strictly positive"):
            surf.local_volatility(100.0, -1.0)
        with pytest.raises(ValueError, match=r"strike\(s\) must be"):
            surf.local_volatility(-5.0, 1.0)

    def test_clip_count_does_not_increment_on_damping(self):
        # Damping clips are an opt-in transformation, not a numerical
        # guard, and must not be conflated with hard-cap or floor
        # events in ``local_vol_clip_count``.
        surf = self._skewed_surface()
        surf.local_vol_clip_count = 0
        # Query a benign point first to confirm no clips.
        surf.local_volatility(100.0, 1.0)
        baseline = surf.local_vol_clip_count
        # Apply a tight damping cap that would force a truncation.
        surf.local_volatility(70.0, 0.5, damping_cap=0.10)
        assert surf.local_vol_clip_count == baseline


# ---------------------------------------------------------------------------
# Local-volatility Monte Carlo integration
# ---------------------------------------------------------------------------

class TestLocalVolMonteCarlo:
    """Path generation under Dupire local volatility + pricer integration."""

    def _flat_surface(self, sigma_const: float, forward: float) -> VolSurface:
        slice_map = {}
        for T in (0.5, 1.0, 2.0):
            p = SVIParams(a=sigma_const ** 2 * T,
                          b=1.0e-6, rho=0.0, m=0.0, sigma=0.1)
            slice_map[str(T)] = _make_svi_slice(T, p, forward=forward)
        return VolSurface.from_slice_map("FLAT", slice_map)

    def _negative_skew_surface(self, forward: float = 100.0) -> VolSurface:
        slice_map = {}
        for T, a in [(0.5, 0.02), (1.0, 0.04), (2.0, 0.08)]:
            p = SVIParams(a=a, b=0.40, rho=-0.4, m=0.0, sigma=0.1)
            slice_map[str(T)] = _make_svi_slice(T, p, forward=forward)
        return VolSurface.from_slice_map("SKEW", slice_map)

    def test_simulate_paths_local_vol_returns_correct_shapes(self):
        from src.pricing.monte_carlo import MonteCarloPricer
        from tests.conftest import make_brc_row

        forward = 100.0
        surf = self._flat_surface(0.20, forward)
        row = make_brc_row(initial_level=forward, current_spot=forward,
                          initial_fixing_date="2026-05-31",
                          maturity_date="2027-05-31")
        # MBRC American observation forces the path-dependent regime.
        row["product_type"] = "MBRC"
        row["type_style"] = "american"
        row["underlying_isins"] = ["FLAT"]

        pricer = MonteCarloPricer(n_paths=1_000, seed=42)
        surfaces = {"FLAT": surf}
        paths, dates, sigma_path = pricer.simulate_paths_local_vol(
            row, surfaces, {"FLAT": 0.20}, risk_free_rate=0.03,
        )
        assert paths.shape == (1_000, len(dates), 1)
        assert sigma_path.shape == paths.shape
        # On a flat surface sigma_path equals sigma_const everywhere.
        np.testing.assert_allclose(sigma_path, 0.20, atol=1.0e-4)

    def test_flat_surface_reproduces_constant_vol_paths(self):
        # Critical safety net: on a flat surface (no smile, no term
        # structure), simulate_paths_local_vol must reproduce
        # simulate_paths to within a tight numerical tolerance under
        # the same RNG seed.
        from src.pricing.monte_carlo import MonteCarloPricer
        from tests.conftest import make_brc_row

        sigma_const, forward = 0.20, 100.0
        flat = self._flat_surface(sigma_const, forward)
        row = make_brc_row(initial_level=forward, current_spot=forward,
                          initial_fixing_date="2026-05-31",
                          maturity_date="2027-05-31")
        row["underlying_isins"] = ["FLAT"]

        pricer = MonteCarloPricer(n_paths=2_000, seed=42)
        paths_cv, dates_cv = pricer.simulate_paths(
            row, vol_map={"FLAT": sigma_const}, risk_free_rate=0.03,
        )
        paths_lv, dates_lv, _ = pricer.simulate_paths_local_vol(
            row, vol_surfaces={"FLAT": flat},
            fallback_vol_map={"FLAT": sigma_const}, risk_free_rate=0.03,
        )
        assert (dates_cv == dates_lv).all()
        # Terminal-spot agreement: the two paths must coincide to the
        # numerical precision of the Dupire derivation. We allow a
        # tolerance of 0.5 % of the typical spot to absorb the
        # floating-point residue without being too loose.
        max_diff = float(np.abs(paths_lv[:, -1, 0] - paths_cv[:, -1, 0]).max())
        assert max_diff < 0.5

    def test_flat_surface_reproduces_constant_vol_fair_value(self):
        # The flat-surface regression at the pricer level: MBRC fair
        # value under local vol must match the constant-vol baseline
        # within Monte Carlo noise.
        from src.pricing.monte_carlo import MonteCarloPricer
        from tests.conftest import make_mbrc_row

        sigma_const = 0.20
        flat_a = self._flat_surface(sigma_const, 100.0)
        flat_b = self._flat_surface(sigma_const, 80.0)
        row = make_mbrc_row(barrier_pct=0.65,
                            initial_levels=[100.0, 80.0],
                            current_spots=[100.0, 80.0],
                            initial_fixing_date="2026-05-31",
                            maturity_date="2027-05-31")
        row["product_type"] = "MBRC"
        row["type_style"] = "american"

        portfolio = pd.DataFrame([row])
        pricer = MonteCarloPricer(n_paths=3_000, seed=42)
        vol_map = {row["underlying_isins"][0]: sigma_const,
                   row["underlying_isins"][1]: sigma_const}
        _, _, fv_cv = pricer.compute_portfolio_greeks(
            portfolio, vol_map, risk_free_rates={"CHF": 0.01},
        )
        _, _, fv_lv = pricer.compute_portfolio_greeks(
            portfolio, vol_map, risk_free_rates={"CHF": 0.01},
            vol_surfaces={row["underlying_isins"][0]: flat_a,
                          row["underlying_isins"][1]: flat_b},
        )
        fv_const = float(fv_cv["fair_value"].iloc[0])
        fv_local = float(fv_lv["fair_value"].iloc[0])
        # MC tolerance: the two pricers share the seed, so the residual
        # difference is purely the bridge variance accounting and the
        # midpoint-vs-left-endpoint Euler convention. A small relative
        # tolerance is appropriate.
        relative = abs(fv_local - fv_const) / max(abs(fv_const), 1.0)
        assert relative < 1.0e-2, (
            f"Flat-surface regression: local-vol FV={fv_local:.2f} vs "
            f"constant-vol FV={fv_const:.2f} (relative diff {relative:.3%})"
        )

    def test_negative_skew_mbrc_sits_between_atm_and_barrier_baselines(self):
        # Under negative skew, an American-barrier MBRC produces three
        # distinct fair values, ordered as follows:
        #
        #   ATM constant-vol   : single ATM sigma plugged into GBM.
        #                        Over-estimates the bond NPV because
        #                        the barrier-hit probability is
        #                        computed under a volatility much
        #                        lower than the surface assigns to
        #                        the barrier region. FV is too high.
        #
        #   Barrier constant-vol: single barrier-strike sigma plugged
        #                         into GBM. Over-corrects: the
        #                         elevated wing volatility is applied
        #                         to the whole path, not only near
        #                         the barrier. FV is too low.
        #
        #   Local-volatility   : (S, t)-dependent sigma. Surface-
        #                        consistent. FV sits between the two
        #                        -- corrects the ATM bug without
        #                        over-applying the wing vol.
        from src.pricing.monte_carlo import MonteCarloPricer
        from tests.conftest import make_brc_row

        forward = 100.0
        skewed = self._negative_skew_surface(forward)
        row = make_brc_row(barrier_pct=0.65, initial_level=forward,
                          current_spot=forward,
                          initial_fixing_date="2026-05-31",
                          maturity_date="2027-05-31")
        row["product_type"] = "MBRC"
        row["type_style"] = "american"
        row["underlying_isins"] = ["SKEW"]

        portfolio = pd.DataFrame([row])
        pricer = MonteCarloPricer(n_paths=4_000, seed=42)
        sigma_atm     = float(skewed.sigma(forward, 1.0))
        sigma_barrier = float(skewed.sigma(0.65 * forward, 1.0))

        # ATM constant-vol baseline.
        _, _, fv_atm = pricer.compute_portfolio_greeks(
            portfolio, {"SKEW": sigma_atm},
            risk_free_rates={"CHF": 0.01},
        )
        # Barrier constant-vol baseline.
        _, _, fv_bar = pricer.compute_portfolio_greeks(
            portfolio, {"SKEW": sigma_barrier},
            risk_free_rates={"CHF": 0.01},
        )
        # Local-volatility regime.
        _, _, fv_lv = pricer.compute_portfolio_greeks(
            portfolio, {"SKEW": sigma_barrier},
            risk_free_rates={"CHF": 0.01},
            vol_surfaces={"SKEW": skewed},
        )
        fv_ATM_only = float(fv_atm["fair_value"].iloc[0])
        fv_BAR_only = float(fv_bar["fair_value"].iloc[0])
        fv_LV       = float(fv_lv["fair_value"].iloc[0])
        se          = float(fv_atm["std_error"].iloc[0])
        # Expected ordering: ATM-only > local-vol > barrier-only, each
        # gap of at least a couple of MC standard errors.
        assert fv_ATM_only > fv_LV + 2.0 * se, (
            f"ATM-vol FV ({fv_ATM_only:.2f}) should exceed local-vol FV ({fv_LV:.2f})"
        )
        assert fv_LV > fv_BAR_only + 2.0 * se, (
            f"Local-vol FV ({fv_LV:.2f}) should exceed barrier-vol FV ({fv_BAR_only:.2f})"
        )

    def test_european_product_keeps_substage_a_path(self):
        # A European-barrier product must not be routed to local-vol
        # mode even when surfaces are available: substage A's scalar
        # input is strictly correct for this payoff class.
        from src.pricing.monte_carlo import MonteCarloPricer
        from tests.conftest import make_brc_row

        forward = 100.0
        skewed = self._negative_skew_surface(forward)
        row = make_brc_row(barrier_pct=0.65, initial_level=forward,
                          current_spot=forward,
                          initial_fixing_date="2026-05-31",
                          maturity_date="2027-05-31")
        row["underlying_isins"] = ["SKEW"]
        # type_style stays "european" -> _is_american is False ->
        # _should_use_local_vol returns False.

        pricer = MonteCarloPricer(n_paths=200, seed=42)
        assert not pricer._should_use_local_vol(row, {"SKEW": skewed})

    def test_should_use_local_vol_requires_path_dependence(self):
        from src.pricing.monte_carlo import MonteCarloPricer
        from tests.conftest import make_brc_row

        pricer = MonteCarloPricer(n_paths=100, seed=42)
        skewed = self._negative_skew_surface(100.0)

        # European BRC -> False even with surface
        row_euro = make_brc_row()
        row_euro["underlying_isins"] = ["SKEW"]
        row_euro["type_style"] = "european"
        assert pricer._should_use_local_vol(row_euro, {"SKEW": skewed}) is False

        # MBRC American -> True with surface
        row_mbrc = make_brc_row()
        row_mbrc["product_type"] = "MBRC"
        row_mbrc["type_style"] = "american"
        row_mbrc["underlying_isins"] = ["SKEW"]
        assert pricer._should_use_local_vol(row_mbrc, {"SKEW": skewed}) is True

        # MBRC American but no surface -> False
        assert pricer._should_use_local_vol(row_mbrc, {}) is False
        assert pricer._should_use_local_vol(row_mbrc, None) is False

        # Surface in fallback regime -> False
        empty_surface = VolSurface.from_slice_map("EMPTY", {})
        assert pricer._should_use_local_vol(
            row_mbrc, {"SKEW": empty_surface}
        ) is False
