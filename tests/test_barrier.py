"""Unit tests for the barrier-observation helpers (`src.barrier`)."""
from __future__ import annotations

import numpy as np

from src.barrier import (
    continuous_survival_prob,
    continuous_survival_prob_from_var,
    european_knock_in,
    sample_knock_in,
)


class TestEuropeanKnockIn:
    def test_breach_when_terminal_at_or_below_barrier(self):
        terminal = np.array([[75.0], [80.0], [85.0]])
        barriers = np.array([80.0])
        out = european_knock_in(terminal, barriers)
        assert out.tolist() == [True, True, False]   # <= barrier counts as breach

    def test_worst_of_any_asset_breaches(self):
        terminal = np.array([[120.0, 75.0], [120.0, 130.0]])
        barriers = np.array([80.0, 80.0])
        out = european_knock_in(terminal, barriers)
        assert out.tolist() == [True, False]          # any underlying below → breach


class TestContinuousSurvival:
    def _bridge(self, sa, sb, b, vol, dt):
        la, lb = np.log(sa / b), np.log(sb / b)
        return 1.0 - np.exp(-2.0 * la * lb / (vol ** 2 * dt))

    def test_matches_closed_form_single_step(self):
        prices = np.array([[[100.0], [85.0]]])        # one path, two points, one asset
        out = continuous_survival_prob(prices, np.array([80.0]), np.array([0.20]),
                                       np.array([1.0]))
        np.testing.assert_allclose(out[0], self._bridge(100.0, 85.0, 80.0, 0.20, 1.0))

    def test_endpoint_at_or_below_barrier_is_certain_knock_in(self):
        prices = np.array([[[100.0], [80.0]]])        # second point sits on the barrier
        out = continuous_survival_prob(prices, np.array([80.0]), np.array([0.20]),
                                       np.array([1.0]))
        assert out[0] == 0.0

    def test_far_from_barrier_survives_almost_surely(self):
        prices = np.array([[[100.0], [100.0], [100.0]]])
        out = continuous_survival_prob(prices, np.array([10.0]), np.array([0.20]),
                                       np.array([0.5, 0.5]))
        assert out[0] > 0.999

    def test_survival_decreases_as_path_nears_barrier(self):
        far  = np.array([[[100.0], [100.0]]])
        near = np.array([[[100.0], [82.0]]])
        b, vol, dt = np.array([80.0]), np.array([0.20]), np.array([1.0])
        s_far  = continuous_survival_prob(far,  b, vol, dt)[0]
        s_near = continuous_survival_prob(near, b, vol, dt)[0]
        assert s_near < s_far

    def test_worst_of_multi_asset_combines(self):
        # Asset 2 dips to the barrier on the second point → certain knock-in.
        prices = np.array([[[100.0, 100.0], [95.0, 80.0]]])
        out = continuous_survival_prob(prices, np.array([80.0, 80.0]),
                                       np.array([0.20, 0.20]), np.array([1.0]))
        assert out[0] == 0.0

    def test_continuous_never_above_european_survival(self):
        # Continuous monitoring can only find *more* breaches than the terminal
        # check, so survival prob <= 1 - european_knock_in along the path.
        prices = np.array([[[100.0], [90.0], [105.0]]])   # dips then recovers above barrier
        b = np.array([88.0])
        cont = continuous_survival_prob(prices, b, np.array([0.25]),
                                        np.array([0.4, 0.4]))[0]
        eur_survives = not european_knock_in(prices[:, -1, :], b)[0]
        assert eur_survives                # terminal 105 > 88 → European never knocks in
        assert cont < 1.0                  # but the interim dip carries breach probability


class TestSurvivalFromVar:
    def test_from_var_matches_vols_dt(self):
        # The convenience wrapper must equal the from-var core with σ²Δt.
        prices = np.array([[[100.0], [92.0], [101.0]]])
        b, vols, dt = np.array([88.0]), np.array([0.25]), np.array([0.4, 0.4])
        step_var = (vols[None, :] ** 2) * dt[:, None]          # (n_steps, n_assets)
        a = continuous_survival_prob(prices, b, vols, dt)
        c = continuous_survival_prob_from_var(prices, b, step_var)
        np.testing.assert_allclose(a, c)


class TestSampleKnockIn:
    def test_breach_rate_matches_one_minus_survival(self):
        # Over many independent paths the sampled breach frequency must track
        # the bridge knock-in probability (1 − survival).
        rng = np.random.default_rng(0)
        n = 40_000
        # All paths identical (100 → 95) so survival is the same scalar p.
        prices = np.tile(np.array([[100.0], [95.0]]), (n, 1, 1))
        b, step_var = np.array([90.0]), np.array([[0.04]])     # σ²Δt = 0.04
        surv = continuous_survival_prob_from_var(prices, b, step_var)[0]
        u = rng.random(n)
        breached = sample_knock_in(prices, b, step_var, u)
        assert abs(breached.mean() - (1.0 - surv)) < 0.01

    def test_uniform_zero_never_breaches_one_always(self):
        prices = np.array([[[100.0], [95.0]], [[100.0], [95.0]]])
        b, step_var = np.array([90.0]), np.array([[0.04]])
        out = sample_knock_in(prices, b, step_var, np.array([0.0, 1.0]))
        assert out.tolist() == [False, True]   # u=0 < surv → survive; u=1 > surv → breach
