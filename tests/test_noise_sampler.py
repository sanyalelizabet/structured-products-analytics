"""Tests for ``NoiseSampler`` — the Common Random Numbers cache."""
from __future__ import annotations

import numpy as np
import pytest

from src.numerics.noise_sampler import NoiseSampler, _stable_isin_seed


# ──────────────────────────────────────────────────────────────────────────
# Shape & reproducibility
# ──────────────────────────────────────────────────────────────────────────

class TestShapesAndReproducibility:
    def test_factor_noise_shape(self):
        s = NoiseSampler(n_paths=50, n_days=300,
                         factor_codes=["MKT", "TECH", "FX"],
                         isins=["A", "B"], seed=1)
        assert s.factor_noise().shape == (50, 300, 3)

    def test_idio_noise_shape_for_subset(self):
        s = NoiseSampler(n_paths=20, n_days=100,
                         factor_codes=["MKT"],
                         isins=["A", "B", "C"], seed=2)
        z = s.idio_noise_for(["A", "C"])
        assert z.shape == (20, 100, 2)

    def test_same_seed_same_draws(self):
        kwargs = dict(n_paths=30, n_days=200,
                      factor_codes=["MKT", "TECH"],
                      isins=["X", "Y"], seed=7)
        s1 = NoiseSampler(**kwargs)
        s2 = NoiseSampler(**kwargs)
        np.testing.assert_array_equal(s1.factor_noise(), s2.factor_noise())
        np.testing.assert_array_equal(s1.idio_noise_for(["X", "Y"]),
                                      s2.idio_noise_for(["X", "Y"]))

    def test_different_seed_different_draws(self):
        a = NoiseSampler(50, 100, ["MKT"], ["A"], seed=1)
        b = NoiseSampler(50, 100, ["MKT"], ["A"], seed=2)
        # At minimum, factor draws differ
        assert not np.array_equal(a.factor_noise(), b.factor_noise())


# ──────────────────────────────────────────────────────────────────────────
# Statistical properties — should be roughly unit-variance, zero-mean
# ──────────────────────────────────────────────────────────────────────────

class TestStatisticalProperties:
    def test_factor_noise_is_standard_normal(self):
        s = NoiseSampler(200, 200, ["MKT", "TECH"], ["A"], seed=42)
        z = s.factor_noise()
        # 200·200·2 = 80,000 obs → mean within ±3·SE ≈ ±0.011
        assert abs(z.mean()) < 0.02
        assert abs(z.std() - 1.0) < 0.02

    def test_idio_noise_is_standard_normal(self):
        s = NoiseSampler(200, 200, ["MKT"], ["A"], seed=42)
        z = s.idio_noise_for(["A"])
        assert abs(z.mean()) < 0.03
        assert abs(z.std() - 1.0) < 0.03


# ──────────────────────────────────────────────────────────────────────────
# Independence between factor block and idio noise (shouldn't share seed)
# ──────────────────────────────────────────────────────────────────────────

class TestIndependence:
    def test_factor_and_idio_uncorrelated(self):
        s = NoiseSampler(500, 300, ["MKT"], ["A"], seed=11)
        f = s.factor_noise().reshape(-1)
        i = s.idio_noise_for(["A"]).reshape(-1)
        n = min(len(f), len(i))
        # Sample correlation should be near zero (~ 1/√n)
        corr = float(np.corrcoef(f[:n], i[:n])[0, 1])
        assert abs(corr) < 0.05

    def test_two_isins_idio_uncorrelated(self):
        s = NoiseSampler(500, 300, ["MKT"], ["A", "B"], seed=11)
        a = s.idio_noise_for(["A"]).reshape(-1)
        b = s.idio_noise_for(["B"]).reshape(-1)
        corr = float(np.corrcoef(a, b)[0, 1])
        assert abs(corr) < 0.05


# ──────────────────────────────────────────────────────────────────────────
# Compatibility check
# ──────────────────────────────────────────────────────────────────────────

class TestMatches:
    @pytest.fixture
    def s(self):
        return NoiseSampler(100, 200, ["MKT", "TECH"], ["A", "B"], seed=1)

    def test_exact_match(self, s):
        assert s.matches(100, 200, ["MKT", "TECH"], ["A", "B"])

    def test_subset_isins_ok(self, s):
        # Only requesting a subset of the cached ISIN universe is fine.
        assert s.matches(100, 200, ["MKT", "TECH"], ["A"])

    def test_extra_isin_invalidates(self, s):
        assert not s.matches(100, 200, ["MKT", "TECH"], ["A", "B", "C"])

    def test_n_paths_change_invalidates(self, s):
        assert not s.matches(101, 200, ["MKT", "TECH"], ["A"])

    def test_n_days_change_invalidates(self, s):
        assert not s.matches(100, 201, ["MKT", "TECH"], ["A"])

    def test_factor_universe_change_invalidates(self, s):
        assert not s.matches(100, 200, ["MKT"], ["A"])
        assert not s.matches(100, 200, ["MKT", "FX"], ["A"])


# ──────────────────────────────────────────────────────────────────────────
# Regeneration — fresh independent sample
# ──────────────────────────────────────────────────────────────────────────

class TestRegenerate:
    def test_regenerate_changes_draws(self):
        s = NoiseSampler(50, 100, ["MKT"], ["A"], seed=1)
        z_before = s.factor_noise().copy()
        s.regenerate()
        z_after = s.factor_noise()
        assert not np.array_equal(z_before, z_after)

    def test_regenerate_preserves_shape(self):
        s = NoiseSampler(50, 100, ["MKT", "TECH"], ["A"], seed=1)
        s.regenerate()
        assert s.factor_noise().shape == (50, 100, 2)
        assert s.idio_noise_for(["A"]).shape == (50, 100, 1)

    def test_explicit_seed_reproducible(self):
        s = NoiseSampler(50, 100, ["MKT"], ["A"], seed=1)
        s.regenerate(seed=999)
        z1 = s.factor_noise().copy()
        s.regenerate(seed=999)
        z2 = s.factor_noise()
        np.testing.assert_array_equal(z1, z2)

    def test_regenerated_samples_uncorrelated(self):
        s = NoiseSampler(500, 300, ["MKT"], ["A"], seed=1)
        a = s.factor_noise().reshape(-1).copy()
        s.regenerate()
        b = s.factor_noise().reshape(-1)
        # Two MC replicates should be near-independent.
        corr = float(np.corrcoef(a, b)[0, 1])
        assert abs(corr) < 0.05


# ──────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────

class TestErrors:
    def test_unknown_isin_raises(self):
        s = NoiseSampler(10, 10, ["MKT"], ["A"], seed=1)
        with pytest.raises(KeyError, match="not in noise sampler"):
            s.idio_noise_for(["A", "GHOST"])

    def test_isin_seed_is_stable(self):
        # The hash function used for ISIN seeding must be deterministic
        # across processes — md5 is used precisely for this reason.
        assert _stable_isin_seed("ABC", 42) == _stable_isin_seed("ABC", 42)
        assert _stable_isin_seed("ABC", 42) != _stable_isin_seed("DEF", 42)
