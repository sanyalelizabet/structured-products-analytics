"""Tests for src.linalg: nearest correlation matrix and safe Cholesky."""
import numpy as np
import pytest

from src.linalg import (
    is_positive_semidefinite,
    nearest_correlation_matrix,
    safe_cholesky,
)


# A classic non-PSD "correlation" matrix: the 3x3 with all off-diagonals -0.6
# has a negative eigenvalue.
NON_PSD = np.array([
    [1.0, -0.6, -0.6],
    [-0.6, 1.0, -0.6],
    [-0.6, -0.6, 1.0],
])

PSD = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.2],
    [0.3, 0.2, 1.0],
])


class TestIsPSD:
    def test_psd_matrix_is_psd(self):
        assert is_positive_semidefinite(PSD)

    def test_non_psd_detected(self):
        assert not is_positive_semidefinite(NON_PSD)


class TestNearestCorrelation:
    def test_output_is_psd(self):
        ncm = nearest_correlation_matrix(NON_PSD)
        assert is_positive_semidefinite(ncm)

    def test_unit_diagonal(self):
        ncm = nearest_correlation_matrix(NON_PSD)
        assert np.allclose(np.diag(ncm), 1.0)

    def test_symmetric(self):
        ncm = nearest_correlation_matrix(NON_PSD)
        assert np.allclose(ncm, ncm.T)

    def test_already_valid_is_unchanged(self):
        ncm = nearest_correlation_matrix(PSD)
        assert np.allclose(ncm, PSD, atol=1e-6)

    def test_offdiagonals_in_range(self):
        ncm = nearest_correlation_matrix(NON_PSD)
        assert ncm.min() >= -1.0 - 1e-9
        assert ncm.max() <= 1.0 + 1e-9


class TestSafeCholesky:
    def test_plain_cholesky_for_psd(self):
        L = safe_cholesky(PSD)
        assert np.allclose(L @ L.T, PSD, atol=1e-8)
        # lower-triangular
        assert np.allclose(np.triu(L, 1), 0.0)

    def test_does_not_raise_on_non_psd(self):
        L = safe_cholesky(NON_PSD)  # would raise with np.linalg.cholesky
        assert L.shape == (3, 3)

    def test_factor_reconstructs_nearest_correlation(self):
        L = safe_cholesky(NON_PSD)
        recon = L @ L.T
        assert is_positive_semidefinite(recon)
        assert np.allclose(np.diag(recon), 1.0, atol=1e-6)

    def test_identity_factor_is_identity(self):
        L = safe_cholesky(np.eye(4))
        assert np.allclose(L, np.eye(4))
