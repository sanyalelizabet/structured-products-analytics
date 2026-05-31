"""Linear-algebra helpers for correlation matrices.

Pairwise sample correlation (different overlap per pair) and NaN→0 fills can
produce a symmetric matrix that is not positive semi-definite, which has no
Cholesky factor. These helpers project such a matrix to the nearest valid
correlation matrix (Higham, 2002) and provide a Cholesky that never raises.
"""
from __future__ import annotations

import numpy as np


def is_positive_semidefinite(A: np.ndarray, tol: float = 1e-8) -> bool:
    """True if ``A`` is symmetric PSD within ``tol`` (smallest eigenvalue ≥ -tol).

    ``tol`` is 1e-8: tight enough to reject genuinely indefinite matrices, loose
    enough for matrices that are PSD up to float64 round-off (e.g. the output of
    :func:`nearest_correlation_matrix`).
    """
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T, atol=1e-12):
        return False
    return float(np.linalg.eigvalsh(A).min()) >= -tol


def _project_to_psd(A: np.ndarray) -> np.ndarray:
    """Nearest PSD matrix in Frobenius norm: clip negative eigenvalues to 0."""
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * w) @ V.T


def nearest_correlation_matrix(
    A: np.ndarray, max_iter: int = 100, tol: float = 1e-8
) -> np.ndarray:
    """Nearest correlation matrix to ``A`` (Higham, 2002).

    Alternating projections between the PSD cone and the set of unit-diagonal
    matrices. Returns a symmetric PSD matrix with unit diagonal. ``A`` must be
    square and symmetric (it is symmetrised on entry).

    Reference: N. J. Higham, "Computing the nearest correlation matrix — a
    problem from finance", IMA J. Numer. Anal. 22(3), 2002.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    Y = (A + A.T) / 2.0
    dS = np.zeros_like(Y)        # Dykstra correction term
    X = Y.copy()

    for _ in range(max_iter):
        R = Y - dS
        X = _project_to_psd(R)            # project onto PSD cone
        dS = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)          # project onto unit diagonal
        denom = np.linalg.norm(Y, "fro")
        if denom > 0 and np.linalg.norm(Y - X, "fro") / denom < tol:
            break

    np.fill_diagonal(Y, 1.0)
    return Y


def safe_cholesky(corr: np.ndarray) -> np.ndarray:
    """Lower-triangular factor ``L`` with ``L @ L.T ≈ corr`` that never raises.

    Tries a plain Cholesky first. If ``corr`` is not positive definite, projects
    to the nearest correlation matrix, retries with escalating diagonal jitter,
    and finally falls back to an eigenvalue square root (always defined for a
    symmetric matrix).
    """
    corr = np.asarray(corr, dtype=float)
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        pass

    fixed = nearest_correlation_matrix(corr)
    eye = np.eye(corr.shape[0])
    jitter = 1e-12
    for _ in range(8):
        try:
            return np.linalg.cholesky(fixed + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # Eigenvalue square root: V·diag(√max(λ,0)) is a valid factor for any
    # symmetric matrix, even singular ones.
    w, V = np.linalg.eigh(fixed)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w))
