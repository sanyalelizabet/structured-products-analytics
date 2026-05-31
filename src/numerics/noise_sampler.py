"""Common Random Numbers sampler for Monte Carlo scenario engines.

When multiple scenarios are run against the *same* sampler, their P&Ls
share the same underlying noise realisation. Differences across
scenarios then reflect only the scenario change — not seed jitter. This
is the standard variance-reduction technique for sensitivity analysis
in stochastic simulation (Glasserman, *Monte Carlo Methods in Financial
Engineering*, ch. 7).

Two tensors are cached:

* ``Z_factor`` — shape ``(n_paths, n_days, K)``. Standard-normal draws
  driving the factor block; the engine multiplies by the Cholesky factor
  of the factor correlation matrix to obtain correlated innovations.

* ``Z_idio``   — ``dict[isin → ndarray of shape (n_paths, n_days)]``.
  Independent standard-normal draws per underlying for the
  idiosyncratic component.

Compatibility with a downstream engine is checked via :meth:`matches`
(paths × days × factor universe × ISIN universe).  Use
:meth:`regenerate` to draw a fresh sample for the user's "another roll
of the dice" button.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def _stable_isin_seed(isin: str, master_seed: int) -> int:
    """Deterministic per-ISIN seed derived from master + ISIN string."""
    h = int(hashlib.md5(str(isin).encode()).hexdigest(), 16)
    return (h ^ master_seed) & 0x7FFFFFFF


class NoiseSampler:
    """Cached Gaussian innovations for path-based MC scenarios."""

    def __init__(
        self,
        n_paths: int,
        n_days:  int,
        factor_codes: Iterable[str],
        isins:        Iterable[str],
        seed: int = 42,
    ):
        self.n_paths      = int(n_paths)
        self.n_days       = int(n_days)
        self.factor_codes = list(factor_codes)
        self.isins        = sorted(isins)
        self.seed         = int(seed)
        self._populate()

    # ------------------------------------------------------------------ data

    def _populate(self):
        K = len(self.factor_codes)
        rng_factor = np.random.default_rng(self.seed)
        self.Z_factor = rng_factor.standard_normal(
            (self.n_paths, self.n_days, K)
        )

        self.Z_idio: dict[str, np.ndarray] = {}
        for isin in self.isins:
            rng_isin = np.random.default_rng(_stable_isin_seed(isin, self.seed))
            self.Z_idio[isin] = rng_isin.standard_normal(
                (self.n_paths, self.n_days)
            )

    # ------------------------------------------------------------- accessors

    def factor_noise(self) -> np.ndarray:
        """Standardised factor innovations, shape ``(n_paths, n_days, K)``."""
        return self.Z_factor

    def knock_in_uniform(self, key: str, n_paths: int | None = None) -> np.ndarray:
        """Reproducible ``U(0,1)`` draws for a continuous-barrier knock-in.

        Keyed by ``key`` (e.g. a product id) so each product gets an independent
        stream, derived from the master seed exactly like the idiosyncratic
        blocks.  This keeps the engine a pure consumer of the sampler (no
        engine-side seeding) and lets :meth:`regenerate` refresh the breach draws
        together with the path noise.

        Returns
        -------
        np.ndarray, shape ``(n_paths,)`` — one uniform per path.
        """
        n = self.n_paths if n_paths is None else int(n_paths)
        rng = np.random.default_rng(_stable_isin_seed(f"knockin:{key}", self.seed))
        return rng.random(n)

    def idio_noise_for(self, isins: Iterable[str]) -> np.ndarray:
        """Stack per-ISIN idio noise for a subset, in the requested order.

        Returns
        -------
        np.ndarray  shape ``(n_paths, n_days, len(isins))``.

        Raises
        ------
        KeyError  if any requested ISIN was not part of the sampler universe.
        """
        try:
            blocks = [self.Z_idio[isin] for isin in isins]
        except KeyError as e:
            raise KeyError(
                f"ISIN {e} not in noise sampler — recreate sampler with the "
                "full asset universe of the portfolio."
            ) from None
        return np.stack(blocks, axis=-1)

    # --------------------------------------------------------- compatibility

    def matches(
        self,
        n_paths: int,
        n_days:  int,
        factor_codes: Iterable[str],
        isins:        Iterable[str],
    ) -> bool:
        """Check whether this sampler can serve a request without re-drawing.

        All four dimensions must match: paths × days × factor list × ISIN
        universe (as a set — order independent).
        """
        return (
            self.n_paths == int(n_paths)
            and self.n_days == int(n_days)
            and list(factor_codes) == self.factor_codes
            and set(isins) <= set(self.isins)
        )

    # ------------------------------------------------------------ regenerate

    def regenerate(self, seed: int | None = None) -> None:
        """Refresh the cached tensors with a new master seed.

        If ``seed`` is omitted the seed is bumped deterministically so the
        next sample is reproducible across runs but independent from the
        previous one.
        """
        if seed is None:
            # LCG-style bump: cheap, reproducible, independent enough.
            self.seed = (self.seed * 16807 + 1) % 2147483647
        else:
            self.seed = int(seed)
        self._populate()
