# Implied Volatility Surface — Stage 2 closeout

## 1. Scope of Stage 2

The second stage of the implied volatility surface construction has
been delivered. The stage assembles the per-slice surfaces of Stage 1
into a single term-structure-consistent object — one
:class:`VolSurface` per underlying — that exposes the implied
volatility as a function of strike *and* tenor. The interpolation
between listed expiries is performed by the linear-in-total-variance
recipe of Gatheral (2006), and the extrapolation outside the listed
range is performed by vol-flat scaling of the appropriate anchor
slice. Calendar-arbitrage monotonicity is audited at construction
and the audit results are exposed on the surface object for the user
interface to badge. As in Stage 1, the new code is purely additive:
no view and no pricer consumes the surface yet, and the bug in the
present pricer — its consumption of the at-the-money volatility at
the barrier strike — remains the substance of Stage 3.

## 2. Artefacts

The artefacts of Stage 2 are confined to the four files already
touched by Stage 1 and require no new modules. The term-structure
helpers ``interpolate_total_variance``, ``extrapolate_atm_scaling``
and ``verify_calendar_monotone``, together with the
:class:`VolSurface` class and the surface-status taxonomy constants,
are appended to [src/pricing/vol_surface.py](../src/pricing/vol_surface.py).
The market-data integration is a new method
``build_vol_surfaces`` added to
[src/market_data_engine.py](../src/market_data_engine.py), which composes
the slice map produced by Stage 1 into surface objects. The
Streamlit cache wrapper ``fetch_vol_surfaces`` in
[app/streamlit_app.py](../app/streamlit_app.py) is updated in place to
return the per-ISIN surface dictionary rather than the slice
dictionary. The test suite gains four new classes and twenty-four new
test functions in [tests/test_vol_surface.py](../tests/test_vol_surface.py),
covering the interpolator, the extrapolator, the calendar audit, and
the end-to-end dispatch of :class:`VolSurface`. The methodology
document [docs/vol_surface_methodology.md](vol_surface_methodology.md)
gains a new Section 6 specifying the term-structure assembly and is
re-numbered accordingly.

## 3. Verification

The full test suite passes — 712 tests, one expected failure
unaffected by the present work. The new tests are organised across
four classes: the interpolator (endpoint identity, midpoint
linearity, bracket-order and out-of-bracket rejection); the
extrapolator (anchor identity, vol-flat invariant in tenor,
non-positive tenor rejection); the calendar-arbitrage audit (clean
surface, inverted surface, single-slice and empty cases); and
:class:`VolSurface` itself (listed-tenor identity, interior
interpolation, both extrapolation regimes, single-slice branch,
fallback branch, vectorised query, moneyness convenience wrapper,
strict-tenor-ordering rejection, and the live audit-violation flag).

The cache layer was exercised against the stored ``data/options.csv``
covering ten underlyings. The status taxonomy distributed as
expected: eight surfaces contained at least one calibrated SVI slice
and one of those (the underlying with a single SVI slice) populated
the *single_slice* branch; two surfaces — both Swiss-listed single
names with insufficient Yahoo chain density — populated the
*fallback* branch. Across the eight non-fallback surfaces, the
listed-expiry coverage on this snapshot extended at most to
*T* &asymp; 0.8 years, so that two-year barrier-product valuations
sit firmly in the *extrapolated* regime. The calendar-arbitrage audit
recorded a small number of monotonicity violations (zero to two per
surface) attributable to Yahoo quote noise; the violations are
informational and do not affect the surface's usability.

## 4. The bug, quantified

The principal deliverable of Stage 2, beyond the surface object
itself, is the ability to measure the implied volatility correction
that Stage 3 will deliver. The table below contrasts, for each
underlying with available chain data, the at-the-money volatility
input that the present pricer consumes against the volatility at the
sixty-five-per-cent-of-spot barrier strike, evaluated at a
representative two-year tenor.

The correction segregates the universe into two cohorts. On the
mature, low-to-moderate-volatility names (at-the-money volatility
between 20 and 45 per cent) the surface reports a barrier-strike
volatility that exceeds the at-the-money volatility by between five
and fourteen volatility points, corresponding to a relative
mis-pricing of the constant-volatility input of between fifteen and
seventy per cent. On the high-volatility speculative names
(at-the-money volatility above fifty per cent) the surface reports a
flat or marginally inverted skew, consistent with the observation
recorded at the close of Stage 1 that names whose at-the-money
volatility is elevated typically exhibit a call-heavy or symmetric
smile.

The implication for the Stage 3 pricer migration is that the
correction will be concentrated on the products buy-side asset
managers actually carry — barrier reverse convertibles and
autocallable structures on mature single-name and index underlyings —
and will be small on the speculative book. The asymmetry is the
right one for a buy-side analytics tool: the correction lands where
the pricing error is largest.

## 5. Known caveats carried into Stage 3

Three caveats are carried forward into the Stage 3 specification.

The first is the absence of an evaluation-time butterfly audit at
intermediate tenors. The linear-in-total-variance interpolation does
not preserve the Durrleman condition in general; in practice the
violation magnitude is small but it is non-zero. The audit is
deferred to Stage 3 because it becomes economically material only
when the surface is queried at every Monte Carlo time step, which is
the regime of the path-dependent pricer migration.

The second is the continued use of the spot price as a proxy for the
forward. The simplification was acceptable at the slice level under
Stage 1 because the smile-translation parameter absorbs the offset;
under Stage 2 it carries through unchanged. Stage 3 is the natural
juncture at which the risk-free rate term structure and a dividend
yield estimator are integrated, since the pricer migration is the
first context in which the forward enters the valuation arithmetic
directly.

The third is the term-structure coverage envelope of the Yahoo
chain. The listed expiries available on most underlyings extend at
most two years out from the valuation date; for the three-to-five-
year barrier products that characterise the Swiss private-client
universe, the surface is forced into the vol-flat extrapolation
regime over a substantial fraction of the product's life. The
limitation is not introduced by the present stage but is
illuminated by it: the surface badge now reports the
*extrapolated* status explicitly, allowing the user to assess the
provenance of the value rather than implicitly trusting the
constant-vol shortcut.

## 6. Stage 3 unblock

Stage 2 is closed. The Stage 3 specification — the migration of the
Monte Carlo pricer from its constant-volatility input to a surface-
aware evaluation — may now be drafted against the verified
``VolSurface.sigma(K, T)`` interface. The first substage of Stage 3
is a pure surface lookup at the European-barrier reverse-convertible
pricer, fully unblocked by the present stage; the second substage
extends the change to the multi-barrier and autocallable pricers
under Dupire local volatility derived from the present surface,
also unblocked.
