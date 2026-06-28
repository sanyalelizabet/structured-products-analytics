# The Implied Volatility Surface

## 1. Purpose and definition

The implied volatility used by the structured-products pricer is the
single most consequential input to the valuation of any product whose
payoff depends on the conditional distribution of the underlying at one
or more future dates. In the constant-volatility regime employed by the
existing Monte Carlo engine, a single scalar &sigma; per underlying is
plugged into a geometric-Brownian-motion dynamics that is then evaluated
at whichever strikes the payoff requires. The economic content of that
choice is the implicit assertion that, for the purposes of the valuation
in hand, the smile is flat.

For European vanilla payoffs the assumption is benign: by Black–Scholes
homogeneity, only one point of the smile enters the valuation, and the
choice of which point is dictated by the strike of the option. For
barrier products, however, the same assumption is materially
inconsistent with the market. The barrier of a typical reverse
convertible sits between fifty-five and seventy per cent of the initial
spot, and on equity index and single-name underlyings the implied
volatility at that strike exceeds the at-the-money volatility by an
amount that is typically five to fifteen volatility points and almost
never zero. Pricing the barrier with the at-the-money volatility is
therefore a known, signed mis-pricing: barrier-hit probabilities are
underestimated, bond components are overvalued, and fair values of
barrier reverse convertibles, multi-barrier reverse convertibles and
autocallable structures are systematically biased upward.

The implied volatility surface is the object that closes this gap. It
is a mapping

```math
\sigma : (K, T) \mapsto \sigma(K, T)
```

assigning, to each strike *K* and tenor *T*, the annualised implied
volatility that the listed option market quotes for the corresponding
European call (or put) on the underlying. The surface is constructed
slice by slice — one slice per listed expiry — and assembled across
expiries into a term-structure-consistent object. The present document
specifies the surface end to end: the slice-level calibration of
Sections 2 to 5, the term-structure assembly of Section 6, and the
two-substage migration of the Monte Carlo pricer to a surface-aware
volatility input in Section 8.

## 2. The raw SVI parameterisation

The slice at a single expiry is represented by the raw SVI
parameterisation of Gatheral (2004). Writing *k* = log(K / F) for the
log-moneyness relative to the forward at the slice maturity, the total
implied variance &nbsp;*w(k)* ≡ &sigma;²(k) &middot; T &nbsp;is taken
to have the functional form

```math
w(k;\, a, b, \rho, m, \varsigma) \;=\; a \;+\; b \left\{\, \rho\,(k - m) \;+\; \sqrt{(k - m)^2 + \varsigma^2}\, \right\},
```

with five free parameters. The parameter *a* is the vertical level of
total variance; *b* &geq; 0 is the overall slope of the wings; the skew
parameter &rho; &in; [−1, 1] controls the asymmetry between the put
and call sides of the smile, and is negative in essentially every
liquid equity context; *m* &in; ℝ is the horizontal translation of the
smile minimum; and &varsigma; > 0 controls the smoothness of the smile
in the at-the-money region. Implied volatility is recovered from total
variance as &sigma;(k, T) = sqrt(w(k) / T).

The choice of the raw parameterisation is deliberate. Although the
natural and (arbitrage-free) SSVI parameterisations of Gatheral and
Jacquier (2014) admit more economic interpretations, the raw form
maintains a one-to-one correspondence between its parameters and the
geometric features of the smile, which makes calibration diagnostics
and post-hoc inspection more direct. The arbitrage-free property is
not relinquished: rather than being imposed by the parameterisation
itself, it is enforced by the explicit Durrleman and Roger Lee gates
described in section 4.

The parameter tuple is constrained at construction time by the
elementary domain conditions *b* &geq; 0, |&rho;| &leq; 1 and &varsigma;
> 0, together with the global non-negativity of total variance at the
smile minimum,

```math
a + b\,\varsigma\,\sqrt{1 - \rho^2} \;\geq\; 0,
```

which ensures that *w(k)* &geq; 0 for every *k* and that the recovered
implied volatility is therefore real.

## 3. Calibration

Calibration of a single slice proceeds in two stages. The first stage
produces an initial guess by the quasi-explicit method of De Marco and
Martini (2009). The second stage refines that guess by a constrained
non-linear least-squares solver acting directly on the five physical
parameters.

The quasi-explicit seed exploits the partial linearity of the SVI
problem. Under the substitution *c* = *b*&varsigma; and *d* = *b*&varsigma;&rho;
the total variance becomes

```math
w_i \;\approx\; a \;+\; d\, x_i \;+\; c\, y_i, \qquad x_i = (k_i - m)/\varsigma, \;\; y_i = \sqrt{x_i^2 + 1},
```

a linear regression of total variance on (1, *x*, *y*) at every fixed
choice of (*m*, &varsigma;). The seed is obtained by anchoring *m* at
the log-moneyness of the empirical variance minimum, optimising &varsigma;
over the interval [0.01, 1.0] by a bounded one-dimensional search, and
extracting the inner linear solution at the optimum &varsigma;. The
resulting tuple is projected onto the admissible SVI cone — *c* &geq;
0, |*d*| &leq; *c* — and clipped to the optimiser bounds.

The refinement is performed by SciPy's Trust-Region Reflective
implementation of bounded non-linear least squares. The residuals are
expressed in volatility units rather than variance units,

```math
r_i \;=\; \sqrt{w_{\mathrm{i}} / T} \;-\; \sigma_{\mathrm{model}}(k_i, T),
```

so that the objective is dimensionally comparable across slices of
different maturity and gives equal economic weight to a
one-volatility-point miss regardless of tenor. The residuals are
weighted by an inverse-spread proxy for the information content of
each market quote: a tight bid-ask spread corresponds to a liquid,
informative observation and receives a higher weight, while a quote
whose spread exceeds the median of the slice receives a proportionally
lower weight. In the absence of bid-ask information the weights
collapse to uniform.

A minimum of five distinct strikes is required for the calibration to
proceed, which is the structural identification floor for the
five-parameter model. Slices that fall below this threshold are not
calibrated and are routed directly to the chain-proxy fallback
described in section 5.

## 4. Arbitrage conditions

The calibrated parameter tuple is subjected to two arbitrage gates
before being exposed to downstream pricing code. Both must pass for
the slice to be considered admissible in the SVI branch.

The first gate is Durrleman's butterfly condition. Durrleman (2010)
showed that the absence of butterfly arbitrage in a smile parameterised
by total variance is equivalent to the non-negativity of the function

```math
g(k) \;=\; \left( 1 - \frac{k\, w'(k)}{2\, w(k)} \right)^{2}
\;-\; \frac{w'(k)^2}{4}\left( \frac{1}{w(k)} + \frac{1}{4} \right)
\;+\; \frac{w''(k)}{2},
```

at every log-moneyness *k*. The quantity *g* is, up to a multiplicative
factor, the density of the risk-neutral distribution implied by the
slice; its non-negativity is therefore both necessary and sufficient
for that distribution to be a probability measure. The gate is
implemented numerically by evaluating *g* on a dense grid that spans
the band *k* &in; [−2.5, 2.5]. The grid is wide enough to surface
violations both at the smile minimum and in the wings, where the SVI
form is most prone to misbehave at calibrated tuples lying near the
boundary of the admissible set.

The second gate is the Roger Lee wing-bound condition. Lee (2004)
established that, in the limit of extreme strikes, the slope of total
implied variance with respect to log-moneyness is constrained by the
absence of static arbitrage in the corresponding strip of call prices.
For the raw SVI parameterisation the asymptotic slopes are *b*(1 +
&rho;) on the right wing and *b*(1 − &rho;) on the left. The
canonical necessary condition for absence of arbitrage in the wings is

```math
b\, T\, (1 + |\rho|) \;\leq\; 4,
```

equivalent to *b*(1 + |&rho;|) &leq; 4 / *T*. The bound is tenor-
dependent: parameter tuples admissible at *T* = 1 year may violate
the condition at *T* = 5 years because the right-hand side, 4 / *T*,
shrinks in tenor while the left-hand side does not. The gate is
implemented by direct evaluation of the inequality at the tenor of
the slice.

## 5. Data-quality gate and fallback policy

The arbitrage gates check the mathematical admissibility of the
calibrated parameter tuple. The data-quality gate complements them by
checking the informational admissibility of the underlying market
data and the closeness of the fit. Three conditions are evaluated:

A minimum number of strikes is required, set by default at five. The
threshold is the identification floor of the SVI model; values below it
cannot produce a stable smile and are rejected outright.

A minimum number of strikes whose bid-ask spread does not exceed
twenty-five per cent of the corresponding mid price is required, set
by default also at five. The condition reflects the empirical
observation that wide-spread quotes are diagnostic of stale or thin
options markets rather than of a genuine implied-volatility surface
feature.

A maximum root-mean-square calibration residual is required. The
ceiling is calibrated to the noise floor of the data source rather
than fixed at a single value. The present implementation consumes the
implied volatilities published in each option chain unchanged,
without re-inverting them from observed option prices, and therefore
inherits the cross-strike inconsistencies introduced by the vendor's
own Black–Scholes conventions — discount rate, dividend treatment,
exercise style. On free retail feeds these inconsistencies produce
per-slice residuals of two to three volatility points on liquid US
single names, and the ceiling is set at three points accordingly to
admit those fits while still rejecting genuinely divergent
calibrations. A migration to a cleaner feed or to in-house
Black–Scholes inversion under consistent conventions would warrant a
tighter ceiling of approximately one and a half volatility points.

When any of the three checks fails, the slice is routed into a
fallback. Two fallback branches are provided. The chain-proxy fallback
retains the raw observed implied volatilities and answers any query
strike by returning the implied volatility of the strike closest to it
in log-moneyness; the function is a step function with discontinuities
at the midpoints between consecutive listed strikes, and is therefore
suitable only for occasional point queries. The constant-volatility
fallback returns a single static volatility regardless of the query
strike, and is used when no chain data is available for the slice at
all.

The user interface exposes the fallback status as one of three values
— *svi*, *proxy*, *fallback* — together with the verbatim reason
recorded by the gate that triggered the fallback. The reason is
displayed on hover or in audit logs, so that the user may distinguish
a slice that has fallen back because of a missing chain from one that
has fallen back because of an arbitrage violation or an excessive
calibration residual. The transparency requirement is non-negotiable:
the no-silent-wrong-model standing directive of the project disallows
any branch that exposes a calibrated surface object without
identifying its provenance.

## 6. Term-structure assembly

The slices defined in the preceding sections constitute, individually,
a calibrated arbitrage-aware smile at one listed expiry. Downstream
applications, and in particular the surface-aware pricer of Section 8,
require the value of the implied volatility at arbitrary tenors that,
in general, do not coincide with any listed expiry. The present section specifies the assembly of the slice-level
surfaces into a single object that exposes the implied volatility as a
function of both strike and tenor.

The assembly procedure follows the standard linear-in-total-variance
recipe of Gatheral (2006, chapter 3). Let the calibrated SVI slices be
indexed by their tenors *T*<sub>1</sub> < *T*<sub>2</sub> < … < *T*<sub>N</sub>,
each carrying a parameter tuple
*θ*<sub>i</sub> = (a, b, ρ, m, ς)<sub>i</sub>. For a query tenor *T*
satisfying *T*<sub>i</sub> ≤ *T* ≤ *T*<sub>i+1</sub> the total implied
variance at log-moneyness *k* is defined as the convex combination

```math
w(k, T) \;=\; w(k, T_i) \;+\; \alpha \bigl[ w(k, T_{i+1}) - w(k, T_i) \bigr],
\qquad \alpha = \frac{T - T_i}{T_{i+1} - T_i},
```

with *w*(*k*, *T*<sub>i</sub>) = *w*<sub>SVI</sub>(*k*; *θ*<sub>i</sub>)
the total variance of the corresponding calibrated slice. The
implied volatility at the query is recovered as
*σ*(*k*, *T*) = sqrt(*w*(*k*, *T*) / *T*). The construction preserves
calendar arbitrage absence at every intermediate tenor whenever, at
every log-moneyness, the total variance of consecutive listed slices
is monotone non-decreasing in tenor. The monotonicity is verified at
construction of the surface on a dense log-moneyness grid spanning the
band most relevant to barrier products, and the location and
magnitude of every violation are recorded on the resulting object so
that the user interface can badge any affected term-structure region
with a quality warning. The audit is informational rather than
blocking: the surface remains usable in the presence of violations,
in the same loud-caveats spirit as the slice-level fallback policy.

Butterfly arbitrage absence at intermediate tenors is *not*
guaranteed by the construction even when the endpoint slices
individually satisfy the Durrleman condition, because the convex
combination of two arbitrage-free smiles is not in general
arbitrage-free in the butterfly direction. In practice the violation,
when it occurs, is small in magnitude and localised in moneyness; an
evaluation-time audit on a grid is acknowledged as a prerequisite for
the path-dependent pricers of Section 8.2, which consume the surface
at every Monte Carlo time step.

For a query tenor that falls outside the convex hull of the listed
expiries — either before the shortest listed tenor or after the
longest — no listed slice brackets the query and an extrapolation is
required. The convention adopted here is the *vol-flat* extension of
the anchor slice (the shortest listed expiry for downward
extrapolation, the longest for upward extrapolation): the implied
volatility at every log-moneyness is held constant in tenor, so that
the total variance scales linearly in tenor,

```math
\sigma(k, T) \;=\; \sigma(k, T_\mathrm{anchor})
\qquad \Longleftrightarrow \qquad
w(k, T) \;=\; w(k, T_\mathrm{anchor}) \cdot \frac{T}{T_\mathrm{anchor}}.
```

The convention is conservative on both ends of the term structure. At
short tenors the vol-flat extension understates the very steep
short-dated skew that listed markets typically exhibit and therefore
biases towards the at-the-money level of the anchor slice; at long
tenors it avoids the silently incorrect choice of extrapolating the
at-the-money variance into a regime where the listed market is silent
and where any linear projection would carry no informational
warrant. The user is informed by the surface status badge that any
value returned through this path is extrapolated, not interpolated.

The surface status taxonomy that the user interface exposes
complements the slice-level taxonomy of Section 5. Four values are
defined: *interpolated* when the query tenor sits strictly between
two listed expiries at which the surface has been calibrated;
*extrapolated* when the query tenor lies outside the convex hull of
the listed expiries; *single_slice* when only one calibrated slice
survives the arbitrage and quality gates of Section 5, in which case
the surface provides no genuine term-structure information and every
query is answered by vol-flat scaling of the single slice; and
*fallback* when no calibrated slice is available, in which case the
surface returns a configurable constant volatility regardless of the
query and the user interface is informed that no calibrated
information underlies the result.

The choice of linear-in-total-variance interpolation in preference to
a full SSVI re-calibration of the surface is deliberate. The SSVI
parameterisation of Gatheral and Jacquier (2014) imposes arbitrage
absence by construction across both the strike and the tenor
directions, but at the cost of a more constrained functional form
that, on the noisy listed chains characteristic of Yahoo Finance
coverage of single-name underlyings, fits each individual slice
materially less well than the raw SVI calibration of Section 3
attains. Preserving the per-slice fit quality is preferred in the
present context to the more parsimonious but globally fitted
alternative, with the residual arbitrage risk acknowledged and
audited rather than precluded.

## 7. Current limitations

Four limitations of the present implementation are recorded
explicitly.

The first is the use of the spot price as a proxy for the forward at
every slice tenor. The simplification is benign at the slice level
because the smile-translation parameter *m* absorbs any small offset
between the true forward and the spot, but it nevertheless distorts
the interpretation of "at the money" in log-moneyness terms by an
amount of order *(r − q) T*. Removing it requires the integration of
the existing risk-free rate term structure together with a dividend
yield estimator and is recorded as future work.

The second is the absence of an evaluation-time butterfly audit at
intermediate tenors. The linear-in-total-variance interpolation does
not in general preserve the Durrleman condition between the endpoint
slices, although in practice the violation magnitude is small. The
audit becomes economically material when the surface is queried at
every Monte Carlo time step under the local-volatility dynamics of
Section 8.2, and is also recorded as future work.

The third is the absence of skew calibration to listed exotics. The
surface is calibrated exclusively to listed vanilla chains; the skew
implied by listed structured product re-offer levels, which on a real
dealer desk acts as an independent calibration constraint, is not
incorporated. The omission is acceptable for a buy-side analytics
tool whose objective is transparent monitoring rather than
dealer-grade hedging, but the user is informed that the calibrated
marks will differ from a dealer's marks by an amount of order the
implicit cost of skew-trading the residual.

The fourth is the reliance on Yahoo Finance for the underlying option
chain. Yahoo provides dense and tight quotes on large-cap US singles,
the major US indices and a number of European indices, but its
coverage on European single-name underlyings and on tenors longer
than approximately two years is materially thinner. On those slices
the calibration falls back to the chain-proxy branch and the assembled
surface is correspondingly forced into the *single_slice* or
*fallback* regime at long tenors. The user is informed by the
surface status badge and the data feed will be evaluated against a
paid alternative (IVolatility, ORATS, OptionMetrics) when the
calibration coverage observed in production usage motivates the
expense.
 
## 8. Surface integration in the Monte Carlo pricer

The Monte Carlo pricer consumes the volatility surface in one of two
regimes, selected per product. European-barrier products are priced
under a *constant-σ regime* in which a single, surface-evaluated
scalar enters the dynamics; American-barrier, autocallable, and
issuer-callable products are priced under a *local-volatility
regime* in which the dynamics evolve under a state-dependent
volatility derived from the surface. The two regimes share the
same fallback policy and the same internal-consistency contract
between path-step and bridge-step.

### 8.1 Constant-σ regime with surface-evaluated input

The volatility consumed by the Monte Carlo pricer for any barrier
product is, for every underlying referenced in the product, the
volatility evaluated at the strike of that underlying's downside
barrier and at the product's residual maturity. Formally, for a
product with underlyings *i* = 1, …, *n*, initial fixing levels
*S*<sub>0,i</sub>, and barrier fraction *b* ∈ (0, 1], the per-underlying
volatility input is

```math
\sigma_i \;=\; \sigma\!\left(\, K_i,\; T_\mathrm{resid} \,\right),
\qquad
K_i \;=\; S_{0,i} \cdot b,
\qquad
T_\mathrm{resid} \;=\; \frac{T_\mathrm{maturity} - T_\mathrm{valuation}}{365.25},
```

with *σ*(*K*, *T*) read from the assembled surface of Section 6. The
Monte Carlo dynamics are otherwise unchanged: paths are generated
under geometric Brownian motion with constant volatility per
underlying, and the Brownian-bridge barrier-hit probability is
computed from the same scalar volatility input — preserving the
internal consistency between the path-step and the bridge-step.

For a *European-barrier* reverse convertible the substitution is
strictly correct. The payoff is observed only at maturity and depends
exclusively on the marginal risk-neutral distribution of the
underlyings at *T*<sub>maturity</sub>; under geometric Brownian motion
that marginal is identified by the volatility at the strike of the
barrier observation, so plugging *σ*(*K*<sub>barrier</sub>,
*T*<sub>maturity</sub>) into the constant-volatility Monte Carlo
recovers the barrier-hit probability that the listed option market
prices.

For *American-barrier*, *autocallable* and *issuer-callable* products,
the constant-σ regime would be a material improvement over an
at-the-money input — the barrier-zone volatility is the
economically relevant quantity for the dominant payoff feature —
but the path dependency of those payoffs is misrepresented by the
constant-volatility dynamics. The constant-σ Monte Carlo generates a
path that, between its initial and terminal date, is governed by a
single volatility rather than by the smile-implied (*S*, *t*)-dependent
local volatility. The error is one-sided in the direction of the
smile shape: a slice more skewed than the constant-σ assumption
implies a different distribution of the maximum drawdown along the
path than the constant-σ dynamics admit. Path-dependent products
are therefore routed to the local-volatility regime of Section 8.2.

The fallback policy mirrors the slice-level discipline of Section 5.
When the surface for a given underlying is unavailable, lies in the
*fallback* regime, or produces an out-of-range value, the
corresponding underlying reverts to a legacy at-the-money volatility
input. Every resolution path — surface or fallback — is recorded on
a per-underlying diagnostic structure so that the user interface can
badge the affected product with the provenance of its mark.

### 8.2 Local-volatility regime

For path-dependent products the pricer derives the Dupire local
volatility function from the assembled total-variance surface and
consumes it inside the Monte Carlo dynamics directly, evaluating the
volatility afresh at every (*S*, *t*) the path visits. The present
subsection specifies the derivation, the layered numerical safety
policy applied to the output, the interpretation that should attach
to extreme values, and the Monte Carlo integration through which the
local volatility enters the pricer.

The local volatility at strike *K* and tenor *T* is defined by the
Dupire identity in total-variance form (Gatheral 2006, equation 1.27):

```math
\sigma_\mathrm{LV}^2(K, T) \;=\; \frac{\partial_T w(k, T)}{\,1 \;-\; \tfrac{k}{w}\partial_k w \;+\; \tfrac{1}{4}\!\left(\!-\tfrac{1}{4} - \tfrac{1}{w} + \tfrac{k^2}{w^2}\!\right)(\partial_k w)^2 \;+\; \tfrac{1}{2}\partial_k^2 w\,},
```

where *k* = ln(*K* / *F*) is the log-moneyness against the surface's
forward and the partial derivatives of total variance are taken from
the assembled surface of Section 6. In the linear-in-total-variance
regime adopted for the term-structure assembly, all four partials
admit closed-form expressions: the strike derivatives ∂<sub>k</sub>w
and ∂²<sub>k</sub>w are the corresponding raw SVI derivatives,
linearly combined between bracketing slices in the interpolated
regime and scaled in tenor in the extrapolated regime; the tenor
derivative ∂<sub>T</sub>w is piecewise constant on each interval
between consecutive listed expiries and equals the slope of the
total-variance line in the extrapolated regime.

#### 8.2.1 Production safety thresholds and warning policy

The Dupire identity occasionally produces values that are
mathematically valid in form but economically extreme in size,
particularly on the deep out-of-the-money put wing of underlyings
whose listed implied surface exhibits a steep skew. The
implementation guards the output through a layered policy:

A **production hard cap** at two hundred per cent and a floor at one
per cent enforce numerical positivity and suppress genuine numerical
explosions. Values inside the interval are returned unchanged; values
outside are clipped and counted on the surface object's
``local_vol_clip_count`` attribute.

A **warning threshold** at one hundred per cent for the absolute
local volatility, and at three for the ratio of local to implied
volatility at the same point, records an informational event
without modifying the returned value. Warnings are an analytical
signal, not a numerical defect: a local volatility above the
threshold identifies a region in which the conditional dynamics of
the underlying are extreme, and informs the user interface that
the figure should be surfaced as such rather than treated as a
normal volatility input.

An **optional damping cap** allows the caller to apply a second,
tighter clip on top of the production hard cap. The damped output
is labelled in the user interface as a *conservative scenario* and
is distinguished from the pure Dupire output, which remains
available alongside it. The damped mode is intended for use when
mark-stability across days is preferred to theoretical purity.

#### 8.2.2 Interpretation of extreme local volatilities

A local volatility well in excess of one hundred per cent at a deep
out-of-the-money put strike should be read as a *conditional*
statement and not as an unconditional volatility input. The
quantity σ<sub>LV</sub>(*K*, *T*) is the instantaneous volatility
that the risk-neutral measure assigns to the underlying *when and
if* the underlying spot crosses into the corresponding region of
strikes at time *T*, given the calibrated implied surface. It is
not a statement about the volatility of the underlying observed
across all states of the world.

The distinction is significant for the interpretation of a
product's mark. A barrier-product fair value computed under such a
local-volatility surface reflects a steep conditional downside
dynamic that is a feature of the listed implied smile, not an
exotic assumption introduced by the pricer. On a name whose
implied smile prices the put wing at thirty-five volatility points
above the at-the-money level, the Dupire-implied conditional
volatility near the barrier strike can comfortably exceed one
hundred per cent without contradicting any economic information
already embedded in the chain; the local volatility merely
reorganises that information into the (*K*, *T*)-dependent form
required by the Monte Carlo stepping scheme.

#### 8.2.3 Monte Carlo integration

The local volatility is consumed by the Monte Carlo through three
components, each preserving the internal-consistency contract by
which every reader of the per-step volatility resolves it through
the same evaluator.

The first component is the step-by-step path generator
``simulate_paths_local_vol``. The path is evolved one business day
at a time under an Euler discretisation of geometric Brownian
motion with state-dependent variance,

```math
S_{j+1} \;=\; S_j \, \exp\!\left( \left( r - \tfrac{1}{2} \sigma_j^2 \right) \Delta t_j + \sigma_j \sqrt{\Delta t_j}\, Z_j \right),
\qquad \sigma_j \;=\; \sigma_\mathrm{LV}\!\left( S_j, \; t_j + \tfrac{1}{2}\Delta t_j \right),
```

the volatility being evaluated at the midpoint of each step to
avoid the *T* → 0 singularity of the Dupire identity at the very
first step. The per-step tensor ``sigma_path`` of shape
``(n_paths, n_steps, n_assets)`` is returned alongside the price
path so that downstream consumers can use precisely the values
that generated the path. The correlation between underlyings is
reproduced by the same Cholesky-decomposed correlated normal
increments that the constant-volatility path generator uses; only
the volatility scaling at each step differs.

The second component is the time-varying Brownian-bridge
barrier-hit probability. The bridge utility
``continuous_survival_prob_from_var`` is generalised to accept a
per-step variance tensor of either the constant-per-path shape
``(n_steps, n_assets)`` or the path-dependent shape
``(n_paths, n_steps, n_assets)`` returned by the local-volatility
generator. The variance assigned to the monitoring interval
between consecutive observations ``i`` and ``i + 1`` is
``sigma_path[:, i + 1, :]² · Δt_i``, identical to the variance
that generated the corresponding path step. Path-step and
bridge-step are thus identical by construction, which is the
structural protection against the implied-realised inconsistency
that motivated the migration.

The third component is the per-product dispatch
``_should_use_local_vol``. The local-volatility regime is
activated for a product when the product is path-dependent under
any of the autocallable, American-barrier or issuer-callable
specialisations *and* at least one of its underlyings carries a
non-fallback surface. European-barrier products always remain on
the constant-σ regime of Section 8.1 — for them the substitution is
strictly correct and the local-volatility generator would add
expense without changing the result. Products whose underlyings
are all in the surface-level fallback regime likewise remain on
the constant-σ path, with the legacy at-the-money volatility
input as the volatility scalar.

## 9. References

De Marco, S. and Martini, C. (2009). *Quasi-explicit calibration of
Gatheral's SVI model*. Zeliade Systems white paper, Paris.

Durrleman, V. (2010). *From implied to spot volatilities*. Finance and
Stochastics 14 (2), 157–177.

Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility
parameterization with application to the valuation of volatility
derivatives*. Presentation, Global Derivatives & Risk Management,
Madrid.

Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.*
Wiley, Hoboken NJ.

Gatheral, J. and Jacquier, A. (2014). *Arbitrage-free SVI volatility
surfaces*. Quantitative Finance 14 (1), 59–71.

Lee, R. W. (2004). *The moment formula for implied volatility at
extreme strikes*. Mathematical Finance 14 (3), 469–480.
