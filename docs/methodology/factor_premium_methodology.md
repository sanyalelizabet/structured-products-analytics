# Estimation of Regime-Conditional Factor Premiums

## 1. Purpose and definition

Within the factor-stress path simulator, each systematic risk factor is
propagated forward as a stochastic process whose deterministic component — its
drift — is held fixed over the pre-shock horizon. The quantity that supplies
this drift is referred to throughout as the *factor premium*.

A factor premium is defined here as the assumed annualised expected
log-return of a factor, conditional on the prevailing market regime:

&nbsp;&nbsp;&nbsp;&nbsp;*μ<sub>f,R</sub> = E[ r<sub>f</sub> | regime = R ]*, annualised,

for factor *f* ∈ {MKT, TECH, HC, FIN, ENERGY, FX} and regime
*R* ∈ {bear, flat, bull}. It is emphasised that this object is a
*physical-measure scenario assumption* — the drift that is assumed to prevail
*if* the market is judged to be in regime *R* — and not a forecast of realised
returns. The distinction matters: the estimator is therefore evaluated by the
plausibility and internal consistency of the assumptions it produces, rather
than by out-of-sample predictive accuracy.

The premiums are assembled into a regime × factor table. At simulation time the
regime selected by the user (the "initial market state") indexes a single row,
and the corresponding per-factor vector is adopted as the pre-shock drift of the
factor paths.

Estimation is conducted over a five-year window of daily history
(`ESTIMATION_LOOKBACK_YEARS = 5`). The choice is a compromise between sample
size and relevance: a five-year span is long enough to encompass a diversity of
market conditions — including a sustained bear episode — so that each regime is
populated by direct observation rather than by the prior alone, yet recent
enough that the estimated behaviour remains representative of the prevailing
market structure.

## 2. Regime classification

Each trading day is assigned to exactly one of the three regimes by reference to
the market factor alone. The classification is computed once from MKT and is
shared unchanged by every factor. The assignment rests on three daily
measurements of the market.

The first measurement is the *trend*, the twelve-month cumulative MKT
log-return.

The second is a *down-stress* condition, which holds when market volatility is
elevated — the realised one-month annualised volatility of the market, or the
VIX if provided, exceeding 20%. The recent one-month return is additionally
taken into account.

The conjunction with a negative recent return is deliberate: volatility is
direction-agnostic, and were high-volatility rebound days admitted to the bear
regime they would contaminate its conditional mean with positive returns.

A day is assigned to the bear regime whenever its trend falls below −5% or the
down-stress condition holds; failing that, to the bull regime when its trend is
at least +10% and volatility is not elevated; and to the flat regime otherwise.
Resolving the bear condition first is what allows an abrupt drawdown to be
recognised even before the slow trailing-return trend has turned — the very
transition that the trend measure, on its own, is too sluggish to register. The
calm requirement attached to the bull regime ensures, symmetrically, that a
turbulent melt-up, a market rising under elevated volatility, is treated as flat
rather than as a clean bull. The volatility signal is configurable, so that a
forward-looking measure may be substituted for the realised default without
altering the surrounding logic.

Formally, let $r^{12\mathrm{m}}_t$ denote the trailing twelve-month cumulative
MKT log-return, $r^{1\mathrm{m}}_t$ the trailing one-month return, and
$\sigma_t$ the volatility signal (realised one-month annualised volatility, or
the VIX if supplied), with stress threshold $\sigma^\star = 20\%$. Define the
elevated-volatility and down-stress indicators

```math
V_t = \mathbf{1}\left[\sigma_t > \sigma^\star\right],
\qquad
D_t = V_t \cdot \mathbf{1}\left[r^{1\mathrm{m}}_t < 0\right].
```

The regime of day $t$ is then

```math
\mathrm{regime}(t) =
\begin{cases}
\text{bear}, & \text{if } r^{12\mathrm{m}}_t < -5\% \ \text{ or } \ D_t = 1, \\
\text{bull}, & \text{else if } r^{12\mathrm{m}}_t \geq +10\% \ \text{ and } \ V_t = 0, \\
\text{flat}, & \text{otherwise.}
\end{cases}
```

The ordering of the cases encodes the precedence: the bear condition is tested
first and therefore overrides an otherwise-bullish trend on a down-stress day.

The rule is illustrated by the representative cases below, in which the trend is
the trailing twelve-month MKT return, "elevated" denotes volatility above the
20% threshold, and the recent return is the trailing one-month figure that
enters the down-stress condition.

| Trend (12m) | Volatility | Recent return (1m) | Regime | Rationale |
|---|---|---|---|---|
| +15% | not elevated | +2% | bull | strong trend under calm conditions |
| +15% | elevated | +2% | flat | rising but turbulent — not a clean bull |
| +3% | not elevated | +1% | flat | trend in the central band |
| −8% | not elevated | −1% | bear | trend below the −5% bear threshold |
| +12% | elevated | −4% | bear | down-stress holds and takes precedence over the bull trend |
| +1% | elevated | +3% | flat | turbulent but rising, so not down-stress, and the trend is too weak for bull |

The fifth and sixth rows isolate the role of the down-stress condition: an
elevated-volatility day is classified as bear only when the recent return is
also negative, and a high trend is overridden in that event, whereas an
elevated-volatility day with a positive recent return carries no stress
implication and is judged on trend alone.

## 3. Estimation of the conditional premium

Both estimators that are implemented target the same quantity *μ<sub>f,R</sub>*
defined in Section 1; they differ only in the extent to which the in-sample
conditional mean is trusted relative to a structural prior. They are most
naturally understood as the two limiting cases, and the interpolation between
them, of a single shrinkage estimator.

Let *n<sub>R</sub>* denote the number of days assigned to regime *R*, let
*r̄<sub>f,R</sub>* denote the sample mean of factor *f*'s daily log-returns over
those days, annualised by the factor 252, and let the clipping operator restrict
any annualised drift to the interval ±25% so that short, concentrated episodes
cannot produce implausible figures.

**The conditional-mean estimator.** In its simplest form the premium is taken to
be the regime-conditional sample mean,

&nbsp;&nbsp;&nbsp;&nbsp;*μ̂<sub>f,R</sub> = clip( r̄<sub>f,R</sub> )*.

Because the sample mean of a drift is estimated with considerable error when few
observations are available, a regime in which fewer than sixty days are observed
is not estimated from data at all; the entire row is instead set to a single
regime-level equity-risk-premium scalar, *ERP<sub>R</sub>* (−10% in bear, 0% in
flat, +12% in bull). This estimator is transparent but has two known
shortcomings: it discards the cross-section of factors whenever it falls back to
the scalar, and it remains noisy near the threshold.

**The shrinkage estimator.** The preferred estimator replaces the sample mean by
a convex combination of that mean and a structural, factor-specific prior. The
prior is constructed in the spirit of the single-factor capital-asset-pricing
relationship: each factor's premium in a regime is anchored to the regime's
equity-risk-premium scaled by the factor's market sensitivity,

&nbsp;&nbsp;&nbsp;&nbsp;*prior<sub>f,R</sub> = β<sub>f</sub> · ERP<sub>R</sub>*,

where *β<sub>f</sub> = Cov(r<sub>f</sub>, r<sub>MKT</sub>) / Var(r<sub>MKT</sub>)*
is estimated over the full sample, with *β<sub>MKT</sub>* identically one. The
premium is then

&nbsp;&nbsp;&nbsp;&nbsp;*μ̂<sub>f,R</sub> = clip( w<sub>R</sub> · r̄<sub>f,R</sub> + (1 − w<sub>R</sub>) · prior<sub>f,R</sub> )*,
&nbsp;&nbsp;&nbsp;&nbsp;*w<sub>R</sub> = n<sub>R</sub> / (n<sub>R</sub> + n<sub>0</sub>)*,

with the prior strength fixed at *n<sub>0</sub> = 252*. The weight is increasing
in the number of regime observations: a data-rich regime (*n<sub>R</sub> ≫
n<sub>0</sub>*) is governed by its sample mean, whereas a data-poor regime
(*n<sub>R</sub> → 0*) reverts to the prior.

**Relationship between the two.** The connection is that both are particular
points of the same convex combination. The conditional-mean estimator
corresponds to the corner *w<sub>R</sub> = 1* (full trust in the data), the pure
structural prior to the corner *w<sub>R</sub> = 0*, and the shrinkage estimator
to an interior weight determined by the sample size. The decisive practical
difference is the behaviour of a thinly populated regime: the conditional-mean
estimator collapses to a *single scalar applied uniformly across all factors*,
whereas the shrinkage estimator collapses to *β<sub>f</sub> · ERP<sub>R</sub>*,
which remains differentiated across factors — a high-beta factor is assigned a
deeper drawdown in a bear regime than a defensive one. For this reason the
shrinkage estimator is recommended, and the conditional-mean estimator is
retained chiefly as a transparent baseline for comparison.

The prior strength of one year reflects the recognition that an annualised drift
inferred from materially fewer than a year of daily observations is dominated by
estimation noise (the amplification by the factor 252 being the proximate
cause); the prior is therefore allowed to govern until roughly a year of
regime-specific evidence has accumulated.

## 4. Output and use

The estimation yields a regime × factor table of annualised drifts, which is
cached separately for each estimator. The market regime chosen at simulation
time selects one row of the table; the resulting per-factor drift vector is
adopted as the deterministic pre-shock drift of the simulated factor paths, from
which asset paths are obtained through the estimated factor loadings. The Factor
Stress view exposes the choice of estimator and displays both tables so that the
data-driven and prior-driven figures may be compared directly.

## 5. Assumptions and limitations

The methodology rests on a small set of assumptions that are stated explicitly
so that they may be reviewed and, where appropriate, recalibrated. The regime
boundaries and the volatility threshold are deliberate, judgemental choices, and
the equity-risk-premium anchors *ERP<sub>R</sub>* are likewise assumptions
rather than estimates. The influence of these anchors is greatest when a regime
is thinly populated, in which case the shrinkage weight is small and the prior
predominates; conversely, the five-year estimation window is intended to ensure
that each regime — including the bear regime, which the window's coverage of a
sustained downturn populates by direct observation — is supported by sufficient
data for the conditional mean to carry meaningful weight. It is nonetheless
acknowledged that drift estimation remains intrinsically noisy, that the anchors
continue to govern any regime that is sparsely represented in a given sample,
and that the estimates should accordingly be read as scenario assumptions rather
than as precise measurements.

## 6. Parameter summary

| Symbol / constant | Value | Role |
|---|---|---|
| `ESTIMATION_LOOKBACK_YEARS` | 5 | history window for estimation |
| Regime trend bands | bear `<−5%`, flat `[−5%,+10%)`, bull `≥+10%` | trailing-12m MKT return cut-offs |
| `TRAILING_WINDOW_DAYS` | 252 | trend window |
| `VOL_WINDOW_DAYS` | 21 | realised-volatility / recent-return window |
| `STRESS_VOL_THRESHOLD` | 0.25 | annualised volatility defining stress |
| `REGIME_ERP` | bear −0.10, flat 0.00, bull +0.12 | regime equity-risk-premium (prior anchor / mean-method fallback) |
| `MIN_OBS_PER_REGIME` | 60 | data threshold below which the mean estimator reverts to the ERP scalar |
| `SHRINKAGE_PRIOR_STRENGTH` (*n<sub>0</sub>*) | 252 | prior strength in the shrinkage weight |
| `DRIFT_CLIP_BAND` | ±0.25 | admissible range for an annualised drift |

## References

- W. F. Sharpe (1964), *Capital Asset Prices*, Journal of Finance 19(3).
- R. C. Merton (1980), *On Estimating the Expected Return on the Market*,
  Journal of Financial Economics 8(4).
- C. Stein (1956); W. James and C. Stein (1961), on shrinkage estimation of means.
- F. Black and R. Litterman (1992), *Global Portfolio Optimization*, Financial
  Analysts Journal 48(5).
