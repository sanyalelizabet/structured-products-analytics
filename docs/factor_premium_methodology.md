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

The market timeline is partitioned, once and jointly, into the three regimes by
reference to the market factor alone; the same daily classification is shared by
every factor, and classification is never performed factor by factor. Two
characteristics of the market are taken into account.

First, a *trend* characteristic is obtained from the trailing twelve-month
cumulative MKT log-return. A day is provisionally assigned to the bear, flat, or
bull band according to whether this quantity falls below −5%, between −5% and
+10%, or at or above +10%, respectively.

Second, a *stress* characteristic is introduced so that abrupt drawdowns, which
the slow trailing-return measure is too sluggish to register, are not
mis-classified. A day is deemed to be under *down-stress* when volatility is
elevated — the realised one-month annualised volatility of the market, or an
externally supplied gauge such as the VIX, exceeding 25% — and the recent
one-month return is simultaneously negative. The requirement that the recent
return be negative is essential: volatility is direction-agnostic, and were
high-volatility rebound days admitted to the bear regime they would contaminate
its conditional mean with positive returns.

The two characteristics are combined as follows. A day is classified as bear if
it lies in the bear trend band or is a down-stress day; as bull only if it lies
in the bull trend band and volatility is not elevated, so that a turbulent
melt-up is treated as flat rather than as a clean bull; and as flat otherwise.
The volatility signal is configurable, which permits a forward-looking measure
to be substituted for the realised default without altering the surrounding
logic.

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
