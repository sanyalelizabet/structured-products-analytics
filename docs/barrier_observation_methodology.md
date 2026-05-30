# Barrier Observation Methodology

## 1. Purpose and scope

This note documents how the down-barrier of a barrier reverse convertible
(BRC, and its multi-underlying and autocallable variants) is *observed* in the
analytics, and how the observation convention enters both the risk-neutral fair
value (`src/pricing/monte_carlo.py`) and the physical-measure stress engines
(`src/scenario_engine.py`, `src/factor_scenario_engine.py`).

A barrier reverse convertible is, in economic terms, a coupon-bearing note
combined with a short down-and-in put on the worst-performing underlying. The
holder receives par at maturity unless the barrier is breached, in which case
redemption converts to the worst underlying's performance. The **observation
convention** determines *when* a breach may occur, and is therefore a
first-order driver of the value of the embedded put.

Two conventions are supported, selected per product by the `type_style` field:

* **European** — the barrier is tested only at final fixing. A breach occurs iff
  some underlying's terminal level is at or below its barrier.
* **American (continuous)** — the barrier is live throughout the product's life.
  A breach occurs if any underlying trades at or below its barrier at any time.

Because continuous observation admits every breach that the terminal test would
detect and more, the American convention assigns the embedded put at least as
much value as the European one; the note is therefore worth no more, and its
stress losses are no smaller.

## 2. Notation

For an underlying indexed by $i$, let $S_{i,t}$ be its price, $K_i$ its strike,
and $B_i = L_i \cdot b$ its down-barrier, where $L_i$ is the initial fixing
level and $b$ the contractual barrier fraction (`barrier_pct`). The simulation
proceeds on a business-day grid $t_0 < t_1 < \dots < t_n$ with step year
fractions $\Delta t_j = (t_j - t_{j-1})/360$ (Actual/360). The model variance of
the log-price increment of underlying $i$ over step $j$ is written
$v_{ij} = \operatorname{Var}\big(\ln S_{i,t_j} - \ln S_{i,t_{j-1}}\big)$.

## 3. The discrete-monitoring bias and its correction

A Monte Carlo path is known only at the grid points. Declaring a breach only
when a *grid* value falls at or below the barrier systematically **understates**
the knock-in probability: between two consecutive grid points the continuous
diffusion may cross the barrier and return above it, leaving both endpoints
above the barrier yet the path having breached. Naïve grid monitoring misses
exactly these excursions, and the resulting fair value is biased upward.

The exact correction is the **Brownian-bridge** crossing probability. In the
logarithm of price the increment is Gaussian, so conditional on the two endpoint
log-levels the path over a step is a Brownian bridge. For a down-barrier with
both endpoints strictly above it, the probability that the bridge never touches
the barrier over the step is, by the reflection principle,

```math
q_{ij} \;=\; 1 - \exp\!\left(-\,\frac{2\,\ln(S_{i,t_{j-1}}/B_i)\,\ln(S_{i,t_j}/B_i)}{v_{ij}}\right),
\qquad S_{i,t_{j-1}},\,S_{i,t_j} > B_i .
```

If either endpoint is at or below the barrier the step is a certain knock-in and
$q_{ij} = 0$. The numerator is the product of the two endpoint log-distances to
the barrier, and the denominator is the step's diffusion variance: the closer
either endpoint sits to the barrier, or the larger the variance, the higher the
crossing probability.

## 4. Worst-of survival over the life

A worst-of barrier knocks in if *any* underlying crosses *its* barrier at *any*
time. Increments are independent across steps and the survival of each
underlying is required jointly, so the probability that the product survives the
whole life without knocking in is the product of the per-step factors over both
the time and the asset axes,

```math
P_{\text{surv}} \;=\; \prod_{j=1}^{n}\;\prod_{i=1}^{m} q_{ij},
```

evaluated per simulated path. This quantity is computed by
`barrier.continuous_survival_prob_from_var`.

## 5. Use of the model's own variance

The bridge requires the per-step diffusion variance $v_{ij}$. It is taken to be
the variance the path-generating model itself assumes — never re-estimated from
the realised paths — so that the barrier observation is exactly consistent with
the dynamics that produced the prices.

**Single-factor stress engine.** The diffusion term is $\sigma_i\sqrt{\Delta t}$
in the log-price, the mean-reversion and discrete shocks being drift. Hence

```math
v_{ij} = \sigma_i^{2}\,\Delta t_j .
```

**Multi-factor stress engine.** Each underlying's return is a systematic part,
$\beta_i^\top f$, plus an idiosyncratic part. With $\Sigma_f$ the annualised
factor covariance the engine uses,
$\Sigma_f = \operatorname{diag}(\sigma_f)\,\mathrm{Corr}_f\,\operatorname{diag}(\sigma_f)$,
the systematic variance is $\beta_i^\top \Sigma_f \beta_i$ per unit time, and the
idiosyncratic variance is $(\lambda\,\sigma_i^{\text{idio}})^2$ injected per daily
step (intensity $\lambda$, the engine's fixed $1/\sqrt{252}$ step). Hence

```math
v_{ij} \;=\; \big(\beta_i^\top \Sigma_f\, \beta_i\big)\,\Delta t_j \;+\; \frac{\big(\lambda\,\sigma_i^{\text{idio}}\big)^2}{252}.
```

**Risk-neutral fair value.** The geometric Brownian motion uses a constant per
underlying volatility, so $v_{ij} = \sigma_i^2 \Delta t_j$ as in the
single-factor case.

## 6. From survival probability to outcome

The two consumers of $P_{\text{surv}}$ differ deliberately, because they answer
different questions.

**Fair value — an expectation.** The fair value is the discounted expected
payoff. Conditioning on each simulated path, the knock-in is integrated out
analytically: redemption is the survival-weighted blend of the par and converted
outcomes,

```math
\text{redemption} \;=\; P_{\text{surv}}\cdot \text{par} \;+\; \big(1 - P_{\text{surv}}\big)\cdot \text{notional}\times \text{worst-of performance},
```

which is a Rao–Blackwellisation of the binary knock-in indicator. Using the
probability rather than a sampled indicator removes avoidable Monte Carlo
variance and, being deterministic given the paths, leaves the bump-and-reprice
Greeks stable under common random numbers.

**Stress — a distribution.** The stress engines report a P&L *distribution*
(percentiles, expected shortfall, breach frequency), not a single expectation.
Blending the payoff by $P_{\text{surv}}$ would replace each path's outcome with a
probability-weighted average and so erase the breached/not-breached bimodality
that drives the downside tail. Each path is therefore assigned a **definite**
outcome by drawing one uniform $u \sim U(0,1)$ and declaring a knock-in when

```math
u \;>\; P_{\text{surv}} .
```

This samples knock-ins at exactly the bridge-correct rate, so the expected breach
frequency matches the fair-value model while the full outcome distribution is
preserved. The uniforms are drawn from the engine's `NoiseSampler`
(`knock_in_uniform`), keyed by product, so stress runs remain reproducible under
common random numbers and refresh together with the path noise on regeneration.

## 7. Autocallables under continuous observation

For an autocallable barrier reverse convertible the early-redemption logic is
unaffected by the observation style: a path that reaches an autocall trigger is
redeemed at par plus accrued coupon, irrespective of the capital barrier.
Continuous observation therefore concerns only the paths that **run to
maturity** — the uncalled paths — whose knock-in is determined over their full
trajectory exactly as in Sections 3–6. In fair value the uncalled paths' breach
is sampled against the bridge survival probability with uniforms fixed by the
pricer seed and product, so the bump-and-reprice Greeks reuse identical draws;
in stress it is sampled from the engine's sampler. When the product has no
autocall observation dates, every path is uncalled and the instrument reduces to
a continuously-observed barrier reverse convertible.

## 8. Assumptions and limitations

* Monitoring is performed over the simulated grid points; the sub-interval
  between today's spot and the first grid date (a single business day) is not
  monitored, which is immaterial to the knock-in probability.
* On the deterministic, flat-spot product projection there is no intermediate
  path, so American and European observation coincide and reduce to the terminal
  test.
* For autocallables, the continuous-observation knock-in of the uncalled paths
  is *sampled* (one definite outcome per path) rather than taken in expectation,
  consistent with the path-event nature of the autocall itself; the Greeks of a
  continuously-observed autocallable are therefore marginally less smooth than
  those of a plain barrier note.

## 9. References

Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
(Brownian-bridge barrier corrections, ch. 6; common random numbers, ch. 7.)
