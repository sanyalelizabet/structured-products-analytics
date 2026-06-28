# Single-Factor Stress-Scenario Methodology

## 1. Purpose and scope

This note documents the single-factor stress engine (`src/scenario_engine.py`),
which underlies the *Stress Testing* view. The engine evaluates a portfolio of
structured products against a user-specified market scenario by Monte Carlo
simulation, and reports the resulting profit-and-loss (P&L) distribution at the
product, currency, and reference-currency levels.

The term *single-factor* refers to the source of systematic risk: each
underlying is driven by the broad equity market through its market beta, and a
scenario is expressed as a shock to that single market factor together with an
assumed drift regime. The richer decomposition into multiple style and sector
factors is treated separately (see the multi-factor methodology).

It is emphasised that the simulation is a *physical-measure* what-if exercise:
the drifts and shocks are scenario assumptions chosen by the analyst, and the
output is to be read as the conditional distribution of P&L under those
assumptions rather than as a risk-neutral valuation.

## 2. Notation

For an underlying indexed by $i$ within a given product, let $S_{i,t}$ denote
its price at time $t$, $\sigma_i$ its annualised volatility, and $\beta_i$ its
market beta. The risk-free rate of the product's currency is denoted $r_f$, the
assumed market drift $\mu_m$, and the speed of mean reversion $\kappa$. The
correlation matrix of the product's underlyings is written $\rho$, with lower
Cholesky factor $L$ satisfying $LL^\top = \rho$. Time is measured in years on an
Actual/360 basis, and $\Delta t$ denotes the year fraction of a simulation step.

## 3. Price dynamics

Each underlying is modelled as a one-factor Schwartz mean-reverting geometric
Brownian motion in the logarithm of price. Writing $X_{i,t} = \ln S_{i,t}$ and
letting $\theta_{i,t}$ denote a (stochastically drifting) log-price target, the
dynamics are

$$
\mathrm{d}X_{i,t} = \Big[\big(\mu_i - \tfrac{1}{2}\sigma_i^2\big)
      + \kappa\,(\theta_{i,t} - X_{i,t})\Big]\,\mathrm{d}t
      + \sigma_i\,\mathrm{d}W_{i,t},
$$

where the instantaneous drift is supplied by the capital-asset-pricing
relationship,

$$
\mu_i = r_f + \beta_i\,(\mu_m - r_f).
$$

Contemporaneous dependence across the underlyings of a product is imposed
through the correlation of the Brownian increments,
$\mathrm{d}W_t = L\,\mathrm{d}B_t$ with $\mathrm{d}B_t$ a vector of independent
standard increments, so that $\operatorname{Corr}(\mathrm{d}W_t) = \rho$. The
mean-reversion term draws the log-price toward the moving target
$\theta_{i,t}$, which itself advances at the deterministic drift; in the limit
$\kappa \to 0$ the process reduces to ordinary geometric Brownian motion.

## 4. Time discretisation

The horizon is the business-day grid from the valuation date to the latest
portfolio maturity. The continuous dynamics are advanced by an Euler–Maruyama
step. With $Z_t = L z_t$ a vector of correlated standard-normal draws, the
target and the log-price evolve as

$$
\ln\theta_{i,t} = \ln\theta_{i,t-1} + \big(\mu_i - \tfrac{1}{2}\sigma_i^2\big)\,\Delta t,
$$

$$
X_{i,t} = X_{i,t-1} + \big(\mu_i - \tfrac{1}{2}\sigma_i^2\big)\,\Delta t
      + \kappa\,(\ln\theta_{i,t-1} - X_{i,t-1})\,\Delta t
      + \sigma_i\sqrt{\Delta t}\;Z_{i,t}.
$$

Prices are recovered as $S_{i,t} = \exp(X_{i,t})$. The half-life of the
mean-reverting component is $\ln 2 / \kappa$, and the stationary log-price
dispersion implied by the specification is $\sigma_i / \sqrt{2\kappa}$.

## 5. Drift-regime schedule

The market drift $\mu_m$ is piecewise constant in time, partitioned by the
scenario's shock schedule. A pre-shock drift applies up to the first shock; a
post-shock (recovery) drift applies thereafter. When a finite recovery horizon
is specified, the drift reverts to a post-recovery rate — typically the ambient
market assumption — once that horizon has elapsed, which prevents a steep
recovery drift from compounding indefinitely.

## 6. Discrete shocks

A scenario shock is applied multiplicatively at the business day nearest each
scheduled shock date, scaled by each underlying's beta. For a market shock of
$s$ per cent, the log-price and its target are displaced by

$$
\Delta X_{i} = \ln\!\Big(\max\big(1 + \tfrac{s}{100}\,\beta_i,\;\varepsilon\big)\Big),
$$

with a small floor $\varepsilon$ guarding against non-positive multipliers. The
target $\theta$ is displaced identically, so that the post-shock mean-reversion
pulls toward the shocked level rather than unwinding the shock.

## 7. Valuation and aggregation

For each simulated path the terminal underlying prices at the product's maturity
are passed to the appropriate payoff (barrier reverse convertible, worst-of,
autocallable, or capital-protection note), yielding a per-path P&L and return.
The product-level distribution is summarised by its mean, median, the 5th and
95th percentiles, the 5 % expected shortfall, and the standard deviation.

Where a barrier product is observed continuously (American observation), the
knock-in is determined over the whole simulated path rather than at maturity
alone, using the Brownian-bridge correction with this engine's own per-step
diffusion variance and sampling one definite outcome per path so the loss
distribution is preserved; see the barrier-observation methodology note. European
observation (the default) uses the terminal fixing only.

Portfolio aggregation is performed by summing the per-path P&L vectors across
products within each currency, preserving the dependence structure induced by
the shared random draws. Where a reference currency and a table of exchange
rates are supplied, the per-currency P&L vectors are converted and summed
path-by-path into a single reference-currency distribution; the foreign-exchange
convention matches that of the portfolio analytics layer.

## 8. Common random numbers

All randomness originates from a shared noise sampler; the engine itself seeds
no generators. Holding the underlying Gaussian draws fixed across runs ensures
that differences between scenarios reflect the change in scenario parameters
alone, which is the basis of clean sensitivity and what-if comparison. This use
of common random numbers is the standard variance-reduction device for
simulation-based sensitivity analysis (Glasserman, 2004, ch. 7). A fresh draw is
obtained only when the caller explicitly requests regeneration.

## 9. Assumptions and limitations

The methodology rests on several deliberate simplifications. Systematic risk is
carried by a single market factor through beta, so sector- or style-specific
co-movements are not represented. Volatilities, betas, and correlations are
held constant over the horizon and are taken from historical estimation.
Innovations are Gaussian, and the mean-reverting log-normal specification
excludes jumps beyond the prescribed discrete shocks. When correlation data are
unavailable for a product's underlyings the identity matrix is substituted,
which treats those underlyings as independent. These choices are appropriate for
a transparent stress tool but should be borne in mind when interpreting tail
statistics.

## References

- E. S. Schwartz (1997), *The Stochastic Behavior of Commodity Prices*,
  Journal of Finance 52(3).
- W. F. Sharpe (1964), *Capital Asset Prices*, Journal of Finance 19(3).
- P. Glasserman (2004), *Monte Carlo Methods in Financial Engineering*,
  Springer — common random numbers, ch. 7.
