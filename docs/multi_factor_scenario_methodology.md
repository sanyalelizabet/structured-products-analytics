# Multi-Factor Stress-Scenario Methodology

## 1. Purpose and scope

This note documents the multi-factor stress engine
(`src/factor_scenario_engine.py`), which underlies the *Factor Stress* view. The
engine simulates a portfolio of structured products against a scenario expressed
as an *event timeline* over a small set of systematic risk factors, and reports
the resulting profit-and-loss (P&L) distribution.

Whereas the single-factor engine carries systematic risk through a single market
beta, the present engine resolves it into six liquid factors — broad equity
(MKT), the technology, healthcare, financials and energy sectors, and a
USD/CHF exchange-rate factor — and projects them onto each underlying through
its estimated factor loadings. The construction is in the spirit of a linear
multi-factor return model (Ross, 1976; Sharpe, 1964): factor paths are simulated
first, and asset behaviour is obtained as a loading-weighted combination of the
factors plus an idiosyncratic residual.

As with the single-factor engine, the exercise is a physical-measure what-if
analysis; the factor drifts, shocks, and recovery dynamics are scenario
assumptions, and the output is the conditional P&L distribution under them.

## 2. Notation

Let the factor index be $k \in \{1,\dots,K\}$ and the underlying index be $i$.
The factor log-level is $G_{k,t} = \ln F_{k,t}$, with annualised factor
volatility $\sigma_k$ and assumed annualised drift $\mu_k$. The factor
correlation matrix is $\Omega$ with lower Cholesky factor $L_F$
($L_F L_F^\top = \Omega$). For underlying $i$, $\beta_{i,k}$ denotes its loading
on factor $k$, $\sigma_i^{\mathrm{id}}$ its idiosyncratic volatility, and
$\lambda \in [0,1]$ a global scaling applied to the idiosyncratic component.
Time is measured in years and $\Delta t$ is the year fraction of a step
(Actual/360 for the factor dynamics).

## 3. Factor dynamics

Each factor follows a correlated Schwartz mean-reverting geometric Brownian
motion in its logarithm,

$$
\mathrm{d}G_{k,t} = \Big[\big(\mu_k - \tfrac{1}{2}\sigma_k^2\big)
      + \kappa\,(\Theta_{k,t} - G_{k,t})\Big]\,\mathrm{d}t
      + \sigma_k\,\mathrm{d}W_{k,t},
$$

where $\Theta_{k,t}$ is a deterministically drifting log-level target and the
Brownian increments are correlated according to $\Omega$, that is
$\mathrm{d}W_t = L_F\,\mathrm{d}B_t$ for independent increments
$\mathrm{d}B_t$. The drift vector $\mu_k$ is the regime-conditional factor
premium; its estimation — the partition of history into bear, flat, and bull
regimes and the shrinkage estimator that produces the per-factor drifts — is
documented separately in the factor-premium methodology and is taken here as a
given input.

## 4. Event timeline and drift segments

A scenario is specified as an initial per-factor drift vector together with an
ordered sequence of dated events. Each event carries a vector of multiplicative
factor shocks and, optionally, a new drift vector that governs the segment
following the event, accompanied by a finite recovery horizon. The active drift
is therefore piecewise constant: it begins at the initial vector, switches at
each event to that event's post-event drift, and reverts to the initial drift
once the event's recovery horizon has elapsed. A partial post-event drift
overrides only the factors it names, leaving the remainder unchanged.

## 5. Time discretisation

The factor block is advanced on the business-day grid from the valuation date to
the latest portfolio maturity by an Euler–Maruyama step. With $Z_t = L_F z_t$ a
vector of correlated standard-normal draws, the target and log-level update as

$$
\Theta_{k,t} = \Theta_{k,t-1} + \big(\mu_k - \tfrac{1}{2}\sigma_k^2\big)\,\Delta t,
$$

$$
G_{k,t} = G_{k,t-1} + \big(\mu_k - \tfrac{1}{2}\sigma_k^2\big)\,\Delta t
      + \kappa\,(\Theta_{k,t-1} - G_{k,t-1})\,\Delta t
      + \sigma_k\sqrt{\Delta t}\;Z_{k,t}.
$$

A scheduled event applies a multiplicative shock at the nearest business day:
for a factor shock of $s_k$ per cent the log-level and its target are displaced
by $\ln\!\big(\max(1 + s_k/100,\,\varepsilon)\big)$. The factor daily
log-returns used in the projection step are the first differences of the
simulated log-levels, $r^{F}_{k,t} = G_{k,t} - G_{k,t-1}$.

## 6. Projection onto assets

Asset returns are obtained from the factor returns through the linear factor
model. For underlying $i$ the daily log-return is

$$
r_{i,t} = \sum_{k=1}^{K} \beta_{i,k}\,r^{F}_{k,t}
      + \lambda\,\sigma_i^{\mathrm{id}}\,\sqrt{\Delta t_{\mathrm{id}}}\;\varepsilon_{i,t},
$$

where $\varepsilon_{i,t}$ are independent standard-normal idiosyncratic draws and
$\Delta t_{\mathrm{id}} = 1/252$ is the daily step used for the residual scaling.
The systematic component is the loading-weighted sum of factor returns; the
idiosyncratic component is scaled by the intensity $\lambda$, which interpolates
between a purely deterministic factor projection ($\lambda = 0$) and the full
historical residual volatility ($\lambda = 1$).

The regression intercept (alpha) is deliberately excluded from the forward
projection: the systematic drift is supplied entirely by the factor premiums of
Section 3, and re-introducing a historically fitted alpha would double-count the
trend. Asset price paths follow by cumulating the projected returns from the
current spot,

$$
\ln S_{i,t} = \ln S_{i,0} + \sum_{u \le t} r_{i,u}, \qquad S_{i,t} = \exp(\ln S_{i,t}).
$$

## 7. Valuation and aggregation

For each path the terminal underlying prices at a product's maturity are passed
to the appropriate payoff (barrier reverse convertible, worst-of, autocallable,
or capital-protection note). The resulting per-path P&L is summarised at the
product level by its mean, median, the 5th and 95th percentiles, the 5 %
expected shortfall, and the standard deviation, and is aggregated to currency
and reference-currency levels by path-wise summation, preserving the dependence
induced by the shared factor and idiosyncratic draws.

A barrier product observed continuously (American observation) has its knock-in
determined over the whole simulated path rather than at maturity alone. The
Brownian-bridge correction is applied with this engine's own per-step diffusion
variance — the systematic part $\beta_i^\top \Sigma_f \beta_i\,\Delta t$ plus the
idiosyncratic part — and one definite knock-in outcome is sampled per path so the
loss distribution is preserved; see the barrier-observation methodology note.
European observation (the default) uses the terminal fixing only.

## 8. Common random numbers

All randomness — both the factor innovations and the idiosyncratic residuals —
is drawn from a shared noise sampler and held fixed across runs, so that
differences between scenarios reflect the scenario change alone. This is the
standard common-random-numbers variance-reduction technique for sensitivity
analysis (Glasserman, 2004, ch. 7); a fresh sample is taken only on explicit
request.

## 9. Assumptions and limitations

The construction assumes a linear factor model with loadings, factor
volatilities, and the factor correlation matrix held constant over the horizon
and estimated from historical data; in reality loadings and correlations drift,
and tend to rise in stress. Innovations are Gaussian and the idiosyncratic
residuals are mutually independent and independent of the factors. The factor
universe is intentionally narrow, so exposures orthogonal to the six factors are
captured only through the idiosyncratic term. As with the single-factor engine,
these simplifications are appropriate for a transparent stress tool, and the
outputs are to be interpreted as scenario-conditional assumptions rather than
forecasts.

## References

- S. A. Ross (1976), *The Arbitrage Theory of Capital Asset Pricing*,
  Journal of Economic Theory 13(3).
- W. F. Sharpe (1964), *Capital Asset Prices*, Journal of Finance 19(3).
- E. S. Schwartz (1997), *The Stochastic Behavior of Commodity Prices*,
  Journal of Finance 52(3).
- P. Glasserman (2004), *Monte Carlo Methods in Financial Engineering*,
  Springer — common random numbers, ch. 7.
