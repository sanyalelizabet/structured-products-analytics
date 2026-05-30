# Issuer-Callable Reverse Convertible Methodology

## 1. Purpose and scope

This note documents the valuation of the issuer-callable barrier reverse
convertible (`IC_BRC`), implemented in
`src/issuer_callable_reverse_convertible.py` and consumed by the fair-value
pricer and both stress engines. It concerns specifically the **issuer's call
right**; the worst-of payoff and the (European or continuously-monitored)
knock-in are treated as in the barrier-observation methodology, to which the
reader is referred.

An `IC_BRC` is a worst-of reverse convertible — periodic coupons and a
down-and-in barrier on the worst-performing share — to which is added a right,
held by the **issuer**, to redeem the note at par on a set of optional
redemption dates. The economic content of that right is the subject of this
note.

## 2. Distinction from the autocallable

It is essential to distinguish the issuer call from the autocall (`AC_BRC`),
because the two are not merely different mechanisms but carry **opposite
optionality for the holder**.

The autocall is *mechanical*: redemption is triggered automatically when the
worst-of level is at or above a contractual trigger on an observation date. It
fires in benign states (high underlyings) and is, if anything, favourable to the
holder, although it also terminates the product early and removes the opportunity
to receive future coupons.

The issuer call is *discretionary and adversarial*: the issuer exercises it to
**minimise the value of the note to the holder**. A rational issuer redeems at
par precisely when continuing the note would be expensive to it — that is, when
the note is *valuable* to the holder. The call therefore **caps** the holder's
value and can only reduce it. Modelling the issuer call as an autocall would
invert this sign and is inadmissible.

## 3. Valuation as an optimal-stopping problem

Let the call dates be $t_1 < t_2 < \dots < t_K$, and let the holder's
continuation value at a call date — the value of the note if it is *not* called
at that date — be $C(t_k)$. The issuer redeems at par $N$ (plus the coupon
accrued to $t_k$, which is paid in either branch). Acting to minimise the
holder's value, the issuer calls whenever continuing is dearer than redeeming,

```math
\text{call at } t_k \iff C(t_k) > N .
```

Equivalently the holder's value at $t_k$ (net of the accrued coupon paid on that
date) is $\min\!\big(N,\; C(t_k)\big)$, the hallmark of a short callable
position. The note value is the risk-neutral expectation of the discounted
cashflows under the issuer's value-minimising stopping policy $\tau^\*$,

```math
V_0 \;=\; \mathbb{E}^{\mathbb{Q}}\!\left[\sum_{t_j \le \tau^\*} D(0,t_j)\,c_j \;+\; D(0,\tau^\*)\,R_{\tau^\*}\right],
```

where $c_j$ are the coupons, $R_{\tau^\*}$ is the redemption at termination (par
on a call date; par or the worst-of conversion at maturity), and $D(0,t)$ is the
risk-free discount factor.

## 4. Longstaff–Schwartz estimation of the continuation value

The continuation value $C(t_k)$ is a conditional expectation and is estimated by
the least-squares Monte Carlo method of Longstaff and Schwartz (2001). Working
**backwards** from the last call date, the realised discounted continuation
payoff of each simulated path is regressed on a low-order polynomial of the
worst-of performance $w_{t_k} = \min_i S_{i,t_k}/K_i$ at that date,

```math
\widehat{C}(t_k) \;=\; \beta_0 + \beta_1\,w_{t_k} + \beta_2\,w_{t_k}^2 ,
```

with $(\beta_0,\beta_1,\beta_2)$ obtained by ordinary least squares across paths.
The issuer is then taken to call on every path where
$\widehat{C}(t_k) > N$; the termination state of those paths is set to $t_k$ with
redemption $N$ plus accrued coupon, and the induction proceeds to the preceding
call date using the updated terminations. With the small number of call dates
typical of these notes, two or three backward regressions suffice.

The regression uses *realised* per-path cashflows, which is why the knock-in is
resolved to a definite per-path indicator (sampled at the Brownian-bridge rate
for continuous observation, terminal for European) rather than to the
survival-probability expectation used for the plain barrier note.

## 5. Coupons, accrual and discounting

Coupons accrue on an Actual/360 basis from the initial fixing; a path terminated
at $t_k$ receives par plus the coupon accrued to $t_k$, and a path surviving to
maturity receives the full coupon together with the worst-of redemption. Each
path is discounted to today from its own termination date, so that early-called
paths are discounted over a shorter horizon. The continuation-value comparison in
Section 4 discounts at the risk-free rate of the product's currency: the call is
the issuer's economic decision and is taken on a risk-neutral basis, which is why
the same exercise logic is applied unchanged when the note is evaluated under the
physical-measure stress scenarios.

## 6. Fair value and stress

In fair value the optimal exercise is solved on the risk-neutral Monte Carlo
paths and the resulting cashflows discounted to today. In the stress engines the
identical optimal-exercise logic is applied to the scenario paths, so that the
issuer call is reflected in the simulated profit-and-loss distribution; the
knock-in is sampled continuously for American observation exactly as for the
other barrier products.

## 7. Assumptions and limitations

* The issuer is assumed to exercise **optimally** (value-minimising). Real
  issuers may call sub-optimally; the model therefore yields the holder's
  *lower* bound on value with respect to call policy, which is the prudent
  convention for a buy-side valuation.
* Coupons are modelled as continuous Actual/360 accrual paid at termination,
  consistent with the rest of the analytics, rather than as discrete scheduled
  amounts.
* The continuation value is approximated by a quadratic polynomial in the
  worst-of level; this is the standard Longstaff–Schwartz basis and is adequate
  for the few exercise dates these notes carry.
* As with the barrier observation generally, the path is monitored from today
  forward and no knock-in is assumed to have occurred before the valuation date.

## 8. References

Longstaff, F. A. and Schwartz, E. S. (2001). *Valuing American Options by
Simulation: A Simple Least-Squares Approach.* Review of Financial Studies 14(1).

Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.* Springer.
