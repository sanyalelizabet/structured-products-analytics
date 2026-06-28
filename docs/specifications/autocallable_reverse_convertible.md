# Autocallable Barrier Reverse Convertible (AC_BRC) — Model Specification

## 1. Model family

This specification covers the Autocallable Barrier Reverse Convertible
(AC_BRC) product family. An AC_BRC is a barrier reverse convertible
extended with a contractual *autocall feature*: a schedule of observation
dates on which the issuer redeems the note early at par if a trigger
condition on the underlyings is met. AC_BRCs may be single-underlying or
multi-underlying worst-of, on the same conventions as the base BRC. The
product reuses the payoff and barrier mechanics specified in
`reverse_convertible.md` for any path that is not called.

The specification is restricted to fixed-coupon AC_BRCs with deterministic
autocall observation schedules. Coupon-memory ("snowball" or
"phoenix-with-memory") variants — in which missed coupons accumulate and
are paid at the next trigger date — are out of scope of the present
implementation and are addressed only as a deferred extension (§14).

## 2. Intended use

The model is intended for buy-side portfolio monitoring, educational
analytics, payoff transparency, scenario analysis, and product-level and
portfolio-level risk reporting. It is not a certified sell-side issuance
pricing model. It can serve as a starting point for theoretical valuation,
but its primary role within this application is monitoring and analytics
under the assumptions documented in this specification and in the
cross-referenced methodology documents.

## 3. Product scope

This specification covers:

* `AC_BRC` — Autocallable Barrier Reverse Convertible, both single-underlying
  and multi-underlying worst-of, with European or American barrier
  observation and without coupon memory.

The following related products are specified separately:

* `BRC` / `MBRC` — base Barrier Reverse Convertible, in
  `reverse_convertible.md`.
* `IC_BRC` — Issuer-Callable Barrier Reverse Convertible.
* `CPN` — Capital Protection Note.

## 4. Notional convention

The notional of a position is decomposed as

```text
denomination     = face value per certificate
position_units   = number of certificates held
total_notional   = denomination × position_units
```

Throughout this specification, the symbol `N` denotes the total position
notional. The convention is identical to `reverse_convertible.md` §4.

## 5. Cost price convention

The cost price is interpreted as a fraction of notional. A value of `1.00`
corresponds to par. The clean purchase cost is

```text
clean_cost = N × cost_price
```


## 6. Coupon

The coupon `c` is an annualised decimal rate, the contractual
schedule `t_1 < … < t_n` with day-count `τ_j` determines the per-period
amount `C_j = N · c · τ_j`, and the post-purchase rule of §6.6 applies to
the uncalled-path projection of the investor return.

The autocall feature introduces one additional coupon convention specific
to the *autocalled path*. On a call date `t_c` (which is not in general
one of the contractual coupon dates `t_j`), the holder receives, in
addition to the par redemption specified in §11, an accrued coupon
calculated on a continuous-time pro-rata basis from the initial fixing
date:

```math
C_{\text{call}} = N \cdot c \cdot \tau(t_0, t_c),
```

where `τ(t_0, t_c)` is the year fraction from initial fixing `t_0` to the
call date `t_c` under the contractual day-count convention. This
pro-rata-from-issuance accrual is distinct from, and does not generally
agree with, the sum of contractual coupons that would have been paid up
to `t_c` under the regular schedule; the contract therefore deliberately
short-cuts the schedule on call.

Coupon-memory variants in which missed coupons accumulate and are
disbursed on call are not in scope (§3, §14).

## 7. Accrued coupon at purchase

The dirty purchase cost is the sum of the clean cost and the coupon
accrued from the last contractual coupon date to the purchase date,
calculated against the contractual schedule:

```text
total_cost = N × cost_price + accrued_at_purchase
```

 The autocalled-path coupon of §6 does not affect the accrued-at-purchase
calculation, which is always referenced to the contractual schedule.

## 8. Barrier convention

The absolute down-barrier of underlying `i` is

```math
B_i = L_i \cdot b,
```

where `L_i` is the initial fixing level and `b` is the contractual
barrier fraction (`barrier_pct`). The convention is identical to
`reverse_convertible.md` §8.

The barrier is only consulted on paths that are not autocalled (§10,
§11); on autocalled paths the trigger condition is, by construction,
stricter than the barrier (the worst-of underlying is at or above its
strike, which is at or above its barrier under the assumption that
`barrier_pct ≤ 1`), and therefore no barrier check is performed.

## 9. Barrier observation

The contractual observation convention for uncalled paths is selected by
the product's `type_style` field and follows `reverse_convertible.md` §9:

* `european` — the barrier is observed only at the final fixing of the
  uncalled path.
* `american` — the barrier is observed continuously over the life of the
  product, up to and including maturity for paths that are never called.

Numerical implementation of the continuous-monitoring case is specified
in `barrier_observation_methodology.md`.

## 10. Autocall observation

The autocall feature is characterised by:

* a contractual sequence of *autocall observation dates*
  `t^{ac}_1 < t^{ac}_2 < … < t^{ac}_m`, all strictly between the initial
  fixing date and the maturity date;
* a contractual *autocall trigger* `g`, expressed as a fraction of strike
  (`autocall_trigger_pct`), with `g = 1.00` denoting a trigger at strike.

At each observation date `t^{ac}_k`, the *trigger condition* is

```math
\min_i \frac{S_{i, t^{ac}_k}}{K_i} \;\ge\; g.
```

If the trigger condition is satisfied at `t^{ac}_k` and no earlier
observation has called the product, the product is autocalled on
`t^{ac}_k` and `t_c := t^{ac}_k` becomes the call date. The trigger is
worst-of: every underlying must be at or above `g · K_i` for the call to
fire.

The autocall denominator is *strike*, not the initial fixing level. The
choice is consistent with the delivery-selection denominator in
`reverse_convertible.md` §10 and is conventional for retail AC_BRCs in
the Swiss market; for typical products with `K_i = L_i` the choice is
numerically immaterial.

## 11. Redemption

The terminal redemption depends on whether the autocall fired and, if
not, on whether the barrier was breached on the uncalled path.

**Autocalled (called at `t_c`).** The holder receives par plus the
pro-rata-from-issuance coupon of §6:

```math
R_{\text{call}} = N + C_{\text{call}} = N \cdot \bigl(1 + c \cdot \tau(t_0, t_c)\bigr).
```

Settlement is in cash. No physical delivery occurs on the autocalled
branch, and no further coupons are paid after `t_c`.

**Uncalled.** The terminal redemption follows `reverse_convertible.md`
§10 in full: par if the barrier held over the contractual observation
convention of §9, and otherwise the worst-of physical delivery (or its
cash equivalent) at strike.

## 12. Total payoff

The total payoff is

```math
V = \begin{cases}
R_{\text{call}}
  & \text{if the autocall fired at some } t^{ac}_k, \\[4pt]
R_{\text{uncalled}} + \displaystyle\sum_{t_j > t_{\text{purchase}}} N \cdot c \cdot \tau_j
  & \text{otherwise.}
\end{cases}
```

On the uncalled branch the contractual coupon stream specified in §6
contributes, with the post-purchase rule of `reverse_convertible.md`
§6.6 applied. On the autocalled branch the only coupon paid is
`C_{\text{call}}` from §6, regardless of any contractual coupons that
would have been due before `t_c` under the regular schedule.

## 13. Analytics conventions

The analytics conventions of `reverse_convertible.md` §12 apply, with
two adaptations for the autocall feature.

* **Profit and loss.** `PnL = V − total_cost`, where `V` is the regime-
  specific payoff of §12.
* **Simple return.** `return_pct = PnL / total_cost`.
* **Yield to maturity.** Computed by XIRR over the cash flows
  `{−total_cost}` at purchase, the contractual coupons scheduled
  strictly after the purchase date, and the terminal redemption. For
  monitoring purposes the YTM is computed against the *uncalled* maturity
  cash flows; a fair-value model that integrates the autocall
  probability into a probability-weighted YTM is out of scope of the
  present analytics.
* **Distance to barrier.** As in `reverse_convertible.md` §12.
* **Break-even.** As in `reverse_convertible.md` §12. The autocall
  feature does not alter the contractual break-even on the uncalled path.

## 14. Exclusions and limitations

The specification inherits all exclusions of `reverse_convertible.md`
§13 (issuer credit risk, funding spread, bid/ask spread, taxes,
liquidity, stochastic rates, dividends beyond constant `q_i`, legal
term-sheet exceptions). The autocall feature introduces two further
limitations:

* **Coupon memory.** Phoenix-with-memory and snowball variants are not
  implemented. The `autocall_coupon_memory` field on the portfolio row is
  reserved for a future extension.
* **Observation grid.** The autocall observation dates are snapped to the
  nearest business day on the Monte Carlo simulation grid. For typical
  daily grids the snap is at most half a business day and immaterial; for
  coarser grids the snap can advance or delay a call event and alter the
  call probability.

---

## Appendix — Pricing dynamics under simulation

The contractual payoff specified in Sections 1–14 is independent of any
particular pricing model. When the product is valued by simulation —
Monte Carlo fair-value pricing, scenario analysis, or stress testing —
the engines assume the dynamics recorded here. These are properties of
the *pricing engines*, not of the *contract*, and are reproduced in
full detail in the methodology documents cross-referenced below.

Fair-value calculations are performed under the risk-neutral measure
`Q` with each underlying `i` following a geometric Brownian motion

```math
\frac{dS_{i,t}}{S_{i,t}} = (r - q_i)\,dt + \sigma_i\,dW^{\mathbb{Q}}_{i,t},
\qquad d\langle W_i, W_j\rangle_t = \rho_{ij}\,dt,
```

where `r` is the discount rate, `q_i` the dividend yield of underlying
`i`, `σ_i` its volatility, and `ρ_ij` the instantaneous correlation. The
discount curve, volatility surface, dividend yield, and correlation
matrix are treated as exogenous inputs whose construction is specified
in `vol_surface_methodology.md`, the correlation methodology, and the
rates module.

Scenario and stress runs apply the same dynamics under the physical
measure `P`, with drifts replaced by user- or factor-specified expected
returns. The autocall observation logic is identical under both
measures: the trigger condition of §10 is checked path-by-path on the
simulated price grid.

The engines presently treat `r`, `σ_i`, `q_i`, and `ρ_ij` as constants
over the life of the product. The deferred-extensions list in
`valuation_production_roadmap.md` records the limitations this
introduces.
