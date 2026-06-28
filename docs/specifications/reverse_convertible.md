# Reverse Convertible Model Specification

## 1. Model family

This specification covers the base reverse-convertible product family. It is
best understood as a *Barrier Reverse Convertible (BRC) payoff and monitoring
specification*, with support for both single-underlying BRCs and
multi-underlying worst-of BRCs (MBRC).

The specification is not complete for autocallable or issuer-callable reverse
convertibles. Those products reuse the base BRC payoff logic but introduce
additional path-dependent call features that are specified separately.

## 2. Intended use

The model is intended for buy-side portfolio monitoring, educational
analytics, payoff transparency, scenario analysis, and product-level and
portfolio-level risk reporting. It is not a certified sell-side issuance
pricing model. It can serve as a starting point for theoretical valuation, but
its primary role within this application is monitoring and analytics under the
assumptions documented in this specification and in the cross-referenced
methodology documents.

## 3. Product scope

This specification covers:

* `BRC` — single-underlying Barrier Reverse Convertible.
* `MBRC` — multi-underlying worst-of Barrier Reverse Convertible.

The following products reuse the payoff logic specified here but require
separate specifications for their additional features:

* `AC_BRC` — Autocallable Barrier Reverse Convertible.
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
notional.

## 5. Cost price convention

The cost price is interpreted as a fraction of notional. A value of `1.00`
corresponds to par, `0.98` to a 2% discount, and `1.02` to a 2% premium. The
clean purchase cost is

```text
clean_cost = N × cost_price
```

## 6. Coupon

### 6.1. Coupon rate

The coupon `c` is an annualised decimal rate (`c = 0.08` denotes an 8 %
per-annum coupon) inherited from the term sheet.

### 6.2. Coupon schedule

The product carries an ordered sequence of contractual coupon payment
dates

```text
t_1 < t_2 < ... < t_n
```

inherited from the term sheet, with the conventional accrual-period
boundary `t_0` equal to the initial fixing date. The `j`-th accrual period
is the half-open interval `(t_{j-1}, t_j]`.

### 6.3. Per-period coupon amount

The coupon amount payable at date `t_j` is

```math
C_j = N \cdot c \cdot \tau_j,
```

where `τ_j = year_fraction(t_{j-1}, t_j ; day_count)` is the year fraction
of the `j`-th accrual period under the contractual day-count convention.

### 6.4. Day-count convention

The day-count convention is contractual and inherited from the term sheet.
The admissible conventions are `30/360`, `ACT/360`, and `ACT/365`. The
implementation defaults to `ACT/360` when the term sheet does not specify
a convention; this default is an analytics fallback, not a contractual
statement, and any product whose term sheet specifies a different
convention must be entered with that convention explicitly.

### 6.5. Single-bullet fallback

When no coupon dates are supplied for a product, the model treats the
product as carrying a single bullet coupon paid at maturity, accruing from
the initial fixing date. This is an analytics simplification used when the
schedule is absent from the input. Products whose contractual coupon
schedule is genuinely multi-periodic must supply their coupon dates
explicitly; relying on the fallback in such cases will understate the
number of coupon payment events and may distort yield and break-even
analytics.

### 6.6. Coupons in the projected investor return

The projected total payoff and PnL include only coupons whose payment date
satisfies

```text
t_j > purchase_date.
```

Coupons whose payment date precedes the purchase date were received by the
previous holder and are not part of the buyer's investor return. They are
accounted for separately through the accrued-at-purchase mechanism
specified in Section 7, which adjusts the dirty purchase cost so that the
buyer compensates the seller for the elapsed portion of the in-progress
accrual period.


### 6.7. Data provenance

The coupon rate, payment dates, and day-count convention are properties of
the term sheet. Their ingestion into the portfolio row is the
responsibility of the data-entry and term-sheet-extraction pipeline and is
outside the scope of this specification.

## 7. Accrued coupon at purchase

The dirty purchase cost is the sum of the clean cost and the coupon accrued
to the purchase date:

```text
total_cost = N × cost_price + accrued_at_purchase
```

The accrual is calculated from the contractual coupon schedule up to the
purchase date. The cost price is therefore a *clean* price; accrued coupon is
added separately to obtain dirty cost, consistent with market convention.

## 8. Barrier convention

The absolute down-barrier of underlying `i` is

```math
B_i = L_i \cdot b,
```

where `L_i` is the initial fixing level of underlying `i` and `b` is the
contractual barrier fraction (`barrier_pct`). The barrier is referenced to
the initial fixing level, not to the strike.

## 9. Barrier observation

The contractual observation convention is selected by the product's
`type_style` field:

* `european` — the barrier is observed only at final fixing. A breach occurs
  iff some underlying's terminal level satisfies `S_i(T) ≤ B_i`.
* `american` — the barrier is observed continuously over the life of the
  product. A breach occurs if any underlying touches or crosses its barrier
  at any time.

Numerical implementation of the continuous-monitoring case, including the
Brownian-bridge crossing correction applied between simulated grid points
and the worst-of bridge survival aggregation, is specified in
`barrier_observation_methodology.md`. The deterministic product-summary path
of the model carries no intermediate price trajectory and therefore reduces
both conventions to a single terminal check; this is an analytics
simplification, not a model statement about the contract.

## 10. Redemption

The terminal redemption depends on whether the barrier was breached under
the contractual observation convention of Section 9.

**No breach.** The holder receives the full notional:

```text
R = N
```

**Breach.** Settlement is contractually by physical delivery of the
worst-performing underlying. The underlying selected for delivery is the one
with the lowest terminal level against its strike:

```math
i^\star = \arg\min_i \frac{S_{i,T}}{K_i},
```

where `K_i` is the contractual strike of underlying `i`. The delivered
quantity is `N / K_{i*}` shares of underlying `i*`, with any fractional
residual paid in cash at the terminal spot. For analytics purposes the
cash-equivalent value

```math
R = N \cdot \pi^\star, \qquad \pi^\star = \frac{S_{i^\star, T}}{K_{i^\star}},
```

is used throughout; the physical and cash-equivalent treatments coincide at
maturity under the assumed settlement mechanics.

## 11. Total payoff

The total payoff at maturity is the sum of the terminal redemption and the
contractual coupon stream scheduled strictly after the purchase date:

```math
V_T = R + \sum_{t_j > t_{\text{purchase}}} N \cdot c \cdot \tau_j.
```

## 12. Analytics conventions

The following quantities are derived analytics built on top of the payoff
specification above. They are not contract features; they are the canonical
definitions used throughout this application's reporting layer.

* **Profit and loss.** Product-level PnL is `PnL = V_T − total_cost`,
  expressed in product currency. FX translation is handled at the portfolio
  level.
* **Simple return.** `return_pct = PnL / total_cost`. Interpreted as a
  holding-period return, not an annualised yield. Returns `NaN` if
  `total_cost` is zero.
* **Yield to maturity.** Computed by an internal XIRR over the cash flows
  `{−total_cost}` at the purchase date, the future coupon payments strictly
  after the purchase date, and `R` at maturity, under an ACT/360 year-fraction
  convention.
* **Distance to barrier.** Per underlying,
  `d_i = (S_i(t) − B_i) / S_i(t)`; for multi-underlying products, the
  reported distance is `min_i d_i`. Positive values indicate the spot lies
  above the barrier.
* **Break-even.** The terminal performance level below which total PnL is
  negative under the contractual coupon and cost conventions, expressed as a
  fraction of strike.

## 13. Exclusions and limitations

The specification does not presently incorporate:

* issuer credit risk and issuer default probability;
* issuer funding spread;
* bid/ask spread;
* taxes, transaction fees, and other frictions;
* liquidity risk;
* stochastic interest rates;
* dividends and borrow costs as path-affecting inputs beyond the constant
  `q_i` introduced in the appendix;
* legal term-sheet exceptions.

The deterministic product-summary path additionally does not monitor
American-style barriers, because it carries no intermediate price
trajectory; path-based monitoring is performed by the Monte Carlo and
scenario engines and is specified in `barrier_observation_methodology.md`.

---

## Appendix — Pricing dynamics under simulation

The contractual payoff specified in Sections 1–13 is independent of any
particular pricing model. When the product is valued by simulation — Monte
Carlo fair-value pricing, scenario analysis, or stress testing — the engines
assume the dynamics recorded here. These are properties of the *pricing
engines*, not of the *contract*, and they are reproduced in full detail in
the methodology documents cross-referenced below.

Fair-value calculations are performed under the risk-neutral measure `Q`
with each underlying `i` following a geometric Brownian motion

```math
\frac{dS_{i,t}}{S_{i,t}} = (r - q_i)\,dt + \sigma_i\,dW^{\mathbb{Q}}_{i,t},
\qquad d\langle W_i, W_j\rangle_t = \rho_{ij}\,dt,
```

where `r` is the discount rate, `q_i` the dividend yield of underlying `i`,
`σ_i` its volatility, and `ρ_ij` the instantaneous correlation. The
discount curve, volatility surface, dividend yield and correlation matrix
are treated as exogenous inputs whose construction is specified in
`vol_surface_methodology.md`, the correlation methodology, and the rates
module.

Scenario and stress runs apply the same dynamics under the physical measure
`P`, with drifts replaced by user- or factor-specified expected returns;
the diffusion specification is otherwise identical, which guarantees that
risk-neutral pricing and physical-measure scenario analysis share the
barrier-observation logic specified in `barrier_observation_methodology.md`.

The engines presently treat `r`, `σ_i`, `q_i` and `ρ_ij` as constants over
the life of the product. The deferred-extensions list in
`valuation_production_roadmap.md` records the limitations this introduces.
