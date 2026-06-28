# Reverse Convertible — Validation Report

Measures the BRC/MBRC implementation against
`docs/specifications/reverse_convertible.md`. Each row maps one spec rule
to its code site and to a test that pins it down. Findings collect the
rows that did not pass.

Status legend:

* **OK** — implementation matches the spec; at least one test exists.
* **Finding F-NN** — implementation deviates or the rule is not pinned
  down; details in the Findings section.
* **No test** — implementation appears correct on reading but no test
  exists.
* **N/A** — spec content has no code obligation (meta / scope text).

---

## Mapping table

| Spec § | Rule (short) | Code | Test | Status |
|---|---|---|---|---|
| 1–3 | Scope, intended use, product list | — | — | N/A |
| 4 | `total_notional = denomination × position_units` | `reverse_convertible.py:_init_notional` | `test_product_info.py::TestDenomination` | OK |
| 5 | `cost_price` is a fraction of notional; `clean_cost = N × cost_price` | `reverse_convertible.py:392 total_cost` | `test_reverse_convertible.py::TestPayoff::test_total_cost` | OK |
| 6.1 | Coupon `c` is annualised decimal | `reverse_convertible.py:131 self.coupon` | `test_reverse_convertible.py::test_coupon_payment_one_year` | OK |
| 6.2 | Ordered dates `t_1 < … < t_n`, `t_0 = initial_fixing` | `coupon_schedule.py:CouponSchedule.__init__` | `test_purchase_date_and_schedule.py::TestScheduleConstruction` | OK |
| 6.3 | `C_j = N · c · τ_j` | `coupon_schedule.py:coupon_amounts` | `test_purchase_date_and_schedule.py::test_total_equals_sum_over_periods` | OK |
| 6.4 | Day-count in `{30/360, ACT/360, ACT/365}`; default `ACT/360` | `coupon_schedule.py:DAY_COUNTS`, `reverse_convertible.py:_build_schedule` | `test_purchase_date_and_schedule.py::test_default_day_count_is_act_360`, `::test_explicit_day_count_used` | OK |
| 6.5 | Missing `coupon_dates` → single bullet at maturity | `reverse_convertible.py:224 _build_schedule` | `test_purchase_date_and_schedule.py::test_default_schedule_is_single_bullet_at_maturity` | OK |
| 6.6 | Projected payoff includes only coupons with `t_j > purchase_date` | `reverse_convertible.py:372 coupon_payment` | `test_purchase_date_and_schedule.py::test_coupon_payment_excludes_pre_purchase_coupons` | OK |
| 6.7 | Schedule provenance out of scope of payoff spec | `portfolio_entry.py`, `term_sheet_extractor.py` | — | N/A |
| 7 | `total_cost = N × cost_price + accrued_at_purchase` | `reverse_convertible.py:392, 376` | `test_purchase_date_and_schedule.py::TestAccrued` | OK |
| 8 | `B_i = L_i × barrier_pct` (initial-level referenced) | `reverse_convertible.py:barrier_levels` | `test_reverse_convertible.py::TestBarrierDistance` | OK |
| 9 (European) | Breach iff `S_i(T) ≤ B_i` | `barrier.py:european_knock_in` | `test_barrier.py`, `test_paths_and_returns.py` | OK |
| 9 (American) | Continuous monitoring via Brownian bridge | `barrier.py:continuous_survival_prob`, `:sample_knock_in` | `test_american_barrier_mc.py`, `test_american_barrier_stress.py` | OK (see F-04) |
| 10 (no breach) | `R = N` | `reverse_convertible.py:363 redemption` | `test_reverse_convertible.py::test_full_redemption_above_strike` | OK |
| 10 (breach: π* and i*) | `i* = argmin S_i(T)/K_i`; `R = N · π*` | `reverse_convertible.py:365`; `vectorised_european_rc_summary:67–82` | `test_reverse_convertible.py`, `test_paths_and_returns.py::test_brc_barrier_breach_redemption_is_performance_times_notional` | OK |
| 11 | `V_T = R + Σ N · c · τ_j` (post-purchase) | `reverse_convertible.py:387 payoff`, `:389 total_payoff` | `test_reverse_convertible.py::test_payoff_equals_redemption_plus_coupon` | OK |
| 12 PnL | `PnL = V_T − total_cost` | `reverse_convertible.py:396 pnl` | `test_reverse_convertible.py::test_pnl_is_payoff_minus_cost` | OK |
| 12 return | `PnL / total_cost`; `NaN` when `total_cost = 0` | `reverse_convertible.py:399 return_pct` | `test_reverse_convertible.py::test_return_pct_zero_cost_is_nan` | OK |
| 12 YTM | XIRR over `cash_flows`, ACT/360 | `reverse_convertible.py:449 ytm`, `:432 _xirr` | `test_purchase_date_and_schedule.py` (YTM cases) | OK |
| 12 distance | `d_i = (S_i − B_i)/S_i`; multi → `min` | `reverse_convertible.py:486, 498` | `test_reverse_convertible.py::TestBarrierDistance` | OK |
| 12 break-even | Performance level where `PnL = 0` under post-purchase coupons | `reverse_convertible.py:504 break_even` | `test_purchase_date_and_schedule.py::test_break_even_higher_after_post_issuance_purchase` | OK |
| 13 | Documented exclusions and limitations | — | — | N/A |
| App. | GBM under Q with constant `r`, `σ`, `q`, `ρ` | `monte_carlo.py`, `vol_surface.py` | `test_monte_carlo.py` | OK |

---

## Tests in place

| Test file | Covers |
|---|---|
| `tests/test_reverse_convertible.py` | Payoff above/below strike, total cost, PnL, return, barrier distances, break-even, worst underlying |
| `tests/test_purchase_date_and_schedule.py` | Coupon schedule construction, day-count, accrued at purchase, cash flows, YTM, post-purchase coupon rule (F-03 regressions) |
| `tests/test_product_info.py` | Denomination, notional alias, coupon scales with notional |
| `tests/test_paths_and_returns.py` | Path-based BRC/MBRC payoffs (no breach, breach, exact fixtures) |
| `tests/test_barrier.py` | European knock-in mask |
| `tests/test_american_barrier_mc.py` | American barrier in Monte Carlo fair-value pricing |
| `tests/test_american_barrier_stress.py` | American barrier in stress / scenario distribution |
| `tests/test_monte_carlo.py` | MC engine, Greeks, dynamics under Q |


---

## Open findings

### F-04 — Worst-of bridge survival assumes within-step asset independence

* **Spec:** §9 / `barrier_observation_methodology.md`.
* **Code:** `barrier.py:continuous_survival_prob_from_var` (product across asset axis).
* **Severity:** LOW (documentation/approximation finding; acknowledged in docstring).

