# Autocallable Reverse Convertible — Validation Report

Measures the AC_BRC implementation against
`docs/specifications/autocallable_reverse_convertible.md`. The base BRC
mechanics this product inherits are validated separately in
`reverse_convertible_validation.md`; rows below that delegate to base
mechanics carry the status `OK (delegates to RC)`.

Status legend:

* **OK** — implementation matches the spec; at least one test exists.
* **OK (delegates to RC)** — rule is inherited from the base BRC and
  validated in the RC report.
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
| 4 | Notional decomposition | inherited via `ReverseConvertible` | RC tests | OK (delegates to RC) |
| 5 | Cost price as fraction of notional | inherited | RC tests | OK (delegates to RC) |
| 6 (contractual schedule) | `C_j = N · c · τ_j`; post-purchase rule | inherited | RC tests | OK (delegates to RC) |
| 6 (called-path accrual) | `C_call = N · c · τ(t_0, t_c)` under contractual day-count | `autocallable_reverse_convertible.py:130` (`rc_proto.schedule.year_fraction(initial_fixing, call_date[p])`) | `test_autocallable_reverse_convertible.py::TestEarliestCall::test_call_payoff_is_par_plus_prorata_coupon`, `::test_call_accrual_respects_contractual_day_count_act_365` | OK |
| 7 | Dirty cost = clean + accrued at purchase | inherited | RC tests | OK (delegates to RC) |
| 8 | `B_i = L_i · b` (initial-level referenced) | inherited | RC tests | OK (delegates to RC) |
| 9 | Barrier observation (European / American) on uncalled paths | `monte_carlo.py` autocall path passes `uncalled_breach_mask` to `vectorised_european_rc_summary` | `test_autocallable_reverse_convertible.py::TestNeverCalled` | OK |
| 10 (obs dates) | Contractual obs schedule between fixing and maturity | `autocallable_reverse_convertible.py:78–79`; grid snap at L100; skip outside grid L98 | `test_autocallable_reverse_convertible.py::TestEarliestCall`, `::TestPathDependentCall`, `::TestNoObservationDates` | OK |
| 10 (trigger) | `min_i S_{i,t}/K_i ≥ g`; default `g = 1.00` | `autocallable_reverse_convertible.py:77 (default), :103 (worst_perf), :106 (compare)` | `test_autocallable_reverse_convertible.py::TestTriggerSensitivity`, `::TestWorstOfBlocksCall` | OK |
| 10 (first-call) | Earliest triggered observation wins | `autocallable_reverse_convertible.py:105 (not_yet_called gate)` | `test_autocallable_reverse_convertible.py::TestEarliestCall::test_above_trigger_throughout_calls_at_first_obs`, `::TestPathDependentCall::test_drop_then_recover_calls_after_recovery_only` | OK |
| 11 (called) | `R_call = N · (1 + c · τ(t_0, t_c))` | `autocallable_reverse_convertible.py:131–132` | `test_autocallable_reverse_convertible.py::TestEarliestCall::test_call_payoff_is_par_plus_prorata_coupon` | OK |
| 11 (called: no breach, no delivery) | Autocalled paths set `barrier_breached=False`, no physical delivery | `autocallable_reverse_convertible.py:147–152` | `test_autocallable_reverse_convertible.py::TestMixedCohort` | OK |
| 11 (uncalled) | Delegates to RC §10 | `autocallable_reverse_convertible.py:154–188` (calls `vectorised_european_rc_summary`) | `test_autocallable_reverse_convertible.py::TestNeverCalled::test_never_calls_pays_full_coupon`, `::test_never_calls_with_breach_matches_brc` | OK (delegates to RC) |
| 12 | Regime-dependent total payoff | composition of §11 branches in `autocallable_reverse_convertible.py:128–188` | `test_autocallable_reverse_convertible.py::TestMixedCohort` | OK |
| 13 PnL / return | `PnL = V − total_cost`; `return_pct = PnL / total_cost` | `autocallable_reverse_convertible.py:138–140`, RC for uncalled | `test_autocallable_reverse_convertible.py::TestMixedCohort`, `test_autocallable_fair_value.py::TestAutocallableFairValue` | OK |
| 13 YTM | Computed against uncalled maturity cash flows | inherited `ReverseConvertible.ytm()` (no AC-specific override) | — | No test (intentional — spec defers AC-probability-weighted YTM) |
| 13 distance / break-even | Per RC §12 | inherited | RC tests | OK (delegates to RC) |
| 14 (memory) | Coupon memory not implemented; field reserved | `autocallable_reverse_convertible.py:42–43` (docstring); field defined in `portfolio_entry.py` | — | OK (acknowledged exclusion) |
| 14 (grid snap) | Obs dates snapped to nearest business day | `autocallable_reverse_convertible.py:100` | implicitly via existing call tests | OK (acknowledged limitation) |
| App. | GBM under Q with constant `r`, `σ`, `q`, `ρ` | `monte_carlo.py`, `vol_surface.py` | `test_autocallable_fair_value.py` | OK |

---

## Tests in place

| Test file | Covers |
|---|---|
| `tests/test_autocallable_reverse_convertible.py` | Earliest-call mechanics, path-dependent recovery, never-called fallback to BRC, worst-of blocking call, trigger sensitivity, empty-obs degeneracy, mixed cohort PnL, engine dispatch through single-factor and factor engines |
| `tests/test_autocallable_fair_value.py` | Fair-value pricing: near-certain call truncates coupon stream; unreachable trigger reduces to European BRC fair value; fair value is finite and priced |

---

## Open findings

*(None.)*
