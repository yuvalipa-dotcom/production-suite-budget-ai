# P4 leakage proof

> **ALL DATA IS SYNTHETIC.** This proves that the Crew 2 model sees only information that was available at each snapshot, and that every evaluation step respects the frozen groups. It says nothing about real-world accuracy.
> Copyright (c) 2026 Yuval Lipa. All rights reserved. No open-source license.

## 1. Prediction moment and target

**Prediction moment.** One prediction is made per eligible snapshot, at the end of `as_of_date`:

| Stage | `as_of_date` |
|---|---|
| S0 | contract day |
| S1 | 35% of lead time |
| S2 | 65% of lead time |
| S3 | 14 days before the event |

A record counts if its date is on or before `as_of_date` (`contracts/column_roles.json`, `as_of_rule`).

**Target.** `is_overrun` = `budget_gap_ils > 0.5` ILS. The gap is the verified final cost, plus the contractual fee known at the snapshot if included, minus the budget in effect at the snapshot (`gate/reconstruct.py`, `snapshot_labels`).
- This is a rounding-tolerance label, **not a commercially material overrun**.
- `is_material_overrun` (2%) is secondary only.

**Population.** The 1,538 Policy C eligible snapshots. The 62 exclusions are 50 `commitment_category_missing` and 12 `manual_cost_undated`.

## 2. Feature availability (as-of rules)

Every feature and its source columns is listed in [`p4_plan.md`](p4_plan.md) §c and in each run's `feature_availability.json`.

Revisable values are always taken from the dated ledgers as of the snapshot, never from a final-state column:
- budget, guest count and revisions to date come from the revision in effect;
- plans with `effective_date ≤ as_of`;
- commitments active at `as_of`;
- manual costs created by `as_of`.

**Fields without a revision ledger** are `archetype`, `event_date` (giving `event_month` and `lead_time_days`), `budget_vat_mode`, `fee_mode` and `fee_included`.
- They come from the single production master record. The generator sets them once and never revises them.
- Owner decision D4 accepts them as fixed from the contract date.
- A test proves that each one is constant across every production's snapshots, both in the frozen reference and in the handoff. It also proves that the cleaned master table has exactly one record per production.

**Primary feature set: `F27`.** It excludes `budget_revisions_to_date` and `client_direct_share`, because both are `PROPOSED` in the product.

## 3. Independent feature reference (C4)

All 1,538 eligible feature rows were reconstructed from the **cleaned** Policy C ledgers with the as-of rules. They were compared with the independently frozen v2 snapshot table `data/synthetic/v2/snapshots.csv`. That table is written by the generator, not by the cleaning or reconstruction code.

| Check | Result |
|---|---|
| Row identity `(production_id, stage)` | 1,538 = 1,538, same keys and order |
| Money amounts (tolerance 0.01 ILS) | worst difference 0.0050 ILS: the reference's 2-decimal rounding |
| Ratios and shares (tolerance 0.0051) | worst difference 0.0050: the same rounding |
| Integers, codes, booleans | exact |
| dtypes and missingness | correct dtypes; no missing value on either side |
| As-of budget | equals the revision in effect for all 1,538 rows, including rows whose production has a later revision |
| Handoff vs reference | equal under the same tolerances |

Nothing was modified to force agreement. `data/`, `contracts/`, `gate/` and `crew1/` are byte-identical to `7d2d38a` (`tests/test_crew2_frozen.py`).

## 4. Leakage tests and results

These are all offline. The full suite results are reported with the P4 run. The tests are in `tests/test_crew2_leakage.py`, `test_crew2_feature_reference.py`, `test_crew2_data.py` and `test_crew2_rules.py`.

| # | Test | Result |
|---|---|---|
| 1 | Outcome-only columns never reach the model input. The declared trace (`FEATURE_SOURCES`) has no outcome-only source. The input columns and the preprocessing columns of `F27` and `F29` contain no outcome or metadata column | pass |
| 2 | Dynamic trace: all 1,538 feature rows rebuilt with every outcome-only raw column removed are identical | pass (0 differing) |
| 3 | As-of invariance, run on 32 sampled snapshots (8 per stage). The following, dated **after** the snapshot, leave its features byte-identical: an added budget revision, an added category plan, an added commitment, an added manual cost, later records revised in place, a cancellation of an active commitment, and changes to `final_pre_vat` and `financial_close_date`. The same changes dated **on** the snapshot day move the features (two-way proof) | pass |
| 4 | A deliberately leaky builder, which ignores revision dates, is caught by the same check | pass |
| 5 | No `production_id` is in two groups or two folds; groups and folds equal `contracts/frozen_split.json`; a moved production is detected | pass |
| 6 | Test labels are not loaded before the final evaluation. Test and demo label fields are dropped as text; label parsing happens only for train rows; the vault is not opened through all 9 CV runs, the rule and the final fit; it opens once, and a second opening is refused | pass |
| 7 | Fit only on training-partition indices. Every declared fit, and every set of rows the `RowSpy` pipeline step saw reaching an estimator, lies inside its training partition. No outer-validation, test or demo row ever appears | pass |
| 8 | C2 split recording. Every split used at every tuning and calibration step, including the final refit, is group-disjoint and inside its partition. Calibration inside tuning uses exactly the tuning train side. The audit catches each planted violation: a production on both sides, a row outside the partition, a fit on the held-out side, calibration on a tuning held-out side, and the spy seeing undeclared rows | pass |
| 9 | The 0.5 threshold is fixed (p ≥ 0.5) and never chosen on data | pass |
| 10 | The selection rule, B0, the grids and the seeds equal the addendum. The addendum and the erratum were committed alone, before any Crew 2 code | pass |

**Runtime checks in every Crew 2 run** (`leakage_report.json`):
- declared trace;
- dynamic trace over all 1,538 snapshots;
- 276 as-of mutations on 20 snapshots;
- group checks;
- the vault is still sealed.

Every split/fit audit of every CV run and both final fits must show 0 violations. Validator W10 enforces this.

## 5. Why scikit-learn's own helpers were not relied on (C2)

Verified on the pinned scikit-learn 1.9.1:
- `CalibratedClassifierCV.fit(X, y, sample_weight=None, **fit_params)` has no `groups` argument. Groups would reach it only through metadata routing, which is disabled by default.
- The outer `GroupKFold` gives no protection to an inner tuning or calibration step.

What P4 does instead:
- It builds every inner split explicitly from the production ids of the current training partition.
- It calibrates by a controlled manual Platt procedure: base model on the inner-train side, sigmoid on the held-out side, mean of the members.
- It records every split and fit, and audits them.
- `CalibratedClassifierCV` and `GridSearchCV` are not used.
