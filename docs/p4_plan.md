# P4 plan: Crew 2 (features, training, evaluation)

> **ALL DATA IS SYNTHETIC.** Plan only (stage P4-0). No code, no training and no LLM call happen until the owner and the reviewer approve this plan.
> Branch `p4-crew2`, base `7d2d38a`. Frozen and not edited by P4: `data/`, `contracts/`, `gate/`, `crew1/`, the split and the Policy C rules.

## a. Prediction moment and target

**Prediction moment.** One prediction is made per scheduled snapshot `(production_id, stage)`, at the end of the day `snapshot_schedule.as_of_date`:

| Stage | `as_of_date` (`raw_data_dictionary.md`) |
|---|---|
| S0 | contract day |
| S1 | 35% of lead time |
| S2 | 65% of lead time |
| S3 | 14 days before the event |

A record is known at the prediction moment if its date is **on or before** `as_of_date` (`column_roles.json`, `as_of_rule`). Same-day records count as known.

**Target.** `is_overrun` is the only target, as pre-registered in `model_selection_rule.md`. It is defined in `gate/reconstruct.py` (`snapshot_labels`) and approved as definition A2 in `target_audit.md`:

- `final_cost_pre_vat` = sum of `commitments.final_pre_vat` over non-cancelled commitments, plus every `manual_costs.pre_vat`;
- `budget_gap_ils` = `final_cost_pre_vat` + contractual fee known at `as_of_date` (only if `fee_included`) − approved pre-VAT budget in effect at `as_of_date`;
- `is_overrun` = `budget_gap_ils > 0.5` (rounding tolerance only).

Contract fields used:
- `column_roles.json`: `commitments.final_pre_vat` and `productions.financial_close_date` are `outcome-only`, and so are the snapshot targets `final_cost_pre_vat`, `budget_gap_ils`, `is_overrun` and `is_material_overrun`.
- `gate/handoff_spec.py`: `COLUMNS` with role `target`, `PRIMARY_KEY = ["production_id", "stage"]`.
- `dataset_contract.json`: `row_count` = 1,538 and `exclusion_counts` = {`commitment_category_missing`: 50, `manual_cost_undated`: 12}.

**Population.** These are the 1,538 snapshots that are eligible under Policy C. The 62 excluded snapshots are never imputed or scored, and are reported as "not scorable". `budget_gap_ils`, `final_cost_pre_vat` and `is_material_overrun` are never features. They are reported only descriptively.

## b. Input

1. Load the 7 raw tables from `data/synthetic/v2/raw/` and check the manifest hashes. The check is the same as `G-SRC-MANIFEST`.
2. Build `clean_data.csv` text in-process with `crew1.tools.cleaning.build_clean_data(raw, frozen_doc, raw_manifest_sha256)`. This is pure Python Policy C with no LLM. It is imported read-only, and `crew1/` is not modified.
3. Assert that the result is the expected deterministic output:
   - its sha256 equals a constant `EXPECTED_CLEAN_SHA256` pinned in P4 code;
   - a P4 test proves the constant is right by comparing the text byte for byte with the independent reference `tests/handoff_reference.py`. That reference is built from the committed clean snapshots and the ground truth, not from the cleaning code;
   - `gate.run --stage handoff` is PASS on it;
   - 1,538 rows, and the exclusion counts are exactly as listed above.

   Any mismatch stops the run.
4. P4 never reads `data/synthetic/v2/snapshots.csv` or ground truth at runtime. Those are test-only, as in P2 and P3.

## c. Feature availability (every candidate feature)

Each of the 29 handoff features is recomputed by `gate/policy_c.snapshot_features`, which is a copy of `gate/reconstruct.py`. The full source columns per feature are declared in `FEATURE_SOURCES`.

**Revisable values** are amounts, budgets, plans, guest counts and commitment status. They are always taken from the dated ledgers as of the snapshot, and never from a final-state column.

The as-of rules:
- **(R)** budget revision in effect = the highest `revision_no` with `effective_date ≤ as_of`;
- **(P)** category plans with `effective_date ≤ as_of`;
- **(A)** active commitments: `selected_date ≤ as_of`, and `cancelled_date` empty or `> as_of`;
- **(M)** manual costs with `created_date ≤ as_of`;
- **(S)** static: the value comes from the production master record, which has no history in the data (see open question Q3).

| # | Feature | Source (raw) | As-of rule |
|---|---|---|---|
| 1 | `stage` | `snapshot_schedule.stage` | the snapshot itself |
| 2 | `archetype` | `productions.archetype` | S |
| 3 | `event_month` | `productions.event_date` | S |
| 4 | `lead_time_days` | `productions.contract_date`, `event_date` | S |
| 5 | `days_since_contract` | `contract_date`, `as_of_date` | computed at `as_of` |
| 6 | `days_to_event` | `event_date`, `as_of_date` | computed at `as_of` |
| 7 | `budget_vat_mode` | `productions.budget_vat_mode` | S |
| 8 | `fee_included` | `productions.fee_included` | S (contractual term) |
| 9 | `fee_mode` | `productions.fee_mode` | S (contractual term) |
| 10 | `guest_count` | `budget_revisions.guest_count` | R. **Not** `final_budget_entered` |
| 11 | `budget_pre_vat` | `budget_revisions.approved_entered_amount` + `budget_vat_mode` | R. **Not** `final_budget_entered` |
| 12 | `budget_revisions_to_date` | `budget_revisions.revision_no`, `effective_date` | R: count of `revision_no > 0`. **Not** `n_budget_revisions` |
| 13 | `fee_contract_pre_vat` | `fee_mode`, `fee_value`, `fee_vat_mode` + budget per R | R. Contractual fee only; **not** `fee_concession_pre_vat` or `final_fee_contract_pre_vat` |
| 14 | `planned_total_pre_vat` | `category_plans.planned_entered`, `vat_mode` | P |
| 15 | `n_categories_planned` | `category_plans.category` | P |
| 16 | `n_categories_committed` | plans + `commitments.category` | P + A |
| 17 | `n_categories_pending` | plans + `commitments.category` | P + A |
| 18 | `n_commitments_active` | `commitments.selected_date`, `cancelled_date` | A |
| 19 | `n_manual_costs` | `manual_costs.created_date` | M |
| 20 | `manual_costs_pre_vat` | `manual_costs.pre_vat` | M |
| 21 | `committed_pre_vat` | `commitments.original_pre_vat` + `manual_costs.pre_vat` | A + M. Accepted quote, **never** `final_pre_vat` |
| 22 | `n_categories_over_plan` | committed by category vs plan | P + A + M |
| 23 | `over_plan_pre_vat` | committed by category vs plan | P + A + M |
| 24 | `n_substitutions_to_date` | `commitments.cancelled_date` | cancellations with `cancelled_date ≤ as_of` |
| 25 | `no_vat_share` | `commitments.vat_mode` among active | A + M |
| 26 | `client_direct_share` | `commitments.payment_responsibility` among active | A + M |
| 27 | `current_position_ils` | committed + included fee − budget | A + M + R |
| 28 | `committed_ratio` | (committed + included fee) / budget | A + M + R |
| 29 | `cost_per_guest_pre_vat` | committed / guests | A + M + R |

**Excluded from all features:**
- every `outcome-only` column;
- every payment column (cash flow, `metadata`);
- `production_id`, `split`, `cv_fold`, `as_of_date`, `event_date`, `batch_id`, `is_synthetic`;
- `replaces_commitment_id`.

No new features are engineered in P4 without a separate approval. The 29 are already as-of and fully covered by the P2 gate checks `G-FIN-RECON`, `G-FIN-ASOF-BUDGET` and `G-NONFIN-RULES`.

## d. Leakage discipline

- **Data.** Model fitting uses `split == "train"` only: 280 productions, at most 1,120 rows before exclusions. CV uses the frozen `cv_fold` 0–4, so the folds are grouped by `production_id` and are never re-drawn.
- **Everything inside a Pipeline.** One scikit-learn `Pipeline` per candidate holds the imputer-free scaler, the one-hot encoder, the model and calibration. It is fitted only on the training-fold rows of each outer fold.
  - Hyper-parameter tuning is nested: the grid is fixed in code before the first run, and searched with an inner `GroupKFold` grouped by `production_id`, on the outer-training rows only.
  - Calibration (`CalibratedClassifierCV`, sigmoid, inner grouped CV) also runs on outer-training rows only.
  - There is no feature-selection step. All 29 features enter, and if one is ever added it goes inside the Pipeline.
- **Selection.** The pre-registered rule applies (`model_selection_rule.md`): lowest mean Brier over the 5 outer folds; within 0.005, higher mean PR-AUC; then the simpler model. It uses CV results only and is applied by deterministic code, not by an agent.
- **Test.** After selection, the chosen Pipeline is refit on all train, with the same nested tuning. Its config hash is frozen, and only then is **test** scored once. Test labels sit in a sealed store that only the final-evaluation step can open, at most once. A second open is an error.
- **Demo.** Demo is inference only: predictions are made and no metric is computed. It is never used to fit, tune, select or calibrate.

## e. Models and metrics

| ID | Candidate | Notes |
|---|---|---|
| B0 | Naive rule `current_position_ils > 0` | No fitting for the decision. Reported per stage. Its probability for Brier needs a definition (see Q1) |
| B-prev | Train-fold prevalence per stage | Reference for Brier skill. Not a selection candidate |
| LR | Calibrated logistic regression | Standardised numerics plus one-hot categoricals. Grid on `C`, L2 penalty |
| GB | `GradientBoostingClassifier`, calibrated | Small grid: `n_estimators`, `max_depth`, `learning_rate` |

There is no preset winner, and the result is reported as measured.

The task needs probabilities, not only a ranking. The product card shows an overrun **probability**, so probability quality comes first:

| Metric | Role | Reason |
|---|---|---|
| Brier score | Primary, pre-registered | Proper scoring rule for probability quality |
| PR-AUC | Tie-break, pre-registered | Ranking quality under about 44% prevalence |
| Log loss, ROC-AUC, reliability curve + ECE (10 bins), Brier skill score vs B-prev | Secondary | Calibration detail and discrimination |

Every metric is reported overall and **per stage**, as CV mean ± sd across folds. Test gets a production-level bootstrap 95% interval.

The known prevalence shift (train 44.5% vs test 59.2% after Policy C) is reported. It is not corrected.

## f. Leakage tests (to write, offline)

1. **Outcome columns.** The model's input column list ∩ `outcome-only` (`column_roles.json`) = ∅.
   - Every input column's `FEATURE_SOURCES` contains no outcome-only source.
   - The full pipeline runs with every outcome-only raw column **dropped** and gives identical feature matrices.
2. **As-of invariance.** For sampled snapshots, mutate a ledger record dated **after** `as_of`:
   - a later budget revision's amount and guest count;
   - a later category plan;
   - a later commitment's `original_pre_vat` and `selected_date`;
   - a `cancelled_date` moved later;
   - a later manual cost;
   - `final_pre_vat`;
   - `financial_close_date`.

   The snapshot's feature row must be byte-identical.

   The same mutation dated **on or before** `as_of` must move the feature. This is the two-way proof.
3. **Groups.** No `production_id` appears in two of train, test and demo, or in two CV folds. Each outer fold's train and validation `production_id` sets are disjoint. The folds equal `frozen_split.json`.
4. **Sealed test.**
   - Test labels are unreadable before `final_evaluation`: opening the sealed store early raises.
   - An access log shows exactly one read, after the model hash was frozen.
   - Selection code paths never receive test or demo rows. A spy on the data loader proves this.
5. **Fit spy.** A wrapper records the row indices of every `fit` / `fit_transform` call, including inside calibration and the grid search. All of them are a subset of the current outer fold's training indices, and no test or demo index ever appears.
6. **Determinism.** Two runs with the same seed give byte-identical `metrics.json` and predictions.

## g. Crew 2 design (CrewAI, the same pattern as Crew 1)

There are 3 agents with sequential tasks. Every number comes from deterministic tools, and the agents only choose tool calls and write interpretation.

| Agent | Task | Tools (its own only) |
|---|---|---|
| C1 Feature & Leakage Engineer | prepare features, prove availability | `build_handoff()` (b), `feature_availability_table()`, `run_leakage_checks()` |
| C2 Model Trainer | grouped CV, rule-based selection, final refit | `list_candidates()`, `run_grouped_cv(candidate_id)`, `apply_selection_rule()` (the rule decides, the agent cannot override), `fit_final_model()` |
| C3 Evaluator & Model Card Author | single test evaluation, demo inference, model card | `evaluate_on_test_once()`, `predict_demo()`, `render_chart(chart_id)`, `write_model_card(sections…)` (typed fields only, strict-schema safe) |

**Deliverables** (workspace `out/`, copied to `--out`):
- `model.joblib`, plus `model_export.json` if LR wins (coefficients + preprocessing constants, for the product);
- `metrics.json`: CV per candidate and per stage, the selection trace, and test metrics with CIs;
- `model_card.md`;
- `demo_predictions.csv`: `production_id`, `stage`, `p_overrun`, labelled "experimental / demo prediction";
- `cv_predictions.csv` (out-of-fold);
- `run_manifest.json`, `trace.jsonl`.

**Isolation** is the same pattern as Crew 1: a fresh workspace outside the repo with raw and the contracts only, logical names, a minimal env, and telemetry off.

**Validators**, all of which must pass:

| ID | Check |
|---|---|
| W1 | artifacts exist |
| W2 | handoff sha256 = `EXPECTED_CLEAN_SHA256` and the gate is PASS |
| W3 | a fresh deterministic recompute of CV metrics, the selection and test metrics equals `metrics.json` |
| W4 | the selected model equals the pre-registered rule's output |
| W5 | test was opened exactly once, after the model hash was frozen; demo has no metrics |
| W6 | demo predictions equal recomputation from `model.joblib` |
| W7 | every number in `model_card.md` matches `metrics.json` (reuses the Crew 1 V5 claim checker) |
| W8 | the model card has its required sections, a synthetic-data and "experimental / demo prediction" label, and no forbidden accuracy claims |
| W9 | the manifest is complete; the trace has no key or absolute path |

Runs are offline with a scripted mock LLM only. A live run needs separate owner approval.

## h. Documentation contributions

- **Synthetic-to-product mapping.** Extend `product_field_mapping.md` with a "used by the selected model" column.
  - Flag that `budget_revisions_to_date` and `client_direct_share` are `PROPOSED` (not computable in the product today).
  - Flag that the fee features are manager-only.
- **Leakage proof.** `docs/p4_leakage_proof.md` gives the as-of table (c), the tests (f) with their results, and the two-way mutation evidence.
- **Copyright.** The model card and the P4 docs state "Copyright (c) 2026 Yuval Lipa. All rights reserved. No open-source license", matching `LICENSE`.

## Dependencies to add (separate `.venv`, Python 3.13)

- `scikit-learn`, pinned to the exact version resolved at install, plus its transitive dependencies `scipy`, `joblib` and `threadpoolctl`, also pinned. They go in a new `requirements-ml.txt` that includes `-r requirements-crew.txt`.
- No other ML libraries: no XGBoost or LightGBM, since sklearn GB is pre-registered. `matplotlib` is already pinned.

## Open questions for the owner

1. **B0 probability.** The rule says "score derived from it" without a definition. Proposal: a 2-bin rate per stage (positive vs non-positive position), fitted on the outer-training fold.
2. **Product-available variant.** Two features are not computable in the product today. Should the main model use all 29 features (the pre-registered setup), with a "product-available" 27-feature variant reported only as a secondary result?
3. **Static master fields.** `event_date`, `fee_*`, `archetype` and `budget_vat_mode` have no history in the data, so they are treated as fixed at contract. Accept this, and document it as a limitation?
4. **Target scope.** Is it `is_overrun` only, as pre-registered? Or should `budget_gap_ils` regression or `is_material_overrun` be added, which would need a separate pre-registration first?
5. **Model artifact format.** `joblib` (pickle) is local only. If GB wins, is a JSON export needed for the product too, or only for LR?
