# Pre-registered model selection rule

> Fixed on 2026-10-18, **before any training**. No model is trained in phase P1.
> **ALL DATA IS SYNTHETIC.**

## Candidates (classification: `is_overrun`)

1. **B0 baseline:** the current deterministic position. Predict an overrun when `current_position_ils > 0`, with the score derived from it. This is "the final cost equals what is committed today".
2. **Calibrated logistic regression:** standardised numeric features and one-hot categories, with calibration inside the cross-validation.
3. **Gradient boosting:** scikit-learn `GradientBoostingClassifier`, with calibration inside the cross-validation.

**No winner is predetermined.** The result is reported exactly as measured.

## Procedure

- Use the **train** group only, with the 5 production-grouped folds from `assign_folds`.
- Hyper-parameters for each candidate come from a small fixed grid, defined in code before the first run.

## Rule

1. **Primary metric:** the lowest **mean Brier score** across the 5 grouped folds.
2. **Tie-break:** if two candidates are within **0.005** Brier of each other, the one with the higher **mean PR-AUC** wins.
3. **Second tie-break:** if still tied, the **simpler** model wins (B0 < logistic < gradient boosting).

## After the rule is applied

- The selected model is refit on all of train and evaluated **once** on **test**.
- The test result is reported, and never used to change the choice.
- **demo is not touched** by selection or evaluation.

## Naive reference rule (added in P1-R, a reference point only)

> **"Already over the budget in effect now ⇒ will overrun"**: predict an overrun when `current_position_ils > 0` (recorded commitments plus the fee if included, against the approved budget in effect at the snapshot).

The rule involves no fitting. It is computed on **train only**, per stage. It is a reference point and **not** a selection gate: the pre-registered procedure above (training data only, production-grouped cross-validation, Brier score first) is unchanged.

<!-- BEGIN GENERATED: naive-baseline -->
| Stage | Rows | Prevalence | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| S0 | 280 | 47.9% | n/a (no positive predictions) | 0.0% | 52.1% |
| S1 | 280 | 46.4% | n/a (no positive predictions) | 0.0% | 53.6% |
| S2 | 280 | 41.8% | 88.9% | 6.8% | 60.7% |
| S3 | 280 | 40.0% | 97.2% | 61.6% | 83.9% |
<!-- END GENERATED: naive-baseline -->

Later evaluation reports **probability quality and calibration** (Brier score, log loss, reliability curves) as well as **discrimination** (ROC-AUC, PR-AUC), per stage.

## Product preference (stated separately from the rule)

For future product integration a simple calibrated logistic model is preferred, because it exports to plain JSON. If gradient boosting wins the rule, that is reported honestly. Shipping a different model in the product would be a separate, explicit owner decision, documented with the measured difference.

---

## Addendum P4-1: fixed before any training (2026-10-19)

> This addendum was committed **on its own, before any P4 training code existed**. Every training commit comes after it. It makes the rule above exact and changes nothing in it. Where the two might be read differently, this addendum is the more precise statement.
> **ALL DATA IS SYNTHETIC.**

### 1. Target (owner decision D1)

- **Primary target:** the frozen Policy C label `is_overrun`, meaning `budget_gap_ils > 0.5` ILS. The 0.5 ILS is a rounding tolerance only.
  - It is **not** a commercially material overrun, and the model card says so.
- **Secondary target:** `is_material_overrun`, meaning `budget_gap_ils > 2%` of the budget in effect. It is reported only, and **never** affects selection (§10).
- **Rows:** the 1,538 Policy C eligible snapshots.
  - Fitting, tuning, calibration and selection use only rows with `split == "train"`.
  - `test` is used once (§9).
  - `demo` is used for inference only.

### 2. Feature sets (D2)

- **Primary: `F27`.** These are the 29 handoff features minus `budget_revisions_to_date` and `client_direct_share`. Both are `PROPOSED` in `product_field_mapping.md`, so the product cannot compute them today.
- **Research: `F29`.** All 29 handoff features. They run under the identical protocol, are reported side by side as research only, and are never used for selection or test evaluation.

Preprocessing is identical for all candidates and always fitted inside the training partition:

| Feature type | Treatment |
|---|---|
| Categorical: `stage`, `archetype`, `budget_vat_mode`, `fee_mode` | One-hot, with `handle_unknown="ignore"` |
| Boolean: `fee_included` | 0/1 |
| Every other feature, including `event_month` | Numeric, standardised (`StandardScaler`) |

Nothing is imputed: the handoff has no nulls (`G-NULLS`).

### 3. Candidates

| Order (simplest first) | Id | Model |
|---|---|---|
| 1 | `B0` | Empirical bucket model (§4). No hyper-parameters, no extra calibration |
| 2 | `LR` | `LogisticRegression` (L2, solver `lbfgs`, `max_iter=10000`), calibrated (§6) |
| 3 | `GB` | `GradientBoostingClassifier` (`random_state=42`), calibrated (§6) |

The hard naive rule (`current_position_ils > 0`) is a **reference only**, reported per stage. It is not a candidate.

### 4. B0 definition (D3)

**Ratio.** `r = current_position_ils / budget_pre_vat`. This is the current position relative to the budget in effect at the snapshot.

**Buckets.** There are 6 right-closed buckets with these boundaries:

| Bucket | Range of `r` |
|---|---|
| 0 | `r ≤ −0.30` |
| 1 | `−0.30 < r ≤ −0.15` |
| 2 | `−0.15 < r ≤ −0.05` |
| 3 | `−0.05 < r ≤ 0` |
| 4 | `0 < r ≤ 0.05` |
| 5 | `r > 0.05` |

The boundary at 0 separates exactly the two sides of the hard rule.

**Cells.** A cell is a `(stage, bucket)` pair, giving 4 × 6 = 24 cells.

**Probability.** Additive (Laplace) smoothing with **α = 1**: `p = (k + 1) / (n + 2)`, where `k` is the number of overruns and `n` the number of rows in the cell. Both are counted on the **training partition only**. A cell with no training rows gets 0.5.

### 5. Hyper-parameter grids (fixed)

| Candidate | Grid, searched in this order |
|---|---|
| `B0` | none |
| `LR` | `C` ∈ (0.01, 0.1, 1.0, 10.0) |
| `GB` | `n_estimators` ∈ (100, 300) × `max_depth` ∈ (2, 3) × `learning_rate` ∈ (0.05, 0.1), iterated in that nested order (8 settings) |

All other parameters are the defaults of the pinned scikit-learn.

**Tuning score.** The mean Brier score over the inner tuning splits (§6). A tie keeps the earlier setting in grid order.

### 6. Group-disjoint resampling (correction C2)

Every resampling step is built **explicitly** from the production ids of the current training partition only. Nothing relies on the outer folds, or on scikit-learn's own handling of groups, to keep an inner step group-disjoint.

**Group partition `K(P, k, seed)`.**
- Take the sorted unique `production_id`s of the partition `P`.
- Permute them with `numpy.random.default_rng(seed)`.
- The production at permuted position `i` goes to part `i mod k`.
- Each part in turn is the held-out side.

**Outer CV.** The 5 frozen `cv_fold`s of `train`. They are never re-drawn.

**Tuning (LR, GB).** Inside each outer training partition `T`:
- the inner tuning splits are `K(T, 3, [42, f, 1])`, where `f` is the outer fold;
- every grid setting is fit and calibrated on each inner-train side, and scored on the held-out side.

**Calibration (LR, GB). This is a controlled manual procedure.** Every fit of a calibrated model on a row set `S` works as follows:
- It uses the group splits `K(S, 3, [42, f, 2, j])`. Here `j` is the inner tuning split index, and `j = 3` for a fit on the whole outer training partition.
- On each split, the base pipeline (preprocessing plus model) is fit on the inner-train side only.
- A sigmoid (Platt) calibrator is fit on the base model's decision values for the held-out side only: a one-feature `LogisticRegression` with `C = 1e12`.
- The model's probability is the mean of the 3 calibrated base models.
- `CalibratedClassifierCV` is **not** used, so no grouping guarantee is assumed from the library.

**Final refit on all of train.**
- Tuning uses `K(train, 3, [42, -1, 1])`.
- Calibration uses `K(·, 3, [42, -1, 2, j])`.

**Split recording.** Every split actually used is recorded with the partition it came from. A test fails if any split puts one production on both sides, or uses a row outside its partition.

### 7. Metrics

- **Primary:** Brier score (`sklearn.metrics.brier_score_loss`).
- **PR-AUC:** defined as `sklearn.metrics.average_precision_score`.
- **Also reported:**
  - log loss (`sklearn.metrics.log_loss`);
  - ROC-AUC (`roc_auc_score`);
  - reliability: 10 equal-width bins on [0, 1], with ECE = Σ_b (n_b / N) · |mean p_b − mean y_b|.
- **Threshold for precision, recall and accuracy:** **0.5**. It is fixed here, never chosen on any data, and never chosen on test (C5).
- **Mean grouped-CV metric:** the unweighted mean of the 5 per-fold values. The standard deviation across folds is also reported.
- **Per stage:** each metric is computed within each fold and stage, then averaged over folds. A metric that is undefined in a fold and stage (only one class) is left out of that mean, and the number left out is reported.

### 8. Selection rule (exact, correction C1)

The candidates are `B0`, `LR` and `GB` on `F27`, with target `is_overrun`.

1. Compute each candidate's mean grouped-CV Brier `b_c`, and the best value `b* = min_c b_c`.
2. **Tie set:** `T = {c : b_c − b* ≤ 0.005}`. The tie is an absolute difference in mean grouped-CV Brier of 0.005 or less **from the best candidate**. This also fixes the order for three candidates: every candidate within 0.005 of the best is tied with it, whatever the distances between the others. Candidates more than 0.005 from the best are eliminated before any tie-break.
3. If `T` has one member, it is selected.
4. **Tie-break 1:** within `T`, the highest mean grouped-CV PR-AUC (`average_precision_score`) wins.
5. **Tie-break 2:** if several members of `T` share that highest PR-AUC (equal when rounded to 6 decimals), the simplest of them wins: `B0 < LR < GB`.

Selection is applied by deterministic code. No agent and no LLM can choose, override or re-rank.

### 9. Final fit, test and demo

1. The selected candidate is refit on all `train` rows (§6), and its serialised bytes are hashed.
2. The secondary model is fitted next, also on all `train` rows, before test is opened:
   - the same candidate type;
   - target `is_material_overrun`;
   - the same procedure.
3. Only then is **test** opened, **once**, for both models together. A guard refuses a second opening. Test labels are not parsed before this step.
4. **Test reporting:**
   - Brier, log loss, ROC-AUC and PR-AUC, with precision, recall and accuracy at 0.5, overall and per stage;
   - reliability bins;
   - a **production-level bootstrap**: 2,000 resamples of test productions with replacement, seed 42, percentile 95% interval. Resamples where a metric is undefined are skipped and counted.
5. **Demo** gets predictions only: no labels are read, and no metric is computed.

### 10. Secondary and research reporting

**`is_material_overrun`.**
- `B0`, `LR` and `GB` on `F27` run under the identical CV protocol and are reported side by side.
- None of this feeds back into the primary choice.
- Its single test result is the secondary model from §9.

**`F29`.**
- `B0`, `LR` and `GB` run under the identical CV protocol on `is_overrun`, reported side by side.
- `B0` uses only `stage`, `current_position_ils` and `budget_pre_vat`, so it is identical in both feature sets.
- CV only: never test, never selection.

### 11. Seed

The seed is 42 everywhere: the inner partitions, `GB` and the bootstrap.
