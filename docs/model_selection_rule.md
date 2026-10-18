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
