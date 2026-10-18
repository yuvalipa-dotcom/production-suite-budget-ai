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

## Product preference (stated separately from the rule)

For future product integration a simple calibrated logistic model is preferred, because it exports to plain JSON. If gradient boosting wins the rule, that is reported honestly. Shipping a different model in the product would be a separate, explicit owner decision, documented with the measured difference.
