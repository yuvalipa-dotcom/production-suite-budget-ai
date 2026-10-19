# Model card: budget overrun probability (Crew 2, P4)

> **About this document.** It is assembled by `crew2/tools/card.py` from the committed narrative generator (`crew2/offline.py`) and the `metrics.json` of the offline run `p4-offline-002`. No model was retrained, no prediction changed, and TEST was not reopened. The run's own artifact card is unchanged; this version adds the grid-boundary, calibration and product-field corrections approved at P4 readiness.

> **ALL DATA IS SYNTHETIC.** Every prediction of this model is an **experimental / demo prediction**. It was trained and
> evaluated only on synthetic productions and is not a validated real-world forecast.
>
> **Target disclosure.** The target `is_overrun` is the frozen Policy C label: the signed budget gap is
> above 0.5 ILS, a rounding tolerance only. It is **not a commercially material overrun**. The 2% material
> overrun (`is_material_overrun`) is reported as a secondary result and never affected model selection.
>
> Copyright (c) 2026 Yuval Lipa. All rights reserved. No open-source license is granted.

## Summary

<!-- narrative:summary -->
This card describes a LR model of the probability that a synthetic production ends above its budget in effect at the snapshot. On the single test evaluation its Brier score is 0.1751 and its PR-AUC is 0.8880. All data are synthetic and every output is an experimental / demo prediction.
<!-- /narrative -->

## Intended use and out-of-scope use

<!-- narrative:intended_use -->
The model supports an academic demonstration of the Production Suite budget card. It is not intended for decisions about real productions, and it is not validated on real data.
<!-- /narrative -->

## Data and target

<!-- narrative:data_and_target -->
The input is the Policy C handoff of 1,538 eligible snapshots. Training uses the 1,084 train rows only; test has 306 rows and demo 148 rows. The target is_overrun is a rounding-tolerance label and not a commercially material overrun.
<!-- /narrative -->

| Group | Rows | Productions | S0 | S1 | S2 | S3 |
|---|---|---|---|---|---|---|
| train | 1084 | 278 | 278 | 272 | 268 | 266 |
| test | 306 | 80 | 80 | 77 | 76 | 73 |
| demo | 148 | 39 | 39 | 37 | 36 | 36 |

## Features

<!-- narrative:features -->
The primary set has 27 snapshot features, each taken as of the snapshot date from the dated ledgers. Of these, 24 correspond to fields the product stores today or are derived from them, and three are only approximations there: lead_time_days and days_since_contract use the event creation date as a proxy for the contract date, and n_substitutions_to_date counts deleted quotes, which are not always substitutions. The 29-feature set adds two fields the product cannot compute today and is research only.
<!-- /narrative -->

Primary `F27` (27 features): `stage`, `archetype`, `event_month`, `lead_time_days`, `days_since_contract`, `days_to_event`, `guest_count`, `budget_vat_mode`, `budget_pre_vat`, `fee_included`, `fee_mode`, `fee_contract_pre_vat`, `planned_total_pre_vat`, `n_categories_planned`, `n_categories_committed`, `n_categories_pending`, `n_commitments_active`, `n_manual_costs`, `manual_costs_pre_vat`, `committed_pre_vat`, `n_categories_over_plan`, `over_plan_pre_vat`, `n_substitutions_to_date`, `no_vat_share`, `current_position_ils`, `committed_ratio`, `cost_per_guest_pre_vat`.

Research only `F29` adds `budget_revisions_to_date` and `client_direct_share` (not computable in the product today).

## Training and evaluation protocol

<!-- narrative:protocol -->
Tuning and calibration ran inside each of the 5 frozen production-grouped folds, on explicit group-disjoint inner splits of the training partition only. Test was opened once, after the final models were frozen, and demo was used for inference only.
<!-- /narrative -->

## Cross-validation results

<!-- narrative:cv_results -->
On grouped CV the mean Brier is 0.2069 for B0, 0.1884 for LR and 0.1992 for GB. The research 29-feature runs score 0.1898 for LR and 0.2002 for GB, and were never used for selection.
<!-- /narrative -->

| Run | Candidate | Feature set | Target | Brier | PR-AUC (AP) | ROC-AUC | Log loss |
|---|---|---|---|---|---|---|---|
| `F27.is_overrun.B0` | B0 | F27 | `is_overrun` | 0.2069 ± 0.0142 | 0.6579 ± 0.0452 | 0.7071 ± 0.0385 | 0.5938 ± 0.0349 |
| `F27.is_overrun.GB` | GB | F27 | `is_overrun` | 0.1992 ± 0.0055 | 0.7321 ± 0.0517 | 0.7624 ± 0.0152 | 0.5825 ± 0.0123 |
| `F27.is_overrun.LR` | LR | F27 | `is_overrun` | 0.1884 ± 0.0147 | 0.7570 ± 0.0779 | 0.7891 ± 0.0414 | 0.5565 ± 0.0346 |

Mean ± sd across the 5 frozen production-grouped folds (train only).

| Run | Brier S0 | Brier S1 | Brier S2 | Brier S3 |
|---|---|---|---|---|
| `F27.is_overrun.B0` | 0.2500 | 0.2460 | 0.2203 | 0.1081 |
| `F27.is_overrun.GB` | 0.2368 | 0.2184 | 0.2143 | 0.1250 |
| `F27.is_overrun.LR` | 0.2245 | 0.1953 | 0.1830 | 0.1493 |

Research (`F29`, never used for selection):

| Run | Candidate | Feature set | Target | Brier | PR-AUC (AP) | ROC-AUC | Log loss |
|---|---|---|---|---|---|---|---|
| `F29.is_overrun.B0` | B0 | F29 | `is_overrun` | 0.2069 ± 0.0142 | 0.6579 ± 0.0452 | 0.7071 ± 0.0385 | 0.5938 ± 0.0349 |
| `F29.is_overrun.GB` | GB | F29 | `is_overrun` | 0.2002 ± 0.0073 | 0.7246 ± 0.0592 | 0.7597 ± 0.0150 | 0.5848 ± 0.0156 |
| `F29.is_overrun.LR` | LR | F29 | `is_overrun` | 0.1898 ± 0.0149 | 0.7534 ± 0.0908 | 0.7863 ± 0.0495 | 0.5605 ± 0.0338 |

Mean ± sd across the 5 frozen production-grouped folds (train only).

## Model selection

<!-- narrative:selection -->
The pre-registered rule, applied by code, selected LR: lowest mean grouped-CV Brier. The best mean Brier was 0.1884 and the tie margin is 0.005. Every fold and the final refit chose C = 10, which is the upper boundary of the pre-registered grid; a larger C was never evaluated.
<!-- /narrative -->

| Candidate | Mean CV Brier | Brier minus best | Mean CV PR-AUC | In tie set |
|---|---|---|---|---|
| B0 | 0.2069 | 0.0184 | 0.6579 | no |
| GB | 0.1992 | 0.0107 | 0.7321 | no |
| LR | 0.1884 | 0.0000 | 0.7570 | yes |

**Selected:** LR (lowest mean grouped-CV Brier).

## Test result (single evaluation)

<!-- narrative:test_results -->
On test (306 rows from 80 productions) the Brier score is 0.1751 with a 95% bootstrap interval of [0.1471, 0.2045], and PR-AUC is 0.8880 [0.8176, 0.9406]. The expected calibration error on test is approximately 0.111 (10 bins), so the probabilities are not well calibrated at the test prevalence. This is the only test evaluation.
<!-- /narrative -->

Rows 306, productions 80, target `is_overrun`, prevalence 0.5915, candidate LR.

| Metric | Test value | 95% production-level bootstrap interval |
|---|---|---|
| Brier | 0.1751 | [0.1471, 0.2045] |
| Log loss | 0.5165 | [0.4459, 0.5924] |
| ROC-AUC | 0.8385 | [0.7745, 0.8966] |
| PR-AUC (AP) | 0.8880 | [0.8176, 0.9406] |
| Precision at p ≥ 0.5 | 0.8712 | |
| Recall at p ≥ 0.5 | 0.6354 | |
| Accuracy at p ≥ 0.5 | 0.7288 | |

| Stage | Rows | Brier | ROC-AUC | PR-AUC (AP) |
|---|---|---|---|---|
| S0 | 80 | 0.2169 | 0.7287 | 0.8382 |
| S1 | 77 | 0.1646 | 0.8621 | 0.9216 |
| S2 | 76 | 0.1525 | 0.8901 | 0.9145 |
| S3 | 73 | 0.1640 | 0.8522 | 0.8786 |

Expected calibration error (10 bins): 0.1107.

Reliability chart: `charts/reliability_test.png` in the local run folder `out/crew2/p4-offline-002/` (not committed).

## Secondary result: material overrun

<!-- narrative:secondary_results -->
For the 2% material overrun the same candidate type scores a test Brier of 0.1564 and a PR-AUC of 0.8143. It is reported only and never affected selection.
<!-- /narrative -->

| Run | Candidate | Feature set | Target | Brier | PR-AUC (AP) | ROC-AUC | Log loss |
|---|---|---|---|---|---|---|---|
| `F27.is_material_overrun.B0` | B0 | F27 | `is_material_overrun` | 0.1911 ± 0.0179 | 0.5730 ± 0.0652 | 0.7076 ± 0.0495 | 0.5590 ± 0.0421 |
| `F27.is_material_overrun.GB` | GB | F27 | `is_material_overrun` | 0.1957 ± 0.0185 | 0.6245 ± 0.0953 | 0.7192 ± 0.0569 | 0.5763 ± 0.0412 |
| `F27.is_material_overrun.LR` | LR | F27 | `is_material_overrun` | 0.1802 ± 0.0164 | 0.6562 ± 0.1092 | 0.7625 ± 0.0543 | 0.5389 ± 0.0384 |

Mean ± sd across the 5 frozen production-grouped folds (train only).

Rows 306, productions 80, target `is_material_overrun`, prevalence 0.4020, candidate LR.

| Metric | Test value | 95% production-level bootstrap interval |
|---|---|---|
| Brier | 0.1564 | [0.1308, 0.1846] |
| Log loss | 0.4755 | [0.4178, 0.5400] |
| ROC-AUC | 0.8580 | [0.7913, 0.9108] |
| PR-AUC (AP) | 0.8143 | [0.6976, 0.8993] |
| Precision at p ≥ 0.5 | 0.7976 | |
| Recall at p ≥ 0.5 | 0.5447 | |
| Accuracy at p ≥ 0.5 | 0.7614 | |

| Stage | Rows | Brier | ROC-AUC | PR-AUC (AP) |
|---|---|---|---|---|
| S0 | 80 | 0.2134 | 0.7289 | 0.6539 |
| S1 | 77 | 0.1670 | 0.8285 | 0.8203 |
| S2 | 76 | 0.1224 | 0.9219 | 0.8969 |
| S3 | 73 | 0.1182 | 0.9251 | 0.8913 |

Expected calibration error (10 bins): 0.0681.

## Limitations

<!-- narrative:limitations -->
All data are synthetic, so nothing here is evidence of real-world accuracy. The test group has an overrun prevalence of 59.2% against 44.5% on train; the split is frozen and was not changed, and the model was neither recalibrated nor retuned on test. Every fold and the final refit chose C = 10, which is the upper boundary of the pre-registered grid; a larger C was never evaluated. Three of the LR model's inputs are approximations in the product, so its behaviour there is untested.
<!-- /narrative -->
