# Crew 2 evaluation report: Flow TRAIN-only run

> **ALL DATA IS SYNTHETIC.** Every output is an experimental / demo prediction, produced by the Flow's TRAIN-only Crew 2 step. It is not evidence about real productions.

This report has two separate parts. **Part A** is computed by this run, from TRAIN rows only. **Part B** is the historical P4 TEST result, pinned and cited; this run did not reopen TEST.

## A. Current TRAIN-only grouped cross-validation (this run)

### A.1 Input and provenance

| item | value |
|---|---|
| input | Crew 1 file handoff (`clean_data.csv`, `dataset_contract.json`), validated by the Flow contract `flow-1` (0 violations) |
| `clean_data.csv` sha256 | `2b8353247ebd7c5e9c63acec4c8a0bbc773380e846bd8c59ba21b58b75aa77e8` |
| `dataset_contract.json` sha256 | `d8774b8dd55b835822e1f0f1179734be536b647b2f0111ec14f2ee59bc5f719c` |
| rows used for fitting | train: 1,084 rows, 278 productions |
| rows never used for fitting | test: 306 rows (labels redacted, vault sealed); demo: 148 rows (labels redacted) |
| train prevalence of `is_overrun` | 0.4446 |
| protocol | 5 frozen production-grouped folds; tuning and Platt calibration on 3 group-disjoint inner splits of each training partition; seed 42 |
| pre-registration | `docs/model_selection_rule.md`, Addendum P4-1 (verified unchanged before fitting) |
| reference | `p4_reference.v1` (approved run `p4-offline-002`) |

### A.2 Candidate comparison (primary: F27, `is_overrun`)

| run | candidate | target | Brier | PR-AUC | ROC-AUC | log loss |
|---|---|---|---|---|---|---|
| `F27.is_overrun.B0` | B0 | `is_overrun` | 0.2069 ± 0.0142 | 0.6579 ± 0.0452 | 0.7071 ± 0.0385 | 0.5938 ± 0.0349 |
| `F27.is_overrun.LR` | LR | `is_overrun` | 0.1884 ± 0.0147 | 0.7570 ± 0.0779 | 0.7891 ± 0.0414 | 0.5565 ± 0.0346 |
| `F27.is_overrun.GB` | GB | `is_overrun` | 0.1992 ± 0.0055 | 0.7321 ± 0.0517 | 0.7624 ± 0.0152 | 0.5825 ± 0.0123 |

Values are mean ± sd over the 5 grouped folds. Grids: B0: 1 setting(s); LR: 4 setting(s); GB: 8 setting(s).

### A.3 Selection (pre-registered rule, applied by code)

- Selected: **LR**, decided by: lowest mean grouped-CV Brier.
- Best mean Brier: 0.188450; tie margin 0.005; tie set: LR.
- Brier minus best: B0 0.018436, LR 0.000000, GB 0.010713.

### A.4 Final models (refit on all of TRAIN)

| model | target | candidate | parameters | train rows | audit violations | sha256 |
|---|---|---|---|---|---|---|
| primary | `is_overrun` | LR | {'C': 10.0} | 1,084 | 0 | `bc7b312e127b01d12999998f4b680180dd5150a2bf199e35c582ac5dd3252d81` |
| secondary | `is_material_overrun` | LR | {'C': 10.0} | 1,084 | 0 | `55a5c9e39c7e532f2583a07a72c65df0ccdc20314e09ca528ee557e98c47ee36` |

### A.5 Reported only: secondary target and research features

Never used for selection.

| run | candidate | target | Brier | PR-AUC | ROC-AUC | log loss |
|---|---|---|---|---|---|---|
| `F27.is_material_overrun.B0` | B0 | `is_material_overrun` | 0.1911 ± 0.0179 | 0.5730 ± 0.0652 | 0.7076 ± 0.0495 | 0.5590 ± 0.0421 |
| `F27.is_material_overrun.LR` | LR | `is_material_overrun` | 0.1802 ± 0.0164 | 0.6562 ± 0.1092 | 0.7625 ± 0.0543 | 0.5389 ± 0.0384 |
| `F27.is_material_overrun.GB` | GB | `is_material_overrun` | 0.1957 ± 0.0185 | 0.6245 ± 0.0953 | 0.7192 ± 0.0569 | 0.5763 ± 0.0412 |

| run | candidate | target | Brier | PR-AUC | ROC-AUC | log loss |
|---|---|---|---|---|---|---|
| `F29.is_overrun.B0` | B0 | `is_overrun` | 0.2069 ± 0.0142 | 0.6579 ± 0.0452 | 0.7071 ± 0.0385 | 0.5938 ± 0.0349 |
| `F29.is_overrun.LR` | LR | `is_overrun` | 0.1898 ± 0.0149 | 0.7534 ± 0.0908 | 0.7863 ± 0.0495 | 0.5605 ± 0.0338 |
| `F29.is_overrun.GB` | GB | `is_overrun` | 0.2002 ± 0.0073 | 0.7246 ± 0.0592 | 0.7597 ± 0.0150 | 0.5848 ± 0.0156 |

### A.6 Reproducibility against the approved P4 CV

- Mode: **exact**: exact equality (pinned reference environment): every CV summary, fold score and TRAIN prediction is identical to the approved P4 run, and both final models are byte-identical (sha256).
- Selected model, decision, tie set, per-fold and final parameters: identical to `p4-offline-002`.
- Maximum absolute deviation: CV means 0; fold scores 0; TRAIN predictions 0.
- Final models byte-identical to the P4 models: primary yes, secondary yes.

### A.7 Naive rule reference (TRAIN, not a candidate)

`current_position_ils > 0` on all train rows: Brier 0.3782, accuracy 0.6218.

## B. Historical P4 TEST evaluation (cited, not recomputed)

> **HISTORICAL.** These numbers are pinned from the approved P4 run `p4-offline-002` (`budget_flow/contracts/p4_reference.v1.json`). TEST was opened once, in P4, after the final models were frozen. **This run did not reopen TEST**: its test vault is sealed, was opened 0 times, and no test metric was computed.

The final models of this run are byte-identical to the models evaluated in P4, so this historical result applies to them.

Test: 306 rows from 80 productions; threshold 0.5 for the threshold metrics; 95% production-level bootstrap intervals (2000 resamples).

| model | target | candidate | rows | Brier [95% CI] | PR-AUC [95% CI] | ROC-AUC | log loss | ECE |
|---|---|---|---|---|---|---|---|---|
| primary | `is_overrun` | LR | 306 | 0.1751 [0.1471, 0.2045] | 0.8880 [0.8176, 0.9406] | 0.8385 | 0.5165 | 0.111 |
| secondary | `is_material_overrun` | LR | 306 | 0.1564 [0.1308, 0.1846] | 0.8143 [0.6976, 0.8993] | 0.8580 | 0.4755 | 0.068 |

Primary model by stage (historical):

| stage | rows | Brier | PR-AUC | prevalence |
|---|---|---|---|---|
| S0 | 80 | 0.2169 | 0.8382 | 0.6500 |
| S1 | 77 | 0.1646 | 0.9216 | 0.6234 |
| S2 | 76 | 0.1525 | 0.9145 | 0.5526 |
| S3 | 73 | 0.1640 | 0.8786 | 0.5342 |
