# Deterministic quality gate (P2)

> **ALL DATA IS SYNTHETIC.** The gate checks data integrity and contract compliance offline. It validates no model and says nothing about real-world predictive accuracy.

P2 has two stages. The **source gate** checks the raw export before anything uses it. The **handoff gate** checks a Crew-1 handoff (`clean_data.csv` plus `dataset_contract.json`) before any model consumes it. In the P5 Flow, the source gate runs before the handoff gate, and the handoff gate runs before model consumption.

```
python -m gate.run --stage source  --raw data/synthetic/v2/raw --out <dir>
python -m gate.run --stage handoff --raw data/synthetic/v2/raw --out <dir> \
    --data <handoff>/clean_data.csv --contract <handoff>/dataset_contract.json
```

Each run writes `gate_result.json` and `gate_report.md`. Both are byte-stable for identical inputs: sorted keys, LF line endings, no timestamps.

**Exit codes:** `0` every HARD check passed · `1` at least one HARD check failed or could not run, including a missing or unreadable input · `2` invalid command line.

## What the gate reads at runtime

- The raw export directory (its seven tables and `manifest.json`).
- The delivered handoff (`clean_data.csv`, `dataset_contract.json`).
- Three committed contracts: `contracts/column_roles.json`, `contracts/dataset_contract.schema.json` and `contracts/frozen_split.json`.

**Nothing else.** The committed clean snapshots, the ground truth and the test fixtures are test-only. Every expected value is rebuilt from the raw export under Policy C ([`raw_cleaning_policy.md`](raw_cleaning_policy.md)). A test traces every file the gate opens, and fails if any of these test-only sources is read.

## The handoff

`clean_data.csv` has one row per **eligible** `(production_id, stage)` snapshot. That is 1,538 of 1,600; the 62 Policy C exclusions are not delivered. Its columns are fixed by `gate/handoff_spec.py`, and every one is required and non-nullable:

- **id:** `production_id`
- **metadata:** `as_of_date`, `split`, `cv_fold`
- **features:** the 29 snapshot features of [`feature_availability.md`](feature_availability.md), including `stage`
- **targets:** `final_cost_pre_vat`, `budget_gap_ils`, `is_overrun`, `is_material_overrun`

Values use the dataset CSV format: amounts rounded to 2 decimals, booleans written as `True` or `False`.

`dataset_contract.json` must satisfy `contracts/dataset_contract.schema.json`. It must declare the right raw provenance, row count and exclusion counts, and its column declarations must agree with the approved specification. **A contract that is merely consistent with its own data does not pass.** Every column, role, dtype, unit, eligibility decision and value is checked independently against the specification and the raw reconstruction.

## Frozen split reference

The raw export does not carry groups or folds. `contracts/frozen_split.json` (version 1.0.0) records the existing frozen assignments for all 400 productions:

- 280 train, 80 test, 40 demo;
- train folds 0–4 with 56 productions each;
- `cv_fold = -1` for every test and demo production.

It is **a contract, not a dataset version**, and no assignment was changed. `python -m tools.build_frozen_split` builds it from the committed v2 assignments, which must equal v1, and cross-checks them against a fresh run of the seeded rules in `generator/split.py`. If the two sources disagree, the build stops instead of choosing one. `--check` verifies that the committed file equals a fresh build, and `tests/test_frozen_split.py` proves the equivalences.

## Checks

A HARD check that fails or cannot run fails the gate. WARN and INFO checks never fail it. A check whose input is unavailable is reported as `NOT_RUN`, and the remaining checks still run wherever their own inputs exist. A check that raises an error is reported as `FAIL`. It is never skipped.

### Source gate

| ID | Validates |
|---|---|
| `G-SRC-MANIFEST` | every raw table is listed in the manifest, present, hash-identical, and has the declared row count |
| `G-SRC-ROLES` | every raw column has a role in `column_roles.json` |
| `G-SRC-OUTCOME` | the outcome-only columns `commitments.final_pre_vat` and `productions.financial_close_date` are declared outcome-only |
| `G-SRC-FROZEN-SPLIT` | the frozen split contract is valid and covers exactly the raw production ids |
| `I-SYNTHETIC` | (INFO) the manifest and every production are flagged synthetic |

### Handoff gate

| ID | Validates |
|---|---|
| `G-IN-FILES` | every required input exists and parses |
| `G-CONTRACT-SCHEMA` | the contract satisfies the JSON schema; column names are unique; the primary key is declared |
| `G-CONTRACT-PROVENANCE` | `raw_manifest_sha256` matches the raw manifest, and the raw files match the manifest |
| `G-SCHEMA-COLUMNS` | data columns equal contract columns, and every required column is present |
| `G-SCHEMA-DTYPES` | contract dtypes and units equal the approved specification; values parse as their dtype |
| `G-ROLES` | contract roles equal the approved specification; no undeclared id or target |
| `G-LEAKAGE-OUTCOME` | no feature is an outcome-only column, is derived from one, or lacks declared raw provenance; no outcome-only column is smuggled in under another role; `outcome_only_exclusions` lists exactly the outcome-only columns |
| `G-KEY-UNIQUE` | `(production_id, stage)` is present, valid and unique |
| `G-ELIGIBILITY-SET` | the delivered snapshots are exactly the Policy C eligible set |
| `G-ELIGIBILITY-COUNTS` | the contract's `row_count` and `exclusion_counts` equal the delivery and Policy C, and together cover every scheduled snapshot |
| `G-TARGET-GAP` | `budget_gap_ils` and `final_cost_pre_vat` equal the raw reconstruction (±0.01) |
| `G-TARGET-LABEL` | `is_overrun` and `is_material_overrun` equal the raw reconstruction |
| `G-TARGET-CONSISTENCY` | each label agrees with its own row's gap (gap > 0.5; gap > 2% of budget). Rows inside a ±0.02 rounding band around a threshold are not judged here; `G-TARGET-LABEL` still checks them exactly |
| `G-FIN-RECON` | every ledger-derived feature (D1) equals the raw reconstruction |
| `G-FIN-ASOF-BUDGET` | `budget_pre_vat` equals the approved budget in effect at the snapshot date |
| `G-NONFIN-RULES` | every non-ledger feature (D2) and `as_of_date` equal the production record and the snapshot schedule |
| `G-SPLIT-GROUP` | each production's `split` equals `contracts/frozen_split.json` |
| `G-SPLIT-FOLD` | each production's `cv_fold` equals `contracts/frozen_split.json` |
| `G-NULLS` | no empty value in any required, id, feature, target or non-nullable column, whatever the contract declares |
| `G-RANGES` | approved ranges and allowed values, plus any contract `min`, `max` and `allowed` |
| `W-PREVALENCE-IMBALANCE` | (WARN) row-level overrun rate per group, rebuilt from raw. It is reported both **before** Policy C eligibility filtering (1,600 snapshots) and **after** it (1,538). It warns when test and train differ by 5 percentage points or more. The documented P1-R figure (+15.0 pp in [`split_report.md`](split_report.md)) is quoted as historical context, not recomputed |
| `I-SYNTHETIC` | (INFO) synthetic-data notice |

## Corruption demonstrations

`tests/test_handoff_gate.py` builds an independent reference handoff. Its values come from the committed clean snapshots, its eligibility from the ground truth, and its column declarations from the dataset `schema.json`. All 19 approved corruptions are applied to copies of it. Where it applies, the copied contract is edited too, so that it stays internally consistent. Each case must fail the gate with exit code 1 and fail its required check.

| # | Corruption | Required failing check |
|---|---|---|
| 1 | one financial amount column × 1,000 | `G-FIN-RECON` |
| 2 | budget and committed amounts × 1,000, ratios preserved | `G-FIN-RECON` |
| 3 | a pre-VAT amount replaced by its VAT-inclusive value (repository VAT rate) | `G-FIN-RECON` |
| 4 | signed gaps shifted, labels kept consistent | `G-TARGET-GAP` |
| 5 | a label and its gap sign flipped together | `G-TARGET-LABEL` and `G-TARGET-GAP` |
| 6 | a label flipped alone | `G-TARGET-CONSISTENCY` |
| 7 | as-of budget replaced by the final approved budget | `G-FIN-ASOF-BUDGET` |
| 8 | as-of budget replaced by the original budget after a revision | `G-FIN-ASOF-BUDGET` |
| 9 | an ineligible snapshot inserted, contract counts updated | `G-ELIGIBILITY-SET` |
| 10 | an eligible snapshot removed, contract counts updated | `G-ELIGIBILITY-SET` |
| 11 | a production moved to another split group | `G-SPLIT-GROUP` |
| 12 | a training fold reassigned | `G-SPLIT-FOLD` |
| 13 | `final_pre_vat` added as a prediction feature | `G-LEAKAGE-OUTCOME` |
| 14 | a column renamed (in data and contract) | `G-SCHEMA-COLUMNS` |
| 15 | a column removed from the contract only | `G-SCHEMA-COLUMNS` or `G-CONTRACT-SCHEMA` |
| 16 | an incorrect raw provenance hash | `G-CONTRACT-PROVENANCE` |
| 17 | a duplicated `(production_id, stage)` key | `G-KEY-UNIQUE` |
| 18 | a null in a required feature (contract marks it nullable) | `G-NULLS` |
| 19 | a required input file removed | `G-IN-FILES` (exit code 1) |

## Limitations

- **The reconstruction and the test oracle share code.** The gate's Policy C and reconstruction modules (`gate/policy_c.py`, `gate/reconstruct.py`) are copies of the P1-R2 test oracle (`tests/raw_repair.py`, `tests/reconstruct.py`). Independence in the tests therefore comes from the generator's committed clean snapshots and the ground truth, not from a second implementation of the rules.
- **The amount tolerance is ±0.01 ILS**, so a corruption smaller than that per cell is not detected.
- **This is synthetic data only.** Passing the gate is not evidence of real-world accuracy.
