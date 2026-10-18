# Feature availability contract: features versus outcome-only columns

> **ALL DATA IS SYNTHETIC.** Machine-readable version: [`contracts/column_roles.json`](../contracts/column_roles.json).

Every column of every table has **exactly one** role:

| Role | Meaning |
|---|---|
| `feature` | Known at prediction time. May be used, directly or through a derived variable, to build snapshot features, subject to the as-of rule below |
| `outcome-only` | Known only after financial close. May be used **only to construct targets** |
| `metadata` | Identifiers, provenance, evaluation-protocol fields, or columns used by neither features nor targets. Never a model input |

## Outcome-only columns

| Table | Column | Why |
|---|---|---|
| `commitments` | `final_pre_vat` | verified final supplier cost, known only at financial close |
| `productions` | `financial_close_date` | known only after close |
| `productions` | `fee_concession_pre_vat` | granted at close; affects company profit only |
| `productions` (clean) | `final_budget_entered`, `n_budget_revisions`, `final_cost_pre_vat`, `final_fee_contract_pre_vat`, `final_company_profit_pre_vat` | aggregates of the whole life cycle |
| `snapshots` | `final_cost_pre_vat`, `budget_gap_ils`, `is_overrun`, `is_material_overrun` | targets |

> ⛔ **Outcome-only columns must never enter a snapshot feature, directly or through any derived variable.**

## As-of rule (for `feature` columns)

A record is available to a snapshot only if its date is on or before the snapshot's `as_of_date`:
- `selected_date` for commitments;
- `created_date` for manual costs;
- `effective_date` for budget revisions and category plans.

A cancellation counts only if `cancelled_date` is on or before `as_of_date`.

## Payments

Payment columns are `metadata`. Payments are cash flow: they feed neither the target nor any feature.

## How the contract is enforced (tests)

- **Declared trace:** every feature's source columns are declared (`tests/reconstruct.py`, `FEATURE_SOURCES`), and a test fails if any source is outcome-only.
- **Dynamic trace:** features are rebuilt from tables with **every outcome-only column removed**. They must be identical, and the rebuild must not fail.
- **Mutation tests:**
  - perturbing `final_pre_vat` changes no feature, and changes each signed gap by exactly the perturbation;
  - a label flips only where its threshold is crossed (both directions are covered);
  - perturbing `financial_close_date` changes nothing at all.
- **Coverage:** every column of every table appears in the contract exactly once, with one of the three roles.

The dataset's own `schema.json` is frozen as part of the committed v2 dataset, and keeps its original `feature` / `target` / `metadata` roles for the snapshot table. This contract extends those roles to every table, without changing a byte of the dataset.
