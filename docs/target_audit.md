# Target audit (P1-R, section A)

> **ALL DATA IS SYNTHETIC.** Numbers below are generated from the committed
> dataset by `tools/reports.py`; a test fails if they are out of date.

## A1 · How the v1 generator computes the label and the signed gap (as found)

Source: `generator/snapshots.py` (`snapshot_row`) and `generator/simulate.py` (`Production`).

| Input | Value used for snapshot *s* (day *d*) | Code |
|---|---|---|
| Budget | the approved budget **in effect at *d***: the initial budget, replaced by each revision whose day ≤ *d* (last one wins), normalised to pre-VAT with the budget's VAT mode | `Production.budget_pre_vat_at(d)` |
| Fee | the **contractual** fee known at *d*: `percent` → percent of the typed budget in effect at *d*; `amount` → the fixed typed amount. Rounded to agorot, then normalised with the fee's VAT mode. No later concession is applied | `Production.fee_pre_vat_at(d)` |
| Fee inclusion | added to the cost side **only if** `fee_included` is true | `snapshot_row` |
| Cost | simulated **verified final** supplier costs of all non-cancelled commitments, plus every manual cost (pre-VAT). The same value for every snapshot of a production | `Production.final_cost_pre_vat()` |
| Signed gap | `final cost + (fee if included) − budget` | `snapshot_row` |
| Label | `gap > 0.5` (rounding tolerance only); the 2% materiality is a separate flag | `snapshot_row` |

**Why the label can differ between snapshots of one production.** The final cost is constant per production. A label changes only when the approved budget in effect changes between stages, through a budget revision. With a `percent` fee, the contractual fee follows the revised budget. A later fee concession is never used.

<!-- BEGIN GENERATED: target-audit -->
- Productions: 400 · snapshot rows: 1600 · budget-revision records: 552 (400 originals + 152 later revisions).
- Later revisions per production: 0 → 285 productions, 1 → 78 productions, 2 → 37 productions.
- Productions whose label changes across stages: **37**; all of them have at least one budget revision (37 of 37).

Examples (first three, ordered by id):

**SYN-0009**

| stage | as_of_date | budget_pre_vat | fee_included | fee_mode | fee_contract_pre_vat | final_cost_pre_vat | budget_gap_ils | is_overrun |
|---|---|---|---|---|---|---|---|---|
| S0 | 2024-02-22 | 344915.25 | False | percent | 47943.22 | 345273.66 | 358.41 | True |
| S1 | 2024-05-15 | 367796.61 | False | percent | 51123.73 | 345273.66 | -22522.95 | False |
| S2 | 2024-07-25 | 367796.61 | False | percent | 51123.73 | 345273.66 | -22522.95 | False |
| S3 | 2024-10-03 | 367796.61 | False | percent | 51123.73 | 345273.66 | -22522.95 | False |

**SYN-0013**

| stage | as_of_date | budget_pre_vat | fee_included | fee_mode | fee_contract_pre_vat | final_cost_pre_vat | budget_gap_ils | is_overrun |
|---|---|---|---|---|---|---|---|---|
| S0 | 2025-11-28 | 363559.32 | True | percent | 34538.14 | 389001.79 | 59980.60 | True |
| S1 | 2026-02-26 | 363559.32 | True | percent | 34538.14 | 389001.79 | 59980.60 | True |
| S2 | 2026-05-14 | 450847.46 | True | percent | 42830.51 | 389001.79 | -19015.16 | False |
| S3 | 2026-07-30 | 450847.46 | True | percent | 42830.51 | 389001.79 | -19015.16 | False |

**SYN-0014**

| stage | as_of_date | budget_pre_vat | fee_included | fee_mode | fee_contract_pre_vat | final_cost_pre_vat | budget_gap_ils | is_overrun |
|---|---|---|---|---|---|---|---|---|
| S0 | 2024-01-04 | 264406.78 | True | amount | 29661.02 | 243839.67 | 9093.91 | True |
| S1 | 2024-03-25 | 264406.78 | True | amount | 29661.02 | 243839.67 | 9093.91 | True |
| S2 | 2024-06-04 | 264406.78 | True | amount | 29661.02 | 243839.67 | 9093.91 | True |
| S3 | 2024-08-11 | 301694.92 | True | amount | 29661.02 | 243839.67 | -28194.23 | False |
<!-- END GENERATED: target-audit -->

## Conformance with the approved definition

| Requirement | v1 | Evidence |
|---|---|---|
| **A2**: label and signed gap use the verified final costs, the contractual fee known at *s* (if included), and the approved budget in effect at *s*; any positive excess is an overrun; later concessions or revisions never change an earlier label | **met** | the table above; `tests/test_versions_and_target_audit.py` recomputes every label and gap independently; a revision dated after a snapshot changes nothing in it |
| **A3**: the original approved budget and every later revision preserved as separate records with effective dates | **not met** | v1 stored only `initial_budget_entered`, `final_budget_entered` and a revision count; intermediate revisions and all effective dates were not exported |

## A4 · Decision: dataset **v2**

A3 was not met, so this is a genuine semantic defect. **v2** was emitted, and **v1 is kept byte-identical** for traceability (a test regenerates and compares it).

v2 adds two ledgers:
- `budget_revisions.csv`: revision 0 = the original approved budget at the contract date, then every later approved revision, with `effective_date`, typed and pre-VAT amounts, and the approved guest count;
- `category_plans.csv`: the category plans as typed, with VAT mode and effective date. These are needed to rebuild the plan-based features from records.

v2 also adds the raw export (`raw/`) with its own manifest.

**Unchanged:**
- production ids, split groups and cross-validation folds (tested);
- every calibration parameter;
- the snapshot table and every v1 file, which are byte-identical in v2 (tested).

**Active dataset version: v2.**
