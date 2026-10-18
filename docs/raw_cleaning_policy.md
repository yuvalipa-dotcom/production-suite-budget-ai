# Raw data cleaning policy

> **ALL DATA IS SYNTHETIC.** These are generic data-quality rules for consumers of
> the raw export (`raw/` + `docs/raw_data_dictionary.md`). They apply to any record
> that fails the dictionary's conventions.

## Normalisation (always allowed, with evidence in the record itself)

- Leading and trailing spaces are removed from every field.
- A coded value is matched to its closed list after lower-casing and mapping spaces or hyphens to `_`. A value that still matches nothing is **invalid**, not guessed.
- Dates are accepted in ISO `YYYY-MM-DD`, and also in the unambiguous forms `YYYY/MM/DD` and `DD-Mon-YYYY`. Anything else is **invalid**.
- Where a record has a typed amount, a VAT mode and a pre-VAT amount: if the pre-VAT amount equals the typed gross amount and the mode is `includes_vat`, the pre-VAT amount is recomputed with the documented 18% rate. Any other mismatch makes the amount **unusable**.
- Two rows with the same key that are identical after normalisation are one record. Two rows with the same key and **different** content are a conflict: neither is used.

## 1 · Target rule

The final-cost target is:
- the verified final cost (`final_pre_vat`) of every non-cancelled commitment;
- plus every manual cost amount;
- plus the contractual fee where it is included in the budget, against the approved pre-VAT budget in effect at the snapshot.

A cost record whose amounts are present **stays in the target**, even when another field of it (for example the category or a date) is missing or invalid. A record is never dropped from the target merely because it has a data-quality problem.

## 2 · Target-unreliable safeguard

If a cost record lacks a usable amount, or its cancellation status cannot be determined (an invalid `cancelled_date`), the production's target is **unreliable**:
- **every** snapshot of that production, at all stages, is ineligible;
- a "verified final cost" is never computed from an incomplete ledger.

A problem in a production's master record, budget revision, category plan or snapshot schedule is treated the same way.

## 3 · Feature-eligibility rule

A snapshot is **ineligible** when any of its features cannot be rebuilt exactly from records available as of its `as_of_date`.

- **Commitment with a missing or invalid field** (other than its dates): the snapshot is ineligible wherever that commitment is **active**, meaning `selected_date ≤ as_of` and not cancelled on or before `as_of`. Snapshots before it is active, or after its cancellation, stay eligible.
- **Commitment without a usable `selected_date`:** its availability cannot be placed in time, so every snapshot of the production is ineligible.
- **Manual cost without a usable `created_date`:** its availability date cannot be established, so every snapshot of the production is ineligible. Its amount still enters the target (rule 1).
- **Manual cost with another missing or invalid field:** every snapshot on or after its `created_date` is ineligible.

Each ineligible snapshot carries a reason code (for example `commitment_category_missing`, `manual_cost_undated`, `target_unreliable`). Codes are joined with `+` when several apply.

## 4 · Payments

Payments feed **neither** the target **nor** any feature. A payment row with a missing, negative, out-of-range or invalid value is:
- recorded in the cleaning log;
- excluded from the cleaned payments table.

It has **no** effect on eligibility.

## 5 · No fabrication

**No value is imputed, inferred or filled.** A missing field stays empty in the cleaned output. An unavailable feature means the snapshot is excluded, never estimated.

## 6 · Frozen groups

Cleaning never changes a production's train/test/demo group or its cross-validation fold. If some of a production's snapshots are ineligible, its remaining snapshots stay in the production's original group.

## Outputs

- The cleaned tables.
- A cleaning log (table, record key, problem list).
- `target_reliable` per production.
- Per snapshot: `eligible` (true/false) and a `reason` code.
