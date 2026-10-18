# Raw data dictionary

> **ALL DATA IS SYNTHETIC.** These tables are a provider-style export of
> fictional event productions. No real customer, vendor or event data.

This dictionary describes the raw export only: its tables, fields, meanings, units, formats, valid values, keys and relationships. It is everything a consumer needs to ingest and validate the export.

## Conventions

| Topic | Convention |
|---|---|
| Currency | Israeli new shekel (ILS) |
| Number format | decimal point, two decimals for money (`12345.67`) |
| Dates | ISO 8601, `YYYY-MM-DD` |
| VAT rate | **18%** (0.18), constant for the whole export |
| Booleans | `True` / `False` |
| Coded values | lower-case, words joined by `_` (closed lists below) |
| Encoding | UTF-8, comma-separated, header row, LF line endings |
| Provenance | every row has `is_synthetic = True` and `batch_id = syn-v1-seed42` |

### VAT modes (per amount)

Every typed amount is stored **as typed**, together with the VAT mode in which it was typed:

| `vat_mode` | Meaning | Pre-VAT value |
|---|---|---|
| `includes_vat` | the typed amount contains VAT | typed ÷ 1.18 |
| `excludes_vat` | the typed amount is net; VAT is added on payment | typed |
| `no_vat` | VAT-exempt supplier; the typed amount is the full cost | typed |

Where a table has both a typed amount and a pre-VAT amount, the pre-VAT amount must equal the rule above, rounded to two decimals.

### Closed lists

- **Event type (`archetype`):** `wedding`, `bar_bat_mitzvah`, `corporate`, `conference`.
- **Cost category (`category`):** `venue_catering`, `music_entertainment`, `photo_video`, `decor_design`, `lighting_sound`, `attire_beauty`, `logistics_transport`, `gifts_hospitality`, `staffing_security`, `staging_av`, `printing_branding`.
- **Fee mode (`fee_mode`):**
  - `percent`: the fee is `fee_value`% of the approved budget as typed;
  - `amount`: the fee is a fixed typed amount, `fee_value`.
- **Payment responsibility:** `company` (paid by the production company) or `client_direct` (paid by the client directly to the supplier).
- **Manual cost kind:** `unexpected`, `cancellation_fee`.
- **Payment direction:** `out` (company → supplier).
- **Stage:** `S0`, `S1`, `S2`, `S3`.

## Tables

### `productions.csv`: one row per production

**Key:** `production_id`.

| Field | Type | Meaning |
|---|---|---|
| `production_id` | text | `SYN-####` |
| `batch_id`, `is_synthetic` | text, boolean | provenance |
| `archetype` | code | event type |
| `contract_date` | date | contract signed |
| `event_date` | date | event day |
| `financial_close_date` | date | production financially closed; final supplier costs are verified by this date |
| `budget_vat_mode` | code | VAT mode of the approved budget |
| `fee_mode`, `fee_value` | code, number | contractual production fee (see fee mode) |
| `fee_vat_mode` | code | VAT mode of the fee |
| `fee_included` | boolean | `True`: the fee is part of the approved budget. `False`: it is charged on top |
| `fee_concession_pre_vat` | ILS ≥ 0 | a fee reduction granted at close; affects the company's profit only |

Valid ranges and order: `contract_date < event_date < financial_close_date`, and `fee_value > 0`.

### `budget_revisions.csv`: approved budget history

**Key:** (`production_id`, `revision_no`).

| Field | Type | Meaning |
|---|---|---|
| `revision_no` | integer ≥ 0 | `0` = the original approved budget, then 1, 2, … for each later approval |
| `effective_date` | date | the date the approved budget takes effect (revision 0 = contract date) |
| `approved_entered_amount` | ILS > 0 | approved total budget, as typed |
| `vat_mode` | code | VAT mode of the typed budget |
| `approved_pre_vat_amount` | ILS > 0 | pre-VAT value of the approved budget |
| `guest_count` | integer > 0 | guest count approved together with this budget |

The budget in effect on a date is the revision with the highest `revision_no` among those with `effective_date` on or before that date.

### `category_plans.csv`: planned cost per category

**Key:** (`production_id`, `category`).

| Field | Type | Meaning |
|---|---|---|
| `category` | code | cost category |
| `planned_entered` | ILS ≥ 0 | planned cost, as typed |
| `vat_mode` | code | VAT mode of the planned cost |
| `effective_date` | date | date the plan was set |

### `commitments.csv`: supplier commitments

**Key:** `commitment_id`. Each row belongs to one `production_id`.

| Field | Type | Meaning |
|---|---|---|
| `commitment_id` | text | `SYN-####-C##` |
| `category` | code | cost category |
| `vat_mode` | code | VAT mode of the quote |
| `payment_responsibility` | code | who pays the supplier |
| `original_entered` | ILS > 0 | the accepted quote, as typed |
| `original_pre_vat` | ILS > 0 | pre-VAT value of the accepted quote |
| `selected_date` | date | the quote was accepted |
| `cancelled_date` | date or empty | the commitment was cancelled (empty = not cancelled) |
| `replaces_commitment_id` | text or empty | the cancelled commitment this one replaces |
| `final_pre_vat` | ILS ≥ 0 | **verified final supplier cost** (pre-VAT), after additions, reductions and credits; known only at financial close. `0` for cancelled commitments |

The accepted quote and the verified final cost are two separate amounts; the second never overwrites the first.

### `manual_costs.csv`: costs recorded directly

**Key:** `manual_cost_id`.

| Field | Type | Meaning |
|---|---|---|
| `manual_cost_id` | text | `SYN-####-M##` |
| `category` | code | cost category |
| `kind` | code | `unexpected` or `cancellation_fee` |
| `vat_mode` | code | VAT mode of the typed amount |
| `entered` | ILS > 0 | amount as typed |
| `pre_vat` | ILS > 0 | pre-VAT value |
| `created_date` | date | date the cost was recorded |

### `payments.csv`: company payments to suppliers

**Key:** `payment_id`.

| Field | Type | Meaning |
|---|---|---|
| `payment_id` | text | `SYN-####-P##` |
| `commitment_id` | text | the commitment paid |
| `direction` | code | `out` |
| `amount_gross` | ILS, 0 < x ≤ 1,000,000 | amount paid, gross |
| `paid_date` | date | payment date |

Payments are **cash flow, not cost**. Commitments paid by the client directly have no company payment.

### `snapshot_schedule.csv`: prediction dates

**Key:** (`production_id`, `stage`).

| Field | Type | Meaning |
|---|---|---|
| `stage` | code | `S0` contract day · `S1` 35% of lead time · `S2` 65% of lead time · `S3` 14 days before the event |
| `as_of_date` | date | the prediction date of that stage |

## Relationships

- `budget_revisions`, `category_plans`, `commitments`, `manual_costs` and `snapshot_schedule` each reference `productions.production_id`.
- `payments.commitment_id` references `commitments.commitment_id`.
- `commitments.replaces_commitment_id` references another commitment of the same production.

## Time rule for prediction

**State as of a snapshot date `d`:**
- the budget revision in effect on `d`;
- category plans effective on or before `d`;
- commitments with `selected_date ≤ d` that are not cancelled on or before `d`;
- manual costs with `created_date ≤ d`.

`final_pre_vat`, `financial_close_date`, `fee_concession_pre_vat`, and payments dated after `d` are **not** known at `d`.
