# Generator field → product field mapping

> **ALL DATA IS SYNTHETIC.** This table describes how each synthetic field
> *would* correspond to a field in the Production Suite product. It is a design aid
> only: it does **not** claim that a model trained here fits real productions.

**Status:**
- `EXISTS`: a field with this meaning exists in the product today.
- `PROPOSED`: it would need a future, separately approved change.

**At prediction time:**
- `yes`: computable from the product's current state at the moment of prediction.
- `no`: known only later.

All amounts in the product are stored as typed, with a per-amount VAT mode, and are normalised to pre-VAT at read time. The same convention is used here.

| Generator field | Product table / field (generic) | Status | At prediction time | Notes |
|---|---|---|---|---|
| `production_id` | `events.id` | EXISTS | yes | synthetic ids are `SYN-####` |
| `archetype` | `events.event_type` | EXISTS | yes | the product has more event types; four are simulated |
| `event_date`, `event_month` | `events.start_date` | EXISTS | yes | |
| `lead_time_days`, `days_since_contract` | `events.created_at` → `start_date` | EXISTS (approx.) | yes | the product has no explicit contract date; creation date is a proxy |
| `days_to_event` | `start_date` − today | EXISTS | yes | |
| `guest_count` | `events.guest_count` / `estimated_guests` | EXISTS | yes | |
| `budget_pre_vat` | `event_finance.total_budget` + `budget_vat_mode` | EXISTS | yes | normalised to pre-VAT |
| `budget_vat_mode` | `event_finance.budget_vat_mode` | EXISTS | yes | |
| `budget_revisions_to_date` | — | PROPOSED | no today | needs budget history; the activity log is partial |
| `fee_included` | `event_finance.commission_included` | EXISTS | yes | **manager-only** in the product |
| `fee_mode` | `event_finance.producer_commission_mode` | EXISTS | yes | manager-only |
| `fee_contract_pre_vat` | `producer_commission_value` + `commission_vat_mode` | EXISTS | yes | contractual fee; manager-only; the later adjustment field is **not** used |
| `planned_total_pre_vat`, `n_categories_planned` | `event_category_budget.planned_cost` + `vat_mode` | EXISTS | yes | |
| `n_commitments_active`, `n_categories_committed`, `n_categories_pending` | `quotes` with status *selected*, not soft-deleted | EXISTS | yes | |
| `committed_pre_vat` | selected `quotes.quoted_amount` + `event_manual_costs.amount`, each normalised by its own VAT mode | EXISTS | yes | this is **committed** cost, not a verified final cost |
| `manual_costs_pre_vat`, `n_manual_costs` | `event_manual_costs` (not soft-deleted) | EXISTS | yes | |
| `n_categories_over_plan`, `over_plan_pre_vat` | committed vs `event_category_budget` per category | EXISTS | yes | |
| `n_substitutions_to_date` | soft-deleted selected quotes | EXISTS (approx.) | yes | a deleted quote is not always a substitution |
| `no_vat_share` | quotes with VAT mode `no_vat` | EXISTS | yes | |
| `client_direct_share` | — | PROPOSED | no today | needs a payment-responsibility field per commitment |
| `current_position_ils` | committed + fee if included − budget | EXISTS (derivable) | yes | the product's current "available to spend" does not subtract an included fee; a fee-aware figure is a separate, approved change |
| `committed_ratio`, `cost_per_guest_pre_vat` | derived | EXISTS (derivable) | yes | |
| `commitments.original_entered` | `quotes.quoted_amount` | EXISTS | yes | |
| `commitments.final_pre_vat` | — | PROPOSED | no | needs a verified final cost per commitment, stored next to (not over) the accepted quote, plus an adjustment audit trail |
| `commitments.payment_responsibility` | — | PROPOSED | — | company-paid vs client-paid |
| `payments.amount_gross`, `paid_date` | `payments.amount`, `paid_date`, `direction = out` | EXISTS | yes | cash flow, never a cost; no link from a payment to a specific quote today (PROPOSED) |
| `final_cost_pre_vat` (target) | — | PROPOSED | no | depends on the verified final costs above |
| `budget_gap_ils`, `is_overrun` (targets) | — | PROPOSED | no | |
| `financial_close_date` | `events.status` = financially closed | EXISTS (status only) | no | no close timestamp today (PROPOSED) |
| `fee_concession_pre_vat`, `final_company_profit_pre_vat` | `event_finance.commission_adjustment` | EXISTS | — | profit, **not** part of the overrun target |
| `is_synthetic`, `batch_id` | `events.is_demo` + a batch tag | EXISTS (partial) | — | see `agency_seed_plan.md` |

## Export-friendly design

- Every feature is a plain number or a small closed category list, with no free text and no per-production history beyond counts.
- A future model can therefore be exported to plain JSON (coefficients, or tree thresholds, plus preprocessing constants) and scored inside the product without Python.
- Whether such a model fits real productions requires later testing on verified real data.

## Fields used by the P4 model (Crew 2)

The selected model (calibrated logistic regression, selected by the pre-registered rule) uses the primary feature set `F27` (owner decision D2). That is every snapshot feature in the table above whose status is `EXISTS`, `EXISTS (approx.)` or `EXISTS (derivable)`:

- **Used:** `stage` (from the snapshot date), `archetype`, `event_month`, `lead_time_days`, `days_since_contract`, `days_to_event`, `guest_count`, `budget_vat_mode`, `budget_pre_vat`, `fee_included`, `fee_mode`, `fee_contract_pre_vat`, `planned_total_pre_vat`, `n_categories_planned`, `n_categories_committed`, `n_categories_pending`, `n_commitments_active`, `n_manual_costs`, `manual_costs_pre_vat`, `committed_pre_vat`, `n_categories_over_plan`, `over_plan_pre_vat`, `n_substitutions_to_date`, `no_vat_share`, `current_position_ils`, `committed_ratio`, `cost_per_guest_pre_vat`.
- **Not used (`PROPOSED`):** `budget_revisions_to_date` and `client_direct_share`. They appear only in the research variant `F29`, which was never used for selection.
- **Manager-only inputs:** `fee_included`, `fee_mode` and `fee_contract_pre_vat`, and through them `current_position_ils` and `committed_ratio`, which include the fee if it is included. A product card built on this model is therefore manager-only, as the roadmap requires.
- **Approximations:** `lead_time_days` and `days_since_contract` (creation date as a proxy for the contract date) and `n_substitutions_to_date` (a deleted quote is not always a substitution) are approximations in the product. Their effect on a real installation is untested.

No commercial integration is part of P4. This mapping is a design aid only.
