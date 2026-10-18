# Synthetic data generation (active version: v2)

> **ALL DATA IS SYNTHETIC.** Every record is produced by the seeded simulation
> described here. No real customer, vendor or event data is used or contained.
> Results obtained on this data are **not** evidence of real-world predictive accuracy.

## Reproduce

```
python -m generator.cli --version v2 --out data/synthetic/v2      # active version (+ raw export)
python -m generator.cli --version v1 --out data/synthetic/v1      # frozen, byte-identical, for traceability
```

**Versions:**
- **v1:** the original export.
- **v2:** adds the approved-budget revision ledger and the category-plan ledger, so that every snapshot feature and label can be rebuilt from dated records. The reason is in [`target_audit.md`](target_audit.md).
- Every v1 file is byte-identical inside v2.
- v2 also contains `raw/`, a provider-style export with realistic data-quality problems, described in [`raw_data_dictionary.md`](raw_data_dictionary.md).

The output is byte-identical for a given seed: sorted rows, fixed numeric format
(2 decimals), LF line endings, and no timestamps or machine-specific values. A
test regenerates the dataset and compares it, file by file, with the committed copy.

| Parameter | Value |
|---|---|
| Seed | `42` |
| Generator version | `1.0.0` |
| Batch id | `syn-v1-seed42` |
| Productions | **400**, 100 per archetype |
| Snapshots per production | **4**: S0, S1, S2, S3 |
| Snapshot rows | **400 × 4 = 1,600** |
| VAT rate | 0.18 |
| Currency / basis | ILS, pre-VAT |

## Output files (`data/synthetic/v2/`)

| File | Grain | Purpose |
|---|---|---|
| `snapshots.csv` | one row per (production, stage) | modelling table: prediction-time features + targets |
| `productions.csv` | one row per production | static attributes, final outcome, profit (kept separate from the overrun) |
| `commitments.csv` | one row per supplier commitment | accepted quote and verified final cost, as two separate amounts |
| `manual_costs.csv` | one row per manual cost | unexpected expenses and cancellation fees |
| `payments.csv` | one row per company payment | cash flow only, **never a cost** |
| `budget_revisions.csv` | one row per approved budget (revision 0 = original) | effective-dated budget and guest-count history (v2) |
| `category_plans.csv` | one row per (production, category) | planned cost as typed, with VAT mode (v2) |
| `raw/` | provider-style tables + own manifest | input for the future data-analyst crew (v2) |
| `schema.json` | — | machine-readable role of every snapshot column |
| `manifest.json` | — | seed, counts, split sizes, sha256 of every file |

Every row carries `is_synthetic = True`, `batch_id`, and a stable synthetic id (`SYN-0001` …).

## Archetypes

`wedding`, `bar_bat_mitzvah`, `corporate`, `conference`. Each archetype has its own:
- distributions for budget, guest count and lead time;
- VAT mode of the budget (private events usually include VAT, business events usually exclude it);
- probability that the fee is included in the budget;
- probability that the client pays suppliers directly;
- category mix;
- peak months.

## Life cycle of one production

1. **Contract (day 0).**
   - The client approves a total budget, typed with a VAT mode.
   - A contractual production fee is set: a percent of the budget or a fixed amount, either *included* in the budget or charged *on top*.
2. **Category plans.**
   - The vendor pool is the pre-VAT budget, minus the fee if the fee is included.
   - It is split into category plans with a Dirichlet draw around the archetype mix.
3. **Supplier commitments.**
   - Each commitment has its own VAT mode (`includes_vat`, `excludes_vat` or `no_vat`) and a payment responsibility (`company` or `client_direct`).
   - Commitments are selected over the lead time; the main venue is often booked at or right after the contract.
4. **Cancellations and substitutions.**
   - A cancelled commitment is replaced by a new one, usually more expensive.
   - A forfeited deposit is recorded as a `cancellation_fee` manual cost.
5. **Unexpected expenses.** Manual costs arrive throughout the life cycle, many of them late or after the event.
6. **Budget revisions.** Some clients raise (or slightly lower) the approved budget and the guest count mid-way.
7. **Verified final costs.**
   - After the event, each commitment receives a final cost: additions, reductions and credits relative to the accepted quote.
   - The accepted quote is **kept unchanged**; the final cost is a second amount.
8. **Payments.**
   - The company pays only the commitments it is responsible for, as a deposit and then a balance.
   - Commitments paid by the client directly have no company payment.
   - Payments are cash flow, never cost.
9. **Financial close.** 14–60 days after the event.
10. **Fee concession.** Some productions get a late reduction of the fee. It lowers the company's final profit, and **never** changes the overrun label.

### Hidden factors

The simulation is driven by four factors that are **never written to any output**:
- client volatility;
- planner discipline;
- venue complexity;
- season pressure.

They are what makes overruns predictable to a degree, without exposing the cause directly.

## Stages (snapshot days)

| Stage | Day |
|---|---|
| S0 | contract day |
| S1 | 35% of lead time |
| S2 | 65% of lead time |
| S3 | 14 days before the event |

A snapshot counts only:
- commitments selected on or before that day and not yet cancelled;
- manual costs created on or before that day;
- the budget, fee and guest count in force on that day.

## Target (decided definition)

```
final_cost_pre_vat = Σ verified final costs of non-cancelled commitments + Σ all manual costs   (pre-VAT)
fee_in_budget_t    = contractual fee known at the snapshot (pre-VAT) if the fee is included, else 0
budget_gap_ils     = final_cost_pre_vat + fee_in_budget_t − approved budget at the snapshot (pre-VAT)
is_overrun         = budget_gap_ils > 0.5
```

- **Any positive excess is an overrun.** The 0.5 ₪ is a floating-point rounding tolerance only.
- `is_material_overrun` (gap > 2% of the budget) is a **warning level**. It is not the overrun definition.
- The fee is the **contractual** fee known at prediction time. A later concession never erases a historical overrun.
- **Overrun against the budget and company profit are different quantities**: profit is in `productions.csv` (`final_company_profit_pre_vat`) and is never a target.

## Resulting rates (seed 42)

| | Overrun rate |
|---|---|
| Overall (1,600 snapshots) | 47.2% |
| Per stage | S0 52.0% · S1 49.8% · S2 44.5% · S3 42.8% |

Rates per split group are listed in `split_protocol.md`.

A naive rule ("already over the budget in effect now ⇒ will overrun") never fires at S0 and S1, where nothing is over budget yet. It becomes informative only near the event. Per-stage figures on train are in [`model_selection_rule.md`](model_selection_rule.md).

## Known limitations

- The distributions are designed assumptions, not estimates from real productions.
- Adjustments to a commitment become known only at the final invoice. In reality some are known earlier.
- Category plans are fixed at contract time.
- Payments by clients to the company are not simulated (they do not affect the target).
- A model trained here may learn artefacts of this generator. Later phases evaluate on held-out productions, and **must not** claim real-world accuracy.
