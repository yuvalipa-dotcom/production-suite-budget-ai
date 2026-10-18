# Demo-installation seeding and removal: PROPOSED design

> **PROPOSAL ONLY.** Nothing in this repository seeds, reads or modifies any database.
> **ALL DATA IS SYNTHETIC.**

The product demonstration (phases P6–P8) would show 3–6 curated synthetic productions, taken from the `demo` split, in one demo installation. This document proposes how such a batch could be identified, seeded and removed.

> ⚠️ **This is not a guaranteed one-operation deletion.** The final mechanism depends on a separately approved verification of the database relationships and of deletion safety during P8.

## Identification

Each seeded record would carry three markers:
1. A stable synthetic id: deterministic ids derived from `SYN-####` and the batch id, so re-running produces the same ids.
2. The product's demo flag on the event (`is_demo = true`).
3. A batch tag in a notes or tag field, unique to this batch (e.g. `[synthetic batch syn-v1-seed42]`).

The batch tag must differ from every tag used by any other demo seeder, so removing this batch can never select another batch's rows.

## Seeding (proposed)

1. **Target gate.** Reuse the installation's existing seed-target gate (declared target, env file and site identity must agree), plus an allow-list containing only the demo installation, plus a check that the installation is in agency mode.
2. **Dry-run by default.** The real run requires typing the full installation domain.
3. **Idempotent.** Clean this batch by its tag, then upsert with the deterministic ids.
4. **Read-back.** Print the counts read back from the server after writing.

## Dependent records to verify before any deletion

Per seeded event, before writing the removal step:
- **Finance row:** one per event.
- **Category budgets:** keyed by event and category.
- **Manual costs:** they have a soft-delete column.
- **Quotes:** they reference suppliers, and suppliers may be shared with other events.
- **Payments:** they reference the event and, for outgoing payments, a supplier.
- **Suppliers created by the batch:** must be removed only if no non-batch record references them.
- **Clients created by the batch:** same rule as suppliers.
- **Activity-log rows:** append-only by design, so they are not deleted.
  - ⚠️ This means removal leaves an audit trace. That is expected, and must be stated.
- **Any further cascades:** documents, tasks, timelines and portal rows must be enumerated from the migrations during P8.

## Proposed deletion order (dependents first)

1. Payments of batch events
2. Manual costs of batch events
3. Quotes of batch events
4. Category budgets of batch events
5. Finance rows of batch events
6. Batch events
7. Suppliers created by the batch, and not referenced elsewhere
8. Clients created by the batch, and not referenced elsewhere

Each step selects **by batch tag and deterministic id together**, never by tag alone. Each step runs as a dry-run first, with the counts shown.

## Rollback

Removal of this batch is the rollback of the seeding. Records outside the batch are never touched.
