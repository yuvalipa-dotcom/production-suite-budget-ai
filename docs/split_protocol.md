# Split and evaluation protocol

> **ALL DATA IS SYNTHETIC.**

## Groups

Every production is assigned to **exactly one** group, deterministically from the production ids and the seed (`generator/split.py`, `assign_split`). All snapshots of a production inherit its group.

| Group | Ratio | Productions | Snapshot rows | Purpose |
|---|---|---|---|---|
| `train` | 70% | 280 | 1,120 | model selection and hyper-parameter tuning, **production-grouped cross-validation only** |
| `test` | 20% | 80 | 320 | reserved for the **single final evaluation** |
| `demo` | 10% | 40 | 160 | reserved for the product demonstration |

`demo` is **never** used for training, tuning, model selection or evaluation, so a demonstration never shows a production the model was trained on.

**Split type:** a seeded random permutation, not a temporal split. The plan this project follows defines a production-grouped (non-temporal) split, so the temporal rule ("test and demo in the later period") does not apply.

## Cross-validation folds

`assign_folds` (`generator/split.py`) assigns each **train** production to one of **5 folds**, seeded and grouped by production id: 56 productions (224 snapshot rows) per fold. Test and demo rows carry `cv_fold = -1`.

All snapshots of a production are always in the same fold, so a model is never validated on a stage of a production it saw at another stage.

## Tests that enforce this

- No production id appears in more than one group.
- The group of every snapshot equals the group of its production.
- No production spans two folds.
- Folds exist only on train and are balanced.
- The split is deterministic, independent of input order, and changes with the seed.

## Overrun rate per group (seed 42, snapshot level)

| Group | Overrun rate |
|---|---|
| train | 44.0% |
| test | 59.1% |
| demo | 46.2% |

The test group happens to have a higher rate than train. This is sampling variation over 80 productions. It is reported, not corrected, because re-drawing the split until it looks balanced would be a form of selection.
