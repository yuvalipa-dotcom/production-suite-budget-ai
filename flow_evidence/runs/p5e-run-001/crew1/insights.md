# Crew 1 insights (synthetic data)

## Observations (descriptive, from eligible synthetic snapshots)

1. The contract declares 1,538 eligible snapshots (SYN-0001 to SYN-0400, 2024-09-04 onwards).
2. Overall overrun prevalence is 47.3%; it is 51.6% at S0 and 42.9% at S3 (Chart 1, Table 2).

## Hypotheses for Crew 2 (not predictions)

- H1: stage S3 snapshots may carry more information about the final gap than S0 snapshots.
- H2: budget revisions to date may be associated with the signed gap; Crew 2 can test this on the train group.

## Frozen split note

The test minus train row-level overrun prevalence difference is +14.7 pp, with a 95% production-level bootstrap interval of [+3.3, +26.2] pp. The split is frozen and was not tuned. The difference was assessed as a sampling outcome, independent of the outcome mechanism, so the split was not re-drawn or rebalanced.

## Data quality and Policy C exclusions

Policy C excluded 62 of 1,600 scheduled snapshots (commitment_category_missing: 50, manual_cost_undated: 12). No value was imputed.

## Limitations

The data are synthetic. There is no real-world validation, and nothing here is evidence of real-world accuracy.
