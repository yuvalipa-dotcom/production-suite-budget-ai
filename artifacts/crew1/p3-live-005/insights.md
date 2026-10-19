## Observations (descriptive, from eligible synthetic snapshots)

The 1,538 eligible snapshots come from 397 productions: 375 fully eligible and 22 partially eligible (3 of the 400 scheduled productions have no eligible snapshot).

Overall overrun prevalence is 47.3% (95% Wilson interval: 44.8% to 49.8%), with 728 overrun snapshots among 1,538 eligible snapshots. Material overrun prevalence is 35.7% (95% interval: 33.3% to 38.1%), with 549 snapshots.

Overrun prevalence varies by stage. S0 shows 51.6% (95% interval: 46.7% to 56.5%), S1 shows 49.7% (95% interval: 44.8% to 54.7%), S2 shows 44.7% (95% interval: 39.8% to 49.8%), and S3 shows 42.9% (95% interval: 38.0% to 48.0%). The prevalence decreases from early to final stages.

The signed gap distribution has a median of -988.9 ILS and a mean of -1,863.6 ILS, with a standard deviation of 29,821.0 ILS. The 5th to 95th percentile range is -46,958.0 ILS to 41,310.1 ILS.

Overrun prevalence by frozen split group shows train at 44.5% (95% bootstrap interval: 38.9% to 50.2%) from 1,084 snapshots, test at 59.2% (95% bootstrap interval: 48.4% to 69.5%) from 306 snapshots, and demo at 43.9% (95% bootstrap interval: 29.7% to 59.2%) from 148 snapshots.

## Hypotheses for Crew 2 (not predictions)

Does the stage progression reflect genuine budget control improvements, or does it reflect selection effects in which productions advance to later stages?

Are the differences in overrun prevalence between fee modes (fixed, percentage, variable) associated with different cost structures or risk profiles?

Does the number of budget revisions to date correlate with overrun risk, and if so, is this association driven by reactive revision patterns or by underlying project complexity?

How do commitment category diversity and the temporal spread of costs (recency and age measures) relate to overrun occurrence?

## Frozen split note

The frozen split was not tuned and was assigned deterministically as a sampling outcome independent of the outcome mechanism. The test group shows higher overrun prevalence than the train group. The test minus train difference in overrun prevalence is +14.7 pp, 95% bootstrap interval [+3.3, +26.2] pp, assessed as a production-level bootstrap in frozen_split_prevalence_difference in stats.json.

## Data quality and Policy C exclusions

Of 1,600 scheduled snapshots, 62 became ineligible under Policy C: 50 due to commitment_category_missing and 12 due to manual_cost_undated. By stage, ineligibility was distributed as S0: 3 snapshots, S1: 14 snapshots, S2: 20 snapshots, and S3: 25 snapshots.

The cleaning process identified defects across three tables: missing category values in commitments, missing created_date values in manual_costs, and out-of-range amount_gross values in payments. Exact duplicates were dropped from commitments, manual_costs, and payments. Normalisations were applied deterministically without imputation across all tables.

Among 400 productions, 375 were fully eligible, 3 were fully ineligible, and 25 had at least one ineligible snapshot. No productions were flagged as target_unreliable.

## Limitations

These data are synthetic and have not been validated against real-world budget outcomes. The observations and associations described here are properties of the synthetic dataset and do not constitute evidence of real-world accuracy or predictive validity. The handoff is complete and the split is frozen; no re-splitting or rebalancing is recommended.