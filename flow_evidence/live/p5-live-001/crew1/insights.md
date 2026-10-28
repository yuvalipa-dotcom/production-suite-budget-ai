# Insights for Crew 2: Eligible Synthetic Snapshots

## Observations (descriptive, from eligible synthetic snapshots)

The dataset comprises 1,538 eligible snapshots from 397 productions across four event stages (S0–S3) and three frozen split groups (train, test, demo). Of the 400 scheduled productions, 375 are fully eligible, 22 are partially eligible (with at least one ineligible snapshot), and 3 have no eligible snapshots.

**Overrun Prevalence by Stage**

Overall overrun prevalence (is_overrun, signed gap exceeding 0.5 ILS) was 47.3% (95% Wilson interval: 44.8%–49.8%). Overrun prevalence declined across stages: S0 at 51.6% (95% interval: 46.7%–56.5%), S1 at 49.7% (95% interval: 44.8%–54.7%), S2 at 44.7% (95% interval: 39.8%–49.8%), and S3 at 42.9% (95% interval: 38.0%–48.0%). Material overrun prevalence (is_material_overrun, gap exceeding 2% of budget) was 35.7% (95% interval: 33.3%–38.1%) overall.

**Signed Budget Gap Distribution**

The signed gap distribution was right-skewed with a median of −988.9 ILS (under budget) and a mean of −1,863.6 ILS. The 5th to 95th percentile range was −46,958.0 ILS to 41,310.1 ILS. By stage, median gaps shifted from positive in S0 (282.4 ILS) to increasingly negative in S1 (−255.7 ILS), S2 (−2,535.0 ILS), and S3 (−3,441.9 ILS). The gap as a share of the as-of budget had a median of −0.5% and a 5th to 95th percentile range of −16.0% to 16.3%.

**Overrun Prevalence by Event Type**

By event type, conference snapshots showed the highest overrun prevalence at 55.1% (95% interval: 50.1%–60.0%), while wedding snapshots showed the lowest at 39.3% (95% interval: 34.5%–44.2%).

**Budget Revisions**

Overrun prevalence was associated with the number of budget revisions to date. Snapshots with zero revisions had an overrun prevalence of 51.7% (95% interval: 49.0%–54.4%) and a mean gap of 1,965.3 ILS. Snapshots with one revision had a prevalence of 25.0% (95% interval: 18.9%–32.3%) and a mean gap of −19,638.0 ILS. Snapshots with two or more revisions had a prevalence of 9.8% (95% interval: 4.6%–19.8%) and a mean gap of −39,325.5 ILS.

**Fee Mode and Fee Inclusion**

Overrun prevalence differed by fee inclusion status. Snapshots with fees included in the budget had an overrun prevalence of 48.7% (95% interval: 45.3%–52.1%), while those without fees included had a prevalence of 45.8% (95% interval: 42.2%–49.4%). By fee mode, snapshots with amount-based fees had an overrun prevalence of 52.6% (95% interval: 48.4%–56.8%), while those with percent-based fees had a prevalence of 44.5% (95% interval: 41.4%–47.6%).

**Feature Correlations (Train Group)**

On the train group (1,084 snapshots), Spearman correlation with the signed gap showed the strongest associations with over_plan_pre_vat (0.266), budget_revisions_to_date (−0.246), and current_position_ils (0.124). Point-biserial correlation with is_overrun showed the strongest associations with over_plan_pre_vat (0.275), budget_revisions_to_date (−0.183), and manual_costs_pre_vat (0.176). All correlations were weak to moderate.

## Hypotheses for Crew 2 (not predictions)

1. Does the association between budget revisions and lower overrun prevalence reflect a causal mechanism (e.g., revisions allow teams to adjust budgets downward), or is it a selection effect (e.g., productions with larger negative gaps are more likely to revise)?

2. Why do conference events show higher overrun prevalence than wedding events? Are there differences in cost structure, planning horizon, or budget discipline by event type?

3. Can the weak correlations between individual features and the signed gap be explained by interactions or non-linear relationships not captured in univariate analysis?

4. Does the association between over_plan_pre_vat and overrun status reflect category-level budget pressure, or does it indicate that certain categories are systematically underestimated?

5. Is the right-skewed gap distribution driven by a subset of productions with large positive gaps, or is it a property of the cost and budget processes across the population?

6. How do the frozen split groups differ in production characteristics (e.g., event type, budget size, revision frequency) beyond their overrun prevalence?

## Frozen split note

The frozen split is documented and was not tuned. The test group had higher overrun prevalence than the train group: test minus train difference was 14.7 percentage points (95% bootstrap interval: 3.3–26.2 pp). This difference was assessed as a sampling outcome independent of the outcome mechanism (frozen_split_prevalence_difference analysis in stats.json, a production-level bootstrap).

## Data quality and Policy C exclusions

Policy C eligibility screening excluded 62 snapshots from the 1,600 scheduled: 50 due to commitment_category_missing and 12 due to manual_cost_undated. Ineligibility increased progressively by stage, from 3 ineligible snapshots in S0 to 25 in S3. The cleaning process identified defects across seven tables and removed exact duplicates and invalid payments. All normalisable data were preserved; only records that could not be reliably reconstructed under Policy C's deterministic rules were excluded. No target_unreliable_productions were flagged.

## Limitations

This analysis describes associations in the eligible snapshots only; ineligible snapshots (62 total) are not represented. The data are synthetic and there is no real-world validation. The frozen split is a documented sampling outcome, not a result of analysis, and the test group's higher overrun prevalence is an observed difference, not a tuning outcome. Correlations are computed on the train group only and are weak to moderate; no feature shows a strong predictive relationship with the gap or overrun status. The analysis does not assess causation or the reliability of the underlying cost and budget data. Budget revisions, fee terms, and event type show associations with overrun prevalence, but these are descriptive findings, not explanations of overrun drivers.