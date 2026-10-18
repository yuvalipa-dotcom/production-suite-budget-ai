# Roadmap

> **ALL DATA IS SYNTHETIC.** No production company uses the product yet, so no real data exists.

## Phases

| Phase | Content | Status |
|---|---|---|
| P1 | Synthetic data generator, tests, documentation | this branch |
| P2 | Dataset contract and executable validation gate | not started |
| P3 | Crew 1: data-analyst agents (3) | not started |
| P4 | Crew 2: data-scientist agents (3), model comparison, model card | not started |
| P5 | Flow and Streamlit demonstration, with one committed real run | not started |
| P6–P8 | Product deliverable (below) | after P5, subject to separate approval |

## Two demonstrations

1. **P5: an independent Streamlit demonstration for academic grading.** It runs the full workflow and shows the artefacts, the gate (including deliberate failures) and the model comparison.
2. **P6–P8: an explicit product deliverable.** A budget prediction card inside the event's Budget tab in the demo installation. It shows:
   - the overrun probability;
   - the estimated signed budget gap (if validated);
   - a clear risk status;
   - data-quality warnings;
   - actionable explanations where supported.

   The card is **manager-only**, and no fee-derived information leaks to other roles.

After P5, P6–P8 are prioritised if time allows before the deadline. They require separate owner approval and coordination with the parallel product work.

## Labelling rule

Any prediction from a model trained only on synthetic data is labelled **"experimental / demo prediction"** everywhere. It is never presented as a validated real-world forecast.

## Later

Real-data validation requires the product first to store verified final supplier costs, separately from accepted quotes, and a financial-close date. That is a separately approved change.
