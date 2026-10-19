# Crew 2: the data-scientist crew (P4)

> **ALL DATA IS SYNTHETIC.** Crew 2 trains and evaluates models on a synthetic dataset. Every prediction is an **experimental / demo prediction**, and nothing here is evidence of real-world accuracy.
> Copyright (c) 2026 Yuval Lipa. All rights reserved. No open-source license.

Crew 2 turns the Policy C handoff into a calibrated overrun-probability model, following the pre-registered protocol. Its deliverables are:
- `model.joblib`;
- `metrics.json`;
- `model_card.md`;
- `demo_predictions.csv`, inference only.

Where to find the rest:
- the protocol: [`model_selection_rule.md`](model_selection_rule.md), Addendum P4-1 and erratum P4-1a, committed alone before any training code;
- the plan: [`p4_plan.md`](p4_plan.md);
- the leakage proof: [`p4_leakage_proof.md`](p4_leakage_proof.md).

## Architecture

| Agent | Task | Tools (its own only) |
|---|---|---|
| C1 Feature & Leakage Engineer | `prepare_and_prove_features` | `build_handoff()`, `feature_availability()`, `run_leakage_checks()` |
| C2 Model Trainer | `train_and_select` | `list_runs()`, `run_grouped_cv(run_id)`, `apply_selection_rule()`, `fit_final_models()` |
| C3 Evaluator & Model Card Author | `evaluate_and_document` | `evaluate_on_test_once()`, `predict_demo()`, `render_chart(chart_id)`, `write_model_card(<10 typed section fields>)` |

**Layout:**

| Path | Content |
|---|---|
| `crew2/ml/` | the deterministic core. It never calls an LLM |
| `crew2/ml/data.py` | the in-process handoff, its verification and the sealed test vault |
| `crew2/ml/features.py` | the feature sets |
| `crew2/ml/models.py` | B0, the LR and GB pipelines, the manual Platt calibration and the fit spy |
| `crew2/ml/resample.py` | the explicit group-disjoint splits and the split/fit audit |
| `crew2/ml/metrics.py` | the metrics |
| `crew2/ml/selection.py` | the rule |
| `crew2/ml/protocol.py` | the ordered session |
| `crew2/ml/leakage.py` | the runtime leakage checks |
| `crew2/tools/` | the CrewAI tool wrappers, the workspace allow-list, and the model card and its charts |
| `crew2/config/` | `agents.yaml` and `tasks.yaml` |
| `crew2/crew.py` | builds the crew |
| `crew2/run.py` | the CLI |
| `crew2/offline.py` | the scripted stand-in for the LLM |
| `crew2/validators.py` | the post-run validators |

**Lessons from P3 that are applied here:**
- Every tool argument is a named, typed field. There is no `dict[str, ...]` argument, because CrewAI sends tools in strict mode (p3-live-003).
- Tool calls are serialised by a run lock.
- Each agent has its own LLM instance and its own `max_iter` and time limits.
- Telemetry, tracing, memory, knowledge, delegation, code execution, reasoning and planning are all off.

## The deterministic boundary (C3)

| Python (deterministic) | LLM (agents) |
|---|---|
| Handoff rebuild and verification, partitioning, the test vault | Which tools to call, in task order |
| Every split, fit, tuning and calibration step, and the audits | Narrative text of the reports |
| Every metric, interval and reliability bin | The ten model-card narrative sections |
| **The model choice** (`apply_selection_rule` takes no argument) | — |
| The model-card header, disclosures and every table | — |

**The input** is rebuilt in-process from the raw export with the deterministic Policy C cleaning function `crew1.tools.cleaning.build_clean_data`, which is imported read-only.
- Its sha256 must equal the pinned `EXPECTED_CLEAN_SHA256`.
- A test proves that this hash agrees with the independent reference, built from the committed clean snapshots and the ground truth.
- The unchanged P2 handoff gate must be PASS on it.

**Train rows keep their labels.** Test and demo rows lose every label column at the text level:
- Test labels are parsed only by `TestVault.open`. It works once, only for frozen model hashes, and only after `fit_final_models`.
- Demo labels are never parsed.

## Running

**Install** (Python 3.13, a separate `.venv`):
```
.venv\Scripts\python -m pip install -r requirements-ml.txt
```

**Offline run.** No LLM call, no key. A scripted stand-in drives the real tools:
```
python -m crew2.run --offline --out out/crew2/<run_id> --jobs 5
```

**Live run.** This makes LLM calls and needs separate owner approval:
```
python -m crew2.run --live --out out/crew2/<run_id>
```
It is refused, with exit code 2, without `ANTHROPIC_API_KEY`.

`--jobs` runs the outer folds in parallel. The results are identical for any value.

**Offline tests.** No key; every socket is blocked:
```
.venv\Scripts\python -m pytest tests/test_crew2_*.py
```
The tests swap the pre-registered grids for one cheap setting per candidate (`tests/crew2_helpers.py`), so the protocol runs in seconds. The real grids are checked against the addendum text.

## Validators (`crew2/validators.py`)

A run is **PASS** only if all ten pass.

| ID | Check |
|---|---|
| W1 | every artifact exists and is non-empty |
| W2 | the handoff sha256 is the expected deterministic output, and the P2 handoff gate is PASS on a fresh rebuild |
| W3 | a fresh, independent recompute of the whole protocol equals `metrics.json` exactly |
| W4 | the selection equals the pre-registered rule applied to the CV results in `metrics.json`, and the final models are that candidate (C3) |
| W5 | test was opened exactly once, after both final models were frozen (shown by the hashes and the trace order); demo has no metric |
| W6 | `demo_predictions.csv` equals a recomputation from `model.joblib`, and every row carries the demo label |
| W7 | every number in the model-card narrative matches `metrics.json` or a pre-registered constant. This is Crew 1's claim checker |
| W8 | every card section, the synthetic and demo labels, the target disclosure, the copyright notice and both charts are present; there is no forbidden claim |
| W9 | the manifest is complete; the trace holds no key, secret shape or absolute path |
| W10 | the leakage report is clean, every CV and final-fit split/fit audit has zero violations, and all 9 pre-registered runs are present |

## Known limitations (P4 closure)

These are reported as measured. Nothing was retrained, retuned or re-selected after the single TEST evaluation, and the pre-registered rule is unchanged.

- **Synthetic data.** Every number comes from a synthetic dataset. The results are **not evidence of real-world production performance**, and every prediction is an experimental / demo prediction.
- **The grid boundary.** The selected logistic regression chose `C = 10`, the upper edge of the pre-registered grid (0.01, 0.1, 1.0, 10.0), in all 5 outer folds and in the final refit. The weakest regularisation in the grid was preferred, and a larger `C` was never evaluated. The grid was fixed before training and is not extended after seeing results.
- **The prevalence shift.** The TEST overrun prevalence is 59.2% (181 of 306 eligible snapshots), against 44.5% on TRAIN (482 of 1,084). The split is frozen, and the difference is reported, not corrected ([`split_report.md`](split_report.md)).
- **Calibration on TEST.** The expected calibration error on TEST is about 0.111 (10 bins). The model was calibrated on TRAIN, at TRAIN's prevalence, so its probabilities are not well calibrated at TEST's higher prevalence. No recalibration on TEST was done or is permitted.
- **The target.** `is_overrun` is a 0.5 ILS rounding-tolerance label, not a commercially material overrun. The 2% material overrun is a secondary result only.
- **Fixed fields.** Fields without a revision ledger are treated as fixed from the contract date (owner decision D4). In a real product they could change.
