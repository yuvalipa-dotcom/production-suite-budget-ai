# Crew 1: the data-analyst crew (P3)

> **ALL DATA IS SYNTHETIC.** Crew 1 cleans and describes a synthetic raw export. Nothing it produces is a model result or evidence of real-world predictive accuracy.

Crew 1 turns the raw export into the handoff that the P2 gate checks and that Crew 2 (P4, not started) will model:
- `clean_data.csv`;
- `dataset_contract.json`;
- `eda_report.html`;
- `insights.md`.

It is a CrewAI crew of three agents running three sequential tasks. **Every data value, Policy C transformation, financial calculation, statistic and structural contract field comes from deterministic Python tools.** The agents choose and call those tools, and write only interpretation text.

## Architecture

| Agent | Task | Tools (its own only) |
|---|---|---|
| A1 Data Ingestion & Cleaning Engineer | T1 `ingest_and_clean`: profile, detect, clean, and write the cleaning narrative | `load_raw_table(name)`, `detect_defects()`, `apply_policy_c()`, `eligibility_summary()` |
| A2 Exploratory Data Analyst | T2 `explore_and_report` (context T1): run analyses, render charts, interpret, `eda_report.html` | `list_analyses()`, `run_analysis(analysis_id)`, `render_chart(chart_id)`, `assemble_eda_report(sections)` |
| A3 Data Contract & Insights Author | T3 `contract_and_insights` (context T1, T2): contract, gate, `insights.md` | `build_dataset_contract(column_descriptions)`, `run_handoff_gate()`, `write_insights(markdown)` |

**Layout:**

| Path | Content |
|---|---|
| `crew1/config/agents.yaml`, `crew1/config/tasks.yaml` | CrewAI-standard configuration |
| `crew1/crew.py` | builds the crew; trace recorder; minimal environment |
| `crew1/tools/` | `workspace.py`, `cleaning.py`, `eda.py`, `contract.py`, `gate_tool.py` and the tool wrappers in `__init__.py` |
| `crew1/templates/eda_report.html` | the report template (stdlib `string.Template`) |
| `crew1/validators.py` | post-run validators |
| `crew1/run.py` | the CLI |

The `run_handoff_gate` tool (`crew1/tools/gate_tool.py`) calls `gate.run` unchanged.

**The approved analysis menu:**
- descriptive statistics;
- overrun prevalence with Wilson 95% intervals by group, stage and event type;
- signed-gap distributions;
- budget-revision effects;
- feature correlations (train group only);
- fee-mode and fee-included breakdowns;
- the frozen-split prevalence difference with a production-level bootstrap interval (2,000 resamples);
- Policy C exclusions.

**Charts** are matplotlib PNGs rendered from `stats.json` only. They use the validated reference palette and are embedded in the report as data URIs, each next to its computed table.

## The deterministic boundary

| Python (deterministic) | LLM (agents) |
|---|---|
| Ingestion, profiling, defect detection | Which tools to call and in which order (within the task) |
| Policy C (`gate/policy_c.py`): normalisation, eligibility, exclusions | The cleaning narrative |
| Every `clean_data.csv` value, in the exact handoff layout | Which analyses and charts to run |
| Every statistic in `stats.json`, every chart, every report table | Interpretation paragraphs of the EDA report |
| Every structural field of `dataset_contract.json` | Optional one-line column descriptions (validated) |
| The P2 handoff gate | `insights.md` text |

**No agent can write or edit a data value.** The cleaning tool writes `clean_data.csv`. The contract tool takes only descriptions, rejects unknown columns and non-plain text, and generates everything else.

The two authoring tools, `assemble_eda_report` and `write_insights`, return the validator findings on the agent's text: missing sections, unverified numbers and forbidden claims. The agent can then correct the text. The post-run validators remain the authority.

**Note on Policy C assurance.** Crew 1 uses the same Policy C library that the gate uses for verification. The independent assurance of Policy C correctness comes from the P2 agreement tests against ground truth and committed data. The handoff gate protects against tampering, LLM error and corruption between the crews. The Crew 1 tests add one more independent cross-check: the tool-built handoff equals the reference built from the committed clean snapshots and the ground truth.

## Isolation (docs/crew1_isolation.md)

**Fresh workspace.** Each run creates a new temporary directory **outside the repository**. It contains a copy of `raw/` (hash-checked against the manifest) and `docs/raw_data_dictionary.md` only. This follows the P3 instruction. The design document also listed `raw_cleaning_policy.md`, but the agents receive the Policy C decisions through tool output instead.
- The crew runs with that directory as its working directory.
- It runs with an allow-listed minimal environment. `PATH` entries and variables that reveal the repository are removed, and `TEMP` points into the workspace.

**Logical names only.**
- Tools take names such as `raw.commitments`, resolved through a fixed allow-list.
- The following are rejected with a code and never echoed:
  - absolute paths;
  - `..`;
  - UNC and device paths;
  - 8.3 short names (`~`);
  - alternate data streams;
  - any path-like text;
  - unknown names.

**Links.** Every path component is checked before opening for symbolic links, junctions and other reparse points. After opening, the handle's final path (`GetFinalPathNameByHandleW` on Windows) must equal the expected path inside the workspace.

**Outputs.** Writes go only to the allow-listed names under `<workspace>/out/`. The CLI copies the artifacts to `--out` after the run, then deletes the workspace.

**No leaking configuration.** No tool description, argument, prompt or environment variable contains the repository, clean v2 or ground-truth paths. The contracts the tools need are read by Python inside the tool, never through an agent argument. These are `column_roles.json`, the meta-contract and `frozen_split.json`.

**CrewAI's kickoff store.** CrewAI keeps a "latest kickoff task outputs" SQLite store for replay, and Crew 1 redirects it into the run workspace, which is deleted after the run. Constructing a crew still creates an empty store under the user app-data folder `CrewAI/psba-crew1`, outside the repository.

**Tests** (`tests/test_crew1_isolation.py`):
- every traversal, absolute, UNC, device, 8.3 and stream attempt is rejected;
- a real Windows junction is rejected;
- a real short-name alias is rejected;
- a foreign file handle is rejected after opening.

Canary tokens are planted in copies of the clean v2 tables and in ground-truth files of a mirror repository. **No canary appears in any tool output, trace or workspace file**, and a file-open trace shows that no tool opens clean v2 or ground truth. A deliberately misconfigured allow-list makes the same canary check detect the leak (two-way proof).

**Windows limitation.** Creating a file symbolic link needs Developer Mode or the symlink privilege. Where it is unavailable, that one test is skipped with the reason, and a supplementary test proves the symlink branch with a simulated `lstat`. The junction test (a real reparse point) runs.

**What this isolation is not.** It is in-process, technical isolation of what the agents can reach through their tools. It is not an operating-system sandbox: the Python process itself can read the repository, and that is how the tools load the committed contracts.

## Runtime safety

These settings are applied in code before CrewAI is imported, by `crew1/__init__.py` and `crew1/crew.py`:
- `CREWAI_DISABLE_TELEMETRY`, `CREWAI_DISABLE_TRACKING` and `OTEL_SDK_DISABLED` are `true`, and `CREWAI_TRACING_ENABLED` is `false`. They are forced, not defaulted.
- Crew and agents have no memory, knowledge, embeddings, delegation, code execution, reasoning or planning, and caching is off.
- Limits:
  - A1: `max_iter` 12, `max_execution_time` 600 s, `max_rpm` 20;
  - A2: `max_iter` 25, `max_execution_time` 900 s, `max_rpm` 20;
  - A3: `max_iter` 15, `max_execution_time` 600 s, `max_rpm` 20;
  - crew: `max_rpm` 20.
- LLM: temperature 0, `max_tokens` 8192, timeout 180 s, 2 retries.
- **Model:** `CREW_MODEL`, default `claude-haiku-4-5-20251001`. This dated Haiku id is listed in the installed `crewai==1.15.28` (`crewai/llms/constants.py`, `ANTHROPIC_MODELS`) and routes to CrewAI's native Anthropic provider (`anthropic==0.73.0`); LiteLLM is not installed.
- **Key:** the provider SDK reads `ANTHROPIC_API_KEY` from the environment. Crew 1 code only checks that it is present. Its value is never read into a log, and any occurrence in the trace is redacted.

## Running

**Install:**
```
.venv\Scripts\python -m pip install -r requirements-crew.txt
```

**Offline tests.** They need no key, and a fixture blocks every socket:
```
.venv\Scripts\python -m pytest tests/test_crew1_wiring.py tests/test_crew1_isolation.py tests/test_crew1_determinism.py tests/test_crew1_validators.py
```
End-to-end CLI tests use a scripted stand-in for the LLM that drives the real tools.

**Live run.** This makes LLM calls and needs the owner's approval:
```
python -m crew1.run --live --out artifacts/crew1/<run_id>
```
- Without `--live`, or without `ANTHROPIC_API_KEY`, the CLI exits `2` and makes no call.
- `--out` must be new or empty.
- Exit codes: `0` PASS, `1` FAIL, `2` usage or configuration error.

**Run outputs:**
- the 4 artifacts;
- `stats.json`, `cleaning_log.json` and `gate_result.json`;
- `trace.jsonl`: every tool call with arguments and output, agent steps and task outputs. Redacted: the key, `sk-` shapes and every absolute path are replaced; workspace paths become `<workspace>`.
- `run_manifest.json`:
  - model id and LLM settings;
  - limits and package versions;
  - input sha256s and Policy C counts;
  - token usage and estimated cost (from the price catalogue bundled with crewai);
  - timings and validator results;
  - status `PASS` or `FAILED`.

## Validators (`crew1/validators.py`)

A run is **PASS** only if all nine pass. Otherwise the CLI exits `1`, the manifest status is `FAILED`, and the artifacts are kept.

| ID | Check |
|---|---|
| V1 | the 4 artifacts exist and are non-empty |
| V2 | `clean_data.csv` is byte-identical to a fresh Policy C recomputation from the repository raw export |
| V3 | the structural fields of `dataset_contract.json` equal a fresh build; only valid free-text descriptions may differ |
| V4 | the P2 handoff gate, re-run independently, is PASS, and the crew's own `gate_result.json` is PASS |
| V5 | every quantitative claim in the report narrative and in `insights.md` matches `stats.json` or the contracts (rules below) |
| V6 | the report has every section with interpretation text and at least 4 embedded PNG charts; no forbidden claim |
| V7 | `insights.md` has the 5 required sections. The frozen split note states the test minus train difference and its interval (pp, from `stats.json`), "frozen", "not tuned", "sampling" and "independent". The Limitations state synthetic data and no real-world validation. No forbidden claim |
| V8 | `run_manifest.json` is complete (model id, package versions, input sha256s, Policy C counts, token usage, timings) |
| V9 | `trace.jsonl` contains no key, no secret shape and no absolute path |

### Quantitative-claim rules (V5)

The tool-generated tables are not narrative. The rules apply to the agents' text: the report's interpretation blocks and the whole of `insights.md`.

**1. Masked as identifiers or metadata.** These are never claims:
- production and record ids (`SYN-0001`, `SYN-0001-C01`);
- dates, and years in date context (`in 2025`);
- version strings and seeds;
- ordinals and percentile labels (`95th`);
- bucket labels (`2+`);
- the confidence level of an interval (`95% Wilson interval`);
- labelled references (`Chart 2`, `Table 1`, `Section 3`, `Crew 2`, `stage 2`);
- list numbering at the start of a line.

Letter-attached tokens (`S0`–`S3`, `H1`, `P2`, `v2`) are not numbers.

**2. Everything else is a quantitative claim**, including numbers that cannot be classified. Its unit comes from the adjacent text:

| Text | Unit |
|---|---|
| `%`, percent | percent |
| pp, percentage points, points | pp |
| ILS, NIS, ₪, shekels | ILS |
| days | days |
| `x`, `×` | ratio |
| no unit | integer or decimal |

An unlabelled number in an interval or range takes its partner's unit (`[3.2, 26.3] pp`).

**3. Matching.**
- The sources:
  - every numeric value of `stats.json`, whose unit is fixed by its key name (an unclassified key is an error);
  - `dataset_contract.json` (row count, exclusion counts, column counts, declared bounds);
  - the counts of `contracts/frozen_split.json`;
  - the documented definitions in `stats.json`: VAT rate, overrun tolerance, materiality share, the stage schedule.
- Compatible units:

  | Claim | Matches |
  |---|---|
  | percent | fractions × 100 |
  | pp | pp |
  | ILS | ILS |
  | days | days |
  | integer | counts and plain numbers |
  | decimal | ratios, fractions and plain numbers |

- **Rounding rule:** a claim matches when it equals the source value rounded to the precision shown in the text (half-up or half-even).
- An unsigned claim matches the magnitude; a signed claim must match the sign.
- Only after "about", "around", "approximately", "roughly", "nearly" or "~" may an integer's trailing zeros mark rounding to that place, and only with at least two significant digits.

**4. Failure.** A claim with no matching source value of a compatible unit fails the run. This catches a wrong unit, for example a pp difference written as `%`.

**Limitation:** the validator checks values and units, not attribution. A true number attached to the wrong subject can pass.

### Forbidden claims (V6, V7)

These are checked per sentence, and a sentence with a negation cue is allowed ("not", "no", "never", "without", "nothing", and so on):
- claims of real-world accuracy, performance or validity;
- "will … predict";
- recommending a re-split, re-draw, rebalancing, re-stratification or over/under-sampling of the frozen split.

**Limitation:** the negation cue is sentence-wide, so a sentence that mixes a recommendation with an unrelated negation is not caught.

## Relationship to the P2 handoff gate

**The gate is unchanged.** Crew 1 calls it through `gate.run` (A3's `run_handoff_gate`, read-only), and V4 re-runs it independently on the copied artifacts.

**Format.** `clean_data.csv` follows `gate/handoff_spec.py`:
- columns in specification order;
- rows sorted by `(production_id, stage)`;
- amounts to 2 decimals;
- booleans written `True` and `False`;
- groups and folds from `contracts/frozen_split.json`.

**Results.** Of the 1,600 scheduled snapshots, 1,538 are eligible and 62 are excluded:
- 50 `commitment_category_missing`;
- 12 `manual_cost_undated`.

The gate's `W-PREVALENCE-IMBALANCE` warning (the frozen test group's higher overrun rate) is expected and is reported in the frozen split note. It is not acted on: the split stays frozen.
