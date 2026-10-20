# P5-F: the Streamlit demonstration app

> **ALL DATA IS SYNTHETIC.** The CrewAI Flow, both crews, their agents and tasks run for real; the LLM responses
> are **scripted** (no inference, no network); every number, check, fit and artifact is computed by Python.

## Two environments (Windows)

The app has its own environment. Streamlit needs `pillow<13`, which conflicts with the ML pin `pillow==13.0.0`;
the app imports no ML code (no crew, scikit-learn, joblib or pickle), so the two never mix.

```
py -3.13 -m venv .venv                     REM the ML environment (crews, Flow, models)
.venv\Scripts\python -m pip install -r requirements-ml.txt

py -3.13 -m venv .venv-app                 REM the app environment (Streamlit only)
.venv-app\Scripts\python -m pip install -r requirements-app.txt
```

Keep the clone in a short folder path: Windows without long-path support can fail to install some wheels in
very deep folders.

## Launch

```
.venv-app\Scripts\streamlit run app\streamlit_app.py
```

The app opens at http://localhost:8501 (bound to localhost only; usage statistics off: `.streamlit/config.toml`).
New runs use `.venv\Scripts\python.exe`; to use another ML environment, set it first:

```
set PSBA_ML_PYTHON=C:\path\to\ml-env\Scripts\python.exe
```

## What it shows

* **Overview**: the committed, verified canonical run (`flow_evidence/runs/p5e-run-001`) immediately: Crew 1 →
  contract gate → Crew 2, the agents that executed, the Flow methods and routes, and a VERIFIED badge only when
  every file matches its recorded sha256.
* **Model results**: the selected model; **A. CURRENT RUN: TRAIN-only grouped CV** (from the run's own tool
  results); **B. HISTORICAL P4 TEST: cited, not recomputed** (from the pinned reference, shown only when its
  sha256 equals the one the run recorded); limitations.
* **Artifacts**: the 8 required outputs with size and sha256; viewers for the reports (Markdown without raw
  HTML), the EDA page (inside an iframe with an empty `sandbox`: no scripts, forms or same-origin access), the
  contract and gate report; CSV files as header and counts only (no rows); `model.joblib` as metadata only
  (never loaded).
* **Contract gate demos**: the five committed breaks with their violation codes and proof that Crew 2 never ran;
  a button runs one again (about 20 s).
* **Run the full Flow**: a full run (about 4–5 minutes with 5 parallel jobs) only after the confirmation box and
  a click; live stage progress and a redacted log tail.

Nothing runs when the app opens or renders. Runs started from the app are written to `out/flow_runs/` (ignored by
git) and appear under **Local runs**.

## Canonical evidence

`flow_evidence/runs/<run_id>/` holds verified copies of Flow runs, each with `evidence.json` (sha256 of every
copied file; a stopped run lists its omitted Crew 1 outputs with their sha256). It is produced, never
overwritten, by:

```
.venv-app\Scripts\python -m app.export_evidence p5e-run-001 p5e-break-money_x100 ...
```

## Tests

```
.venv\Scripts\python -m pytest tests/test_app_evidence.py      REM evidence layer, app/ML separation (ML env)
.venv-app\Scripts\python -m pytest app/tests -q                 REM rendering and user actions (app env)
```
