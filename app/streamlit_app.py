"""Production Suite budget-overrun AI: the Flow demonstration app (P5-F). ALL DATA IS SYNTHETIC.

    .venv-app\\Scripts\\streamlit run app\\streamlit_app.py

Opens on the committed, verified canonical run; nothing runs until the user asks. See docs/app.md.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repository root, for the app package

import streamlit as st  # noqa: E402

from app import evidence as E  # noqa: E402
from app import runner as RUN  # noqa: E402

st.set_page_config(page_title="Budget Overrun AI: Flow demo", page_icon="📊", layout="wide")

SCENARIO_TEXT = {
    "column_rename": "Column rename: budget_pre_vat renamed to budget_ils in clean_data.csv",
    "money_x100": "Money unit x100: budget_pre_vat x100, and the attacker regenerates a matching flow contract",
    "row_deletion": "Required data row deleted from clean_data.csv",
    "contract_column_deletion": "Required column definition deleted from dataset_contract.json",
    "label_leakage": "Outcome label leaked as an input column (declared as a feature)",
}


def f4(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def disclosure_badges():
    c = st.columns(4)
    c[0].badge("ALL DATA IS SYNTHETIC", icon=":material/science:", color="orange")
    c[1].badge("CrewAI orchestration: REAL", icon=":material/account_tree:", color="green")
    c[2].badge("Python computation: REAL", icon=":material/calculate:", color="green")
    c[3].badge("LLM responses: SCRIPTED", icon=":material/description:", color="violet")


# ---------------------------------------------------------------- sidebar: which run

with st.sidebar:
    st.header("Run evidence")
    source_label = st.radio("Source", ["Canonical (verified, committed)", "Local runs (out/flow_runs)"], index=0)
    source = "canonical" if source_label.startswith("Canonical") else "local"
    runs = E.list_runs(source)
    normal = [r for r in runs if "break" not in r]
    default = st.session_state.get("selected_run")
    run_id = st.selectbox("Run", runs, index=runs.index(default) if default in runs else
                          (runs.index(normal[0]) if normal else 0)) if runs else None
    st.divider()
    st.caption("The CrewAI Flow, both crews, their agents and tasks run for real. The model responses are "
               "scripted by an in-process transport: no LLM inference and no network call. Every number, check, "
               "fit and artifact is computed by deterministic Python.")
    py = RUN.ml_python()
    st.caption("ML environment for new runs: " + ("found" if py else "not configured (set PSBA_ML_PYTHON)"))

run = E.load_run(source, run_id) if run_id else None

st.title("Budget Overrun AI: CrewAI Flow demonstration")
st.write("Event productions: will a production end above its budget? An academic demonstration of two CrewAI "
         "crews joined by a contract gate.")
disclosure_badges()

if run is None:
    st.info("No run evidence found. Start a run in the **Run the full Flow** tab, or export canonical evidence "
            "(docs/app.md).")

tabs = st.tabs(["Overview", "Model results", "Artifacts", "Contract gate demos", "Run the full Flow",
                "How it works"])

# ---------------------------------------------------------------- overview

with tabs[0]:
    if run is not None:
        m = run.manifest
        status_color = {"PASS": "green", "STOPPED_CONTRACT": "orange"}.get(run.status, "red")
        top = st.columns([2, 2, 3])
        top[0].metric("Run", run.run_id)
        with top[1]:
            st.markdown(f"**Status** :{status_color}[{run.status}]")
            st.caption(m.get("status_meaning", ""))
        with top[2]:
            if run.verified:
                st.success(f"VERIFIED: {run.verification['files_checked']} files match their recorded sha256.",
                           icon=":material/verified:")
            else:
                st.error("NOT VERIFIED: " + "; ".join(run.verification.get("problems", [])[:3]),
                         icon=":material/gpp_bad:")
        if m.get("scenario") not in (None, "none"):
            st.warning(f"Deliberate contract break: {SCENARIO_TEXT.get(m['scenario'], m['scenario'])}")
        timing = m.get("timing", {})
        c1, a1, g, a2, c2 = st.columns([5, 1, 4, 1, 5], vertical_alignment="center")
        crew1, gate, crew2 = m.get("crew1", {}), m.get("gate", {}), m.get("crew2", {})
        with c1.container(border=True):
            st.subheader("1 · Crew 1: data analyst")
            for a in crew1.get("agents_executed", []):
                st.write(f"✅ {a}")
            ev = crew1.get("crewai_events", {})
            st.caption(f"kickoff events: {ev.get('crew_kickoff_started', 0)} started / "
                       f"{ev.get('crew_kickoff_completed', 0)} completed · tasks {ev.get('task_completed', 0)} · "
                       f"scripted LLM requests {crew1.get('llm_requests', 0)} · Python tool calls "
                       f"{crew1.get('python_tool_calls', 0)} · {timing.get('crew1_kickoff', 'n/a')} s")
        a1.markdown("### ➜")
        with g.container(border=True):
            st.subheader("2 · Contract gate")
            decision = gate.get("decision", "not reached")
            if decision == "PASS":
                st.success("PASS: flow-1 contract, 0 violations", icon=":material/check_circle:")
            elif decision == "STOPPED_CONTRACT":
                st.error("STOPPED_CONTRACT", icon=":material/block:")
                st.code("\n".join(gate.get("violation_codes", [])), language=None)
            else:
                st.info(decision)
            st.caption(f"router contract_gate → {m.get('flow', {}).get('routes', {}).get('contract_gate', 'n/a')}")
        a2.markdown("### ➜")
        with c2.container(border=True):
            st.subheader("3 · Crew 2: data scientist")
            if crew2.get("kickoff_invoked"):
                for a in crew2.get("agents_executed", []):
                    st.write(f"✅ {a}")
                sm = crew2.get("selected_model", {})
                st.caption(f"TRAIN-only · selected {sm.get('candidate')} {sm.get('params')} · scripted LLM requests "
                           f"{crew2.get('llm_requests', 0)} · Python tool calls {crew2.get('python_tool_calls', 0)} · "
                           f"{timing.get('crew2_kickoff', 'n/a')} s")
            else:
                st.write("⛔ **Crew 2 never ran.**")
                st.caption(f"kickoffs: {crew2.get('kickoffs', 0)} · no CV run and no model fit took place "
                           "(proven by call counters in tests/test_flow_e2e.py)")
        flow = m.get("flow", {})
        st.caption("Flow methods executed (CrewAI event bus): " + " → ".join(flow.get("methods_executed", [])) +
                   f" · total {timing.get('total_seconds', 'n/a')} s")

# ---------------------------------------------------------------- model results

with tabs[1]:
    if run is None or run.status != "PASS":
        st.info("Model results exist only for a PASS run: a stopped run never trains a model.")
    else:
        c2 = run.manifest["crew2"]
        sm, rep = c2["selected_model"], c2["reproducibility"]
        k = st.columns(4)
        k[0].metric("Selected model", sm["candidate"])
        k[1].metric("Parameters", ", ".join(f"{a}={b}" for a, b in sm["params"].items()) or "none")
        k[2].metric("Reproduction of approved P4 CV", rep["mode"])
        k[3].metric("TEST vault opens (this run)", c2["test_vault"]["opens"])
        st.caption(f"Decided by the pre-registered rule: {sm['decided_by']}. Final models refit on "
                   f"{sm['train_rows']:,} TRAIN rows; byte-identical to the P4 models: "
                   f"{all(rep['final_model_sha256_equal_to_p4'].values())}.")

        st.subheader("A · CURRENT RUN: TRAIN-only grouped cross-validation")
        st.caption("Computed by this run from TRAIN rows only (5 production-grouped folds). Mean ± sd over folds.")
        cv = run.cv_results()
        rows = [{"run": r["run_id"], "role": r["role"], "Brier": f"{f4(r['brier_mean'])} ± {f4(r['brier_sd'])}",
                 "PR-AUC": f"{f4(r['average_precision_mean'])} ± {f4(r['average_precision_sd'])}",
                 "ROC-AUC": f"{f4(r['roc_auc_mean'])} ± {f4(r['roc_auc_sd'])}"} for r in cv]
        st.dataframe([r for r in rows if r["role"] == "primary"], hide_index=True, width="stretch")
        with st.expander("Reported only, never used for selection: secondary target and research features"):
            st.dataframe([r for r in rows if r["role"] != "primary"], hide_index=True, width="stretch")

        st.subheader("B · HISTORICAL P4 TEST: cited, not recomputed")
        ht = E.historical_test(run)
        if ht is None:
            st.warning("The pinned P4 reference does not match the one this run recorded; historical metrics hidden.")
        else:
            st.caption(f"Pinned from the approved run {ht['source_run']}. TEST was opened once in P4, after the final "
                       "models were frozen. This run did NOT reopen TEST.")
            trows = []
            for name in ("primary", "secondary"):
                t = ht[name]
                o, ci = t["overall"], t["bootstrap_ci"]
                trows.append({"model": name, "target": t["target"], "rows": o["n"],
                              "Brier [95% CI]": f"{f4(o['brier'])} [{f4(ci['brier']['lo'])}, {f4(ci['brier']['hi'])}]",
                              "PR-AUC [95% CI]": f"{f4(o['average_precision'])} [{f4(ci['average_precision']['lo'])}, "
                                                 f"{f4(ci['average_precision']['hi'])}]",
                              "ROC-AUC": f4(o["roc_auc"]), "ECE (10 bins)": f4(t["reliability"]["ece"], 3)})
            st.dataframe(trows, hide_index=True, width="stretch")

        st.subheader("Limitations")
        st.markdown(
            "- **All data are synthetic**: nothing here is evidence of real-world accuracy.\n"
            "- The historical test group has a higher overrun prevalence than train; the probabilities were not "
            "well calibrated there (ECE above).\n"
            "- The selected C = 10 is the upper boundary of the pre-registered grid.\n"
            "- `is_overrun` is a rounding-tolerance label, not a commercially material overrun.\n"
            "- The agents' text is scripted; the demonstration shows orchestration, not model reasoning.")

# ---------------------------------------------------------------- artifacts

with tabs[2]:
    if run is not None:
        st.subheader("The 8 required outputs")
        recorded = run.manifest.get("required_outputs_sha256", {})
        rows = []
        for rel in E.REQUIRED:
            present = run.exists(rel)
            rows.append({"crew": "Crew 1" if rel.startswith("crew1") else "Crew 2", "file": rel,
                         "bytes": run.size(rel) if present else None,
                         "sha256": (recorded.get(rel) or "")[:16] + ("…" if recorded.get(rel) else ""),
                         "status": ("verified" if present and run.verified else "present" if present else
                                    "not produced (Crew 2 never ran)" if rel.startswith("crew2") else "not exported")})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("model.joblib is never loaded by this app (no unpickling): only its size and sha256 are shown.")
        options = [rel for rel in E.VIEWABLE if run.exists(rel)]
        if options:
            choice = st.selectbox("View", options, format_func=lambda r: E.VIEWABLE[r][1])
            kind = E.VIEWABLE[choice][0]
            if kind == "markdown":
                with st.container(border=True):
                    st.markdown(run.read_text(choice))
            elif kind == "html":
                import streamlit.components.v1 as components
                components.html(E.sandboxed_html(run.read_text(choice)), height=900)
            elif kind == "json":
                st.json(run.read_json(choice), expanded=1)
            elif kind == "csv_summary":
                s = run.csv_summary(choice)
                st.write(f"{s['rows']:,} rows, {s['n_columns']} columns; rows by split: {s['rows_by_split']}")
                st.write("Outcome columns present: " + (", ".join(s["outcome_columns_present"]) or "none"))
                st.caption("Rows are not displayed.")
                st.code(", ".join(s["columns"]), language=None)
            else:
                st.write(f"{run.size(choice):,} bytes · sha256 `{recorded.get(choice, 'n/a')}`")
                st.caption("Not loaded: a pickle file is never opened by the demo app.")

# ---------------------------------------------------------------- contract gate demos

with tabs[3]:
    st.write("Each break tampers with the Flow's **own copy** of the files Crew 1 wrote in the same run. Crew 1's "
             "outputs and all committed evidence stay unchanged. The gate fails closed: Crew 2 never starts.")
    for rid in [r for r in E.list_runs("canonical") if "break" in r]:
        br = E.load_run("canonical", rid)
        bm, bg = br.manifest, br.manifest.get("gate", {})
        with st.expander(f"{SCENARIO_TEXT.get(bm.get('scenario'), rid)} → {br.status}  ·  "
                         f"{', '.join(bg.get('violation_codes', []))}"):
            cols = st.columns(3)
            cols[0].metric("Crew 2 kickoffs", bm.get("crew2", {}).get("kickoffs", "n/a"))
            cols[1].metric("Crew 2 outputs", sum(1 for r in bm.get("required_outputs_sha256", {}) if r.startswith("crew2")))
            cols[2].metric("Evidence", "verified" if br.verified else "NOT verified")
            st.caption("Flow methods: " + " → ".join(bm.get("flow", {}).get("methods_executed", [])))
            st.dataframe([{k: v.get(k) for k in ("code", "column", "detail")} for v in bg.get("violations", [])],
                         hide_index=True, width="stretch")
    st.divider()
    st.subheader("Run a contract break now")
    scen = st.selectbox("Scenario", RUN.SCENARIOS, format_func=lambda s: SCENARIO_TEXT[s])
    st.caption("About 20 seconds: Crew 1 runs for real (scripted LLM responses), then the gate stops the run.")
    if st.button("Run this contract break", disabled=RUN.ml_python() is None or bool(RUN.active_runs())):
        try:
            st.session_state["last_started"] = RUN.start_flow(scen, 1)
        except (RuntimeError, ValueError) as e:
            st.error(str(e))
    last = st.session_state.get("last_started", "")
    if last.startswith("app-break-"):
        p = RUN.progress(last)
        if p["running"]:
            st.info(f"{last}: running ({p['elapsed_s'] or 0:.0f} s) · Crew 1 "
                    f"{'done' if p['stages']['crew1'] else 'running'} · gate {p['gate_decision'] or 'pending'}",
                    icon=":material/hourglass_top:")
        elif last in E.list_runs("local"):
            lr = E.load_run("local", last)
            lm = lr.manifest
            st.error(f"{last}: {lr.status} · " + ", ".join(lm.get("gate", {}).get("violation_codes", [])),
                     icon=":material/block:")
            cols = st.columns(3)
            cols[0].metric("Crew 2 kickoffs", lm.get("crew2", {}).get("kickoffs"))
            cols[1].metric("Crew 2 outputs", "present" if lr.exists("crew2/model.joblib") else "none")
            cols[2].metric("Evidence", "verified" if lr.verified else "NOT verified")

# ---------------------------------------------------------------- full run

with tabs[4]:
    st.write("Runs the whole Flow locally: Crew 1 kickoff → contract gate → Crew 2 kickoff (TRAIN-only: the 9 "
             "pre-registered grouped-CV runs, selection, final fit) → verification of the 8 outputs.")
    st.info("Expected runtime: about 4–5 minutes with 5 parallel jobs (Crew 1 ≈ 15 s, Crew 2 ≈ 4 min); slower "
            "with fewer CPU cores. The grids are never shortened. TEST is never reopened.", icon=":material/timer:")
    n_jobs = st.slider("Parallel jobs", 1, max(1, os.cpu_count() or 1), min(5, os.cpu_count() or 1))
    ok = st.checkbox("I understand this starts a CPU-heavy local run of several minutes.")
    if st.button("Start the full Flow run", type="primary",
                 disabled=not ok or RUN.ml_python() is None or bool(RUN.active_runs())):
        try:
            st.session_state["last_started"] = RUN.start_flow("none", n_jobs)
        except (RuntimeError, ValueError) as e:
            st.error(str(e))

    for rid in RUN.known_runs():                     # oldest first: a new run is appended, never re-uses a box
        p = RUN.progress(rid)
        with st.container(border=True):
            st.markdown(f"**{rid}**: " + ("running" if p["running"] else f"finished (exit code {p['exit_code']})"))
            s = p["stages"]
            st.progress(p["fraction"], text=" · ".join([
                ("✅" if s["crew1"] else "⏳") + " Crew 1",
                ("✅ " + str(p["gate_decision"]) if s["gate"] else "⏳ gate"),
                ("⏳ Crew 2 running" if s["crew2_started"] and not s["finished"] else
                 "✅ Crew 2" if s["crew2_started"] else "— Crew 2 not started"),
                ("✅ manifest" if s["finished"] else "⏳ manifest")]))
            if p["running"] and p["elapsed_s"] is not None:
                st.caption(f"elapsed {p['elapsed_s']:.0f} s")
            elif s["finished"] and rid in E.list_runs("local"):
                st.caption(f"Flow total {E.load_run('local', rid).manifest.get('timing', {}).get('total_seconds')} s "
                           "(from its flow_manifest.json)")
            tail = RUN.log_tail(rid)
            if tail:
                st.code(tail, language=None)
            if s["finished"] and st.button("Show this run", key=f"show-{rid}"):
                st.session_state["selected_run"] = rid
                st.info("Select **Local runs** in the sidebar to view it.")

# ---------------------------------------------------------------- how it works

with tabs[5]:
    st.markdown("""
**What is real and what is scripted**

| Part | Real or scripted |
|---|---|
| CrewAI Flow (`@start`, `@listen`, `@router`), crews, agents, tasks, tool execution | **Real** CrewAI execution |
| Data cleaning, EDA, contract, gate, features, grouped CV, selection, final fit, reports | **Real** deterministic Python |
| The agents' model responses (which tool to call, final answer text) | **Scripted**: an in-process transport answers every model request; no inference, no network |

**Boundaries**

* Crew 2 reads only the Flow's copy of Crew 1's two files; it never imports Crew 1 or reads the raw export.
* TEST is never reopened. The historical P4 TEST result is cited from a pinned reference.
* This app runs in its own environment, imports no ML code, never loads `model.joblib`, shows no data rows,
  and starts a run only when you press a button.
""")
    st.caption("Commands and details: docs/flow.md and docs/app.md.")

# ---------------------------------------------------------------- live refresh (only while a run is active)

if RUN.active_runs():
    time.sleep(2)
    st.rerun()
