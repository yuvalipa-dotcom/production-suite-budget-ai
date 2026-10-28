"""Crew 2 for the Flow: a real CrewAI ``crew.kickoff()`` with three agents (P5-D). ALL DATA IS SYNTHETIC.

    python -m budget_flow.crew2_kickoff --clean-data PATH --contract PATH --out DIR [--flow-contract PATH] [--n-jobs N]

The CrewAI orchestration is real: CrewAI builds the agents and tasks, runs the tasks in order, sends each
agent's requests to its LLM, executes the tool calls the responses ask for, and passes task context on.
By default the model responses are NOT LLM inference: ``crew_common.scripted.ScriptedTransport`` answers
every provider request in-process from a fixed script, every scripted text is marked, and no request leaves
the process. With ``llm_mode="live"`` every response comes from the Anthropic API instead
(``crew_common.live.RecordingTransport``: each provider call recorded, ``max_rpm`` 20 as in Crew 1, a shared
request and token budget), with no fallback to scripted responses. In both modes every number, check, fit
and artifact is computed by deterministic Python tools.

Agents (``budget_flow/config/crew2_agents.yaml``) and their tools:

* Handoff & Feature Validator: ``validate_handoff`` (the P5-B ``load_handoff``: flow-1 contract, labels
  redacted, test vault sealed), ``write_features`` (features.csv and its schema);
* TRAIN-only Model Trainer: ``check_preregistration``, ``list_runs``, ``run_grouped_cv`` (x9),
  ``apply_selection_rule``, ``fit_final_models``, ``verify_reproducibility`` (the P5-C ``TrainOnlySession``
  and ``p4_reference.verify``);
* Evaluation & Ethics Author: ``write_model``, ``write_evaluation_report``, ``write_model_card`` (the P5-C
  renderers), ``verify_artifacts``.

Boundary: the inputs are Crew 1's delivered files only, copied into a workspace outside the repository;
no tool takes a path; nothing imports Crew 1; the raw export is never read; TEST stays sealed. A contract
violation, a TEST access, a changed pre-registration or a failed reproduction is fatal: it stops every
later tool call and fails the run after kickoff, and nothing is written.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

import crew_common  # noqa: F401  (forces the telemetry settings before CrewAI is imported)
from crew_common import kickoff as K
from crew_common import live as L
from crew_common import scripted as S
from crew_common import tools as CT
from crew_common import workspace as W
from crew_common.llm import DEFAULT_MODEL, LLM_SETTINGS, make_llm
from crew_common.recorder import Recorder
from crew2.ml import protocol as P
from crew2.ml.features import F27

from . import contract as C
from . import handoff as FH
from . import outputs as O
from . import p4_reference as R
from . import train as T

CONFIG = Path(__file__).resolve().parent / "config"
STORAGE_DIR = "psba-flow-crew2"
TOOLSETS = {
    "handoff_validator": ("validate_handoff", "write_features"),
    "train_only_modeler": ("check_preregistration", "list_runs", "run_grouped_cv", "apply_selection_rule",
                           "fit_final_models", "verify_reproducibility"),
    "evaluation_author": ("write_model", "write_evaluation_report", "write_model_card", "verify_artifacts"),
}
# One model request per executor iteration; sized for one tool call per turn plus the final answer.
AGENT_LIMITS = {
    "handoff_validator": {"max_iter": 6, "max_execution_time": 600},
    "train_only_modeler": {"max_iter": 20, "max_execution_time": 7200},
    "evaluation_author": {"max_iter": 8, "max_execution_time": 600},
}
AGENT_MAX_RETRY_LIMIT = 0
LIVE_MAX_RPM = 20                                         # live runs only: the Crew 1 (P3) request-rate limit
READ_NAMES = {"in.clean_data": "in/clean_data.csv", "in.dataset_contract": "in/dataset_contract.json",
              "in.flow_contract": "in/flow_contract.json"}
OUTPUT_NAMES = {"out.features": "out/features.csv", "out.features_schema": "out/features.schema.json",
                "out.model": "out/model.joblib", "out.evaluation_report": "out/evaluation_report.md",
                "out.model_card": "out/model_card.md"}
OUTPUT_FILES = {k: v.split("/", 1)[1] for k, v in OUTPUT_NAMES.items()}
class Crew2RunFailed(RuntimeError):
    """The Crew 2 kickoff did not complete every step and check: nothing is written."""


FATAL = (C.ContractViolationError, FH.TestAccessError, FH.PathPolicyError, R.ReproducibilityError, T.FlowPathError,
         Crew2RunFailed)


class RunArgs(BaseModel):
    run_id: str = Field(description="One pre-registered run id from list_runs: " + ", ".join(P.RUNS))


# ---------------------------------------------------------------- the tools' shared run state

class Crew2FlowContext:
    """What the tools of one Crew 2 run share. Agents never see a path: the inputs are workspace names."""

    def __init__(self, ws: W.Workspace, n_jobs: int = 1, has_flow_contract: bool = False):
        self.ws, self.n_jobs, self.has_flow_contract = ws, n_jobs, has_flow_contract
        self.ref, self.pred_ref = R.load()
        self.handoff: FH.FlowHandoff | None = None
        self.session: T.TrainOnlySession | None = None
        self.metrics: dict | None = None
        self.repro: dict | None = None
        self.preregistration_checked = False
        self.step_seconds: dict[str, float] = {}
        self.written: dict[str, bytes] = {}
        self.checks: dict | None = None

    def _need(self, cond, msg):
        if not cond:
            raise LookupError(msg)

    def input_sha(self) -> dict:
        names = {"clean_data.csv": "in.clean_data", "dataset_contract.json": "in.dataset_contract"}
        if self.has_flow_contract:
            names["flow_contract.json"] = "in.flow_contract"
        return {k: R.sha256_hex(self.ws.read_bytes(v)) for k, v in names.items()}

    def _write(self, name: str, data: bytes) -> dict:
        self.ws.write(name, data)
        self.written[name] = data
        return {"written": OUTPUT_FILES[name], "bytes": len(data), "sha256": R.sha256_hex(data)}

    # ------------------------------------------------------------ handoff & feature validator
    def validate_handoff(self):
        self._need(self.handoff is None, "the handoff is already validated")
        h = FH.load_handoff(self.ws.resolve("in.clean_data"), self.ws.resolve("in.dataset_contract"),
                            self.ws.resolve("in.flow_contract") if self.has_flow_contract else None)
        self.handoff, self.session = h, T.TrainOnlySession(h, n_jobs=self.n_jobs)
        return {"contract": C.FLOW_CONTRACT_VERSION, "ok": h.report.ok, "violations": len(h.report.violations),
                "clean_data_sha256": h.report.clean_data_sha256,
                "rows": {k: v["rows"] for k, v in h.counts.items()},
                "productions": {k: v["productions"] for k, v in h.counts.items()},
                "monetary_scale_ratio_range": [round(min(h.report.scale_ratios.values()), 4),
                                               round(max(h.report.scale_ratios.values()), 4)],
                "non_train_labels": "redacted before validation", "test_vault": type(h.parts.vault).__name__}

    def write_features(self):
        self._need(self.handoff is not None, "call validate_handoff first")
        out = self._write("out.features", O.render_features(self.handoff.parts))
        self._write("out.features_schema", O.render_schema())
        return out | {"rows": sum(c["rows"] for c in self.handoff.counts.values()), "features": len(F27),
                      "columns": list(O.FEATURE_COLUMNS[:4]) + [f"... {len(F27)} F27 features"],
                      "outcome_columns_present": False, "schema": "features.schema.json"}

    # ------------------------------------------------------------ TRAIN-only model trainer
    def check_preregistration(self):
        R.check_preregistration(self.ref)
        self.preregistration_checked = True
        return {"unchanged": True, "reference": R.REFERENCE_VERSION, "seed": self.ref["preregistration"]["seed"],
                "candidates": self.ref["preregistration"]["candidate_order"],
                "grid_sizes": {k: len(v) for k, v in self.ref["preregistration"]["grids"].items()},
                "brier_tie": self.ref["preregistration"]["brier_tie"]}

    def list_runs(self):
        return {"runs": list(P.RUNS), "primary_runs_for_selection": list(R.PRIMARY_RUNS),
                "selection_rule": "applied by code: lowest mean grouped-CV Brier; tie within 0.005 -> higher mean "
                                  "PR-AUC -> simpler model (B0 < LR < GB)"}

    def run_grouped_cv(self, run_id: str):
        self._need(self.session is not None, "call validate_handoff first (handoff validator)")
        self._need(self.preregistration_checked, "call check_preregistration first")
        run_id = run_id.strip()
        t = time.monotonic()
        r = self.session.run_cv(run_id)
        self.step_seconds.setdefault(run_id, round(time.monotonic() - t, 1))
        return {"run_id": run_id, "role": r["role"],
                "summary": {m: {"mean": v["mean"], "sd": v["sd"]} for m, v in r["summary"].items()
                            if m in ("brier", "average_precision", "roc_auc")},
                "params_per_fold": [f["params"] for f in r["folds"]], "audit_violations": r["audit"]["violations"]}

    def apply_selection_rule(self):
        self._need(self.session is not None, "call validate_handoff first (handoff validator)")
        sel = self.session.apply_selection_rule()
        return {k: sel[k] for k in ("selected", "decided_by", "tie_set", "best_mean_brier", "brier_minus_best")}

    def fit_final_models(self):
        self._need(self.session is not None and self.session.selection is not None, "apply the selection rule first")
        t = time.monotonic()
        fin = self.session.fit_final_models()
        self.step_seconds.setdefault("final_refit", round(time.monotonic() - t, 1))
        return {k: {kk: v[kk] for kk in ("candidate", "target", "params", "train_rows", "model_sha256")}
                | {"audit_violations": v["audit"]["violations"]} for k, v in fin.items()}

    def verify_reproducibility(self):
        self._need(self.session is not None and self.session.final is not None, "fit the final models first")
        metrics = self.session.train_only_metrics()
        repro = R.verify(metrics["cv"], metrics["selection"], metrics["final"], R._prediction_rows(self.session),
                         self.ref, self.pred_ref, R.mode_for(R.environment(), self.ref))
        self.metrics, self.repro = metrics, repro
        return repro

    # ------------------------------------------------------------ evaluation & ethics author
    def _ready(self):
        self._need(self.repro is not None, "the trainer has not verified the reproduction yet")

    def write_model(self):
        self._ready()
        return self._write("out.model", O.render_model(self.session, self.metrics, self.handoff))

    def write_evaluation_report(self):
        self._ready()
        text = O.render_report(self.metrics, self.ref, self.repro, self.handoff, self.input_sha())
        return self._write("out.evaluation_report", text.encode("utf-8")) | {
            "sections": ["A. Current TRAIN-only grouped cross-validation (this run)",
                         "B. Historical P4 TEST evaluation (cited, not recomputed)"]}

    def write_model_card(self):
        self._ready()
        return self._write("out.model_card", O.render_card(self.metrics, self.ref, self.repro).encode("utf-8"))

    def verify_artifacts(self):
        self._ready()
        files = {k: self.ws.read_output(k) for k in ("out.features", "out.model", "out.evaluation_report",
                                                     "out.model_card")}
        header = files["out.features"].decode("utf-8").split("\n", 1)[0].split(",")
        report = files["out.evaluation_report"].decode("utf-8")
        card = files["out.model_card"].decode("utf-8")
        vault = self.session.parts.vault
        checks = {
            "four_artifacts_written": all(files.values()),
            "artifacts_equal_what_the_tools_wrote": all(files[k] == self.written.get(k) for k in files),
            "features_header_is_ids_plus_27_features": header == list(O.FEATURE_COLUMNS),
            "features_rows": files["out.features"].count(b"\n") - 1 == sum(c["rows"] for c in self.handoff.counts.values()),
            "no_outcome_column": not set(header) & set(C.LABEL_COLUMNS),
            "report_separates_current_cv_and_historical_test":
                0 <= report.find("## A. Current TRAIN-only") < report.find("## B. Historical P4 TEST"),
            "report_states_test_not_reopened": "This run did not reopen TEST" in report,
            "card_has_ethical_considerations": "## Ethical considerations" in card,
            "card_discloses_synthetic_data": "## Synthetic-data disclosure" in card,
            "no_production_id_in_documents": "SYN-" not in report + card,
            "model_file_under_200_kb": len(files["out.model"]) < 200_000,
            "test_vault_sealed_and_unopened": isinstance(vault, FH.SealedVault) and not vault.opened
                                              and self.session.test is None,
            "reproduction_verified": bool(self.repro["ok"]),
        }
        self.checks = checks
        if not all(checks.values()):
            raise Crew2RunFailed(f"artifact checks failed: {[k for k, v in checks.items() if not v]}")
        return {"all_passed": True, "checks": checks}


SPECS = {
    "validate_handoff": (CT.NoArgs, "Validate Crew 1's delivered clean_data.csv and dataset_contract.json against the "
                                    "Flow contract flow-1 (every non-train label redacted first) and partition them "
                                    "by the frozen split; the test vault is sealed."),
    "write_features": (CT.NoArgs, "Write features.csv (identifiers, split and fold, the 27 approved F27 features, no "
                                  "outcome column) and features.schema.json."),
    "check_preregistration": (CT.NoArgs, "Verify that the seed, folds, grids, features and selection rule equal the "
                                         "approved P4 pre-registration. Must run before any fitting."),
    "list_runs": (CT.NoArgs, "List the 9 pre-registered grouped-CV runs and the selection rule."),
    "run_grouped_cv": (RunArgs, "Run one pre-registered grouped-CV run on the train group only (5 frozen folds; "
                                "tuning and calibration on group-disjoint inner splits)."),
    "apply_selection_rule": (CT.NoArgs, "Apply the pre-registered selection rule in code to the primary CV results. "
                                        "It takes no argument: the rule, not an agent, selects the model."),
    "fit_final_models": (CT.NoArgs, "Refit the selected candidate on all of train (primary is_overrun, secondary "
                                    "is_material_overrun) and hash both models."),
    "verify_reproducibility": (CT.NoArgs, "Compare the CV results, selection, parameters, predictions and model "
                                          "hashes with the approved P4 run (exact in the pinned environment, "
                                          "pre-registered tolerances elsewhere). A difference fails the run."),
    "write_model": (CT.NoArgs, "Write model.joblib: the two TRAIN-only final models with their hashes and provenance."),
    "write_evaluation_report": (CT.NoArgs, "Write evaluation_report.md: A. the current TRAIN-only CV, selection and "
                                           "provenance; B. the historical P4 TEST result, cited, not recomputed."),
    "write_model_card": (CT.NoArgs, "Write model_card.md: purpose, training data, evaluation, limitations, "
                                    "synthetic-data disclosure and ethical considerations."),
    "verify_artifacts": (CT.NoArgs, "Check the four Crew 2 artifacts (features.csv, model.joblib, "
                                    "evaluation_report.md, model_card.md) and the sealed test vault."),
}
assert set(SPECS) == {t for ts in TOOLSETS.values() for t in ts}


# ---------------------------------------------------------------- the scripted model responses

def _validator_final(ctx: Crew2FlowContext) -> str:
    c = ctx.handoff.counts
    return (f"Handoff accepted under contract {C.FLOW_CONTRACT_VERSION} with {len(ctx.handoff.report.violations)} "
            f"violations: train {c['train']['rows']}, test {c['test']['rows']}, demo {c['demo']['rows']} rows; "
            f"non-train labels redacted; test vault sealed. features.csv written with {len(F27)} features and no "
            f"outcome column. (Values quoted from the tool results by the script.)")


def _trainer_final(ctx: Crew2FlowContext) -> str:
    s, fin = ctx.metrics["selection"], ctx.metrics["final"]["primary"]
    return (f"The pre-registered rule selected {s['selected']} ({s['decided_by']}), parameters {fin['params']}, "
            f"refit on {fin['train_rows']} train rows. Reproduction of the approved P4 CV: {ctx.repro['mode']}, "
            f"ok={ctx.repro['ok']}. (Values quoted from the tool results by the script.)")


def _author_final(ctx: Crew2FlowContext) -> str:
    return (f"model.joblib, evaluation_report.md and model_card.md written; {len(ctx.checks)} artifact checks "
            f"passed, test vault sealed and unopened. (Values quoted from the tool results by the script.)")


PLANS = {
    "handoff_validator": [[("validate_handoff", {})], [("write_features", {})], _validator_final],
    "train_only_modeler": [[("check_preregistration", {})], [("list_runs", {})],
                           *[[("run_grouped_cv", {"run_id": r})] for r in P.RUNS],
                           [("apply_selection_rule", {})], [("fit_final_models", {})],
                           [("verify_reproducibility", {})], _trainer_final],
    "evaluation_author": [[("write_model", {})], [("write_evaluation_report", {})], [("write_model_card", {})],
                          [("verify_artifacts", {})], _author_final],
}
AGENT_BY_TOOLS = {frozenset(names): key for key, names in TOOLSETS.items()}


# ---------------------------------------------------------------- the crew

def load_config() -> tuple[dict, dict]:
    return (yaml.safe_load((CONFIG / "crew2_agents.yaml").read_text(encoding="utf-8")),
            yaml.safe_load((CONFIG / "crew2_tasks.yaml").read_text(encoding="utf-8")))


def build_crew(ctx: Crew2FlowContext, recorder: Recorder, state: CT.RunState, model: str = DEFAULT_MODEL,
               max_rpm: int | None = None):
    """Three agents, each with only its own tools and its own LLM; three sequential tasks; the Crew 1/P4 safety
    settings (no memory, knowledge, embeddings, delegation, code execution, reasoning, planning or cache)."""
    import warnings

    from crewai import Agent, Crew, Process, Task
    from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage

    warnings.filterwarnings("ignore", message="method callbacks cannot be serialized")
    agents_cfg, tasks_cfg = load_config()
    tools = CT.build_tools(SPECS, TOOLSETS, ctx, recorder, state)
    agents = {}
    for key in TOOLSETS:
        cfg = agents_cfg[key]
        agents[key] = Agent(
            role=cfg["role"].strip(), goal=cfg["goal"].strip(), backstory=cfg["backstory"].strip(),
            tools=tools[key], llm=make_llm(model), allow_delegation=False, memory=False, knowledge_sources=None,
            embedder=None, allow_code_execution=False, reasoning=False, planning=False, multimodal=False,
            inject_date=False, cache=False, max_retry_limit=AGENT_MAX_RETRY_LIMIT, respect_context_window=True,
            step_callback=recorder.step, verbose=False, max_rpm=max_rpm, **AGENT_LIMITS[key])
    tasks = {}
    for name, cfg in tasks_cfg.items():
        tasks[name] = Task(name=name, description=cfg["description"].strip(),
                           expected_output=cfg["expected_output"].strip(), agent=agents[cfg["agent"]],
                           context=[tasks[c] for c in cfg.get("context", [])] or None)
    crew = Crew(agents=list(agents.values()), tasks=list(tasks.values()), process=Process.sequential,
                memory=False, cache=False, planning=False, tracing=False, share_crew=False,
                knowledge_sources=None, embedder=None, max_rpm=max_rpm, verbose=False,
                output_log_file=None, stream=False, task_callback=recorder.task)
    crew._task_output_handler.storage = KickoffTaskOutputsSQLiteStorage(db_path=str(ctx.ws.root / "tmp" / "kickoff.db"))
    return crew


@contextlib.contextmanager
def _in_workspace(root: Path):
    """The workspace as working directory and Crew 2's own CrewAI app-data folder name; both restored after."""
    saved_cwd, saved_storage = os.getcwd(), os.environ.get("CREWAI_STORAGE_DIR")
    os.chdir(root)
    os.environ["CREWAI_STORAGE_DIR"] = STORAGE_DIR
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        if saved_storage is None:
            os.environ.pop("CREWAI_STORAGE_DIR", None)
        else:
            os.environ["CREWAI_STORAGE_DIR"] = saved_storage


def _python_tool_calls(recorder: Recorder) -> dict:
    calls = [e for e in recorder.events if e["event"] == "tool_call"]
    by_agent: dict[str, list[str]] = {}
    for e in calls:
        by_agent.setdefault(e["agent"], []).append(e["tool"])
    return {"computed_by": "python", "calls": len(calls), "failed": sum(not e["ok"] for e in calls),
            "by_agent": by_agent}


def run_crew2_kickoff(clean_csv: Path, contract_json: Path, out_dir: Path, flow_contract_json: Path | None = None,
                      n_jobs: int = 1, log=None, llm_mode: str = "scripted", budget: L.LiveBudget | None = None) -> dict:
    """Kick off the Flow's Crew 2 on Crew 1's delivered files; write the artifacts and the evidence to ``out_dir``."""
    if llm_mode not in ("scripted", "live"):
        raise ValueError(f"unknown llm_mode {llm_mode!r}")
    live = llm_mode == "live"
    source = L.SOURCE if live else S.SOURCE
    out = T._check_out_dir(out_dir)
    for p in (clean_csv, contract_json, *((flow_contract_json,) if flow_contract_json else ())):
        FH._read(Path(p))                                  # the path policy: never the raw export, never a link
    inputs = {"in/clean_data.csv": Path(clean_csv), "in/dataset_contract.json": Path(contract_json)}
    if flow_contract_json:
        inputs["in/flow_contract.json"] = Path(flow_contract_json)
    started, t0 = datetime.now(timezone.utc), time.monotonic()
    ws = W.stage("crew2_flow_ws_", inputs, READ_NAMES, OUTPUT_NAMES)
    try:
        recorder = Recorder(ws.root)
        recorder.add("run_started", orchestration="CrewAI crew.kickoff()", llm_responses=source)
        ctx = Crew2FlowContext(ws, n_jobs=n_jobs, has_flow_contract=bool(flow_contract_json))
        state = CT.RunState(FATAL)
        crew = build_crew(ctx, recorder, state, max_rpm=LIVE_MAX_RPM if live else None)
        transport = (L.RecordingTransport(AGENT_BY_TOOLS, budget or L.LiveBudget.default(), "crew2", ctx, log) if live
                     else S.ScriptedTransport(PLANS, AGENT_BY_TOOLS, ctx))
        evidence = K.KickoffEvidence()
        kickoff_started = L.utc_now()
        with transport.installed(), evidence.listening(), _in_workspace(ws.root):
            result = crew.kickoff()
        kickoff_finished = L.utc_now()
        kickoff_seconds = round(time.monotonic() - t0, 1)
        recorder.add("run_finished", fatal=state.fatal)
        if state.fatal:
            raise Crew2RunFailed(state.fatal)
        if ctx.checks is None or not all(ctx.checks.values()):
            raise Crew2RunFailed("the evaluation author did not verify the artifacts")
        if transport.network_blocked:
            raise Crew2RunFailed(f"network access was attempted: {transport.network_blocked[:3]}")
        silent = [a for a in TOOLSETS if live and not transport.summary()["successful_responses_by_agent"].get(a)]
        if silent:
            raise Crew2RunFailed(f"agents without a live provider response: {silent}")
        out.mkdir(parents=True, exist_ok=True)
        for name, fname in OUTPUT_FILES.items():
            (out / fname).write_bytes(ws.read_output(name))
        kickoff = {
            "synthetic": True,
            "orchestration": "real CrewAI crew.kickoff(): Crew, 3 Agents, 3 Tasks, sequential process",
            "llm_mode": llm_mode,
            "llm_responses": source,
            "disclosure": (("The CrewAI orchestration is real and every agent response came from live Anthropic "
                            f"inference ({DEFAULT_MODEL}); each provider call is recorded in llm_transport.exchanges "
                            "(no header, no key). ") if live else
                           ("The CrewAI orchestration is real. The model responses are scripted (not LLM inference); "
                            f"every scripted text starts with '{S.SCRIPTED_MARK}'. "))
                          + "Every number, check, fit and artifact was computed by deterministic Python tools "
                            "(computed_by: python).",
            "agents": [{"key": k, "role": a.role, "tools": [t.name for t in a.tools]}
                       for k, a in zip(TOOLSETS, crew.agents)],
            "tasks": [{"name": t.name, "agent": t.agent.role} for t in crew.tasks],
            "task_outputs": [{"task": o.name, "agent": o.agent, "source": "llm" if live else "scripted",
                              "text": recorder.redact(o.raw)}
                             for o in result.tasks_output],
            "crewai_events": evidence.summary() | {"events": evidence.events},
            "llm_transport": transport.summary() | {"exchanges": transport.exchanges},
            "python_tool_calls": _python_tool_calls(recorder),
            "token_usage_reported_by_crewai": (crew.usage_metrics.model_dump()
                                               if hasattr(crew.usage_metrics, "model_dump") else None),
            "model_id_configured": f"anthropic/{DEFAULT_MODEL}" + ("" if live else
                                   " (never called: answered by the scripted transport)"),
            "llm_settings": LLM_SETTINGS, "agent_limits": AGENT_LIMITS, "agent_max_retry_limit": AGENT_MAX_RETRY_LIMIT,
            "max_rpm": LIVE_MAX_RPM if live else None,
            "timing": {"kickoff_started_utc": kickoff_started, "kickoff_finished_utc": kickoff_finished},
        }
        files = {fname: ctx.written[name] for name, fname in OUTPUT_FILES.items()}
        manifest = O.make_manifest(ctx.session, ctx.metrics, ctx.ref, ctx.repro, ctx.handoff, ctx.input_sha(), files,
                                   {"started_at": started.isoformat(timespec="seconds"),
                                    "kickoff_seconds": kickoff_seconds, "steps_seconds": ctx.step_seconds,
                                    "n_jobs": n_jobs})
        manifest |= {"step": "Flow Crew 2 kickoff (P5-D)", "orchestration": {
            "kickoff": "crew_kickoff.json", "trace": "trace.jsonl", "agents_executed":
                kickoff["crewai_events"]["agents_executed"], "llm_responses": source}}
        (out / "trace.jsonl").write_text(recorder.jsonl(), encoding="utf-8", newline="\n")
        (out / "crew_kickoff.json").write_text(json.dumps(kickoff, indent=2, default=str) + "\n", encoding="utf-8",
                                               newline="\n")
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        if log:
            log(f"Crew 2 kickoff finished in {kickoff_seconds} s; agents executed: "
                f"{kickoff['crewai_events']['agents_executed']}")
        return manifest
    finally:
        ws.cleanup()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean-data", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--flow-contract", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args(argv)
    m = run_crew2_kickoff(args.clean_data, args.contract, args.out, args.flow_contract, args.n_jobs,
                          log=lambda s: print(s, flush=True))
    print(f"selected {m['selected_model']['candidate']} {m['selected_model']['params']}; artifacts in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
