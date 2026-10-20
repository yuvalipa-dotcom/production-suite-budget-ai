"""The agent-facing Crew-2 tools. Each agent gets ONLY its own tools; there is no generic file,
shell, code-execution or directory tool. Every tool runs deterministic Python (``crew2.ml``),
takes typed named arguments only (never a free-form mapping: CrewAI sends tools to Anthropic in
strict mode, p3-live-003), and records its call in the run trace.

No tool lets an agent choose, override or re-rank the model: ``apply_selection_rule`` takes no
argument and applies the pre-registered rule in code (correction C3).
"""

from __future__ import annotations

import io
import json
import threading
import time
from typing import Any, Callable

import joblib
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from crew1.tools.workspace import IsolationError, Workspace

from .. import claims as CL
from ..ml import data as D
from ..ml import features as F
from ..ml import leakage as L
from ..ml.protocol import RUNS, OrderError, Session
from ..ml.resample import LeakageError
from . import card as K
from .workspace import CHART_IDS, CONTRACTS_DIR

TOOLSETS = {
    "feature_engineer": ("build_handoff", "feature_availability", "run_leakage_checks"),
    "model_trainer": ("list_runs", "run_grouped_cv", "apply_selection_rule", "fit_final_models"),
    "evaluator": ("evaluate_on_test_once", "predict_demo", "render_chart", "write_model_card"),
}
OUTPUT_LIMIT = 12000


def compact(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def dumps(obj) -> str:
    """Byte-stable JSON: sorted keys, two-space indent, LF, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


class NoArgs(BaseModel):
    pass


class RunArgs(BaseModel):
    run_id: str = Field(description="One pre-registered run id from list_runs: " + ", ".join(RUNS))


class ChartArgs(BaseModel):
    chart_id: str = Field(description="One chart id: " + ", ".join(CHART_IDS))


_SECTION_TEXT = "Plain-text narrative for the model-card section '{}' (2 to 5 sentences; every number copied from " \
                "a tool result)."


class ModelCardArgs(BaseModel):
    summary: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["summary"]))
    intended_use: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["intended_use"]))
    data_and_target: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["data_and_target"]))
    features: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["features"]))
    protocol: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["protocol"]))
    cv_results: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["cv_results"]))
    selection: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["selection"]))
    test_results: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["test_results"]))
    secondary_results: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["secondary_results"]))
    limitations: str = Field(description=_SECTION_TEXT.format(K.SECTIONS["limitations"]))


assert tuple(ModelCardArgs.model_fields) == tuple(K.SECTIONS)


class Crew2Tool(BaseTool):
    """A deterministic tool bound to one run. Errors are returned to the agent, never raised."""

    _fn: Callable[..., Any] = PrivateAttr()
    _recorder: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    def bind(self, fn, recorder, lock=None) -> "Crew2Tool":
        self._fn, self._recorder = fn, recorder
        if lock is not None:
            self._lock = lock
        return self

    def _run(self, **kwargs) -> str:
        with self._lock:                              # one tool call at a time (CrewAI may run them in threads)
            start = time.monotonic()
            try:
                out = self._fn(**kwargs)
                text = out if isinstance(out, str) else compact(out)
                ok = True
            except IsolationError as e:
                text, ok = f"ERROR: rejected ({e.code}).", False
            except (OrderError, D.TestAlreadyOpened, KeyError, LookupError, ValueError) as e:
                text, ok = f"ERROR: {type(e).__name__}: {e}", False
            if len(text) > OUTPUT_LIMIT:
                text = text[:OUTPUT_LIMIT] + " ...[truncated]"
            if self._recorder is not None:
                self._recorder.add("tool_call", tool=self.name, args=kwargs, ok=ok, output=text,
                                   seconds=round(time.monotonic() - start, 3))
            return text


def _brief(summary: dict) -> dict:
    return {m: {"mean": v["mean"], "sd": v["sd"]} for m, v in summary.items()}


class ToolContext:
    """State shared by the tools of one run: the workspace and the deterministic protocol session."""

    def __init__(self, ws: Workspace, n_jobs: int = 1, session: Session | None = None):
        self.ws = ws
        self.session = session or Session(n_jobs=n_jobs, raw_dir=ws.raw_dir(), contracts_dir=CONTRACTS_DIR)
        self.lock = threading.Lock()
        self.leakage: dict | None = None
        self.charts: list[str] = []

    # ------------------------------------------------------------ persistence
    def _write_metrics(self) -> None:
        s = self.session
        doc = {"synthetic": True, "complete": False, "data": s.data_summary() if s.parts is not None else None,
               "cv": dict(s.cv), "selection": s.selection, "final": s.final, "test": s.test}
        try:
            doc = s.metrics() | {"complete": True}
        except OrderError:
            pass
        self.ws.write("out.metrics", dumps(doc))
        if s.oof:
            rows = [df.assign(run_id=k) for k, df in sorted(s.oof.items())]
            import pandas as pd
            out = pd.concat(rows)[["run_id", "production_id", "stage", "cv_fold", "p"]]
            self.ws.write("out.cv_predictions", out.to_csv(index=False, lineterminator="\n", float_format="%.10f"))

    def metrics(self) -> dict:
        if not self.ws.has_output("out.metrics"):
            raise LookupError("metrics.json does not exist yet")
        return json.loads(self.ws.read_output("out.metrics"))

    # ------------------------------------------------------------ C1 feature engineer
    def build_handoff(self):
        info = self.session.build_handoff()
        self._write_metrics()
        return {"verified": True, "clean_data_sha256": info["clean_data_sha256"], "rows": info["rows"],
                "exclusion_counts": info["exclusion_counts"], "counts": info["counts"],
                "note": "built in-process with the deterministic Policy C cleaning function; equal to the expected "
                        "deterministic output"}

    def feature_availability(self):
        from gate.reconstruct import FEATURE_SOURCES
        outcome = F.outcome_only_columns()
        table = [{"feature": f, "in_F27": f in F.F27, "in_F29": True,
                  "sources": sorted(FEATURE_SOURCES[f]),
                  "outcome_only_sources": sorted(s for s in FEATURE_SOURCES[f] if s.split(".", 1)[1] in outcome)}
                 for f in F.F29]
        doc = {"as_of_rule": "a record counts only if its date (effective_date, selected_date, created_date) is on or "
                             "before the snapshot's as_of_date; a cancellation counts only if cancelled_date is on or "
                             "before as_of_date",
               "features": table, "excluded_from_primary": list(F.PROPOSED_IN_PRODUCT),
               "outcome_only_columns": sorted(outcome)}
        self.ws.write("out.feature_availability", dumps(doc))
        return {"features": len(table), "F27": len(F.F27), "F29": len(F.F29),
                "features_with_outcome_only_sources": [t["feature"] for t in table if t["outcome_only_sources"]],
                "excluded_from_primary": doc["excluded_from_primary"], "written": "feature_availability.json"}

    def run_leakage_checks(self):
        s = self.session
        if s.parts is None:
            raise OrderError("build the handoff first")
        raw, _ = D.load_raw(s.raw_dir)
        ledgers, _ = L.cleaned_ledgers(raw)
        import pandas as pd
        hand = pd.read_csv(io.StringIO(s.handoff_text), dtype=str, keep_default_na=False)
        report = {"declared_trace": L.declared_trace(), "dynamic_trace": L.dynamic_trace(ledgers, hand),
                  "as_of_mutations": L.as_of_mutations(ledgers, L.sample_keys(hand, 5)),
                  "groups": L.group_checks(s.parts.train, s.parts.test_features, s.parts.demo_features,
                                           D.frozen_split(CONTRACTS_DIR)),
                  "test_vault_opened": s.parts.vault.opened}
        report["ok"] = all(report[k]["ok"] for k in ("declared_trace", "dynamic_trace", "as_of_mutations", "groups")) \
            and not report["test_vault_opened"]
        self.leakage = report
        self.ws.write("out.leakage_report", dumps(report))
        return {k: (v["ok"] if isinstance(v, dict) else v) for k, v in report.items()} | {
            "mutations_checked": report["as_of_mutations"]["mutations_checked"], "written": "leakage_report.json"}

    # ------------------------------------------------------------ C2 model trainer
    def list_runs(self):
        return {"runs": RUNS, "selection_rule": "applied by code: lowest mean grouped-CV Brier; tie within 0.005 of "
                                                "the best -> higher mean PR-AUC -> simpler model (B0 < LR < GB)",
                "required_before_selection": [k for k, v in RUNS.items() if v["role"] == "primary"]}

    def run_grouped_cv(self, run_id: str):
        r = self.session.run_cv(run_id.strip() if isinstance(run_id, str) else run_id)
        self._write_metrics()
        return {"run_id": r["run_id"], "role": r["role"], "summary": _brief(r["summary"]),
                "params_per_fold": [f["params"] for f in r["folds"]], "audit": r["audit"]}

    def apply_selection_rule(self):
        sel = self.session.apply_selection_rule()
        self._write_metrics()
        return sel

    def fit_final_models(self):
        fin = self.session.fit_final_models()
        buf = io.BytesIO()
        joblib.dump({"synthetic": True, "label": K.DEMO_LABEL, "feature_set": F.PRIMARY_FEATURE_SET,
                     "features": list(F.F27), "models": self.session.final_models,
                     "model_sha256": {k: v["model_sha256"] for k, v in fin.items()}}, buf)
        self.ws.write("out.model", buf.getvalue())
        self._write_metrics()
        return {k: {kk: v[kk] for kk in ("candidate", "target", "params", "model_sha256", "train_rows", "audit")}
                for k, v in fin.items()} | {"written": "model.joblib"}

    # ------------------------------------------------------------ C3 evaluator
    def evaluate_on_test_once(self):
        t = self.session.evaluate_on_test_once()
        self._write_metrics()
        return {"opened": t["opened"], "rows": t["rows"], "productions": t["productions"],
                **{n: {"overall": t[n]["overall"], "bootstrap_ci": t[n]["bootstrap_ci"],
                       "ece": t[n]["reliability"]["ece"]} for n in ("primary", "secondary")}}

    def predict_demo(self):
        demo = self.session.predict_demo()
        self.ws.write("out.demo_predictions", demo.to_csv(index=False, lineterminator="\n", float_format="%.6f"))
        self._write_metrics()
        return {"rows": len(demo), "metrics": None, "label": K.DEMO_LABEL, "written": "demo_predictions.csv"}

    def render_chart(self, chart_id: str):
        if chart_id not in CHART_IDS:
            raise KeyError(f"unknown chart id; choose one of {list(CHART_IDS)}")
        m = self.metrics()
        if chart_id == "reliability_test" and not m.get("test"):
            raise LookupError("evaluate_on_test_once first")
        if chart_id == "cv_brier" and not m.get("cv"):
            raise LookupError("no CV results yet")
        data = K.render_chart(chart_id, m)
        self.ws.write(f"out.chart.{chart_id}", data)
        self.charts.append(chart_id)
        return {"chart_id": chart_id, "rendered": True, "png_bytes": len(data)}

    def write_model_card(self, **sections: str):
        m = self.metrics()
        if not m.get("complete"):
            raise OrderError("the model card needs every CV run, the test evaluation and the demo predictions")
        page = K.assemble(sections, m)
        self.ws.write("out.model_card", page)
        text = "\n\n".join(K.narrative(page).values())
        return {"written": "model_card.md",
                "sections_missing_or_short": [s for s in K.SECTIONS if len(sections.get(s, "").strip()) < K.MIN_NARRATIVE],
                "unverified_quantitative_claims": CL.unverified(text, m),
                "forbidden_claims": CL.forbidden_claims(text),
                "charts_rendered": sorted(set(self.charts))}


SPECS = {
    "build_handoff": (NoArgs, "Rebuild the Policy C clean data in-process from the raw export with the deterministic "
                              "cleaning function, verify it equals the expected deterministic output, and partition "
                              "it by the frozen groups (test labels stay sealed)."),
    "feature_availability": (NoArgs, "Write feature_availability.json: each of the 29 features with its raw source "
                                     "columns, the as-of rule and the primary (F27) / research (F29) membership."),
    "run_leakage_checks": (NoArgs, "Run the leakage checks (declared and dynamic outcome trace, as-of mutation "
                                   "two-way proof, frozen group disjointness) and write leakage_report.json."),
    "list_runs": (NoArgs, "List the 9 pre-registered grouped-CV runs and the selection rule."),
    "run_grouped_cv": (RunArgs, "Run one pre-registered grouped-CV run on train (5 frozen folds; tuning and "
                                "calibration on group-disjoint inner splits). Returns the fold summary."),
    "apply_selection_rule": (NoArgs, "Apply the pre-registered selection rule in code to the primary CV results. "
                                     "It takes no argument: the rule, not an agent, selects the model."),
    "fit_final_models": (NoArgs, "Refit the selected candidate on all of train (primary is_overrun and secondary "
                                 "is_material_overrun), freeze and hash both, and write model.joblib."),
    "evaluate_on_test_once": (NoArgs, "Open the sealed test set ONCE and evaluate both frozen models (metrics, "
                                      "per stage, reliability, production-level bootstrap intervals). A second call "
                                      "is refused."),
    "predict_demo": (NoArgs, "Predict the demo snapshots (inference only, no metric) and write "
                             "demo_predictions.csv, labelled as experimental / demo predictions."),
    "render_chart": (ChartArgs, "Render one chart PNG from metrics.json only."),
    "write_model_card": (ModelCardArgs, "Write model_card.md: your narrative in each of the ten section fields; the "
                                        "header, the disclosures and every table are inserted by the tool. Returns "
                                        "checks of your text (missing sections, unverified numbers, forbidden "
                                        "claims)."),
}
assert set(SPECS) == {t for ts in TOOLSETS.values() for t in ts}


def build_tools(ctx: ToolContext, recorder=None) -> dict[str, list[BaseTool]]:
    out = {}
    for agent, names in TOOLSETS.items():
        tools = []
        for name in names:
            schema, description = SPECS[name]
            tool = Crew2Tool(name=name, description=description, args_schema=schema)
            tools.append(tool.bind(getattr(ctx, name), recorder, ctx.lock))
        out[agent] = tools
    return out


__all__ = ["TOOLSETS", "ToolContext", "build_tools", "LeakageError"]
