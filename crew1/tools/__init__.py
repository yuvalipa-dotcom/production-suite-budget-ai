"""The agent-facing Crew-1 tools. Each agent gets ONLY its own tools; there is no generic file,
shell, code-execution or directory tool. Every tool runs deterministic Python, takes logical
names (never paths), and records its call in the run trace."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from . import cleaning as C
from . import contract as K
from . import eda as E
from . import narrative_checks as N
from .gate_tool import run_gate
from .workspace import RAW_TABLES, IsolationError, Sources, Workspace

TOOLSETS = {
    "ingestion_engineer": ("load_raw_table", "detect_defects", "apply_policy_c", "eligibility_summary"),
    "exploratory_analyst": ("list_analyses", "run_analysis", "render_chart", "assemble_eda_report"),
    "contract_author": ("build_dataset_contract", "run_handoff_gate", "write_insights"),
}
OUTPUT_LIMIT = 12000


def compact(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class NoArgs(BaseModel):
    pass


class TableArgs(BaseModel):
    name: str = Field(description="Logical raw table name: " + ", ".join(f"raw.{t}" for t in RAW_TABLES))


class AnalysisArgs(BaseModel):
    analysis_id: str = Field(description="One id from list_analyses: " + ", ".join(E.ANALYSES))


class ChartArgs(BaseModel):
    chart_id: str = Field(description="One chart id from list_analyses: " + ", ".join(E.CHARTS))


# CrewAI 1.15.28 sends every native tool to Anthropic with strict: true, which closes every object
# (additionalProperties: false). A dict-typed argument then admits only {} (p3-live-003: 22 empty
# assemble_eda_report calls), so every argument below is a named field or a list of typed objects.
_SECTION_TEXT = "Plain-text interpretation for the report section '{}'; blank lines separate paragraphs."


class ReportArgs(BaseModel):
    overview: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["overview"][0]))
    data_quality: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["data_quality"][0]))
    prevalence: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["prevalence"][0]))
    signed_gap: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["signed_gap"][0]))
    budget_revisions: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["budget_revisions"][0]))
    fee_terms: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["fee_terms"][0]))
    correlations: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["correlations"][0]))
    frozen_split: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["frozen_split"][0]))
    limitations: str = Field(description=_SECTION_TEXT.format(E.SECTIONS["limitations"][0]))


assert tuple(ReportArgs.model_fields) == tuple(E.SECTIONS)


class ColumnDescription(BaseModel):
    column: str = Field(description="An exact column name of clean_data.csv.")
    description: str = Field(description="One line of plain text describing the column.")


class ContractArgs(BaseModel):
    column_descriptions: list[ColumnDescription] | None = Field(
        default=None, description="Optional one-line plain-text descriptions, one entry per column. Free text "
                                  "only: every structural field is built by the tool.")


class InsightsArgs(BaseModel):
    markdown: str = Field(description="The complete insights.md document in Markdown.")


class Crew1Tool(BaseTool):
    """A deterministic tool bound to one workspace. Errors are returned to the agent, never raised."""

    _fn: Callable[..., Any] = PrivateAttr()
    _recorder: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    def bind(self, fn, recorder, lock=None) -> "Crew1Tool":
        self._fn, self._recorder = fn, recorder
        if lock is not None:
            self._lock = lock
        return self

    def _run(self, **kwargs) -> str:
        # CrewAI runs the tool calls of one model turn in parallel threads; the shared run lock makes them
        # execute one at a time (stats.json is read, merged and rewritten; p3-live-002 lost a write).
        with self._lock:
            start = time.monotonic()
            try:
                out = self._fn(**kwargs)
                text = out if isinstance(out, str) else compact(out)
                ok = True
            except IsolationError as e:
                text, ok = f"ERROR: rejected ({e.code}). Use a logical name from the tool description.", False
            except (K.ContractError, KeyError, LookupError, ValueError) as e:
                text, ok = f"ERROR: {type(e).__name__}: {e}", False
            if len(text) > OUTPUT_LIMIT:
                text = text[:OUTPUT_LIMIT] + " ...[truncated]"
            if self._recorder is not None:
                self._recorder.add("tool_call", tool=self.name, args=kwargs, ok=ok, output=text,
                                   seconds=round(time.monotonic() - start, 3))
            return text


class ToolContext:
    """State shared by the tools of one run: the workspace and the (never agent-visible) contract sources."""

    def __init__(self, ws: Workspace, sources: Sources):
        self.ws, self.sources = ws, sources
        self.analyses_requested: list[str] = []
        self.charts_rendered: list[str] = []
        self._raw = None
        self.lock = threading.Lock()                              # shared by every tool of this run

    def raw(self):
        if self._raw is None:
            self._raw = C.load_raw(self.ws)
        return self._raw

    def frozen(self) -> dict:
        return json.loads((self.sources.contracts_dir / "frozen_split.json").read_text(encoding="utf-8"))

    def frozen_counts(self) -> dict:
        return self.frozen()["counts"]

    def log(self) -> dict:
        if not self.ws.has_output("out.cleaning_log"):
            raise LookupError("cleaning_log.json does not exist yet: run apply_policy_c first")
        return json.loads(self.ws.read_output("out.cleaning_log"))

    def contract(self) -> dict | None:
        if not self.ws.has_output("out.dataset_contract"):
            return None
        return json.loads(self.ws.read_output("out.dataset_contract"))

    def narrative_facts(self) -> dict:
        log = self.log() if self.ws.has_output("out.cleaning_log") else None
        return N.narrative_facts(log, self.contract())

    # ------------------------------------------------------------ A1
    def load_raw_table(self, name: str):
        if not (isinstance(name, str) and name.strip().startswith("raw.") and name.strip() != "raw.manifest"):
            from .workspace import classify_name
            code = classify_name(name, {f"raw.{t}" for t in RAW_TABLES})
            raise IsolationError(code or "not_allowlisted")
        return C.profile_table(self.ws, name.strip())

    def detect_defects(self):
        return C.detect_defects(self.raw())

    def apply_policy_c(self):
        sha = C.sha256_bytes(self.ws.read_bytes("raw.manifest"))
        text, log = C.build_clean_data(self.raw(), self.frozen(), sha)
        self.ws.write("out.clean_data", text)
        self.ws.write("out.cleaning_log", C.dumps(log))
        return {"written": ["clean_data.csv", "cleaning_log.json"], "rows_delivered": log["snapshots"]["eligible"],
                "snapshots": log["snapshots"], "exclusion_counts": log["exclusion_counts"],
                "columns": len(text.split("\n", 1)[0].split(",")),
                "clean_data_sha256": C.sha256_bytes(text.encode("utf-8")),
                "records_with_problems": len(log["issues"]),
                "invalid_payments_excluded": len(log["excluded_payments"]),
                "target_unreliable_productions": len(log["target_unreliable_productions"])}

    def eligibility_summary(self):
        return C.eligibility_summary(self.log())

    # ------------------------------------------------------------ A2
    def list_analyses(self):
        return {"analyses": E.ANALYSES, "charts": {c: {"needs": n, "title": t} for c, (n, t) in E.CHARTS.items()},
                "report_sections": {s: t for s, (t, _, _) in E.SECTIONS.items()},
                "minimum_charts": E.MIN_CHARTS,
                "units": "rates are fractions (write them as %); differences between rates are percentage points "
                         "(pp); amounts are ILS pre-VAT",
                "narrative_facts": self.narrative_facts()}

    def run_analysis(self, analysis_id: str):
        if analysis_id not in E.ANALYSES:
            raise KeyError(f"unknown analysis id; choose one of {list(E.ANALYSES)}")
        result = E.run_analysis(self.ws, analysis_id)
        self.analyses_requested.append(analysis_id)
        return {"analysis_id": analysis_id, "result": result}

    def render_chart(self, chart_id: str):
        if chart_id not in E.CHARTS:
            raise KeyError(f"unknown chart id; choose one of {list(E.CHARTS)}")
        try:
            data = E.render_chart(self.ws, chart_id)
        except LookupError:
            raise LookupError(f"run_analysis('{E.CHARTS[chart_id][0]}') first") from None
        self.charts_rendered.append(chart_id)
        return {"chart_id": chart_id, "rendered": True, "png_bytes": len(data), "title": E.CHARTS[chart_id][1]}

    def assemble_eda_report(self, **sections: str):
        from ..validators import eda_narrative, lint_narrative
        page = E.assemble_report(self.ws, sections)
        narrative, images = eda_narrative(page)
        stats = E.load_stats(self.ws)
        text = "\n\n".join(narrative.values())
        return {"written": "eda_report.html",
                "sections_missing_or_short": [s for s in E.SECTIONS if len(narrative.get(s, "")) < 40],
                "charts_embedded": len(images), "minimum_charts": E.MIN_CHARTS,
                **lint_narrative(text, stats, self.contract(), self.frozen_counts()),
                "context_problems": N.context_problems(text, stats, self.contract(), self.frozen_counts(),
                                                       self.log())}

    # ------------------------------------------------------------ A3
    def build_dataset_contract(self, column_descriptions: list | None = None):
        if not self.ws.has_output("out.clean_data"):
            raise LookupError("clean_data.csv does not exist yet")
        sha = C.sha256_bytes(self.ws.read_bytes("raw.manifest"))
        log = self.log()
        contract = K.build_contract(self.ws.read_output("out.clean_data"), self.sources.contracts_dir, sha,
                                    log["exclusion_counts"], K.descriptions_from_entries(column_descriptions))
        K.write_contract(self.ws, contract)
        return {"written": "dataset_contract.json", "row_count": contract["row_count"],
                "exclusion_counts": contract["exclusion_counts"], "columns": len(contract["columns"]),
                "described_columns": sum(1 for c in contract["columns"] if "description" in c),
                "raw_manifest_sha256": contract["raw_manifest_sha256"],
                "narrative_facts": self.narrative_facts()}

    def run_handoff_gate(self):
        result = run_gate(self.ws)
        checks = [{"id": c["id"], "severity": c["severity"], "status": c["status"],
                   **({"details": c["details"][:400]} if c["severity"] == "HARD" and c["status"] != "PASS" else {})}
                  for c in result["checks"]]
        return {"status": result["status"], "written": "gate_result.json", "checks": checks}

    def write_insights(self, markdown: str):
        from ..validators import INSIGHT_SECTIONS, _norm_heading, frozen_note_problems, lint_narrative, \
            markdown_sections
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("markdown must be non-empty text")
        self.ws.write("out.insights", markdown)
        secs = markdown_sections(markdown)
        stats = E.load_stats(self.ws)
        return {"written": "insights.md",
                "required_headings": [f"## {t}" for t in INSIGHT_SECTIONS],
                "sections_missing_or_empty": [t for t in INSIGHT_SECTIONS if not secs.get(_norm_heading(t))],
                "frozen_split_note_problems": frozen_note_problems(secs.get(_norm_heading("Frozen split note"), ""),
                                                                   stats),
                "frozen_split_values": N.frozen_split_values(stats),
                **lint_narrative(markdown, stats, self.contract(), self.frozen_counts()),
                "context_problems": N.context_problems(markdown, stats, self.contract(), self.frozen_counts(),
                                                       self.log())}


SPECS = {
    "load_raw_table": (TableArgs, "Load one raw table by logical name and return its schema and profile summary "
                                  "(rows, per-column empty and distinct counts, raw coded values)."),
    "detect_defects": (NoArgs, "Detect data-quality defects in all raw tables with the documented detection rules; "
                               "returns per-type counts of problems, normalisations and duplicates."),
    "apply_policy_c": (NoArgs, "Apply Policy C deterministically: writes clean_data.csv (the handoff) and "
                               "cleaning_log.json, and returns the counts. No values are imputed."),
    "eligibility_summary": (NoArgs, "Summarise Policy C snapshot eligibility: eligible and ineligible snapshots, "
                                    "by stage and by exclusion reason."),
    "list_analyses": (NoArgs, "List the fixed menu of approved analyses, the chart ids and the report section ids, "
                              "and return the narrative facts (target definitions, eligible production counts)."),
    "run_analysis": (AnalysisArgs, "Run one approved analysis on clean_data.csv; results are added to stats.json "
                                   "and returned."),
    "render_chart": (ChartArgs, "Render one chart (PNG) from stats.json only. Its analysis must have been run."),
    "assemble_eda_report": (ReportArgs, "Write eda_report.html: your interpretation text in each of the nine "
                                        "section fields, with the computed tables and rendered charts inserted by "
                                        "the tool. Returns checks of your text (unverified numbers, forbidden "
                                        "claims, missing sections, statements the tool outputs do not support)."),
    "build_dataset_contract": (ContractArgs, "Build dataset_contract.json deterministically from the actual "
                                             "clean_data.csv and the committed contracts. You may add only "
                                             "one-line column descriptions."),
    "run_handoff_gate": (NoArgs, "Run the P2 handoff quality gate (read-only) on clean_data.csv and "
                                 "dataset_contract.json; writes gate_result.json and returns the check statuses."),
    "write_insights": (InsightsArgs, "Write insights.md. Returns the required headings, the frozen split values "
                                     "from stats.json, and checks of your text (missing sections, unverified "
                                     "numbers, forbidden claims, frozen split note requirements, statements the "
                                     "tool outputs do not support)."),
}
assert set(SPECS) == {t for ts in TOOLSETS.values() for t in ts}


def build_tools(ctx: ToolContext, recorder=None) -> dict[str, list[BaseTool]]:
    """The tool list of each agent (by agent key)."""
    out = {}
    for agent, names in TOOLSETS.items():
        tools = []
        for name in names:
            schema, description = SPECS[name]
            tool = Crew1Tool(name=name, description=description, args_schema=schema)
            tools.append(tool.bind(getattr(ctx, name), recorder, ctx.lock))
        out[agent] = tools
    return out
