"""Read-only access to verified Flow runs for the demo app (P5-F). Standard library only. ALL DATA IS SYNTHETIC.

The app imports nothing from the ML code (no crew, no scikit-learn, no joblib): it reads JSON, Markdown and
CSV headers, recomputes sha256 digests and never unpickles anything. ``model.joblib`` is shown as metadata only.

Two run sources:

* ``flow_evidence/`` (committed): the canonical runs exported by ``python -m app.export_evidence``, each with an
  ``evidence.json`` listing every file and its sha256;
* ``out/flow_runs/`` (local, ignored by git): runs started from the app or the command line.

A run is VERIFIED only if every file listed in its ``flow_manifest.json`` (and, for a canonical run, in its
``evidence.json``) exists and has the recorded sha256. Every file access goes through ``RunView.path``, which
accepts only allow-listed relative names inside the run folder and refuses links.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO / "flow_evidence"
LOCAL_RUNS_DIR = REPO / "out" / "flow_runs"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}")
CREW1_REQUIRED = ("crew1/clean_data.csv", "crew1/eda_report.html", "crew1/insights.md", "crew1/dataset_contract.json")
CREW2_REQUIRED = ("crew2/features.csv", "crew2/model.joblib", "crew2/evaluation_report.md", "crew2/model_card.md")
REQUIRED = CREW1_REQUIRED + CREW2_REQUIRED
VIEWABLE = {
    "crew2/evaluation_report.md": ("markdown", "Evaluation report (Crew 2)"),
    "crew2/model_card.md": ("markdown", "Model card with ethical considerations (Crew 2)"),
    "crew1/insights.md": ("markdown", "Insights (Crew 1)"),
    "crew1/eda_report.html": ("html", "EDA report (Crew 1)"),
    "crew1/dataset_contract.json": ("json", "Dataset contract (Crew 1)"),
    "handoff/gate_report.json": ("json", "Contract gate report (Flow)"),
    "crew2/features.schema.json": ("json", "features.csv schema (Crew 2)"),
    "crew2/features.csv": ("csv_summary", "features.csv: header and counts only"),
    "crew1/clean_data.csv": ("csv_summary", "clean_data.csv: header and counts only"),
    "crew2/model.joblib": ("binary_metadata", "model.joblib: metadata only (never loaded)"),
}
LABEL_COLUMNS = ("final_cost_pre_vat", "budget_gap_ils", "is_overrun", "is_material_overrun")


class EvidenceError(Exception):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_link(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    return os.path.islink(p) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def redact(text: str) -> str:
    """Remove the repository location, the user's home folder and any other absolute Windows/Unix path."""
    for root in sorted({str(REPO), str(REPO).replace("\\", "/"), str(Path.home()), str(Path.home()).replace("\\", "/")},
                       key=len, reverse=True):
        text = re.sub(re.escape(root), "<path>", text, flags=re.I)
    text = re.sub(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s\"'<>|]*", "<path>", text)
    return re.sub(r"(?<![\w.])/(?:home|Users|tmp|var|root|mnt)/[^\s\"'<>|]*", "<path>", text)


def sandboxed_html(page: str, height: int = 880) -> str:
    """An HTML report wrapped in an iframe with an empty ``sandbox``: no script, form, popup or same-origin access.
    (Streamlit's own component frame allows scripts and same-origin, so it is not a sandbox by itself.)"""
    return (f'<iframe sandbox="" referrerpolicy="no-referrer" title="report" srcdoc="{html.escape(page, quote=True)}" '
            f'style="width:100%;height:{int(height)}px;border:0"></iframe>')


@dataclass
class RunView:
    source: str                       # "canonical" or "local"
    run_id: str
    root: Path
    manifest: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)

    # ------------------------------------------------------------ safe file access
    def path(self, rel: str) -> Path:
        if not isinstance(rel, str) or rel.startswith(("/", "\\")) or ".." in rel or ":" in rel or "\\" in rel:
            raise EvidenceError("not an allowed relative name")
        parts = rel.split("/")
        cur = self.root
        for part in parts:
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", part):
                raise EvidenceError("not an allowed relative name")
            cur = cur / part
            if _is_link(cur):
                raise EvidenceError("links are refused")
        real = Path(os.path.realpath(cur))
        if Path(os.path.realpath(self.root)) not in real.parents:
            raise EvidenceError("outside the run folder")
        return cur

    def exists(self, rel: str) -> bool:
        try:
            return self.path(rel).is_file()
        except EvidenceError:
            return False

    def read_text(self, rel: str, limit: int = 2_000_000) -> str:
        p = self.path(rel)
        if p.stat().st_size > limit:
            raise EvidenceError("file too large to display")
        return p.read_text(encoding="utf-8")

    def read_json(self, rel: str) -> dict:
        return json.loads(self.read_text(rel))

    def size(self, rel: str) -> int:
        return self.path(rel).stat().st_size

    # ------------------------------------------------------------ derived views
    @property
    def status(self) -> str:
        return self.manifest.get("status", "UNKNOWN")

    @property
    def verified(self) -> bool:
        return bool(self.verification.get("verified"))

    def csv_summary(self, rel: str) -> dict:
        """Header, row count and split counts of a CSV: never its rows."""
        text = self.read_text(rel, limit=20_000_000)
        reader = csv.DictReader(io.StringIO(text))
        splits: dict[str, int] = {}
        n = 0
        for r in reader:
            n += 1
            if "split" in r:
                splits[r["split"]] = splits.get(r["split"], 0) + 1
        header = list(reader.fieldnames or [])
        return {"columns": header, "n_columns": len(header), "rows": n, "rows_by_split": splits,
                "outcome_columns_present": [c for c in header if c in LABEL_COLUMNS]}

    def cv_results(self) -> list[dict]:
        """The current run's grouped-CV summaries, from the Python tool results recorded in Crew 2's trace."""
        if not self.exists("crew2/trace.jsonl"):
            return []
        out = []
        for line in self.read_text("crew2/trace.jsonl", limit=20_000_000).splitlines():
            e = json.loads(line)
            if e.get("event") == "tool_call" and e.get("tool") == "run_grouped_cv" and e.get("ok"):
                r = json.loads(e["output"])
                out.append({"run_id": r["run_id"], "role": r["role"], **{
                    f"{m}_{k}": r["summary"][m][k] for m in ("brier", "average_precision", "roc_auc")
                    for k in ("mean", "sd")}, "params_per_fold": r["params_per_fold"]})
        return out


def _verify(run: RunView, evidence_index: dict | None) -> dict:
    problems, checked = [], 0
    expected = dict(run.manifest.get("required_outputs_sha256") or {})
    if evidence_index:
        # A stopped canonical run omits Crew 1's outputs on purpose (identical bytes; listed with their sha256).
        omitted = evidence_index.get("omitted_sha256", {})
        if any(expected.get(rel) != sha for rel, sha in omitted.items()):
            problems.append("an omitted output does not match the manifest")
        expected = {rel: sha for rel, sha in expected.items() if rel not in omitted}
        expected.update(evidence_index.get("files_sha256", {}))
    c2 = run.root / "crew2" / "run_manifest.json"
    if c2.is_file() and not _is_link(c2):
        arts = json.loads(c2.read_text(encoding="utf-8")).get("artifacts", {})
        expected.update({f"crew2/{k}": v["sha256"] for k, v in arts.items()})
    for rel, sha in sorted(expected.items()):
        checked += 1
        try:
            if sha256_file(run.path(rel)) != sha:
                problems.append(f"{rel}: sha256 differs")
        except (EvidenceError, OSError):
            problems.append(f"{rel}: missing or not readable")
    if run.status == "PASS" and any(r not in expected for r in REQUIRED):
        problems.append("a PASS run must list all 8 required outputs")
    return {"verified": not problems and checked > 0, "files_checked": checked, "problems": problems[:20]}


def load_run(source: str, run_id: str) -> RunView:
    if not RUN_ID_RE.fullmatch(run_id or ""):
        raise EvidenceError("invalid run id")
    base = EVIDENCE_DIR / "runs" if source == "canonical" else LOCAL_RUNS_DIR
    root = base / run_id
    if _is_link(root) or not root.is_dir():
        raise EvidenceError("unknown run")
    run = RunView(source, run_id, root)
    try:
        run.manifest = run.read_json("flow_manifest.json")
    except (EvidenceError, OSError, ValueError):
        run.verification = {"verified": False, "files_checked": 0, "problems": ["no readable flow_manifest.json"]}
        return run
    index = run.read_json("evidence.json") if source == "canonical" and run.exists("evidence.json") else None
    if source == "canonical" and index is None:
        run.verification = {"verified": False, "files_checked": 0, "problems": ["no evidence.json"]}
        return run
    run.verification = _verify(run, index)
    return run


def list_runs(source: str) -> list[str]:
    base = EVIDENCE_DIR / "runs" if source == "canonical" else LOCAL_RUNS_DIR
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and not _is_link(p) and RUN_ID_RE.fullmatch(p.name)
                  and (p / "flow_manifest.json").is_file())


def historical_test(run: RunView) -> dict | None:
    """The pinned approved P4 TEST metrics, cited (never recomputed): read from the committed reference, and shown
    only when its sha256 equals the one the run recorded as its input."""
    ref = REPO / "budget_flow" / "contracts" / "p4_reference.v1.json"
    recorded = (run.manifest.get("crew2", {}).get("input_sha256") or {}).get("budget_flow/contracts/p4_reference.v1.json")
    if not ref.is_file() or not recorded or sha256_file(ref) != recorded:
        return None
    return json.loads(ref.read_text(encoding="utf-8"))["historical_test"]
