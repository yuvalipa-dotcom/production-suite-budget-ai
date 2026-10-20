"""Post-run validators for Crew 2 (docs/crew2.md). A run is PASS only if every one passes.

| ID  | Check |
|-----|-------|
| W1  | every artifact exists and is non-empty |
| W2  | the handoff sha256 is the expected deterministic output, and the P2 handoff gate is PASS on it |
| W3  | a fresh deterministic recompute of the whole protocol equals metrics.json |
| W4  | the selection equals the pre-registered rule applied to metrics.json (C3: no agent choice) |
| W5  | test was opened exactly once, after both models were frozen; demo has no metric and no label |
| W6  | demo predictions equal a recomputation from model.joblib |
| W7  | every number in the model-card narrative matches metrics.json or a pre-registered constant |
| W8  | the card has every section, the disclosures and the copyright notice, the charts, no forbidden claim |
| W9  | the manifest is complete; the trace holds no key, secret shape or absolute path |
| W10 | the leakage report and every split/fit audit are clean |
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from crew1.validators import trace_problems

from . import claims as CL
from .ml import data as D
from .ml.protocol import RUNS, Session
from .ml.selection import apply_rule
from .tools import card as K
from .tools.workspace import CHART_IDS, CONTRACTS_DIR, RAW_DIR

ARTIFACTS = ("metrics.json", "model.joblib", "model_card.md", "demo_predictions.csv", "cv_predictions.csv",
             "leakage_report.json", "feature_availability.json", *(f"charts/{c}.png" for c in CHART_IDS))
MANIFEST_FIELDS = ("run_id", "status", "mode", "model", "llm_settings", "limits", "package_versions",
                   "input_sha256", "token_usage", "timings", "artifacts_sha256")
DEMO_COLUMNS = ["production_id", "stage", "p_overrun", "label"]


def _res(vid: str, name: str, problems: list) -> dict:
    return {"id": vid, "name": name, "status": "PASS" if not problems else "FAIL", "problems": problems[:20]}


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def gate_on(text: str, log: dict) -> dict:
    """Run the unchanged P2 handoff gate on a handoff text (contract built by the Crew 1 contract code)."""
    from crew1.tools.contract import build_contract
    from gate.run import main as gate_main

    raw, sha = D.load_raw(RAW_DIR)
    with tempfile.TemporaryDirectory(prefix="crew2_gate_") as tmp:
        t = Path(tmp)
        (t / "clean_data.csv").write_text(text, encoding="utf-8", newline="")
        contract = build_contract(text.encode("utf-8"), CONTRACTS_DIR, sha, log["exclusion_counts"], None)
        (t / "dataset_contract.json").write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
        code = gate_main(["--stage", "handoff", "--raw", str(RAW_DIR), "--out", str(t / "gate"),
                          "--data", str(t / "clean_data.csv"), "--contract", str(t / "dataset_contract.json")])
        result = json.loads((t / "gate" / "gate_result.json").read_text(encoding="utf-8"))
    return {"exit_code": code, "status": result["status"],
            "failed": [c["id"] for c in result["checks"] if c["severity"] == "HARD" and c["status"] != "PASS"]}


def w1(out: Path) -> dict:
    return _res("W1", "artifacts", [f"missing or empty: {a}" for a in ARTIFACTS
                                    if not (out / a).is_file() or (out / a).stat().st_size == 0])


def w2(m: dict | None) -> dict:
    problems = []
    if not m or not m.get("data"):
        return _res("W2", "handoff", ["metrics.json has no data section"])
    if m["data"]["handoff"]["clean_data_sha256"] != D.EXPECTED_CLEAN_SHA256:
        problems.append("handoff sha256 differs from the expected deterministic output")
    text, log = D.build_handoff()
    try:
        D.verify_handoff(text, log)
    except D.HandoffMismatch as e:
        problems.append(f"fresh handoff: {e}")
    g = gate_on(text, log)
    if g["status"] != "PASS":
        problems.append(f"handoff gate {g['status']}: {g['failed']}")
    return _res("W2", "handoff", problems)


def w3(m: dict | None, fresh: dict | None) -> dict:
    if not m or fresh is None:
        return _res("W3", "deterministic recompute", ["metrics.json or the recompute is missing"])
    problems = [] if canonical(m) == canonical(fresh | {"complete": True}) else \
        [f"section differs: {k}" for k in sorted(set(m) | set(fresh))
         if k != "complete" and canonical(m.get(k)) != canonical(fresh.get(k))] or ["metrics.json differs"]
    return _res("W3", "deterministic recompute", problems)


def w4(m: dict | None) -> dict:
    if not m or not m.get("selection"):
        return _res("W4", "selection rule", ["no selection in metrics.json"])
    prim = {r["candidate"]: {"brier": r["summary"]["brier"]["mean"],
                             "average_precision": r["summary"]["average_precision"]["mean"]}
            for r in m["cv"].values() if r["role"] == "primary"}
    expected = apply_rule(prim)
    problems = []
    if expected["selected"] != m["selection"]["selected"]:
        problems.append(f"selected {m['selection']['selected']} but the rule gives {expected['selected']}")
    if any(v["candidate"] != expected["selected"] for v in m["final"].values()):
        problems.append("a final model is not the rule's candidate")
    return _res("W4", "selection rule", problems)


def w5(out: Path, m: dict | None, trace_text: str) -> dict:
    problems = []
    if not m or not m.get("test"):
        return _res("W5", "single test evaluation", ["no test section"])
    if m["test"]["opened"] != 1:
        problems.append(f"test opened {m['test']['opened']} times")
    for k in ("primary", "secondary"):
        if m["test"][k]["model_sha256"] != m["final"][k]["model_sha256"]:
            problems.append(f"{k}: evaluated model differs from the frozen model")
    events = [json.loads(line) for line in trace_text.splitlines() if line.strip()]
    calls = [e for e in events if e.get("event") == "tool_call"]
    ok_test = [i for i, e in enumerate(calls) if e["tool"] == "evaluate_on_test_once" and e["ok"]]
    fit = [i for i, e in enumerate(calls) if e["tool"] == "fit_final_models" and e["ok"]]
    if len(ok_test) != 1:
        problems.append(f"{len(ok_test)} successful test evaluations in the trace")
    elif not fit or fit[0] > ok_test[0]:
        problems.append("test was evaluated before the final models were frozen")
    if (m.get("demo") or {}).get("metrics") is not None:
        problems.append("demo has metrics")
    try:
        demo = pd.read_csv(out / "demo_predictions.csv")
        if list(demo.columns) != DEMO_COLUMNS:
            problems.append(f"demo columns {list(demo.columns)}")
    except (OSError, ValueError):
        problems.append("demo_predictions.csv unreadable")
    return _res("W5", "single test evaluation", problems)


def w6(out: Path) -> dict:
    problems = []
    try:
        bundle = joblib.load(out / "model.joblib")
        demo = pd.read_csv(out / "demo_predictions.csv", dtype={"p_overrun": str})
    except Exception as e:                                    # noqa: BLE001  (any unreadable artifact fails)
        return _res("W6", "demo predictions", [f"unreadable: {type(e).__name__}"])
    text, _ = D.build_handoff()
    feats = D.partition(text).demo_features
    p = bundle["models"]["primary"].predict(feats)
    expected = pd.DataFrame({"production_id": feats["production_id"], "stage": feats["stage"],
                             "p_overrun": [f"{x:.6f}" for x in p]}).sort_values(["production_id", "stage"])
    got = demo[["production_id", "stage", "p_overrun"]].sort_values(["production_id", "stage"])
    if expected.reset_index(drop=True).to_dict("records") != got.reset_index(drop=True).to_dict("records"):
        problems.append("demo predictions differ from model.joblib")
    if set(demo["label"]) != {K.DEMO_LABEL}:
        problems.append("demo rows are not all labelled as experimental / demo predictions")
    return _res("W6", "demo predictions", problems)


def w7(card: str | None, m: dict | None) -> dict:
    if card is None:
        return _res("W7", "card numbers", ["model_card.md missing"])
    text = "\n\n".join(K.narrative(card).values())
    return _res("W7", "card numbers", [f"unverified: {c}" for c in CL.unverified(text, m)])


def w8(out: Path, card: str | None) -> dict:
    if card is None:
        return _res("W8", "card content", ["model_card.md missing"])
    problems = [f"missing phrase: {p}" for p in K.REQUIRED_PHRASES if p not in card]
    nar = K.narrative(card)
    problems += [f"section missing or short: {s}" for s in K.SECTIONS if len(nar.get(s, "")) < K.MIN_NARRATIVE]
    problems += [f"heading missing: {t}" for t in K.SECTIONS.values() if f"## {t}" not in card]
    problems += [f"forbidden: {f}" for f in CL.forbidden_claims("\n\n".join(nar.values()))]
    problems += [f"chart missing: {c}" for c in CHART_IDS if not (out / "charts" / f"{c}.png").is_file()]
    return _res("W8", "card content", problems)


def w9(manifest: dict | None, trace_text: str, secret: str | None) -> dict:
    problems = [] if manifest else ["manifest missing"]
    if manifest:
        problems += [f"manifest field missing: {f}" for f in MANIFEST_FIELDS if manifest.get(f) in (None, "", {})
                     and f != "token_usage"]
        problems += [] if isinstance(manifest.get("token_usage"), dict) else ["token usage missing"]
    problems += trace_problems(trace_text, secret)
    return _res("W9", "manifest and trace", problems)


def w10(out: Path, m: dict | None) -> dict:
    rep = _json(out / "leakage_report.json")
    problems = [] if rep and rep.get("ok") else ["leakage report missing or not ok"]
    if rep and rep.get("test_vault_opened"):
        problems.append("test vault was open during the leakage checks")
    if m:
        for run_id, r in (m.get("cv") or {}).items():
            if r["audit"]["violations"] or not r["audit"]["splits_checked"] and r["candidate"] != "B0":
                problems.append(f"{run_id}: split audit")
        for k, v in (m.get("final") or {}).items():
            if v["audit"]["violations"]:
                problems.append(f"final {k}: split audit")
        if set(m.get("cv") or {}) != set(RUNS):
            problems.append("not every pre-registered CV run is present")
    return _res("W10", "leakage and audits", problems)


def fresh_metrics(n_jobs: int = 1) -> dict:
    return Session(n_jobs=n_jobs).run_all()


def run_validators(out: Path, manifest: dict | None, secret: str | None, recompute=None) -> list[dict]:
    m = _json(out / "metrics.json")
    card = (out / "model_card.md").read_text(encoding="utf-8") if (out / "model_card.md").is_file() else None
    trace_text = (out / "trace.jsonl").read_text(encoding="utf-8") if (out / "trace.jsonl").is_file() else ""
    try:
        fresh = (recompute or fresh_metrics)()
    except Exception as e:                                    # noqa: BLE001  (a failed recompute fails W3)
        fresh = None
        _ = e
    return [w1(out), w2(m), w3(m, fresh), w4(m), w5(out, m, trace_text), w6(out), w7(card, m), w8(out, card),
            w9(manifest, trace_text, secret), w10(out, m)]

