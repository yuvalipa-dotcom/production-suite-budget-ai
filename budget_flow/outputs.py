"""The Crew 2 artifacts of the Flow's TRAIN-only step (P5-C). ALL DATA IS SYNTHETIC.

* ``features.csv``: one row per handoff snapshot (train, test and demo): identifiers, split and fold metadata,
  and the 27 approved features (F27). No outcome column of any row; schema in ``features.schema.json``.
* ``model.joblib``: the two final models refit on TRAIN only (primary ``is_overrun``, secondary
  ``is_material_overrun``), their sha256 and their provenance. No data rows.
* ``evaluation_report.md``: A. the current TRAIN-only grouped CV, selection and provenance; B. the historical
  P4 TEST result, pinned and cited, with the statement that TEST was not reopened.
* ``model_card.md``: purpose, training data, evaluation, limitations, synthetic-data disclosure and ethical
  considerations.

``run_manifest.json`` records the environment, seeds, versions, input and artifact hashes, timings and the
reproducibility comparison. Only the manifest holds timings, so the four artifacts are deterministic.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import time
from pathlib import Path

import joblib

from crew2.ml import protocol as P
from crew2.ml.features import F27, PRIMARY_FEATURE_SET, ROW
from crew2.ml.models import GRIDS, K_INNER, SEED
from crew2.ml.selection import BRIER_TIE
from gate import handoff_spec as H

from . import contract as C
from . import p4_reference as R

ARTIFACTS = ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md")
ID_COLUMNS = ("handoff_row", "production_id", "split", "cv_fold")
FEATURE_COLUMNS = ID_COLUMNS + tuple(F27)
LABEL = "experimental / demo prediction"
BANNER = ("> **ALL DATA IS SYNTHETIC.** Every output is an experimental / demo prediction, produced by the Flow's "
          "TRAIN-only Crew 2 step. It is not evidence about real productions.")


def _f(x, d: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _ms(s: dict) -> str:
    return f"{_f(s['mean'])} ± {_f(s['sd'])}"


def _cell(v) -> str:
    if isinstance(v, float):
        return repr(v)
    return str(v)


# ---------------------------------------------------------------- features.csv

def features_rows(parts) -> list[list[str]]:
    frames = [parts.train, parts.test_features, parts.demo_features]
    rows = []
    for df in frames:
        for rec in df[[ROW, "production_id", "split", "cv_fold", *F27]].itertuples(index=False, name=None):
            rows.append(rec)
    rows.sort(key=lambda r: int(r[0]))
    return [[_cell(int(r[0])), *(_cell(v.item() if hasattr(v, "item") else v) for v in r[1:])] for r in rows]


def render_features(parts) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FEATURE_COLUMNS)
    w.writerows(features_rows(parts))
    return buf.getvalue().encode("utf-8")


def features_schema() -> dict:
    cols = {"handoff_row": {"type": "integer", "role": "identifier",
                            "description": "0-based row position in the delivered clean_data.csv"},
            "production_id": {"type": "string", "role": "identifier", "description": "synthetic production id"},
            "split": {"type": "string", "role": "split metadata", "allowed": sorted(H.ALLOWED["split"]),
                      "description": "frozen split (contracts/frozen_split.json); only train rows are used for fitting"},
            "cv_fold": {"type": "integer", "role": "split metadata", "range": [-1, 4],
                        "description": "frozen production-grouped CV fold; -1 for every non-train row"}}
    for c in F27:
        dtype, _, unit = H.COLUMNS[c]
        cols[c] = {"type": dtype, "role": "feature (F27)", **({"unit": unit} if unit else {}),
                   **({"allowed": sorted(H.ALLOWED[c])} if c in H.ALLOWED else {}),
                   "description": "as-of-snapshot value from Crew 1's validated handoff"}
    return {"file": "features.csv", "synthetic": True, "encoding": "utf-8, LF, comma-separated, header row",
            "key": ["production_id", "stage"], "feature_set": PRIMARY_FEATURE_SET, "n_features": len(F27),
            "excluded": {"outcome columns (never present)": list(H.TARGETS),
                         "research-only features (not in F27)": ["budget_revisions_to_date", "client_direct_share"],
                         "metadata": ["as_of_date"]},
            "columns": cols}


# ---------------------------------------------------------------- model.joblib

def render_model(session, metrics: dict, handoff) -> bytes:
    fin = metrics["final"]
    buf = io.BytesIO()
    joblib.dump({"synthetic": True, "label": LABEL, "feature_set": PRIMARY_FEATURE_SET, "features": list(F27),
                 "models": session.final_models,
                 "model_sha256": {k: v["model_sha256"] for k, v in fin.items()},
                 "provenance": {"step": "Flow TRAIN-only Crew 2 (P5-C)", "trained_on": "train split only",
                                "train_rows": fin["primary"]["train_rows"],
                                "clean_data_sha256": handoff.report.clean_data_sha256, "seed": SEED,
                                "selection": {k: metrics["selection"][k] for k in ("selected", "decided_by")},
                                "params": {k: v["params"] for k, v in fin.items()},
                                "targets": {k: v["target"] for k, v in fin.items()},
                                "reference": R.REFERENCE_VERSION}}, buf)
    return buf.getvalue()


# ---------------------------------------------------------------- evaluation_report.md

def _cv_table(cv: dict, role: str) -> list[str]:
    out = ["| run | candidate | target | Brier | PR-AUC | ROC-AUC | log loss |", "|---|---|---|---|---|---|---|"]
    for k, r in cv.items():
        if r["role"] == role:
            s = r["summary"]
            out.append(f"| `{k}` | {r['candidate']} | `{r['target']}` | {_ms(s['brier'])} | "
                       f"{_ms(s['average_precision'])} | {_ms(s['roc_auc'])} | {_ms(s['log_loss'])} |")
    return out


def _test_row(name: str, t: dict) -> str:
    o, ci = t["overall"], t["bootstrap_ci"]
    return (f"| {name} | `{t['target']}` | {t['candidate']} | {o['n']} | {_f(o['brier'])} "
            f"[{_f(ci['brier']['lo'])}, {_f(ci['brier']['hi'])}] | {_f(o['average_precision'])} "
            f"[{_f(ci['average_precision']['lo'])}, {_f(ci['average_precision']['hi'])}] | {_f(o['roc_auc'])} | "
            f"{_f(o['log_loss'])} | {_f(t['reliability']['ece'], 3)} |")


def render_report(metrics: dict, ref: dict, repro: dict, handoff, input_sha: dict) -> str:
    cv, sel, fin, data = metrics["cv"], metrics["selection"], metrics["final"], metrics["data"]
    c = data["counts"]
    ht = ref["historical_test"]
    dev = repro["max_abs_deviation"]
    same_models = all(repro["final_model_sha256_equal_to_p4"].values())
    tol = ("exact equality (pinned reference environment): every CV summary, fold score and TRAIN prediction is "
           "identical to the approved P4 run, and both final models are byte-identical (sha256)."
           if repro["mode"] == "exact" else
           f"pre-registered tolerances (not the pinned reference environment): CV means <= "
           f"{R.TOLERANCES['cv_mean']:g}, fold scores <= {R.TOLERANCES['fold_score']:g}, predictions <= "
           f"{R.TOLERANCES['prediction']:g}.")
    naive = metrics["naive_rule_reference"]["overall"]
    L = ["# Crew 2 evaluation report: Flow TRAIN-only run", "", BANNER, "",
         "This report has two separate parts. **Part A** is computed by this run, from TRAIN rows only. "
         "**Part B** is the historical P4 TEST result, pinned and cited; this run did not reopen TEST.", "",
         "## A. Current TRAIN-only grouped cross-validation (this run)", "",
         "### A.1 Input and provenance", "",
         "| item | value |", "|---|---|",
         f"| input | Crew 1 file handoff (`clean_data.csv`, `dataset_contract.json`), validated by the Flow "
         f"contract `{C.FLOW_CONTRACT_VERSION}` ({len(handoff.report.violations)} violations) |",
         f"| `clean_data.csv` sha256 | `{handoff.report.clean_data_sha256}` |",
         f"| `dataset_contract.json` sha256 | `{input_sha['dataset_contract.json']}` |",
         f"| rows used for fitting | train: {c['train']['rows']:,} rows, {c['train']['productions']} productions |",
         f"| rows never used for fitting | test: {c['test']['rows']} rows (labels redacted, vault sealed); demo: "
         f"{c['demo']['rows']} rows (labels redacted) |",
         f"| train prevalence of `is_overrun` | {_f(data['train_prevalence']['is_overrun']['overall'])} |",
         f"| protocol | {P.N_FOLDS} frozen production-grouped folds; tuning and Platt calibration on "
         f"{K_INNER} group-disjoint inner splits of each training partition; seed {SEED} |",
         f"| pre-registration | `docs/model_selection_rule.md`, Addendum P4-1 (verified unchanged before fitting) |",
         f"| reference | `{R.REFERENCE_VERSION}` (approved run `{R.SOURCE_RUN}`) |", "",
         "### A.2 Candidate comparison (primary: F27, `is_overrun`)", "", *_cv_table(cv, "primary"), "",
         "Values are mean ± sd over the 5 grouped folds. Grids: "
         + "; ".join(f"{k}: {len(v)} setting(s)" for k, v in GRIDS.items()) + ".", "",
         "### A.3 Selection (pre-registered rule, applied by code)", "",
         f"- Selected: **{sel['selected']}**, decided by: {sel['decided_by']}.",
         f"- Best mean Brier: {_f(sel['best_mean_brier'], 6)}; tie margin {BRIER_TIE}; tie set: "
         f"{', '.join(sel['tie_set'])}.",
         "- Brier minus best: " + ", ".join(f"{k} {_f(v, 6)}" for k, v in sel["brier_minus_best"].items()) + ".", "",
         "### A.4 Final models (refit on all of TRAIN)", "",
         "| model | target | candidate | parameters | train rows | audit violations | sha256 |",
         "|---|---|---|---|---|---|---|",
         *[f"| {k} | `{v['target']}` | {v['candidate']} | {v['params']} | {v['train_rows']:,} | "
           f"{v['audit']['violations']} | `{v['model_sha256']}` |" for k, v in fin.items()], "",
         "### A.5 Reported only: secondary target and research features", "",
         "Never used for selection.", "", *_cv_table(cv, "secondary"), "", *_cv_table(cv, "research"), "",
         "### A.6 Reproducibility against the approved P4 CV", "",
         f"- Mode: **{repro['mode']}**: {tol}",
         f"- Selected model, decision, tie set, per-fold and final parameters: identical to `{R.SOURCE_RUN}`.",
         f"- Maximum absolute deviation: CV means {dev['cv_mean']:.3g}; fold scores {dev['fold_score']:.3g}; "
         f"TRAIN predictions {dev['prediction']:.3g}.",
         f"- Final models byte-identical to the P4 models: "
         + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in repro["final_model_sha256_equal_to_p4"].items()) + ".",
         "", "### A.7 Naive rule reference (TRAIN, not a candidate)", "",
         f"`current_position_ils > 0` on all train rows: Brier {_f(naive['brier'])}, accuracy {_f(naive['accuracy'])}.",
         "", "## B. Historical P4 TEST evaluation (cited, not recomputed)", "",
         f"> **HISTORICAL.** These numbers are pinned from the approved P4 run `{ht['source_run']}` "
         f"(`budget_flow/contracts/{R.REFERENCE_FILE.name}`). TEST was opened once, in P4, after the final models "
         f"were frozen. **This run did not reopen TEST**: its test vault is sealed, was opened 0 times, and no test "
         f"metric was computed.", "",
         ("The final models of this run are byte-identical to the models evaluated in P4, so this historical result "
          "applies to them." if same_models else
          "The final models of this run are not byte-identical to the P4 models (another environment), so this "
          "historical result describes the P4 models only."), "",
         f"Test: {ht['rows']} rows from {ht['productions']} productions; threshold {ht['threshold']} for the "
         f"threshold metrics; 95% production-level bootstrap intervals ({ht['primary']['bootstrap_ci']['reps']} "
         f"resamples).", "",
         "| model | target | candidate | rows | Brier [95% CI] | PR-AUC [95% CI] | ROC-AUC | log loss | ECE |",
         "|---|---|---|---|---|---|---|---|---|",
         _test_row("primary", ht["primary"]), _test_row("secondary", ht["secondary"]), "",
         "Primary model by stage (historical):", "",
         "| stage | rows | Brier | PR-AUC | prevalence |", "|---|---|---|---|---|",
         *[f"| {s} | {v['n']} | {_f(v['brier'])} | {_f(v['average_precision'])} | {_f(v['prevalence'])} |"
           for s, v in ht["primary"]["by_stage"].items()], ""]
    return "\n".join(L)


# ---------------------------------------------------------------- model_card.md

def render_card(metrics: dict, ref: dict, repro: dict) -> str:
    sel, fin, data, cv = metrics["selection"], metrics["final"], metrics["data"], metrics["cv"]
    c = data["counts"]
    p = fin["primary"]
    best = cv[f"{PRIMARY_FEATURE_SET}.is_overrun.{sel['selected']}"]["summary"]
    ht = ref["historical_test"]["primary"]
    tp = data["train_prevalence"]["is_overrun"]["overall"]
    L = ["# Model card: Crew 2 budget-overrun model (Flow, TRAIN-only)", "", BANNER, "",
         "## Purpose and intended use", "",
         "The model estimates the probability that a synthetic event production ends above the budget in effect at "
         "a budget snapshot (`is_overrun`). It supports an academic demonstration of a production budget card. A "
         "secondary model estimates a material overrun (`is_material_overrun`, above 2%); it is reported only.", "",
         "Out of scope: decisions about real productions, people, suppliers or clients; any use on real data. The "
         "model is not validated on real data.", "",
         "## Model", "",
         f"- Candidate selected by the pre-registered rule: **{sel['selected']}** ({sel['decided_by']}), "
         f"parameters {p['params']}.",
         f"- Probabilities are the mean of {K_INNER} members, each a pipeline (standardised numeric features, "
         "one-hot categories) with a Platt calibrator fit on a group-disjoint held-out side.",
         f"- Features: the 27 approved snapshot features (F27), listed in `features.csv` and "
         "`features.schema.json`. Three are only approximations in the product (`lead_time_days`, "
         "`days_since_contract`, `n_substitutions_to_date`).",
         "", "## Training data", "",
         f"- Source: Crew 1's validated file handoff (Policy C), {data['handoff']['rows']:,} eligible snapshots.",
         f"- Fitting uses the train split only: {c['train']['rows']:,} rows from {c['train']['productions']} "
         f"productions; `is_overrun` prevalence {100 * tp:.1f}%.",
         f"- Test ({c['test']['rows']} rows) and demo ({c['demo']['rows']} rows) labels were redacted before Crew 2 "
         "saw the data; they were never used.", "",
         "## Evaluation", "",
         f"- Current run, TRAIN-only grouped CV (5 production-grouped folds): mean Brier {_ms(best['brier'])}, "
         f"PR-AUC {_ms(best['average_precision'])}, ROC-AUC {_ms(best['roc_auc'])}.",
         f"- Reproducibility: {repro['mode']} reproduction of the approved P4 CV (see `evaluation_report.md`, A.6).",
         f"- Historical P4 TEST (cited, not recomputed; TEST was not reopened by this run): Brier "
         f"{_f(ht['overall']['brier'])} [{_f(ht['bootstrap_ci']['brier']['lo'])}, "
         f"{_f(ht['bootstrap_ci']['brier']['hi'])}], PR-AUC {_f(ht['overall']['average_precision'])}, expected "
         f"calibration error {_f(ht['reliability']['ece'], 3)}.", "",
         "## Limitations", "",
         "- All data are synthetic; nothing here is evidence of real-world accuracy.",
         f"- The historical test group has an overrun prevalence of {100 * ht['overall']['prevalence']:.1f}% against "
         f"{100 * tp:.1f}% on train, and the probabilities were not well calibrated there.",
         f"- The selected C = {p['params'].get('C', 'n/a')} is the upper boundary of the pre-registered grid; a "
         "larger value was never evaluated." if p["params"].get("C") == max(g.get("C", 0) for g in GRIDS["LR"])
         else "- The selected parameters are inside the pre-registered grid.",
         "- `is_overrun` is a rounding-tolerance label, not a commercially material overrun.",
         "- Outside the pinned reference environment, results are checked against tolerances, not exactly.", "",
         "## Synthetic-data disclosure", "",
         "Every production, budget, supplier figure and outcome is generated by the project's synthetic data "
         "generator. No real client, person or event is represented.", "",
         "## Ethical considerations", "",
         "- **Decision support only.** A probability is not a verdict about a producer's competence. Any real use "
         "would need human review of every flagged production and a documented appeal path.",
         "- **Automation bias.** A confident score can crowd out the producer's knowledge of the event; the card "
         "should always show the underlying budget position next to the score.",
         "- **Fairness across segments.** Error rates differ by stage and could differ by event archetype. A real "
         "deployment would need per-segment evaluation on real data before use, and should not be used to price "
         "or rank clients or suppliers.",
         "- **Calibration and drift.** The historical test showed miscalibration under a prevalence shift; scores "
         "must not be read as exact probabilities, and a real system would need monitoring and recalibration.",
         "- **Privacy.** The data are synthetic and hold no personal data. A real system would hold commercially "
         "sensitive budgets and would need access control and data minimisation.",
         "- **Transparency.** The selection rule was pre-registered and applied by code; TEST was evaluated once in "
         "P4 and never reopened; the protocol and its limits are disclosed here.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- writing

def _threadpools() -> list[dict]:
    try:
        from threadpoolctl import threadpool_info
        return [{k: i.get(k) for k in ("internal_api", "version", "num_threads")} for i in threadpool_info()]
    except Exception:                                       # informational only
        return []


def render_schema() -> bytes:
    return (json.dumps(features_schema(), indent=2) + "\n").encode("utf-8")


def write_all(out: Path, session, metrics: dict, ref: dict, repro: dict, handoff, inputs: dict, timing: dict) -> dict:
    t0 = time.monotonic()
    input_sha = {name: R.sha256_hex(Path(p).read_bytes()) for name, p in inputs.items()}
    files = {
        "features.csv": render_features(session.parts),
        "features.schema.json": render_schema(),
        "model.joblib": render_model(session, metrics, handoff),
        "evaluation_report.md": render_report(metrics, ref, repro, handoff, input_sha).encode("utf-8"),
        "model_card.md": render_card(metrics, ref, repro).encode("utf-8"),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = make_manifest(session, metrics, ref, repro, handoff, input_sha, files,
                             {**timing, "write_seconds": round(time.monotonic() - t0, 1)})
    manifest["timing"]["total_seconds"] = round(timing["train_only_seconds"] + manifest["timing"]["write_seconds"], 1)
    (out / "run_manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def make_manifest(session, metrics: dict, ref: dict, repro: dict, handoff, input_sha: dict, files: dict,
                  timing: dict) -> dict:
    """Provenance of one TRAIN-only run: environment, seeds, versions, input and artifact hashes, TEST sealing."""
    fin = metrics["final"]
    return {
        "synthetic": True, "step": "Flow TRAIN-only Crew 2 (P5-C)", "status": "PASS",
        "selected_model": {"candidate": metrics["selection"]["selected"],
                           "decided_by": metrics["selection"]["decided_by"], "params": fin["primary"]["params"],
                           "model_sha256": {k: v["model_sha256"] for k, v in fin.items()},
                           "train_rows": fin["primary"]["train_rows"],
                           "audit_violations": {k: v["audit"]["violations"] for k, v in fin.items()}},
        "test_vault": {"class": type(session.parts.vault).__name__, "opened": session.parts.vault.opened,
                       "opens": len(session.parts.vault.opens), "test_metrics_computed": session.test is not None},
        "historical_test": {"source_run": ref["historical_test"]["source_run"], "cited_from": R.REFERENCE_FILE.name,
                            "recomputed": False},
        "reproducibility": repro,
        "contract": {"version": C.FLOW_CONTRACT_VERSION, "ok": handoff.report.ok,
                     "violations": len(handoff.report.violations), "scale_ratios": handoff.report.scale_ratios},
        "environment": {**R.environment(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "threadpools": _threadpools()},
        "seeds": {"seed": SEED, "tuning_splits": "[seed, fold code, 1]", "calibration_splits": "[seed, fold code, 2, j]",
                  "fold_code": "outer fold 0-4; final refit 5", "gradient_boosting_random_state": SEED},
        "input_sha256": {**input_sha,
                         "contracts/frozen_split.json": R.sha256_hex(C.FROZEN_SPLIT_FILE.read_bytes()),
                         f"budget_flow/contracts/{C.REFERENCE_FILE.name}": C.REFERENCE_SHA256,
                         f"budget_flow/contracts/{R.REFERENCE_FILE.name}": R.REFERENCE_SHA256,
                         f"budget_flow/contracts/{R.PREDICTIONS_FILE.name}": R.PREDICTIONS_SHA256,
                         "docs/model_selection_rule.md": R.sha256_hex(R.SELECTION_RULE_DOC.read_bytes()),
                         "crew2/ml/selection.py": R.sha256_hex(R.SELECTION_CODE.read_bytes())},
        "artifacts": {name: {"sha256": R.sha256_hex(data), "bytes": len(data)} for name, data in files.items()},
        "timing": dict(timing),
    }
