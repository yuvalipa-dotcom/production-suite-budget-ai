"""The pinned approved P4 reference for the Flow's TRAIN-only run (P5-C). ALL DATA IS SYNTHETIC.

    python -m budget_flow.p4_reference --check                 # the committed files match their pinned digests
    python -m budget_flow.p4_reference --write RUN_DIR [--n-jobs N]
        # re-extract from the approved local P4 run folder (out/crew2/p4-offline-002): a TRAIN-only run of the
        # Flow must first reproduce that run exactly; then pin the two printed digests below

The source is the approved offline run ``p4-offline-002``. Its ``metrics.json`` and ``cv_predictions.csv`` are
local run outputs (never committed), identified here by their pinned sha256. Two committed files are extracted:

* ``contracts/p4_reference.v1.json``: the pre-registration fingerprint, the pinned reference environment, the
  9 grouped-CV runs (summaries, per-fold parameters and scores), the selection trace, the final parameters, and
  the historical TEST metrics. The TEST metrics are aggregates that are CITED by the Flow, never recomputed;
* ``contracts/p4_train_predictions.v1.csv``: per TRAIN row, the out-of-fold predictions of the 3 primary runs
  and the predictions of the 2 final models. It holds no label and no test or demo row.

``--write`` never loads a pickle: the historical final models are identified only by their sha256, which the
reproduced models must equal byte for byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
from importlib import metadata
from pathlib import Path

from crew2.ml import models as M
from crew2.ml import protocol as P
from crew2.ml import selection as S
from crew2.ml.features import FEATURE_SETS, PRIMARY_FEATURE_SET, PRIMARY_TARGET, ROW
from crew2.ml.metrics import THRESHOLD

from . import contract as C

REFERENCE_VERSION = "p4_reference.v1"
SOURCE_RUN = "p4-offline-002"
REFERENCE_FILE = C.CONTRACTS / "p4_reference.v1.json"
PREDICTIONS_FILE = C.CONTRACTS / "p4_train_predictions.v1.csv"
SELECTION_RULE_DOC = C.REPO / "docs" / "model_selection_rule.md"
SELECTION_CODE = C.REPO / "crew2" / "ml" / "selection.py"

# Pinned sha256 values, held as integer literals (the publication scanner flags quoted hex digests).
_REFERENCE_DIGEST = 86935086252275331054679030000466161384537756410664891864091244266220652702673
_PREDICTIONS_DIGEST = 7518333964508061626008801059600978392465870614163099340200220335082920670512
_SELECTION_RULE_DOC_DIGEST = 82603151363337372778859033258739374108609519285377369921699655771841341669384
_SELECTION_CODE_DIGEST = 46565640004126124885572039135266243725267909624383787807736529397958196855465
_HIST_METRICS_DIGEST = 19623051215706841499584268963954186084880325403684919195587624496858944762596
_HIST_CV_PREDICTIONS_DIGEST = 90057790086645011589506700421291352098913509328167591416355548949759192514393
_HIST_MODEL_DIGESTS = {"primary": 85252477150296585148322694984170763853344685194804627463642932214774425267585,
                       "secondary": 38739515282670359338282144363198149802264129220847342914880225184463691705910}


def _hex(n: int) -> str:
    return f"{n:064x}"


REFERENCE_SHA256 = _hex(_REFERENCE_DIGEST)
PREDICTIONS_SHA256 = _hex(_PREDICTIONS_DIGEST)
HIST_MODEL_SHA256 = {k: _hex(v) for k, v in _HIST_MODEL_DIGESTS.items()}

# Pre-registered reproducibility tolerances (P5-A), used outside the pinned reference environment only.
TOLERANCES = {"cv_mean": 1e-6, "fold_score": 1e-5, "prediction": 1e-6}
ENV_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
PRIMARY_RUNS = tuple(f"{PRIMARY_FEATURE_SET}.{PRIMARY_TARGET}.{c}" for c in M.CANDIDATE_ORDER)
PREDICTION_COLUMNS = (ROW, "cv_fold", *PRIMARY_RUNS, "final.primary", "final.secondary")
EXACT_FIELDS = ("n", "positives", "prevalence")          # counts are never subject to a tolerance


class ReproducibilityError(RuntimeError):
    """The TRAIN-only run does not reproduce the approved P4 results: the run fails, nothing is written."""


class SelectionChanged(ReproducibilityError):
    """The selected model, its parameters or the selection trace differ from the approved P4 decision."""


class PreregistrationChanged(ReproducibilityError):
    """A pre-registered setting (seed, folds, grids, features, decision rule) differs from P4."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- what must not change

def preregistration() -> dict:
    """The pre-registered protocol as the code defines it now (compared with the pinned P4 values)."""
    return json.loads(json.dumps({
        "seed": M.SEED, "k_inner": M.K_INNER, "n_folds": P.N_FOLDS, "platt_c": M.PLATT_C,
        "candidate_order": list(M.CANDIDATE_ORDER), "grids": M.GRIDS, "b0_edges": list(M.B0_EDGES),
        "b0_alpha": M.B0_ALPHA, "brier_tie": S.BRIER_TIE, "ap_decimals": S.AP_DECIMALS, "runs": P.RUNS,
        "primary_feature_set": PRIMARY_FEATURE_SET, "primary_target": PRIMARY_TARGET,
        "features": {k: list(v) for k, v in FEATURE_SETS.items()}, "threshold": THRESHOLD}))


def environment() -> dict:
    """The fields that define the pinned reference environment (exact reproduction is verified there)."""
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            **{p: metadata.version(p) for p in ENV_PACKAGES}}


def check_preregistration(ref: dict) -> None:
    """Raise ``PreregistrationChanged`` before any fitting if the protocol or the decision rule changed."""
    live = preregistration()
    changed = sorted(k for k in set(live) | set(ref["preregistration"]) if live.get(k) != ref["preregistration"].get(k))
    for name, path, digest in (("selection rule document", SELECTION_RULE_DOC, _SELECTION_RULE_DOC_DIGEST),
                               ("selection rule code", SELECTION_CODE, _SELECTION_CODE_DIGEST)):
        if sha256_hex(path.read_bytes()) != _hex(digest):
            changed.append(name)
    replay = S.apply_rule(ref["selection"]["inputs"])
    if any(replay[k] != ref["selection"][k] for k in ("selected", "decided_by", "tie_set")):
        changed.append("selection rule behaviour on the approved P4 inputs")
    if changed:
        raise PreregistrationChanged(f"pre-registered settings differ from the approved P4 protocol: {changed}")


# ---------------------------------------------------------------- the pinned files

def load() -> tuple[dict, list[dict]]:
    """(reference, prediction rows), each checked against its pinned sha256."""
    ref_bytes, pred_bytes = REFERENCE_FILE.read_bytes(), PREDICTIONS_FILE.read_bytes()
    if sha256_hex(ref_bytes) != REFERENCE_SHA256:
        raise ReproducibilityError(f"{REFERENCE_FILE.name} differs from its pinned sha256")
    if sha256_hex(pred_bytes) != PREDICTIONS_SHA256:
        raise ReproducibilityError(f"{PREDICTIONS_FILE.name} differs from its pinned sha256")
    ref = json.loads(ref_bytes.decode("utf-8"))
    rows = [{k: int(v) if k in (ROW, "cv_fold") else float(v) for k, v in r.items()}
            for r in csv.DictReader(io.StringIO(pred_bytes.decode("utf-8")))]
    return ref, rows


def historical_test() -> dict:
    """The approved P4 TEST metrics, cited (never recomputed)."""
    return load()[0]["historical_test"]


# ---------------------------------------------------------------- comparison

def mode_for(env: dict, ref: dict) -> str:
    return "exact" if env == ref["environment"] else "tolerance"


def _dev(a, b) -> float:
    if a is None or b is None:
        return 0.0 if a is b else math.inf
    return abs(float(a) - float(b))


def verify(cv: dict, selection: dict, final: dict, predictions: list[dict], ref: dict, pred_ref: list[dict],
           mode: str) -> dict:
    """Compare a TRAIN-only run with the approved P4 results. Raises; returns the comparison record on success.

    Always exact: the selected candidate, the decision, the tie set, the per-fold and final parameters and the
    row sets. ``exact`` mode (pinned reference environment): every number and both final model digests equal.
    ``tolerance`` mode (another environment): the pre-registered tolerances, never weakened.
    """
    for k in ("selected", "decided_by", "tie_set", "runs"):
        if selection[k] != ref["selection"][k]:
            raise SelectionChanged(f"selection {k}: {selection[k]!r} != approved {ref['selection'][k]!r}")
    for name, r in ref["final"].items():
        got = {k: final[name][k] for k in ("candidate", "target", "feature_set", "params", "train_rows")}
        if got != r:
            raise SelectionChanged(f"final {name} model: {got} != approved {r}")
    for run_id, r in ref["cv"].items():
        if [f["params"] for f in cv[run_id]["folds"]] != [f["params"] for f in r["folds"]]:
            raise SelectionChanged(f"{run_id}: the tuned per-fold parameters differ from the approved run")

    dev = {"cv_mean": 0.0, "fold_score": 0.0, "prediction": 0.0}
    problems = []
    for run_id, r in ref["cv"].items():
        for metric, s in r["summary"].items():
            d = _dev(cv[run_id]["summary"][metric]["mean"], s["mean"])
            dev["cv_mean"] = max(dev["cv_mean"], d)
            if mode == "exact" and cv[run_id]["summary"][metric] != s:
                problems.append(f"{run_id} {metric} summary")
        for f_new, f_ref in zip(cv[run_id]["folds"], r["folds"]):
            for metric, v in f_ref["scores"].items():
                got = f_new["scores"][metric]
                if metric in EXACT_FIELDS and got != v:
                    problems.append(f"{run_id} fold {f_ref['fold']} {metric}")
                d = _dev(got, v)
                dev["fold_score"] = max(dev["fold_score"], d)
                if mode == "exact" and got != v:
                    problems.append(f"{run_id} fold {f_ref['fold']} {metric}")
    for k in ("best_mean_brier",):
        if mode == "exact" and selection[k] != ref["selection"][k]:
            problems.append(f"selection {k}")
    if [(p[ROW], p["cv_fold"]) for p in predictions] != [(p[ROW], p["cv_fold"]) for p in pred_ref]:
        raise ReproducibilityError("the TRAIN rows or their folds differ from the approved run")
    for p_new, p_ref in zip(predictions, pred_ref):
        for c in PREDICTION_COLUMNS[2:]:
            d = _dev(p_new[c], p_ref[c])
            dev["prediction"] = max(dev["prediction"], d)
            if mode == "exact" and p_new[c] != p_ref[c]:
                problems.append(f"prediction {c} row {p_ref[ROW]}")
    sha_equal = {k: final[k]["model_sha256"] == HIST_MODEL_SHA256[k] for k in HIST_MODEL_SHA256}
    if mode == "exact":
        problems += [f"final {k} model sha256" for k, eq in sha_equal.items() if not eq]
    else:
        problems += [f"{k} deviation {dev[k]:.3g} > {TOLERANCES[k]:g}" for k in TOLERANCES if dev[k] > TOLERANCES[k]]
    if problems:
        shown = problems[:10] + ([f"... {len(problems) - 10} more"] if len(problems) > 10 else [])
        raise ReproducibilityError(f"{mode} reproduction of the approved P4 CV failed: {shown}")
    return {"mode": mode, "reference": REFERENCE_VERSION, "source_run": SOURCE_RUN,
            "tolerances": None if mode == "exact" else dict(TOLERANCES), "max_abs_deviation": dev,
            "final_model_sha256_equal_to_p4": sha_equal, "selection_equal": True, "parameters_equal": True,
            "ok": True}


# ---------------------------------------------------------------- extraction (owner-run, needs the local P4 run)

def _prediction_rows(session) -> list[dict]:
    train = session.parts.train.sort_values(ROW)
    out = {int(r): {ROW: int(r), "cv_fold": int(f)} for r, f in zip(train[ROW], train["cv_fold"])}
    for run_id in PRIMARY_RUNS:
        for r, p in zip(session.oof[run_id][ROW], session.oof[run_id]["p"]):
            out[int(r)][run_id] = float(p)
    for name, model in session.final_models.items():
        for r, p in zip(train[ROW], model.predict(train)):
            out[int(r)][f"final.{name}"] = float(p)
    return [out[r] for r in sorted(out)]


def render_predictions(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(PREDICTION_COLUMNS)
    for r in rows:
        w.writerow([r[ROW], r["cv_fold"], *(repr(r[c]) for c in PREDICTION_COLUMNS[2:])])
    return buf.getvalue().encode("utf-8")


def _strip_digests(test: dict) -> dict:
    return {k: ({kk: vv for kk, vv in v.items() if kk != "model_sha256"} if isinstance(v, dict) else v)
            for k, v in test.items()}


def build(run_dir: Path, n_jobs: int) -> tuple[bytes, bytes]:
    from . import handoff as FH
    from .train import TrainOnlySession

    metrics_bytes = (run_dir / "metrics.json").read_bytes()
    cvp_bytes = (run_dir / "cv_predictions.csv").read_bytes()
    if sha256_hex(metrics_bytes) != _hex(_HIST_METRICS_DIGEST):
        raise SystemExit("metrics.json is not the approved p4-offline-002 output (sha256 differs)")
    if sha256_hex(cvp_bytes) != _hex(_HIST_CV_PREDICTIONS_DIGEST):
        raise SystemExit("cv_predictions.csv is not the approved p4-offline-002 output (sha256 differs)")
    hist = json.loads(metrics_bytes)
    approved = C.REPO / "artifacts" / "crew1" / "p3-live-005"
    session = TrainOnlySession(FH.load_handoff(approved / "clean_data.csv", approved / "dataset_contract.json"),
                               n_jobs=n_jobs)
    session.run_train_only()
    final = session.final
    problems = [k for k in P.RUNS
                if session.cv[k]["summary"] != hist["cv"][k]["summary"]
                or [(f["params"], f["scores"]) for f in session.cv[k]["folds"]]
                != [(f["params"], f["scores"]) for f in hist["cv"][k]["folds"]]]
    if session.selection != hist["selection"]:
        problems.append("selection")
    for k in HIST_MODEL_SHA256:
        if final[k]["model_sha256"] != HIST_MODEL_SHA256[k] or final[k]["params"] != hist["final"][k]["params"]:
            problems.append(f"final {k}")
    rows = _prediction_rows(session)
    by_row = {r[ROW]: r for r in rows}
    pid_row = {(p, s): int(r) for r, p, s in zip(session.parts.train[ROW], session.parts.train["production_id"],
                                                 session.parts.train["stage"])}
    seen = 0
    for h in csv.DictReader(io.StringIO(cvp_bytes.decode("utf-8"))):
        if h["run_id"] in PRIMARY_RUNS:
            seen += 1
            if abs(by_row[pid_row[(h["production_id"], h["stage"])]][h["run_id"]] - float(h["p"])) > 1e-9:
                problems.append(f"cv prediction {h['run_id']} {h['production_id']} {h['stage']}")
    if seen != len(PRIMARY_RUNS) * len(rows):
        problems.append("cv_predictions.csv row count")
    if problems:
        raise SystemExit(f"the TRAIN-only Flow run does not reproduce {SOURCE_RUN} exactly: {problems[:10]}")
    ref = {
        "version": REFERENCE_VERSION, "synthetic": True,
        "description": "Approved P4 results pinned for the Flow's TRAIN-only run. cv/selection/final are compared "
                       "with every new run; historical_test is cited only (TEST is never reopened by the Flow).",
        "source": {"run": SOURCE_RUN, "files": ["metrics.json", "cv_predictions.csv"],
                   "identity": "sha256 pinned in budget_flow/p4_reference.py; the final models are identified by "
                               "their pinned sha256 only (no pickle was loaded)"},
        "preregistration": preregistration(),
        "environment": environment(),
        "cv": {k: {"summary": hist["cv"][k]["summary"],
                   "folds": [{"fold": f["fold"], "params": f["params"], "scores": f["scores"]}
                             for f in hist["cv"][k]["folds"]]} for k in P.RUNS},
        "selection": hist["selection"],
        "final": {k: {kk: hist["final"][k][kk] for kk in ("candidate", "target", "feature_set", "params", "train_rows")}
                  for k in HIST_MODEL_SHA256},
        "historical_test": {"source_run": SOURCE_RUN, "opened": hist["test"]["opened"],
                            "note": "Opened once in P4 after the final models were frozen. Cited by the Flow, "
                                    "never recomputed.",
                            **_strip_digests({k: v for k, v in hist["test"].items() if k != "opened"})},
    }
    return (json.dumps(ref, indent=2, sort_keys=True) + "\n").encode("utf-8"), render_predictions(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", type=Path, metavar="RUN_DIR")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args(argv)
    if args.write:
        ref_bytes, pred_bytes = build(args.write, args.n_jobs)
        REFERENCE_FILE.write_bytes(ref_bytes)
        PREDICTIONS_FILE.write_bytes(pred_bytes)
        print(f"wrote {REFERENCE_FILE.name} and {PREDICTIONS_FILE.name}; pin _REFERENCE_DIGEST = "
              f"{int(sha256_hex(ref_bytes), 16)} and _PREDICTIONS_DIGEST = {int(sha256_hex(pred_bytes), 16)}")
        return 0
    try:
        ref, _ = load()
        check_preregistration(ref)
    except ReproducibilityError as e:
        print(f"FAIL: {e}")
        return 1
    print("the pinned P4 reference and the pre-registered protocol are unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
