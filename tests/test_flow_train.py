"""P5-C: Crew 2's TRAIN-only step of the Flow. Offline. ALL DATA IS SYNTHETIC.

Fast tests: the TEST steps are refused (also through the P4 base-class methods), the pre-registration and the
pinned P4 reference are enforced, and the reproducibility comparison fails hard on a changed selection, changed
parameters or deviations beyond the pre-registered tolerances.

The ``guarded_run`` tests run the full pre-registered protocol once (about 12 minutes) in a child process whose
every interpreter, including the joblib workers, refuses to import Crew 1 or CrewAI and to open the raw export
or the network, with every TEST entry point replaced by a counting refusal. They check the four artifacts.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

import pytest

from budget_flow import contract as C
from budget_flow import handoff as FH
from budget_flow import outputs as O
from budget_flow import p4_reference as R
from budget_flow import train as T
from crew2.ml import data as D
from crew2.ml import models as M
from crew2.ml import protocol as P
from crew2.ml import selection as S
from crew2.ml.features import F27, ROW, outcome_only_columns
from tests.conftest import REPO

APPROVED = REPO / "artifacts" / "crew1" / "p3-live-005"


@pytest.fixture(scope="module")
def handoff():
    return FH.load_handoff(APPROVED / "clean_data.csv", APPROVED / "dataset_contract.json")


@pytest.fixture(scope="module")
def pinned():
    return R.load()


# ---------------------------------------------------------------- TEST stays sealed

def test_every_test_step_is_refused(handoff):
    s = T.TrainOnlySession(handoff)
    for call, err in ((s.evaluate_on_test_once, FH.TestAccessError), (s.run_all, FH.TestAccessError),
                      (s.metrics, FH.TestAccessError), (s.build_handoff, T.FlowPathError),
                      (s.predict_demo, T.FlowPathError)):
        with pytest.raises(err):
            call()


def test_the_p4_base_class_test_evaluation_hits_the_sealed_vault(handoff):
    """Even the unchanged P4 method, called directly on the Flow session, cannot reach a test label."""
    s = T.TrainOnlySession(handoff)
    s.final_models = {"primary": {"stand-in": 1}}
    s.final = {"primary": {"model_sha256": P.model_sha256(s.final_models["primary"])}}
    with pytest.raises(FH.TestAccessError):
        P.Session.evaluate_on_test_once(s)
    with pytest.raises(T.FlowPathError):
        P.Session.run_all(s)                       # starts by rebuilding the handoff from raw: refused
    assert s.test is None and not s.parts.vault.opened


def test_even_a_p4_vault_built_from_the_flow_text_holds_no_test_label(handoff):
    header, rows = C.parse_and_redact((APPROVED / "clean_data.csv").read_bytes())
    vault = D.partition(C.redacted_csv_text(header, rows)).vault
    with pytest.raises(ValueError):                # the label fields are empty: nothing to parse
        vault.open({"primary": "0" * 64})


def test_a_session_refuses_an_unsealed_or_invalid_handoff(handoff):
    unsealed = FH.FlowHandoff(D.partition(C.redacted_csv_text(*C.parse_and_redact(
        (APPROVED / "clean_data.csv").read_bytes()))), handoff.report)
    with pytest.raises(FH.TestAccessError):
        T.TrainOnlySession(unsealed)
    bad = copy.copy(handoff.report)
    bad.violations = [C.Violation("UNIT_SCALE_ANOMALY", "budget_pre_vat", "x")]
    with pytest.raises(C.ContractViolationError):
        T.TrainOnlySession(FH.FlowHandoff(handoff.parts, bad))


_TEST_ENTRY_POINTS = {"evaluate_on_test_once", "run_all", "fresh_metrics", "bootstrap_ci", "open", "metrics",
                      "build_handoff", "TestVault"}


def test_no_flow_module_calls_a_test_entry_point():
    """Statically: no call in budget_flow/ to a TEST evaluation, W3, the bootstrap, a vault or the raw rebuild
    (the refusing overrides in train.py are definitions, not calls)."""
    for path in sorted((REPO / "budget_flow").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
                owner = f.value.attr if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) else ""
                if name == "open" and owner != "vault":
                    continue                          # file opens are covered by the runtime guard
                assert name not in _TEST_ENTRY_POINTS, (path.name, name, node.lineno)


# ---------------------------------------------------------------- pre-registration and pinned reference

def test_the_pinned_reference_and_preregistration_hold():
    assert R.main(["--check"]) == 0


@pytest.mark.parametrize("patch", [
    ("M", "SEED", 7), ("M", "K_INNER", 4), ("S", "BRIER_TIE", 0.01), ("S", "AP_DECIMALS", 3),
    ("M", "GRIDS", {**M.GRIDS, "LR": [{"C": c} for c in (0.01, 0.1, 1.0)]}),
    ("M", "CANDIDATE_ORDER", ("LR", "B0", "GB")), ("P", "N_FOLDS", 4),
])
def test_a_changed_preregistered_setting_fails_before_fitting(monkeypatch, pinned, patch):
    mod, name, value = patch
    monkeypatch.setattr({"M": M, "S": S, "P": P}[mod], name, value)
    with pytest.raises(R.PreregistrationChanged):
        R.check_preregistration(pinned[0])


def test_a_changed_decision_rule_fails_before_fitting(monkeypatch, pinned, tmp_path):
    monkeypatch.setattr(S, "apply_rule", lambda means: {"selected": "GB", "decided_by": "x", "tie_set": ["GB"]})
    with pytest.raises(R.PreregistrationChanged, match="behaviour"):
        R.check_preregistration(pinned[0])
    monkeypatch.undo()
    edited = tmp_path / "selection.py"
    edited.write_bytes(R.SELECTION_CODE.read_bytes() + b"\n# edited\n")
    monkeypatch.setattr(R, "SELECTION_CODE", edited)
    with pytest.raises(R.PreregistrationChanged, match="selection rule code"):
        R.check_preregistration(pinned[0])


def test_a_changed_reference_file_is_refused(monkeypatch, tmp_path):
    for attr in ("REFERENCE_FILE", "PREDICTIONS_FILE"):
        f = tmp_path / getattr(R, attr).name
        f.write_bytes(getattr(R, attr).read_bytes().replace(b"0.1", b"0.2", 1))
        monkeypatch.setattr(R, attr, f)
        with pytest.raises(R.ReproducibilityError, match="pinned sha256"):
            R.load()
        monkeypatch.undo()


def test_the_reference_holds_no_row_level_test_data_and_only_train_predictions(pinned, handoff):
    ref, rows = pinned
    assert "SYN-" not in R.REFERENCE_FILE.read_text(encoding="utf-8")
    assert "SYN-" not in R.PREDICTIONS_FILE.read_text(encoding="utf-8")
    assert [r[ROW] for r in rows] == sorted(handoff.train[ROW].astype(int))
    assert list(rows[0]) == list(R.PREDICTION_COLUMNS)            # no label column
    assert ref["historical_test"]["opened"] == 1 and ref["historical_test"]["source_run"] == "p4-offline-002"
    assert ref["selection"]["selected"] == "LR" and ref["final"]["primary"]["params"] == {"C": 10.0}


# ---------------------------------------------------------------- reproducibility comparison

def _as_run(ref, rows):
    """A run identical to the approved one, built from the pinned reference."""
    cv = copy.deepcopy(ref["cv"])
    final = {k: {**v, "model_sha256": R.HIST_MODEL_SHA256[k]} for k, v in copy.deepcopy(ref["final"]).items()}
    return cv, copy.deepcopy(ref["selection"]), final, copy.deepcopy(rows)


def _verify(run, ref, rows, mode):
    return R.verify(*run, ref, rows, mode)


def test_the_approved_run_passes_in_both_modes(pinned):
    ref, rows = pinned
    for mode in ("exact", "tolerance"):
        out = _verify(_as_run(ref, rows), ref, rows, mode)
        assert out["ok"] and out["max_abs_deviation"] == {"cv_mean": 0.0, "fold_score": 0.0, "prediction": 0.0}


@pytest.mark.parametrize("mutate", ["selected", "decided_by", "tie_set", "final_params", "fold_params"])
def test_a_changed_selection_or_parameter_is_a_hard_failure_in_every_mode(pinned, mutate):
    ref, rows = pinned
    for mode in ("exact", "tolerance"):
        cv, sel, final, preds = _as_run(ref, rows)
        if mutate == "selected":
            sel["selected"] = "GB"
        elif mutate == "decided_by":
            sel["decided_by"] = "tie-break 1: higher mean PR-AUC (average_precision_score)"
        elif mutate == "tie_set":
            sel["tie_set"] = ["LR", "GB"]
        elif mutate == "final_params":
            final["primary"]["params"] = {"C": 1.0}
        else:
            cv["F27.is_overrun.GB"]["folds"][2]["params"] = {"n_estimators": 999, "max_depth": 9, "learning_rate": 0.5}
        with pytest.raises(R.SelectionChanged):
            R.verify(cv, sel, final, preds, ref, rows, mode)


def _shift(run, what, delta):
    cv, sel, final, preds = run
    if what == "cv_mean":
        cv["F27.is_overrun.LR"]["summary"]["brier"]["mean"] += delta
    elif what == "fold_score":
        cv["F27.is_overrun.LR"]["folds"][1]["scores"]["log_loss"] += delta
    else:
        preds[5]["final.primary"] += delta
    return run


@pytest.mark.parametrize("what, ok, bad", [("cv_mean", 5e-7, 2e-6), ("fold_score", 5e-6, 2e-5),
                                           ("prediction", 5e-7, 2e-6)])
def test_preregistered_tolerances_outside_the_reference_environment(pinned, what, ok, bad):
    ref, rows = pinned
    assert _verify(_shift(_as_run(ref, rows), what, ok), ref, rows, "tolerance")["ok"]
    with pytest.raises(R.ReproducibilityError, match=what):
        _verify(_shift(_as_run(ref, rows), what, bad), ref, rows, "tolerance")
    with pytest.raises(R.ReproducibilityError, match="exact"):        # the pinned environment allows no deviation
        _verify(_shift(_as_run(ref, rows), what, 1e-12), ref, rows, "exact")


def test_counts_and_model_digests(pinned):
    ref, rows = pinned
    run = _as_run(ref, rows)
    run[0]["F27.is_overrun.B0"]["folds"][0]["scores"]["n"] += 1
    with pytest.raises(R.ReproducibilityError):
        _verify(run, ref, rows, "tolerance")
    run = _as_run(ref, rows)
    run[2]["primary"]["model_sha256"] = "0" * 64
    with pytest.raises(R.ReproducibilityError, match="sha256"):
        _verify(run, ref, rows, "exact")
    out = _verify(run, ref, rows, "tolerance")       # elsewhere the bytes may differ; recorded, not required
    assert out["final_model_sha256_equal_to_p4"] == {"primary": False, "secondary": True}


def test_the_mode_is_exact_only_in_the_pinned_environment(pinned):
    ref = pinned[0]
    assert R.mode_for(dict(ref["environment"]), ref) == "exact"
    assert R.mode_for({**ref["environment"], "scikit-learn": "0.0"}, ref) == "tolerance"


# ---------------------------------------------------------------- output folder policy

def test_historical_and_protected_folders_are_never_written(tmp_path):
    for d in (REPO / "artifacts" / "x", REPO / "out" / "crew2" / "p4-offline-002", REPO / "contracts" / "x",
              REPO / "budget_flow" / "contracts" / "x", REPO / "data" / "x"):
        with pytest.raises(T.FlowPathError):
            T._check_out_dir(d)
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(T.FlowPathError, match="not empty"):
        T._check_out_dir(tmp_path)


def test_historical_p4_files_are_unchanged():
    """Against the merged P4 main: only the P5-B import changes in crew2; no contract, artifact, document,
    validator or tool."""
    git = shutil.which("git")
    if not git:
        pytest.skip("git unavailable")
    p4_docs = ("docs/crew2.md", "docs/p4_model_card.md", "docs/p4_plan.md", "docs/p4_leakage_proof.md",
               "docs/model_selection_rule.md")
    res = subprocess.run([git, "diff", "--name-only", "ea7366f", "--", "crew2", "crew1", "contracts", "artifacts",
                          "tools", "gate", "generator", "data", *p4_docs, *(f"tests/test_crew2_{n}.py" for n in (
                              "crew", "data", "feature_reference", "frozen", "leakage", "rules"))],
                         cwd=REPO, capture_output=True, text=True)
    if res.returncode:
        pytest.skip("git history unavailable")
    assert sorted(res.stdout.split()) == ["crew2/__init__.py", "crew2/ml/__init__.py", "crew2/ml/data.py",
                                          "tools/publication_check.py", "tools/publish_scan.py"]
    # P5-G: crew2/ml/data.py is normalized to LF; ignoring line endings, only the deferred Crew 1 import changes
    data_diff = subprocess.run([git, "diff", "-U0", "--ignore-cr-at-eol", "ea7366f", "--", "crew2/ml/data.py"],
                               cwd=REPO, capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    assert [ln for ln in data_diff if ln[:1] in "+-" and ln[:3] not in ("+++", "---")] == [
        "-from crew1.tools.cleaning import build_clean_data",
        "+    from crew1.tools.cleaning import build_clean_data   # imported here: the file-based Flow path never "
        "loads Crew 1", "+"]
    # P5-F: the scanner only GAINS the canonical-evidence exemption; no earlier rule line is removed or changed
    scan_diff = subprocess.run([git, "diff", "-U0", "ea7366f", "--", "tools/publish_scan.py"], cwd=REPO,
                               capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    removed = [ln[1:] for ln in scan_diff if ln.startswith("-") and not ln.startswith("---")]
    assert removed == ["            secrets += [f for f in _secret_findings(disk, logical) "
                       "if not _approved_p3_run_digest(logical, blob, f)]"]


# ---------------------------------------------------------------- the guarded full run

_SITECUSTOMIZE = r'''
import importlib.abc, os, sys
_REPO = os.environ["PSBA_GUARD_REPO"]
_LOG = os.environ["PSBA_GUARD_LOG"]
_BLOCKED = tuple(os.environ.get("PSBA_GUARD_BLOCK", "crew1,crewai").split(","))
def _norm(p):
    return os.path.realpath(p).replace("\\", "/").lower()
_FORBIDDEN = [_norm(os.path.join(_REPO, d)) for d in ("data", "crew1")]
def _violation(msg):
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(f"{os.getpid()} {msg}\n")
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            _violation("import " + name)
            raise ImportError("import of " + name + " is forbidden in the Flow")
        return None
sys.meta_path.insert(0, _Block())
_LOG_N = _norm(_LOG)
def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = _norm(os.fsdecode(args[0]))
        if p != _LOG_N and (any(p == f or p.startswith(f + "/") for f in _FORBIDDEN) or "/raw/" in p):
            _violation("open " + p)
            raise PermissionError("forbidden read in the Flow: " + p)
    elif event == "socket.connect":
        addr = args[1] if len(args) > 1 else None
        if not (isinstance(addr, tuple) and addr and addr[0] in ("127.0.0.1", "::1", "localhost")):
            _violation("connect " + repr(addr))
            raise PermissionError("network is blocked in the Flow")
sys.addaudithook(_hook)
'''

_DRIVER = r'''
import json, os, sys
from budget_flow import handoff as FH, train as T
from crew2.ml import data as D, metrics as MT, protocol as P
calls = {"TestVault.open": 0, "SealedVault.open": 0, "Session.evaluate_on_test_once": 0, "Session.run_all": 0,
         "Session.metrics": 0, "bootstrap_ci": 0}
def counting(name, real=None):
    def f(*a, **k):
        calls[name] += 1
        if real is not None:
            return real(*a, **k)
        raise RuntimeError(name + " is forbidden in the Flow")
    return f
D.TestVault.open = counting("TestVault.open")
FH.SealedVault.open = counting("SealedVault.open", FH.SealedVault.open)
P.Session.evaluate_on_test_once = counting("Session.evaluate_on_test_once")
P.Session.run_all = counting("Session.run_all")
P.Session.metrics = counting("Session.metrics")
P.bootstrap_ci = MT.bootstrap_ci = counting("bootstrap_ci")
reads = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        reads.append(os.path.realpath(os.fsdecode(args[0])))
sys.addaudithook(hook)
m = T.run_flow_train(CLEAN, CONTRACT, OUT, n_jobs=N_JOBS)
inputs = {os.path.realpath(CLEAN), os.path.realpath(CONTRACT)}
print(json.dumps({"manifest": m, "calls": calls,
                  "crew_modules": sorted(x for x in sys.modules if x.split(".")[0] in ("crew1", "crewai")),
                  "w3_or_tools_loaded": sorted(x for x in sys.modules if x in ("crew2.validators", "crew2.tools",
                                                                                "crew2.crew", "crew2.offline")),
                  "input_reads": sorted(p for p in set(reads) if p in inputs)}))
'''


@pytest.fixture(scope="session")
def guarded_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("p5c_guarded")
    for name in ("clean_data.csv", "dataset_contract.json"):
        shutil.copyfile(APPROVED / name, base / name)
    harness = base / "harness"
    harness.mkdir()
    (harness / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    log = base / "violations.log"
    out = base / "flow_out"
    n_jobs = max(1, min(5, os.cpu_count() or 1))
    code = (f"CLEAN = {str(base / 'clean_data.csv')!r}\nCONTRACT = {str(base / 'dataset_contract.json')!r}\n"
            f"OUT = {str(out)!r}\nN_JOBS = {n_jobs}\n") + _DRIVER
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONPATH=os.pathsep.join([str(harness), str(REPO)]), PSBA_GUARD_REPO=str(REPO),
               PSBA_GUARD_LOG=str(log), PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(base), env=env,
                         timeout=5400)
    assert res.returncode == 0, res.stderr[-4000:]
    evidence = json.loads(res.stdout.strip().splitlines()[-1])
    evidence["violations"] = log.read_text(encoding="utf-8") if log.exists() else ""
    evidence["out"] = out
    evidence["inputs"] = (base / "clean_data.csv", base / "dataset_contract.json")
    return evidence


def test_guard_the_run_loaded_no_crew1_read_no_raw_file_and_never_touched_test(guarded_run):
    assert guarded_run["violations"] == ""                     # no process (parent or worker) tried
    assert guarded_run["crew_modules"] == [] and guarded_run["w3_or_tools_loaded"] == []
    assert guarded_run["calls"] == {k: 0 for k in guarded_run["calls"]}
    assert len(guarded_run["input_reads"]) == 2                # exactly Crew 1's two delivered files
    tv = guarded_run["manifest"]["test_vault"]
    assert tv == {"class": "SealedVault", "opened": False, "opens": 0, "test_metrics_computed": False}
    assert guarded_run["manifest"]["historical_test"]["recomputed"] is False


def test_guard_the_harness_blocks_a_worker_style_child(tmp_path):
    """Control: the same sitecustomize stops a fresh interpreter that tries a raw read or a Crew 1 import."""
    harness = tmp_path / "h"
    harness.mkdir()
    (harness / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONPATH=os.pathsep.join([str(harness), str(REPO)]), PSBA_GUARD_REPO=str(REPO),
               PSBA_GUARD_LOG=str(tmp_path / "v.log"))
    raw = REPO / "data" / "synthetic" / "v2" / "raw" / "productions.csv"
    for code in (f"open({str(raw)!r}).read(1)", "import crew1"):
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env)
        assert res.returncode != 0
    assert len((tmp_path / "v.log").read_text(encoding="utf-8").splitlines()) == 2


def test_four_artifacts_are_written_and_small(guarded_run):
    out = guarded_run["out"]
    arts = guarded_run["manifest"]["artifacts"]
    for name in O.ARTIFACTS:
        data = (out / name).read_bytes()
        assert data and arts[name] == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert arts["model.joblib"]["bytes"] < 200_000


def test_selection_and_approved_cv_equivalence(guarded_run, pinned):
    m = guarded_run["manifest"]
    assert m["selected_model"]["candidate"] == "LR" and m["selected_model"]["params"] == {"C": 10.0}
    assert m["selected_model"]["decided_by"] == "lowest mean grouped-CV Brier"
    assert m["selected_model"]["audit_violations"] == {"primary": 0, "secondary": 0}
    rep = m["reproducibility"]
    assert rep["ok"] and rep["selection_equal"] and rep["parameters_equal"]
    assert rep["mode"] == R.mode_for(R.environment(), pinned[0])
    if rep["mode"] == "exact":
        assert rep["max_abs_deviation"] == {"cv_mean": 0.0, "fold_score": 0.0, "prediction": 0.0}
        assert m["selected_model"]["model_sha256"] == R.HIST_MODEL_SHA256
    else:
        assert all(rep["max_abs_deviation"][k] <= R.TOLERANCES[k] for k in R.TOLERANCES)
    assert m["timing"]["train_only_seconds"] > 0 and set(m["timing"]["steps_seconds"]) == {*P.RUNS, "final_refit"}


def test_features_csv(guarded_run, handoff):
    text = (guarded_run["out"] / "features.csv").read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    header = list(rows[0])
    assert header == list(O.FEATURE_COLUMNS) and len(header) == 4 + 27 and header[4:] == list(F27)
    assert not set(header) & (set(C.LABEL_COLUMNS) | outcome_only_columns())
    assert len(rows) == 1538 and [int(r["handoff_row"]) for r in rows] == list(range(1538))
    assert {s: sum(r["split"] == s for r in rows) for s in ("train", "test", "demo")} == {
        s: handoff.counts[s]["rows"] for s in ("train", "test", "demo")}
    assert all(r["cv_fold"] == "-1" for r in rows if r["split"] != "train")
    train = handoff.train.set_index(ROW)
    for r in rows[:50]:
        if r["split"] == "train":
            src = train.loc[int(r["handoff_row"])]
            assert r["production_id"] == src["production_id"] and r["budget_pre_vat"] == repr(float(src["budget_pre_vat"]))
    schema = json.loads((guarded_run["out"] / "features.schema.json").read_text(encoding="utf-8"))
    assert list(schema["columns"]) == header and schema["n_features"] == 27


def test_model_joblib(guarded_run, pinned, handoff):
    import joblib
    path = guarded_run["out"] / "model.joblib"
    data = path.read_bytes()
    # loaded only after its digest equals the one this guarded run recorded: never an untrusted pickle
    assert hashlib.sha256(data).hexdigest() == guarded_run["manifest"]["artifacts"]["model.joblib"]["sha256"]
    assert b"SYN-" not in data
    obj = joblib.load(io.BytesIO(data))
    assert obj["synthetic"] and obj["features"] == list(F27) and obj["provenance"]["trained_on"] == "train split only"
    # model_sha256 is the digest of the fitted object when it was frozen (as in P4); a re-dump of an unpickled
    # object is not byte-stable, so the loaded models are identified by their predictions instead
    assert obj["model_sha256"] == guarded_run["manifest"]["selected_model"]["model_sha256"]
    train = handoff.train.sort_values(ROW)
    for name in ("primary", "secondary"):
        preds = obj["models"][name].predict(train)
        worst = max(abs(a - r[f"final.{name}"]) for a, r in zip(preds, pinned[1]))
        assert worst == 0.0 if guarded_run["manifest"]["reproducibility"]["mode"] == "exact" else worst <= 1e-6


def test_evaluation_report_separates_current_cv_from_cited_historical_test(guarded_run, pinned):
    text = (guarded_run["out"] / "evaluation_report.md").read_text(encoding="utf-8")
    a, b = text.index("## A. Current TRAIN-only"), text.index("## B. Historical P4 TEST")
    assert a < b
    part_a, part_b = text[a:b], text[b:]
    assert "This run did not reopen TEST" in part_b and "opened 0 times" in part_b
    ht = pinned[0]["historical_test"]
    for t in ("primary", "secondary"):                         # the cited values are the pinned P4 values
        assert f"{ht[t]['overall']['brier']:.4f} [{ht[t]['bootstrap_ci']['brier']['lo']:.4f}" in part_b
    assert "test" not in "".join(l for l in part_a.splitlines() if l.startswith("| `F27")).lower()
    lr = pinned[0]["cv"]["F27.is_overrun.LR"]["summary"]["brier"]
    assert f"{lr['mean']:.4f} ± {lr['sd']:.4f}" in part_a and "Selected: **LR**" in part_a


def test_model_card_sections(guarded_run):
    text = (guarded_run["out"] / "model_card.md").read_text(encoding="utf-8")
    for h in ("## Purpose and intended use", "## Training data", "## Evaluation", "## Limitations",
              "## Synthetic-data disclosure", "## Ethical considerations"):
        assert h in text
    assert "ALL DATA IS SYNTHETIC" in text and "not recomputed" in text


def test_no_row_level_or_test_label_content_in_any_artifact(guarded_run):
    out = guarded_run["out"]
    for name in ("evaluation_report.md", "model_card.md", "run_manifest.json", "features.schema.json"):
        assert "SYN-" not in (out / name).read_text(encoding="utf-8"), name
    header = (out / "features.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert not set(header) & set(C.LABEL_COLUMNS)
