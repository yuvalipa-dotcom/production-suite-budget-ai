"""The P4 protocol as one ordered session (Addendum P4-1; docs/p4_plan.md section d).

    handoff -> grouped CV (9 pre-registered runs) -> selection rule -> final refit on all
    of train (primary + secondary) -> TEST opened once -> DEMO inference

Every step is deterministic. A step refuses to run before its prerequisites, the test
vault refuses a second opening, and every unit of fitting work is audited for group
disjointness (``resample.audit``) before its result is accepted.
"""

from __future__ import annotations

import hashlib
import io

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import data as D
from .features import FEATURE_SETS, PRIMARY_FEATURE_SET, PRIMARY_TARGET, ROW, TARGETS
from .metrics import THRESHOLD, by_stage, bootstrap_ci, fold_summary, reliability, scores
from .models import CANDIDATE_ORDER, GRIDS, SEED, fit_candidate, set_spy_sink
from .resample import LeakageError, Trail, audit
from .selection import apply_rule

N_FOLDS = 5
STAGES = ("S0", "S1", "S2", "S3")
RUN_GROUPS = (("is_overrun", "F27", "primary"), ("is_material_overrun", "F27", "secondary"),
              ("is_overrun", "F29", "research"))
RUNS = {f"{fs}.{target}.{c}": {"target": target, "feature_set": fs, "candidate": c, "role": role}
        for target, fs, role in RUN_GROUPS for c in CANDIDATE_ORDER}
DEMO_LABEL = "experimental / demo prediction"


class OrderError(RuntimeError):
    """A protocol step was requested before its prerequisites."""


def model_sha256(model) -> str:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _pid_map(train: pd.DataFrame) -> dict[int, str]:
    return dict(zip(train[ROW].astype(int), train["production_id"]))


# ---------------------------------------------------------------- one unit of fitting work

def _fit_unit(train_part: pd.DataFrame, target: str, fs: str, kind: str, fold: int, forbidden: set[int]):
    """Fit one candidate on ``train_part`` only (tuning + calibration inside it), then audit the trail."""
    rows = np.asarray(sorted(train_part[ROW].astype(int)))
    trail = Trail(_pid_map(train_part), root=rows.tolist())
    sink: list = []
    set_spy_sink(sink)
    try:
        model, params, tuning = fit_candidate(kind, fs, target, train_part, rows, fold, trail)
    finally:
        set_spy_sink(None)
    trail.spy = sink
    summary = audit(trail, forbidden)
    return model, params, tuning, summary, trail


def outer_fold(train: pd.DataFrame, target: str, fs: str, kind: str, fold: int, forbidden: set[int],
               keep_trail: bool = False) -> dict:
    part = train[train["cv_fold"] != fold].reset_index(drop=True)
    valid = train[train["cv_fold"] == fold].reset_index(drop=True)
    if set(part["production_id"]) & set(valid["production_id"]):
        raise LeakageError(f"outer fold {fold}: a production on both sides")
    model, params, tuning, summary, trail = _fit_unit(part, target, fs, kind, fold, forbidden)
    p = model.predict(valid)
    oof = valid[[ROW, "production_id", "stage", "cv_fold"]].assign(y=valid[target].astype(int).to_numpy(), p=p)
    out = {"fold": fold, "params": params, "tuning": tuning, "scores": scores(oof["y"], oof["p"]),
           "by_stage": by_stage(oof, "y", "p"), "audit": summary, "oof": oof,
           "partition_rows": int(len(part)), "validation_rows": int(len(valid))}
    if keep_trail:
        out["trail"] = trail
    return out


# ---------------------------------------------------------------- the session

class Session:
    def __init__(self, n_jobs: int = 1, raw_dir=D.RAW_DIR, contracts_dir=D.CONTRACTS_DIR,
                 expected_sha256: str | None = None):
        self.n_jobs = n_jobs
        self.raw_dir, self.contracts_dir = raw_dir, contracts_dir
        self.expected_sha256 = expected_sha256 or D.EXPECTED_CLEAN_SHA256
        self.handoff_text: str | None = None
        self.handoff_info: dict | None = None
        self.parts: D.Partitioned | None = None
        self.cv: dict[str, dict] = {}
        self.oof: dict[str, pd.DataFrame] = {}
        self.selection: dict | None = None
        self.final: dict | None = None
        self.final_models: dict = {}
        self.test: dict | None = None
        self.test_predictions: pd.DataFrame | None = None
        self.demo_predictions: pd.DataFrame | None = None

    # ------------------------------------------------------------ input
    def build_handoff(self) -> dict:
        text, log = D.build_handoff(self.raw_dir, self.contracts_dir)
        info = D.verify_handoff(text, log, self.expected_sha256)
        self.handoff_text, self.handoff_info = text, info
        self.parts = D.partition(text)
        return info | {"counts": self.parts.counts}

    def _need(self, cond, msg):
        if not cond:
            raise OrderError(msg)

    @property
    def forbidden_rows(self) -> set[int]:
        p = self.parts
        return set(p.test_features[ROW].astype(int)) | set(p.demo_features[ROW].astype(int))

    # ------------------------------------------------------------ grouped CV
    def run_cv(self, run_id: str) -> dict:
        self._need(self.parts is not None, "build the handoff first")
        if run_id not in RUNS:
            raise KeyError(f"unknown run id; choose one of {list(RUNS)}")
        if run_id in self.cv:
            return self.cv[run_id]
        spec = RUNS[run_id]
        train = self.parts.train
        folds = Parallel(n_jobs=self.n_jobs)(
            delayed(outer_fold)(train, spec["target"], spec["feature_set"], spec["candidate"], f, self.forbidden_rows)
            for f in range(N_FOLDS))
        oof = pd.concat([f.pop("oof") for f in folds]).sort_values(ROW).reset_index(drop=True)
        stage_summary = {}
        for s in STAGES:
            stage_summary[s] = fold_summary([f["by_stage"][s] for f in folds])
        result = {
            **spec, "run_id": run_id, "grid": GRIDS[spec["candidate"]],
            "folds": [{k: f[k] for k in ("fold", "params", "scores", "partition_rows", "validation_rows", "audit")}
                      | {"tuning": f["tuning"]} for f in folds],
            "summary": fold_summary([f["scores"] for f in folds]),
            "by_stage": stage_summary,
            "reliability_oof": {"all": reliability(oof["y"], oof["p"]),
                                **{s: reliability(g["y"], g["p"]) for s, g in sorted(oof.groupby("stage"))}},
            "audit": {k: sum(f["audit"][k] for f in folds) for k in folds[0]["audit"]},
        }
        self.cv[run_id], self.oof[run_id] = result, oof
        return result

    # ------------------------------------------------------------ selection
    def apply_selection_rule(self) -> dict:
        ids = [f"{PRIMARY_FEATURE_SET}.{PRIMARY_TARGET}.{c}" for c in CANDIDATE_ORDER]
        self._need(all(i in self.cv for i in ids), f"run the grouped CV for {ids} first")
        means = {c: {"brier": self.cv[i]["summary"]["brier"]["mean"],
                     "average_precision": self.cv[i]["summary"]["average_precision"]["mean"]}
                 for c, i in zip(CANDIDATE_ORDER, ids)}
        self.selection = apply_rule(means) | {"inputs": means, "runs": ids}
        return self.selection

    # ------------------------------------------------------------ final refit (before test)
    def fit_final_models(self) -> dict:
        self._need(self.selection is not None, "apply the selection rule first")
        if self.final is not None:
            return self.final
        kind = self.selection["selected"]
        train = self.parts.train
        out = {}
        for name, target in (("primary", PRIMARY_TARGET), ("secondary", "is_material_overrun")):
            model, params, tuning, summary, _ = _fit_unit(train, target, PRIMARY_FEATURE_SET, kind, -1,
                                                          self.forbidden_rows)
            self.final_models[name] = model
            out[name] = {"candidate": kind, "target": target, "feature_set": PRIMARY_FEATURE_SET, "params": params,
                         "tuning": tuning, "audit": summary, "train_rows": int(len(train)),
                         "model_sha256": model_sha256(model)}
        self.final = out
        return out

    # ------------------------------------------------------------ the single test evaluation
    def evaluate_on_test_once(self) -> dict:
        self._need(self.final is not None, "fit the final models first")
        shas = {k: v["model_sha256"] for k, v in self.final.items()}
        for k, m in self.final_models.items():          # the models evaluated are exactly the frozen ones
            if model_sha256(m) != shas[k]:
                raise LeakageError("a final model changed after it was frozen")
        labels = self.parts.vault.open(shas)             # raises on a second opening
        feats = self.parts.test_features
        df = feats[[ROW, "production_id", "stage"]].merge(labels, on=["production_id", "stage"], validate="1:1")
        df = df.assign(p_primary=self.final_models["primary"].predict(feats),
                       p_secondary=self.final_models["secondary"].predict(feats))
        result = {"opened": len(self.parts.vault.opens), "rows": int(len(df)),
                  "productions": int(df["production_id"].nunique()), "threshold": THRESHOLD}
        for name, target, p in (("primary", "is_overrun", "p_primary"),
                                ("secondary", "is_material_overrun", "p_secondary")):
            result[name] = {"target": target, "candidate": self.final[name]["candidate"],
                            "model_sha256": shas[name], "overall": scores(df[target], df[p]),
                            "by_stage": by_stage(df, target, p),
                            "reliability": reliability(df[target], df[p]),
                            "bootstrap_ci": bootstrap_ci(df, target, p)}
        self.test, self.test_predictions = result, df
        return result

    # ------------------------------------------------------------ demo inference only
    def predict_demo(self) -> pd.DataFrame:
        self._need(self.final is not None, "fit the final models first")
        feats = self.parts.demo_features
        self.demo_predictions = pd.DataFrame({
            "production_id": feats["production_id"], "stage": feats["stage"],
            "p_overrun": self.final_models["primary"].predict(feats),
            "label": DEMO_LABEL}).sort_values(["production_id", "stage"]).reset_index(drop=True)
        return self.demo_predictions

    # ------------------------------------------------------------ reference and report
    def naive_reference(self) -> dict:
        """The hard rule ``current_position_ils > 0`` on train, per stage: a reference only, never a candidate."""
        t = self.parts.train
        pred = (t["current_position_ils"] > 0).astype(float)
        out = {"rule": "current_position_ils > 0", "rows": "train", "overall": scores(t["is_overrun"], pred)}
        out["by_stage"] = {s: scores(g["is_overrun"], pred[g.index]) for s, g in sorted(t.groupby("stage"))}
        return out

    def data_summary(self) -> dict:
        t = self.parts.train
        return {"handoff": self.handoff_info, "counts": self.parts.counts,
                "train_prevalence": {tgt: {"overall": float(t[tgt].mean()),
                                           **{s: float(g[tgt].mean()) for s, g in sorted(t.groupby("stage"))}}
                                     for tgt in TARGETS},
                "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()}}

    def metrics(self) -> dict:
        self._need(len(self.cv) == len(RUNS) and self.test is not None and self.demo_predictions is not None,
                   "the metrics document needs every CV run, the test evaluation and the demo predictions")
        return {"synthetic": True, "seed": SEED, "preregistration": "docs/model_selection_rule.md, Addendum P4-1",
                "data": self.data_summary(), "naive_rule_reference": self.naive_reference(),
                "cv": {k: self.cv[k] for k in RUNS}, "selection": self.selection, "final": self.final,
                "test": self.test, "demo": {"rows": int(len(self.demo_predictions)), "metrics": None,
                                            "label": DEMO_LABEL}}

    def run_all(self) -> dict:
        self.build_handoff()
        for run_id in RUNS:
            self.run_cv(run_id)
        self.apply_selection_rule()
        self.fit_final_models()
        self.evaluate_on_test_once()
        self.predict_demo()
        return self.metrics()
