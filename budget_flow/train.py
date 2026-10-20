"""Crew 2's TRAIN-only execution in the Flow (P5-C). ALL DATA IS SYNTHETIC.

    python -m budget_flow.train --clean-data PATH --contract PATH --out DIR [--flow-contract PATH] [--n-jobs N]

Input: only the validated file handoff of P5-B (``handoff.load_handoff``): Crew 1's ``clean_data.csv`` and
``dataset_contract.json``, every non-train label redacted, the test vault sealed.

Protocol: the P4 session (``crew2.ml.protocol``) unchanged: the same 9 pre-registered grouped-CV runs, seeds,
grids, features and selection rule, then the final refit on all of TRAIN. Every step that would touch TEST
(``evaluate_on_test_once``, ``run_all``, the P4 ``metrics`` document) is refused, and even the base-class
method would hit the ``SealedVault``. The historical P4 TEST result is cited from the pinned reference.

Before any fitting, the pre-registration is compared with the approved P4 protocol; after fitting, the results
are compared with the approved P4 CV (``p4_reference.verify``). Any difference in the selected model, its
parameters or the decision rule is a hard failure, and nothing is written.

Output (``outputs.write_all``): ``features.csv``, ``model.joblib``, ``evaluation_report.md``, ``model_card.md``,
plus ``features.schema.json`` and ``run_manifest.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from crew2.ml import protocol as P

from . import contract as C
from . import handoff as FH
from . import outputs as O
from . import p4_reference as R

PROTECTED_OUTPUT_DIRS = (C.REPO / "artifacts", C.REPO / "out" / "crew2", C.REPO / "data", C.REPO / "contracts",
                         C.CONTRACTS)


class FlowPathError(RuntimeError):
    """A step outside the Flow's file-based, TRAIN-only path was requested."""


class TrainOnlySession(P.Session):
    """The P4 session on the validated Flow handoff, with every TEST step refused."""

    def __init__(self, handoff: FH.FlowHandoff, n_jobs: int = 1):
        if not handoff.report.ok:
            raise C.ContractViolationError(handoff.report)
        if not isinstance(handoff.parts.vault, FH.SealedVault):
            raise FH.TestAccessError("the Flow runs only on a handoff whose test vault is sealed")
        super().__init__(n_jobs=n_jobs, raw_dir=None, contracts_dir=None,
                         expected_sha256=handoff.report.clean_data_sha256)
        self.parts = handoff.parts
        self.handoff_info = {"source": "Crew 1 file handoff, validated by the Flow contract flow-1",
                             "clean_data_sha256": handoff.report.clean_data_sha256,
                             "rows": sum(c["rows"] for c in handoff.counts.values())}

    def build_handoff(self):
        raise FlowPathError("the Flow never rebuilds the handoff from the raw export; it uses load_handoff")

    def evaluate_on_test_once(self):
        raise FH.TestAccessError("the Flow never evaluates on TEST; the historical P4 TEST result is cited")

    def run_all(self):
        raise FH.TestAccessError("run_all evaluates on TEST; the Flow uses run_train_only")

    def metrics(self):
        raise FH.TestAccessError("the P4 metrics document needs a TEST evaluation; the Flow uses train_only_metrics")

    def predict_demo(self):
        raise FlowPathError("demo inference is not part of the TRAIN-only Flow step")

    def run_train_only(self, log=None) -> dict:
        timings = {}
        for run_id in P.RUNS:
            t = time.monotonic()
            self.run_cv(run_id)
            timings[run_id] = round(time.monotonic() - t, 1)
            if log:
                log(f"grouped CV {run_id}: {timings[run_id]} s")
        self.apply_selection_rule()
        t = time.monotonic()
        self.fit_final_models()
        timings["final_refit"] = round(time.monotonic() - t, 1)
        if log:
            log(f"selected {self.selection['selected']}; final refit {timings['final_refit']} s")
        self.step_seconds = timings
        return self.train_only_metrics()

    def train_only_metrics(self) -> dict:
        self._need(len(self.cv) == len(P.RUNS) and self.final is not None,
                   "the TRAIN-only metrics need every CV run and the final models")
        return {"synthetic": True, "seed": P.SEED, "preregistration": "docs/model_selection_rule.md, Addendum P4-1",
                "data": self.data_summary(), "naive_rule_reference": self.naive_reference(),
                "cv": {k: self.cv[k] for k in P.RUNS}, "selection": self.selection, "final": self.final,
                "test": None}


def _check_out_dir(out_dir: Path) -> Path:
    out = Path(os.path.realpath(out_dir))
    for d in PROTECTED_OUTPUT_DIRS:
        d = Path(os.path.realpath(d))
        if out == d or d in out.parents:
            raise FlowPathError(f"refusing to write inside a protected folder: {d.name}")
    if out.exists() and any(out.iterdir()):
        raise FlowPathError("the output folder exists and is not empty; historical outputs are never overwritten")
    return out


def run_flow_train(clean_csv: Path, contract_json: Path, out_dir: Path, flow_contract_json: Path | None = None,
                   n_jobs: int = 1, log=None) -> dict:
    """The TRAIN-only Crew 2 step of the Flow. Returns the run manifest; raises on any failed check."""
    out = _check_out_dir(out_dir)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    ref, pred_ref = R.load()
    R.check_preregistration(ref)                                   # before any fitting
    handoff = FH.load_handoff(clean_csv, contract_json, flow_contract_json)
    session = TrainOnlySession(handoff, n_jobs=n_jobs)
    metrics = session.run_train_only(log=log)
    predictions = R._prediction_rows(session)
    env = R.environment()
    repro = R.verify(metrics["cv"], metrics["selection"], metrics["final"], predictions, ref, pred_ref,
                     R.mode_for(env, ref))
    if session.parts.vault.opened or session.test is not None:
        raise FH.TestAccessError("the test vault was touched")
    fit_seconds = round(time.monotonic() - t0, 1)
    manifest = O.write_all(out, session, metrics, ref, repro, handoff,
                           inputs={"clean_data.csv": Path(clean_csv), "dataset_contract.json": Path(contract_json),
                                   **({"flow_contract.json": Path(flow_contract_json)} if flow_contract_json else {})},
                           timing={"started_at": started.isoformat(timespec="seconds"),
                                   "train_only_seconds": fit_seconds, "steps_seconds": session.step_seconds,
                                   "n_jobs": n_jobs})
    if log:
        log(f"TRAIN-only Flow step finished in {manifest['timing']['total_seconds']} s ({repro['mode']} reproduction)")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean-data", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--flow-contract", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args(argv)
    m = run_flow_train(args.clean_data, args.contract, args.out, args.flow_contract, args.n_jobs,
                       log=lambda s: print(s, flush=True))
    print(f"selected {m['selected_model']['candidate']} {m['selected_model']['params']}; "
          f"artifacts in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
