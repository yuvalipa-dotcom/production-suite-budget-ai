"""The offline stand-in for the LLM (no model call). ALL DATA IS SYNTHETIC.

``scripted_runner`` drives the REAL agent tools, through their CrewAI wrappers, in task order,
with narrative text built from the computed metrics. It is what ``python -m crew2.run --offline``
and the end-to-end tests use. A live run (real agents) needs separate owner approval.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from .ml.protocol import RUNS
from .tools import ToolContext, build_tools
from .tools.workspace import CHART_IDS


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def card_sections(m: dict) -> dict[str, str]:
    sel, fin, t = m["selection"], m["final"], m["test"]
    p, s = t["primary"], t["secondary"]
    po, pci, so = p["overall"], p["bootstrap_ci"], s["overall"]
    cv = m["cv"]
    prim = {r["candidate"]: r["summary"] for r in cv.values() if r["role"] == "primary"}
    res = {r["candidate"]: r["summary"] for r in cv.values() if r["role"] == "research"}
    c = m["data"]["counts"]
    tp = m["data"]["train_prevalence"]["is_overrun"]["overall"]
    params = fin["primary"]["params"]
    run = cv[f"F27.is_overrun.{sel['selected']}"]
    if "C" in params:
        top = max(g["C"] for g in run["grid"])
        who = "Every fold and the final refit" if all(f["params"]["C"] == params["C"] for f in run["folds"])             else "The final refit"
        boundary = (f" {who} chose C = {params['C']:g}, which is the upper boundary of the pre-registered grid; a "
                    f"larger C was never evaluated." if params["C"] == top else
                    f" The final refit chose C = {params['C']:g}.")
    else:
        boundary = ""
    return {
        "summary": f"This card describes a {sel['selected']} model of the probability that a synthetic production "
                   f"ends above its budget in effect at the snapshot. On the single test evaluation its Brier score "
                   f"is {_f(po['brier'])} and its PR-AUC is {_f(po['average_precision'])}. All data are synthetic and "
                   f"every output is an experimental / demo prediction.",
        "intended_use": "The model supports an academic demonstration of the Production Suite budget card. It is not "
                        "intended for decisions about real productions, and it is not validated on real data.",
        "data_and_target": f"The input is the Policy C handoff of {m['data']['handoff']['rows']:,} eligible snapshots. "
                           f"Training uses the {c['train']['rows']:,} train rows only; test has {c['test']['rows']} "
                           f"rows and demo {c['demo']['rows']} rows. The target is_overrun is a rounding-tolerance "
                           f"label and not a commercially material overrun.",
        "features": "The primary set has 27 snapshot features, each taken as of the snapshot date from the dated "
                    "ledgers. Of these, 24 correspond to fields the product stores today or are derived from them, "
                    "and three are only approximations there: lead_time_days and days_since_contract use the event "
                    "creation date as a proxy for the contract date, and n_substitutions_to_date counts deleted "
                    "quotes, which are not always substitutions. The 29-feature set adds two fields the product "
                    "cannot compute today and is research only.",
        "protocol": "Tuning and calibration ran inside each of the 5 frozen production-grouped folds, on explicit "
                    "group-disjoint inner splits of the training partition only. Test was opened once, after the "
                    "final models were frozen, and demo was used for inference only.",
        "cv_results": f"On grouped CV the mean Brier is {_f(prim['B0']['brier']['mean'])} for B0, "
                      f"{_f(prim['LR']['brier']['mean'])} for LR and {_f(prim['GB']['brier']['mean'])} for GB. The "
                      f"research 29-feature runs score {_f(res['LR']['brier']['mean'])} for LR and "
                      f"{_f(res['GB']['brier']['mean'])} for GB, and were never used for selection.",
        "selection": f"The pre-registered rule, applied by code, selected {sel['selected']}: {sel['decided_by']}. "
                     f"The best mean Brier was {_f(sel['best_mean_brier'])} and the tie margin is 0.005.{boundary}",
        "test_results": f"On test ({t['rows']} rows from {t['productions']} productions) the Brier score is "
                        f"{_f(po['brier'])} with a 95% bootstrap interval of [{_f(pci['brier']['lo'])}, "
                        f"{_f(pci['brier']['hi'])}], and PR-AUC is {_f(po['average_precision'])} "
                        f"[{_f(pci['average_precision']['lo'])}, {_f(pci['average_precision']['hi'])}]. The expected "
                        f"calibration error on test is approximately {p['reliability']['ece']:.3f} (10 bins), so the "
                        f"probabilities are not well calibrated at the test prevalence. This is the only test "
                        f"evaluation.",
        "secondary_results": f"For the 2% material overrun the same candidate type scores a test Brier of "
                             f"{_f(so['brier'])} and a PR-AUC of {_f(so['average_precision'])}. It is reported only and "
                             f"never affected selection.",
        "limitations": f"All data are synthetic, so nothing here is evidence of real-world accuracy. The test group "
                       f"has an overrun prevalence of {100 * po['prevalence']:.1f}% against {100 * tp:.1f}% on "
                       f"train; the split is frozen and was not changed, and the model was neither recalibrated nor "
                       f"retuned on test.{boundary} Three of the {fin['primary']['candidate']} model's inputs are "
                       f"approximations in the product, so its behaviour there is untested.",
    }


def scripted_runner(sections=None, extra_test_call: bool = False, corrupt=None):
    """An offline runner: real tools, scripted narrative, no LLM. ``corrupt(ctx)`` runs before the card."""

    def run(ctx: ToolContext, model, recorder):
        tools = {t.name: t for ts in build_tools(ctx, recorder).values() for t in ts}
        tools["build_handoff"]._run()
        tools["feature_availability"]._run()
        tools["run_leakage_checks"]._run()
        recorder.task(SimpleNamespace(name="prepare_and_prove_features", agent="Feature & Leakage Engineer",
                                      raw="scripted"))
        tools["list_runs"]._run()
        for run_id in RUNS:
            tools["run_grouped_cv"].run(run_id=run_id)          # validated, as CrewAI calls it
        tools["apply_selection_rule"]._run()
        tools["fit_final_models"]._run()
        recorder.task(SimpleNamespace(name="train_and_select", agent="Model Trainer", raw="scripted"))
        tools["evaluate_on_test_once"]._run()
        if extra_test_call:
            tools["evaluate_on_test_once"]._run()
        tools["predict_demo"]._run()
        for c in CHART_IDS:
            tools["render_chart"].run(chart_id=c)
        if corrupt:
            corrupt(ctx)
        m = json.loads(ctx.ws.read_output("out.metrics"))
        tools["write_model_card"].run(**(sections or card_sections)(m))
        recorder.task(SimpleNamespace(name="evaluate_and_document", agent="Evaluator & Model Card Author",
                                      raw="scripted"))
        return {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "successful_requests": 0}}

    return run
