"""The pre-registered rules in code (Addendum P4-1 / P4-1a): selection, B0, grids, metrics.
Offline. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from crew2.ml import metrics as MT
from crew2.ml import models as M
from crew2.ml.features import F27, F29, PROPOSED_IN_PRODUCT
from crew2.ml.resample import FINAL_FOLD_CODE, seed_fold
from crew2.ml.selection import apply_rule
from tests.conftest import REPO

ADDENDUM_COMMIT = "dc830a8"
ERRATUM_COMMIT = "aa16b98"


def _m(b0, lr, gb):
    return {"B0": {"brier": b0[0], "average_precision": b0[1]}, "LR": {"brier": lr[0], "average_precision": lr[1]},
            "GB": {"brier": gb[0], "average_precision": gb[1]}}


# ---------------------------------------------------------------- selection rule

def test_a_clear_brier_winner_is_selected():
    r = apply_rule(_m((0.20, 0.9), (0.18, 0.5), (0.19, 0.99)))
    assert r["selected"] == "LR" and r["tie_set"] == ["LR"] and r["decided_by"].startswith("lowest")


def test_a_tie_within_0005_is_broken_by_pr_auc():
    r = apply_rule(_m((0.30, 0.9), (0.180, 0.60), (0.184, 0.70)))
    assert r["tie_set"] == ["LR", "GB"] and r["selected"] == "GB" and r["decided_by"].startswith("tie-break 1")


def test_exactly_0005_is_a_tie_despite_float_rounding():
    assert 0.185 - 0.18 > 0.005                                   # the float artefact the rule guards against
    r = apply_rule(_m((0.185, 0.9), (0.18, 0.5), (0.3, 0.1)))
    assert r["tie_set"] == ["B0", "LR"] and r["selected"] == "B0"
    r = apply_rule(_m((0.1851, 0.9), (0.18, 0.5), (0.3, 0.1)))
    assert r["tie_set"] == ["LR"]


def test_equal_pr_auc_goes_to_the_simpler_model():
    r = apply_rule(_m((0.300, 0.5), (0.200, 0.7000001), (0.201, 0.7000004)))
    assert r["tie_set"] == ["LR", "GB"] and r["selected"] == "LR" and r["decided_by"].startswith("tie-break 2")


def test_three_candidate_ties_are_measured_from_the_best():
    """A chain (GB within 0.005 of LR, LR within 0.005 of B0) does not tie GB to B0."""
    r = apply_rule(_m((0.200, 0.1), (0.204, 0.2), (0.208, 0.9)))
    assert r["tie_set"] == ["B0", "LR"] and r["selected"] == "LR"
    r = apply_rule(_m((0.201, 0.5), (0.200, 0.5), (0.203, 0.5)))
    assert r["tie_set"] == ["B0", "LR", "GB"] and r["selected"] == "B0"


def test_the_rule_requires_exactly_the_three_candidates():
    with pytest.raises(ValueError):
        apply_rule({"LR": {"brier": 0.1, "average_precision": 0.5}})


# ---------------------------------------------------------------- B0

def test_b0_buckets_are_right_closed_with_the_registered_edges():
    budget = np.full(9, 100.0)
    pos = np.array([-50, -30, -29.9, -15, -5, 0, 0.01, 5, 6])
    assert M.b0_bucket(pos, budget).tolist() == [0, 0, 1, 1, 2, 3, 4, 4, 5]
    assert M.B0_EDGES == (-0.30, -0.15, -0.05, 0.0, 0.05) and M.B0_ALPHA == 1.0


def test_b0_is_laplace_smoothed_per_stage_and_bucket():
    df = pd.DataFrame({"stage": ["S0"] * 4 + ["S3"] * 2, "current_position_ils": [-90, -90, -90, 10, 10, 10],
                       "budget_pre_vat": [100.0] * 6})
    m = M.B0Model().fit(df, np.array([1, 0, 0, 1, 1, 1]))
    assert m.table["S0"][0] == (1 + 1) / (3 + 2) and m.table["S0"][5] == (1 + 1) / (1 + 2)
    assert m.table["S3"][5] == (2 + 1) / (2 + 2) and m.table["S1"] == [0.5] * 6       # empty cells
    assert m.predict(df.iloc[[0, 3]]).tolist() == [0.4, 2 / 3]


# ---------------------------------------------------------------- grids, sets, seeds

def test_grids_and_feature_sets_equal_the_addendum():
    assert M.GRIDS["B0"] == [{}]
    assert M.GRIDS["LR"] == [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
    assert M.GRIDS["GB"] == [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
                             for n in (100, 300) for d in (2, 3) for lr in (0.05, 0.1)]
    assert M.CANDIDATE_ORDER == ("B0", "LR", "GB") and M.K_INNER == 3 and M.SEED == 42
    assert len(F27) == 27 and set(F29) - set(F27) == set(PROPOSED_IN_PRODUCT)
    assert seed_fold(-1) == FINAL_FOLD_CODE == 5 and seed_fold(3) == 3
    text = (REPO / "docs" / "model_selection_rule.md").read_text(encoding="utf-8")
    for s in ("`C` ∈ (0.01, 0.1, 1.0, 10.0)", "`n_estimators` ∈ (100, 300)", "**α = 1**", "`r ≤ −0.30`",
              "b_c − b* ≤ 0.005", "**p ≥ 0.5**", "fold code **5**"):
        assert s in text, s


def test_the_addendum_and_erratum_were_committed_alone_before_any_crew2_code():
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

    try:
        for c in (ADDENDUM_COMMIT, ERRATUM_COMMIT):
            assert git("show", "--name-only", "--format=", c).splitlines() == ["docs/model_selection_rule.md"]
        first_code = git("log", "--reverse", "--format=%h", "--", "crew2").splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history unavailable")
    if first_code:
        assert git("merge-base", "--is-ancestor", ERRATUM_COMMIT, first_code[0]) == ""


# ---------------------------------------------------------------- metrics

def test_metric_definitions():
    y, p = [0, 1, 1, 0, 1], [0.1, 0.8, 0.4, 0.6, 0.9]
    s = MT.scores(y, p)
    from sklearn.metrics import average_precision_score, brier_score_loss
    assert s["brier"] == brier_score_loss(y, p) and s["average_precision"] == average_precision_score(y, p)
    assert MT.scores([0, 0], [0.2, 0.7])["roc_auc"] is None and MT.scores([0, 0], [0.2, 0.7])["average_precision"] is None
    rel = MT.reliability([0, 1, 1], [0.05, 0.95, 1.0])
    assert rel["bins"][9]["n"] == 2 and abs(rel["ece"] - (1 / 3 * 0.05 + 2 / 3 * 0.025)) < 1e-12


def test_bootstrap_is_production_level_and_deterministic():
    df = pd.DataFrame({"production_id": [f"P{i // 2}" for i in range(40)], "y": [i % 3 == 0 for i in range(40)],
                       "p": np.linspace(0.05, 0.95, 40)})
    a, b = MT.bootstrap_ci(df, "y", "p", reps=200), MT.bootstrap_ci(df, "y", "p", reps=200)
    assert a == b and a["productions"] == 20 and a["brier"]["lo"] <= MT.scores(df["y"], df["p"])["brier"] <= a["brier"]["hi"]
    assert a["level"] == 0.95 and a["reps"] == 200
