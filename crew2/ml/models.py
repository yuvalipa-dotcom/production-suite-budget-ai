"""The three pre-registered candidates (Addendum P4-1, sections 3-6).

* ``B0``: empirical overrun probability per (stage, position bucket), Laplace-smoothed.
* ``LR`` / ``GB``: a preprocessing + model ``Pipeline``, tuned on explicit group-disjoint
  inner splits and calibrated by a controlled manual Platt procedure: each base pipeline
  is fit on an inner-train side only, its sigmoid calibrator on the held-out side only,
  and the probability is the mean of the calibrated members. ``CalibratedClassifierCV``
  is not used, so no grouping guarantee is assumed from the library.

Every fit is recorded in the ``Trail`` of its unit of work; ``RowSpy`` (the first step of
each pipeline) independently records the rows that actually reach the estimator.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import BOOLEAN, CATEGORICAL, FEATURE_SETS, ROW, model_input, numeric
from .resample import Trail, group_partition, seed_fold

SEED = 42
K_INNER = 3
PLATT_C = 1e12
CANDIDATE_ORDER = ("B0", "LR", "GB")                     # simplest first (tie-break 2)
GRIDS = {
    "B0": [{}],
    "LR": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "GB": [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
           for n, d, lr in itertools.product((100, 300), (2, 3), (0.05, 0.1))],
}
B0_EDGES = (-0.30, -0.15, -0.05, 0.0, 0.05)              # right-closed buckets 0..5
B0_ALPHA = 1.0
STAGES = ("S0", "S1", "S2", "S3")

# ---------------------------------------------------------------- the fit spy

_SPY_SINK: list | None = None


def set_spy_sink(sink: list | None) -> None:
    global _SPY_SINK
    _SPY_SINK = sink


class RowSpy(BaseEstimator, TransformerMixin):
    """Pass-through first pipeline step: records the row ids every ``fit`` receives."""

    def __init__(self, context: str = ""):
        self.context = context

    def fit(self, X, y=None):
        if _SPY_SINK is not None:
            _SPY_SINK.append({"context": self.context, "rows": sorted(int(r) for r in X[ROW])})
        return self

    def transform(self, X):
        return X


# ---------------------------------------------------------------- B0

def b0_bucket(position: np.ndarray, budget: np.ndarray) -> np.ndarray:
    """Right-closed buckets of r = position / budget: bucket i holds edges[i-1] < r <= edges[i]."""
    r = np.asarray(position, dtype=float) / np.asarray(budget, dtype=float)
    return np.searchsorted(np.asarray(B0_EDGES), r, side="left")


class B0Model:
    """p = (k + 1) / (n + 2) per (stage, bucket) cell, counted on the training rows only; 0.5 for an empty cell."""

    kind = "B0"

    def __init__(self):
        self.table: dict[str, list[float]] = {}
        self.counts: dict[str, list[list[int]]] = {}

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "B0Model":
        b = b0_bucket(df["current_position_ils"], df["budget_pre_vat"])
        y = np.asarray(y, dtype=int)
        for s in STAGES:
            k, n = [0] * (len(B0_EDGES) + 1), [0] * (len(B0_EDGES) + 1)
            m = (df["stage"] == s).to_numpy()
            for bucket, label in zip(b[m], y[m]):
                n[bucket] += 1
                k[bucket] += int(label)
            self.counts[s] = [k, n]
            self.table[s] = [(k[i] + B0_ALPHA) / (n[i] + 2 * B0_ALPHA) for i in range(len(n))]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        b = b0_bucket(df["current_position_ils"], df["budget_pre_vat"])
        return np.array([self.table[s][i] for s, i in zip(df["stage"], b)], dtype=float)


# ---------------------------------------------------------------- LR / GB

def base_pipeline(kind: str, params: dict, feature_set: str, context: str) -> Pipeline:
    feats = FEATURE_SETS[feature_set]
    prep = ColumnTransformer(
        [("num", StandardScaler(), list(numeric(feats))),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in CATEGORICAL if c in feats]),
         ("bool", "passthrough", [c for c in BOOLEAN if c in feats])],
        remainder="drop")                                    # drops the row id
    if kind == "LR":
        model = LogisticRegression(max_iter=10000, **params)
    elif kind == "GB":
        model = GradientBoostingClassifier(random_state=SEED, **params)
    else:
        raise KeyError(kind)
    return Pipeline([("spy", RowSpy(context)), ("prep", prep), ("model", model)])


class CalibratedModel:
    """Mean of K_INNER (base pipeline, Platt calibrator) members."""

    def __init__(self, kind: str, feature_set: str, params: dict):
        self.kind, self.feature_set, self.params = kind, feature_set, dict(params)
        self.members: list[tuple[Pipeline, LogisticRegression]] = []

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = model_input(df, self.feature_set)
        out = np.zeros(len(df))
        for pipe, cal in self.members:
            out += cal.predict_proba(pipe.decision_function(X).reshape(-1, 1))[:, 1]
        return out / len(self.members)


def fit_calibrated(kind: str, params: dict, feature_set: str, target: str, train: pd.DataFrame,
                   rows: np.ndarray, seed: list[int], context: str, trail: Trail) -> CalibratedModel:
    """Fit on ``rows`` (a subset of the training partition) with group-disjoint calibration splits."""
    by_row = train.set_index(ROW, drop=False)
    model = CalibratedModel(kind, feature_set, params)
    for j, (tr, ho) in enumerate(group_partition(trail.pid_of_row, rows, K_INNER, seed)):
        ctx = f"{context}/cal{j}"
        trail.split(ctx, rows, tr, ho)
        d_tr, d_ho = by_row.loc[tr], by_row.loc[ho]
        pipe = base_pipeline(kind, params, feature_set, ctx)
        trail.fit(ctx, "base", tr, rows, "train")
        pipe.fit(model_input(d_tr, feature_set), d_tr[target].astype(int).to_numpy())
        y_ho = d_ho[target].astype(int).to_numpy()
        if len(set(y_ho)) < 2:
            raise ValueError(f"{ctx}: calibration side has one class")
        cal = LogisticRegression(C=PLATT_C, max_iter=10000)
        trail.fit(ctx, "calibrator", ho, rows, "held_out")
        cal.fit(pipe.decision_function(model_input(d_ho, feature_set)).reshape(-1, 1), y_ho)
        model.members.append((pipe, cal))
    return model


def fit_candidate(kind: str, feature_set: str, target: str, train: pd.DataFrame, rows: np.ndarray, fold: int,
                  trail: Trail) -> tuple[object, dict, list[dict]]:
    """Tune (LR, GB) on inner group splits of ``rows``, then fit the chosen setting on all of ``rows``.
    ``fold`` is the outer fold (-1 for the final refit on all of train). Returns (model, params, tuning)."""
    by_row = train.set_index(ROW, drop=False)
    f = seed_fold(fold)
    context = f"fold{fold}/{feature_set}/{target}/{kind}"
    if kind == "B0":
        trail.fit(context, "b0", rows, rows, "partition")
        d = by_row.loc[rows]
        return B0Model().fit(d, d[target].astype(int).to_numpy()), {}, []
    tune_splits = group_partition(trail.pid_of_row, rows, K_INNER, [SEED, f, 1])
    for j, (tr, ho) in enumerate(tune_splits):
        trail.split(f"{context}/tune{j}", rows, tr, ho)
    tuning, best = [], None
    for params in GRIDS[kind]:
        scores = []
        for j, (tr, ho) in enumerate(tune_splits):
            m = fit_calibrated(kind, params, feature_set, target, train, tr, [SEED, f, 2, j],
                               f"{context}/tune{j}/{_tag(params)}", trail)
            d_ho = by_row.loc[ho]
            scores.append(brier_score_loss(d_ho[target].astype(int).to_numpy(), m.predict(d_ho)))
        mean = float(np.mean(scores))
        tuning.append({"params": params, "inner_brier": [float(s) for s in scores], "mean_inner_brier": mean})
        if best is None or mean < best[1]:                    # a tie keeps the earlier setting
            best = (params, mean)
    params = best[0]
    model = fit_calibrated(kind, params, feature_set, target, train, rows, [SEED, f, 2, K_INNER],
                           f"{context}/final/{_tag(params)}", trail)
    return model, params, tuning


def _tag(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items()) or "none"


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df)
