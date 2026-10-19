"""Metrics (Addendum P4-1, section 7). The 0.5 threshold is fixed and never chosen on data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

THRESHOLD = 0.5
N_BINS = 10
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42
METRICS = ("brier", "log_loss", "roc_auc", "average_precision", "precision", "recall", "accuracy")
CI_METRICS = ("brier", "log_loss", "roc_auc", "average_precision")


def _r(x) -> float | None:
    return None if x is None else float(x)


def scores(y, p) -> dict:
    """Every metric of one set of rows; a metric that is undefined there is None."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    both = len(set(y.tolist())) == 2
    pred = (p >= THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": _r(y.mean()) if len(y) else None,
        "brier": _r(brier_score_loss(y, p)),
        "log_loss": _r(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "roc_auc": _r(roc_auc_score(y, p)) if both else None,
        "average_precision": _r(average_precision_score(y, p)) if y.sum() > 0 else None,
        "precision": _r(tp / pred.sum()) if pred.sum() else None,
        "recall": _r(tp / y.sum()) if y.sum() else None,
        "accuracy": _r((pred == y).mean()),
    }


def reliability(y, p, n_bins: int = N_BINS) -> dict:
    """10 equal-width bins on [0, 1] (the last bin includes 1.0) and the expected calibration error."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n:
            ece += n / len(y) * abs(p[m].mean() - y[m].mean())
        bins.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n,
                     "mean_p": _r(p[m].mean()) if n else None, "observed_rate": _r(y[m].mean()) if n else None})
    return {"bins": bins, "ece": _r(ece)}


def by_stage(df: pd.DataFrame, y_col: str, p_col: str) -> dict:
    return {s: scores(g[y_col], g[p_col]) for s, g in sorted(df.groupby("stage"))}


def bootstrap_ci(df: pd.DataFrame, y_col: str, p_col: str, reps: int = BOOTSTRAP_REPS,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """Production-level bootstrap: resample productions with replacement; percentile 95% interval."""
    groups = {pid: g for pid, g in df.groupby("production_id", sort=True)}
    pids = sorted(groups)
    ys = {k: groups[k][y_col].astype(int).to_numpy() for k in pids}
    ps = {k: groups[k][p_col].astype(float).to_numpy() for k in pids}
    rng = np.random.default_rng(seed)
    values = {m: [] for m in CI_METRICS}
    skipped = {m: 0 for m in CI_METRICS}
    for _ in range(reps):
        draw = rng.integers(0, len(pids), len(pids))
        y = np.concatenate([ys[pids[i]] for i in draw])
        p = np.concatenate([ps[pids[i]] for i in draw])
        s = scores(y, p)
        for m in CI_METRICS:
            if s[m] is None:
                skipped[m] += 1
            else:
                values[m].append(s[m])
    return {m: {"lo": _r(np.percentile(values[m], 2.5)), "hi": _r(np.percentile(values[m], 97.5)),
                "resamples_used": len(values[m]), "resamples_skipped": skipped[m]} for m in CI_METRICS} | {
        "reps": reps, "seed": seed, "productions": len(pids), "level": 0.95}


def fold_summary(fold_scores: list[dict]) -> dict:
    """Unweighted mean and sd across folds; folds where a metric is undefined are left out and counted."""
    out = {}
    for m in METRICS:
        vals = [f[m] for f in fold_scores if f[m] is not None]
        out[m] = {"mean": _r(np.mean(vals)) if vals else None, "sd": _r(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                  "folds_used": len(vals), "folds_undefined": len(fold_scores) - len(vals)}
    return out
