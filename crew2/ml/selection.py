"""The pre-registered selection rule (Addendum P4-1, section 8), applied by code only.

No agent and no LLM can choose, override or re-rank: the CrewAI tool that exposes this
takes no argument, and validator W4 recomputes the choice from metrics.json.
"""

from __future__ import annotations

from .models import CANDIDATE_ORDER

BRIER_TIE = 0.005
AP_DECIMALS = 6


def apply_rule(means: dict[str, dict[str, float]]) -> dict:
    """``means[c] = {"brier": mean grouped-CV Brier, "average_precision": mean grouped-CV PR-AUC}``."""
    if set(means) != set(CANDIDATE_ORDER):
        raise ValueError(f"the rule needs exactly the candidates {CANDIDATE_ORDER}")
    best = min(means[c]["brier"] for c in CANDIDATE_ORDER)
    # "0.005 or less" in exact decimal terms: the difference is rounded to 12 places first, so a float
    # artefact (0.185 - 0.18 = 0.0050000000000000044) cannot push an exact 0.005 out of the tie set.
    tie_set = [c for c in CANDIDATE_ORDER if round(means[c]["brier"] - best, 12) <= BRIER_TIE]
    trace = {"best_mean_brier": best, "tie_margin": BRIER_TIE, "tie_set": tie_set,
             "brier_minus_best": {c: means[c]["brier"] - best for c in CANDIDATE_ORDER}}
    if len(tie_set) == 1:
        return trace | {"selected": tie_set[0], "decided_by": "lowest mean grouped-CV Brier"}
    top_ap = max(round(means[c]["average_precision"], AP_DECIMALS) for c in tie_set)
    ap_best = [c for c in tie_set if round(means[c]["average_precision"], AP_DECIMALS) == top_ap]
    trace |= {"tie_break_1_highest_mean_average_precision": ap_best}
    if len(ap_best) == 1:
        return trace | {"selected": ap_best[0], "decided_by": "tie-break 1: higher mean PR-AUC (average_precision_score)"}
    simplest = min(ap_best, key=CANDIDATE_ORDER.index)
    return trace | {"selected": simplest, "decided_by": "tie-break 2: simpler model (B0 < LR < GB)"}
