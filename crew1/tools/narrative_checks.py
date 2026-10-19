"""The authoritative narrative facts for the Crew 1 agents, and the deterministic rules for their text.

* ``narrative_facts``: the target definitions (P2 contract: docs/data_generation.md, gate/reconstruct.py),
  the eligible population and production counts (from the cleaning log), what kind of statement each
  finding allows, and which analysis each figure comes from. The agents receive it from list_analyses
  and build_dataset_contract; ``DEFINITIONS_TEXT`` is also in their instructions, verbatim.
* ``frozen_split_values``: the frozen split difference and interval from stats.json, with units.
* ``findings``: statements that contradict the facts. The rules in ``BLOCKING`` fail V6 (EDA report)
  and V7 (insights) in crew1/validators.py; every rule is fed back to the agent by the authoring tools
  (``context_problems``). They were written for the errors of p3-live-003 and p3-live-004:
    N-DEFINITION  the signed gap or an overrun described through payments or commitments;
    N-SIGN        a positive gap called under budget, or a negative gap called over budget;
    N-POPULATION  another production count attached to the eligible snapshots (375 or 400, not 397);
    N-CAUSAL      a cause, effect or explanation asserted for revisions, stages, fees or cost outcomes;
    N-ATTRIBUTION a finding credited to the wrong source (the frozen split figures to the handoff gate,
                  the documented sampling-outcome assessment to an analysis);
    N-EXCLUSION   an exclusion reason the cleaning log does not record (feedback only).
"""

from __future__ import annotations

import re

from gate.reconstruct import OVERRUN_TOLERANCE, WARNING_THRESHOLD

from .eda import production_eligibility

BLOCKING = ("N-DEFINITION", "N-SIGN", "N-POPULATION", "N-CAUSAL", "N-ATTRIBUTION")

DEFINITIONS = {
    "signed_gap": "the verified final cost plus the contractual fee when it is included in the budget, minus the "
                  "approved budget in effect at the snapshot (ILS, pre-VAT)",
    "sign": "a positive gap means over budget; a negative gap means under budget",
    "is_overrun": f"the signed gap exceeds {OVERRUN_TOLERANCE:g} ILS",
    "is_material_overrun": f"the signed gap exceeds {WARNING_THRESHOLD:.0%} of that budget (a warning level, "
                           "not the overrun definition)",
    "not_part_of_the_gap": "payments and commitment totals are not part of the signed gap or of either label",
    "source": "P2 contract: docs/data_generation.md (target definition), gate/reconstruct.py (thresholds)",
}
DEFINITIONS_TEXT = (
    "Authoritative definitions (P2 contract; copy them, never infer them): the signed gap is "
    f"{DEFINITIONS['signed_gap']}. A positive gap means over budget; a negative gap means under budget. "
    f"is_overrun means {DEFINITIONS['is_overrun']}; is_material_overrun means the signed gap exceeds "
    f"{WARNING_THRESHOLD:.0%} of that budget. Payments and commitment totals are not part of the signed gap "
    "or of either label.")
STATEMENT_KINDS = {
    "fact": "the definitions and counts in narrative_facts; state them exactly",
    "association": "differences between stages, groups, revision buckets or fee terms, and correlations; describe "
                   "them as associations, never as causes, effects, impacts or explanations",
    "hypothesis": "questions for Crew 2 to test, written as questions",
}
ATTRIBUTION = {
    "overrun prevalence intervals": "overrun_prevalence (Wilson 95% intervals)",
    "frozen split difference and its interval": "frozen_split_prevalence_difference in stats.json "
                                                "(production-level bootstrap), not the handoff gate",
    "split is frozen, not tuned, a sampling outcome": "the documented decision when the split was frozen, not a "
                                                      "result of any analysis",
    "correlations": "feature_correlations (train group only): Spearman with the signed gap, point-biserial "
                    "with is_overrun",
    "exclusions": "policy_c_exclusions and the cleaning log (Policy C)",
}

SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
NEGATION = re.compile(r"\b(?:no|not|never|without|nor|cannot|neither|none)\b|n't\b", re.I)
PRODUCTIONS = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)\s+(?:[A-Za-z-]+\s+){0,3}productions\b", re.I)
FROZEN_FIGURE = re.compile(r"bootstrap|\binterval\b|test\s+minus\s+train|prevalence\s+difference|"
                           r"frozen_split_prevalence_difference", re.I)
GATE = re.compile(r"\bgate\b", re.I)
ASSESSMENT_CREDITED = re.compile(
    r"(?:sampling\s+outcome|independent\s+of\s+the\s+outcome\s+mechanism|not\s+(?:been\s+)?tuned)[^.;]*?"
    r"\b(?:via|by|from|according\s+to|shown\s+by|based\s+on)\s+(?:the\s+)?"
    r"(?:frozen_split_prevalence_difference|stats\.json|bootstrap|analysis|handoff\s+gate|gate)", re.I)
GAP_OR_LABEL = re.compile(r"\boverruns?\b|\bunderruns?\b|\bgaps?\b|\bis_overrun\b", re.I)
PAYMENT = re.compile(r"\bpayments?\b", re.I)
COMMITMENT_DEFINED = re.compile(r"\bgaps?\s*\(\s*commitments?|\bcommitments?\s+minus\b|\bminus\s+commitments?\b",
                                re.I)
GAP_CLAUSE = re.compile(r"\bgaps?\b", re.I)
SIGNED_AMOUNT = re.compile(r"(?:(?<=[\s(\[:,=])|^)([−-]?)(\d[\d,]*(?:\.\d+)?)\s*(?:ILS|NIS|%)")
SIGN_WORD = re.compile(r"\b(negative|positive)\s+(?:(?:mean|median|signed)\s+)*gaps?\b", re.I)
OVER_BUDGET = re.compile(r"\bover[\s-]budget\b|\bunder-?budget(?:ing|ed)\b|\bover-?spen\w*|\bcost\s+overruns?\b|"
                         r"\bexceed\w*\s+(?:the\s+|its\s+|their\s+)?budgets?\b", re.I)
UNDER_BUDGET = re.compile(r"\bunder[\s-]budget\b(?!ing|ed)|\bover-?budget(?:ing|ed)\b|\bunder-?spen\w*|"
                          r"\bunder-?runs?\b|\bbelow\s+(?:the\s+|its\s+|their\s+)?budgets?\b|"
                          r"\bwithin\s+(?:the\s+|its\s+|their\s+)?budgets?\b", re.I)
CAUSAL = re.compile(r"\b(?:reflect(?:s|ed|ing)?|due\s+to|because|caus(?:e|es|ed|ing)|lead(?:s|ing)?\s+to|led\s+to|"
                    r"result(?:s|ed|ing)?\s+in|driv(?:e|es|en|ing)|reduc(?:e|es|ed|ing)|"
                    r"(?:lower|rais|increas|decreas)(?:e|es|ed|ing|s)?\s+the|impacts?\s+(?:of|on)|"
                    r"effects?\s+(?:of|on)|influenc(?:e|es|ed|ing)|as\s+a\s+result)\b", re.I)
CAUSAL_TOPIC = re.compile(r"revision|\bstages?\b|\bS[0-3]\b|\bfees?\b|cost\s+outcomes?|final\s+costs?|"
                          r"as\s+(?:events|productions)\s+(?:approach|progress)|over\s+time|overrun|\bgaps?\b", re.I)
HYPOTHESIS = re.compile(r"\bwhether\b|\bhypothes\w*", re.I)
REASON = re.compile(r"\bbecause\b|\bdue\s+to\b", re.I)   # an exclusion's reason: N-EXCLUSION checks it
EXCLUDED = re.compile(r"exclu|invalid|ineligib|removed|dropped", re.I)

# Reason wordings and the recorded problem codes that support them; None: no tool output records it.
REASONS = (
    ("out of range", re.compile(r"out[- ]of[- ]range|outside\s+(?:the\s+)?(?:valid\s+|allowed\s+)?range", re.I),
     lambda code: code.startswith("out_of_range")),
    ("missing value", re.compile(r"\bmissing\b|\bblank\b|\bnull\b", re.I), lambda code: "missing" in code),
    ("undated", re.compile(r"\bundated\b|without\s+(?:a\s+)?(?:creation\s+)?date|no\s+(?:creation\s+)?date", re.I),
     lambda code: "undated" in code or code.endswith("_date")),
    ("duplicate", re.compile(r"duplicat", re.I), lambda code: "duplicate" in code),
    ("unmatched reference", re.compile(r"no\s+matching|unmatched|orphan|unknown\s+(?:commitment|supplier)|"
                                       r"without\s+(?:a\s+)?(?:matching\s+)?commitment", re.I), None),
    ("currency", re.compile(r"\bcurrenc", re.I), None),
    ("date order", re.compile(r"future[- ]dated|dated\s+after|paid\s+(?:before|after)|date\s+mismatch", re.I), None),
    ("error or fraud", re.compile(r"\bfraud|entry\s+error|\btypo", re.I), None),
)


def _fmt(n: int) -> set[str]:
    return {f"{n:,}", str(n)}


def frozen_split_values(stats: dict | None) -> dict | None:
    """The three frozen split values exactly as stats.json holds them, and how to write them."""
    a = ((stats or {}).get("analyses") or {}).get("frozen_split_prevalence_difference")
    if not a:
        return None
    pp, lo, hi = a["test_minus_train_pp"], a["test_minus_train_lo_pp"], a["test_minus_train_hi_pp"]
    return {"source": "stats.json, analysis frozen_split_prevalence_difference (not the handoff gate)",
            "method": a.get("method"), "unit": "pp (percentage points)",
            "test_minus_train_pp": pp, "test_minus_train_lo_pp": lo, "test_minus_train_hi_pp": hi,
            "write_as": f"{pp:+.1f} pp, 95% bootstrap interval [{lo:+.1f}, {hi:+.1f}] pp",
            "eligible_snapshots_per_group": {g: v.get("n") for g, v in (a.get("groups") or {}).items()}}


def _recorded_reasons(log: dict | None, contract: dict | None) -> dict[str, set[str]]:
    log = log or {}
    payments = {p for i in log.get("issues") or [] if i.get("table") == "payments" for p in i.get("problems", [])}
    raw, clean = log.get("raw_rows") or {}, log.get("clean_rows") or {}
    if raw.get("payments", 0) - clean.get("payments", 0) - len(log.get("excluded_payments") or []) > 0:
        payments.add("exact_duplicate")
    snapshots = set(log.get("exclusion_counts") or {}) | set((contract or {}).get("exclusion_counts") or {})
    return {"payments": payments, "snapshots": snapshots}


def _population(log: dict | None, stats: dict | None, frozen_counts: dict | None) -> dict | None:
    if log and log.get("ineligible_snapshots") is not None and log.get("raw_rows"):
        p = production_eligibility(log)
        return {"eligible_snapshots": log["snapshots"]["eligible"],
                "scheduled_snapshots": log["snapshots"]["scheduled"],
                "productions_scheduled": p["n_productions_scheduled"],
                "productions_with_eligible_snapshots": p["n_productions_with_eligible_snapshots"],
                "productions_fully_eligible": p["n_productions_fully_eligible"],
                "productions_partially_eligible": p["n_productions_partially_eligible"],
                "productions_fully_excluded": p["n_productions_fully_excluded"]}
    analyses = (stats or {}).get("analyses") or {}
    x = analyses.get("policy_c_exclusions")
    if "descriptive_statistics" in analyses and x:
        return {"eligible_snapshots": x["eligible"],
                "productions_with_eligible_snapshots": analyses["descriptive_statistics"].get("productions")}
    if x and frozen_counts and isinstance(frozen_counts.get("productions"), int):
        return {"eligible_snapshots": x["eligible"],
                "productions_with_eligible_snapshots": frozen_counts["productions"] - x["n_productions_fully_excluded"]}
    return None


def narrative_facts(log: dict | None = None, contract: dict | None = None, stats: dict | None = None,
                    frozen_counts: dict | None = None) -> dict:
    """The facts the narrative must state exactly, the statement kinds, and the source of each figure."""
    pop = _population(log, stats, frozen_counts)
    if pop and contract and isinstance(contract.get("row_count"), int):
        pop["eligible_snapshots"] = contract["row_count"]
    if pop and "productions_partially_eligible" in pop:
        pop["sentence"] = (f"The {pop['eligible_snapshots']:,} eligible snapshots come from "
                           f"{pop['productions_with_eligible_snapshots']} productions: "
                           f"{pop['productions_fully_eligible']} fully eligible and "
                           f"{pop['productions_partially_eligible']} partially eligible "
                           f"({pop['productions_fully_excluded']} of the {pop['productions_scheduled']} scheduled "
                           "productions have no eligible snapshot).")
    return {"definitions": DEFINITIONS, "definitions_text": DEFINITIONS_TEXT, "population": pop,
            "statement_kinds": STATEMENT_KINDS, "attribution": ATTRIBUTION,
            "recorded_exclusion_reasons": {k: sorted(v) for k, v in _recorded_reasons(log, contract).items()}}


def _gap_signs(clause: str) -> set[str]:
    signs = {"negative" if m.group(1) else "positive" for m in SIGNED_AMOUNT.finditer(clause)}
    signs |= {m.group(1).lower() for m in SIGN_WORD.finditer(clause)}
    return signs


def findings(text: str, facts: dict) -> list[tuple[str, str]]:
    """(rule, message) for every statement that contradicts the facts, in text order."""
    out: list[tuple[str, str]] = []
    pop = facts.get("population") or {}
    rows, productions = pop.get("eligible_snapshots"), pop.get("productions_with_eligible_snapshots")
    recorded = {k: set(v) for k, v in (facts.get("recorded_exclusion_reasons") or {}).items()}
    for s in (x.strip() for x in SENTENCE.split(text or "")):
        if not s:
            continue
        quote = f"\"{s[:160]}\""
        negated, question = bool(NEGATION.search(s)), s.endswith("?") or bool(HYPOTHESIS.search(s))
        exclusion = bool(EXCLUDED.search(s))   # exclusion reasons are N-EXCLUSION's, checked against the log
        if GAP_OR_LABEL.search(s) and not negated and (COMMITMENT_DEFINED.search(s)
                                                       or (PAYMENT.search(s) and not exclusion)):
            out.append(("N-DEFINITION", "the signed gap is the verified final cost plus the included fee minus the "
                                        "approved budget; payments and commitment totals are not part of it or "
                                        f"of either label: {quote}"))
        for clause in s.split(";"):
            if not GAP_CLAUSE.search(clause):
                continue
            signs = _gap_signs(clause)
            if signs == {"negative"} and OVER_BUDGET.search(clause):
                out.append(("N-SIGN", f"a negative gap means under budget, not over budget: {quote}"))
            elif signs == {"positive"} and UNDER_BUDGET.search(clause):
                out.append(("N-SIGN", f"a positive gap means over budget, not under budget: {quote}"))
        if rows and productions and any(f in s for f in _fmt(rows)) and not any(f in s for f in _fmt(productions)):
            for m in PRODUCTIONS.finditer(s):
                n = int(m.group(1).replace(",", ""))
                if n != productions:
                    out.append(("N-POPULATION", f"'{m.group(0)}' is attached to the {rows:,} eligible snapshots, "
                                                f"which come from {productions} productions (stats.json); {n} is "
                                                f"not their production count: {quote}"))
        causal = REASON.sub(" ", s) if exclusion else s
        if CAUSAL.search(causal) and CAUSAL_TOPIC.search(s) and not (negated or question):
            out.append(("N-CAUSAL", "the analyses show associations only; no tool output supports a cause, effect "
                                    f"or explanation for revisions, stages, fees or cost outcomes: {quote}"))
        if FROZEN_FIGURE.search(s) and GATE.search(s):
            out.append(("N-ATTRIBUTION", "the test minus train difference and its bootstrap interval come from the "
                                         "stats.json analysis frozen_split_prevalence_difference, not from the "
                                         f"handoff gate: {quote}"))
        if ASSESSMENT_CREDITED.search(s):
            out.append(("N-ATTRIBUTION", "that the split is frozen, not tuned and a sampling outcome is the "
                                         "documented decision when it was frozen, not a result of an analysis or "
                                         f"of the gate: {quote}"))
        if not exclusion:
            continue
        scopes = [k for k, rx in (("payments", r"payment"), ("snapshots", r"snapshot|ineligib")) if re.search(rx, s, re.I)]
        if not scopes:
            continue
        codes = set().union(*(recorded.get(k, set()) for k in scopes))
        for label, rx, supports in REASONS:
            if rx.search(s) and (supports is None or not any(supports(c) for c in codes)):
                out.append(("N-EXCLUSION", f"exclusion reason '{label}' for {' and '.join(scopes)} is not recorded "
                                           f"in any tool output (recorded: {sorted(codes) or 'none'}): {quote}"))
    return out


def blocking_problems(text: str, facts: dict) -> list[str]:
    """The findings that fail a post-run validator (V6, V7)."""
    return [f"{rule}: {msg}" for rule, msg in findings(text, facts) if rule in BLOCKING]


def context_problems(text: str, stats: dict | None, contract: dict | None, frozen_counts: dict | None,
                     log: dict | None) -> list[str]:
    """Every finding, as feedback for the authoring tools."""
    return [msg for _, msg in findings(text, narrative_facts(log, contract, stats, frozen_counts))]
