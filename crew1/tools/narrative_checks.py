"""Deterministic feedback on agent narrative for the authoring tools (p3-live-003). Non-blocking:
the post-run validators (crew1/validators.py) are unchanged and remain the authority.

* ``frozen_split_values``: the frozen split difference and interval from stats.json, with units.
* ``context_problems``: statements that misread a number or claim something no tool output records:
  the production count of the eligible snapshots, the source of the frozen split figures, and
  exclusion reasons (payments and snapshots) that the cleaning log does not record.
"""

from __future__ import annotations

import re

SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
PRODUCTIONS = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)\s+productions\b", re.I)
FROZEN_FIGURE = re.compile(r"bootstrap|\binterval\b|test\s+minus\s+train|prevalence\s+difference|"
                           r"frozen_split_prevalence_difference", re.I)
GATE = re.compile(r"\bgate\b", re.I)
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


def eligible_productions(stats: dict | None, frozen_counts: dict | None) -> int | None:
    """Productions that contribute eligible snapshots, from stats.json (or the frozen total minus fully excluded)."""
    analyses = (stats or {}).get("analyses") or {}
    if "descriptive_statistics" in analyses:
        return analyses["descriptive_statistics"].get("productions")
    x = analyses.get("policy_c_exclusions")
    if x and frozen_counts and isinstance(frozen_counts.get("productions"), int):
        return frozen_counts["productions"] - x["n_productions_fully_excluded"]
    return None


def _recorded_reasons(log: dict | None, contract: dict | None) -> dict[str, set[str]]:
    log = log or {}
    payments = {p for i in log.get("issues") or [] if i.get("table") == "payments" for p in i.get("problems", [])}
    raw, clean = log.get("raw_rows") or {}, log.get("clean_rows") or {}
    if raw.get("payments", 0) - clean.get("payments", 0) - len(log.get("excluded_payments") or []) > 0:
        payments.add("exact_duplicate")
    snapshots = set(log.get("exclusion_counts") or {}) | set((contract or {}).get("exclusion_counts") or {})
    return {"payments": payments, "snapshots": snapshots}


def context_problems(text: str, stats: dict | None, contract: dict | None, frozen_counts: dict | None,
                     log: dict | None) -> list[str]:
    problems: list[str] = []
    rows = (contract or {}).get("row_count") or (((stats or {}).get("analyses") or {})
                                                 .get("policy_c_exclusions") or {}).get("eligible")
    productions = eligible_productions(stats, frozen_counts)
    recorded = _recorded_reasons(log, contract)
    for s in (x.strip() for x in SENTENCE.split(text or "")):
        if not s:
            continue
        if rows and productions and any(f in s for f in _fmt(rows)) and not any(f in s for f in _fmt(productions)):
            for m in PRODUCTIONS.finditer(s):
                n = int(m.group(1).replace(",", ""))
                if n != productions:
                    problems.append(f"'{m.group(0)}' is attached to the {rows:,} eligible snapshots, which come "
                                    f"from {productions} productions (stats.json); {n} is not their production "
                                    f"count: \"{s[:160]}\"")
        if FROZEN_FIGURE.search(s) and GATE.search(s):
            problems.append("the test minus train difference and its bootstrap interval come from the stats.json "
                            "analysis frozen_split_prevalence_difference, not from the handoff gate: "
                            f"\"{s[:160]}\"")
        if not EXCLUDED.search(s):
            continue
        scopes = [k for k, rx in (("payments", r"payment"), ("snapshots", r"snapshot|ineligib")) if re.search(rx, s, re.I)]
        if not scopes:
            continue
        codes = set().union(*(recorded[k] for k in scopes))
        for label, rx, supports in REASONS:
            if rx.search(s) and (supports is None or not any(supports(c) for c in codes)):
                problems.append(f"exclusion reason '{label}' for {' and '.join(scopes)} is not recorded in any "
                                f"tool output (recorded: {sorted(codes) or 'none'}): \"{s[:160]}\"")
    return problems
