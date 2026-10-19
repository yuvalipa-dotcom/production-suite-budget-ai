"""Quantitative-claim and forbidden-claim checks for the Crew 2 narrative (validator W7/W8).

The claim extractor and matcher are Crew 1's (``crew1/validators.py``, rules in docs/crew1.md);
only the sources differ: every numeric value of ``metrics.json`` plus the pre-registered
protocol constants. A number in the agent's text with no matching source fails the run.
"""

from __future__ import annotations

from crew1.validators import claim_matches, extract_claims, forbidden_claims  # noqa: F401  (re-exported)

# Pre-registered protocol constants (Addendum P4-1) the narrative may quote.
PROTOCOL_COUNTS = (5, 3, 4, 6, 10, 24, 27, 29, 2000, 1538, 1600, 62, 50, 12, 280, 80, 40, 400)
PROTOCOL_NUMBERS = (0.005, 0.5, 0.02, 0.95, 0.01, 0.1, 1.0, 10.0, 0.05, 100, 300, 2, 3, -0.30, -0.15, -0.05, 0.0)


def numeric_leaves(obj):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int):
        yield float(obj), "int"
    elif isinstance(obj, float):
        yield obj, "float"
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from numeric_leaves(v)


def sources(metrics: dict | None) -> dict[str, list[float]]:
    by_unit: dict[str, list[float]] = {"count": [], "number": [], "fraction": [], "ratio": []}
    for v, kind in numeric_leaves(metrics or {}):
        if kind == "int":
            by_unit["count"].append(v)
            by_unit["number"].append(v)
        else:
            by_unit["fraction"].append(v)
            by_unit["number"].append(v)
    by_unit["count"] += [float(x) for x in PROTOCOL_COUNTS]
    by_unit["number"] += [float(x) for x in PROTOCOL_COUNTS + PROTOCOL_NUMBERS]
    by_unit["fraction"] += [float(x) for x in PROTOCOL_NUMBERS]
    return by_unit


def unverified(text: str, metrics: dict | None) -> list[str]:
    src = sources(metrics)
    return [f"{c.text} [{c.unit}]" for c in extract_claims(text) if not claim_matches(c, src)]
