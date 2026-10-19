"""A1 tools: ingest, profile, detect defects and clean under Policy C. Deterministic Python only.

Policy C itself is the library in ``gate/policy_c.py`` (the same code the P2 gate uses to
verify a handoff). ``clean_data.csv`` is written in the exact handoff layout of
``gate/handoff_spec.py``: one row per eligible (production_id, stage) snapshot, columns in
specification order, rows sorted by key, values in the dataset CSV format.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter

from gate import handoff_spec as H
from gate import policy_c as P
from gate.checks import LEDGERS, RAW_TABLES

from .workspace import Workspace, sha256_bytes

CODED_COLUMNS = set(P.CODED)
PRE_VAT_COLUMNS = {pre for _, pre in P.VAT_PAIRS.values()}


def dumps(obj) -> str:
    """Byte-stable JSON: sorted keys, two-space indent, LF, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def load_raw(ws: Workspace) -> dict:
    return {t: ws.read_csv(f"raw.{t}") for t in RAW_TABLES}


def frozen_assignments(frozen_doc: dict) -> dict[str, tuple[str, int]]:
    return {r["production_id"]: (r["split"], int(r["cv_fold"])) for r in frozen_doc["productions"]}


# ---------------------------------------------------------------- profiling and detection

def profile_table(ws: Workspace, name: str) -> dict:
    """Schema and profile summary of one raw table (by logical name, e.g. ``raw.commitments``)."""
    df = ws.read_csv(name)
    manifest = json.loads(ws.read_text("raw.manifest"))
    file = name.split(".", 1)[1] + ".csv"
    cols = {}
    for c in df.columns:
        values = df[c]
        stripped = values.str.strip()
        entry = {"empty": int((stripped == "").sum()), "distinct": int(stripped.nunique()),
                 "with_surrounding_spaces": int((values != stripped).sum())}
        if c in CODED_COLUMNS:
            entry["raw_values"] = dict(sorted(Counter(values).most_common(12)))
        cols[c] = entry
    return {"table": name, "rows": int(len(df)), "manifest_rows": manifest["rows"].get(file),
            "sha256_matches_manifest": sha256_bytes(ws.read_bytes(name)) == manifest["files_sha256"].get(file),
            "columns": cols}


def detect_defects(raw: dict) -> dict:
    """Per-type defect counts under the documented detection rules (docs/raw_cleaning_policy.md).

    Problems are the rule violations Policy C records (missing, invalid, inconsistent,
    out of range, conflicting duplicate). Normalisations are the always-allowed repairs.
    """
    problems: dict[str, Counter] = {}
    normalised: dict[str, Counter] = {}
    duplicates: dict[str, int] = {}
    for table, df in raw.items():
        prob, norm = Counter(), Counter()
        seen: dict[str, dict] = {}
        exact = 0
        for rec in df.to_dict("records"):
            cleaned, probs = P._clean_record(table, rec)
            for col, orig in rec.items():
                s = orig.strip()
                if orig != s:
                    norm["whitespace_trimmed"] += 1
                if s != cleaned[col]:
                    if col in CODED_COLUMNS:
                        norm["code_normalised"] += 1
                    elif col in P.DATES:
                        norm["date_reformatted"] += 1
                    elif col in PRE_VAT_COLUMNS:
                        norm["pre_vat_recomputed_from_gross"] += 1
            key = "#".join(cleaned[k] for k in P.KEYS[table])
            if key in seen:
                if seen[key] == cleaned:
                    exact += 1
                continue
            seen[key] = cleaned
            for p in probs:
                prob[p] += 1
        problems[table] = prob
        normalised[table] = norm
        duplicates[table] = exact
    result = P.apply_policy_c(raw)
    conflicts = Counter(i["table"] for i in result.issues if "conflicting_duplicate" in i["problems"])
    return {
        "problems_by_table": {t: dict(sorted(c.items())) for t, c in problems.items() if c},
        "problem_totals": dict(sorted(sum(problems.values(), Counter()).items())),
        "normalisations_by_table": {t: dict(sorted(c.items())) for t, c in normalised.items() if c},
        "exact_duplicates_dropped": {t: n for t, n in duplicates.items() if n},
        "conflicting_duplicates": dict(sorted(conflicts.items())),
        "records_with_problems": len(result.issues),
        "invalid_payments_excluded": len(result.excluded_payments),
    }


# ---------------------------------------------------------------- Policy C and the handoff

def build_clean_data(raw: dict, frozen_doc: dict, raw_manifest_sha256: str) -> tuple[str, dict]:
    """(clean_data.csv text, cleaning log). Pure and deterministic."""
    result = P.apply_policy_c(raw)
    ix = P.index({n: result.tables[n] for n in LEDGERS})
    groups = frozen_assignments(frozen_doc)
    columns = list(H.COLUMNS)
    elig = result.eligibility.sort_values(["production_id", "stage"]).to_dict("records")
    rows, ineligible = [], []
    for r in elig:
        pid, stage, as_of = r["production_id"], r["stage"], r["as_of_date"]
        if not r["eligible"]:
            ineligible.append({"production_id": pid, "stage": stage, "reason": r["reason"]})
            continue
        split, fold = groups[pid]
        values = {"production_id": pid, "as_of_date": as_of, "split": split, "cv_fold": fold,
                  **P.snapshot_features(ix, pid, stage, as_of), **P.snapshot_labels(ix, pid, as_of)}
        rows.append([P.as_csv_text(values[c]) for c in columns])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
    exclusions = dict(sorted(Counter(x["reason"] for x in ineligible).items()))
    log = {
        "policy": "C (docs/raw_cleaning_policy.md)",
        "synthetic": True,
        "raw_manifest_sha256": raw_manifest_sha256,
        "raw_rows": {t: int(len(df)) for t, df in raw.items()},
        "clean_rows": {t: int(len(df)) for t, df in result.tables.items()},
        "issues": sorted(({"table": i["table"], "record_id": i["record_id"], "problems": list(i["problems"])}
                          for i in result.issues), key=lambda i: (i["table"], i["record_id"])),
        "excluded_payments": sorted(result.excluded_payments),
        "target_unreliable_productions": sorted(p for p, ok in result.target_reliable.items() if not ok),
        "snapshots": {"scheduled": len(elig), "eligible": len(rows), "ineligible": len(ineligible)},
        "exclusion_counts": exclusions,
        "ineligible_snapshots": ineligible,
    }
    return buf.getvalue(), log


def eligibility_summary(log: dict) -> dict:
    """Eligibility by stage and reason, from a cleaning log."""
    by_stage = Counter(x["stage"] for x in log["ineligible_snapshots"])
    prods = Counter(x["production_id"] for x in log["ineligible_snapshots"])
    n_prod = log["raw_rows"]["productions"]
    return {
        "scheduled_snapshots": log["snapshots"]["scheduled"],
        "eligible_snapshots": log["snapshots"]["eligible"],
        "ineligible_snapshots": log["snapshots"]["ineligible"],
        "ineligible_by_stage": {s: by_stage.get(s, 0) for s in P.STAGES},
        "exclusions_by_reason": log["exclusion_counts"],
        "productions_with_an_ineligible_snapshot": len(prods),
        "productions_fully_ineligible": sum(1 for n in prods.values() if n == len(P.STAGES)),
        "productions_fully_eligible": n_prod - len(prods),
        "target_unreliable_productions": len(log["target_unreliable_productions"]),
        "invalid_payments_excluded": len(log["excluded_payments"]),
    }
