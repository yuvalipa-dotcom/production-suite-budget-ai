"""Pure, deterministic checks for the P2 source and handoff gates.

Runtime reads are limited to: the raw export directory, the delivered handoff
(``clean_data.csv``, ``dataset_contract.json``) and three committed contracts
(``column_roles.json``, ``dataset_contract.schema.json``, ``frozen_split.json``).
The gate never reads the committed clean snapshots, ground truth or test fixtures;
every expected value is rebuilt from the raw export under Policy C.

Each check declares the inputs it needs. A check whose input is unavailable is
NOT_RUN (a HARD failure); a check that raises is FAIL. Corrupt input therefore
never crashes the gate and never passes silently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from . import handoff_spec as H
from .json_schema import errors as schema_errors
from .policy_c import D1, D2, STAGES, apply_policy_c, as_csv_text, index, snapshot_features, snapshot_labels
from .reconstruct import FEATURE_SOURCES, LABELS, OVERRUN_TOLERANCE, WARNING_THRESHOLD

RAW_TABLES = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs", "payments",
              "snapshot_schedule")
LEDGERS = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs")
HARD, WARN, INFO = "HARD", "WARN", "INFO"
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"
TOLERANCE = 0.01                 # delivered values are rounded to 2 decimals
CONSISTENCY_BAND = 0.02          # rounding band around a label threshold that is not judged internally
PREVALENCE_WARN_PP = 5.0
FROZEN_COUNTS = {"productions": 400, "train": 280, "test": 80, "demo": 40}
N_FOLDS = 5
NON_TRAIN_FOLD = -1
HISTORICAL_IMBALANCE = ("historical context, not recomputed: docs/split_report.md records a test-minus-train "
                        "row-level overrun gap of +15.0 pp for the frozen split")


def result(id, severity, status, details=""):
    return {"id": id, "severity": severity, "status": status, "details": details}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(problems) -> str:
    shown = ", ".join(str(p) for p in problems[:5])
    return f"{len(problems)} problem(s): {shown}" + (" ..." if len(problems) > 5 else "")


def _contracts(root) -> dict[str, Path]:
    c = Path(root) / "contracts"
    return {"roles": c / "column_roles.json", "schema": c / "dataset_contract.schema.json",
            "frozen": c / "frozen_split.json"}


def load_raw(raw) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(Path(raw) / f"{n}.csv", dtype=str, keep_default_na=False) for n in RAW_TABLES}


def raw_manifest_hash(raw) -> str:
    return sha(Path(raw) / "manifest.json")


def role_contract(root) -> dict:
    return json.loads(_contracts(root)["roles"].read_text(encoding="utf-8"))


def outcome_only(roles: dict) -> set[str]:
    return {f"{t}.{c}" for t, cols in roles["tables"].items() for c, s in cols.items() if s["role"] == "outcome-only"}


def expected(raw):
    """Policy C eligibility, features and labels for every scheduled snapshot, rebuilt from raw only."""
    p = apply_policy_c(raw)
    ix = index({n: p.tables[n] for n in LEDGERS})
    rows = {}
    for r in p.eligibility.to_dict("records"):
        k = (r["production_id"], r["stage"])
        rows[k] = {"eligibility": bool(r["eligible"]), "reason": r["reason"], "as_of_date": r["as_of_date"],
                   "features": snapshot_features(ix, *k, r["as_of_date"]) if r["eligible"] else None,
                   "labels": snapshot_labels(ix, k[0], r["as_of_date"])}
    return p, rows


def validate_frozen(doc) -> tuple[list[str], dict[str, tuple[str, int]]]:
    """Structural and invariant checks of the frozen split contract; returns (problems, id -> (split, fold))."""
    if not isinstance(doc, dict) or doc.get("contract") != "frozen_split" or not isinstance(doc.get("productions"), list):
        return ["not a frozen_split contract"], {}
    problems, mapping = [], {}
    rows = doc["productions"]
    ids = [r.get("production_id") for r in rows if isinstance(r, dict)]
    if len(ids) != len(rows) or ids != sorted(ids) or len(set(ids)) != len(ids):
        problems.append("productions must be unique objects sorted by production_id")
    for r in rows:
        if not isinstance(r, dict):
            continue
        g, f = r.get("split"), r.get("cv_fold")
        if g not in H.ALLOWED["split"] or not isinstance(f, int) or isinstance(f, bool):
            problems.append(f"invalid assignment: {r.get('production_id')}")
        elif (g == "train") != (0 <= f < N_FOLDS) or (g != "train" and f != NON_TRAIN_FOLD):
            problems.append(f"fold not allowed for group: {r.get('production_id')}")
        mapping[r.get("production_id")] = (g, f)
    counts = Counter(g for g, _ in mapping.values())
    actual = {"productions": len(mapping), **{g: counts[g] for g in ("train", "test", "demo")}}
    if actual != FROZEN_COUNTS:
        problems.append(f"counts {actual} != approved {FROZEN_COUNTS}")
    folds = Counter(f for g, f in mapping.values() if g == "train")
    if sorted(folds) != list(range(N_FOLDS)):
        problems.append("train folds must be exactly 0..4")
    declared = doc.get("counts", {})
    if {k: declared.get(k) for k in FROZEN_COUNTS} != actual or \
            declared.get("train_folds") != {str(k): folds[k] for k in range(N_FOLDS)}:
        problems.append("declared counts do not match the listed assignments")
    return problems, mapping


# ---------------------------------------------------------------- input loading

def _load(ctx: dict, key: str, loader, problems: list[str], label: str) -> None:
    try:
        ctx[key] = loader()
    except FileNotFoundError:
        ctx[key] = None
        problems.append(f"{label}: missing")
    except Exception as e:                                   # unreadable input is reported, never raised
        ctx[key] = None
        problems.append(f"{label}: unreadable ({type(e).__name__})")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_common(ctx: dict, root, rawdir, problems: list[str]) -> None:
    files = _contracts(root)
    _load(ctx, "roles", lambda: _json(files["roles"]), problems, "contracts/column_roles.json")
    _load(ctx, "frozen", lambda: _json(files["frozen"]), problems, "contracts/frozen_split.json")
    _load(ctx, "raw_manifest", lambda: _json(Path(rawdir) / "manifest.json"), problems, "raw manifest.json")
    ctx["raw_manifest_sha"] = raw_manifest_hash(rawdir) if ctx["raw_manifest"] is not None else None
    ctx["rawdir"] = Path(rawdir)
    _load(ctx, "raw", lambda: load_raw(rawdir), problems, "raw export tables")


def _run(specs, ctx) -> list[dict]:
    out = []
    for cid, severity, needs, fn in specs:
        missing = [n for n in needs if ctx.get(n) is None]
        if missing:
            out.append(result(cid, severity, NOT_RUN, "input unavailable: " + ", ".join(missing)))
            continue
        try:
            got = fn(ctx)
        except Exception as e:
            out.append(result(cid, severity, FAIL, f"check error: {type(e).__name__}: {e}"))
            continue
        if isinstance(got, tuple):                           # WARN / INFO checks report their own status
            out.append(result(cid, severity, *got))
        else:
            out.append(result(cid, severity, FAIL if got else PASS, _summary(got) if got else ""))
    return out


# ---------------------------------------------------------------- shared helpers

def _missing_cols(data: pd.DataFrame, cols) -> list[str]:
    return [f"column missing: {c}" for c in cols if c not in data.columns]


def _mismatch(delivered: str, value) -> bool:
    if isinstance(value, float) and not isinstance(value, bool):
        try:
            return not abs(float(delivered) - value) <= TOLERANCE
        except ValueError:
            return True
    return delivered != as_csv_text(value)


def _value_problems(ctx, cols, part: str) -> list:
    """Delivered cells that differ from the raw reconstruction. Rows outside the scheduled,
    reconstructable set are left to the eligibility checks."""
    data, exp = ctx["data"], ctx["expected"]
    bad = _missing_cols(data, cols + H.PRIMARY_KEY)
    if bad:
        return bad
    for r in data.to_dict("records"):
        e = exp.get((r["production_id"], r["stage"]))
        source = e and e[part]
        if not source:
            continue
        bad += [(r["production_id"], r["stage"], c) for c in cols if _mismatch(r[c], source[c])]
    return bad


def _float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


_PARSERS = {
    "string": lambda v: True,
    "integer": lambda v: re.fullmatch(r"-?\d+", v) is not None,
    "number": lambda v: _float(v) is not None,
    "boolean": lambda v: v in ("True", "False"),
    "date": lambda v: re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) is not None and _iso(v),
}


def _iso(v: str) -> bool:
    try:
        date.fromisoformat(v)
        return True
    except ValueError:
        return False


def _specs(contract) -> dict[str, dict]:
    return {c["name"]: c for c in contract.get("columns", []) if isinstance(c, dict) and "name" in c}


# ---------------------------------------------------------------- source gate

def _src_manifest(ctx):
    m, raw = ctx["raw_manifest"], ctx["rawdir"]
    files, rows = m.get("files_sha256", {}), m.get("rows", {})
    bad = [f"not listed: {n}.csv" for n in RAW_TABLES if f"{n}.csv" not in files]
    for name, digest in sorted(files.items()):
        path = raw / name
        if not path.is_file():
            bad.append(f"missing: {name}")
        elif sha(path) != digest:
            bad.append(f"hash mismatch: {name}")
        elif name in rows and len(pd.read_csv(path, dtype=str, keep_default_na=False)) != rows[name]:
            bad.append(f"row count mismatch: {name}")
    return bad


def _src_roles(ctx):
    roles = ctx["roles"]["tables"]
    return [f"{n}: columns without a role {sorted(set(df.columns) - set(roles.get(n, {})))}"
            for n, df in ctx["raw"].items() if not set(df.columns) <= set(roles.get(n, {}))]


def _src_outcome(ctx):
    wanted = {"commitments.final_pre_vat", "productions.financial_close_date"}
    return sorted(wanted - outcome_only(ctx["roles"]))


def _src_frozen(ctx):
    problems, mapping = validate_frozen(ctx["frozen"])
    raw_ids = {p.strip() for p in ctx["raw"]["productions"]["production_id"]}
    if set(mapping) != raw_ids:
        problems.append(f"production ids differ from the raw export ({len(set(mapping) ^ raw_ids)} differ)")
    return problems


def _synthetic(ctx):
    flags = set(ctx["raw"]["productions"]["is_synthetic"].str.strip())
    ok = ctx["raw_manifest"].get("synthetic") is True and flags == {"True"}
    return (PASS if ok else WARN,
            "synthetic data; no real-world validation" if ok else "synthetic flag not confirmed")


SOURCE_CHECKS = (
    ("G-SRC-MANIFEST", HARD, ("raw_manifest",), _src_manifest),
    ("G-SRC-ROLES", HARD, ("raw", "roles"), _src_roles),
    ("G-SRC-OUTCOME", HARD, ("roles",), _src_outcome),
    ("G-SRC-FROZEN-SPLIT", HARD, ("frozen", "raw"), _src_frozen),
    ("I-SYNTHETIC", INFO, ("raw_manifest", "raw"), _synthetic),
)


def source_checks(root, rawdir) -> list[dict]:
    ctx, problems = {}, []
    _load_common(ctx, root, rawdir, problems)
    return _run(SOURCE_CHECKS, ctx)


# ---------------------------------------------------------------- handoff gate

def _in_files(ctx):
    return ctx["input_problems"]


def _contract_schema(ctx):
    c = ctx["contract"]
    bad = schema_errors(c, ctx["schema"])
    if not bad:
        names = [x["name"] for x in c["columns"]]
        bad += [f"duplicate column: {n}" for n, k in sorted(Counter(names).items()) if k > 1]
        bad += [f"primary key column not declared: {k}" for k in H.PRIMARY_KEY if k not in names]
    return bad


def _provenance(ctx):
    bad = [] if ctx["contract"].get("raw_manifest_sha256") == ctx["raw_manifest_sha"] else \
        ["raw_manifest_sha256 does not match the raw export manifest"]
    return bad + _src_manifest(ctx)


def _columns(ctx):
    data_cols, names = list(ctx["data"].columns), set(_specs(ctx["contract"]))
    bad = [f"in data, not in contract: {c}" for c in data_cols if c not in names]
    bad += [f"in contract, not in data: {c}" for c in sorted(names - set(data_cols))]
    bad += [f"required column missing: {c}" for c in H.COLUMNS if c not in names or c not in data_cols]
    return bad


def _dtypes(ctx):
    data, bad = ctx["data"], []
    for name, s in sorted(_specs(ctx["contract"]).items()):
        dtype = s.get("dtype")
        if dtype not in H.DTYPES:
            bad.append(f"unknown dtype: {name}={dtype}")
            continue
        if name in H.COLUMNS:
            want_dtype, _, want_unit = H.COLUMNS[name]
            if dtype != want_dtype:
                bad.append(f"dtype differs from the approved spec: {name}")
            if s.get("unit") != want_unit:
                bad.append(f"unit differs from the approved spec: {name}")
        if name in data.columns:
            values = data[name]
            if not values[values != ""].map(_PARSERS[dtype]).all():
                bad.append(f"values do not parse as {dtype}: {name}")
    return bad


def _roles(ctx):
    bad = []
    for name, s in sorted(_specs(ctx["contract"]).items()):
        role = s.get("role")
        if name in H.COLUMNS and role != H.COLUMNS[name][1]:
            bad.append(f"role differs from the approved spec: {name}={role}")
        elif name not in H.COLUMNS and role in ("id", "target"):
            bad.append(f"undeclared {role} column: {name}")
    return bad


def _leakage(ctx):
    outcomes = outcome_only(ctx["roles"])
    names = {x.rsplit(".", 1)[1] for x in outcomes}
    specs = _specs(ctx["contract"])
    bad = []
    for name, s in sorted(specs.items()):
        role = s.get("role")
        if role == "feature":
            if name in names or name in LABELS:
                bad.append(f"outcome-only column used as a feature: {name}")
            elif name not in FEATURE_SOURCES:
                bad.append(f"feature without declared raw provenance: {name}")
            elif FEATURE_SOURCES[name] & outcomes:
                bad.append(f"feature derived from outcome-only data: {name}")
        elif name in names and not (role == "target" and name in LABELS):
            bad.append(f"outcome-only column delivered as {role}: {name}")
    bad += [f"outcome-only column delivered without a contract entry: {c}"
            for c in ctx["data"].columns if c in names and c not in specs]
    if set(ctx["contract"].get("outcome_only_exclusions", [])) != outcomes:
        bad.append("outcome_only_exclusions does not list exactly the outcome-only columns of column_roles.json")
    return bad


def _key_unique(ctx):
    data = ctx["data"]
    bad = _missing_cols(data, H.PRIMARY_KEY)
    if bad:
        return bad
    keys = list(zip(data["production_id"], data["stage"]))
    bad += [f"duplicate key: {k}" for k, n in sorted(Counter(keys).items()) if n > 1]
    bad += [f"invalid key: {k}" for k in sorted(set(keys)) if not k[0] or k[1] not in STAGES]
    return bad


def _delivered_keys(ctx) -> set:
    data = ctx["data"]
    return set(zip(data["production_id"], data["stage"]))


def _eligibility_set(ctx):
    bad = _missing_cols(ctx["data"], H.PRIMARY_KEY)
    if bad:
        return bad
    eligible = {k for k, v in ctx["expected"].items() if v["eligibility"]}
    delivered = _delivered_keys(ctx)
    return ([f"delivered but not eligible: {k}" for k in sorted(delivered - eligible)]
            + [f"eligible but not delivered: {k}" for k in sorted(eligible - delivered)])


def _eligibility_counts(ctx):
    c, exp, n = ctx["contract"], ctx["expected"], len(ctx["data"])
    actual = dict(sorted(Counter(v["reason"] for v in exp.values() if not v["eligibility"]).items()))
    bad = [] if c.get("row_count") == n else [f"row_count {c.get('row_count')} != delivered rows {n}"]
    declared = c.get("exclusion_counts")
    if declared != actual:
        bad.append(f"exclusion_counts {declared} != Policy C {actual}")
    if isinstance(declared, dict) and isinstance(c.get("row_count"), int) and \
            c["row_count"] + sum(v for v in declared.values() if isinstance(v, int)) != len(exp):
        bad.append(f"row_count + exclusions != {len(exp)} scheduled snapshots")
    return bad


def _target_consistency(ctx):
    data = ctx["data"]
    cols = ["budget_gap_ils", "is_overrun", "is_material_overrun", "budget_pre_vat"]
    bad = _missing_cols(data, cols + H.PRIMARY_KEY)
    if bad:
        return bad
    for r in data.to_dict("records"):
        gap, budget = _float(r["budget_gap_ils"]), _float(r["budget_pre_vat"])
        if gap is None or budget is None:
            bad.append((r["production_id"], r["stage"], "unparsable gap or budget"))
            continue
        for col, threshold in (("is_overrun", OVERRUN_TOLERANCE), ("is_material_overrun", WARNING_THRESHOLD * budget)):
            if abs(gap - threshold) > CONSISTENCY_BAND and (r[col] == "True") != (gap > threshold):
                bad.append((r["production_id"], r["stage"], col))
    return bad


def _non_financial(ctx):
    bad = _value_problems(ctx, list(D2), "features")
    if bad and isinstance(bad[0], str):
        return bad
    bad += _missing_cols(ctx["data"], ["as_of_date"])
    if "as_of_date" in ctx["data"].columns:
        for r in ctx["data"].to_dict("records"):
            e = ctx["expected"].get((r["production_id"], r["stage"]))
            if e and r["as_of_date"] != e["as_of_date"]:
                bad.append((r["production_id"], r["stage"], "as_of_date"))
    return bad


def _split(column: str, position: int):
    def check(ctx):
        problems, frozen = validate_frozen(ctx["frozen"])
        if problems:
            return ["frozen split contract invalid: " + "; ".join(problems)]
        data = ctx["data"]
        bad = _missing_cols(data, [column, "production_id"])
        if bad:
            return bad
        for pid, value in sorted(set(zip(data["production_id"], data[column]))):
            if pid not in frozen:
                bad.append(f"production not in the frozen split: {pid}")
            elif value != str(frozen[pid][position]):
                bad.append(f"{pid}: {column}={value}, frozen {frozen[pid][position]}")
        return bad
    return check


def _nulls(ctx):
    data, specs = ctx["data"], _specs(ctx["contract"])
    required = set(H.COLUMNS) | {n for n, s in specs.items()
                                 if s.get("role") in ("id", "feature", "target") or s.get("nullable") is False}
    bad = [f"required column missing: {c}" for c in sorted(required) if c not in data.columns]
    bad += [f"{c}: {int((data[c] == '').sum())} empty value(s)" for c in sorted(required)
            if c in data.columns and (data[c] == "").any()]
    return bad


def _ranges(ctx):
    data, specs, bad = ctx["data"], _specs(ctx["contract"]), []
    bounds = {c: list(b) for c, b in H.RANGES.items()}
    for name, s in specs.items():
        if "min" in s or "max" in s:
            lo, hi = bounds.get(name, [None, None])
            bounds[name] = [s.get("min", lo), s.get("max", hi)]
    for name, (lo, hi) in sorted(bounds.items()):
        if name not in data.columns:
            continue
        values = [_float(v) for v in data[name] if v != ""]
        if any(v is None or (lo is not None and v < lo) or (hi is not None and v > hi) for v in values):
            bad.append(f"out of range [{lo}, {hi}]: {name}")
    allowed = {c: set(v) for c, v in H.ALLOWED.items()}
    for name, s in specs.items():
        if isinstance(s.get("allowed"), list):
            allowed[name] = allowed.get(name, set(map(str, s["allowed"]))) & set(map(str, s["allowed"]))
    for name, values in sorted(allowed.items()):
        if name in data.columns and not set(data[name][data[name] != ""]) <= values:
            bad.append(f"value outside the allowed set: {name}")
    return bad


def _rate(keys, exp, groups) -> str:
    n, k = Counter(), Counter()
    for key in keys:
        g = groups.get(key[0], ("unassigned",))[0]
        n[g] += 1
        k[g] += exp[key]["labels"]["is_overrun"]
    return ", ".join(f"{g} {100 * k[g] / n[g]:.1f}% ({k[g]}/{n[g]})" for g in ("train", "test", "demo") if n[g])


def _prevalence(ctx):
    problems, groups = validate_frozen(ctx["frozen"])
    if problems:
        return WARN, "not computed: frozen split contract invalid"
    exp = ctx["expected"]
    before = sorted(exp)
    after = sorted(k for k, v in exp.items() if v["eligibility"])

    def gap(keys):
        rate = {g: [exp[k]["labels"]["is_overrun"] for k in keys if groups[k[0]][0] == g] for g in ("train", "test")}
        return 100 * (sum(rate["test"]) / len(rate["test"]) - sum(rate["train"]) / len(rate["train"]))

    diff = gap(after)
    details = (f"row-level is_overrun rate rebuilt from raw, groups from the frozen split; "
               f"before Policy C eligibility filtering ({len(before)} snapshots): {_rate(before, exp, groups)}; "
               f"after Policy C eligibility filtering ({len(after)} snapshots): {_rate(after, exp, groups)}; "
               f"test minus train: {gap(before):+.1f} pp before, {diff:+.1f} pp after "
               f"(warning at {PREVALENCE_WARN_PP:.1f} pp); {HISTORICAL_IMBALANCE}")
    return (WARN if abs(diff) >= PREVALENCE_WARN_PP else PASS), details


HANDOFF_CHECKS = (
    ("G-IN-FILES", HARD, (), _in_files),
    ("G-CONTRACT-SCHEMA", HARD, ("contract", "schema"), _contract_schema),
    ("G-CONTRACT-PROVENANCE", HARD, ("contract", "raw_manifest"), _provenance),
    ("G-SCHEMA-COLUMNS", HARD, ("data", "contract"), _columns),
    ("G-SCHEMA-DTYPES", HARD, ("data", "contract"), _dtypes),
    ("G-ROLES", HARD, ("contract",), _roles),
    ("G-LEAKAGE-OUTCOME", HARD, ("data", "contract", "roles"), _leakage),
    ("G-KEY-UNIQUE", HARD, ("data",), _key_unique),
    ("G-ELIGIBILITY-SET", HARD, ("data", "expected"), _eligibility_set),
    ("G-ELIGIBILITY-COUNTS", HARD, ("data", "contract", "expected"), _eligibility_counts),
    ("G-TARGET-GAP", HARD, ("data", "expected"),
     lambda ctx: _value_problems(ctx, ["budget_gap_ils", "final_cost_pre_vat"], "labels")),
    ("G-TARGET-LABEL", HARD, ("data", "expected"),
     lambda ctx: _value_problems(ctx, ["is_overrun", "is_material_overrun"], "labels")),
    ("G-TARGET-CONSISTENCY", HARD, ("data",), _target_consistency),
    ("G-FIN-RECON", HARD, ("data", "expected"), lambda ctx: _value_problems(ctx, list(D1), "features")),
    ("G-FIN-ASOF-BUDGET", HARD, ("data", "expected"), lambda ctx: _value_problems(ctx, ["budget_pre_vat"], "features")),
    ("G-NONFIN-RULES", HARD, ("data", "expected"), _non_financial),
    ("G-SPLIT-GROUP", HARD, ("data", "frozen"), _split("split", 0)),
    ("G-SPLIT-FOLD", HARD, ("data", "frozen"), _split("cv_fold", 1)),
    ("G-NULLS", HARD, ("data", "contract"), _nulls),
    ("G-RANGES", HARD, ("data", "contract"), _ranges),
    ("W-PREVALENCE-IMBALANCE", WARN, ("expected", "frozen"), _prevalence),
    ("I-SYNTHETIC", INFO, ("raw_manifest", "raw"), _synthetic),
)


def handoff_checks(root, data_path, contract_path, rawdir) -> list[dict]:
    ctx, problems = {}, []
    _load(ctx, "data", lambda: pd.read_csv(data_path, dtype=str, keep_default_na=False), problems, H.DATA_FILE)
    _load(ctx, "contract", lambda: _json(contract_path), problems, H.CONTRACT_FILE)
    if ctx["contract"] is not None and not isinstance(ctx["contract"], dict):
        ctx["contract"] = None
        problems.append(f"{H.CONTRACT_FILE}: not a JSON object")
    _load(ctx, "schema", lambda: _json(_contracts(root)["schema"]), problems, "contracts/dataset_contract.schema.json")
    _load_common(ctx, root, rawdir, problems)
    ctx["expected"] = None
    if ctx["raw"] is not None:
        try:
            ctx["expected"] = expected(ctx["raw"])[1]
        except Exception as e:
            problems.append(f"raw export: Policy C reconstruction failed ({type(e).__name__})")
    ctx["input_problems"] = problems
    return _run(HANDOFF_CHECKS, ctx)
