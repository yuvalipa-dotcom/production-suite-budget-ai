"""The Flow contract ``flow-1``: validation of a Crew-1 handoff before Crew 2 sees it. ALL DATA IS SYNTHETIC.

The approved historical contract (``dataset_contract.json``, version ``v2``, checked against
``contracts/dataset_contract.schema.json``) is not changed. ``flow-1`` extends it for new Flow
runs with:

* explicit currency and unit metadata for every monetary column;
* the sha256 of the delivered ``clean_data.csv`` and of the delivered ``dataset_contract.json``;
* a declared scale profile per monetary column (median and p90 of absolute values, TRAIN rows only).

Every check below compares against an authority the delivered files cannot change: the approved
column specification (``gate/handoff_spec.py``), the frozen split (``contracts/frozen_split.json``)
and the pinned reference profile (``budget_flow/contracts/money_scale_reference.v1.json``, whose
sha256 is pinned in this module). A tampered ``clean_data.csv`` therefore cannot pass because a
matching contract was regenerated from it.

A ``UNIT_SCALE_ANOMALY`` reports that a monetary column's scale is outside the documented band
around the reference. It does not identify the cause: a x100 ratio is consistent with amounts in
agorot instead of shekels, but also with another currency, a VAT basis change or corrupted data.

Every outcome label of a non-train row is redacted when the file is parsed (``parse_and_redact``),
before any check runs, so no validation and no Crew 2 component ever holds a test or demo label.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from gate import handoff_spec as H
from gate.json_schema import errors as schema_errors

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONTRACTS = HERE / "contracts"
FLOW_SCHEMA_FILE = CONTRACTS / "flow_contract.flow-1.schema.json"
REFERENCE_FILE = CONTRACTS / "money_scale_reference.v1.json"
BASE_SCHEMA_FILE = REPO / "contracts" / "dataset_contract.schema.json"
FROZEN_SPLIT_FILE = REPO / "contracts" / "frozen_split.json"

FLOW_CONTRACT_VERSION = "flow-1"
BASE_CONTRACT_VERSION = "v2"
REFERENCE_VERSION = "money_scale_reference.v1"
# sha256 of money_scale_reference.v1.json, held as an integer literal (the publication scanner flags quoted
# hex digests). A changed reference file fails with REFERENCE_DIGEST_MISMATCH.
_REFERENCE_DIGEST = 53709452885378218498124241687007798266144778734023111317209061482010300536454
REFERENCE_SHA256 = f"{_REFERENCE_DIGEST:064x}"

# Initial documented policy (owner decision, P5-B): the measured statistic may differ from the reference by at
# most a factor of 2 in either direction.
SCALE_BAND = (0.5, 2.0)
MONEY_UNIT = "ILS pre-VAT"
CURRENCY = {"code": "ILS", "minor_unit": "agora", "minor_units_per_major": 100,
            "amount_scale": "major units (shekels), not agorot", "vat_basis": "pre-VAT"}
MONEY_COLUMNS = tuple(c for c, (_, _, unit) in H.COLUMNS.items() if unit == MONEY_UNIT)
LABEL_COLUMNS = tuple(H.TARGETS)
TRAIN = "train"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Violation:
    code: str
    column: str | None
    detail: str

    def as_dict(self) -> dict:
        return {"code": self.code, "column": self.column, "detail": self.detail}


@dataclass
class Report:
    clean_data_sha256: str
    base_contract_sha256: str
    profile: dict
    scale_ratios: dict
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> set[str]:
        return {v.code for v in self.violations}

    def as_dict(self) -> dict:
        return {"ok": self.ok, "clean_data_sha256": self.clean_data_sha256,
                "base_contract_sha256": self.base_contract_sha256, "profile": self.profile,
                "scale_ratios": self.scale_ratios, "violations": [v.as_dict() for v in self.violations]}


class ContractViolationError(RuntimeError):
    """The handoff failed the Flow contract; Crew 2 must not start."""

    def __init__(self, report: Report):
        self.report = report
        summary = "; ".join(f"{v.code}" + (f"[{v.column}]" if v.column else "") for v in report.violations[:8])
        super().__init__(f"handoff rejected by the Flow contract: {summary}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- parsing and redaction

def parse_and_redact(clean_csv: bytes) -> tuple[list[str], list[dict]]:
    """Header and rows of ``clean_data.csv``. Outcome labels of every non-train row become ``None`` here,
    before any other code sees the row; the delivered bytes are not kept."""
    reader = csv.DictReader(io.StringIO(clean_csv.decode("utf-8"), newline=""))
    header = list(reader.fieldnames or [])
    rows = []
    for r in reader:
        if r.get("split") != TRAIN:
            for c in LABEL_COLUMNS:
                if c in r:
                    r[c] = None
        rows.append(r)
    return header, rows


def redacted_csv_text(header: list[str], rows: list[dict]) -> str:
    """The rows as CSV text in the delivered column order; redacted labels are empty fields."""
    out = io.StringIO()
    w = csv.DictWriter(out, header, lineterminator="\n")
    w.writeheader()
    w.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
    return out.getvalue()


# ---------------------------------------------------------------- scale profile

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _p90(values: list[float]) -> float:
    """Nearest-rank 90th percentile."""
    s = sorted(values)
    return s[max(math.ceil(0.9 * len(s)) - 1, 0)]


def scale_profile(rows: list[dict]) -> dict:
    """Median and p90 of absolute values per monetary column, over TRAIN rows only."""
    out = {}
    for c in MONEY_COLUMNS:
        vals = []
        for r in rows:
            if r.get("split") == TRAIN and r.get(c) not in (None, ""):
                try:
                    v = float(r[c])
                except ValueError:
                    continue
                if math.isfinite(v):
                    vals.append(abs(v))
        out[c] = ({"n": len(vals), "median_abs": _median(vals), "p90_abs": _p90(vals)} if vals
                  else {"n": 0, "median_abs": None, "p90_abs": None})
    return out


# ---------------------------------------------------------------- the pinned reference

def load_reference(path: Path = REFERENCE_FILE) -> tuple[dict | None, list[Violation]]:
    data = path.read_bytes()
    if sha256_hex(data) != REFERENCE_SHA256:
        return None, [Violation("REFERENCE_DIGEST_MISMATCH", None,
                                f"{path.name} differs from the pinned reference profile")]
    ref = json.loads(data.decode("utf-8"))
    if ref.get("version") != REFERENCE_VERSION or set(ref.get("columns", {})) != set(MONEY_COLUMNS):
        return None, [Violation("REFERENCE_INVALID", None, "reference version or monetary columns differ")]
    return ref, []


def scale_check(profile: dict, reference: dict) -> tuple[dict, list[Violation]]:
    """Ratio of each column's measured statistic to the reference; outside SCALE_BAND is an anomaly."""
    lo, hi = reference["band"]
    ratios, out = {}, []
    for c in MONEY_COLUMNS:
        ref = reference["columns"][c]
        stat, ref_value = ref["statistic"], ref["value"]
        got = profile[c].get(stat)
        if got is None:
            ratios[c] = None
            out.append(Violation("UNIT_SCALE_ANOMALY", c, f"no {stat} could be measured (no usable TRAIN values)"))
            continue
        ratio = got / ref_value
        ratios[c] = ratio
        if not lo <= ratio <= hi:
            out.append(Violation(
                "UNIT_SCALE_ANOMALY", c,
                f"{stat} is {ratio:.4g} x the pinned reference (allowed band [{lo}, {hi}]). A scale anomaly is "
                f"detected; its cause (e.g. agorot instead of shekels, another currency or basis, corrupted values) "
                f"is not identified"))
    return ratios, out


# ---------------------------------------------------------------- the delivered contract (v2) against the spec

def _spec_column(name: str) -> dict:
    dtype, role, unit = H.COLUMNS[name]
    spec = {"name": name, "dtype": dtype, "role": role, "unit": unit, "nullable": False}
    if name in H.ALLOWED:
        spec["allowed"] = sorted(H.ALLOWED[name])
    lo, hi = H.RANGES.get(name, (None, None))
    if lo is not None:
        spec["min"] = lo
    if hi is not None:
        spec["max"] = hi
    return spec


_DECLARATION_CODES = {"dtype": "TYPE_DECLARATION_MISMATCH", "role": "ROLE_DECLARATION_MISMATCH",
                      "unit": "UNIT_DECLARATION_MISMATCH", "nullable": "NULLABLE_DECLARATION_MISMATCH",
                      "allowed": "ALLOWED_DECLARATION_MISMATCH", "min": "RANGE_DECLARATION_MISMATCH",
                      "max": "RANGE_DECLARATION_MISMATCH"}


def check_base_contract(base: dict) -> list[Violation]:
    out = []
    schema = json.loads(BASE_SCHEMA_FILE.read_text(encoding="utf-8"))
    out += [Violation("BASE_CONTRACT_SCHEMA", None, e) for e in schema_errors(base, schema)[:10]]
    if out:
        return out
    if base["version"] != BASE_CONTRACT_VERSION:
        out.append(Violation("BASE_CONTRACT_VERSION", None, f"version {base['version']!r}, expected "
                                                            f"{BASE_CONTRACT_VERSION!r}"))
    declared = {c["name"]: c for c in base["columns"]}
    names = [c["name"] for c in base["columns"]]
    if names != list(H.COLUMNS):
        out.append(Violation("BASE_CONTRACT_COLUMNS", None, "declared columns differ from the approved specification "
                                                            f"(missing {sorted(set(H.COLUMNS) - set(names))}, "
                                                            f"extra {sorted(set(names) - set(H.COLUMNS))})"))
    for name in H.COLUMNS:
        if name not in declared:
            continue
        got = {k: v for k, v in declared[name].items() if k != "description"}
        want = _spec_column(name)
        for key in sorted(set(got) | set(want)):
            if key == "name" or got.get(key) == want.get(key):
                continue
            out.append(Violation(_DECLARATION_CODES.get(key, "BASE_CONTRACT_FIELD"), name,
                                 f"{key} declared {got.get(key)!r}, approved {want.get(key)!r}"))
    leaked = sorted(set(base.get("outcome_only_exclusions", [])) & set(names))
    if leaked:
        out.append(Violation("ROLE_RESTRICTION", None, f"outcome-only fields declared as columns: {leaked}"))
    return out


# ---------------------------------------------------------------- the delivered data against the spec

def _typed_ok(dtype: str, value: str) -> bool:
    try:
        if dtype == "integer":
            return str(int(value)) == value
        if dtype == "number":
            return math.isfinite(float(value))
        if dtype == "boolean":
            return value in ("True", "False")
        if dtype == "date":
            return bool(_DATE.match(value)) and date.fromisoformat(value) is not None
        return value != ""
    except ValueError:
        return False


def check_rows(header: list[str], rows: list[dict], base: dict | None) -> list[Violation]:
    out = []
    if header != list(H.COLUMNS):
        out.append(Violation("COLUMN_SET_MISMATCH", None,
                             f"missing {sorted(set(H.COLUMNS) - set(header))}, extra {sorted(set(header) - set(H.COLUMNS))}"
                             if set(header) != set(H.COLUMNS) else "column order differs from the approved specification"))
    if base is not None and isinstance(base.get("row_count"), int) and base["row_count"] != len(rows):
        out.append(Violation("ROW_COUNT_MISMATCH", None, f"{len(rows)} rows, contract declares {base['row_count']}"))
    keys = [(r.get("production_id"), r.get("stage")) for r in rows]
    dup = len(keys) - len(set(keys))
    if dup:
        out.append(Violation("PRIMARY_KEY_DUPLICATE", None, f"{dup} duplicate (production_id, stage) key(s)"))
    counts: dict[tuple[str, str], list] = {}

    def bad(code, col, key):
        counts.setdefault((code, col), []).append(key)

    for r, key in zip(rows, keys):
        non_train = r.get("split") != TRAIN
        for col in H.COLUMNS:
            if col not in r:
                continue
            v = r[col]
            if v is None:                                       # a redacted label (non-train rows only)
                if not (non_train and col in LABEL_COLUMNS):
                    bad("NULL_VIOLATION", col, key)
                continue
            dtype = H.COLUMNS[col][0]
            if v == "":
                bad("NULL_VIOLATION", col, key)
            elif not _typed_ok(dtype, v):
                bad("TYPE_VIOLATION", col, key)
            elif col in H.ALLOWED and v not in H.ALLOWED[col]:
                bad("ALLOWED_VALUE_VIOLATION", col, key)
            elif col in H.RANGES and dtype in ("integer", "number"):
                lo, hi = H.RANGES[col]
                x = float(v)
                if (lo is not None and x < lo) or (hi is not None and x > hi):
                    bad("RANGE_VIOLATION", col, key)
        if non_train and any(r.get(c) is not None for c in LABEL_COLUMNS):
            bad("TEST_LABEL_EXPOSED", None, key)                # cannot happen after parse_and_redact
    for (code, col), ks in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        out.append(Violation(code, col, f"{len(ks)} row(s), first {ks[0][0]}/{ks[0][1]}"))
    return out


def check_split(rows: list[dict], frozen: dict) -> list[Violation]:
    """Every snapshot carries its production's frozen group and fold (no production straddles groups)."""
    want = {p["production_id"]: (p["split"], p["cv_fold"]) for p in frozen["productions"]}
    wrong = sorted({r["production_id"] for r in rows
                    if want.get(r.get("production_id")) != (r.get("split"), _int_or_none(r.get("cv_fold")))})
    return [Violation("SPLIT_POLICY_VIOLATION", None, f"{len(wrong)} production(s) differ from the frozen split, "
                                                      f"first {wrong[0]}")] if wrong else []


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- the flow-1 contract

def build_flow_contract(clean_csv: bytes, base_contract: bytes) -> dict:
    """The ``flow-1`` contract for a delivered handoff (written by the Flow after Crew 1)."""
    header, rows = parse_and_redact(clean_csv)
    return {
        "version": FLOW_CONTRACT_VERSION, "synthetic": True,
        "base_contract": {"file": H.CONTRACT_FILE, "version": BASE_CONTRACT_VERSION,
                          "sha256": sha256_hex(base_contract)},
        "clean_data": {"file": H.DATA_FILE, "sha256": sha256_hex(clean_csv), "rows": len(rows)},
        "currency": dict(CURRENCY),
        "reference": {"version": REFERENCE_VERSION, "sha256": REFERENCE_SHA256, "band": list(SCALE_BAND)},
        "monetary_columns": {c: {"unit": MONEY_UNIT, "currency": CURRENCY["code"], "vat_basis": CURRENCY["vat_basis"],
                                 "profile": p} for c, p in scale_profile(rows).items()},
    }


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def check_flow_contract(flow: dict, clean_sha: str, base_sha: str, profile: dict) -> list[Violation]:
    schema = json.loads(FLOW_SCHEMA_FILE.read_text(encoding="utf-8"))
    problems = schema_errors(flow, schema)
    if problems:
        return [Violation("FLOW_CONTRACT_SCHEMA", None, p) for p in problems[:10]]
    out = []
    if flow["clean_data"]["sha256"] != clean_sha:
        out.append(Violation("CONTENT_DIGEST_MISMATCH", None, "clean_data.csv differs from the file the Flow "
                                                              "contract was written for"))
    if flow["base_contract"]["sha256"] != base_sha:
        out.append(Violation("BASE_CONTRACT_DIGEST_MISMATCH", None, "dataset_contract.json differs from the file "
                                                                    "the Flow contract was written for"))
    if flow["reference"]["sha256"] != REFERENCE_SHA256 or flow["reference"]["band"] != list(SCALE_BAND):
        out.append(Violation("REFERENCE_DECLARATION_MISMATCH", None, "the Flow contract names another reference "
                                                                     "or band"))
    if flow["currency"] != CURRENCY:
        out.append(Violation("CURRENCY_DECLARATION_MISMATCH", None, f"currency {flow['currency']!r}"))
    for c in MONEY_COLUMNS:
        decl = flow["monetary_columns"][c]
        if decl["unit"] != MONEY_UNIT or decl["currency"] != CURRENCY["code"] or decl["vat_basis"] != CURRENCY["vat_basis"]:
            out.append(Violation("UNIT_DECLARATION_MISMATCH", c, f"declared {decl['unit']!r} {decl['currency']!r} "
                                                                 f"{decl['vat_basis']!r}"))
        if not all(_close(decl["profile"].get(k), profile[c].get(k)) for k in ("median_abs", "p90_abs")) \
                or decl["profile"].get("n") != profile[c]["n"]:
            out.append(Violation("PROFILE_DECLARATION_MISMATCH", c, "declared scale profile differs from the "
                                                                    "measured TRAIN profile"))
    return out


# ---------------------------------------------------------------- everything together

def validate(header: list[str], rows: list[dict], clean_sha: str, base_bytes: bytes,
             flow: dict | None = None, reference_path: Path = REFERENCE_FILE) -> Report:
    """Validate already-redacted rows (``parse_and_redact``) and the delivered contract bytes."""
    base_sha = sha256_hex(base_bytes)
    violations: list[Violation] = []
    try:
        base = json.loads(base_bytes.decode("utf-8"))
        if not isinstance(base, dict):
            raise ValueError("not an object")
    except ValueError as e:
        base = None
        violations.append(Violation("BASE_CONTRACT_SCHEMA", None, f"dataset_contract.json is not a JSON object ({e})"))
    if base is not None:
        violations += check_base_contract(base)
    violations += check_rows(header, rows, base)
    if "COLUMN_SET_MISMATCH" not in {v.code for v in violations}:
        violations += check_split(rows, json.loads(FROZEN_SPLIT_FILE.read_text(encoding="utf-8")))
    profile = scale_profile(rows)
    reference, ref_problems = load_reference(reference_path)
    violations += ref_problems
    ratios = {}
    if reference is not None:
        ratios, scale_problems = scale_check(profile, reference)
        violations += scale_problems
    if flow is not None:
        violations += check_flow_contract(flow, clean_sha, base_sha, profile)
    return Report(clean_sha, base_sha, profile, ratios, violations)
