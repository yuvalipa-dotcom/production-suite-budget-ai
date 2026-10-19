"""A3 tool: the dataset contract (deterministic). The gate call is in ``gate_tool.py``.

Every structural field of ``dataset_contract.json`` is built by code from the ACTUAL
clean_data.csv, ``contracts/column_roles.json``, the meta-contract
(``contracts/dataset_contract.schema.json``), the raw manifest hash and the Policy C counts.
The agent may add only free-text column descriptions, which are validated here.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from gate import handoff_spec as H
from gate.checks import outcome_only
from gate.json_schema import errors as schema_errors

from .cleaning import dumps
from .workspace import Workspace

DATASET = "production-suite-budget-ai Crew-1 handoff (synthetic)"
VERSION = "v2"
SPLIT_POLICY = "frozen: contracts/frozen_split.json"
MAX_DESCRIPTION = 300
DESCRIPTION_RE = re.compile(r"[^\x00-\x1f<>`]+")


class ContractError(ValueError):
    pass


def descriptions_from_entries(entries) -> dict[str, str] | None:
    """The tool's typed list of {column, description} entries as a column -> text mapping."""
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ContractError("column_descriptions must be a list of {column, description} entries")
    out: dict[str, str] = {}
    for e in entries:
        e = e.model_dump() if hasattr(e, "model_dump") else e
        if not isinstance(e, dict) or set(e) != {"column", "description"}:
            raise ContractError("each column description must have exactly the fields column and description")
        if e["column"] in out:
            raise ContractError(f"duplicate column in descriptions: {e['column']}")
        out[e["column"]] = e["description"]
    return out


def validate_descriptions(descriptions, columns: list[str]) -> dict[str, str]:
    """Free-text descriptions only: known columns, plain single-line text of bounded length."""
    if descriptions in (None, ""):
        return {}
    if isinstance(descriptions, str):
        try:
            descriptions = json.loads(descriptions)
        except ValueError:
            raise ContractError("descriptions must be an object mapping column name to text") from None
    if not isinstance(descriptions, dict):
        raise ContractError("descriptions must be an object mapping column name to text")
    out = {}
    for name, text in descriptions.items():
        if name not in columns:
            raise ContractError(f"unknown column in descriptions: {name}")
        if not isinstance(text, str) or not text.strip():
            raise ContractError(f"description must be non-empty text: {name}")
        text = text.strip()
        if len(text) > MAX_DESCRIPTION or not DESCRIPTION_RE.fullmatch(text):
            raise ContractError(f"description must be one line of plain text, at most {MAX_DESCRIPTION} "
                                f"characters, without control characters, angle brackets or backticks: {name}")
        out[name] = text
    return out


def build_contract(clean_csv: bytes, contracts_dir: Path, raw_manifest_sha256: str, exclusion_counts: dict,
                   descriptions=None) -> dict:
    """The dataset contract for a delivered clean_data.csv. Structural fields are never agent input."""
    reader = csv.reader(io.StringIO(clean_csv.decode("utf-8")))
    header = next(reader)
    n_rows = sum(1 for _ in reader)
    extra = [c for c in header if c not in H.COLUMNS]
    missing = [c for c in H.COLUMNS if c not in header]
    if extra or missing:
        raise ContractError(f"clean_data.csv columns differ from the approved specification "
                            f"(extra {extra}, missing {missing})")
    roles = json.loads((contracts_dir / "column_roles.json").read_text(encoding="utf-8"))
    schema = json.loads((contracts_dir / "dataset_contract.schema.json").read_text(encoding="utf-8"))
    desc = validate_descriptions(descriptions, header)
    columns = []
    for name in header:
        dtype, role, unit = H.COLUMNS[name]
        spec = {"name": name, "dtype": dtype, "role": role, "unit": unit, "nullable": False}
        if name in H.ALLOWED:
            spec["allowed"] = sorted(H.ALLOWED[name])
        lo, hi = H.RANGES.get(name, (None, None))
        if lo is not None:
            spec["min"] = lo
        if hi is not None:
            spec["max"] = hi
        if name in desc:
            spec["description"] = desc[name]
        columns.append(spec)
    contract = {
        "dataset": DATASET,
        "version": VERSION,
        "raw_manifest_sha256": raw_manifest_sha256,
        "primary_key": list(H.PRIMARY_KEY),
        "row_count": n_rows,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "columns": columns,
        "split_policy": SPLIT_POLICY,
        "outcome_only_exclusions": sorted(outcome_only(roles)),
    }
    problems = schema_errors(contract, schema)
    if problems:
        raise ContractError("contract does not satisfy the meta-contract: " + "; ".join(map(str, problems[:5])))
    return contract


def structural(contract: dict) -> dict:
    """The contract without the free-text column descriptions."""
    out = json.loads(json.dumps(contract))
    for c in out.get("columns", []):
        if isinstance(c, dict):
            c.pop("description", None)
    return out


def write_contract(ws: Workspace, contract: dict) -> None:
    ws.write("out.dataset_contract", dumps(contract))
