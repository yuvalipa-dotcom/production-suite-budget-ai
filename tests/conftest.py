"""Shared fixtures. ALL DATA IS SYNTHETIC.

Everything is generated at runtime into temporary directories, including the
ground truth (defect log and catalogue), which is never committed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from generator import config as C
from generator.cli import FILES_V2, RAW_FILES, generate, write, write_ground_truth, write_raw

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic"
COMMITTED_DATA = DATA / "v1"
ACTIVE_VERSION = "v2"
ACTIVE_DATA = DATA / ACTIVE_VERSION


def _read(path: Path, as_text: bool = False) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, **({"dtype": str} if as_text else {}))


@pytest.fixture(scope="session")
def tables_mem():
    return generate(C.SEED)


@pytest.fixture(scope="session")
def generated(tmp_path_factory, tables_mem):
    out = tmp_path_factory.mktemp("gen_v2")
    manifest = write(tables_mem, out, C.SEED, "v2")
    raw_manifest, log = write_raw(tables_mem, out / "raw", C.SEED)
    gt = tmp_path_factory.mktemp("ground_truth")
    write_ground_truth(log, gt)
    tables = {name: _read(out / f"{name}.csv") for name in FILES_V2}
    return {"dir": out, "manifest": manifest, "raw_manifest": raw_manifest, "gt_dir": gt, **tables}


@pytest.fixture(scope="session")
def snaps(generated):
    return generated["snapshots"]


@pytest.fixture(scope="session")
def raw_text(generated):
    return {name: _read(generated["dir"] / "raw" / f"{name}.csv", as_text=True) for name in RAW_FILES}


@pytest.fixture(scope="session")
def defect_log(generated):
    return _read(generated["gt_dir"] / "defect_log.csv", as_text=True)


@pytest.fixture(scope="session")
def expected_ineligible(defect_log, generated):
    """Expected ineligible snapshots, derived from ground truth + clean ledger dates (independent of the cleaner)."""
    from tests.raw_repair import COST_TABLES, STAGES

    snaps = generated["snapshots"]
    asof = {(r.production_id, r.stage): r.as_of_date for r in snaps.itertuples()}
    com = generated["commitments"].set_index("commitment_id")
    out: dict[tuple[str, str], set[str]] = {}
    nonrec = defect_log[(defect_log["recoverable"] == "False") & defect_log["table"].isin(COST_TABLES)]
    for d in nonrec.itertuples():
        pid = "-".join(d.record_id.split("-")[:2])
        if d.table == "commitments":
            c = com.loc[d.record_id]
            for s in STAGES:
                a = asof[(pid, s)]
                if c.selected_date <= a and (c.cancelled_date == "" or c.cancelled_date > a):
                    out.setdefault((pid, s), set()).add(f"commitment_{d.column}_missing")
        elif d.column == "created_date":
            for s in STAGES:
                out.setdefault((pid, s), set()).add("manual_cost_undated")
    return {k: "+".join(sorted(v)) for k, v in out.items()}
