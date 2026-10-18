"""Shared fixtures. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from generator import config as C
from generator.cli import FILES, generate, write

REPO = Path(__file__).resolve().parents[1]
COMMITTED_DATA = REPO / "data" / "synthetic" / "v1"


@pytest.fixture(scope="session")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    manifest = write(generate(C.SEED), out, C.SEED)
    tables = {name: pd.read_csv(out / f"{name}.csv", keep_default_na=False) for name in FILES}
    return {"dir": out, "manifest": manifest, **tables}


@pytest.fixture(scope="session")
def snaps(generated):
    return generated["snapshots"]
