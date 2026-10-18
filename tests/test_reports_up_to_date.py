"""Generated documentation sections must match the committed dataset (no hand-typed numbers)."""

import importlib.util

from tests.conftest import REPO

spec = importlib.util.spec_from_file_location("reports", REPO / "tools" / "reports.py")
reports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reports)


def test_generated_sections_are_up_to_date():
    for rel, blocks in reports.SECTIONS.items():
        path = REPO / rel
        assert reports.render(path, blocks) == path.read_text(encoding="utf-8"), f"{rel} is out of date"


def test_wilson_interval_reference_values():
    lo, hi = reports.wilson(50, 100)
    assert round(lo, 4) == 0.4038 and round(hi, 4) == 0.5962
    lo, hi = reports.wilson(0, 10)
    assert lo < 1e-12 and round(hi, 4) == 0.2775
