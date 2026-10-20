"""Crew 2 input (plan section b): the in-process handoff is the expected deterministic output.
Offline. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from crew2.ml import data as D
from crew2.validators import gate_on
from gate.handoff_spec import COLUMNS
from tests.crew2_helpers import no_network  # noqa: F401
from tests.handoff_reference import reference_frames

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture(scope="module")
def handoff():
    return D.build_handoff()


def test_the_in_process_handoff_is_the_pinned_deterministic_output(handoff):
    text, log = handoff
    info = D.verify_handoff(text, log)
    assert info["clean_data_sha256"] == D.EXPECTED_CLEAN_SHA256
    assert info["rows"] == 1538 and log["snapshots"] == {"scheduled": 1600, "eligible": 1538, "ineligible": 62}
    assert log["exclusion_counts"] == {"commitment_category_missing": 50, "manual_cost_undated": 12}
    again, _ = D.build_handoff()
    assert again == text                                    # byte-identical on a rebuild


def test_the_pinned_hash_is_backed_by_the_independent_reference(handoff, expected_ineligible):
    """The pinned output equals the reference built from the committed clean snapshots and the ground truth
    (never from the cleaning code), within the dataset's 2-decimal CSV rounding (0.01)."""
    text, _ = handoff
    got = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    ref, ref_contract = reference_frames(expected_ineligible)
    key = ["production_id", "stage"]
    got, ref = got.sort_values(key).reset_index(drop=True), ref.sort_values(key).reset_index(drop=True)
    assert list(got.columns) == list(COLUMNS) and set(ref.columns) == set(got.columns)
    assert len(got) == len(ref) == ref_contract["row_count"] == 1538
    assert (got[key] == ref[key]).all().all()
    for c in got.columns:
        if COLUMNS[c][0] == "number":
            assert (pd.to_numeric(got[c]) - pd.to_numeric(ref[c])).abs().max() <= 0.01 + 1e-9, c
        else:
            assert (got[c] == ref[c]).all(), c


def test_the_unchanged_p2_handoff_gate_passes_on_the_in_process_handoff(handoff):
    text, log = handoff
    g = gate_on(text, log)
    assert g["status"] == "PASS" and g["exit_code"] == 0 and g["failed"] == []


def test_any_change_to_the_handoff_stops_the_run(handoff):
    text, log = handoff
    with pytest.raises(D.HandoffMismatch):
        D.verify_handoff(text.replace("True", "False", 1), log)
    with pytest.raises(D.HandoffMismatch):
        D.verify_handoff(text, log | {"exclusion_counts": {"manual_cost_undated": 62}})


def test_partition_drops_test_and_demo_labels_at_the_text_level(handoff):
    text, _ = handoff
    p = D.partition(text)
    for frame in (p.test_features, p.demo_features):
        assert not set(D.LABEL_COLUMNS) & set(frame.columns)
    assert set(D.LABEL_COLUMNS) <= set(p.train.columns)
    assert (p.train["split"] == "train").all() and (p.test_features["split"] == "test").all()
    assert sum(c["rows"] for c in p.counts.values()) == 1538
    assert p.counts["train"]["productions"] + p.counts["test"]["productions"] + p.counts["demo"]["productions"] == 397
    assert not p.vault.opened


def test_label_fields_are_parsed_only_for_train_rows(handoff, monkeypatch):
    text, _ = handoff
    seen = []
    real = D._typed
    monkeypatch.setattr(D, "_typed", lambda c, v: (seen.append(c) if c in D.LABEL_COLUMNS else None) or real(c, v))
    p = D.partition(text)
    assert len(seen) == len(p.train) * len(D.LABEL_COLUMNS)


def test_the_vault_opens_once_and_only_for_frozen_models(handoff):
    text, _ = handoff
    p = D.partition(text)
    with pytest.raises(ValueError):
        p.vault.open({})
    assert not p.vault.opened
    labels = p.vault.open({"primary": "a" * 64})
    assert len(labels) == len(p.test_features)
    full = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    exp = full[full["split"] == "test"].sort_values(["production_id", "stage"])
    assert labels["is_overrun"].tolist() == (exp["is_overrun"] == "True").tolist()
    with pytest.raises(D.TestAlreadyOpened):
        p.vault.open({"primary": "a" * 64})
