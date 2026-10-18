"""Raw export: defects, leakage safety, ground-truth isolation and clean reproduction (B1-B8)."""

import copy
import subprocess
from datetime import datetime

import pandas as pd

from generator import config as C
from generator.cli import RAW_FILES, generate, write_raw
from generator.raw import PLAN, RAW_COLUMNS, inject
from tests.conftest import ACTIVE_DATA, REPO
from tests.raw_repair import DATES, KEYS, repair_table
from tests.reconstruct import FEATURE_RULES, LABELS, as_csv_text, features_at, index

TARGET_LIKE = {"final_cost_pre_vat", "budget_gap_ils", "is_overrun", "is_material_overrun",
               "final_company_profit_pre_vat", "final_fee_contract_pre_vat", "split", "cv_fold"}
FORBIDDEN_REFERENCES = ("ground_truth", "ground truth", "defect", "clean", "snapshots.csv", "../",
                        "data/synthetic/v2/", "split_report", "target_audit")


def _clean_text(name):
    src = "snapshots" if name == "snapshot_schedule" else name
    return pd.read_csv(ACTIVE_DATA / f"{src}.csv", dtype=str, keep_default_na=False)


def test_raw_contains_no_target_or_label_columns(raw_text):
    for name, df in raw_text.items():
        assert TARGET_LIKE.isdisjoint(df.columns), name
    # the snapshot schedule carries no features at all: only keys, provenance and the snapshot date
    assert list(raw_text["snapshot_schedule"].columns) == ["production_id", "batch_id", "is_synthetic",
                                                           "stage", "as_of_date"]


def test_raw_files_and_dictionary_do_not_reference_clean_data_or_ground_truth():
    texts = {p.name: p.read_text(encoding="utf-8").lower() for p in (ACTIVE_DATA / "raw").iterdir()}
    texts["raw_data_dictionary.md"] = (REPO / "docs" / "raw_data_dictionary.md").read_text(encoding="utf-8").lower()
    for name, text in texts.items():
        for token in FORBIDDEN_REFERENCES:
            assert token not in text, (name, token)


def test_raw_is_labelled_synthetic(raw_text):
    for name, df in raw_text.items():
        assert (df["is_synthetic"].str.strip() == "True").all(), name
        assert (df["batch_id"].str.strip() == C.BATCH_ID).all(), name


def test_defect_selection_is_independent_of_outcomes(tables_mem):
    """Changing every outcome value leaves the selected defects identical."""
    ledgers = {n: tables_mem[n] for n in RAW_FILES}
    _, log_a = inject(ledgers, C.SEED)
    altered = copy.deepcopy(ledgers)
    altered["commitments"]["final_pre_vat"] = altered["commitments"]["final_pre_vat"] * 1.7 + 11
    altered["productions"]["fee_concession_pre_vat"] = 0.0
    _, log_b = inject(altered, C.SEED)
    sig = lambda log: [(d["table"], d["record_id"], d["defect_type"], d["column"]) for d in log]
    assert sig(log_a) == sig(log_b)
    for name in RAW_FILES:
        assert TARGET_LIKE.isdisjoint(ledgers[name][RAW_COLUMNS[name] or list(ledgers[name].columns)].columns)


def test_defect_incidence_is_not_driven_by_the_label(defect_log, generated):
    s0 = generated["snapshots"][generated["snapshots"]["stage"] == "S0"].set_index("production_id")["is_overrun"]
    recs = generated["commitments"][["commitment_id", "production_id"]]
    hit = set(defect_log.loc[defect_log["table"] == "commitments", "record_id"])
    recs = recs.assign(defect=recs["commitment_id"].isin(hit),
                       overrun=recs["production_id"].map(s0).astype(str) == "True")
    rates = recs.groupby("overrun")["defect"].mean()
    assert abs(rates[True] - rates[False]) < 0.02, rates.to_dict()


def test_no_defect_moves_a_date(defect_log):
    def parse(v):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y"):
            try:
                return datetime.strptime(v.strip(), fmt).date().isoformat()
            except ValueError:
                pass
        return None
    dated = defect_log[defect_log["column"].isin(DATES)]
    assert set(dated["defect_type"]) <= {"date_format", "missing_value"}
    fmt = dated[dated["defect_type"] == "date_format"]
    assert len(fmt) > 0
    assert all(parse(d) == o for d, o in zip(fmt["defective_value"], fmt["original_value"]))
    assert (dated.loc[dated["defect_type"] == "missing_value", "defective_value"] == "").all()


def test_defect_rates_follow_the_declared_rates(defect_log, generated):
    """Per table, the observed per-record defect rate stays within the declared total rate + 3 s.e."""
    for table, specs in PLAN.items():
        n = len(generated["snapshots"]) if table == "snapshot_schedule" else len(generated[table])
        declared = sum(spec[2] for spec in specs)
        observed = (defect_log["table"] == table).sum() / n
        bound = declared + 3 * (declared * (1 - declared) / n) ** 0.5
        assert observed <= bound, (table, observed, declared)
    nonrec_declared = max(sum(s[2] for s in specs if not s[3]) for specs in PLAN.values())
    nonrec = defect_log[defect_log["recoverable"] == "False"]
    assert len(nonrec) > 0
    assert (nonrec.groupby("table").size() / pd.Series(
        {t: len(generated[t]) for t in nonrec["table"].unique()})).max() <= 2 * nonrec_declared


def test_clean_reproduction_from_raw(raw_text, defect_log, generated):
    """B8: recoverable records reproduce the clean data exactly; the rest are excluded as documented."""
    repaired, excluded = {}, {}
    for name in RAW_FILES:
        rep, exc = repair_table(name, raw_text[name])
        expected_exc = set(defect_log.loc[(defect_log["table"] == name) & (defect_log["recoverable"] == "False"),
                                          "record_id"])
        assert exc == expected_exc, name                     # exactly the documented exclusions, nothing more
        clean = _clean_text(name)[list(raw_text[name].columns)]
        clean["_k"] = clean[KEYS[name]].agg("#".join, axis=1)
        clean = clean[~clean["_k"].isin(exc)].sort_values("_k").drop(columns="_k").reset_index(drop=True)
        assert rep.reset_index(drop=True).equals(clean), name
        repaired[name], excluded[name] = rep, exc

    # Snapshot features and labels rebuilt from the repaired raw ledgers, for every production
    # not affected by an excluded cost record.
    affected = {k.split("-")[0] + "-" + k.split("-")[1] for t in ("commitments", "manual_costs") for k in excluded[t]}
    idx = index({n: repaired[n] for n in
                 ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs")})
    clean_snaps = pd.read_csv(ACTIVE_DATA / "snapshots.csv", dtype=str, keep_default_na=False)
    schedule = {(r["production_id"], r["stage"]): r["as_of_date"] for r in repaired["snapshot_schedule"].to_dict("records")}
    reproduced = 0
    for r in clean_snaps.to_dict("records"):
        if r["production_id"] in affected:
            continue
        f = features_at(idx, r["production_id"], r["stage"], schedule[(r["production_id"], r["stage"])])
        for col in list(FEATURE_RULES) + list(LABELS):
            assert as_csv_text(f[col]) == r[col], (r["production_id"], r["stage"], col)
        reproduced += 1
    assert reproduced == 4 * (400 - len(affected))
    assert 0 < len(affected) < 40


def test_ground_truth_is_regenerable_and_git_ignored(generated, tmp_path):
    _, log_again = write_raw(generate(C.SEED), tmp_path / "raw", C.SEED)
    first = pd.read_csv(generated["gt_dir"] / "defect_log.csv", dtype=str, keep_default_na=False)
    assert len(first) == len(log_again)
    for path in ("ground_truth/v2/defect_log.csv", "ground_truth/v2/defect_catalogue.json",
                 "private/process_log.md", "scan_output/x.json"):
        res = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", path])
        assert res.returncode == 0, f"{path} is not git-ignored"
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "ground_truth", "private", "scan_output"],
                             capture_output=True, text=True).stdout
    assert tracked.strip() == ""
