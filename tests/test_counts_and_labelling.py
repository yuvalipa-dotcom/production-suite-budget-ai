"""Counts, synthetic labelling and provenance."""

from generator import config as C


def test_counts(generated, snaps):
    prods = generated["productions"]
    assert len(prods) == 400
    assert prods["production_id"].is_unique
    assert len(snaps) == 400 * 4 == 1600
    assert snaps.groupby("production_id").size().eq(4).all()
    assert set(snaps["stage"]) == set(C.STAGES)
    assert not snaps.duplicated(["production_id", "stage"]).any()
    assert prods["archetype"].value_counts().to_dict() == {
        "wedding": 100, "bar_bat_mitzvah": 100, "corporate": 100, "conference": 100}


def test_manifest_counts(generated):
    counts = generated["manifest"]["counts"]
    assert counts["productions"] == 400
    assert counts["snapshots_per_production"] == 4
    assert counts["snapshot_rows"] == 1600


def test_every_table_is_labelled_synthetic(generated):
    for name in ("snapshots", "productions", "commitments", "manual_costs", "payments"):
        df = generated[name]
        assert df["is_synthetic"].astype(str).eq("True").all(), name
        assert df["batch_id"].eq(C.BATCH_ID).all(), name
        assert df["production_id"].str.match(r"^SYN-\d{4}$").all(), name


def test_manifest_declares_synthetic(generated):
    m = generated["manifest"]
    assert m["synthetic"] is True
    assert "SYNTHETIC" in m["notice"]
    assert m["seed"] == C.SEED
    assert m["batch_id"] == C.BATCH_ID


def test_hidden_latent_factors_are_never_exported(generated):
    for name in ("snapshots", "productions", "commitments", "manual_costs", "payments"):
        cols = " ".join(generated[name].columns).lower()
        for latent in ("volatility", "discipline", "complexity", "season", "latent"):
            assert latent not in cols, (name, latent)
