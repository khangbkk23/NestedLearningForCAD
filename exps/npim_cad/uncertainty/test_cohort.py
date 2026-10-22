import numpy as np
from collections import Counter

from exps.npim_cad.uncertainty.cohort import (
    BOUNDARY_CATEGORIES,
    audit,
    eligible,
    fit_records,
    load_normal_image,
    load_probes,
    witness_records,
)


def test_probe_counts_and_split():
    records = load_probes()
    assert len(records) == 59
    assert Counter(record.category for record in records) == {"bottle": 19, "carpet": 20, "hazelnut": 20}
    assert all(record.split in ("fit", "witness") for record in records)
    for category in ("bottle", "carpet", "hazelnut"):
        subset = [record for record in records if record.category == category]
        assert sum(record.split == "fit" for record in subset) == len(subset) // 2
    for boundary in BOUNDARY_CATEGORIES:
        fit = fit_records(records, boundary)
        witness = witness_records(records, boundary)
        assert fit and witness
        assert not ({record.relative_path for record in fit} & {record.relative_path for record in witness})
        assert all(record.category in eligible(boundary) for record in fit + witness)
    assert len(fit_records(records, 100)) == 9
    assert len(witness_records(records, 100)) == 10
    assert len(fit_records(records, 200)) == 19
    assert len(witness_records(records, 200)) == 20
    assert len(fit_records(records, 300)) == 29
    assert len(witness_records(records, 300)) == 30


def test_probes_are_normal_train_good_and_disjoint_paths():
    records = load_probes()
    for record in records:
        parts = record.relative_path.split("/")
        assert parts[1:3] == ["train", "good"]
        assert "test" not in parts
        assert len(record.sha256) == 64
    assert len({record.sha256 for record in records}) == 59


def test_split_digest_is_deterministic_and_audit_matches():
    first = audit(load_probes())
    second = audit(load_probes())
    assert first["split_digest"] == second["split_digest"]
    assert first["fit_witness_by_boundary"]["100"] == {"categories": ["bottle"], "fit": 9, "witness": 10}
    assert first["fit_witness_by_boundary"]["300"] == {
        "categories": ["bottle", "carpet", "hazelnut"], "fit": 29, "witness": 30
    }


def test_load_normal_image_geometry_and_content():
    record = load_probes()[0]
    image = load_normal_image(record)
    assert tuple(image.shape) == (3, 224, 224)
    assert np.isfinite(image.numpy()).all()
    assert float(image.min()) > -3.0 and float(image.max()) < 3.0
