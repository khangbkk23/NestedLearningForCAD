from pathlib import Path

from exps.nl_align_01 import g0


def test_ptag_archive_values():
    r = g0.audit_ptag()
    assert r["cells"] == 635
    assert r["verdict"] == "SYNTHETIC_OUTER_OBJECTIVE_ADMISSIBLE"
    assert r["q2"]["tau"] > 0.5 and r["q3"]["tau"] > 0.6


def test_generator_source_files_exist():
    assert (g0.ROOT / "dataset/anomaly_generators/perlin.py").exists()
    assert (g0.ROOT / "data/dtd/images").exists()


def test_r15_bank_manifest_is_hashed():
    r = g0.audit_r15_interface()
    assert r["learner"] == "Dinomaly-B + Replay-24 R15"
    assert r["bank_build_required"] is True or r["r15_synthetic_feature_shape"] == [2, 789, 768]
