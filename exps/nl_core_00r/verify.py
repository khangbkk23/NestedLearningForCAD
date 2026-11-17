"""Dependency-light checks for NL-CORE-00R."""

from __future__ import annotations

import torch

from .forensics import constant_key_fixed_point, run_stream, throughput


def main() -> None:
    fp = constant_key_fixed_point(alpha=1.0, eta=0.1, steps=250)
    assert abs(fp["observed_final"] - 0.5) < 1e-8
    assert abs(fp["perfect_association_update"] - 0.9) < 1e-8
    for conflicting in (False, True):
        rows = {name: run_stream(name, conflicting=conflicting, repeats=12)
                for name in ("nl", "fixed_controller", "frozen_controller", "delta_rule", "nonnested_controller")}
        assert all(row["finite"] for row in rows.values())
        assert all(len(row["probe_matrix"]) == 4 and len(row["probe_matrix"][0]) == 3 for row in rows.values())
        assert all(row["persistent_bytes"] == rows["nl"]["persistent_bytes"] for row in rows.values())
    speed = throughput(dim=768, steps=4)
    assert speed["persistent_bytes_fp32"] == 4 * (768 * 768 + 2 * 769) + 8
    assert speed["nl_updates_per_second"] > 0 and speed["delta_updates_per_second"] > 0
    print("NL-CORE-00R verification: PASS")
    print("fixed_point=PASS controls=PASS matrices=PASS bytes=PASS throughput=PASS")


if __name__ == "__main__":
    main()
