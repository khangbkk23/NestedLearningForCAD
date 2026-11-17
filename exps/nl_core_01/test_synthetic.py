from __future__ import annotations

from .synthetic import POLICIES, ORDERS, TOPOLOGIES, run_one


def test_all_controls_have_boundary_matrices_and_finite_state():
    for topology in TOPOLOGIES:
        for order in ORDERS:
            for policy in POLICIES:
                result = run_one(policy, topology, order, seed=3, repeats=4)
                assert result["finite"]
                assert len(result["boundaries"]) == 4
                assert all(len(row) == 3 for row in result["boundaries"])
                assert len(result["acquisition"]) == 3
                assert len(result["forgetting"]) == 3
