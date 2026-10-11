# exps/cms_nm01/test_forensic.py
"""Synthetic regression checks for the two confirmed reporting/protocol bugs."""

import math

import numpy as np
import pytest
import torch

from exps.cms_nm01.cms_nd import mixture_nll
from exps.cms_nm01.reporting import level_rows, markdown_level_table
from exps.cms_nm01.run import CATEGORIES, ORDER_SEEDS, evaluate_gates


def test_mixture_pointwise_and_mean_bounds_with_crossing_experts():
    nll = torch.tensor([[[1., 900., 2.], [3., 4., 600.]],
                        [[200., 2., 3.], [4., 600., 5.]],
                        [[400., 300., 4.], [5., 6., 7.]]], dtype=torch.float64)
    mixed = mixture_nll(tuple(-value for value in nll))
    direct = -torch.log(torch.exp(-nll).mean(0))
    assert torch.allclose(mixed, direct, atol=1e-12, rtol=0)
    assert torch.all(mixed >= nll.amin(0) - 1e-12)
    assert torch.all(mixed <= nll.amin(0) + math.log(3) + 1e-12)
    assert mixed.mean() <= nll.mean((1, 2)).min() + math.log(3) + 1e-12
    # Mixture-before-reduction differs from mixing the level means.
    assert mixed.mean() < mixture_nll(tuple(-value.mean() for value in nll)) - 1


def test_category_level_table_axes_are_explicit():
    record = {"order_seed": 0, "arm": "CMS-ND-3", "stages": [{"stage": 3, "metrics": {"per_category": {
        "bottle": {"level_mean": [112., 68., 80.], "mixture_mean": 63.},
        "carpet": {"level_mean": [55., 71., 81.], "mixture_mean": 56.},
        "hazelnut": {"level_mean": [32., 65., 75.], "mixture_mean": 33.},
    }}}]}
    rows = level_rows(record)
    assert [(row["category"], row["fast"], row["medium"], row["slow"]) for row in rows] == [
        ("bottle", 112., 68., 80.), ("carpet", 55., 71., 81.), ("hazelnut", 32., 65., 75.)]
    assert "bottle | 63.000000 | 112.000000 | 68.000000 | 80.000000" in markdown_level_table([record])
    record["stages"][0]["metrics"]["per_category"]["bottle"]["mixture_mean"] = 75.
    with pytest.raises(ValueError):
        level_rows(record)


def test_written_gate_is_distinct_from_undeclared_post_acquisition_rule():
    def metrics(mean):
        return {"per_category": {category: {"image_nll": [mean]*20, "mixture_mean": mean} for category in CATEGORIES},
                "finite_state": True, "representation": {"finite": True, "projection_effective_rank": 20., "level_scale_saturation": [0., 0., 0.]}}
    records = []
    for seed in ORDER_SEEDS:
        for arm in ("CMS-ND-3", "CMS-ND-FROZEN", "GAUSS-64"):
            final = metrics(80. if arm == "CMS-ND-3" else 100.)
            records.append({"arm": arm, "order_seed": seed, "order": list(CATEGORIES), "calibration_metrics": metrics(100.),
                            "final_metrics": final, "stages": [{"stage": 0, "metrics": metrics(100.)}] + [
                                {"stage": stage, "metrics": metrics(50. if stage < 3 and arm == "CMS-ND-3" else final["per_category"]["bottle"]["mixture_mean"])} for stage in (1,2,3)]})
    result = evaluate_gates(records)
    assert result["g1"]["pass"]
    assert result["g2"]["pass"]
    assert not result["g2"]["post_acquisition_diagnostic_pass_at_20pct"]
