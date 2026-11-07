"""CPU tests for the M4 full-stream summary (synthetic inputs; no GPU, no MVTec label)."""
import numpy as np
import pytest

from exps.nl_align_02 import m4


def _fake():
    names = m4.M1.acquired(15)
    ap, iau, q99, boot, pool, acqb = {}, {}, {}, {}, {}, {}
    for sd in m4.SEEDS:
        for a, s, c in m4.cells(sd):
            j = names.index(c) + 1
            v = 0.6 - 0.01 * (s - j) + (0.003 if a == "FS_B" else 0.0)
            ap[(a, sd, s, c)], iau[(a, sd, s, c)], q99[(a, sd, s, c)] = v, 0.95, 0.2
            boot[(a, sd, s, c)] = np.full(m4.M1.REPS, v)
            pool[(a, sd, s)] = [np.array([1., 0.]), np.array([2., 2.])]
        for a in m4.ARMS:
            for c in names:
                acqb[(a, sd, c)] = (0.6, np.full(m4.M1.REPS, 0.6))
    return ap, iau, q99, boot, pool, acqb, names


def test_cells_have_no_future_categories():
    order = m4.M1.acquired(15)
    for a, s, c in m4.cells(2711):
        assert order.index(c) < s
    assert len(m4.cells(2711)) == 3 * sum(range(10, 16))


def test_summary_final_contrast_and_forgetting():
    out = m4._summarize(*_fake())
    d = out["contrasts"]["FS_B-ID"]["final"]
    assert d["delta"] == pytest.approx(0.003) and d["ci95"] == pytest.approx([0.003, 0.003])
    f = out["forgetting_acq_to_final"]["ID"]
    # F_c = 0.6 - (0.6 - 0.01 (15 - j)) = 0.01 (15 - j), j = 1..14
    assert f["macro_F"] == pytest.approx(np.mean([0.01 * (15 - j) for j in range(1, 15)]))
    assert "zipper" not in f["per_category"]
    assert out["forgetting_acq_to_final"]["FS_B-ID"]["delta_macro_F"] == pytest.approx(-0.003)
