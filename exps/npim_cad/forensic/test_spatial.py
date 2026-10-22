# exps/npim_cad/forensic/test_spatial.py
"""G1 spatial-correctness tests: flat index, bilinear, orientation, non-square."""

from __future__ import annotations

import numpy as np

from exps.npim_cad.forensic.spatial import (
    delta_fixture,
    flat_index,
    geometric_audit,
    manual_bilinear,
    production_bilinear,
)


def test_flat_index_row_major():
    assert flat_index(0, 0) == 0
    assert flat_index(0, 1) == 1
    assert flat_index(1, 0) == 28
    assert flat_index(14, 14) == 14 * 28 + 14
    assert flat_index(27, 27) == 783


def test_manual_bilinear_matches_production():
    for height, width in ((32, 32), (45, 71), (7, 11)):
        source = np.random.default_rng(height + width).random((28, 28))
        manual = manual_bilinear(source, height, width)
        production = production_bilinear(source, height, width)
        assert np.abs(manual - production).max() <= 2e-6


def test_delta_orientation_and_non_square():
    cases = {
        "top_left": (0, 0, 32, 32),
        "bottom_right": (27, 27, 32, 32),
        "top_right": (0, 27, 45, 71),
        "bottom_left": (27, 0, 45, 71),
        "middle": (14, 9, 48, 64),
    }
    for name, (row, column, height, width) in cases.items():
        values = production_bilinear(delta_fixture(row, column), height, width)
        peak_row, peak_column = np.unravel_index(int(np.argmax(values)), values.shape)
        assert (peak_row < height / 2) == (row < 14), name
        assert (peak_column < width / 2) == (column < 14), name
        # Non-square width is independent of height.
        if height != width:
            assert peak_column <= width, name


def test_geometric_audit_cases_all_pass():
    audit = geometric_audit()
    assert audit["flat_index_ok"]
    assert audit["manual_vs_production_max_abs"] <= 2e-6
    assert audit["all_cases_ok"], audit["cases"]
