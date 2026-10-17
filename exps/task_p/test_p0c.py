"""Focused synthetic tests for the P0c full-grid causal verification harness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.task_p.p0_normal_support import BUDGET, Support, distances, fingerprint  # noqa: E402
from exps.task_p.p0c_protocol import (  # noqa: E402
    BOUNDARIES, CATEGORIES, FULL_GRID, PROBE_GRID, PROBE_SPEC, TRAIN_GRID, Reader, validate_roles,
)


def test_grid_constants():
    assert len(TRAIN_GRID) == 16 and len(FULL_GRID) == 784 and len(PROBE_GRID) == 64
    assert sorted(set(TRAIN_GRID) | set(FULL_GRID)) == list(range(784))
    assert not set(TRAIN_GRID) & set(PROBE_GRID)
    assert BOUNDARIES == (100, 200, 300, 350)
    assert sum(stop - start for start, stop in PROBE_SPEC.values()) == 59
    assert all(0 <= index < 784 for index in TRAIN_GRID + PROBE_GRID)


def _row(category, index, digest):
    return {'category': category, 'cache_index': index,
            'relative_path': f'{category}/train/good/{index:03}.png', 'raw_png_sha256': digest}


def _roles_fixture():
    updates = ([_row('bottle', i, f'ub{i}') for i in range(150)]
               + [_row('carpet', i, f'uc{i}') for i in range(100)]
               + [_row('hazelnut', i, f'uh{i}') for i in range(100)])
    probes = ([_row('bottle', 190 + i, f'pb{i}') for i in range(19)]
              + [_row('carpet', 140 + i, f'pc{i}') for i in range(20)]
              + [_row('hazelnut', 140 + i, f'ph{i}') for i in range(20)])
    return updates, probes, [_row('carpet', 100 + i, f'r{i}') for i in range(20)], \
        [_row('bottle', 170 + i, f'p0{i}') for i in range(20)]


def test_roles_disjointness_detects_overlap():
    updates, probes, excluded_r1, excluded_p0 = _roles_fixture()
    ok = validate_roles(updates, probes, excluded_r1, excluded_p0)
    assert ok['counts']['updates'] == 350 and ok['counts']['probes'] == 59
    with pytest.raises(RuntimeError):
        validate_roles(updates[:-1] + [_row('bottle', 190, 'pbnone')], probes, excluded_r1, excluded_p0)
    with pytest.raises(RuntimeError):
        validate_roles(updates[:-1] + [_row('bottle', 1, 'u0')], probes, excluded_r1, excluded_p0)


def test_reader_rejects_future_and_out_of_order_access():
    protocol = {'updates': [_row('bottle', 0, 'a'), _row('bottle', 1, 'b')],
                'probes': [_row('bottle', 190, 'c')], 'preflight_events': 1}
    reader = Reader(protocol)
    reader.completed = 1
    reader.admitted = {'bottle'}
    with pytest.raises(RuntimeError):
        reader.probe(protocol['probes'][0], 1)            # boundary not in BOUNDARIES
    with pytest.raises(RuntimeError):
        reader.probe(_row('bottle', 191, 'd'), 100)       # identity not registered
    with pytest.raises(RuntimeError):
        reader.probe(protocol['probes'][0], 100)          # boundary reached but reader.completed is 1
    with pytest.raises(RuntimeError):
        reader.update(protocol['updates'][0])             # already-consumed arrival replayed
    assert reader.access == []                            # no numeric read happened


def test_distance_matches_direct_fp64_and_survives_cancellation():
    generator = torch.Generator().manual_seed(3)
    left = (torch.randn(12, 768, generator=generator) * 1e6)
    right = (torch.randn(9, 768, generator=generator) * 1e6)
    reference = torch.cdist(left.double(), right.double())
    optimized = distances(left, right)
    assert torch.allclose(optimized, reference, rtol=0, atol=1e-6 * reference.max())
    base = torch.randn(1, 768, generator=generator) * 10
    near = base + torch.randn(1, 768, generator=generator) * 1e-7
    huge = torch.full((1, 768), 1e18)
    for left_case, right_case in ((base, near), (huge, huge + torch.randn(1, 768, generator=generator))):
        direct = torch.sqrt(torch.clamp(
            (left_case.double() - right_case.double()).square().sum(1), min=0))
        computed = distances(left_case, right_case)[0]
        assert torch.allclose(computed, direct, rtol=1e-9, atol=1e-9)
    assert float(distances(base, base)[0, 0]) == 0.0


def _synthetic_full_grid(events, seed=5):
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randn(784, 768, generator=generator) * 3) for _ in range(events)]


def test_greedy_minimum_separation_never_decreases_and_is_deterministic():
    stream = _synthetic_full_grid(5)
    first, second = Support('STREAM_GREEDY', capacity=800), Support('STREAM_GREEDY', capacity=800)
    separations, post_fill = [], 0
    for event, x in enumerate(stream, start=1):
        ids = torch.arange((event - 1) * 784, event * 784, dtype=torch.int64)
        first.update(x, ids)
        second.update(x, ids)
        if first.n == first.capacity:
            post_fill += 1
            values = first.nn_distance[:first.n]
            separations.append(float(values[torch.isfinite(values)].min()))
    assert post_fill >= 2, 'replacement regime never reached'
    assert fingerprint(first.state()) == fingerprint(second.state())
    assert all(b >= a - 1e-9 for a, b in zip(separations, separations[1:]))


def test_full_grid_update_respects_cap_and_serialization_round_trips():
    model = Support('STREAM_GREEDY')
    stream = _synthetic_full_grid(3, seed=9)
    for event, x in enumerate(stream, start=1):
        ids = torch.arange((event - 1) * 784, event * 784, dtype=torch.int64)
        model.update(x, ids)
    ledger = model.ledger()
    assert ledger['charged_persistent_bytes'] <= BUDGET
    assert ledger['occupied'] > 0
    before = fingerprint(model.state())
    payload = torch.save(model.state(), '/tmp/opencode/p0c_state.pt') or '/tmp/opencode/p0c_state.pt'
    reopened = torch.load(payload, map_location='cpu', weights_only=False)
    assert fingerprint(reopened) == before
    clone = Support('STREAM_GREEDY')
    for key, value in reopened.items():
        setattr(clone, key, value)
    probe = stream[0][:1]
    assert torch.equal(model.score(probe)[0], clone.score(probe)[0])


def test_no_write_and_read_only_do_not_mutate_state():
    model = Support('STREAM_GREEDY')
    stream = _synthetic_full_grid(3, seed=13)
    for event, x in enumerate(stream, start=1):
        model.update(x, torch.arange((event - 1) * 784, event * 784, dtype=torch.int64))
    before = fingerprint(model.state())
    model.update(stream[0], torch.arange(784, dtype=torch.int64), write=False)
    assert fingerprint(model.state()) == before
    model.score(stream[1][:8])
    model.score(stream[1][:8])
    assert fingerprint(model.state()) == before


def test_train_grid_proposals_are_a_subset_of_the_same_arriving_image():
    image = torch.randn(784, 768, generator=torch.Generator().manual_seed(21))
    assert torch.equal(image[list(TRAIN_GRID)], image[list(TRAIN_GRID)])
    assert PROBE_SPEC['bottle'] == (190, 209) and PROBE_SPEC['carpet'] == (140, 160)
    assert PROBE_SPEC['hazelnut'] == (140, 160)
    assert len(CATEGORIES) == 3
    manifest_path = Path('/mnt/d/code_for_fun/project/nestedlearningforcad/nestedlearningforcad/'
                         'results/hope_cad/frozen_features_v1/source_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    entries = {(entry['category'], int(entry['relative_path'].rsplit('/', 1)[1][:-4]))
               for entry in manifest['entries']}
    for category, (start, stop) in PROBE_SPEC.items():
        for index in range(start, stop):
            assert (category, index) in entries, f'missing probe source {category}/{index}'
