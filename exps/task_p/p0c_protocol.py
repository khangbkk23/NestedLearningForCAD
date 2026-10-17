"""P0c — full-grid causal verification of the observation-grid bottleneck.

One bounded, preregistered, three-arm causal experiment under the P0 8-MiB
online-state cap, using the identical frozen ViT-B/8 clean caches and the
identical 350-image arrival chronology:

  A  GRID16_GREEDY_REFERENCE : P0's exact streaming greedy on 16 positions (control; must
                               reproduce the saved P0 STREAM_GREEDY states bitwise)
  B  FULL784_GREEDY          : the identical algorithm fed all 784 patch positions
  C  FULL784_RESERVOIR       : Algorithm R over all arriving 784-position descriptors

Only this module may open the three allowed clean caches.  New P0c probes
(bottle 190-208, carpet 140-159, hazelnut 140-159) are numerically read only at
their registered read-only boundaries, never by the learners.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p0_normal_support import (
    ARMS as P0_ARMS, BUDGET, GREEDY_CAPACITY, RESERVOIR_CAPACITY, Support,
    distances, fingerprint, require, serialized_bytes,
)

ROOT = Path('/mnt/d/code_for_fun/project/nestedlearningforcad/nestedlearningforcad')
FEATURES = ROOT / 'results/hope_cad/frozen_features_v1'
OUT = ROOT / 'results/hope_cad/task_p/p0c'
P0 = ROOT / 'results/hope_cad/task_p/p0'
CATEGORIES = ('bottle', 'carpet', 'hazelnut')
TRAIN_GRID = tuple(r * 28 + c for r in (3, 10, 17, 24) for c in (3, 10, 17, 24))
FULL_GRID = tuple(range(784))
PROBE_GRID = tuple(r * 28 + c for r in (1, 4, 8, 11, 15, 18, 22, 26) for c in (1, 4, 8, 11, 15, 18, 22, 26))
CP_SHA = '755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415'
SOURCE_DIGEST = '060579457d344992eef159b1d4f1a86c34dd9283b97b70d3c3a386006b0f4dcc'
CACHE_SHA = {
    'clean/bottle_train.pt': 'ef4f3180d2381b64b4cdafea447b9df254b7522b9338f24992f4a857114ad6b3',
    'clean/carpet_train.pt': 'f3b1957d7b9d12add45c4ff9c33a5f9f5e237ff7ce6dbdb7693c81527c89b179',
    'clean/hazelnut_train.pt': '7bd97accb4a927285881d8c8b61cbc5db8106e57e2ceee1fe623f5903f62d180',
}
PROBE_SPEC = {'bottle': (190, 209), 'carpet': (140, 160), 'hazelnut': (140, 160)}
EXCLUDED_R1 = {'bottle': (150, 170), 'carpet': (100, 120), 'hazelnut': (100, 120)}
EXCLUDED_P0 = {'bottle': (170, 190), 'carpet': (120, 140), 'hazelnut': (120, 140)}
ARMS = ('GRID16_GREEDY_REFERENCE', 'FULL784_GREEDY', 'FULL784_RESERVOIR')
ARM_GRID = {'GRID16_GREEDY_REFERENCE': TRAIN_GRID, 'FULL784_GREEDY': FULL_GRID, 'FULL784_RESERVOIR': FULL_GRID}
ARM_METHOD = {'GRID16_GREEDY_REFERENCE': 'STREAM_GREEDY', 'FULL784_GREEDY': 'STREAM_GREEDY',
              'FULL784_RESERVOIR': 'RESERVOIR'}
BOUNDARIES = (100, 200, 300, 350)
CODE = ('exps/task_p/p0c_protocol.py', 'exps/task_p/analyze_p0c.py', 'exps/task_p/test_p0c.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def environment():
    return {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'device': 'cpu', 'threads': torch.get_num_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled()}


def rows_for(source, category, start, stop, role):
    entries = {entry['relative_path']: entry for entry in source['entries']}
    out = []
    for index in range(start, stop):
        path = f'{category}/train/good/{index:03}.png'
        require(path in entries, f'missing approved identity: {path}')
        entry = entries[path]
        require(entry['category'] == category and entry['category_index'] in (0, 3, 5), 'category mismatch')
        out.append({'category': category, 'cache_index': index, 'relative_path': path,
                    'raw_png_sha256': entry['raw_png_sha256'], 'role': role})
    return out


def reconstruct_updates(source):
    """The original seeded within-phase P0 arrival order, reconstructed independently."""
    from scripts.exps.hope_image_synchronous_memory import manifest
    rows, probes, availability = manifest(0)
    by_path = {entry['relative_path']: entry for entry in source['entries']}
    reconstructed = []
    for position, row in enumerate(rows, start=1):
        path = row['relative_path']
        reconstructed.append({'category': row['class_name'], 'cache_index': row['cache_index'],
                              'relative_path': path, 'event': position,
                              'raw_png_sha256': by_path[path]['raw_png_sha256']})
    return reconstructed, [probe['relative_path'] for probe in probes], availability


def validate_roles(updates, probes, excluded_r1, excluded_p0):
    groups = {'updates': updates, 'probes': probes, 'excluded_r1': excluded_r1, 'excluded_p0': excluded_p0}
    for name, rows in groups.items():
        for row in rows:
            require(row['relative_path'] == f"{row['category']}/train/good/{row['cache_index']:03}.png",
                    'forbidden source path')
        require(len({r['relative_path'] for r in rows}) == len(rows), f'duplicate identity in {name}')
        require(len({r['raw_png_sha256'] for r in rows}) == len(rows), f'duplicate content in {name}')
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            require(not {r['relative_path'] for r in groups[left]} & {r['relative_path'] for r in groups[right]},
                    f'split identity overlap {left}/{right}')
            require(not {r['raw_png_sha256'] for r in groups[left]} & {r['raw_png_sha256'] for r in groups[right]},
                    f'split content overlap {left}/{right}')
    require(len(updates) == 350 and len(probes) == 59, 'unexpected role counts')
    return {'counts': {name: len(rows) for name, rows in groups.items()},
            'filename_content_disjoint': True,
            'probe_counts_by_category': {c: sum(1 for r in probes if r['category'] == c) for c in CATEGORIES}}


class Reader:
    """Policy-enforcing cache reader: numeric reads only for authorized roles and boundaries."""

    def __init__(self, protocol, preflight=False):
        self.p = protocol
        self.completed = 0
        self.admitted = set()
        self.current = None
        self.obj = None
        self.access = []
        self.preflight = preflight
        self.full_grid_reads = 0
        self.train_grid_reads = 0

    def _open(self, category):
        require(category in CATEGORIES, 'forbidden category')
        obj = torch.load(FEATURES / f'clean/{category}_train.pt', mmap=True, weights_only=True)
        entries = [r for r in json.loads((FEATURES / 'source_manifest.json').read_text())['entries']
                   if r['category'] == category]
        require(obj['view'] == 'clean' and obj['class_name'] == category, 'wrong cache view')
        require(obj['relative_paths'] == [r['relative_path'] for r in entries], 'cache paths mismatch')
        require(obj['per_image_source_sha256'] == [r['raw_png_sha256'] for r in entries], 'cache source hashes mismatch')
        require(obj['patches'].shape == (len(entries), 784, 768) and obj['patches'].dtype == torch.float32,
                'cache geometry mismatch')
        require(obj['source_manifest_digest'] == SOURCE_DIGEST and obj['metadata']['checkpoint_sha256'] == CP_SHA,
                'cache provenance mismatch')
        return obj

    def _select(self, obj, row, grid, role):
        require(obj['relative_paths'][row['cache_index']] == row['relative_path'], 'cache row mismatch')
        full = obj['patches'][row['cache_index']]
        h = hashlib.sha256(str((tuple(full.shape), full.dtype)).encode() + full.contiguous().numpy().tobytes()).hexdigest()
        require(h == obj['per_image_feature_sha256'][row['cache_index']], 'feature row digest mismatch')
        result = full[list(grid)].clone()
        require(bool(torch.isfinite(result).all()), 'nonfinite authorized features')
        self.access.append({'role': role, 'path': row['relative_path'], 'event': self.completed,
                            'positions': len(grid), 'selected_sha256': fingerprint({'x': result})})
        if len(grid) == 784:
            self.full_grid_reads += 1
        else:
            self.train_grid_reads += 1
        return result

    def update(self, row):
        require(self.completed < len(self.p['updates']) and row == self.p['updates'][self.completed],
                'future or unauthorized update')
        require(not self.preflight or self.completed < self.p['preflight_events'], 'preflight beyond allowed events')
        category = row['category']
        if self.current != category:
            self.obj = None
            self.current = category
            self.admitted.add(category)
            self.obj = self._open(category)
        self.completed += 1
        # All arms observe the same arriving image; each arm slices its own proposal grid.
        return self._select(self.obj, row, FULL_GRID, 'feasibility_training' if self.preflight else 'update')

    def probe(self, row, boundary):
        require(not self.preflight, 'preflight probes forbidden')
        require(boundary == self.completed and boundary in BOUNDARIES, 'probe at forbidden boundary')
        require(row in self.p['probes'] and row['category'] in self.admitted, 'future or unauthorized probe')
        obj = self._open(row['category'])
        return self._select(obj, row, PROBE_GRID, 'probe')


def feasibility(protocol):
    """Non-evaluative timing check on already-arrived update identities only."""
    reference = torch.load(P0 / 'states_t100.pt', map_location='cpu', weights_only=False)['models']
    timings = {}
    access = []
    for arm in ARMS:
        reader = Reader(protocol, preflight=True)
        model = Support(ARM_METHOD[arm])
        saved = reference['STREAM_GREEDY'] if arm == 'GRID16_GREEDY_REFERENCE' else None
        if saved is not None:
            for key, value in saved.items():
                if key in ('method', 'capacity'):
                    continue
                setattr(model, key, value.clone() if isinstance(value, torch.Tensor) else value)
            model.capacity = int(saved['capacity'])
        else:
            generator = torch.Generator().manual_seed(11)
            model.bank[:] = torch.randn(model.bank.shape, generator=generator) * 2
            model.ids[:] = torch.arange(model.capacity)
            model.n = model.capacity
            model.seen = model.capacity
            if arm == 'FULL784_GREEDY':
                for start in range(0, model.n, 128):
                    stop = min(model.n, start + 128)
                    d = distances(model.bank[start:stop], model.bank[:model.n])
                    d[torch.arange(stop - start), torch.arange(start, stop)] = torch.inf
                    values, indices = d.min(1)
                    model.nn_distance[start:stop] = values
                    model.nn_index[start:stop] = indices
        seconds = []
        for event in range(1, protocol['preflight_events'] + 1):
            row = protocol['updates'][event - 1]
            full = reader.update(row)
            grid = ARM_GRID[arm]
            x = full[list(grid)].clone()
            stride = len(grid)
            ids = torch.arange((event - 1) * stride, event * stride, dtype=torch.int64)
            started = time.perf_counter()
            model.update(x, ids)
            seconds.append(time.perf_counter() - started)
        timings[arm] = {'events': protocol['preflight_events'], 'mean_update_seconds': float(np.mean(seconds)),
                        'max_update_seconds': float(max(seconds)), 'ledger': model.ledger(),
                        'occupancy': int(model.n)}
        access.extend(reader.access)
        del model, reader
        gc.collect()
    projected = 350 * sum(value['mean_update_seconds'] for value in timings.values())
    allowance = 3 * projected + 120
    result = {'no_probe_reads': True, 'preflight_events': protocol['preflight_events'],
              'first_admitted_training_path': protocol['updates'][0]['relative_path'],
              'measurements': timings, 'projected_update_seconds': projected,
              'threefold_plus_120s_seconds': allowance,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'access_log': access}
    save('feasibility.json', result)
    require(allowance <= 3600, 'BLOCKED_COMPUTE: conservative throughput projection exceeds 60 minutes')
    require(result['peak_rss_bytes'] <= 6 * 1024**3, 'BLOCKED_COMPUTE: feasibility peak RSS exceeds 6 GiB')
    return result


def prepare(preflight_events=12):
    require(not (OUT / 'protocol_resolved.json').exists() and not (OUT / 'run_started.json').exists(),
            'refusing overwrite or second attempt')
    require('passed' in (OUT / 'unit_tests.txt').read_text(), 'focused tests missing')
    source = json.loads((FEATURES / 'source_manifest.json').read_text())
    canonical = hashlib.sha256(json.dumps({k: v for k, v in source.items() if k != 'manifest_digest'},
                                          sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(canonical == SOURCE_DIGEST == source['manifest_digest'], 'source canonical digest mismatch')
    for name, expected in CACHE_SHA.items():
        require(sha(FEATURES / name) == expected, 'cache byte hash mismatch: ' + name)
        print('verified', name, flush=True)
    require(sha(ROOT / 'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth') == CP_SHA,
            'checkpoint digest mismatch')

    updates, r1_probe_paths, availability = reconstruct_updates(source)
    frozen = json.loads((P0 / 'protocol_resolved.json').read_text())
    require([r['relative_path'] for r in frozen['updates']] == [r['relative_path'] for r in updates],
            'reconstructed arrival order differs from the frozen P0 order')
    require(sorted(r1_probe_paths) == sorted(r['relative_path'] for r in frozen['excluded_r1_probes']),
            'reconstructed R1 probe set differs from the frozen P0 record')

    probes = [row for category, (start, stop) in PROBE_SPEC.items()
              for row in rows_for(source, category, start, stop, 'probe')]
    excluded_r1 = [row for category, (start, stop) in EXCLUDED_R1.items()
                   for row in rows_for(source, category, start, stop, 'excluded_r1')]
    excluded_p0 = [row for category, (start, stop) in EXCLUDED_P0.items()
                   for row in rows_for(source, category, start, stop, 'excluded_p0')]
    require(sorted(r['relative_path'] for r in excluded_p0) == sorted(r['relative_path'] for r in frozen['probes']),
            'P0 probe identities differ from the frozen P0 record')
    disjointness = validate_roles(updates, probes, excluded_r1, excluded_p0)

    p = {'schema': 'task_p_p0c_v1', 'root': str(ROOT),
         'locked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'checkpoint_sha256': CP_SHA, 'cache_sha256': CACHE_SHA, 'source_canonical_digest': SOURCE_DIGEST,
         'input_document_sha256': {name: sha(FEATURES / name) for name in
                                   ('source_manifest.json', 'extractor_contract.json', 'checkpoint_provenance.json',
                                    'cache_integrity.json', 'independent_verification.json')},
         'p0_protocol_sha256': sha(P0 / 'protocol_resolved.json'),
         'source_code_sha256': {name: sha(ROOT / name) for name in CODE},
         'environment': environment(),
         'updates': updates, 'probes': probes, 'excluded_r1_probes': excluded_r1, 'excluded_p0_probes': excluded_p0,
         'disjointness': disjointness, 'availability': availability,
         'train_grid': list(TRAIN_GRID), 'full_grid': 'all 784 row-major ViT patch positions',
         'probe_grid': list(PROBE_GRID), 'arrival_boundaries': list(BOUNDARIES),
         'order_seed': 0, 'reservoir_seed': 7202,
         'budget_bytes': BUDGET, 'metadata_reserve_bytes': 16384,
         'capacities': {'GRID16_GREEDY_REFERENCE': GREEDY_CAPACITY, 'FULL784_GREEDY': GREEDY_CAPACITY,
                        'FULL784_RESERVOIR': RESERVOIR_CAPACITY},
         'arm_grids': {'GRID16_GREEDY_REFERENCE': 'train16', 'FULL784_GREEDY': 'full784',
                       'FULL784_RESERVOIR': 'full784'},
         'arms': {
             'GRID16_GREEDY_REFERENCE': 'unchanged P0 STREAM_GREEDY fed the 16 training-grid positions',
             'FULL784_GREEDY': 'identical algorithm fed all 784 patch positions of the arriving image',
             'FULL784_RESERVOIR': 'Algorithm R over all arriving 784-position descriptors'},
         'distance': ('FP64 accumulated raw Euclidean norm-dot with direct-difference repair at '
                      '<=64*float64_eps*(sum squared norms); stable first-index min ties'),
         'risk': 'mean largest 7 of 64 probe patch distances; per image then mean images; equal-category macro',
         'scale_formula': ('median of 150 L2 differences between consecutive row-major candidate positions '
                           'from first ten admitted images per category; evaluator-only; 16-position grid '
                           'retained for exact comparability with P0'),
         'primary': 'FULL784_GREEDY versus GRID16_GREEDY_REFERENCE at event 350 on new P0c probes',
         'bootstrap': {'replicates': 10000, 'seed': 9201,
                       'unit': 'whole source image; paired stratified within category',
                       'interval': 'percentile 2.5/97.5; fixed training/order seed'},
         'gates': {'final_gain': 0.08, 'ci_lower': 0.0, 'per_category_final_deterioration': 0.10,
                   'near_zero_normalized_risk': 1e-10, 'all_integrity_required': True,
                   'offline_calibration_within': 0.15},
         'calibration': ('offline full-grid farthest-point bank at the same 8-MiB budget, plus unbounded '
                         'proposal and full-grid oracles; evaluator-only, never learner inputs'),
         'stop_rules': {'computation_seconds': 3600, 'peak_rss_bytes': 6 * 1024**3, 'output_bytes': 512 * 1024**2,
                        'no_recovery_training_or_tuning': True},
         'decision_rules': {
             'P0C_GRID_CONFIRMED': 'gain >= 8%, bootstrap CI lower > 0, every category within 10%, integrity valid',
             'P0C_ALLOCATION_TRADEOFF': 'gain >= 8% and CI lower > 0, but at least one category worse than -10%',
             'P0C_STREAMING_SELECTOR_BOTTLENECK': ('primary gates fail while the offline evaluator-only full-grid '
                                                   'FPS bank improves on the reference arm by >= 8%'),
             'P0C_GRID_NOT_CONFIRMED': ('primary gates fail and the offline evaluator-only full-grid FPS bank '
                                        'improves on the reference arm by < 8%'),
             'BLOCKED': 'execution/integrity/reproduction gate failure'},
         'decision_note': ('rules were fixed before any new P0c probe tensor was read; the offline calibration is '
                           'evaluator-only and can never influence a learner'),
         'limitations': ['new probes are disjoint from R1/P0/P0b probe sets but not certified fresh to the '
                         'historical project', 'unknown object/acquisition grouping', 'one seed/order',
                         'normal-support metric only; no anomaly claim'],
         'preserved_sha256': json.loads((P0 / 'preflight_preservation.json').read_text())['preserved_sha256'],
         'preflight_events': preflight_events}
    feasibility(p)
    save('input_verification.json', {'passed': True, 'cache_sha256': CACHE_SHA, 'checkpoint_sha256': CP_SHA,
                                     'canonical_source_digest': canonical, 'disjointness': disjointness,
                                     'p0_arrival_reconstruction_matches': True,
                                     'numeric_future_cache_access_before_arrival': False})
    save('protocol_resolved.json', p)
    (OUT / 'protocol.sha256').write_text(sha(OUT / 'protocol_resolved.json') + '\n')
    print('PROTOCOL_LOCKED', sha(OUT / 'protocol_resolved.json'), flush=True)


def run():
    p = json.loads((OUT / 'protocol_resolved.json').read_text())
    protocol_sha = sha(OUT / 'protocol_resolved.json')
    require(protocol_sha == (OUT / 'protocol.sha256').read_text().strip(), 'protocol changed')
    require(all(sha(ROOT / name) == value for name, value in p['source_code_sha256'].items()), 'frozen source changed')
    require(p['environment'] == environment(), 'environment changed')
    with (OUT / 'run_started.json').open('x') as stream:
        json.dump({'protocol_sha256': protocol_sha, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                  stream)

    started = time.perf_counter()
    reader = Reader(p)
    models = {arm: Support(ARM_METHOD[arm]) for arm in ARMS}
    records, boundaries, trace, chain = [], [], [], []
    scales, scale_values = {}, {category: [] for category in CATEGORIES}
    timing = {arm: {'update_seconds': 0.0, 'read_seconds': 0.0, 'read_images': 0, 'descriptors': 0,
                    'max_persistent_bytes': 0} for arm in ARMS}
    previous = {arm: fingerprint(model.state()) for arm, model in models.items()}
    p0_reproduction = []

    for row in p['updates']:
        require(time.perf_counter() - started < 3600, 'BLOCKED_COMPUTE: run time exceeded')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 6 * 1024**3, 'BLOCKED_COMPUTE: RSS exceeded')
        event = row['event']
        category = row['category']
        full = reader.update(row)          # policy-enforced single read of the arriving image
        train = full[list(TRAIN_GRID)].clone()
        proposals = {'GRID16_GREEDY_REFERENCE': ('train16', train),
                     'FULL784_GREEDY': ('full784', full.clone()),
                     'FULL784_RESERVOIR': ('full784', full.clone())}
        if category not in scales:
            values = (train[1:].double() - train[:-1].double()).norm(dim=-1).tolist()
            scale_values[category].extend(values)
            if len(scale_values[category]) == 150:
                value = float(np.median(scale_values[category]))
                require(np.isfinite(value) and value > 0, 'invalid training-only scale')
                scales[category] = {'value': value, 'available_at_event': event}
                scale_values[category].clear()
        before = {arm: fingerprint(model.state()) for arm, model in models.items()}
        require(before == previous, 'state reset between events')
        for arm in ARMS:
            grid_name, x = proposals[arm]
            stride = x.shape[0]
            ids = torch.arange((event - 1) * stride, event * stride, dtype=torch.int64)
            input_hash = fingerprint({'x': x, 'ids': ids})
            model = models[arm]
            if event == 1:
                h = fingerprint(model.state())
                model.update(x, ids, write=False)
                require(h == fingerprint(model.state()), 'no-write mutated state')
            started_update = time.perf_counter()
            stats = model.update(x, ids)
            timing[arm]['update_seconds'] += time.perf_counter() - started_update
            timing[arm]['descriptors'] += int(x.shape[0])
            require(input_hash == fingerprint({'x': x, 'ids': ids}), 'learner mutated common proposals')
            ledger = model.ledger()
            timing[arm]['max_persistent_bytes'] = max(timing[arm]['max_persistent_bytes'],
                                                      ledger['charged_persistent_bytes'])
            trace.append({'event': event, 'arm': arm, 'grid': grid_name, 'input_sha256': input_hash,
                          'after_sha256': fingerprint(model.state()),
                          'score_bank_sha256': fingerprint({'bank': model.bank, 'ids': model.ids}),
                          'ledger': ledger, **{k: v for k, v in stats.items() if k != 'decisions'}})
            del x
        # cross-arm source identity: arm A proposals must equal the full-grid rows at the 16 training positions
        require(torch.equal(proposals['GRID16_GREEDY_REFERENCE'][1],
                            proposals['FULL784_GREEDY'][1][list(TRAIN_GRID)]),
                'train-grid proposals are not a subset of the same arriving image')
        del proposals, train, full
        previous = {arm: fingerprint(model.state()) for arm, model in models.items()}
        chain.append({'event': event, 'before': before, 'after': previous})

        if event in BOUNDARIES:
            boundary = {'event': event, 'scales': {k: v for k, v in scales.items()}, 'methods': {}, 'categories': {}}
            for arm in ARMS:
                model = models[arm]
                by_category = {c: 0 for c in CATEGORIES}
                counts = {}
                for serial in model.ids[:model.n].tolist():
                    image = serial // (16 if arm == 'GRID16_GREEDY_REFERENCE' else 784)
                    counts[p['updates'][image]['category']] = counts.get(p['updates'][image]['category'], 0) + 1
                boundary['methods'][arm] = {'ledger': model.ledger(),
                                            'serialized_state_bytes': serialized_bytes(model.state()),
                                            'support_counts_audit_only': counts}
            for category in CATEGORIES:
                if category not in reader.admitted:
                    continue
                group = []
                for probe in [q for q in p['probes'] if q['category'] == category]:
                    probe_obj = reader._open(category)
                    q = reader._select(probe_obj, probe, PROBE_GRID, 'probe')
                    entry = {'event': event, 'category': category, 'path': probe['relative_path'],
                             'raw_png_sha256': probe['raw_png_sha256'], 'scale': scales[category]['value'],
                             'arms': {}}
                    for arm in ARMS:
                        model = models[arm]
                        h = fingerprint(model.state())
                        started_read = time.perf_counter()
                        d, idx = model.score(q)
                        timing[arm]['read_seconds'] += time.perf_counter() - started_read
                        timing[arm]['read_images'] += 1
                        repeat, _ = model.score(q)
                        require(torch.equal(d, repeat), 'read replay differs')
                        require(h == fingerprint(model.state()), 'read mutated learner state')
                        entry['arms'][arm] = {'tail_raw': float(d.topk(7).values.mean()),
                                              'tail_normalized': float(d.topk(7).values.mean()) / scales[category]['value'],
                                              'mean': float(d.mean()), 'max': float(d.max()),
                                              'distances': d.tolist(), 'nearest_serials': idx.tolist()}
                    records.append(entry)
                    group.append(entry)
                boundary['categories'][category] = {
                    arm: {'normalized_risk': float(np.mean([e['arms'][arm]['tail_normalized'] for e in group]))}
                    for arm in ARMS}
            boundaries.append(boundary)
            torch.save({'protocol_sha256': protocol_sha,
                        'models': {arm: model.state() for arm, model in models.items()}},
                       OUT / f'states_t{event}.pt')
            save('per_image_results.json', records)
            save('boundaries.json', boundaries)
            save('update_trace.json', trace)
            save('state_chain.json', chain)
            save('access_log.json', reader.access)
            print('boundary_complete', event, 'occupancy', {arm: model.n for arm, model in models.items()}, flush=True)

    # reproduction gate: arm A must match the saved P0 STREAM_GREEDY states bitwise
    for event in BOUNDARIES:
        reference = torch.load(P0 / f'states_t{event}.pt', map_location='cpu', weights_only=False)['models']['STREAM_GREEDY']
        state = torch.load(OUT / f'states_t{event}.pt', map_location='cpu', weights_only=False)['models']['GRID16_GREEDY_REFERENCE']
        same = (int(state['n']) == int(reference['n'])
                and torch.equal(state['bank'][:state['n']], reference['bank'][:reference['n']])
                and torch.equal(state['ids'][:state['n']], reference['ids'][:reference['n']])
                and bool(torch.equal(state['nn_distance'][:state['n']], reference['nn_distance'][:reference['n']]))
                and bool(torch.equal(state['nn_index'][:state['n']], reference['nn_index'][:reference['n']])))
        p0_reproduction.append({'event': event, 'bitwise_state_equal': bool(same)})
        require(same, f'GRID16 reference failed to reproduce P0 STREAM_GREEDY state at t={event}')
    save('p0_reproduction.json', {'passed': all(r['bitwise_state_equal'] for r in p0_reproduction),
                                  'checks': p0_reproduction})
    require(all(sha(ROOT / name) == value for name, value in p['preserved_sha256'].items()),
            'protected source changed during the run')

    resources = {'methods': {arm: {**value, 'update_ms_per_arrival': 1000 * value['update_seconds'] / 350,
                                   'descriptors_per_second': value['descriptors'] / max(1e-9, value['update_seconds']),
                                   'inference_ms_per_image': 1000 * value['read_seconds'] / max(1, value['read_images'])}
                             for arm, value in timing.items()},
                 'shared': {'runtime_seconds': time.perf_counter() - started,
                            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            'transient_full_grid_input_bytes': 784 * 768 * 4,
                            'checkpoint_file_bytes': (ROOT / 'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth').stat().st_size,
                            'clean_cache_file_bytes': sum((FEATURES / name).stat().st_size for name in CACHE_SHA)}}
    save('resources.json', resources)
    print('MEASUREMENTS_COMPLETE', resources['shared']['runtime_seconds'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--preflight-events', type=int, default=12)
    args = parser.parse_args()
    configure()
    try:
        prepare(args.preflight_events) if args.mode == 'prepare' else run()
    except Exception as error:
        save('execution_error.json', {'mode': args.mode, 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
