"""Exclusive artifacts, pinned inputs and real/synthetic identity-gated readers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p1_data import native_mask, sha, write_json
from exps.task_p.p2.data import (NormalReader, Replay, evaluation_features, require,
                                 tensor_bytes, tree_digest)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/hope_cad/task_p/p2e'
P2D = ROOT / 'results/hope_cad/task_p/p2d'
P2 = ROOT / 'results/hope_cad/task_p/p2'
P2D_SHA = '182601095371de52627e8a8e2bae26299a3573f5bcf7a63a88f07c762e44d94b'
CATEGORIES = ('bottle', 'carpet', 'hazelnut')
REFERENCES = ('B', 'C2', 'D2')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pinned_sources():
    names = [path for path in (ROOT / 'exps/task_p/p2e').glob('*.py')]
    names += [ROOT / name for name in (
        'AGENTS.md', 'agents/tasks/current_task_p2e.md', 'exps/task_p/p1_data.py',
        'exps/task_p/p1_readout.py', 'exps/task_p/p1_bootstrap_bounds.py',
        'exps/task_p/p2/data.py', 'exps/task_p/p2/model.py', 'exps/task_p/p2/metrics.py',
        'exps/task_p/p2d/model.py')]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(names)}


def inherited():
    require(sha(P2D / 'protocol_resolved.json') == P2D_SHA, 'P2d lock changed')
    protocol = read(P2D / 'protocol_resolved.json')
    pins = dict(protocol['code_sha256'])
    pins.update(read(P2 / 'protocol_resolved.json')['code_sha256'])
    pins.update(read(P2D / 'inheritance.json')['preserved'])
    pins.update({str((P2 / name).relative_to(ROOT)): record['sha256']
                 for name, record in read(P2 / 'artifact_integrity.json')['files'].items()})
    for name, expected in pins.items():
        require(sha(ROOT / name) == expected, f'inherited hash differs: {name}')
    paths = {ROOT / name for name in pins}
    paths.update(path for path in P2D.rglob('*') if path.is_file())
    paths.update(path for path in (ROOT / 'exps/task_p/p2d').rglob('*') if path.is_file())
    paths.update(path for path in (ROOT / 'exps/task_p/p2').rglob('*') if path.is_file())
    paths.update(path for path in (ROOT / 'agents/reports').glob('*.md'))
    paths.update(path for path in (ROOT / 'agents/tasks').glob('*.md')
                 if path.name != 'current_task_p2e.md')
    preserved = {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}
    return {'passed': True, 'preserved': preserved, 'p2d_protocol_sha256': P2D_SHA,
            'reference_groups': 162, 'sealed_payload_reads': 0}


def protocol_at(directory):
    directory = Path(directory)
    path = directory / 'protocol_resolved.json'
    require(sha(path) == (directory / 'protocol.sha256').read_text().strip(), 'protocol digest')
    protocol = read(path)
    require(protocol['schema'] == 'p2e_single_mix_intervention_v1', 'protocol schema')
    require(isinstance(protocol['orders'], dict), 'orders must be a mapping')
    require(protocol['arms'] == ['E'] and protocol['mix'] == 0.0, 'single intervention')
    for name, expected in protocol['code_sha256'].items():
        require(sha(ROOT / name) == expected, f'frozen source changed: {name}')
    if not protocol['synthetic']:
        require(sha(P2D / 'protocol_resolved.json') == P2D_SHA, 'reference protocol drift')
        for name, expected in protocol['reference_sha256'].items():
            require(sha(ROOT / name) == expected, f'reference drift: {name}')
    return protocol


def model_arguments(protocol):
    return dict(protocol['model_dimensions'])


def fixtures(directory):
    return torch.load(Path(directory) / 'synthetic_inputs.pt', map_location='cpu', weights_only=True)


class Reader:
    def __init__(self, directory, protocol, rows):
        self.protocol, self.rows, self.index, self.access = protocol, rows, 0, []
        self.inner = None if protocol['synthetic'] else NormalReader(rows)
        self.fixture = fixtures(directory) if protocol['synthetic'] else None

    def get(self, row):
        require(self.index < len(self.rows) and row == self.rows[self.index], 'future normal source')
        require(row['role'] == 'normal_train' and '/train/good/' in row['relative_path'], 'not normal')
        if self.inner is not None:
            feature = self.inner.get(row)
            self.access = self.inner.access
        else:
            feature = self.fixture['normal'][row['relative_path']].clone()
            self.access.append({'event': row['event'], 'source': row['relative_path'],
                                'feature_sha256': tree_digest(feature)})
        self.index += 1
        return feature


def eval_features(directory, row, protocol):
    require(row in protocol['evaluations'], 'unadmitted evaluation identity')
    require(row['relative_path'] not in protocol['sealed_identity_paths'], 'sealed identity')
    if protocol['synthetic']:
        return fixtures(directory)['evaluation'][row['relative_path']].clone()
    return evaluation_features(row, protocol)


def masks_at(directory, protocol):
    if protocol['synthetic']:
        values = fixtures(directory)['masks']
        return {row['relative_path']: values[row['relative_path']].numpy().astype(bool)
                for row in protocol['evaluations']}
    return {row['relative_path']: native_mask(protocol['dataset_root'], row)
            for row in protocol['evaluations']}


def image_draws(protocol):
    result = {}
    for category_index, category in enumerate(CATEGORIES):
        rows = [row for row in protocol['evaluations'] if row['category'] == category]
        generator = np.random.default_rng(102701 + category_index)
        counts = []
        for repetition in range(protocol['bootstrap_draws']):
            weights = np.zeros(len(rows), dtype=np.int64)
            for label in (0, 1):
                choices = np.array([index for index, row in enumerate(rows) if row['label'] == label])
                require(len(choices) > 0, 'degenerate image stratum')
                weights += np.bincount(generator.choice(choices, size=len(choices), replace=True), minlength=len(rows))
            counts.append(weights)
        result[category] = np.stack(counts)
    return result


def reference_artifacts(protocol):
    points = read(P2D / 'point_metrics.json')
    samples = read(P2D / 'bootstrap_samples.json')
    keep = lambda key: key.split('/')[-1] in REFERENCES
    return ({key: value for key, value in points.items() if keep(key)},
            {key: value for key, value in samples.items() if keep(key)})
