"""P2f paths, pinned references, and real/synthetic identity-gated readers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p1_data import native_mask, sha, write_json
from exps.task_p.p2.data import (NormalReader, Replay, evaluation_features, require,
                                 tensor_bytes, tree_digest)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/hope_cad/task_p/p2f'
P2E = ROOT / 'results/hope_cad/task_p/p2e'
P2D = ROOT / 'results/hope_cad/task_p/p2d'
P2 = ROOT / 'results/hope_cad/task_p/p2'
P1 = ROOT / 'results/hope_cad/task_p/p1'
P2E_SHA = '18ea5eb104b6cdccce1e89073f3b509d84a583bf37f6ac83212bd5c5fbb6fb75'
P2D_SHA = '182601095371de52627e8a8e2bae26299a3573f5bcf7a63a88f07c762e44d94b'
P2_SHA = 'be3a0fa0a351f4acf0e087667090d25b3257e5b5cfccc5dac8b0daa3b1ea0fa2'
P1_SHA = '11c1cc03b20cb1f79a6e8b7e472bb6f44758b43046b61043368ea11b4ce1e08f'
CATEGORIES = ('bottle', 'carpet', 'hazelnut')
REFERENCES = ('E', 'B', 'C2', 'D2')


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
    names = [path for path in (ROOT / 'exps/task_p/p2f').glob('*.py')]
    names += [ROOT / name for name in (
        'AGENTS.md', 'agents/tasks/current_task_p2f.md',
        'exps/task_p/p1_data.py', 'exps/task_p/p1_readout.py',
        'exps/task_p/p1_bootstrap_bounds.py',
        'exps/task_p/p2/data.py', 'exps/task_p/p2/model.py',
        'exps/task_p/p2/metrics.py',
        'exps/task_p/p2d/model.py', 'exps/task_p/p2e/model.py')]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(names)}


def _verify_protocol(directory, expected_sha):
    path = Path(directory) / 'protocol_resolved.json'
    require(sha(path) == expected_sha, f'protocol changed: {directory}')
    protocol = read(path)
    for name, expected in protocol.get('code_sha256', {}).items():
        require(sha(ROOT / name) == expected, f'locked source changed: {name}')
    return protocol


def inherited():
    """Verify the complete P0-P2e inheritance and build the preserved-file ledger."""
    p2e = _verify_protocol(P2E, P2E_SHA)
    p2d = _verify_protocol(P2D, P2D_SHA)
    p2 = _verify_protocol(P2, P2_SHA)
    p1 = _verify_protocol(P1, P1_SHA)
    # P2e's own 610 pinned historical files plus its Stage A evidence.
    for name, expected in p2e['reference_sha256'].items():
        require(sha(ROOT / name) == expected, f'P2e reference changed: {name}')
    for name, expected in p2e['stage_a_sha256'].items():
        require(sha(P2E / name) == expected, f'P2e Stage A changed: {name}')
    ledger = read(P2E / 'artifact_integrity.json')['files']
    for name, record in ledger.items():
        require(sha(P2E / name) == record['sha256'], f'P2e artifact changed: {name}')
    preserved = dict(p2e['reference_sha256'])
    preserved.update({str((P2E / name).relative_to(ROOT)): record['sha256']
                      for name, record in ledger.items()})
    preserved.update(p2e['code_sha256'])
    for path in (ROOT / 'agents/reports').glob('*.md'):
        preserved[str(path.relative_to(ROOT))] = sha(path)
    for path in (ROOT / 'agents/tasks').glob('*.md'):
        if path.name != 'current_task_p2f.md':
            preserved[str(path.relative_to(ROOT))] = sha(path)
    preserved[str(ROOT / 'AGENTS.md')] = sha(ROOT / 'AGENTS.md')
    return {'passed': True, 'preserved': preserved,
            'p2e_protocol_sha256': P2E_SHA, 'p2d_protocol_sha256': P2D_SHA,
            'p2_protocol_sha256': P2_SHA, 'p1_protocol_sha256': P1_SHA,
            'p2e_reference_files': len(p2e['reference_sha256']),
            'p2e_artifacts': len(ledger), 'sealed_payload_reads': 0}


def key_references():
    names = {
        'p2e_protocol': P2E / 'protocol_resolved.json',
        'p2e_point_metrics': P2E / 'point_metrics.json',
        'p2e_bootstrap_samples': P2E / 'bootstrap_samples.json',
        'p2e_summary': P2E / 'summary.json',
        'p2e_score_separation': P2E / 'score_separation.json',
        'p2e_probe_state': P2E / 'probe_state_t100.pt',
        'p2e_probe_trace': P2E / 'probe_trace.json',
        'p2e_stage_a': P2E / 'stage_a.json',
        'p2e_synthetic_validation': P2E / 'synthetic_validation.json',
        'p2e_artifact_integrity': P2E / 'artifact_integrity.json',
        'p2d_protocol': P2D / 'protocol_resolved.json',
        'p2_protocol': P2 / 'protocol_resolved.json',
        'p1_protocol': P1 / 'protocol_resolved.json',
    }
    return {str(path.relative_to(ROOT)): sha(path) for path in names.values()}


def protocol_at(directory):
    directory = Path(directory)
    path = directory / 'protocol_resolved.json'
    require(sha(path) == (directory / 'protocol.sha256').read_text().strip(), 'protocol digest')
    protocol = read(path)
    require(protocol['schema'] == 'p2f_conditional_consolidation_v1', 'protocol schema')
    require(protocol['arms'] == ['F'], 'single F arm')
    for name, expected in protocol['code_sha256'].items():
        require(sha(ROOT / name) == expected, f'frozen source changed: {name}')
    if not protocol['synthetic']:
        require(protocol['p2e_protocol_sha256'] == P2E_SHA, 'P2e reference protocol drift')
        require(protocol['p2d_protocol_sha256'] == P2D_SHA, 'P2d reference protocol drift')
        require(protocol['p2_protocol_sha256'] == P2_SHA, 'P2 reference protocol drift')
        for name, expected in protocol['reference_sha256'].items():
            require(sha(ROOT / name) == expected, f'reference drift: {name}')
        require(sha(directory / 'inheritance.json') == protocol['inheritance_sha256'],
                'inheritance drift')
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
        for _ in range(protocol['bootstrap_draws']):
            weights = np.zeros(len(rows), dtype=np.int64)
            for label in (0, 1):
                choices = np.array([index for index, row in enumerate(rows) if row['label'] == label])
                require(len(choices) > 0, 'degenerate image stratum')
                weights += np.bincount(generator.choice(choices, size=len(choices), replace=True),
                                       minlength=len(rows))
            counts.append(weights)
        result[category] = np.stack(counts)
    return result


def reference_artifacts(protocol):
    points = read(P2E / 'point_metrics.json')
    samples = read(P2E / 'bootstrap_samples.json')
    keep = lambda key: key.split('/')[-1] in REFERENCES
    return ({key: value for key, value in points.items() if keep(key)},
            {key: value for key, value in samples.items() if keep(key)})


def reference_points():
    return read(P2E / 'point_metrics.json')


def reference_samples():
    return read(P2E / 'bootstrap_samples.json')
