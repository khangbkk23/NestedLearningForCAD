"""W7 transport/experiment contract. Does not change the frozen v4 learner."""
from copy import deepcopy
from pathlib import Path
import json
import random

from dataset.benchmark_manifest_v1 import canonical_digest, manifest_digest
from dataset.benchmark_protocol_v2 import split_prefix, validate_prefix_manifest, file_hash
from training.hope_runtime_v4 import ROOT, source_identity, atomic_json

METHODS = ('M', 'C0', 'C1', 'C2', 'C3', 'C4', 'A1')
SEEDS = (0, 1, 2)
W6_SHA256 = 'ae9d257ecf1ced738df651fa07887b838a96da5fe4cdda84e46c51533c9bcd71'
W6_MANIFEST_SHA256 = '5892af759a096a6f2ba658eff5a3095aadc94c05bdb7c5849bfb6b122f37d4cd'
# Public dataset integrity counts, NOT performance/selection feedback. MVTec AD Table 1:
# https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf
# (train normal, test normal, test anomaly)
MVTEC_COUNTS = dict(bottle=(209,20,63),cable=(224,58,92),capsule=(219,23,109),
    carpet=(280,28,89),grid=(264,21,57),hazelnut=(391,40,70),leather=(245,32,92),
    metal_nut=(220,22,93),pill=(267,26,141),screw=(320,41,119),tile=(230,33,84),
    toothbrush=(60,12,30),transistor=(213,60,40),wood=(247,19,60),zipper=(240,32,119))


def source_hashes():
    result = source_identity()
    for pattern in ('training/hope_w7_*.py', 'scripts/hope_cad_v2/*w7*.py',
                    'tests/hope_cad_v2/test_w7*.py', 'docs/main_spec/w7*'):
        for path in ROOT.glob(pattern):
            if path.is_file():
                result[path.relative_to(ROOT).as_posix()] = file_hash(path)
    # Baseline implementations are also immutable dependencies of the suite.
    for path in (ROOT/'models/cadic_paperfaithful').glob('*.py'):
        result[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return dict(sorted(result.items()))


def lock_json(path, identity):
    path = Path(path)
    record = dict(identity=identity, digest=canonical_digest(identity))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError(f'locked identity changed: {path}')
    else:
        atomic_json(record, path)
    return record


def jobs():
    """35 streams; M/seed0 is produced by 7.1+7.2, never retrained in 7.3."""
    result = []
    for seed in SEEDS:
        for method in METHODS:
            result.append(dict(id=f'{method}_seed{seed}', method=method, seed=seed,
                               order_seed=None, budget=None))
        for budget in (2500, 10000):
            result.append(dict(id=f'CADIC{budget}_seed{seed}', method='CADIC', seed=seed,
                               order_seed=None, budget=budget))
    for order in (100, 200):
        for method, budget in (('M', None), ('C4', None), ('CADIC', 2500), ('CADIC', 10000)):
            label = f'CADIC{budget}' if budget else method
            result.append(dict(id=f'{label}_seed0_order{order}', method=method,
                               seed=0, order_seed=order, budget=budget))
    return result


def job_by_id(name):
    matches = [job for job in jobs() if job['id'] == name]
    if len(matches) != 1:
        raise ValueError(f'unregistered W7 job: {name}')
    return matches[0]


def job_protocol(context, job):
    protocol = deepcopy(context['protocol'])
    manifest = deepcopy(context['manifest'])
    if job['order_seed'] is not None:
        tail = list(protocol['task_order'][1:])
        random.Random(job['order_seed']).shuffle(tail)
        protocol['task_order'] = [protocol['task_order'][0], *tail]
        protocol['order_seed'] = job['order_seed']
        protocol['id'] += f"_order{job['order_seed']}"
    tasks = protocol['task_order']
    entries = deepcopy(manifest['entries'])
    for row in entries:
        row['task_id'] = tasks.index(row['task_name'])
    entries.sort(key=lambda row: (row['task_id'], row['relative_path']))
    manifest.update(task_order=tasks, protocol_id=protocol['id'], seed=job['seed'], entries=entries)
    manifest['roles'], manifest['online'] = split_prefix(entries, tasks)
    manifest['digest'] = manifest_digest(manifest)
    validate_prefix_manifest(manifest, protocol, verify_bytes=False)
    # Holding out exactly the same prefix images is essential for matched baselines/orders.
    if manifest['roles'] != context['manifest']['roles']:
        raise ValueError('job changed the first-task calibration permissions')
    return protocol, manifest


def validate_w6_review(folder):
    """Require the concrete reviewed evidence, not an arbitrary handoff PASS label."""
    folder = Path(folder)
    receipt = json.loads((folder/'w6_import.json').read_text(encoding='utf-8'))
    if receipt.get('archive_sha256') != W6_SHA256:
        raise ValueError('W7 requires the reviewed completed W6 v4 archive')
    handoff = json.loads((folder/'handoff.json').read_text(encoding='utf-8'))
    if (handoff.get('status') != 'ready_for_artifact_review'
            or not handoff.get('engineering_gates_passed')
            or handoff.get('official_test_accessed') is not False):
        raise ValueError('W6 handoff is not the reviewed normal-only completion')
    return receipt
