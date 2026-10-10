"""Prefix-calibrated protocol. Calibration receives only first-task role manifests.

V1 is deliberately untouched. Train byte hashes, not file names alone, bind
resume and matched baselines. Official test discovery stays in the evaluator.
"""
from copy import deepcopy
import csv
import hashlib
from pathlib import Path, PurePosixPath
import torch
from torch.utils.data import DataLoader
from .benchmark_manifest_v1 import canonical_digest, manifest_digest
from .benchmark_protocol_v1 import CANONICAL_MVTEC_TASKS, MVTecContinualProtocol, _ImageDataset

CANONICAL_VISA_TASKS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
                        'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
ROLES = ('meta_support', 'meta_query', 'normal_validation', 'normal_probe')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if (not isinstance(relative, str) or '\\' in relative or ':' in relative or rel.is_absolute()
            or '..' in rel.parts or rel.as_posix() != relative):
        raise ValueError('noncanonical relative path')
    root = Path(root).resolve()
    path = root / relative
    if not path.is_file() or path.resolve() != path or not path.is_relative_to(root):
        raise ValueError(f'missing/symlinked sample: {relative}')
    return path


def split_prefix(entries, tasks):
    first = [row for row in entries if row['task_name'] == tasks[0]]
    p = min(100, len(first))
    if p < 32:
        raise ValueError('first task needs at least 32 normal images')
    sizes = (p // 2, p // 5, (15 * p) // 100)
    bounds = (0, sizes[0], sizes[0] + sizes[1], sum(sizes), p)
    roles = {role: deepcopy(first[bounds[i]:bounds[i + 1]]) for i, role in enumerate(ROLES)}
    excluded = {row['relative_path'] for role in ROLES[2:] for row in roles[role]}
    online = [deepcopy(row) for row in entries if row['relative_path'] not in excluded]
    return roles, online


def build_prefix_manifest(config, *, seed=0, source_commit='unavailable'):
    root, tasks = Path(config['dataset']['root']).resolve(), config['task_order']
    name = config['dataset']['name']
    entries, split_sha = [], None
    if name == 'MVTec AD':
        paths = [(task, p.relative_to(root).as_posix()) for task in tasks
                 for p in sorted((root / task / 'train/good').glob('*.png'))]
    elif name == 'VisA':
        csv_path = safe_path(root, config['dataset']['split_csv'])
        split_sha = file_hash(csv_path)
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        identities = [r['image'] for r in rows]
        if len(set(identities)) != len(identities):
            raise ValueError('duplicate or train/test overlapping VisA identities')
        if any(r['split'] not in ('train','test') or r['label'] not in ('normal','anomaly')
               or r['object'] not in CANONICAL_VISA_TASKS for r in rows):
            raise ValueError('invalid official VisA split/label/category')
        if set(tasks) == set(CANONICAL_VISA_TASKS):
            counts = (sum(r['split']=='train' for r in rows),sum(r['split']=='test' for r in rows),
                      sum(r['split']=='test' and r['label']=='anomaly' for r in rows))
            if counts != (8659,2162,1200):
                raise ValueError(f'VisA official 1cls counts mismatch: {counts}')
        paths = [(r['object'], r['image']) for r in rows
                 if r['split'] == 'train' and r['object'] in tasks]
        if any(r['label'] != 'normal' for r in rows if r['split'] == 'train'):
            raise ValueError('VisA train split must be normal-only')
        paths.sort(key=lambda r: (tasks.index(r[0]), r[1]))
    else:
        raise ValueError('unsupported dataset')
    for task, relative in paths:
        path = safe_path(root, relative)
        entries.append(dict(task_id=tasks.index(task), task_name=task, split='train',
                            relative_path=relative, sha256=file_hash(path)))
    roles, online = split_prefix(entries, tasks)
    manifest = dict(schema_version=2, protocol_id=config['id'], dataset_name=name,
                    task_order=list(tasks), seed=seed, source_commit=source_commit,
                    ordering_rule='task_order_then_lexicographic_relative_path',
                    official_split_sha256=split_sha, entries=entries, roles=roles, online=online)
    manifest['digest'] = manifest_digest(manifest)
    validate_prefix_manifest(manifest, config, verify_bytes=True)
    return manifest


def validate_prefix_manifest(manifest, config, *, verify_bytes=True):
    if (manifest.get('schema_version') != 2 or manifest.get('digest') != manifest_digest(manifest)
            or manifest['protocol_id'] != config['id'] or manifest['task_order'] != config['task_order']
            or manifest['dataset_name'] != config['dataset']['name']):
        raise ValueError('prefix manifest identity mismatch')
    entries, tasks = manifest['entries'], manifest['task_order']
    paths = [r['relative_path'] for r in entries]
    if len(set(paths)) != len(paths) or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('duplicate identities/tasks')
    if entries != sorted(entries, key=lambda r: (r['task_id'], r['relative_path'])):
        raise ValueError('noncanonical train order')
    for i, task in enumerate(tasks):
        if not any(r['task_id'] == i for r in entries):
            raise ValueError('empty train task')
    for row in entries:
        i = row['task_id']
        if type(i) is not int or not 0 <= i < len(tasks) or row['task_name'] != tasks[i] or row['split'] != 'train':
            raise ValueError('invalid train identity')
        if manifest['dataset_name'] == 'MVTec AD':
            path = PurePosixPath(row['relative_path'])
            if len(path.parts) != 4 or path.parts[:3] != (tasks[i], 'train', 'good') or path.suffix != '.png':
                raise ValueError('only official train/good PNGs may enter training')
        if verify_bytes and file_hash(safe_path(config['dataset']['root'], row['relative_path'])) != row['sha256']:
            raise ValueError('training bytes changed')
    roles, online = split_prefix(entries, tasks)
    if manifest['roles'] != roles or manifest['online'] != online:
        raise ValueError('prefix role/online partition mismatch')
    if manifest['dataset_name'] == 'VisA' and verify_bytes:
        root = config['dataset']['root']
        split = safe_path(root, config['dataset']['split_csv'])
        if file_hash(split) != manifest['official_split_sha256']:
            raise ValueError('official VisA split changed')
        with split.open(newline='', encoding='utf-8-sig') as stream:
            allowed = {(r['object'], r['image']) for r in csv.DictReader(stream)
                       if r['split'] == 'train' and r['label'] == 'normal' and r['object'] in tasks}
        if {(r['task_name'], r['relative_path']) for r in entries} != allowed:
            raise ValueError('manifest differs from official VisA train split')


def calibration_roles(manifest):
    """No root, future identities, test annotations, or outside-prefix rows escape."""
    result = dict(schema='hope_prefix_roles_v3', first_task=manifest['task_order'][0],
                  roles=deepcopy(manifest['roles']))
    result['digest'] = canonical_digest(result)
    return result


class HashedNormalDataset(_ImageDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        if file_hash(safe_path(self.root, row['relative_path'])) != row['sha256']:
            raise ValueError('normal sample bytes changed since manifest creation')
        return super().__getitem__(index)


class NonzeroMaskDataset(_ImageDataset):
    def __getitem__(self, index):
        # Keep the v1 RGB transform; load masks by raw nonzero encoding, including
        # palette/0-1 masks whose convert(L) colour could erase positive indices.
        import numpy as np
        from PIL import Image
        row = self.rows[index]
        normal = dict(row, label=0)
        single = _ImageDataset(self.root, [normal], self.image_size,
                               self.mean.flatten().tolist(), self.std.flatten().tolist(), False)
        result = single[0]
        result['labels'] = row['label']
        if row['label']:
            from .visa_mask_qa_v2 import inspect_mask
            inspect_mask(self.root, row, self.image_size)
            with Image.open(safe_path(self.root,row['mask_path'])) as source:
                raw = np.asarray(source)
                if raw.ndim == 3: raw = raw[...,0]
                mask = Image.fromarray((raw != 0).astype(np.uint8)).resize((self.image_size,self.image_size),Image.Resampling.NEAREST)
                result['masks'] = torch.from_numpy(np.asarray(mask).copy()).bool()
        return result


class PrefixContinualProtocol(MVTecContinualProtocol):
    def __init__(self, config, manifest, method, seed, num_workers=0):
        validate_prefix_manifest(manifest, config)
        if manifest['seed'] != seed:
            raise ValueError('seed/manifest mismatch')
        self.config, self.manifest, self.seed = config, manifest, seed
        self.root, self.task_names = Path(config['dataset']['root']).resolve(), list(config['task_order'])
        self.batch_size, self.num_workers = method['runtime']['batch_size'], num_workers
        pre = method['preprocessing']
        self.image_size, self.mean, self.std = pre['image_size'], pre['mean'], pre['std']

    def training_paths(self, task_id):
        return [deepcopy(r) for r in self.manifest['online'] if r['task_id'] == task_id]

    def _loader(self, rows, training):
        dataset_type = HashedNormalDataset if training else NonzeroMaskDataset
        return DataLoader(dataset_type(self.root, rows, self.image_size, self.mean, self.std, training),
                          batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.num_workers, generator=torch.Generator().manual_seed(self.seed))

    def test_paths(self, task_id):
        if self.config['dataset']['name'] == 'MVTec AD':
            return super().test_paths(task_id)
        with safe_path(self.root, self.config['dataset']['split_csv']).open(newline='', encoding='utf-8-sig') as stream:
            rows = [dict(relative_path=r['image'], label=int(r['label'] == 'anomaly'),
                         mask_path=r['mask'] if r['label'] == 'anomaly' else None)
                    for r in csv.DictReader(stream) if r['split'] == 'test' and r['object'] == self.task_names[task_id]]
        if not rows:
            raise ValueError('empty official VisA test task')
        return sorted(rows, key=lambda r: r['relative_path'])

    def metadata(self):
        return {**super().metadata(), 'prefix_roles_digest': calibration_roles(self.manifest)['digest'],
                'online_digest': canonical_digest(self.manifest['online']),
                'data_access': 'prefix_calibrated_not_strict_single_pass'}
