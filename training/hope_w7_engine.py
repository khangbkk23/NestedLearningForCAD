"""Resumable W7 image transactions and reusable, exact per-checkpoint evaluations.

7.1 trains once and scores the final row. 7.2 only fills missing triangular cells.
Raw image scores/maps are retained for paired image bootstrap; metrics never tune a model.
"""
from copy import deepcopy
from pathlib import Path
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import file_hash, safe_path
from training.benchmark_engine_v1 import _clone_state, _state_equal
from training.benchmark_metrics_v1 import compute_metrics, macro_task_metrics, forgetting_matrix
from training.hope_outer_learning_v4 import capture_rng, restore_rng
from training.hope_runtime_v4 import atomic_json, atomic_torch_save
from training.hope_w7_contract import MVTEC_COUNTS

METRICS = ('i_auroc', 'p_aupr', 'i_ap', 'p_auroc')


class SessionLimit(Exception):
    """A saved event/shard boundary, never a scientific failure."""


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def remaining_batches(loader, offset=0, batch_size=1):
    # HOPE extracts per image; CADIC keeps its locked loader extraction batch.
    # Both adapters still apply normal updates sequentially per IMAGE.
    return DataLoader(Subset(loader.dataset, range(offset, len(loader.dataset))),
                      batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=loader.collate_fn,
                      generator=torch.Generator().manual_seed(0))


class W7Engine:
    def __init__(self, adapter, dataset, root, run_digest, *, device='cpu',
                 deadline=None, checkpoint_every=16):
        self.adapter, self.dataset = adapter, dataset
        self.root, self.device = Path(root), device
        self.run_digest, self.deadline = run_digest, deadline
        self.checkpoint_every = checkpoint_every
        if checkpoint_every < 1:
            raise ValueError('checkpoint interval must be positive')
        (self.root/'states').mkdir(parents=True, exist_ok=True)
        self.events = 0

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def _state_payload(self, task, cursor):
        return dict(schema='hope_w7_task_state_v1', run_digest=self.run_digest,
                    task_id=task, event_cursor=cursor, adapter=self.adapter.state_dict(), rng=capture_rng())

    def load_checkpoint(self, task):
        path = self.root/'states'/f'task_{task:02d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        expected = sum(len(self.dataset.training_paths(i)) for i in range(task+1))
        if (payload.get('schema') != 'hope_w7_task_state_v1'
                or payload['run_digest'] != self.run_digest or payload['task_id'] != task
                or payload['event_cursor'] != expected):
            raise ValueError('task checkpoint identity/cursor mismatch')
        self.adapter.load_state_dict(payload['adapter'])
        restore_rng(payload['rng'])
        return path

    def train(self, *, stop_after_events=None):
        """Atomic image-boundary resume, including partial slow-CMS accumulators/RNG."""
        path = self.root/'states/stream_resume.pt'
        task, offset, cursor, completed, prior = 0, 0, 0, [], 0.0
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved.get('schema') != 'hope_w7_train_resume_v1' or saved['run_digest'] != self.run_digest:
                raise ValueError('stream source/protocol/runtime identity mismatch')
            task, offset, cursor, completed = (saved[k] for k in ('task','offset','cursor','completed'))
            if (type(task) is not int or type(offset) is not int or type(cursor) is not int
                    or not 0 <= task <= len(self.dataset.task_names) or offset < 0
                    or (task == len(self.dataset.task_names) and offset)
                    or [row['task_id'] for row in completed] != list(range(task))
                    or cursor != sum(len(self.dataset.training_paths(i)) for i in range(task))+offset):
                raise ValueError('corrupt stream cursor')
            self.adapter.load_state_dict(saved['adapter'])
            if hasattr(self.adapter, 'session') and self.adapter.session.cursor != cursor:
                raise ValueError('adapter cursor does not match stream progress')
            restore_rng(saved['rng'])
            prior = saved['elapsed_seconds']
        elif list((self.root/'states').glob('task_*.pt')):
            raise ValueError('task states without stream resume are not a valid continuation')
        synchronize(self.device)
        started = time.perf_counter()

        def save():
            synchronize(self.device)
            atomic_torch_save(dict(schema='hope_w7_train_resume_v1', run_digest=self.run_digest,
                task=task, offset=offset, cursor=cursor, completed=completed,
                elapsed_seconds=prior+time.perf_counter()-started,
                adapter=self.adapter.state_dict(), rng=capture_rng()), path)

        for task in range(task, len(self.dataset.task_names)):
            loader = self.dataset.build_train_loader(task)
            count = len(loader.dataset)
            if offset > count:
                raise ValueError('stream cursor exceeds the current task')
            print(f'[W7 TRAIN] {self.dataset.task_names[task]} {offset}/{count}; total={cursor}', flush=True)
            transport=getattr(self.adapter,'train_transport_batch_size',1)
            for batch in remaining_batches(loader, offset, transport):
                if self.expired():
                    save()
                    raise SessionLimit('training saved before next image')
                self.adapter.fit_task(task, self.dataset.task_names[task], [batch])
                images=int(batch['images'].shape[0])
                cursor += images
                offset += images
                self.events += images
                if offset % self.checkpoint_every == 0 or offset == count:
                    save()
                    print(f'[W7 TRAIN] task={task} image={offset}/{count} total={cursor}', flush=True)
                if stop_after_events is not None and self.events >= stop_after_events:
                    save()
                    raise SessionLimit('explicit test interruption after a completed image')
            atomic_torch_save(self._state_payload(task, cursor), self.root/'states'/f'task_{task:02d}.pt')
            completed.append(dict(task_id=task, task_name=self.dataset.task_names[task],
                                  train_images=count, event_cursor=cursor))
            offset = 0
            task += 1
            save()
            atomic_json(completed, self.root/'train_tasks.json')
        atomic_json(dict(status='train_complete', tasks=completed, event_cursor=cursor,
                         elapsed_seconds=prior+time.perf_counter()-started), self.root/'train_summary.json')
        return completed

    def test_manifest(self, task):
        rows = []
        for raw in self.dataset.test_paths(task):
            row = dict(raw)
            row['image_sha256'] = file_hash(safe_path(self.dataset.root, row['relative_path']))
            row['mask_sha256'] = (file_hash(safe_path(self.dataset.root, row['mask_path']))
                                  if row.get('mask_path') else None)
            rows.append(row)
        paths = [row['relative_path'] for row in rows]
        if not rows or len(paths) != len(set(paths)):
            raise ValueError('empty or duplicated official test identities')
        name=self.dataset.task_names[task]
        if name in MVTEC_COUNTS:
            actual=(sum(row['label']==0 for row in rows),sum(row['label']==1 for row in rows))
            if actual!=MVTEC_COUNTS[name][1:]:
                raise ValueError(f'incomplete official MVTec test inventory for {name}: {actual}')
        manifest = dict(task_name=self.dataset.task_names[task], rows=rows)
        manifest['digest'] = canonical_digest(manifest)
        path = self.root/'test_manifests'/f'{task:02d}.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('official test bytes/identities changed')
        if not path.exists():
            atomic_json(manifest, path)
        return manifest

    def _cell_identity(self, checkpoint, task, manifest):
        path = self.root/'states'/f'task_{checkpoint:02d}.pt'
        return dict(run_digest=self.run_digest, checkpoint=checkpoint, test_task=task,
                    checkpoint_sha256=file_hash(path), test_manifest_digest=manifest['digest'])

    def evaluate_cell(self, checkpoint, task, *, stop_after_shards=None):
        if task > checkpoint:
            raise ValueError('future-task evaluation is forbidden')
        manifest = self.test_manifest(task)
        identity = self._cell_identity(checkpoint, task, manifest)
        folder = self.root/'scores'/f'{checkpoint:02d}_{task:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        progress_path = folder/'progress.json'
        progress = dict(identity=identity, offset=0, shards=[], elapsed_seconds=0.0)
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress['identity'] != identity:
                raise ValueError('score resume checkpoint/test/source mismatch')
        offset = 0
        for shard in progress['shards']:
            if shard['start'] != offset or shard['end'] <= offset:
                raise ValueError('noncontiguous score shards')
            if file_hash(folder/shard['file']) != shard['sha256']:
                raise ValueError('score shard bytes changed')
            offset = shard['end']
        if offset != progress['offset'] or offset > len(manifest['rows']):
            raise ValueError('score cursor differs from committed shards')
        result_path = folder/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result.get('status')!='complete' or result['identity'] != identity
                    or offset != len(manifest['rows'])
                    or any(not isinstance(result['metrics'].get(k),(int,float))
                           or not np.isfinite(result['metrics'][k]) for k in METRICS)):
                raise ValueError('stale completed evaluation cell')
            return result['metrics']
        loader = self.dataset.build_test_loader(task)
        if len(loader.dataset) != len(manifest['rows']):
            raise ValueError('test loader omits official images')
        # Evaluation owns no training state or RNG, even when paused/failed.
        before, rng = _clone_state(self.adapter.state_dict()), capture_rng()
        synchronize(self.device)
        started, done = time.perf_counter(), 0
        try:
            remaining = DataLoader(Subset(loader.dataset, range(offset, len(loader.dataset))),
                                   batch_size=loader.batch_size, shuffle=False, num_workers=0,
                                   collate_fn=loader.collate_fn,
                                   generator=torch.Generator().manual_seed(0))
            for batch in remaining:
                if self.expired():
                    raise SessionLimit('evaluation saved before next score shard')
                n = int(batch['images'].shape[0])
                rows = manifest['rows'][offset:offset+n]
                # Recheck the exact bytes before use, including masks/labels in the evaluator only.
                for row in rows:
                    if (file_hash(safe_path(self.dataset.root,row['relative_path'])) != row['image_sha256']
                            or (row.get('mask_path') and file_hash(safe_path(self.dataset.root,row['mask_path'])) != row['mask_sha256'])):
                        raise ValueError('test bytes changed during evaluation')
                out = self.adapter.score_batch(batch)
                scores = np.asarray(out['image_scores'], dtype=np.float32)
                maps = np.asarray(out['anomaly_maps'], dtype=np.float32)
                labels = np.asarray(batch['labels'], dtype=np.int64)
                masks = np.asarray(batch['masks'], dtype=np.uint8)
                if (scores.shape != (n,) or maps.shape != masks.shape or maps.ndim != 3
                        or not np.isfinite(scores).all() or not np.isfinite(maps).all()
                        or labels.tolist() != [r['label'] for r in rows]
                        or not np.isin(masks,[0,1]).all()):
                    raise ValueError('invalid score/map/label/mask payload')
                if not _state_equal(before, _clone_state(self.adapter.state_dict())):
                    raise RuntimeError('evaluation mutated persistent model state')
                if not _state_equal(rng, capture_rng()):
                    raise RuntimeError('evaluation mutated training RNG')
                name = f'{offset:06d}.npz'
                atomic_npz(folder/name, image_scores=scores, maps=maps, labels=labels, masks=masks,
                           identities=np.asarray([r['relative_path'] for r in rows]))
                synchronize(self.device)
                progress['shards'].append(dict(file=name,start=offset,end=offset+n,sha256=file_hash(folder/name)))
                offset += n
                progress.update(offset=offset, elapsed_seconds=progress['elapsed_seconds']+time.perf_counter()-started)
                atomic_json(progress, progress_path)
                started = time.perf_counter()
                done += 1
                print(f'[W7 EVAL] checkpoint={checkpoint} task={task} images={offset}/{len(loader.dataset)}', flush=True)
                if stop_after_shards is not None and done >= stop_after_shards:
                    raise SessionLimit('explicit test interruption after a committed score shard')
        finally:
            unchanged = _state_equal(before, _clone_state(self.adapter.state_dict()))
            restore_rng(rng)
            if not unchanged:
                raise RuntimeError('evaluation mutated persistent model state')
        arrays = read_cell(folder)
        metrics = compute_metrics(arrays['image_scores'], arrays['labels'], arrays['maps'], arrays['masks'])
        if any(not np.isfinite(metrics[key]) for key in METRICS):
            atomic_json(dict(identity=identity,status='undefined_metric',metrics={k:(v if np.isfinite(v) else None) for k,v in metrics.items()}),result_path)
            raise ValueError('undefined official metric; do not omit this category')
        atomic_json(dict(identity=identity, status='complete', metrics=metrics,
                         evaluation_state_and_rng_unchanged=True, elapsed_seconds=progress['elapsed_seconds']), result_path)
        return metrics

    def final(self):
        final = len(self.dataset.task_names)-1
        self.load_checkpoint(final)
        per = {name:self.evaluate_cell(final,task) for task,name in enumerate(self.dataset.task_names)}
        atomic_json(per,self.root/'metrics/final_per_task.json')
        macro = macro_task_metrics(per)
        atomic_json(macro,self.root/'metrics/final_macro.json')
        cases={}
        for task,name in enumerate(self.dataset.task_names):
            arrays=read_cell(self.root/'scores'/f'{final:02d}_{task:02d}')
            scores,labels,identities=arrays['image_scores'],arrays['labels'],arrays['identities']
            normal=np.flatnonzero(labels==0);anomaly=np.flatnonzero(labels==1)
            cases[name]={label:str(identities[index]) for label,index in (
                ('first_normal',normal[0]),('first_anomaly',anomaly[0]),
                ('highest_score_normal',normal[np.argmax(scores[normal])]),
                ('lowest_score_anomaly',anomaly[np.argmin(scores[anomaly])]))}
        atomic_json(dict(rule='first by canonical identity; highest-score normal; lowest-score anomaly',
                         cases=cases,all_maps_retained=True),self.root/'metrics/qualitative_cases.json')
        return macro

    def matrix(self):
        n = len(self.dataset.task_names)
        matrices = {key:[[None]*n for _ in range(n)] for key in METRICS}
        for checkpoint in range(n):
            self.load_checkpoint(checkpoint)
            for task in range(checkpoint+1):
                measured = self.evaluate_cell(checkpoint,task)
                for key in METRICS:
                    matrices[key][checkpoint][task] = measured[key]
        result = {}
        for key,matrix in matrices.items():
            array = np.asarray(matrix,dtype=float)
            summary = forgetting_matrix(array)
            summary['bwt'] = float(np.mean(array[-1,:-1]-np.diag(array)[:-1])) if n>1 else None
            summary['final_macro'] = float(array[-1].mean())
            result[key] = summary
        atomic_json(matrices,self.root/'metrics/task_matrices.json')
        atomic_json(result,self.root/'metrics/continual_summary.json')
        # Restore the terminal deployment state; adding a matrix never changes the trained run.
        self.load_checkpoint(n-1)
        return result


def read_cell(folder):
    folder = Path(folder)
    progress = json.loads((folder/'progress.json').read_text())
    values = {key:[] for key in ('image_scores','maps','labels','masks','identities')}
    for shard in progress['shards']:
        path = folder/shard['file']
        if file_hash(path) != shard['sha256']:
            raise ValueError('raw score checksum mismatch')
        with np.load(path,allow_pickle=False) as arrays:
            for key in values:
                values[key].append(arrays[key])
    return {key:np.concatenate(parts) for key,parts in values.items()}
