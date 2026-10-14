"""HOPE stream checkpoints at completed image boundaries; v1 runner is unchanged."""
from pathlib import Path
import json
import time
import torch
from torch.utils.data import DataLoader, Subset
from training.benchmark_artifacts_v1 import BenchmarkArtifacts
from training.benchmark_engine_v1 import BenchmarkEngineV1
from training.hope_runtime_v3 import atomic_torch_save, atomic_json


class AtomicBenchmarkArtifacts(BenchmarkArtifacts):
    def save_state(self, name, state):
        atomic_torch_save(state, self.states / name)

    def write_json(self, relative, value):
        atomic_json(value, self.root / relative)


class HOPEBenchmarkEngineV3(BenchmarkEngineV1):
    def train(self, dataset, max_tasks=None, max_train_images=None, resume=False,
              *, checkpoint_every=16, stop_after_events=None):
        if checkpoint_every < 1:
            raise ValueError('checkpoint interval must be positive')
        if not hasattr(dataset, 'manifest'):
            raise ValueError('event resume requires a locked dataset manifest')
        limit = min(len(dataset.task_names), max_tasks or len(dataset.task_names))
        identity = dict(manifest_digest=dataset.manifest['digest'], limit=limit,
                        max_train_images=max_train_images, checkpoint_every=checkpoint_every)
        path = self.artifacts.states / 'stream_resume.pt'
        task_index, offset, rows, prior_seconds = 0, 0, [], 0.0
        if resume and path.exists():
            payload = torch.load(path, map_location=self.device, weights_only=False)
            if payload.get('schema') != 'hope_stream_resume_v3' or payload['identity'] != identity:
                raise ValueError('stream resume protocol/caps/schema mismatch')
            task_index, offset, rows = payload['next_task'], payload['next_index'], payload['completed_tasks']
            if (type(task_index) is not int or type(offset) is not int or not 0 <= task_index <= limit
                    or offset < 0 or len(rows) != task_index or (task_index == limit and offset)):
                raise ValueError('invalid stream task/image cursor')
            if [r['task_id'] for r in rows] != list(range(task_index)):
                raise ValueError('noncontiguous completed tasks')
            expected_cursor = sum(r['train_samples'] for r in rows) + offset
            if payload['adapter']['session']['cursor'] != expected_cursor:
                raise ValueError('image cursor does not match completed event count')
            self.adapter.load_state_dict(payload['adapter'])
            prior_seconds = payload['elapsed_seconds']
        elif any(self.artifacts.states.glob('*.pt')):
            raise ValueError('existing states require a valid event resume checkpoint')
        started, executed = time.perf_counter(), 0
        def save(next_task, next_index):
            atomic_torch_save(dict(schema='hope_stream_resume_v3', identity=identity,
                next_task=next_task, next_index=next_index, completed_tasks=rows,
                adapter=self.adapter.state_dict(),
                elapsed_seconds=prior_seconds+time.perf_counter()-started), path)
        for task_id in range(task_index, limit):
            loader = self._load(dataset.build_train_loader, task_id, max_train_images)
            count = len(loader.dataset)
            if offset > count:
                raise ValueError('stream cursor exceeds task length')
            print(f'[TRAIN] {dataset.task_names[task_id]}: resume {offset}/{count}', flush=True)
            remaining = DataLoader(Subset(loader.dataset, range(offset, count)), batch_size=1,
                                   shuffle=False, num_workers=0, collate_fn=loader.collate_fn,
                                   generator=torch.Generator().manual_seed(0))
            for batch in remaining:
                self.adapter.fit_task(task_id, dataset.task_names[task_id], [batch])
                offset += 1
                executed += 1
                if offset % checkpoint_every == 0 or offset == count or executed == stop_after_events:
                    save(task_id, offset)
                if stop_after_events is not None and executed >= stop_after_events:
                    return dict(status='paused_at_event_boundary', cursor=self.adapter.session.cursor)
            self.artifacts.save_state(f'task_{task_id:02d}.pt', self.adapter.state_dict())
            rows.append(dict(task_id=task_id, task_name=dataset.task_names[task_id], train_samples=count,
                             cursor=self.adapter.session.cursor, commits=list(self.adapter.session.state.commits)))
            offset = 0
            save(task_id + 1, 0)
            self.artifacts.write_json('logs/train.json', rows)
        self.artifacts.save_state('final.pt', self.adapter.state_dict())
        self.times['train_wall_seconds'] = prior_seconds+time.perf_counter()-started
        return rows
