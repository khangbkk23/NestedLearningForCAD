"""Short normal-only check of the actual W7 engine, selected M, and ViT checkpoint."""
from pathlib import Path
import gc
import json
import subprocess
import sys
import time
import torch
from torch.utils.data import DataLoader

from dataset.benchmark_protocol_v2 import HashedNormalDataset
from training.benchmark_engine_v1 import _clone_state,_state_equal
from training.hope_outer_learning_v4 import capture_rng,restore_rng
from training.hope_runtime_v4 import atomic_json,resource_usage
from training.hope_w7_engine import W7Engine,SessionLimit,synchronize


def run_preflight(args,context,suite):
    from scripts.hope_cad_v2.run_w7 import adapter_from_initial
    root=Path(args.work_dir)
    # Fixtures include real W7 orchestration/CI checks, and the unchanged v4 regression suite.
    verify=root/'verification_w7.json'
    code=subprocess.call([sys.executable,'-B','scripts/hope_cad_v2/verify_w7.py','--output',str(verify)])
    if code or not json.loads(verify.read_text()).get('success'):
        raise ValueError('W7 regression suite failed')
    initial=torch.load(Path(args.w6_dir)/'selection/M/deployment_initialization.pt',map_location='cpu',weights_only=False)
    rows=context['roles']['roles']['meta_support'][:9]
    pre=context['method']['preprocessing']
    data=HashedNormalDataset(context['protocol']['dataset']['root'],rows,pre['image_size'],pre['mean'],pre['std'],True)
    class NormalOnly:
        task_names=['normal_prefix_part0','normal_prefix_part1']
        def training_paths(self,task): return rows[:6] if task==0 else rows[6:]
        def build_train_loader(self,task):
            from torch.utils.data import Subset
            subset=Subset(data,range(6) if task==0 else range(6,9))
            return DataLoader(subset,batch_size=1,shuffle=False,num_workers=0,
                              generator=torch.Generator().manual_seed(0))
    dataset=NormalOnly()
    base=root/'preflight_engine'
    if (base/'uninterrupted/states/stream_resume.pt').exists():
        # A preflight interruption is rerunnable only in a fresh attempt directory.
        attempt=1
        while (root/f'preflight_engine_{attempt}').exists(): attempt+=1
        base=root/f'preflight_engine_{attempt}'
    adapter=adapter_from_initial(initial,context,{'normal_only_preflight':True},args.device)
    try:
        restore_rng(initial['rng'])
        synchronize(args.device)
        start=time.perf_counter()
        W7Engine(adapter,dataset,base/'uninterrupted',suite['digest'],device=args.device).train()
        synchronize(args.device)
        train_seconds=time.perf_counter()-start
        expected=_clone_state(adapter.state_dict())
        expected_rng=capture_rng()
        batch=next(iter(dataset.build_train_loader(0)))
        before=_clone_state(adapter.state_dict())
        synchronize(args.device);start=time.perf_counter()
        expected_scores=adapter.score_batch(batch)
        synchronize(args.device);score_seconds=time.perf_counter()-start
        if not _state_equal(before,_clone_state(adapter.state_dict())) or not _state_equal(expected_rng,capture_rng()):
            raise RuntimeError('normal-only scoring changed state/RNG')
    finally:
        adapter.close()
    del adapter;gc.collect()
    adapter=adapter_from_initial(initial,context,{'normal_only_preflight':True},args.device)
    try:
        restore_rng(initial['rng'])
        try:
            W7Engine(adapter,dataset,base/'resumed',suite['digest'],device=args.device).train(stop_after_events=5)
        except SessionLimit:
            pass
    finally:
        adapter.close()
    del adapter;gc.collect()
    adapter=adapter_from_initial(initial,context,{'normal_only_preflight':True},args.device)
    try:
        W7Engine(adapter,dataset,base/'resumed',suite['digest'],device=args.device).train()
        if not _state_equal(expected,_clone_state(adapter.state_dict())) or not _state_equal(expected_rng,capture_rng()):
            raise RuntimeError('W7 real-RGB image resume differs from uninterrupted training')
        actual=adapter.score_batch(batch)
        if not _state_equal(actual,expected_scores):
            raise RuntimeError('resumed real-RGB scores differ')
        total_train=len(context['manifest']['online'])
        report=dict(schema='hope_w7_preflight_v1',passed=True,suite_lock_digest=suite['digest'],
            exact_state_rng_score_resume=True,normal_images=9,interrupted_after=5,
            simulated_task_boundary_after=6,normal_image_events_executed=18,
            normal_score_images_executed=2,
            official_test_accessed=False,reportable=False,
            train_seconds_per_image_including_checkpoint_overhead=train_seconds/9,
            score_seconds_per_image=score_seconds,resources=resource_usage(args.device),
            projection=dict(train_images=total_train,train_seconds=(train_seconds/9)*total_train,
                            evaluation='not measured on official test; normal fixture only',
                            caveat='short normal-prefix measurement, not a promised completion time'))
        atomic_json(report,root/'preflight.json')
        return dict(status='w7_preflight_passed',suite_digest=suite['digest'],reportable=False,
                    next='7.1',official_test_accessed=False)
    finally:
        adapter.close()
