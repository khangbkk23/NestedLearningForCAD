"""Bounded A2/A3 sensitivity on already-seen normals; no retraining or selection."""
from copy import deepcopy
from dataclasses import replace
import time
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader

from dataset.benchmark_protocol_v2 import HashedNormalDataset
from models.hope_cad_v2.visual_memory import stability_matched_step,GuardedStep
from training.benchmark_engine_v1 import _clone_state,_state_equal
from training.hope_outer_learning_v4 import capture_rng,restore_rng
from training.hope_runtime_v4 import atomic_json


def run_interventions(adapter,context,path,*,past_checkpoint=None):
    if path.exists():return
    before=_clone_state(adapter.state_dict());rng=capture_rng()
    rows=context['roles']['roles']['meta_support'][:8]
    pre=context['method']['preprocessing']
    data=HashedNormalDataset(context['protocol']['dataset']['root'],rows,pre['image_size'],pre['mean'],pre['std'],True)
    batches=list(DataLoader(data,batch_size=1,num_workers=0,generator=torch.Generator().manual_seed(0)))
    results={};started=time.perf_counter()
    try:
        reference=[adapter.score_batch(batch) for batch in batches]
        modes=['reset_CMS_readout','single_scan_readout','no_spectral_cap','no_injection_cap',
               'continue_on_8_seen_normals','freeze_CMS_on_8_seen_normals']
        if past_checkpoint is not None:modes.append('earlier_first_task_history_readout')
        for mode in modes:
            adapter.load_state_dict(before)
            restore_rng(rng)
            if mode=='reset_CMS_readout':
                adapter.session.state=adapter.session.model.cms.initial_state()
            if mode=='single_scan_readout':
                with torch.no_grad():adapter.session.model.visual.memory.fusion[1:].zero_()
            if mode in ('continue_on_8_seen_normals','freeze_CMS_on_8_seen_normals'):
                try:
                    for batch in batches:
                        retained=adapter.session.state.weights
                        adapter.fit_task(0,'seen_normal_diagnostic',[batch])
                        if mode=='freeze_CMS_on_8_seen_normals':
                            # Preserve legal clocks/accumulators, suppress both weight applications.
                            adapter.session.state=replace(adapter.session.state,weights=retained)
                except (FloatingPointError,RuntimeError,ValueError) as error:
                    results[mode]=dict(status='unstable_intervention',error=repr(error))
                    continue
            if mode=='earlier_first_task_history_readout':
                previous=torch.load(past_checkpoint,map_location='cpu',weights_only=False)
                if previous.get('schema')!='hope_w7_task_state_v1' or previous['task_id']!=0:
                    raise ValueError('history intervention requires the registered first-task checkpoint')
                adapter.load_state_dict(previous['adapter'])
            def altered(key,delta,raw_eta,alpha,**kwargs):
                original=stability_matched_step(key,delta,raw_eta,alpha,**kwargs)
                if mode=='no_injection_cap':
                    executed=torch.minimum(raw_eta,original.spectral_limit)
                else:
                    safe=torch.where(original.injection_limit>0,original.injection_limit,torch.ones_like(raw_eta))
                    executed=original.injection_limit*(-torch.expm1(-raw_eta/safe))
                return GuardedStep(executed,original.injection_limit,original.spectral_limit)
            try:
                if mode.startswith('no_'):
                    with patch('models.hope_cad_v2.visual_memory.stability_matched_step',side_effect=altered):
                        measured=[adapter.score_batch(batch) for batch in batches]
                else:measured=[adapter.score_batch(batch) for batch in batches]
                delta=torch.cat([(a['anomaly_maps']-b['anomaly_maps']).reshape(-1) for a,b in zip(measured,reference)])
                results[mode]=dict(status='finite',map_delta_rms=delta.double().square().mean().sqrt().item(),
                    image_score_delta_mean=float(torch.cat([a['image_scores']-b['image_scores'] for a,b in zip(measured,reference)]).double().mean()))
            except (FloatingPointError,RuntimeError,ValueError) as error:
                results[mode]=dict(status='unstable_intervention',error=repr(error))
    finally:
        adapter.load_state_dict(before);restore_rng(rng)
        if not _state_equal(before,_clone_state(adapter.state_dict())):
            raise RuntimeError('paired diagnostics mutated the main checkpoint')
    atomic_json(dict(schema='hope_w7_paired_normal_interventions_v1',
        scope='8 predeclared first-task normal identities at terminal M state, all four score views',
        identities=[r['relative_path'] for r in rows],interventions=results,
        retrained_ablations=False,selection_feedback=False,official_test_accessed=False,
        historical_state_policy='read-only intervention clones, no subsequent updates',
        continuation_policy='two discarded 8-event normal continuations: active vs frozen weights with legal clocks',
        continuation_evaluation='same 8 seen normals; in-sample sensitivity, NOT generalization/AD accuracy',
        main_state_and_rng_unchanged=True,elapsed_seconds=time.perf_counter()-started,reportable=False),path)
