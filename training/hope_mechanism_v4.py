"""Post-selection normal development diagnostics with continuous histories.

No probe value enters selection, recipe export or acceptance thresholds.
"""
from copy import deepcopy
from dataclasses import replace
import time
import torch
from models.hope_cad_v2.state import tensor_digest
from training.hope_outer_learning_v4 import capture_rng, restore_rng, unroll_support
from training.hope_mechanism_v3 import score_probe, teacher_controls
from scripts.hope_cad_v2.diagnostics_v4 import visual_diagnostics


def canonical_history(model, data, probe, *, freeze_level=None):
    """0/16/32/70 (or pool length) all belong to ONE canonical history."""
    pool=[(identity,role) for role in ('meta_support','meta_query') for identity in data.ids(role)]
    state=model.cms.initial_state()
    measurements={}
    checkpoints={0,16,32,len(pool)}
    applications=[0]*len(state.weights)
    for t in range(len(pool)+1):
        if t in checkpoints:
            measured,_=score_probe(model,state,probe)
            measured.update(clock=state.clock,stream_events_seen=t,commits=list(state.commits),
                            pending=list(state.pending),actual_weight_applications=list(applications))
            measurements[str(t)]=measured
        if t==len(pool) or not model.dynamics.cms_updates:
            continue
        identity,role=pool[t]
        x,target,mask=data.get(identity,t%4,role)
        with torch.no_grad(): encoded=model.encode(x)
        result=model.cms.event(encoded,target,mask,model.decoder,state)
        weights=list(result.state.weights)
        for level,diagnostic in enumerate(result.diagnostics):
            if level==freeze_level:
                weights[level]=state.weights[level]
            else:
                applications[level]+=int(diagnostic['committed'])
        state=replace(result.state,weights=tuple(weights)).detached()
    return dict(policy='one continuous canonical support+query history',
                freeze_level=freeze_level,checkpoints=measurements)


def mechanism_report(models, data, schedule, probe):
    before={name:tensor_digest((dict(model.named_parameters()),)) for name,model in models.items()}
    rng=capture_rng(); started=time.perf_counter()
    report=dict(schema='hope_mechanism_evidence_v4',reportable=False,selection_feedback=False,
                normal_probe_status='development_exposed_not_independent_confirmation',
                official_test_accessed=False,methods={},interventions={},controls={},
                normal_probe_identities=[identity for identity,_ in probe])
    try:
        report['controls']=teacher_controls(data,probe)
        x,_,_=data.get(data.ids('meta_support')[0],0,'meta_support')
        for name,model in models.items():
            measured=canonical_history(model,data,probe)
            supported=(unroll_support(model,data,schedule['validation_support'],meta=False)
                       if model.dynamics.cms_updates else model.cms.initial_state())
            separate,_=score_probe(model,supported,probe)
            separate.update(clock=supported.clock,policy='independent fixed randomized support history; NOT a point on the canonical curve')
            measured['separate_fixed_16_support']=separate
            measured['selected_visual_diagnostics']=visual_diagnostics(model,x)
            report['methods'][name]=measured
        model=models['M']
        visual_off=deepcopy(model)
        visual_off.dynamics=replace(visual_off.dynamics,visual_updates=False)
        report['interventions']['visual_off_during_adaptation_and_scoring']=canonical_history(visual_off,data,probe)
        for level in range(len(model.cms.levels)):
            report['interventions'][f'freeze_level_{level+1}_during_adaptation']=canonical_history(model,data,probe,freeze_level=level)
        if before!={name:tensor_digest((dict(model.named_parameters()),)) for name,model in models.items()}:
            raise RuntimeError('diagnostics mutated selected learned weights')
        report.update(status='complete_pending_scientific_review',engineering_passed=True,
                      mutable_state_isolated=True,utility_claim='pending_review_no_AD_claim',
                      scope='normal first-task development diagnostic; no cross-task/AD conclusion')
    finally:
        restore_rng(rng)
    report['elapsed_seconds']=time.perf_counter()-started
    report['feature_reads']=dict(data.reads)
    pool_length=sum(len(data.ids(role)) for role in ('meta_support','meta_query'))
    canonical_probe_checkpoints=len({0,16,32,pool_length} & set(range(pool_length+1)))
    report['direct_probe_reads']=len(probe)*4*(
        len(models)*(canonical_probe_checkpoints+1)+(1+len(models['M'].cms.levels))*canonical_probe_checkpoints)
    report['probe_read_accounting']='direct tensor scoring; PrefixFeatures cannot read normal_probe'
    return report
