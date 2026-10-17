"""Discarded clone interventions. No anomaly metrics, label input or model selection."""
from copy import deepcopy
from pathlib import Path
import time
import torch
from models.hope_cad_v2.cms import masked_teacher_loss
from models.hope_cad_v2.state import finite_tensors
from training.hope_outer_learning_v4 import capture_rng,restore_rng,episode_loss,make_schedule,short_validation_components
from .contract import PROTOCOL,canon,digest,atomic_json,atomic_torch_save

class BudgetPause(Exception):pass

def boundary(deadline):
    if deadline is not None and time.monotonic()>=deadline:raise BudgetPause('saved boundary; resume latest diagnostic output')

def load(path):return torch.load(path,map_location='cpu',weights_only=False)

def clone_model(model):
    result=deepcopy(model);result.eval();result.requires_grad_(False)
    return result

def state_fingerprint(state):
    from models.hope_cad_v2.state import tensor_digest
    return digest(dict(clock=state.clock,pending=state.pending,commits=state.commits,weights=tensor_digest(state.weights),accumulators=tensor_digest(state.accumulators)))

@torch.no_grad()
def evaluate(model,state,groups):
    result={}
    for name,events in groups.items():
        totals=[];norms=[];cosines=[]
        for x,target,mask in events:
            pred=model.decoder(model.cms(x,state)[-1])[:,mask];truth=target[:,mask]
            finite_tensors((pred,truth),'diagnostic evaluation')
            totals.append((pred-truth).square().sum(-1).double().mean())
            norms.append(pred.norm(dim=-1).double().mean())
            cosines.append(torch.nn.functional.cosine_similarity(pred,truth,dim=-1,eps=1e-8).double().mean())
        if not totals:raise ValueError('empty diagnostic evaluation group')
        result[name]=dict(loss=.5*float(torch.stack(totals).mean()),prediction_norm=float(torch.stack(norms).mean()),cosine=float(torch.stack(cosines).mean()),image_view_count=len(events))
    return result

def parameter_change(before,after):
    return float(torch.stack([(a.detach().double()-b.detach().double()).square().sum() for a,b in zip(before,after)]).sum().sqrt())

def capacity_arm(parent,parent_state,groups,fit_events,cursor,arm,folder,*,identity,deadline=None,steps=None,checkpoints=None):
    if arm not in PROTOCOL['arms']:raise ValueError('unknown diagnostic arm')
    steps=PROTOCOL['capacity_steps'] if steps is None else steps
    checkpoints=PROTOCOL['capacity_checkpoints'] if checkpoints is None else checkpoints
    folder=Path(folder);done=folder/'result.json';active=folder/'active.pt'
    identity=canon(dict(**identity,arm=arm,steps=steps,checkpoints=checkpoints))
    if done.exists():
        out=__import__('json').loads(done.read_text())
        if out['identity']!=identity:raise ValueError('completed arm identity changed')
        active.unlink(missing_ok=True);return out
    model=clone_model(parent);state=parent_state.detached()
    cms_params=[p for w in state.weights for p in w.values()]
    for p in cms_params:p.requires_grad_(arm in ('cms_adam','joint_adam','deployed_cms'))
    if arm in ('decoder_adam','joint_adam'):model.decoder.requires_grad_(True)
    decoder_params=list(model.decoder.parameters())
    params=([p for p in cms_params] if arm in ('cms_adam','joint_adam') else [])+([p for p in decoder_params] if arm in ('decoder_adam','joint_adam') else [])
    optimizer=torch.optim.Adam(params,**{k:v for k,v in PROTOCOL['aux_adam'].items() if k!='clip_norm'}) if params else None
    before_cms=[p.detach().clone() for p in cms_params];before_decoder=[p.detach().clone() for p in decoder_params]
    from .contract import event_order
    order=event_order(cursor,steps)
    trace=[];start=0
    if active.exists():
        saved=load(active)
        if saved['identity']!=identity:raise ValueError('arm resume identity changed')
        model.load_state_dict(saved['model']);state=model.cms.load_state(saved['state'])
        cms_params=[p for w in state.weights for p in w.values()]
        for p in cms_params:p.requires_grad_(arm in ('cms_adam','joint_adam','deployed_cms'))
        params=(cms_params if arm in ('cms_adam','joint_adam') else [])+(list(model.decoder.parameters()) if arm in ('decoder_adam','joint_adam') else [])
        optimizer=torch.optim.Adam(params,**{k:v for k,v in PROTOCOL['aux_adam'].items() if k!='clip_norm'}) if params else None
        if optimizer:optimizer.load_state_dict(saved['optimizer'])
        trace,start=saved['trace'],saved['step'];restore_rng(saved['rng'])
    def save(step):
        atomic_torch_save(dict(identity=identity,model=model.state_dict(),state=state.checkpoint(),optimizer=optimizer.state_dict() if optimizer else None,trace=trace,step=step,rng=capture_rng()),active)
    if not trace:
        trace.append(dict(step=0,groups=evaluate(model,state,groups)));save(0)
    for step in range(start+1,steps+1):
        boundary(deadline)
        image,view=order[step-1];x,target,mask=fit_events[image][view]
        if arm=='deployed_cms':state=model.cms.event(x,target,mask,model.decoder,state,meta=False).state
        else:
            optimizer.zero_grad(set_to_none=True)
            loss=masked_teacher_loss(model.decoder(model.cms(x,state)[-1]),target,mask)
            finite_tensors((loss,),'auxiliary fit');loss.backward()
            if any(p.grad is None for p in params):raise RuntimeError('disconnected auxiliary parameter')
            torch.nn.utils.clip_grad_norm_(params,PROTOCOL['aux_adam']['clip_norm'],error_if_nonfinite=True)
            optimizer.step();finite_tensors(params,'auxiliary proposal')
            finite_tensors((v for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v)),'auxiliary optimizer')
        if step in checkpoints:trace.append(dict(step=step,groups=evaluate(model,state,groups)))
        save(step)
    out=dict(identity=identity,status='complete',trace=trace,
        cms_change_l2=parameter_change(before_cms,[p for w in state.weights for p in w.values()]),
        decoder_change_l2=parameter_change(before_decoder,list(model.decoder.parameters())),
        clock_before=parent_state.clock,clock_after=state.clock,
        clock_meaning='exact deployment CMS events' if arm=='deployed_cms' else 'unchanged metadata; Adam intervention is not deployment dynamics',
        fitted_weights_discarded=True,independent_holdout=False,reportable=False)
    atomic_json(out,done);active.unlink(missing_ok=True)
    return out

def latent_oracle(model,state,event,folder,*,identity,deadline=None,steps=None):
    folder=Path(folder);done=folder/'result.json';active=folder/'active.pt'
    steps=PROTOCOL['latent_oracle']['steps'] if steps is None else steps
    identity=canon(dict(**identity,steps=steps))
    if done.exists():
        out=__import__('json').loads(done.read_text())
        if out['identity']!=identity:raise ValueError('oracle identity changed')
        active.unlink(missing_ok=True);return out
    x,target,mask=event;decoder=deepcopy(model.decoder).requires_grad_(False).eval()
    with torch.no_grad():h0=model.cms(x,state)[-1][:,mask].clone()
    h=h0.detach().clone().requires_grad_(True);truth=target[:,mask]
    optimizer=torch.optim.Adam([h],lr=PROTOCOL['latent_oracle']['lr'])
    def loss():return .5*(decoder(h)-truth).square().sum(-1).mean()
    initial=float(loss().detach());best=initial;start=0
    if active.exists():
        old=load(active)
        if old['identity']!=identity:raise ValueError('oracle resume identity changed')
        with torch.no_grad():h.copy_(old['latent'].to(h))
        optimizer.load_state_dict(old['optimizer']);best,start=old['best'],old['step']
    for step in range(start+1,steps+1):
        boundary(deadline);optimizer.zero_grad(set_to_none=True);value=loss()
        finite_tensors((value,),'latent oracle');value.backward();finite_tensors((h.grad,),'latent gradient');optimizer.step()
        terminal=float(loss().detach());finite_tensors((h,loss()),'latent proposal');best=min(best,terminal)
        atomic_torch_save(dict(identity=identity,latent=h.detach().cpu(),optimizer=optimizer.state_dict(),best=best,step=step),active)
    out=dict(identity=identity,initial_loss=initial,terminal_loss=float(loss().detach()),best_found_loss=best,
        latent_change_l2=float((h.detach()-h0).double().norm()),latent_norm=float(h.detach().norm(dim=-1).mean()),
        scope=PROTOCOL['latent_oracle']['scope'],query_teacher_used_for_same_image_fit=True,
        lower_bound_certificate=False,weights_discarded=True,reportable=False)
    atomic_json(out,done);active.unlink(missing_ok=True);return out

def continuation(model,data,parent,folder,*,identity,deadline=None,extra_steps=None,validation_offsets=None):
    # Auxiliary 400->500 continuation, not the frozen fit/selection implementation.
    cfg=PROTOCOL['continuation'];extra=cfg['to_step']-cfg['from_step'] if extra_steps is None else extra_steps
    offsets=[0,25,50,75,100] if validation_offsets is None else validation_offsets
    if parent['step']!=cfg['from_step'] or parent['max_steps']!=400 or parent['outer_lr']!=cfg['outer_lr']:raise ValueError('wrong original outer checkpoint')
    if canon(parent['metadata'])!=canon(model.metadata()):raise ValueError('outer model metadata differs')
    schedule=make_schedule(data,cfg['seed'],steps=cfg['from_step']+extra)
    prior=parent['schedule']
    if (canon(schedule['episodes'][:400])!=canon(prior['episodes']) or schedule['roles_digest']!=prior['roles_digest'] or schedule['validation_support']!=prior['validation_support']):raise ValueError('outer schedule prefix changed')
    folder=Path(folder);done=folder/'result.json';active=folder/'active.pt'
    identity=canon(dict(**identity,schedule_digest=schedule['digest'],extra_steps=extra,validation_offsets=offsets))
    if done.exists():
        out=__import__('json').loads(done.read_text())
        if out['identity']!=identity:raise ValueError('continuation identity changed')
        active.unlink(missing_ok=True);return out
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=cfg['outer_lr'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    saved=load(active) if active.exists() else None
    if saved and saved['identity']!=identity:raise ValueError('continuation resume changed')
    model.load_state_dict(saved['weights'] if saved else parent['weights']);optimizer.load_state_dict(deepcopy(saved['optimizer'] if saved else parent['optimizer']))
    restore_rng(saved['rng'] if saved else parent['rng']);data.reads=dict(saved['reads'] if saved else parent['feature_reads_total'])
    trace=saved['trace'] if saved else [];start=saved['step'] if saved else 400;training=saved['training'] if saved else []
    def save(step):atomic_torch_save(dict(identity=identity,weights=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(),reads=data.reads,trace=trace,training=training,step=step),active)
    if not trace:
        boundary(deadline);trace.append(dict(step=400,**short_validation_components(model,data,schedule['validation_support'])));save(400)
    for step in range(start+1,400+extra+1):
        boundary(deadline);started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        loss=episode_loss(model,data,schedule['episodes'][step-1]);finite_tensors((loss,),'outer continuation');loss.backward()
        if any(p.grad is None for p in params):raise RuntimeError('missing outer gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();finite_tensors(params,'outer proposal')
        finite_tensors((v for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v)),'outer optimizer')
        training.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-started))
        if step-400 in offsets:trace.append(dict(step=step,**short_validation_components(model,data,schedule['validation_support'])))
        save(step)
        if step%5==0:print('[CONTINUATION]',step,'/ 500 loss',float(loss.detach()),flush=True)
    out=dict(identity=identity,status='complete',trace=trace,training=training,reads=data.reads,selected=False,probe_reads=data.reads['normal_probe'],fitted_weights_discarded=True,reportable=False)
    if out['probe_reads']:raise RuntimeError('probe accessed')
    atomic_json(out,done);active.unlink(missing_ok=True);return out
