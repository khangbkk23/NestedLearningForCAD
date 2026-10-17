"""Read-only normal diagnostics: actual stages, radial/angular residual and local proposals."""
from pathlib import Path
import json
import hashlib
import torch
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from dataset.hope_mask_views import patch_masks,mask_normalized_rgb
from models.hope_cad_v2.teacher_head import frozen_teacher
from models.hope_cad_v2.cms import masked_teacher_loss
from training.benchmark_engine_v1 import _clone_state,_state_equal
from training.hope_outer_learning_v4 import capture_rng,restore_rng
from .common import (atomic_json,sha,digest,PROTOCOL,sampled_rows,assert_legal_normal,check_deadline)


def statistics(values):
    v=values.detach().double().cpu().flatten()
    if not v.numel() or not bool(torch.isfinite(v).all()):raise FloatingPointError('nonfinite/empty diagnostic')
    q=torch.quantile(v,torch.tensor([.5,.95,.99],dtype=torch.float64))
    return dict(count=v.numel(),mean=float(v.mean()),p50=float(q[0]),p95=float(q[1]),p99=float(q[2]),max=float(v.max()),min=float(v.min()))


def residual_components(prediction,target):
    r,s=prediction.norm(dim=-1),target.norm(dim=-1)
    dot=(prediction*target).sum(-1)
    score=(prediction-target).square().sum(-1)
    radial=(r-s).square()
    angular=2*(r*s-dot)
    cosine=(dot/(r*s).clamp_min(1e-12)).clamp(-1,1)
    error=float((score-radial-angular).abs().max())
    if error>1e-5*max(1.,float(score.max())):raise RuntimeError('residual decomposition arithmetic failed')
    return dict(score=score,radial=radial,angular=angular,cosine=cosine,prediction_norm=r,teacher_norm=s)


def measure_normal(adapter,image,*,gradients=True):
    """Never update adapter; the one-event proposal is discarded, not an observed trajectory."""
    before=_clone_state(adapter.state_dict());rng=capture_rng()
    model,state=adapter.session.model,adapter.session.state
    c=model.visual.config; n=c.height*c.width
    image=image.to(adapter.device)
    gathered={k:torch.empty(n,device=image.device) for k in ('score','radial','angular','cosine','prediction_norm','teacher_norm')}
    stages={};covered=torch.zeros(n,dtype=torch.int32,device=image.device);one=None
    masks=patch_masks(c.height,c.width,device=image.device)
    try:
        with torch.no_grad():
            clean=adapter._extract(image);target=frozen_teacher(clean)
            for view in range(4):
                features=adapter._extract(mask_normalized_rgb(image,view,height=c.height,width=c.width))
                encoded=model.encode(features);outputs=model.cms(encoded,state)
                pred=model.decoder(outputs[-1]);components=residual_components(pred,target)
                for key,values in components.items():gathered[key][masks[view]]=values[0,masks[view]]
                covered[masks[view]]+=1
                stage=dict(raw_masked=features,projection=model.visual.projection(features),visual_output=encoded)
                stage.update({f'cms_{j+1}':h for j,h in enumerate(outputs)})
                for key,h in stage.items():stages.setdefault(key,[]).append(h.norm(dim=-1)[0,masks[view]])
                if view==adapter.session.cursor%4:one=(encoded,target,masks[view])
            if not bool((covered==1).all()):raise ValueError('diagnostic views do not cover patches exactly once')
            winner=int(gathered['score'].argmax())
            measured=dict(normal_loss=.5*float(gathered['score'].mean()),image_score=float(gathered['score'].max()),
                score_components={key:statistics(value) for key,value in gathered.items()},
                activation_norms={key:statistics(torch.cat(value)) for key,value in stages.items()},
                clean_raw_norm=statistics(clean.norm(dim=-1)),
                prediction_norm_gt1_fraction=float((gathered['prediction_norm']>1.00001).float().mean()),
                prediction_norm_gt2_fraction=float((gathered['prediction_norm']>2).float().mean()),
                max_score_patch=dict(index=winner,row=winner//c.width,column=winner%c.width,
                    components={key:float(value[winner]) for key,value in gathered.items()}))
        if gradients and model.dynamics.cms_updates:
            encoded,target,mask=one
            result=model.cms.event(encoded,target,mask,model.decoder,state,meta=False)
            with torch.no_grad():
                after=model.decoder(model.cms(encoded,result.state)[-1])
                after_loss=masked_teacher_loss(after,target,mask)
            measured['discarded_local_proposal']=dict(scope='same-normal-image next-view sensitivity; not held-out/generalization',
                pre_clock=state.clock,proposal_clock=result.state.clock,view=adapter.session.cursor%4,
                losses_before=[float(v) for v in result.losses],final_loss_after=float(after_loss),
                levels=list(result.diagnostics),applied_to_main_state=False)
        else:measured['discarded_local_proposal']=dict(disabled=True,reason='static control or explicit preflight')
        if not _state_equal(before,_clone_state(adapter.state_dict())):raise RuntimeError('diagnostic mutated model/state')
        if not _state_equal(rng,capture_rng()):raise RuntimeError('diagnostic consumed training RNG')
        measured['main_state_and_rng_unchanged']=True
        return measured
    finally:
        restore_rng(rng)
        if not _state_equal(before,_clone_state(adapter.state_dict())):
            adapter.load_state_dict(before)
            raise RuntimeError('diagnostic mutation rolled back; result rejected')


def run_diagnostics(adapter,dataset,engine,folder,job,study_digest,*,deadline=None):
    folder=Path(folder);allowed={r['relative_path']:r for r in dataset.manifest['online']}
    samples={j:sampled_rows(dataset,j,PROTOCOL['diagnostic']['samples_per_task']) for j in range(len(dataset.task_names))}
    pre=dataset
    manifest=dict(schema='recovery_normal_sample_manifest_v1',selection='identity SHA256 order; first 8 online normals per task',
        source='manifest.online only; no validation/probe/test',samples={dataset.task_names[j]:rows for j,rows in samples.items()},
        independent_holdout=False,official_test_accessed=False,selection_feedback=False)
    sample_path=folder/'sample_manifest.json'
    if sample_path.exists() and json.loads(sample_path.read_text())!=manifest:raise ValueError('normal sample manifest changed')
    atomic_json(manifest,sample_path)
    terminal=len(dataset.task_names)-1
    for checkpoint in range(len(dataset.task_names)):
        tasks=list(range(len(dataset.task_names))) if checkpoint==terminal else [checkpoint]
        checkpoint_path=engine.root/'states'/f'task_{checkpoint:02d}.pt'
        checkpoint_sha=sha(checkpoint_path)
        engine.load_checkpoint(checkpoint)
        for task in tasks:
            for row in samples[task]:
                assert_legal_normal(row,allowed,checkpoint)
                check_deadline(deadline)
                identity=dict(study_digest=study_digest,job=job,checkpoint=checkpoint,checkpoint_sha256=checkpoint_sha,
                    row=row,protocol_digest=digest(PROTOCOL),normal_only=True)
                key=hashlib.sha256(row['relative_path'].encode()).hexdigest()
                target=folder/'samples'/f'{checkpoint:02d}_{task:02d}'/(key+'.json')
                if target.exists():
                    if json.loads(target.read_text())['identity']!=identity:raise ValueError('diagnostic resume identity changed')
                    continue
                loader=HashedNormalDataset(dataset.root,[row],pre.image_size,pre.mean,pre.std,True)
                image=loader[0]['images'].unsqueeze(0)
                measured=measure_normal(adapter,image)
                atomic_json(dict(identity=identity,measurement=measured),target)
            print(f'[NORMAL DIAGNOSTIC] {job} checkpoint={checkpoint} task={task} samples={len(samples[task])}',flush=True)
    summary={}
    for task,name in enumerate(dataset.task_names):
        values={}
        for history,checkpoint in [('diagonal',task),('final',terminal)]:
            records=[json.loads(p.read_text())['measurement'] for p in sorted((folder/'samples'/f'{checkpoint:02d}_{task:02d}').glob('*.json'))]
            if len(records)!=len(samples[task]):raise ValueError('incomplete diagnostic category')
            values[history]=dict(mean_normal_loss=sum(x['normal_loss'] for x in records)/len(records),
                mean_max_score=sum(x['image_score'] for x in records)/len(records),
                mean_prediction_norm=sum(x['score_components']['prediction_norm']['mean'] for x in records)/len(records))
        values['final_minus_diagonal_normal_loss']=values['final']['mean_normal_loss']-values['diagonal']['mean_normal_loss']
        summary[name]=values
    engine.load_checkpoint(terminal)
    report=dict(schema='recovery_normal_diagnostics_v1',status='complete',job=job,
        samples_per_task=PROTOCOL['diagnostic']['samples_per_task'],category_summaries=summary,
        main_state_and_rng_unchanged=True,official_test_accessed=False,
        interpretation='reused train normals: descriptive acquisition quality and retention, not independent validation or AD gain',
        selection_feedback=False,reportable=False)
    atomic_json(report,folder/'summary.json')
    return report