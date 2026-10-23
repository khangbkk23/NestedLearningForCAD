"""Explicit Vast/SSH research execution: bounded fresh full-train cohorts only."""
import gc
import json
import time
from pathlib import Path
import torch
from hope_cad.contract import digest,canonical,file_sha
from hope_cad.research import contract,method_identity,study_digest,job,session_for
from hope_cad.runtime.identity import (source_digest,configure_runtime,runtime_identity,
    check_remote_runtime,output_directory)
from hope_cad.runtime.execution import exclusive_work,verify_checkpoint
from hope_cad.runtime.system import atomic_json,resource_usage
from hope_cad.runtime.evidence import export_archive
from hope_cad.runtime.backend import activation
from hope_cad.runtime.budget import Budget
from hope_cad.runtime.measurement import Measurement
from hope_cad.benchmark.engine import train,evaluate,save_or_verify,Pause,boundary,maps_from_patches
from hope_cad.benchmark.study import lock
from hope_cad.benchmark.stream import _state_equal
from hope_cad.data.fulltrain import build_manifest

EVIDENCE_SCHEMA='hope_hq_fulltrain_evidence_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def binding(rt,rows_digest,checkpoint):
    return canonical(dict(source_digest=source_digest(),research_study_digest=study_digest(),runtime=rt,
        normal_rows_digest=rows_digest,backbone_sha256=checkpoint,protocol=contract()['study']['protocol_id'],
        producer='frozen_ViT_block8_784x768_FP32_ImageNet_direct224bilinear',backend='fixed_shape_cached_NN_imagewise_Eq9b9'))


def assert_preflight(work,identity):
    p=read(Path(work)/'preflight.json')
    if p.get('status')!='research_preflight_passed_pending_review' or p.get('binding')!=identity or p.get('binding_digest')!=digest(identity):raise ValueError('Research preflight identity/acceptance mismatch')


def record_review(work,phase,note):
    """Explicit operator attestation after independent review; never infer acceptance."""
    work=Path(work).resolve()
    if phase not in ('preflight','pilot') or not note.strip():raise ValueError('Review phase and explanation required')
    receipt=work/(phase+'.json');v=read(receipt)
    if (phase=='preflight' and v.get('status')!='research_preflight_passed_pending_review') or (phase=='pilot' and v.get('passed') is not True):raise ValueError('Cannot accept failed/incomplete technical/scientific gates')
    value=dict(accepted=True,phase=phase,note=note,receipt_sha256=file_sha(receipt),source_digest=source_digest(),candidate_adopted=False)
    if phase=='pilot':value['pilot_receipt_sha256']=value['receipt_sha256']
    with exclusive_work(work):save_or_verify(work/(phase+'_review.json'),value)
    return value


def history_numerical_gate(device='cpu'):
    from hope_cad.models.learner import UtilityModel
    from hope_cad.models.consolidation import query_objective
    from hope_cad.models.primitives.state import finite_tensors
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(45);model=UtilityModel().to(device).eval()
    gen=torch.Generator().manual_seed(88)
    rs=[dict(tokens=torch.randn(1,784,32,generator=gen).to(device),stats=torch.rand(16,generator=gen).to(device),
        c=3.,gamma=.02,target=(-1)**i*.3) for i in range(10)]
    params=[p for _,p in model.slow_parameters()];initial=model.initial_fast();queries=rs[6:8]+rs[8:10]
    exact,losses=query_objective(model,initial,rs[:4],queries,'H32_EXACT')
    historical=torch.autograd.grad(losses[2:].mean(),params,retain_graph=True)
    ge=torch.autograd.grad(exact,params)
    fo,_=query_objective(model,initial,rs[:4],queries,'H32_FO');gf=torch.autograd.grad(fo,params)
    finite_tensors((exact,fo,*ge,*gf,*historical),'full-grid historical exact/FO')
    if not torch.equal(exact.detach(),fo.detach()) or not any(bool(v.abs().max()>0) for v in historical):raise ValueError('Historical gate inactive or FO forward differs')
    return dict(status='passed',grid=[28,28],channels=32,all_active_connected=True,historical_gradient_nonzero=True,
        exact_FO_forward_equal=True,exact_loss=float(exact.detach()),scope='discarded synthetic numerical acceptance, not AD')


def preflight(work,rows,provider,device,deadline):
    from hope_cad.baselines.cadic.cache import reference_parity
    if (work/'preflight.json').exists():raise ValueError('Preflight already published; use pilot, do not rerun/tune')
    result=dict(numerical=history_numerical_gate(device),clones={})
    for variant in ('CADIC','R_EXACT','H32_EXACT','H32_DIRECT'):
        a=session_for(variant+'_seed0',device);middle=None
        with activation(True):
            for i,row in enumerate(rows[:28]):
                boundary(deadline);a.observe(provider.get(row),row['relative_path'])
                if i==23:middle=a.state_dict()
                if (i+1)%4==0:print('[RESEARCH PREFLIGHT]',variant,i+1,'/28',flush=True)
            b=session_for(variant+'_seed0',device);b.load_state_dict(middle)
            for row in rows[24:28]:b.observe(provider.get(row),row['relative_path'])
            if not _state_equal(a.state_dict(),b.state_dict()):raise ValueError('Discarded clone resume mismatch')
            before=a.state_dict();raw=provider.get(rows[0]);scores=a.score(raw)
            if not _state_equal(before,a.state_dict()) or not all(torch.isfinite(x).all() for x in scores):raise ValueError('Read-only score gate')
        result['clones'][variant]=dict(images=24,extra_resume_images=4,counters=a.counters,memory=a.memory_profile())
        if variant=='CADIC':
            # Check real full-budget cache/NN Eq9 in both routes after leaving activation below.
            result['full_budget_pair_parity']=reference_parity(a.bank,raw[0,:4],raw[:,:7])
            with torch.no_grad():ref=a.score(raw)
            with activation(True),torch.no_grad():fast=a.score(raw)
            if not _state_equal(ref,fast):raise ValueError('Full-budget optimized NN/Eq9 differs')
            result['optimized_NN_Eq9_exact']=True
        del a,b,middle,before;gc.collect();torch.cuda.empty_cache()
    return result


def pilot_gates(work,rows,provider,device):
    cfg=contract()['study'];summaries={};actors={}
    for v in cfg['B_pilot_job_order']:
        name=v+'_seed0';folder=work/'pilot'/name
        done=read(folder/'train_complete.json');value=torch.load(folder/'active.pt',map_location='cpu',weights_only=True)
        actor=session_for(name,device);actor.load_state_dict(value['session'])
        if done['images']!=256 or actor.seen_ids!=[r['relative_path'] for r in rows]:raise ValueError('Incomplete pilot identities')
        events=[read(folder/'events'/f'{i:05d}.json') for i in range(256)]
        labels=actor.counters['labels_completed'];slow=actor.learner.slow_updates if actor.learner else 0
        episodes=[e['credit']['learning']['episode'] for e in events if e.get('credit') and 'episode' in e['credit']['learning']]
        summaries[v]=dict(labels=labels,slow_updates=slow,clock_passed=(v=='CADIC' or labels>=64) and
            (slow>=8 if v in ('R_EXACT','R_FO','H32_EXACT','H32_FO','H32_DIRECT') else slow==0),
            historical_gradient_nonzero=any((e['historical_gradient_norm'] or 0)>0 for e in episodes),
            prequential_mean_last128=sum(e['prequential_risk'] for e in events[128:])/128,
            counters=actor.counters,memory=actor.memory_profile(),elapsed_train_seconds=done['elapsed_seconds'],
            episode_count=len(episodes),clip_count=sum(e['clipped'] is True for e in episodes),zero_gradient_count=sum(e['zero_gradient'] is True for e in episodes))
        if v in ('CADIC','H32_EXACT'):actors[v]=actor
        else:del actor
    a,b=actors['H32_EXACT'],actors['CADIC'];ca=a.counters
    rate=ca['rejected_learned']/ca['learned_decisions'] if ca['learned_decisions'] else None
    maxdiff=0.;before=(a.state_dict(),b.state_dict())
    probe_ids=[]
    for row in rows[:24]:
        x=provider.get(row);ap=a.score(x)[0].detach().cpu().numpy();bp=b.score(x)[0].detach().cpu().numpy()
        diff=abs(maps_from_patches(ap)-maps_from_patches(bp)).max();maxdiff=max(maxdiff,float(diff));probe_ids.append(row['relative_path'])
    if not _state_equal(before,(a.state_dict(),b.state_dict())):raise ValueError('Pilot probes mutated state')
    ratio=summaries['H32_EXACT']['prequential_mean_last128']/summaries['CADIC']['prequential_mean_last128']
    gates=dict(identities_and_clocks=all(s['clock_passed'] for s in summaries.values()),
        historical_gradient=summaries['H32_EXACT']['historical_gradient_nonzero'],reject_fraction=rate is not None and .01<=rate<=.99,
        different_bank=not torch.equal(a.bank.features,b.bank.features),score_effect=maxdiff>1e-6,normal_risk=ratio<=1.10)
    return dict(gates=gates,passed=all(gates.values()),arms=summaries,risk_ratio=ratio,reject_fraction=rate,max_map_difference=maxdiff,
        probe_ids=probe_ids,probe_manifest_digest=digest(rows[:24]),risk_window_ids=[r['relative_path'] for r in rows[128:]],
        interpretation='normal-prefix mechanism/safety, not AD or independent generalization')


def run(c):
    configure_runtime();started=time.monotonic();work=output_directory(c['run_dir'],data_root=c['data_root'],checkpoint=c['checkpoint'])
    phase=c['phase'];cfg=contract()['study'];failure=None;budget=None;extractor=None;meter=None;status='failed_preserve_evidence'
    if phase not in ('preflight','pilot','main','extension'):raise ValueError('Invalid research stage')
    with exclusive_work(work):
        try:
            allocation=read(c['allocation']) if c.get('allocation') else None
            budget=Budget(work,'B' if phase in ('preflight','pilot') else ('C' if phase=='main' else 'D'),c['session_hours'],c.get('external_billed_seconds',0),allocation)
            meter=Measurement(work,budget.entry['id'],c['device'])
            rt=check_remote_runtime(c['device'],c['expected_gpu'])
            manifest,rd=build_manifest(c['data_root'],source_digest());checkpoint=verify_checkpoint(c['checkpoint']);identity=binding(rt,rd,checkpoint)
            lock(work/'suite_lock.json',identity);save_or_verify(work/'manifest_train.json',manifest)
            if phase!='preflight':
                assert_preflight(work,identity)
                review=read(work/'preflight_review.json')
                if review.get('accepted') is not True or review.get('source_digest')!=identity['source_digest'] or review.get('receipt_sha256')!=file_sha(work/'preflight.json'):raise ValueError('Review preflight before pilot/main')
            if phase in ('main','extension'):
                pg=read(work/'pilot_review.json')
                if pg.get('accepted') is not True or pg.get('source_digest')!=identity['source_digest'] or pg.get('pilot_receipt_sha256')!=file_sha(work/'pilot.json') or not read(work/'pilot.json')['passed']:raise ValueError('Review completed pilot before main/extension')
            if phase=='extension':
                if not read(work/'development_screen.json')['passed']:raise ValueError('Stage C screen did not permit extension')
            from hope_cad.models.features import CADICViTConfig,CADICViTFeatureExtractor
            from hope_cad.benchmark.data import NormalFeatures,EvaluationFeatures,test_manifest
            extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),c['device'])
            with activation(phase!='preflight'):
                if phase=='preflight':
                    from hope_cad.benchmark.pilot import NormalFeatureCache
                    provider=NormalFeatureCache(c['data_root'],extractor,work/'discarded_cache',work/'preflight_features',c['device'])
                    with meter.phase('discarded_preflight'):
                        checks=preflight(work,manifest['online'],provider,c['device'],budget.deadline)
                    status='research_preflight_passed_pending_review'
                    atomic_json(dict(status=status,binding=identity,binding_digest=digest(identity),checks=checks,learned_state_reused=False),work/'preflight.json')
                elif phase=='pilot':
                    rows=manifest['online'][:256]
                    for variant in cfg['B_pilot_job_order']:
                        name=variant+'_seed0';folder=work/'pilot'/name;provider=NormalFeatures(c['data_root'],extractor,folder/'feature_receipts',c['device'])
                        ji=dict(job=name,mode=variant,seed=0,suite_digest=digest(identity),method_digest=digest(method_identity(variant)),scope='discarded_pilot_256')
                        with meter.phase('train',name):
                            actor=train(folder,rows,manifest['task_order'][:2],ji,provider,lambda:session_for(name,c['device']),deadline=budget.deadline)
                            del actor;gc.collect();torch.cuda.empty_cache()
                    boundary(budget.deadline)
                    provider=NormalFeatures(c['data_root'],extractor,work/'probe_features',c['device'])
                    with meter.phase('pilot_review_probes_multiple_actors'):
                        summary=pilot_gates(work,rows,provider,c['device']);save_or_verify(work/'pilot.json',summary)
                    status='pilot_complete_pending_review' if summary['passed'] else 'pilot_complete_scientific_gate_failed'
                else:
                    name=c['job'];variant,seed=job(name,phase);folder=work/'runs'/name;rows=manifest['online']
                    ji=dict(job=name,mode=variant,seed=seed,suite_digest=digest(identity),method_digest=digest(method_identity(variant)),scope='fresh_fulltrain_3629')
                    provider=NormalFeatures(c['data_root'],extractor,folder/'feature_receipts',c['device']);factory=lambda:session_for(name,c['device'])
                    with meter.phase('train',name):
                        actor=train(folder,rows,manifest['task_order'],ji,provider,factory,deadline=budget.deadline)
                        del actor;gc.collect();torch.cuda.empty_cache()
                    tm=test_manifest(c['data_root'],manifest['task_order'])
                    if len(tm['rows'])!=1725:raise ValueError('Official test scope changed')
                    save_or_verify(work/'test_manifest.json',tm)
                    ep=EvaluationFeatures(c['data_root'],extractor,c['device']);ei=dict(provider=ep,test_digest=tm['digest'])
                    with meter.phase('final_evaluation',name):
                        evaluate(folder,manifest['task_order'],tm['rows'],factory,ei,deadline=budget.deadline)
                    with meter.phase('matrix_evaluation',name):
                        evaluate(folder,manifest['task_order'],tm['rows'],factory,ei,matrix=True,deadline=budget.deadline)
                    status='matrix_complete_pending_review'
                    if all((work/'runs'/j/'matrix_complete.json').exists() for j in cfg['C_full_seed0_jobs']):
                        data={j:read(work/'runs'/j/'metrics/final_macro.json') for j in cfg['C_full_seed0_jobs']}
                        h,r,b=[data[j] for j in ('H32_EXACT_seed0','R_EXACT_seed0','CADIC_seed0')]
                        gates=dict(history_pixel=h['macro_final_p_ap']-r['macro_final_p_ap']>0,history_image=h['macro_final_i_auroc']-r['macro_final_i_auroc']>=-.02,
                            cadic_image=h['macro_final_i_auroc']-b['macro_final_i_auroc']>=-.02,cadic_pixel=h['macro_final_p_ap']-b['macro_final_p_ap']>=-.02)
                        save_or_verify(work/'development_screen.json',dict(passed=all(gates.values()),gates=gates,metrics=data,scope='spend_screen_only_not_confirmatory_NI'))
        except Pause:status='paused_at_atomic_boundary'
        except BaseException as exc:failure=exc
        finally:
            if extractor is not None:
                try:extractor.close()
                except BaseException as exc:failure=failure or exc
            if failure:status='budget_unresolved' if 'budget_unresolved' in str(failure) else 'failed_preserve_evidence'
            try:resources=resource_usage(c['device'])
            except BaseException as exc:
                resources=dict(unavailable=repr(exc));failure=failure or exc;status='failed_preserve_evidence'
            atomic_json(dict(status=status,phase=phase,job=c.get('job'),reportable=False,error=repr(failure) if failure else None,
                elapsed_seconds_before_export=time.monotonic()-started,resources=resources,
                resource_scope='Invocation GPU maxima can be reset by per-phase telemetry; use telemetry for per-job peaks. Host peak is process lifetime.',
                accounting_status='reserved_until_verified_export',candidate_adopted=False),work/'handoff.json')
        if budget:budget.begin_export()
        # Hold the writer lock through export and ledger reconciliation.
        receipt=export_archive(work,work.parent/(work.name+'.zip'),schema=EVIDENCE_SCHEMA)
        accounting=budget.finish(receipt) if budget else None
    if failure:raise RuntimeError('Research run failed; evidence exported '+receipt['path']) from failure
    return dict(status=status,evidence=receipt,accounting=accounting)
