"""One frozen causal normal-only test. Stops at the first failed safety gate."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from exps.ecad.r1.distance import configure
from exps.ecad.r1.train_normal import (ROOT,FEATURES,SOURCE_SHA,CATEGORIES,ArrivalCache,
                                      writer_plan,file_sha,tensor_sha)
from .normal_compatibility import NormalCompatibility,VERSION
from .precise_distance import PreciseEuclidean,VERSION as DISTANCE_VERSION
from .test_normal_compatibility import numerical_gate,synthetic_stream
from .test_mechanism import equal_state

MODES=('raw','legacy','boxes','shuffled')
REFERENCE=ROOT/'results/hope_cad/ecad/r1b/normal_fit_8011_20261018'
CONTEXT_REFERENCE=ROOT/'results/hope_cad/ecad/ccnm/ccnm_01_8011_20261018'


def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)


def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    return x


def checkpoint(model,path):
    with path.open('xb') as f:torch.save(cpu(model.state_dict()),f)
    return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':file_sha(path),
            'ledger':model.ledger()}


def witness_plan(source,writers):
    rows=[]
    for category in CATEGORIES:
        available=[r for r in source['entries'] if r['category']==category]
        available.sort(key=lambda r:hashlib.sha256(('ecad-r1a-split-11021:'+r['raw_png_sha256']).encode()).hexdigest())
        for role,selected in [('reserved_normal',available[24:32]),('existing_normal',available[190:206])]:
            rows.extend({**r,'witness_role':role} for r in selected)
    assert len(rows)==72 and len({r['raw_png_sha256'] for r in rows+writers})==422
    assert all(Path(r['relative_path']).parts[1:3]==('train','good') for r in rows)
    return rows


def paired_oracle(model,grid,context,outputs):
    """Direct differences, independent of norm/dot implementation and chunking."""
    b=model.base;maximum=0.
    for mode,out in outputs.items():
        rows=torch.searchsorted(b.model.serials,out['ids'])
        geometric=(grid.double()-b.model.bank[rows].double()).square().sum(1)
        if mode=='raw':penalty=torch.zeros_like(geometric)
        elif mode=='legacy':penalty=(b.contexts[rows].double()-context.double()).square().sum(1)
        else:
            lo,hi=model.lower,model.upper
            if mode=='shuffled':
                p=torch.randperm(len(lo),generator=torch.Generator().manual_seed(52011)).to(lo.device)
                lo,hi=lo[p],hi[p]
            delta=(lo[rows].double()-context.double()).clamp_min(0)+(context.double()-hi[rows].double()).clamp_min(0)
            penalty=delta.square().sum(1)
        ref=(geometric+penalty*b.alpha.double().square()).sqrt().float()
        torch.testing.assert_close(out['scores'],ref,atol=2e-6,rtol=2e-6)
        maximum=max(maximum,float((out['scores']-ref).abs().max()))
    return maximum


def evaluate(model,source,rows,checked,outdir,count,summary,resumed):
    reader=ArrivalCache(source,rows);reader.checked=checked.copy()
    records=[];arrays={mode:[] for mode in MODES};ids={mode:[] for mode in MODES}
    before=model.state_dict()
    try:
        for event,row in enumerate(rows):
            grid=reader.next(event).to(model.base.model.bank.device)
            context=model.base.context(grid)
            outputs={mode:model.score(grid,mode=mode,query_context=context) for mode in MODES}
            if event==0:
                for mode,out in outputs.items():
                    assert equal_state(out,resumed.score(grid,mode=mode,query_context=context))
                # Chunk parity on a real-normal subset; full inferred image context.
                a=model.score(grid[:19],chunk=7,query_context=context)
                b=model.score(grid[:19],chunk=256,query_context=context)
                torch.testing.assert_close(a['scores'],b['scores'],atol=2e-6,rtol=2e-6)
                assert torch.equal(a['ids'],b['ids'])
                assert equal_state(model.score(grid,weight=0),outputs['raw'])
                x,y=grid[:19],model.base.model.bank[:31]
                ref=(x[:,None].double()-y[None].double()).square().sum(2).sqrt().float()
                torch.testing.assert_close(PreciseEuclidean(7).matrix(x,y),ref,atol=2e-6,rtol=2e-6)
            error=paired_oracle(model,grid,context,outputs)
            raw,boxes,legacy=(outputs[k]['scores'] for k in ('raw','boxes','legacy'))
            assert bool((boxes+2e-6>=raw).all() and (boxes<=legacy+2e-6).all())
            selected=torch.searchsorted(model.base.model.serials,outputs['raw']['ids'])
            compatible=((model.lower[selected]<=context)&(context<=model.upper[selected])).all(1)
            record={'completed':count,'category':row['category'],'relative_path':row['relative_path'],
                    'witness_role':row['witness_role'],'direct_fp64_selected_pair_max_error':error,
                    'raw_nn_context_compatible_fraction':float(compatible.float().mean()),
                    'new_nn_changed_fraction':float((outputs['boxes']['ids']!=outputs['raw']['ids']).float().mean()),
                    'scores':{mode:{'mean':float(out['scores'].mean()),'q99':float(torch.quantile(out['scores'],.99)),
                                    'max':float(out['scores'].max())} for mode,out in outputs.items()}}
            record['new_to_raw_max_ratio']=float(boxes.max()/raw.max())
            records.append(record);summary['normal_image_evaluations']+=1
            for mode,out in outputs.items():
                assert bool(torch.isfinite(out['scores']).all())
                arrays[mode].append(out['scores'].cpu().numpy());ids[mode].append(out['ids'].cpu().numpy())
            if record['new_to_raw_max_ratio']>1.05:
                summary['normal_guard_failure']=record
                raise RuntimeError('NORMAL_MAX_GUARD: new image max exceeds 1.05 * precise raw max')
    finally:
        assert equal_state(before,model.state_dict()),'score-time mutation'
        category=rows[0]['category']
        write(outdir/f'normal_{count:03}_{category}.json',records)
        np.savez_compressed(outdir/f'normal_{count:03}_{category}.npz',
            **{f'{m}_scores':np.array(arrays[m]) for m in MODES},
            **{f'{m}_ids':np.array(ids[m]) for m in MODES})
        summary['normal_groups'].append({'completed':count,'category':category,'images':len(records),
            'max_new_to_raw_max_ratio':max((r['new_to_raw_max_ratio'] for r in records),default=0),
            'mean_raw_nn_compatible_fraction':float(np.mean([r['raw_nn_context_compatible_fraction'] for r in records])) if records else None})


def boundary(model,count,args,source,writers,witnesses,reader,summary):
    ref=torch.load(REFERENCE/f'contcore_{count:03}.pt',weights_only=True,map_location=args.device)
    assert torch.equal(model.base.model.bank,ref['bank']) and torch.equal(model.base.model.serials,ref['serials'])
    original=torch.load(CONTEXT_REFERENCE/f'ccnm_{count:03}.pt',weights_only=True,map_location=args.device)
    assert torch.equal(model.base.contexts,original['contexts']) and torch.equal(model.base.alpha,original['alpha'])
    path=args.out/f'compatibility_{count:03}.pt';saved=checkpoint(model,path)
    saved['allocator_and_source_anchor_parity']='EXACT'
    saved['mean_box_width']=float((model.upper-model.lower).mean())
    saved['expanded_profiles']=int(((model.upper-model.lower)>0).any(1).sum())
    saved['retained_by_category']={c:sum(writers[int(s)//784]['category']==c for s in model.base.model.serials.cpu().tolist()) for c in CATEGORIES}
    summary['checkpoints'].append(saved)
    resumed=NormalCompatibility.from_state_dict(torch.load(path,weights_only=True,map_location='cpu'),device=args.device)
    assert equal_state(model.state_dict(),resumed.state_dict())
    eligible={r['category'] for r in writers[:count]}
    for category in CATEGORIES:
        if category in eligible:
            evaluate(model,source,[r for r in witnesses if r['category']==category],reader.checked,args.out,count,summary,resumed)
    return resumed


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume-from',type=Path);args=p.parse_args()
    args.out=args.out.resolve()
    configure();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();model=None
    summary={'status':'IN_PROGRESS','version':VERSION,'normal_writers_completed':0,'normal_image_evaluations':0,
             'normal_groups':[],'checkpoints':[],'normal_guard_failure':None,
             'anomaly_scoring':False,'official_test_access':False,'cost_profile_performed':False}
    def timeout(*_):raise TimeoutError('900-second worker safety cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:
        gates={'numerical':numerical_gate(args.device),'stream_invariants':synthetic_stream(args.device)}
        write(args.out/'gates.json',gates)
        source=json.loads((FEATURES/'source_manifest.json').read_text())
        digest=hashlib.sha256(json.dumps({k:v for k,v in source.items() if k!='manifest_digest'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert digest==source['manifest_digest']==SOURCE_SHA
        writers,_=writer_plan(source);witnesses=witness_plan(source,writers)
        prior=json.loads((CONTEXT_REFERENCE/'contract.json').read_text());assert writers==prior['updates']
        for paths in (prior['core_pins']['files'],prior['source_hashes']):
            for path,sha in paths.items():assert file_sha(ROOT/path)==sha,(path,'frozen source changed')
        code={str(path.relative_to(ROOT)):file_sha(path) for path in Path(__file__).parent.glob('*.py')}
        write(args.out/'contract.json',{'version':VERSION,'distance_version':DISTANCE_VERSION,
            'source_sha':SOURCE_SHA,'writers':writers,'witnesses':witnesses,'code_sha256':code,
            'normal_max_guard_ratio':1.05,'compatibility_radius':'half nearest-other old-bank distance',
            'shuffle_seed':52011,'frozen_alpha_rule':prior['calibration'],'torch':torch.__version__,
            'device':torch.cuda.get_device_name(0) if args.device=='cuda' else args.device,
            'historical_allocator':'unchanged pinned FP32 kernel; numerical repair applies to all new readouts and compatibility geometry',
            'access':'train/good arriving writers and eligible disjoint normal witnesses only',
            'reproduce':f'python -m exps.ecad.ccnm.run_normal_compatibility --out NEW_OUTPUT_DIRECTORY --device {args.device}'})
        reader=ArrivalCache(source,writers);model=NormalCompatibility(device=args.device)
        start_event=0
        if args.resume_from:
            parent=args.resume_from.resolve().parent
            old_contract=json.loads((parent/'contract.json').read_text())
            old_summary=json.loads((parent/'summary.json').read_text())
            assert old_contract['writers']==writers and old_contract['witnesses']==witnesses
            assert old_summary['normal_guard_failure'] is None and old_summary['normal_image_evaluations']==0
            for name,sha in old_contract['code_sha256'].items():
                if name!=str(Path(__file__).relative_to(ROOT)):
                    assert file_sha(ROOT/name)==sha,('mechanism changed on resume',name)
            model=NormalCompatibility.from_state_dict(torch.load(args.resume_from,weights_only=True,map_location='cpu'),device=args.device)
            start_event=model.base.model.next_event
            assert start_event==old_summary['normal_writers_completed']==100
            old_updates=[json.loads(line) for line in (parent/'updates.jsonl').read_text().splitlines()]
            assert [r['completed'] for r in old_updates]==list(range(1,start_event+1))
            reader.next_event=start_event # restore the committed input cursor; no old grid is delivered
            summary['normal_writers_completed']=start_event
            summary['resume']={'checkpoint':str(args.resume_from),'sha256':file_sha(args.resume_from),
                'prior_summary_sha256':file_sha(parent/'summary.json'),'prior_execution_seconds':old_summary['elapsed_seconds'],
                'reason':'metadata relative-path bug only; mechanism unchanged; no writer replay'}
            signal.alarm(max(1,int(900-old_summary['elapsed_seconds'])))
            model=boundary(model,start_event,args,source,writers,witnesses,reader,summary)
        with (args.out/'updates.jsonl').open('x') as log:
            for event in range(start_event,350):
                grid=reader.next(event).to(args.device)
                stats=model.update(grid,event=event);count=event+1
                summary['normal_writers_completed']=count
                log.write(json.dumps({'completed':count,**{k:int(v) for k,v in stats.items()}})+'\n');log.flush()
                if count%25==0:print(json.dumps({'writers':count,'elapsed':time.monotonic()-started}),flush=True)
                if count not in (100,200,300,350):continue
                model=boundary(model,count,args,source,writers,witnesses,reader,summary)
        summary.update(status='DONE',verdict='CCNM_READY_FOR_INDEPENDENT_ANOMALY_TEST',
            checkpoint_resume='PASS',score_read_only='PASS',allocator_parity='EXACT',
            inference_category_labels_used=False,efficacy_claim=False)
    except Exception as exc:
        summary.update(status='STOPPED',error=str(exc),traceback=traceback.format_exc(),
            verdict='CCNM_NO_MECHANISTIC_ADVANTAGE' if summary['normal_guard_failure'] else 'CCNM_INCONCLUSIVE')
        if model is not None and not (args.out/f'compatibility_{model.base.model.next_event:03}.pt').exists():
            summary['partial_checkpoint']=checkpoint(model,args.out/'partial_checkpoint.pt')
    finally:
        signal.alarm(0);summary['elapsed_seconds']=time.monotonic()-started
        if model is not None:
            summary['final_ledger']=model.ledger()
            summary['final_bank_sha256']=tensor_sha(model.base.model.bank)
        write(args.out/'summary.json',summary)
        print(json.dumps({k:summary.get(k) for k in ('status','verdict','normal_writers_completed','normal_image_evaluations','error','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
