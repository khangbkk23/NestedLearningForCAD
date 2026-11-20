"""B320: first two B319-CTX slow events, no evaluation or method changes.

Instrumentation observes the unchanged production step. The reference validates
optimizer/policy derivatives using actual captured query/support gradients; it
does not independently prove query loss differentiation or sampling semantics.
"""
import argparse
from datetime import datetime,timezone
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024**2):value.update(block)
    return value.hexdigest()


def save(path,value):
    staged=path.with_suffix('.partial')
    staged.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    staged.replace(path)


def access(output,operation,content_class,path,bytes_count,digest,reference=None):
    record=dict(at=datetime.now(timezone.utc).isoformat(),operation=operation,
                content_class=content_class,relative_path=path,bytes=bytes_count,
                sha256=digest,verification_ref=reference)
    with (output/'access.jsonl').open('a',encoding='utf8',newline='\n') as stream:
        stream.write(json.dumps(record,allow_nan=False)+'\n')


def fingerprint(t,values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        h.update(name.encode('utf8'));h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class EpisodeAudit:
    def __init__(self,t,decoder,optimizer,controller):
        self.t=t;self.decoder=decoder;self.optimizer=optimizer;self.controller=controller
        self.reference_alignments=[];self.expected=None;self.capture=False;self.contexts=None
        self.parameters=None;self.reference=None;self.branch_report=None
        self.actual_policy_gradient=None

    def __enter__(self):
        import nested_replay_cad.fast_update as fast
        import nested_replay_cad.core.learning.adam as slow_adam
        from episode_reference import reference_forward,reference_vjp
        self.fast=fast;self.original_functional=fast.functional_step;self.original_vjp=fast.conditional_vjp
        self.original_zero=self.optimizer.zero_grad;self.original_weights=self.controller.policy.weights
        self.slow_adam=slow_adam;self.original_adam=slow_adam.adam_propose
        t=self.t
        def functional(parent,grads,moments,step,**kwargs):
            self.expected=reference_forward(t,parent,grads,moments,step)
            answer=self.original_functional(parent,grads,moments,step,**kwargs)
            actual=kwargs.get('destination') or answer[0]
            for name in self.expected:
                t.testing.assert_close(actual[name].detach().cpu(),self.expected[name],atol=1e-6,rtol=1e-5)
            return answer
        def vjp(grads,moments,step,query,**kwargs):
            cpu=t.get_rng_state().clone();cuda=t.cuda.get_rng_state(0).clone()
            protected={f'{n}.{k}':v for n,row in moments.items() for k,v in row.items()}
            before=fingerprint(t,protected)
            self.reference,self.branch_report=reference_vjp(t,grads,moments,step,query)
            if before!=fingerprint(t,protected):raise ValueError('Reference mutated optimizer moments')
            if not t.equal(cpu,t.get_rng_state()) or not t.equal(cuda,t.cuda.get_rng_state(0)):
                raise ValueError('Reference mutated RNG')
            answer=self.original_vjp(grads,moments,step,query,**kwargs)
            self.capture=True
            return answer
        def zero(*args,**kwargs):
            if self.capture:
                grads={n:p.grad for n,p in self.decoder.named_parameters()}
                present=[g is not None for g in grads.values()]
                if any(present):
                    if not all(present):raise ValueError('Partial replay gradient population')
                    alignment=.5*math_sum(float((self.reference[n]*g.detach().double().cpu()).sum()) for n,g in grads.items())
                    self.reference_alignments.append(alignment)
                    if len(self.reference_alignments)==4:self.capture=False
            return self.original_zero(*args,**kwargs)
        def weights(phi,contexts):
            if self.contexts is None:
                self.contexts=contexts.detach().clone()
                self.parameters={n:p.detach().clone() for n,p in phi.parameters.items()}
            return self.original_weights(phi,contexts)
        def capture_adam(phi,state,gradient,*args,**kwargs):
            if self.actual_policy_gradient is not None:
                raise ValueError('Multiple slow Adam proposals in one episode')
            self.actual_policy_gradient={n:g.detach().clone() for n,g in gradient.items()}
            return self.original_adam(phi,state,gradient,*args,**kwargs)
        slow_adam.adam_propose=capture_adam
        fast.functional_step=functional;fast.conditional_vjp=vjp
        self.optimizer.zero_grad=zero;self.controller.policy.weights=weights
        return self

    def __exit__(self,*_):
        self.fast.functional_step=self.original_functional;self.fast.conditional_vjp=self.original_vjp
        self.slow_adam.adam_propose=self.original_adam
        self.optimizer.zero_grad=self.original_zero;self.controller.policy.weights=self.original_weights

    def finish(self,event):
        from episode_reference import directions,directional_report,policy_gradient
        from nested_replay_cad.core.contracts import PhiState
        t=self.t
        if len(self.reference_alignments)!=4 or self.expected is None or self.contexts is None or self.actual_policy_gradient is None:
            raise ValueError('Incomplete actual episode instrumentation')
        maximum=0
        for name,p in self.decoder.named_parameters():
            actual=p.detach().cpu();expected=self.expected.pop(name)
            t.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            maximum=max(maximum,float((actual-expected).abs().max()))
        refs=policy_gradient(t,self.parameters,self.contexts,self.reference_alignments,kl=False)
        ref_kl=policy_gradient(t,self.parameters,self.contexts,[0.]*4,kl=True)
        uniform=policy_gradient(t,self.parameters,self.contexts,self.reference_alignments,kl=False,uniform=True)
        if any(t.count_nonzero(v) for v in uniform.values()):raise ValueError('Uniform-weight null control')
        zero_query=policy_gradient(t,self.parameters,self.contexts,[0.]*4,kl=True)
        if any(not t.equal(zero_query[n],ref_kl[n]) for n in ref_kl):raise ValueError('Zeroquery KL control')
        # Query and KL components are recomputed for separate diagnostic gates.
        # The authoritative total is captured directly before production Adam.
        # Graph replay is diagnostic: FP32 accumulation can depend on query/KL
        # graph construction order, so replay is not required to be bitwise equal.
        p={n:v.detach().clone().requires_grad_() for n,v in self.parameters.items()}
        w=self.original_weights(PhiState(p),self.contexts)
        kl=(w*t.log(4*w)).sum()
        objective=(w.double()*t.tensor(event['slow']['alignments'],dtype=t.float64)).sum()
        total=objective+.001*kl
        ag=t.autograd.grad(objective,tuple(p.values()),retain_graph=True)
        kg=t.autograd.grad(.001*(w*t.log(4*w)).sum(),tuple(p.values()),retain_graph=True)
        tg=t.autograd.grad(total,tuple(p.values()))
        actual={n:v for n,v in zip(p,ag)};actual_kl={n:v for n,v in zip(p,kg)}
        if set(self.actual_policy_gradient)!=set(p):
            raise ValueError('Captured slow-gradient population mismatch')
        recomputed={n:v for n,v in zip(p,tg)}
        replay_equal=all(t.equal(recomputed[n],self.actual_policy_gradient[n]) for n in p)
        replay_maxabs=max(float((recomputed[n]-self.actual_policy_gradient[n]).abs().max()) for n in p)
        for n,v in self.actual_policy_gradient.items():
            recorded=t.tensor(event['slow']['gradient'][n],dtype=v.dtype).reshape(v.shape)
            if not t.equal(v,recorded):raise ValueError('Recorded production gradient differs from live capture '+n)
        vectors=directions(t,p)
        query_rows=directional_report(t,actual,refs,vectors)
        kl_rows=directional_report(t,actual_kl,ref_kl,vectors)
        total_rows=directional_report(t,self.actual_policy_gradient,{n:refs[n]+ref_kl[n] for n in refs},vectors)
        signal=max(abs(row['reference']) for row in query_rows)
        passed=all(r['passed'] for r in query_rows+kl_rows+total_rows)
        disposition='PASS_SCOPED' if passed and signal>2e-8 else ('INCONCLUSIVE_LOW_SIGNAL' if passed else 'FAIL_MISMATCH')
        return dict(status=disposition,query=query_rows,KL=kl_rows,total=total_rows,
                    signal_maxabs=signal,fast_forward_all_coordinates_maxabs=maximum,
                    reference_alignments=self.reference_alignments,
                    production_alignments=event['slow']['alignments'],branches=self.branch_report,
                    recomputation_diagnostic=dict(bitwise_equal=replay_equal,maxabs=replay_maxabs,
                        acceptance_gate=False,reason='Graph replay can differ by FP32 accumulation order; actual total captured before Adam'),
                    controls=dict(uniform_null=True,zeroquery_KL=True,moments_RNG_unchanged=True,
                                  actual_gradient_record_matches_live_capture=True),
                    scope='actual query gradients and replay support gradients shared with production; optimizer and policy derivative implementations independent')


def math_sum(values):
    import math
    return math.fsum(values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/b320.json')
    parser.add_argument('--self-check-only',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(experiment='B320',status='FAILED',episodes=[],new_normal_training=False,
                AD_executed=False,scientific_acceptance=False,VisA_access=False)
    try:
        import torch as t
        from episode_reference import self_check
        if args.device.startswith('cuda'):
            from nested_replay_cad.trainer import configure
            t=configure();t.cuda.init()
        result['self_checks']=self_check(t,args.device)
        if args.self_check_only:
            result['status']='SYNTHETIC_REFERENCE_PASS';return 0
        cfg=json.loads(args.config.read_text(encoding='utf8'))
        if cfg.get('seed')!=0 or cfg.get('max_arrivals',64)!=64 or cfg.get('target_slow_events',2)!=2:
            raise ValueError('B320 protocol identity')
        required={'experiment':'B320','arm':'a0_tail_ctx','directions':8,'direction_seed':31901,
                  'directional_atol':2e-8,'directional_rtol':2e-4,'fast_atol':1e-6,'fast_rtol':1e-5,
                  'minimum_query_direction_signal':2e-8,'base_recipe':'configs/b319.json'}
        if any(cfg.get(k)!=v for k,v in required.items()):raise ValueError('Changed B320 diagnostic protocol')
        expected_caps={'whole_seconds':3600,'worker_seconds':2400,'group_RSS_bytes':8*1024**3,
                       'VRAM_bytes':12*1024**3,'payload_bytes':1024**3,'cpu_seconds':7200}
        if cfg.get('caps')!=expected_caps:raise ValueError('Changed B320 caps')
        recipe=json.loads((ROOT/'configs/b319.json').read_text(encoding='utf8'))
        result.update(config_sha256=sha(args.config),source_sha256={p.relative_to(ROOT).as_posix():sha(p)
            for folder in ('src','scripts','tests') for p in sorted((ROOT/folder).rglob('*.py'))},
            runtime={'python':sys.version,'torch':t.__version__,'CUDA':t.version.cuda},
            counters={'optimizer':0,'support_forwards':0,'controller_context_forwards':0,
                      'slow_query_forwards':0,'slow_replay_forwards':0})
        if (recipe.get('method')!='B319' or recipe.get('seed')!=0
                or recipe.get('arms')!=['er_uniform','a0_tail','a0_tail_ctx']
                or recipe.get('primary')!='a0_tail_ctx'
                or recipe.get('hard_mining_final')!=.9
                or recipe.get('hard_mining_ramp_arrivals')!=26):
            raise ValueError('Changed base B319 recipe')
        manifest=ROOT/recipe['full_manifest_path']
        if sha(manifest)!=recipe['full_manifest_sha256'] or sha(args.weights)!=recipe['encoder']['sha256']:
            raise ValueError('Input/weights identity mismatch')
        weight_sha=sha(args.weights)
        weight_bytes=args.weights.stat().st_size
        access(args.output,'hash','weights',args.weights.name,weight_bytes,weight_sha)
        from nested_replay_cad.data import selection
        from nested_replay_cad.core.data.manifest import checked_root,checked_path,read_verified_bytes
        from nested_replay_cad.core.memory.rng import seed_sampling
        from nested_replay_cad.core.memory.sampler import sample_reservations,admission_slot
        from nested_replay_cad.model import decoder as official,encoder
        from nested_replay_cad.replay import NestedReplay,trim_workspace
        rows=selection(json.loads(manifest.read_text(encoding='utf8')))['records'][:64]
        dataset=checked_root(args.data)
        result.update(seed=0,manifest_sha256=sha(manifest),weights_sha256=sha(args.weights),
                      inputs='train/good prefix only; no cohort or scorer',GPU=t.cuda.get_device_name(0))
        t.manual_seed(0);service=encoder.load(args.weights,device='cuda:0')
        access(args.output,'load','weights',args.weights.name,weight_bytes,weight_sha,'verified_weight_sha256')
        t.manual_seed(0);decoder=official.Decoder(checkpoint=True);official.init_trainable(decoder);decoder.to('cuda:0')
        optimizer=official.make_optimizer(decoder);controller=NestedReplay(0,'tail',normalize_context=True)
        t.manual_seed(1000000)
        rng=seed_sampling(0);reservoir=();fifo=();store={};metas={}
        deadline=started+min(cfg.get('caps',{}).get('worker_seconds',2400),2400)
        def guard():
            if time.monotonic()>deadline:raise TimeoutError('Worker cap2400s')
            import resource
            usage=resource.getrusage(resource.RUSAGE_SELF)
            if usage.ru_utime+usage.ru_stime>7200:raise TimeoutError('Aggregate process CPU cap7200s')
            if usage.ru_maxrss*1024>8*1024**3:raise MemoryError('RAM8GiB')
            if t.cuda.max_memory_allocated(0)>12*1024**3:raise MemoryError('VRAM12GiB')
            if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>1024**3:raise OSError('Output1GiB')
        def release():
            nonlocal service
            service=None;trim_workspace()
        for arrival,row in enumerate(rows,1):
            guard()
            reserved,advanced=sample_reservations(reservoir,fifo,rng)
            if row['arrival']!=arrival or row['split']!='train/good' or '/train/good/' not in row['relative_path']:
                raise ValueError('Training identity/split')
            raw=read_verified_bytes(dataset,row['relative_path'])
            if hashlib.sha256(raw).hexdigest()!=row['file_sha256'] or len(raw)!=row['bytes']:
                raise ValueError('Training image byte identity')
            access(args.output,'read_verified','train_image',row['relative_path'],len(raw),row['file_sha256'])
            if service is None:
                random_before=t.get_rng_state().clone()
                try:
                    service=encoder.load(args.weights,device='cuda:0')
                    access(args.output,'reload','weights',args.weights.name,weight_bytes,weight_sha,'verified_weight_sha256')
                finally:t.set_rng_state(random_before)
            with t.no_grad():
                image=encoder.preprocess(checked_path(dataset,row['relative_path'])).unsqueeze(0).to('cuda:0')
                current=service.groups(image).detach()[0].contiguous()
            access(args.output,'preprocess_read','train_image',row['relative_path'],row['bytes'],row['file_sha256'],'same_path_verified_immediately_before_preprocess')
            del image
            if current.shape!=(2,789,768) or current.dtype!=t.float32 or not t.isfinite(current).all():
                raise ValueError('Grouped feature contract')
            p=recipe['hard_mining_final']*min(arrival/recipe['hard_mining_ramp_arrivals'],1.)
            due=reserved.eligible and (controller.eligible_commits+1)%16==0
            identity=row['opaque_id']
            common=(decoder,optimizer,identity,current,reserved.R_reserved,reserved.Q_recent,
                    reserved.Q_history,store,metas,arrival,reserved.eligible,p,release)
            if due:
                with EpisodeAudit(t,decoder,optimizer,controller) as audit:
                    event=controller.step(*common)
                result['counters']['optimizer']+=1
                for key,count in event['counters'].items():result['counters'][key]+=count
                result.update(processed_arrivals=arrival,new_normal_training=True)
                report=audit.finish(event)
                report.update(arrival=arrival,slow_step=controller.slow_steps,
                              support=list(reserved.R_reserved),recent=list(reserved.Q_recent),history=list(reserved.Q_history))
                result['episodes'].append(report)
                print('B320_EPISODE',json.dumps(report),flush=True)
                save(args.output/'result.json',result)
                if report['status']!='PASS_SCOPED':
                    result['status']=report['status'];return 2
            else:
                event=controller.step(*common)
                result['counters']['optimizer']+=1
                for key,count in event['counters'].items():result['counters'][key]+=count
                result.update(processed_arrivals=arrival,new_normal_training=True)
            guard()
            slot,admission_rng=admission_slot(advanced.admission,arrival)
            staged=list(reservoir)
            if slot is not None:
                if arrival<=24:staged.append(identity)
                else:staged[slot]=identity
            reservoir=tuple(staged);fifo=(*fifo,identity)[-8:]
            retained=set(reservoir)|set(fifo);store[identity]=current
            metas[identity]=dict(arrival=arrival,raw_sha256=row['file_sha256'])
            store={k:store[k] for k in retained};metas={k:metas[k] for k in retained}
            rng=replace(advanced,admission=admission_rng)
            result.update(arrivals=arrival,new_normal_training=True)
            print('B320_PREFIX',arrival,'/64','slow',controller.slow_steps,flush=True)
            if controller.slow_steps==2:break
        result['status']='PASS_SCOPED' if len(result['episodes'])==2 else 'INCONCLUSIVE_MISSING_EPISODES'
        result['scope']='2early episodes/8directions; not full97D qualification or AD gain; old B317/B319 records unchanged'
        return 0 if result['status']=='PASS_SCOPED' else 2
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:
        result['seconds']=time.monotonic()-started
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            result.update(peak_VRAM_allocated=torch.cuda.max_memory_allocated(0),peak_VRAM_reserved=torch.cuda.max_memory_reserved(0))
        if sys.platform=='linux':
            import resource
            use=resource.getrusage(resource.RUSAGE_SELF)
            result.update(cpu_seconds=use.ru_utime+use.ru_stime,peak_RSS_bytes=use.ru_maxrss*1024)
        save(args.output/'result.json',result)
        print('B320_TERMINAL',result['status'],flush=True)


if __name__=='__main__':raise SystemExit(main())
