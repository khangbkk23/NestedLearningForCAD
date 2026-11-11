"""Remote-only small-tensor qualification for the isolated forward-bound VJP.

Importing this file does not import Torch or execute tests. run() is invoked by
the Kaggle preflight, never by local metadata/AST checks.
"""
import argparse
import json
import math
from pathlib import Path


def _oracle(t,parameters,gradients,moments,step,maximum=.1,threshold=1.):
    """Independent autograd smooth proxy with actual-operation primal ordering."""
    norm=t.linalg.vector_norm(t.stack([t.linalg.vector_norm(g,2) for g in gradients.values()]),2)
    h={n:g*t.clamp(maximum/(norm+1e-6),max=1.) for n,g in gradients.items()}
    c1=1-.9**(step+1);c2=1-.999**(step+1);answer={}
    for n,p in parameters.items():
        g=h[n];state=moments[n]
        m=(state['exp_avg']*.9).add(g,alpha=1-.9)
        v=(state['exp_avg_sq']*.999).addcmul(g,g,value=1-.999)
        d=(t.maximum(state['max_exp_avg_sq'],v).sqrt()/math.sqrt(c2)).add(1e-10)
        r=t.linalg.vector_norm(g/d,2)/math.sqrt(g.numel())
        scaled=r/threshold
        scale=scaled if float(scaled.detach())>1. else g.new_ones(())
        answer[n]=p*(1-.002*1e-4)-(.002/c1)*m/d/scale
    return answer


def _actual(t,official,parameters,gradients,moments,step,maximum,threshold):
    values={n:t.nn.Parameter(p.detach().clone()) for n,p in parameters.items()}
    optimizer=official.StableAdamW(list(values.values()),lr=.002,betas=(.9,.999),eps=1e-10,
        weight_decay=1e-4,amsgrad=True,clip_threshold=threshold)
    for n,p in values.items():
        optimizer.state[p].update(step=step,**{k:v.detach().clone() for k,v in moments[n].items()})
        p.grad=gradients[n].detach().clone()
    t.nn.utils.clip_grad_norm_(list(values.values()),maximum)
    optimizer.step()
    return {n:p.detach().clone() for n,p in values.items()}


def run(device='cuda:0'):
    import torch as t
    from nested_replay_cad.model import decoder as official
    from nested_replay_cad.fast_update import functional_step,conditional_vjp,forward_primals
    if device.startswith('cuda') and not t.cuda.is_available():raise RuntimeError('CUDA qualification requested but unavailable')
    generator=t.Generator(device='cpu').manual_seed(31320261108)
    def noise(shape,dtype):return t.randn(shape,generator=generator,dtype=dtype).to(device)
    records=[];fd_records=[];checks=0
    def check(value,message):
        nonlocal checks
        if not value:raise ValueError(message)
        checks+=1
    def compare(parameters,gradients,moments,step,case,maximum=.1,threshold=1.,fd=False):
        before={('p',n):v.detach().clone() for n,v in parameters.items()}
        before.update({(k,n):v.detach().clone() for n,r in moments.items() for k,v in r.items()})
        q={n:noise(g.shape,g.dtype) for n,g in gradients.items()}
        with t.no_grad():
            proposal,_=functional_step(parameters,gradients,moments,step,maximum=maximum,clip_threshold=threshold)
            actual=_actual(t,official,parameters,gradients,moments,step,maximum,threshold)
        dtype=next(iter(gradients.values())).dtype
        atol,rtol=(1e-6,1e-5) if dtype==t.float32 else (1e-12,1e-12)
        max_forward=0.
        for n in parameters:
            t.testing.assert_close(proposal[n],actual[n],atol=atol,rtol=rtol)
            max_forward=max(max_forward,float((proposal[n]-actual[n]).abs().max()))
        adjoint,branches=conditional_vjp(gradients,moments,step,q,maximum=maximum,clip_threshold=threshold)
        leaves={n:g.detach().clone().requires_grad_() for n,g in gradients.items()}
        oracle=_oracle(t,parameters,leaves,moments,step,maximum,threshold)
        reference=t.autograd.grad(sum((oracle[n]*q[n]).sum() for n in oracle),tuple(leaves.values()))
        atol,rtol=(2e-8,2e-4) if dtype==t.float32 else (1e-12,1e-8)
        max_raw_vjp=0.
        for n,expected in zip(leaves,reference):
            # B272 registered the replay-logit VJP gate, not a stricter FP32
            # gate on every raw mixed-gradient coordinate. Preserve that scope.
            if dtype==t.float64:t.testing.assert_close(adjoint[n],expected.double(),atol=atol,rtol=rtol)
            max_raw_vjp=max(max_raw_vjp,float((adjoint[n]-expected.double()).abs().max()))
        logits=t.tensor([-.6,-.1,.25,.9],dtype=dtype,device=device,requires_grad=True)
        weights=.025+.9*t.softmax(logits,dim=0);fixed=weights.detach()
        support=[{n:noise(g.shape,g.dtype)*.01 for n,g in gradients.items()} for _ in range(4)]
        # Centered intervention preserves exact branch-test primals at z=z0.
        mixed={n:g+.5*sum((weights[i]-fixed[i])*support[i][n] for i in range(4)) for n,g in gradients.items()}
        oracle_policy=_oracle(t,parameters,mixed,moments,step,maximum,threshold)
        direct=t.autograd.grad(sum((oracle_policy[n]*q[n]).sum() for n in q),logits)[0]
        upstream=t.stack([.5*sum((adjoint[n]*support[i][n].double()).sum() for n in gradients) for i in range(4)])
        probabilities=t.softmax(logits.detach(),dim=0).double()
        explicit=.9*probabilities*(upstream-(probabilities*upstream).sum())
        t.testing.assert_close(explicit,direct.double(),rtol=rtol,atol=atol)
        max_vjp=float((explicit-direct.double()).abs().max())
        for (kind,n),v in before.items():
            check(t.equal(v,parameters[n] if kind=='p' else moments[n][kind]),'Parent/moment mutation '+case)
        destination={n:p.detach().clone() for n,p in parameters.items()}
        with t.no_grad():functional_step(parameters,gradients,moments,step,maximum=maximum,clip_threshold=threshold,destination=destination)
        check(all(t.equal(destination[n],proposal[n]) for n in proposal),'Destination forward parity '+case)
        if fd:
            direction={n:noise(g.shape,g.dtype)*.1 for n,g in gradients.items()}
            delta=1e-4
            with t.no_grad():
                left={n:g-delta*direction[n] for n,g in gradients.items()};right={n:g+delta*direction[n] for n,g in gradients.items()}
                pa=forward_primals(left,moments,step,maximum=maximum,clip_threshold=threshold)
                pb=forward_primals(right,moments,step,maximum=maximum,clip_threshold=threshold)
                check(pa['clip_derivative_active']==pb['clip_derivative_active'],'FD crosses clip branch '+case)
                for n in gradients:
                    check(t.equal(pa['tensors'][n]['ams_gt'],pb['tensors'][n]['ams_gt']) and t.equal(pa['tensors'][n]['ams_eq'],pb['tensors'][n]['ams_eq']) and pa['tensors'][n]['rms_scaled']==pb['tensors'][n]['rms_scaled'],'FD crosses optimizer branch '+case)
                va,_=functional_step(parameters,left,moments,step,maximum=maximum,clip_threshold=threshold)
                vb,_=functional_step(parameters,right,moments,step,maximum=maximum,clip_threshold=threshold)
                observed=sum(float(((vb[n]-va[n])*q[n]).sum()) for n in gradients)/(2*delta)
                expected=math.fsum(float((adjoint[n]*direction[n]).sum()) for n in gradients)
                check(abs(observed-expected)<=1e-9+5e-4*abs(expected),'FP64 smooth finite difference '+case)
                fd_records.append(dict(case=case,actual=observed,expected=expected,absolute_error=abs(observed-expected)))
        records.append(dict(case=case,dtype=str(dtype),forward_maxabs=max_forward,replay_logit_VJP_maxabs=max_vjp,
            raw_gradient_VJP_maxabs_diagnostic=max_raw_vjp,raw_gradient_FP32_is_diagnostic_only=dtype==t.float32,branches=branches))
        return branches
    for dtype in (t.float32,t.float64):
        for step in (0,7):
            for scale in (.001,2.):
                gradients={n:(noise(shape,dtype).abs()+.2)*scale for n,shape in (('a',(3,)),('b',(2,2)))}
                parameters={n:noise(g.shape,dtype)*.01 for n,g in gradients.items()}
                moments={}
                for n,g in gradients.items():
                    v=t.full_like(g,0. if step==0 else (1e-8 if scale==2. else .01))
                    moments[n]=dict(exp_avg=t.zeros_like(g) if step==0 else noise(g.shape,dtype)*.001,
                        exp_avg_sq=v,max_exp_avg_sq=v.clone())
                compare(parameters,gradients,moments,step,f'{dtype}_step{step}_scale{scale}',fd=dtype==t.float64 and step==7)
    # Explicit source-dtype AMSGrad tie, adjacent values, and raw clip coefficient==1.
    dtype=t.float32
    v=t.tensor([.001],dtype=dtype,device=device);g=v.sqrt()
    parameters={'x':t.tensor([.01],dtype=dtype,device=device)}
    moments={'x':dict(exp_avg=t.zeros_like(g),exp_avg_sq=v.clone(),max_exp_avg_sq=v.clone())}
    branches=compare(parameters,{'x':g},moments,7,'FP32_AMSGrad_rounding_tie')
    check(branches['tensors']['x']['amsgrad_equal']==1,'Constructed AMS tie absent')
    legacy_v=.999*v.double()+.001*g.double().square()
    check(bool((legacy_v>v.double()).all()),'Expected historical FP64 branch disagreement absent')
    for name,direction in (('below',-math.inf),('above',math.inf)):
        shifted=t.nextafter(g,t.full_like(g,direction))
        compare(parameters,{'x':shifted},moments,7,'FP32_AMS_adjacent_'+name)
    # Vary the old maximum by one ULP around a fixed computed variance. Old
    # exp_avg_sq0 <= old maximum, so these are feasible AMSGrad orderings.
    one=t.ones((1,),dtype=dtype,device=device)
    variance=(t.zeros_like(one)*.999).addcmul(one,one,value=1-.999)
    for name,direction,wanted in (('active',-math.inf,True),('inactive',math.inf,False)):
        oldmax=t.nextafter(variance,t.full_like(one,direction))
        mo={'x':dict(exp_avg=t.zeros_like(one),exp_avg_sq=t.zeros_like(one),max_exp_avg_sq=oldmax)}
        branches=compare({'x':t.zeros_like(one)},{'x':one},mo,7,'FP32_AMS_oneULP_'+name,maximum=10.)
        check(branches['tensors']['x']['amsgrad_equal']==0,'One-ULP branch should not be tied')
        primal=forward_primals({'x':one},mo,7,maximum=10.)
        check(bool(primal['tensors']['x']['ams_gt'].all())==wanted,'Wrong one-ULP AMS branch')
    g=t.tensor([.02,.03],dtype=dtype,device=device)
    parameters={'x':t.zeros_like(g)};v=t.full_like(g,.01)
    moments={'x':dict(exp_avg=t.zeros_like(g),exp_avg_sq=v,max_exp_avg_sq=v.clone())}
    maximum=float(t.linalg.vector_norm(t.stack([t.linalg.vector_norm(g,2)]),2)+1e-6)
    branches=compare(parameters,{'x':g},moments,7,'FP32_clip_exact_boundary',maximum=maximum)
    check(branches['clip_exact_boundary'] and branches['clip_derivative_active'],'Clip tie convention mismatch')
    v=t.full_like(g,1e-8);moments={'x':dict(exp_avg=t.zeros_like(g),exp_avg_sq=v,max_exp_avg_sq=v.clone())}
    with t.no_grad():threshold=float(forward_primals({'x':g},moments,7)['tensors']['x']['rms'])
    branches=compare(parameters,{'x':g},moments,7,'FP32_RMS_exact_boundary',threshold=threshold)
    check(not branches['tensors']['x']['rms_scaled'],'RMS tie should choose constant-first branch')
    zero_records=[]
    # Zero composite derivative: autograd through sqrt(0) alone is not an oracle.
    for dtype in (t.float32,t.float64):
        zero=t.zeros((2,),device=device,dtype=dtype);mom={'x':dict(exp_avg=zero.clone(),exp_avg_sq=zero.clone(),max_exp_avg_sq=zero.clone())}
        actual,_=conditional_vjp({'x':zero},mom,0,{'x':t.ones_like(zero)})
        # Independent IEEE source-dtype epsilon, not helper-derived output.
        # FP32 forward stores epsilon1e-10 as1.000000013351432e-10.
        import struct
        source_eps=struct.unpack('<f',struct.pack('<f',1e-10))[0] if dtype==t.float32 else 1e-10
        expected_zero=-.002/source_eps
        t.testing.assert_close(actual['x'],t.full_like(actual['x'],expected_zero),rtol=1e-8,atol=1e-8)
        zero_records.append(dict(dtype=str(dtype),source_epsilon=source_eps,
            expected=expected_zero,actual=actual['x'].tolist(),ideal_FP64_reference=-.002/1e-10))
        check(bool(t.isfinite(actual['x']).all()),'Zero-composite derivative nonfinite')
    # Identical replay gradients have equal upstreams; pairwise policy contrast0.
    g=t.tensor([.02,.01],dtype=t.float64,device=device);v=t.ones_like(g)*.01
    mom={'x':dict(exp_avg=t.zeros_like(g),exp_avg_sq=v,max_exp_avg_sq=v.clone())}
    adj,_=conditional_vjp({'x':g},mom,7,{'x':t.ones_like(g)})
    align=[.5*float((adj['x']*g).sum()) for _ in range(4)]
    check(all(align[i]-align[j]==0 for i in range(4) for j in range(4)),'Equal-gradient null contrast')
    check(any(r['branches']['global_clip_active'] for r in records) and any(not r['branches']['global_clip_active'] for r in records),'Both global clip branches required')
    check(any(b['rms_scaled'] for r in records for b in r['branches']['tensors'].values()),'RMS-active branch required')
    return dict(status='PASS',tests=len(records)+3,checks=checks,cases=records,finite_differences=fd_records,
        device=device,zero_composite_source_dtype_oracles=zero_records,FP32_forward_tolerance=dict(rtol=1e-5,atol=1e-6),FP64_forward_tolerance=dict(rtol=1e-12,atol=1e-12),
        FP32_VJP_tolerance=dict(rtol=2e-4,atol=2e-8),FP64_VJP_tolerance=dict(rtol=1e-8,atol=1e-12),
        source_primal_branch_binding=True,historical_branch_gap_preserved=True,new_normal_training=False,AD_executed=False,
        scientific_acceptance=False,scope='Tiny conditional derivative qualification only; no full-model/AD benefit or literal quantized derivative claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=run(args.device)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
