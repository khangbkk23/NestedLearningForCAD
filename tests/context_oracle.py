"""REMOTE-only CTX-RN-01 qualification; no AD inputs or model training.

Run inside the already qualified source/runtime. Importing this file is stdlib-only.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def qualify():
    import torch as t
    from nested_replay_cad.context import normalize_context,CONTEXT_MODE
    from nested_replay_cad.core.contracts import PhiState
    from nested_replay_cad.core.policy.layout import phi_shapes
    from nested_replay_cad.core.policy.numerical import policy_weights_first_vjp
    if t.__version__!='2.12.0+cu126':raise ValueError('Qualified Torch version required')
    checks=[];max_gradient_error=0.;max_fd_error=0.
    def ck(value,name):
        if not value:raise ValueError(name)
        checks.append(name)
    def params(head_zero=False):
        result={}
        for name,shape in sorted(phi_shapes().items()):
            n=1
            for s in shape:n*=s
            x=t.linspace(-.2,.3,n,dtype=t.float32).reshape(shape).clone()
            if head_zero and name.startswith('out.'):x.zero_()
            result[name]=x.contiguous().requires_grad_()
        return result
    def oracle(p,c):
        h=t.tanh(t.nn.functional.linear(c,p['hidden.weight'],p['hidden.bias']))
        logits=2*t.tanh(t.nn.functional.linear(h,p['out.weight'],p['out.bias']).squeeze(-1))
        return .1/len(c)+.9*t.softmax(logits,dim=0)
    base=t.tensor([[.08,.12,.1,.9],[.081,.123,.2,.8],[.077,.11,.8,.7],[.085,.13,.5,.85]],dtype=t.float32)
    rng=t.get_rng_state().clone()
    for m in range(5):
        raw=base[:m].clone().contiguous();before=raw.clone();c=normalize_context(raw)
        ck(t.equal(raw,before),'input unchanged m'+str(m))
        ck(c.shape==(m,4) and c.dtype==t.float32 and c.device.type=='cpu' and c.is_contiguous() and not c.requires_grad and c.grad_fn is None,'output contract m'+str(m))
        ck(t.equal(c[:,2:],raw[:,2:]),'preserve age/similarity m'+str(m))
        ck(bool(((c[:,:2]>=0)&(c[:,:2]<=1)).all()),'bounded residuals m'+str(m))
        if m:
            raw64=raw.double();center=raw64[:,:2]-raw64[:,:2].mean(0,keepdim=True)
            ref=.5+.25*(center/center.square().mean(0,keepdim=True).sqrt().clamp_min(.001)).clamp(-2,2)
            t.testing.assert_close(c[:,:2].double(),ref,atol=1e-6,rtol=1e-5)
            ck(True,'FP64 normalization formula m'+str(m))
        if m==1:ck(t.equal(c[0,:2],t.full((2,),.5)),'single replay residuals0.5')
        for head_zero in (False,True):
            p=params(head_zero);values=tuple(p.values());v=t.linspace(-.001,.002,m)
            weights=policy_weights_first_vjp(PhiState(p),c)
            if not m:
                ck(weights.numel()==0,'empty policy output');continue
            expected_forward=oracle(p,c)
            ck(t.equal(weights.detach(),expected_forward.detach()),'FP32 forward oracle m'+str(m)+' zero'+str(head_zero))
            gs=t.autograd.grad((weights*v).sum(),values)
            p64={name:x.detach().double().requires_grad_() for name,x in p.items()}
            ref=t.autograd.grad((oracle(p64,c.double())*v.double()).sum(),tuple(p64.values()))
            flat=t.cat([x.reshape(-1) for x in gs]).double();reference=t.cat([x.reshape(-1) for x in ref])
            err=float((flat-reference).abs().max());max_gradient_error=max(max_gradient_error,err)
            ck(bool(((flat-reference).abs()<=1e-8+1e-3*reference.abs()).all()),'FP64 first VJP component tolerance m'+str(m)+' zero'+str(head_zero))
            if float(reference.norm())>1e-10:ck(float((flat-reference).norm()/reference.norm())<=.01,'first VJP relativeL2 m'+str(m)+' zero'+str(head_zero))
            if m==1:ck(float(flat.abs().max())<=1e-12,'single replay null gradient')
            if m==3 and not head_zero:
                for name,index in (('out.weight',3),('hidden.weight',9),('hidden.bias',2)):
                    q={n:x.detach().double().clone() for n,x in p.items()};h=1e-5
                    q[name].view(-1)[index]+=h;plus=(oracle(q,c.double())*v.double()).sum()
                    q[name].view(-1)[index]-=2*h;minus=(oracle(q,c.double())*v.double()).sum()
                    actual=gs[tuple(p).index(name)].view(-1)[index].double()
                    error=float((actual-(plus-minus)/(2*h)).abs());max_fd_error=max(max_fd_error,error)
                    ck(error<=1e-8,'independent FP64 finite difference '+name)
    permutation=t.tensor([2,0,3,1]);normal=normalize_context(base)
    permuted=normalize_context(base[permutation].contiguous())
    t.testing.assert_close(permuted,normal[permutation],atol=1e-6,rtol=1e-5);ck(True,'row permutation equivariance')
    p=params();w=policy_weights_first_vjp(PhiState(p),normal);wp=policy_weights_first_vjp(PhiState(p),permuted)
    t.testing.assert_close(wp,w[permutation],atol=1e-6,rtol=1e-5);ck(True,'policy weight permutation equivariance')
    for m in (1,2,3,4):
        equal=normalize_context(t.full((m,4),.5));weights=policy_weights_first_vjp(PhiState(params()),equal)
        t.testing.assert_close(weights,t.full((m,),1/m),atol=1e-7,rtol=1e-6);ck(True,'identical contexts uniform m'+str(m))
        g=t.autograd.grad((policy_weights_first_vjp(PhiState(p),equal)*t.arange(m,dtype=t.float32)).sum(),tuple(p.values()))
        ck(all(float(x.abs().max())<=1e-12 for x in g),'identical contexts null choice gradient m'+str(m))
    extreme=t.tensor([[0.,1.,.2,.7],[1.,0.,.8,.9]],dtype=t.float32)
    normalized=normalize_context(extreme)
    t.testing.assert_close(normalized[:,:2],t.tensor([[.25,.75],[.75,.25]]),atol=1e-6,rtol=1e-5);ck(True,'extreme bounded contrasts')
    tiny=base.clone();tiny[:,:2]=.1;tiny[0,0]+=.000001
    ck(bool(t.isfinite(normalize_context(tiny)).all()),'sigma floor finite tiny contrast')
    invalid=[base.double(),base.clone().requires_grad_(),t.full((4,4),float('nan')),t.full((4,4),1.1),t.zeros((5,4)),t.zeros((2,3)),t.zeros((4,8))[:,::2]]
    for i,x in enumerate(invalid):
        try:normalize_context(x)
        except ValueError:ck(True,'invalid input rejection '+str(i))
        else:raise ValueError('invalid input accepted '+str(i))
    ck(t.equal(rng,t.get_rng_state()),'global CPU RNG unchanged')
    return dict(status='PASS',context_mode=CONTEXT_MODE,checks=len(checks),details=checks,
        max_FP64_gradient_abs_error=max_gradient_error,max_FP64_finite_difference_error=max_fd_error,
        thresholds=dict(VJP_atol=1e-8,VJP_rtol=1e-3,VJP_relativeL2=.01,FD_abs=1e-8,FD_step=1e-5,
                        context_atol=1e-6,context_rtol=1e-5,null_abs=1e-12),
        no_AD_inputs=True,no_model_training=True,higher_order_qualified=False,scientific_acceptance=False)


def run():
    """Aggregator API: return JSON-serializable qualification or raise."""
    return qualify()

