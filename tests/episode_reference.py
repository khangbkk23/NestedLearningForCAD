"""Independent forward-bound optimizer/policy reference for B320.

No production optimizer derivative helpers are imported. Native autograd operates
on FP64 proxies whose primals are rebound to independently computed FP32 values.
The global clipping dependency is assembled in two passes to bound memory.
"""
import math


def bind(raw, proxy):
    return raw.detach().double() + (proxy - proxy.detach())


def bound_norm(t, raw, values, divisor=1.):
    """First-order native graph with norm's *source-rounded output* in its VJP.

    Rebinding an ordinary FP64 norm's value alone would leave its backward using
    the internally recomputed FP64 norm. This local norm linearization instead
    differentiates to values/(raw*divisor**2), exactly the declared source-output
    convention. At source norm zero the declared composite derivative is zero.
    """
    if float(raw)==0:
        return bind(raw, values.sum()*0)
    proxy=values.square().sum()/(2*raw.detach().double()*divisor**2)
    return bind(raw,proxy)


def raw_row(t, g, state, c, step):
    h = g * c
    m = (state['exp_avg'] * .9).add(h, alpha=1-.9)
    v = (state['exp_avg_sq'] * .999).addcmul(h, h, value=1-.999)
    old = state['max_exp_avg_sq']
    vmax = t.maximum(old, v)
    root = vmax.sqrt()
    den = (root / math.sqrt(1-.999**(step+1))).add(1e-10)
    ratio = h / den
    rms = t.linalg.vector_norm(ratio, 2) / math.sqrt(g.numel())
    scale = rms if float(rms) > 1 else g.new_ones(())
    return dict(h=h, m=m, v=v, old=old, vmax=vmax, root=root,
                den=den, ratio=ratio, rms=rms, scale=scale)


def reference_forward(t, parent, gradients, moments, step):
    """Stream independent actual FP32 operation ordering to CPU."""
    with t.no_grad():
        norm = t.linalg.vector_norm(t.stack([t.linalg.vector_norm(g, 2) for g in gradients.values()]), 2)
        c = (.1 / (norm + 1e-6)).clamp(max=1.)
        result = {}
        for name, p in parent.items():
            r = raw_row(t, gradients[name], moments[name], c, step)
            scale = .002 / (1-.9**(step+1))
            if float(r['rms']) > 1.:scale = float(scale/r['scale'])
            # StableAdamW converts the RMS-scaled coefficient to a scalar.
            value = t.addcdiv(p * (1-.002*1e-4), r['m'], r['den'], value=-float(scale))
            result[name] = value.detach().cpu()
        return result


def reference_vjp(t, gradients, moments, step, query):
    """Independent native-autograd conditional VJP, CPU result, streamed GPU graph."""
    with t.no_grad():
        norms = t.stack([t.linalg.vector_norm(g, 2) for g in gradients.values()])
        total = t.linalg.vector_norm(norms, 2)
        plus = total + 1e-6
        raw = .1 / plus
        c = raw.clamp(max=1.)
    with t.enable_grad():
        leaves = norms.detach().double().requires_grad_()
        n = bound_norm(t,total,leaves)
        pl = bind(plus, n + 1e-6)
        rc = bind(raw, .1 / pl)
        cp = bind(c, rc.clamp(max=1.))
        dc_dnorm = t.autograd.grad(cp, leaves)[0].detach().cpu()
    direct, global_bars, rows = {}, [], []
    for name, g0 in gradients.items():
        with t.no_grad():
            r = raw_row(t, g0, moments[name], c, step)
        with t.enable_grad():
            g = g0.detach().double().requires_grad_()
            cl = c.detach().double().requires_grad_()
            h = bind(r['h'], g * cl)
            m = bind(r['m'], moments[name]['exp_avg'].double()*.9 + h*(1-.9))
            v = bind(r['v'], moments[name]['exp_avg_sq'].double()*.999 + h.square()*(1-.999))
            old = r['old'].double()
            active, tied = r['v'] > r['old'], r['v'] == r['old']
            mx = bind(r['vmax'], t.where(active, v, t.where(tied, .5*(v+old), old)))
            positive = r['vmax'] > 0
            safe = t.where(positive, r['root'].double(), t.ones_like(mx))
            root = bind(r['root'], t.where(positive,(mx-mx.detach())/(2*safe),t.zeros_like(mx)))
            den = bind(r['den'], root/math.sqrt(1-.999**(step+1)) + 1e-10)
            ratio = bind(r['ratio'], h/den)
            if float(r['rms']) > 1.:
                rms = bound_norm(t,r['rms'],ratio,math.sqrt(g.numel()))
                scale = bind(r['scale'], rms)
            else:
                scale = r['scale'].detach().double()
            delta = -(.002/(1-.9**(step+1))) * m/den/scale
            objective = (delta * query[name].double()).sum()
            dg, dc = t.autograd.grad(objective, (g, cl))
            direct[name] = dg.detach().cpu()
            global_bars.append(float(dc))
            rows.append(dict(name=name, ams_ties=int(tied.sum()),
                             root_zero=int((~positive).sum()), rms_scaled=float(r['rms']) > 1))
        del r, g, cl, h, m, v, mx, safe, root, den, ratio, scale, delta, objective, dg, dc
    bar = math.fsum(global_bars)
    for index, (name, g) in enumerate(gradients.items()):
        # Each leaf here is the source-bound per-tensor norm. Its independent
        # native derivative is g/norm, with the declared zero-norm derivative0.
        with t.enable_grad():
            x = g.detach().double().requires_grad_()
            if float(norms[index]) != 0:
                norm = bound_norm(t,norms[index],x)
                coupling = t.autograd.grad(norm, x, grad_outputs=x.new_tensor(bar*float(dc_dnorm[index])))[0]
                direct[name].add_(coupling.detach().cpu())
        if not t.isfinite(direct[name]).all():
            raise ValueError('Reference nonfinite '+name)
    return direct, dict(global_clip_active=float(raw) < 1., tensors=rows,
                        convention='independent FP32 primal binding, native FP64 local autograd and global clip coupling')


def native_policy(t, parameters, contexts):
    """Independent bound-primal native graph; no custom pairwise backward."""
    raw = {n:p.detach() for n,p in parameters.items()}
    p = {n:v.detach().double().requires_grad_() for n,v in parameters.items()}
    c = contexts.detach().double()
    with t.no_grad():
        hp = t.nn.functional.linear(contexts, raw['hidden.weight'], raw['hidden.bias'])
        hidden = t.tanh(hp)
        op = t.nn.functional.linear(hidden, raw['out.weight'], raw['out.bias']).squeeze(-1)
        logits = 2*t.tanh(op)
        prob = t.softmax(logits, dim=0)
        weights = .1/len(contexts) + .9*prob
    hidden_proxy = bind(hidden, t.tanh(bind(hp, t.nn.functional.linear(c,p['hidden.weight'],p['hidden.bias']))))
    op_proxy = bind(op, t.nn.functional.linear(hidden_proxy,p['out.weight'],p['out.bias']).squeeze(-1))
    logit_proxy = bind(logits, 2*t.tanh(op_proxy))
    probability_proxy = bind(prob, t.softmax(logit_proxy, dim=0))
    w = bind(weights, .1/len(contexts)+.9*probability_proxy)
    return p, w


def policy_gradient(t, parameters, contexts, alignments, kl=True, uniform=False):
    p, w = native_policy(t, parameters, contexts)
    query = (w*t.tensor(alignments, dtype=t.float64)).sum()
    if uniform:
        query = (w.detach()*t.tensor(alignments,dtype=t.float64)).sum() + sum(v.sum()*0 for v in p.values())
    objective = query + (.001*(w*t.log(len(contexts)*w)).sum() if kl else 0)
    result = t.autograd.grad(objective, tuple(p.values()))
    return {n:g.detach() for n,g in zip(p,result)}


def directions(t, parameters):
    gen = t.Generator().manual_seed(31901)
    count = sum(p.numel() for p in parameters.values())
    return [t.randint(0,2,(count,),generator=gen,dtype=t.int64).double().mul_(2).sub_(1)/math.sqrt(count) for _ in range(8)]


def directional_report(t, actual, expected, vectors):
    a=t.cat([actual[n].detach().double().flatten() for n in sorted(actual)])
    b=t.cat([expected[n].detach().double().flatten() for n in sorted(expected)])
    records=[]
    for i,d in enumerate(vectors):
        av,bv=float(t.dot(a,d)),float(t.dot(b,d))
        tolerance=2e-8+2e-4*max(abs(av),abs(bv))
        records.append(dict(direction=i,actual=av,reference=bv,error=abs(av-bv),tolerance=tolerance,passed=abs(av-bv)<=tolerance))
    return records


def self_check(t, device):
    """Synthetic nonzero/tie/zero branches before actual episodes."""
    from nested_replay_cad.fast_update import conditional_vjp
    rows=[]
    for kind in ('smooth','tie','zero'):
        g=t.tensor([.02,.03],device=device,dtype=t.float32)
        v=t.full_like(g,.01)
        if kind=='zero':g.zero_();v.zero_()
        if kind=='tie':g=t.tensor([.001],device=device).sqrt();v=t.full_like(g,.001)
        mo={'x':dict(exp_avg=t.zeros_like(g),exp_avg_sq=v.clone(),max_exp_avg_sq=v.clone())}
        grads={'x':g};q={'x':t.ones_like(g)}
        reference,_=reference_vjp(t,grads,mo,7,q)
        actual,_=conditional_vjp(grads,mo,7,q)
        error=float((reference['x']-actual['x'].cpu()).abs().max())
        t.testing.assert_close(reference['x'],actual['x'].cpu(),atol=2e-8,rtol=2e-4)
        rows.append(dict(case=kind,error=error))
        zero,_=reference_vjp(t,grads,mo,7,{'x':t.zeros_like(g)})
        if t.count_nonzero(zero['x']):raise ValueError('Synthetic full zero-query VJP control')
    for kind in ('global_clip_RMS_active','mixed_tensor_coupling'):
        grads={'x':t.tensor([2.,-3.,1.],dtype=t.float32,device=device)}
        if kind=='mixed_tensor_coupling':grads['y']=t.tensor([[.5,4.],[-2.,.8]],dtype=t.float32,device=device)
        mo={n:dict(exp_avg=t.zeros_like(g),exp_avg_sq=t.ones_like(g)*1e-8,
                   max_exp_avg_sq=t.ones_like(g)*1e-8) for n,g in grads.items()}
        q={n:t.arange(1,g.numel()+1,dtype=g.dtype,device=device).reshape(g.shape)*.1 for n,g in grads.items()}
        expected,branches=reference_vjp(t,grads,mo,7,q)
        actual,observed=conditional_vjp(grads,mo,7,q)
        if not branches['global_clip_active'] or not observed['global_clip_active']:
            raise ValueError('Synthetic active clip branch absent')
        if not all(row['rms_scaled'] for row in branches['tensors']):
            raise ValueError('Synthetic RMS scaled branch absent')
        for name in grads:t.testing.assert_close(expected[name],actual[name].cpu(),atol=2e-8,rtol=2e-4)
        forward=reference_forward(t,{n:t.zeros_like(g) for n,g in grads.items()},grads,mo,7)
        corrupted=forward['x']+1.
        if t.allclose(corrupted,forward['x'],atol=1e-6,rtol=1e-5):
            raise ValueError('Intentional forward corruption was not rejected')
        rows.append(dict(case=kind,global_clip_active=True,RMS_scaled=True,
                         max_error=max(float((expected[n]-actual[n].cpu()).abs().max()) for n in grads),
                         corrupted_forward_rejected=True))
    params={'hidden.weight':t.arange(64,dtype=t.float32).reshape(16,4)/100,
            'hidden.bias':t.zeros(16),'out.weight':t.linspace(-.2,.2,16).reshape(1,16),
            'out.bias':t.zeros(1)}
    contexts=t.tensor([[.1,.2,.3,.4],[.2,.6,.1,.9],[.3,.1,.8,.2],[.9,.3,.2,.1]],dtype=t.float32)
    align=[.2,-.1,.7,-.4]
    nonzero=policy_gradient(t,params,contexts,align,kl=False)
    if not any(t.count_nonzero(x) for x in nonzero.values()):raise ValueError('Synthetic policy nonzero control')
    null=policy_gradient(t,params,contexts,align,kl=False,uniform=True)
    if any(t.count_nonzero(x) for x in null.values()):raise ValueError('Synthetic uniform policy null control')
    vectors=directions(t,params)
    perturbed={n:v+1 for n,v in nonzero.items()}
    negative=directional_report(t,perturbed,nonzero,vectors)
    if all(x['passed'] for x in negative):raise ValueError('Perturbed gradient negative control did not fail')
    rows.append(dict(case='full_zero_query_all_cases',passed=True))
    rows.append(dict(case='policy_nonzero_uniform_null',passed=True))
    rows.append(dict(case='intentional_gradient_perturbation_rejected',passed=True))
    return rows
