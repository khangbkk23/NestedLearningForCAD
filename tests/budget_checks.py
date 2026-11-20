"""B321 independent tiny composition and synthetic full-model batch checks."""
import math


def _native(t,params,context):
    hidden=t.tanh(t.nn.functional.linear(context.double(),params['hidden.weight'].double(),params['hidden.bias'].double()))
    logits=2*t.tanh(t.nn.functional.linear(hidden,params['out.weight'].double(),params['out.bias'].double()).squeeze(-1))
    return .025+.9*t.softmax(logits,dim=0)


def _shadow(t,g,m,v,old,step):
    h=g*t.clamp(.1/(t.linalg.vector_norm(g)+1e-6),max=1.)
    mean=.9*m+.1*h;variance=.999*v+.001*h.square()
    denominator=t.maximum(old,variance).sqrt()/math.sqrt(1-.999**(step+1))+1e-10
    rms=t.linalg.vector_norm(h/denominator)/math.sqrt(g.numel())
    scale=rms if float(rms.detach())>1 else g.new_ones(())
    return -(.002/(1-.9**(step+1)))*mean/denominator/scale


def composition_checks(t):
    from nested_replay_cad.core.contracts import PhiState
    from nested_replay_cad.core.policy.layout import phi_shapes
    from nested_replay_cad.core.policy.policy import ReplayPolicy
    from nested_replay_cad.fast_update import conditional_vjp
    params={n:t.linspace(-.2,.3,math.prod(s),dtype=t.float32).reshape(s).requires_grad_() for n,s in phi_shapes().items()}
    contexts=[t.tensor([[.1,.2,.3,.4],[.2,.6,.1,.9],[.3,.1,.8,.2],[.9,.3,.2,.1]],dtype=t.float32),t.tensor([[.4,.7,.2,.3],[.1,.3,.8,.9],[.6,.2,.4,.1],[.5,.9,.1,.8]],dtype=t.float32)]
    supports=t.linspace(-.3,.4,24,dtype=t.float64).reshape(8,3)
    g0=t.tensor([.015,-.025,.035],dtype=t.float64);query=t.tensor([2.,-3.,5.],dtype=t.float64)
    m=t.tensor([.002,-.001,.003],dtype=t.float64);v=t.full_like(g0,.01);old=v.clone()
    adj,_=conditional_vjp({'x':g0},{'x':{'exp_avg':m,'exp_avg_sq':v,'max_exp_avg_sq':old}},7,{'x':query})
    from budget_train import alignment_value,grouped_objective
    align=t.tensor([alignment_value([(adj['x'],row)]) for row in supports],dtype=t.float64)
    production=[ReplayPolicy().weights(PhiState(params),c) for c in contexts]
    objective=grouped_objective(t,production,align,include_kl=False)
    grads=t.autograd.grad(objective,tuple(params.values()),retain_graph=True)
    q={n:p.detach().clone().requires_grad_() for n,p in params.items()};weights=[_native(t,q,c) for c in contexts]
    centered=g0+.25*sum(((w-w.detach())[:,None]*supports[i*4:i*4+4]).sum(0) for i,w in enumerate(weights))
    direct=t.autograd.grad((_shadow(t,centered,m,v,old,7)*query).sum(),tuple(q.values()))
    maximum=0.
    for a,b in zip(grads,direct):
        t.testing.assert_close(a.double(),b.double(),atol=2e-8,rtol=2e-4);maximum=max(maximum,float((a.double()-b.double()).abs().max()))
    signal=max(float(x.abs().max()) for x in direct)
    if signal<=2e-8:raise ValueError('Tiny composition self-check is not informative')
    # Doubling the alignment coefficient must be rejected by the same gate.
    wrong=t.autograd.grad(2*objective,tuple(params.values()))
    if all(t.allclose(a.double(),b.double(),atol=2e-8,rtol=2e-4) for a,b in zip(wrong,direct)):raise ValueError('Incorrect coefficient negative control escaped')
    z={n:p.detach().clone().requires_grad_() for n,p in params.items()};zw=[_native(t,z,c) for c in contexts]
    uniform=sum((w.detach()*align[i*4:i*4+4]).sum() for i,w in enumerate(zw))+sum(p.sum()*0 for p in z.values())
    if any(t.count_nonzero(x) for x in t.autograd.grad(uniform,tuple(z.values()))):raise ValueError('Uniform null control')
    # Check actual grouped-objective assembly, including mean rather than sum KL.
    ka={n:p.detach().clone().requires_grad_() for n,p in params.items()};kw=[ReplayPolicy().weights(PhiState(ka),c) for c in contexts]
    got=t.autograd.grad(grouped_objective(t,kw,[0.]*8),tuple(ka.values()))
    kr={n:p.detach().clone().requires_grad_() for n,p in params.items()};rw=[_native(t,kr,c) for c in contexts]
    wanted=t.autograd.grad(.0005*sum((w*t.log(4*w)).sum() for w in rw),tuple(kr.values()))
    for a,b in zip(got,wanted):t.testing.assert_close(a.double(),b.double(),atol=2e-8,rtol=2e-4)
    # The uniform ER grouped coefficients are exactly 1/16 for all sixteen samples.
    coef=[.5/8]*8+[.25/4]*8
    if len(coef)!=16 or any(x!=1/16 for x in coef) or sum(coef)!=1:raise ValueError('ER batch coefficients')
    return {'status':'PASS_SCOPED','tiny_phi_maxabs':maximum,'nonzero_signal':signal,'controls':{'two_quartet_chain':True,'wrong_half_coefficient_rejected':True,'uniform_null':True,'ER_sixteen_equal_coefficients':True,'two_quartet_mean_KL':True},'scope':'Independent tiny smooth optimizer/native-policy composition; not actual episode derivative qualification'}


def batch_model_check(t,model,features,cfg):
    """Actual decoder, dropout off: microbatch4 vs per-image loss accumulation."""
    from nested_replay_cad.model import decoder as official
    model.eval();model.training=True
    before={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    cpu=t.get_rng_state().clone();cuda=t.cuda.get_rng_state(0).clone()
    model.zero_grad(set_to_none=True);x=features.to('cuda:0')
    a,b=model(x[:,0],x[:,1]);batch_loss=sum(official.global_cosine_hm_percent([v[i:i+1] for v in a],[v[i:i+1] for v in b],p=.9,factor=.1) for i in range(len(x)))/len(x)
    observed=float(batch_loss.detach());batch_loss.backward();batched={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters()}
    del a,b,batch_loss,x;model.zero_grad(set_to_none=True)
    serial=0.
    for f in features:
        x=f.unsqueeze(0).to('cuda:0');a,b=model(x[:,0],x[:,1]);loss=official.global_cosine_hm_percent(a,b,p=.9,factor=.1)/len(features)
        serial+=float(loss.detach());loss.backward();del a,b,loss,x
    error=reference=0.
    for n,v in model.named_parameters():
        g=v.grad.detach().cpu().double();delta=batched[n].double()-g
        error+=float(delta.square().sum());reference+=float(g.square().sum())
    relative=math.sqrt(error/max(reference,1e-300));limit=cfg['gradientcheck_relative_L2']
    if relative>limit:raise ValueError('Batched/serial no-dropout gradient relativeL2 '+str(relative))
    atol=cfg.get('loss_atol',1e-5);rtol=cfg.get('loss_rtol',1e-4)
    if abs(observed-serial)>atol+rtol*max(abs(observed),abs(serial)):raise ValueError('Batched/serial loss mismatch')
    if any(not t.equal(v.detach().cpu(),before[n]) for n,v in model.state_dict().items()):raise ValueError('Batch qualification mutated parameters')
    if not t.equal(cpu,t.get_rng_state()) or not t.equal(cuda,t.cuda.get_rng_state(0)):raise ValueError('Dropout-off batch qualification mutated RNG')
    model.zero_grad(set_to_none=True);model.train()
    return {'status':'PASS_SCOPED','loss_batch':observed,'loss_serial':serial,'gradient_relative_L2':relative,'relative_L2_limit':limit,'scope':'Full actual decoder, fixed synthetic four-image batch, dropout off; not optimizer-update equivalence across batch kernels or all inputs'}
