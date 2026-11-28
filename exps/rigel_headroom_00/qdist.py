import torch, math, sys, json
from rigel_cad import storage as ST, ops
torch.manual_seed(0)
def absmax4(x):
    q,s=ST.quantize4(x); return ST.dequantize(q,s)
def group4(x,g=96):
    N,D=x.shape; xs=x.reshape(N*D//g,g); return absmax4(xs).reshape(N,D)
def rot(D,seed=1234):
    g=torch.Generator().manual_seed(seed); A=torch.randn(D,D,generator=g,dtype=torch.float64)
    Q,R=torch.linalg.qr(A); Q=Q*torch.sign(torch.diag(R)); return Q.float()
out={}
for ds in sys.argv[1:]:
    st=torch.load(f'results/rigel_cad/bench/states/{ds}_E.rigel',weights_only=False)
    k=st['keys'].float(); W,_=ops.whitener(st['S'],st['n']); W=W.float()
    Q=rot(k.shape[1])
    pr = k.abs().amax(1)/k.pow(2).mean(1).sqrt()
    res={'n_keys':len(k),'peak_to_rms_median':float(pr.median()),'peak_to_rms_rot_median':float((k@Q).abs().amax(1).div((k@Q).pow(2).mean(1).sqrt()).median())}
    kw=k@W
    for name,f in [('absmax_int4',absmax4),('group96_int4',group4),('rotQ_int4',lambda x:absmax4(x@Q)@Q.T),('rotQ_group96_int4',lambda x:group4(x@Q)@Q.T),('int8',lambda x: (lambda qs:(qs[0].float()*qs[1].float()[:,None]))(( (lambda s:(torch.round(x/s.float()[:,None]).clamp(-127,127),s))((x.abs().amax(1)/127).half()) )))]:
        kh=f(k); e=(kh-k).norm(dim=1)/k.norm(dim=1); c=torch.nn.functional.cosine_similarity(kh,k,dim=1)
        ew=((kh-k)@W).norm(dim=1)/kw.norm(dim=1)
        # NN preservation among keys: for 2000 keys, nearest other key in whitened space, true vs quantised keys
        idx=torch.randperm(len(k))[:2000]; khw=kh@W
        d0=ops.sqdist(kw[idx],kw); d0[torch.arange(2000),idx]=float('inf')
        d1=ops.sqdist(kw[idx],khw); d1[torch.arange(2000),idx]=float('inf')
        res[name]={'relL2_med':float(e.median()),'relL2_p99':float(e.quantile(.99)),'cos_min':float(c.min()),'cos_med':float(c.median()),
                   'white_rel_med':float(ew.median()),'nn_changed':float((d0.argmin(1)!=d1.argmin(1)).float().mean())}
    out[ds]=res
print(json.dumps(out,indent=1))
