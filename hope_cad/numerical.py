"""Canonical synthetic meta gate on the requested device, no optimizer/run."""
import torch
from models.hope_cad_v5_1.learner import UtilityModel,meta_loss
from models.hope_cad_v2.state import finite_tensors


def canonical_meta_gate(device='cpu'):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(45)
        model=UtilityModel().to(device).eval()
    gen=torch.Generator().manual_seed(88)
    records=[dict(tokens=torch.randn(1,784,32,generator=gen).to(device),
                  stats=torch.rand(16,generator=gen).to(device),c=3.,gamma=.02,target=(-1)**i*.3)
             for i in range(8)]
    initial=model.initial_fast();params=[p for _,p in model.slow_parameters()]
    exact=meta_loss(model,initial,records,exact=True)
    ge=torch.autograd.grad(exact,params)
    first=meta_loss(model,initial,records,exact=False)
    gf=torch.autograd.grad(first,params)
    finite_tensors((exact,first,*ge,*gf),'canonical exact/FO meta')
    if not torch.equal(exact.detach(),first.detach()):raise RuntimeError('exact/FO canonical forward differs')
    if not all(p.dtype==torch.float32 for p in model.parameters()) or exact.dtype!=torch.float64:
        raise RuntimeError('canonical parameter/scalar dtypes changed')
    if not any(bool(g.abs().max()>0) for g in ge):raise RuntimeError('canonical slow path inactive')
    return dict(status='passed',device=str(device),grid=[28,28],channels=32,head_dim=8,
                support=4,query=4,parameter_dtype='float32',scalar_dtype='float64',
                exact_loss=float(exact.detach()),first_order_loss=float(first.detach()),
                exact_fo_forward_equal=True,all_active_gradients_finite=True,
                scope='synthetic numerical gate only; discarded, no dataset/optimizer step or performance claim')
