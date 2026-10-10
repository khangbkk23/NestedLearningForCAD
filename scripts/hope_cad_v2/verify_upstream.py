"""Pinned VisionHOPE FP32 operator parity, including observed chunk gain buffers.

Loads only the unmodified low-level modules, avoiding upstream training extras.
No upstream operator is substituted into the exact higher-order W4 backend.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import types
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from models.hope_cad_v2.visual_memory import VisualMemoryState, scan_chunked
from models.hope_cad_v2.state import tensor_digest
from scripts.hope_cad_v2.artifacts import provenance, write_json


def load_upstream(root):
    pin = json.loads(Path(__file__).with_name('upstream_pin.json').read_text())
    for relative, expected in pin['source_sha256'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'upstream source differs from pinned commit: {relative}')
    if any(k.startswith('VISIONHOPE_') for k in os.environ):
        raise ValueError('unset VISIONHOPE_* environment overrides for a clean parity gate')
    for name, folder in (('visionhope', root), ('visionhope.models', root/'models'),
                         ('visionhope.models.srnl_core', root/'models/srnl_core'),
                         ('visionhope.utils', root/'utils')):
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        sys.modules[name] = module
    return importlib.import_module('visionhope.models.srnl_core.chunk'), pin


def comparison(a, b, *, gradient=False):
    a, b = a.detach().double(), b.detach().double()
    error = (a-b).abs()
    return dict(passed=bool(torch.allclose(a, b, atol=1e-5, rtol=1e-3 if gradient else 1e-4)),
                max_absolute_error=error.max().item(),
                relative_l2_error=error.norm().item()/max(b.norm().item(), 1e-30))


@torch.no_grad()
def observed_states(upstream, state, x, q, chunk):
    """Use public extension output buffers (save_g=1), without patching upstream.

    G_final_chunk contains chunk-relative gains; applying each gain in order
    exposes all five boundary/final maps. Padded tails are reported separately.
    """
    d, n = x.shape[-1], x.shape[-2]
    padded = (chunk-n % chunk) % chunk
    if padded:
        x, q = F.pad(x, (0,0,0,padded)), F.pad(q, (0,0,0,padded))
    n = x.shape[-2]
    flat, chunks = x.numel()//(n*d), n//chunk
    mats = [m.contiguous().reshape(flat, -1, d) for m in state.tensors()]
    prefix = torch.empty(flat, max(chunks-1,0), d, d, device=x.device)
    key, value = torch.empty(flat,n,d,device=x.device), torch.empty(flat,n,d,device=x.device)
    scalars = [torch.empty(flat,n,device=x.device) for _ in range(5)]
    token_gain = torch.empty(flat,n,d,d,device=x.device)
    final_gain = torch.empty(flat,chunks,d,d,device=x.device)
    extension = upstream.load_cuda(torch.float32)
    extension.srnl_forward(mats[0], mats[2], mats[1], mats[3].reshape(flat,d), mats[4].reshape(flat,d),
                           x.contiguous().reshape(flat,n,d), q.contiguous().reshape(flat,n,d),
                           prefix, key, value, *scalars, token_gain, final_gain, chunk, d, flat, 1)
    states = []
    for index in range(chunks):
        mats = [m @ final_gain[:, index] for m in mats]
        states.append(VisualMemoryState(*(m.reshape(*x.shape[:-2], -1, d) for m in mats)))
    return states, padded


def run_fixture(upstream, seed, d, chunk, n, *, out_of_range=False):
    torch.manual_seed(seed)
    shape = (2,4,1)
    x = .3*torch.randn(*shape,n,d,device='cuda')
    q = F.normalize(torch.randn_like(x), dim=-1, eps=1e-6)
    state = [(.04 if i < 3 else .01)*torch.randn(*shape,d if i < 3 else 1,d,device='cuda') for i in range(5)]
    if out_of_range:
        x.fill_(1)
        state[4].fill_(30)
    tensors = (x, q, *state)
    left = [t.clone().requires_grad_() for t in tensors]
    right = [t.clone().requires_grad_() for t in tensors]
    local = scan_chunked(left[0], left[1], VisualMemoryState(*left[2:]), chunk, capture=True)
    y = upstream.srnl_normalized_query(right[2],right[4],right[3],right[5],right[6],right[0],right[1],chunk)
    result = dict(seed=seed, head_dim=d, chunk=chunk, tokens=n,
                  fixture_sha256=tensor_digest(({str(i):t for i,t in enumerate(tensors)},)),
                  output=comparison(local.output,y), expected_divergence=out_of_range)
    if out_of_range:
        result['scope'] = 'outside upstream alpha-logit clamp; difference expected, excluded from parity acceptance'
        return result
    cotangent = torch.randn_like(y)
    a = torch.autograd.grad((local.output*cotangent).sum(), left, allow_unused=True)
    b = torch.autograd.grad((y*cotangent).sum(), right, allow_unused=True)
    result['gradients'] = {name: comparison(ga if ga is not None else torch.zeros_like(t),
                                           gb if gb is not None else torch.zeros_like(t), gradient=True)
                           for name,ga,gb,t in zip(('input','normalized_query','content','key','value','eta','alpha'),a,b,tensors)}
    states, padding = observed_states(upstream, VisualMemoryState(*state),x,q,chunk)
    # A padded upstream terminal cannot be compared with the last real-token state.
    px,pq = (F.pad(x,(0,0,0,padding)),F.pad(q,(0,0,0,padding))) if padding else (x,q)
    reference = scan_chunked(px,pq,VisualMemoryState(*state),chunk,capture=True)
    current = VisualMemoryState(*state)
    result['states'] = []
    with torch.no_grad():
        for index, info in enumerate(reference.chunks):
            current = VisualMemoryState(*(m @ info.gains[..., -1,:,:] for m in current.tensors()))
            result['states'].append({name:comparison(a,b) for name,a,b in
                                    zip(('content','key','value','eta','alpha'),current.tensors(),states[index].tensors())})
    result['padding_tokens'] = padding
    result['state_scope'] = 'padded_terminal' if padding else 'real_boundary_and_final'
    # Main controls are provably in the common unclamped region by Cauchy-Schwarz
    # and the operator's chunk-relative nonexpansion bound.
    xmax = x.norm(dim=-1).max().item()
    result['control_logit_bounds'] = dict(eta=xmax*state[3].norm(dim=-1).max().item(),
                                         biased_alpha=2.197224577+xmax*state[4].norm(dim=-1).max().item())
    if max(result['control_logit_bounds'].values()) >= 10:
        raise ValueError('main parity fixture does not guarantee clamp-inactive controls')
    result['passed'] = (result['output']['passed'] and all(g['passed'] for g in result['gradients'].values())
                        and all(g['passed'] for s in result['states'] for g in s.values()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = dict(**provenance(), status='running', reportable=False,
                  scope='low-level normalized-query FP32 forward/state/first-gradient parity; not full-backbone or Hessian parity', fixtures=[])
    try:
        upstream, pin = load_upstream(args.upstream)
        report['upstream'] = pin
        if not torch.cuda.is_available() or shutil.which('nvcc') is None or sys.platform == 'win32':
            report.update(status='pending_linux_cuda_toolchain', reason='run pinned extension on Linux CUDA with nvcc and ninja')
        else:
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            for seed in (103,104,105):
                for d in (4,16):
                    for chunk in (1,7,28):
                        n = 784 if chunk == 28 else 3*chunk
                        report['fixtures'].append(run_fixture(upstream,seed,d,chunk,n))
                        write_json(args.output,report)
            report['fixtures'].append(run_fixture(upstream,103,4,28,29))
            report['expected_divergence'] = run_fixture(upstream,103,4,7,21,out_of_range=True)
            report['status'] = 'passed' if all(f['passed'] for f in report['fixtures']) else 'failed_parity'
    except Exception as error:
        report.update(status='failed', reason=f'{type(error).__name__}: {error}')
    write_json(args.output,report)
    print(report['status'])
    return 0 if report['status']=='passed' else (2 if report['status'].startswith('pending') else 1)


if __name__ == '__main__': raise SystemExit(main())
