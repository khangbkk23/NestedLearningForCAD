"""B326 M1a follow-up: element-wise gap between the Kaggle CPU BASE (seed 2712) and the same seed generated here.

Run through win_cpu_base.py's shim is not needed: this script imports only torch and the driver's base_from_seed.
    python measure_base_diff.py <kaggle BASE.pt> <driver dir> <hope_cad dir> <out json>
"""
import json
import os
import sys
import types

sys.modules.setdefault('resource', types.ModuleType('resource'))     # Unix-only import in Khang's full.py
os.environ['ATEN_CPU_CAPABILITY'] = 'default'
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

kaggle_pt, driver_dir, hope_dir, out = sys.argv[1:5]
sys.path[:0] = [driver_dir, hope_dir]
import b326_memory as M  # noqa: E402

ref = torch.load(kaggle_pt, weights_only=True, map_location='cpu')
loc = M.base_from_seed(2712, torch, nn)
assert list(ref) == list(loc), 'key order differs'
rows, tot_n, tot_diff, max_abs, max_ulp, max_rel = [], 0, 0, 0.0, 0, 0.0
ulp_hist = {}
for k in ref:
    a, b = ref[k].float(), loc[k].float()
    assert a.shape == b.shape, k
    d = (a - b).abs()
    neq = int((a != b).sum())
    ia, ib = a.view(torch.int32).long(), b.view(torch.int32).long()
    ulp = (ia - ib).abs()
    if neq:
        vals, cnt = torch.unique(ulp[ulp > 0], return_counts=True)
        for v, c in zip(vals.tolist(), cnt.tolist()):
            ulp_hist[v] = ulp_hist.get(v, 0) + c
    nz = a.abs() > 0
    rel = float((d[nz] / a.abs()[nz]).max()) if nz.any() else 0.0
    rows.append({'key': k, 'numel': a.numel(), 'n_differ': neq, 'max_abs': float(d.max()), 'max_ulp': int(ulp.max()),
                 'max_rel': rel, 'scale_abs_max': float(a.abs().max())})
    tot_n += a.numel(); tot_diff += neq
    max_abs, max_ulp, max_rel = max(max_abs, float(d.max())), max(max_ulp, int(ulp.max())), max(max_rel, rel)
summary = {'seed': 2712, 'kaggle_canonical_sha256': M.canonical_sha(ref), 'local_canonical_sha256': M.canonical_sha(loc),
           'torch_local': torch.__version__, 'cpu_capability_local': torch.backends.cpu.get_cpu_capability(),
           'tensors': len(rows), 'tensors_with_any_diff': sum(r['n_differ'] > 0 for r in rows),
           'elements': tot_n, 'elements_differ': tot_diff, 'fraction_differ': tot_diff / tot_n,
           'max_abs_diff': max_abs, 'max_ulp_diff': max_ulp, 'max_rel_diff': max_rel,
           'ulp_histogram': {str(k): v for k, v in sorted(ulp_hist.items())},
           'per_tensor_differing': [r for r in rows if r['n_differ']]}
with open(out, 'w', encoding='utf8', newline='\n') as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'per_tensor_differing'}, indent=1))
print('differing tensors (first 8):', [(r['key'], r['n_differ'], r['max_ulp']) for r in summary['per_tensor_differing'][:8]])
