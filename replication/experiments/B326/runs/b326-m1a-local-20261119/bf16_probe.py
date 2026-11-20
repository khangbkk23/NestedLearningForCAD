# Verbatim copy of the probe run on 2026-11-19 that added `bf16_cast`, `weight_matrices` and
# `differing_tensors_are_weight_matrices` to base_diff_2712.json. Same temporary env as measure_base_diff.py.
# Usage: python bf16_probe.py <worktree root> <kaggle BASE.pt> <base_diff_2712.json>
import sys, types, os, json
sys.modules.setdefault('resource', types.ModuleType('resource'))
os.environ['ATEN_CPU_CAPABILITY'] = 'default'
import torch, torch.nn as nn
W = sys.argv[1]
sys.path[:0] = [W + r'\scripts', W + r'\third_party\hope_cad']
import b326_memory as M
ref = torch.load(sys.argv[2], weights_only=True, map_location='cpu')
loc = M.base_from_seed(2712, torch, nn)
n = d16 = 0; lin = [k for k, v in ref.items() if v.dim() == 2]
for k in ref:
    a, b = ref[k].float().bfloat16(), loc[k].float().bfloat16()
    n += a.numel(); d16 += int((a != b).sum())
p = sys.argv[3]
s = json.load(open(p, encoding='utf8'))
s['bf16_cast'] = {'elements_differ': d16, 'fraction_differ': d16 / n}
s['weight_matrices'] = len(lin)
s['differing_tensors_are_weight_matrices'] = sorted(r['key'] for r in s['per_tensor_differing']) == sorted(lin)
json.dump(s, open(p, 'w', encoding='utf8', newline='\n'), indent=1)
print(s['bf16_cast'], s['weight_matrices'], s['differing_tensors_are_weight_matrices'])
