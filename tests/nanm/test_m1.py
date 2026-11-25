"""CPU tests for the M1 nanm package: import boundary, frozen contract, kernel parity vs the sealed oracle (bitwise).

    python -B -m pytest -q -p no:cacheprovider tests/nanm/test_m1.py
The oracle is imported here (test harness only); the nanm package itself must never import it.
"""
import ast
import pathlib

import torch

PKG = pathlib.Path(__file__).resolve().parents[2] / 'nanm'
ALLOWED = {'__future__', 'torch', 'numpy', 'PIL', 'math', 'hashlib', 'json', 'dataclasses', 'pathlib', 'nanm',
           'os', 're', 'uuid', 'platform', 'safetensors'}
CONTRACT_SHA = '8f7f669d77cb06edcabd565eda6bc9cdb403b63662af67150f5182945708af7e'


def test_import_boundary():
    for p in PKG.glob('*.py'):
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for n in names:
                assert n.split('.')[0] in ALLOWED, f'{p.name} imports {n}'


def test_contract_matches_oracle_constants():
    from exps.bench_g3 import run as G
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    from exps.nanm_r2b import run as B
    from exps.nar_lite_01 import run as NL
    from exps.dinomaly import encoder as E
    from nanm import G3Config
    c = G3Config()
    assert (c.dim, c.query_chunk, c.proj_dim, c.seed) == (G.D, G.QC, A.PROJ_DIM, 2711)
    assert (c.shrink_alpha, c.sigma, c.ksize, c.top_fraction) == (B.ALPHA, NL.SIGMA, NL.KSIZE, NL.TOP)
    assert (c.image_size, c.patch, c.encoder_blocks) == (E.SIZE, E.PATCH, E.GROUPS[1])
    assert c.contract_sha256() == CONTRACT_SHA


def test_kernels_bitwise_cpu():
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    from exps.nanm_r2b import run as B
    from nanm import geometry as NG, memory as NM, readout as NR, retrieval as NRT
    torch.manual_seed(0)
    x16 = A.unit(torch.randn(3000, 768)).half()
    idx = A.greedy_coreset(x16, 40, seed=2711)
    assert torch.equal(NM.greedy_kcenter(x16, 40, 2711, 128), idx)
    f32 = x16.float()
    So, no = B.residual_stats(f32, idx)
    Sn, nn_ = NG.residual_moment(f32, idx, 2048)
    assert torch.equal(So, Sn) and no == nn_
    Wo, co = B.whiten(So + So.T.contiguous() * 0 + torch.eye(768, dtype=torch.float64), no)
    Wn, cn = NG.whiten(So + So.T.contiguous() * 0 + torch.eye(768, dtype=torch.float64), no, 0.5, 1e-12)
    assert torch.equal(Wo, Wn) and co == cn
    q, m = torch.randn(64, 768), torch.randn(100, 768)
    assert torch.equal(R2.sqdist(q, m), NRT.sqdist(q, m))
    grid = torch.rand(3, 28, 28) * 500
    mo = R2.maps_of(grid)
    mn = NR.maps392(grid, 392, 4, 33)
    assert torch.equal(mo, mn)
    assert torch.equal(torch.from_numpy(A.image_scores(mo)), NR.image_score(mn, 0.01))
