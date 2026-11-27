"""Bitwise equivalence of RIGEL-CAD with the frozen research code that produced the paper numbers.

Needs CUDA, the repository's exps/ tree and the locked caches on K: (skipped otherwise). Checked on MPDD order a,
seed 2711 (the T1 cell of G3E-PAPER-01 and the NL03-MULTIRATE-01 benchmark cell):
  1. encoder tokens == cached canonical test tokens;
  2. RIGEL-CAD @ 8 MiB, unbounded, fast-only (F) and FIFO grids == locked grids at every stage;
  3. metrics on the locked grids == the locked METRICS json (per stage) and bootstrap replicates;
  4. task orders == the registered orders;
  5. storage='int4_triangular' == the research Q4T implementation (key state, whitener and grids at every stage).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
G3E = Path('/mnt/k/NLCAD/cache/g3e_paper_01/cells')
MRB = Path('/mnt/k/NLCAD/cache/nl03_multirate_01/bench')
METRICS = ROOT / 'results/hope_cad/nl03_multirate_01/bench/METRICS_mpdd_a2711.json'
need = pytest.mark.skipif(not (torch.cuda.is_available() and G3E.exists() and METRICS.exists()),
                          reason='needs CUDA and the locked research caches')


def _ctx():
    from exps.nl03_multirate_01 import final as FN
    from exps.nl03_multirate_01 import run as RN
    from exps.nl03_pressure_00 import core as PC
    G = PC.bind()
    PC.use(G, 'mpdd')
    return FN, RN, PC, G


@need
def test_orders_match_registered():
    from exps.nl03_multirate_01 import final as FN
    from rigel_cad import data
    for ds in ('mvtec', 'visa', 'mpdd'):
        for ok in 'abc':
            assert [f'{ds}/{c}' for c in data.stream_orders(ds)[ok]] == list(FN.bench_order(ds, ok)), (ds, ok)


@pytest.fixture
def restore_torch_globals():
    """The research encoder loader (exps.nar_pilot_00.full.configure) switches the whole process to deterministic
    algorithms, 2 CPU threads and a 5 GiB memory cap. The locked grids were computed without those settings, so they
    must not leak into later tests (test_stream_grids_bitwise fails after this test otherwise)."""
    det, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    thr, tf32m, tf32c = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.use_deterministic_algorithms(det, warn_only=warn)
    torch.set_num_threads(thr)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32m, tf32c
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()


@need
def test_encoder_tokens_bitwise(restore_torch_globals):
    """RIGEL-CAD encoder == the research encoder code run on the same machine (bitwise). The locked token caches were
    written in an earlier software/hardware environment; against them the difference is at most one fp16 ulp on a
    tiny fraction of entries (measured 2026-11-26: max 6.1e-5 on 0.0045 % of entries)."""
    FN, RN, PC, G = _ctx()
    from exps.bench_g3 import run as G0
    from exps.dinomaly import encoder as E
    from exps.nanm_r0 import audit as A
    from exps.nar_b14_01 import run as b14
    from rigel_cad.encoders import RigelDinoEncoder
    ex = PC.extract_rec(G, 'mpdd')
    q, rows = G.load_test('mpdd', 'bracket_black', ex)
    paths = [Path(G.data_root('mpdd')) / r for r in rows[:16]]
    enc = G0.encoder()
    with torch.no_grad():
        ref = torch.cat([A.unit(b14.tokens(enc, torch.stack([E.preprocess(p) for p in paths[i:i + 8]]).cuda())
                                .reshape(-1, 768)).half().cpu() for i in range(0, 16, 8)])
    del enc
    mine = RigelDinoEncoder().encode_paths(paths, batch=8).reshape(-1, 768)
    assert torch.equal(mine, ref)
    d = (mine.float() - q[:16 * 784].float()).abs()
    assert float(d.max()) <= 1.23e-4 and float((d > 0).float().mean()) < 1e-3


@need
def test_stream_grids_bitwise():
    FN, RN, PC, G = _ctx()
    from rigel_cad import RigelCAD, RigelConfig
    order = FN.bench_order('mpdd', 'a')
    arms = {'E@8': RigelConfig(budget_mib=8), 'U': RigelConfig(budget_mib=None),
            'F': RigelConfig(budget_mib=8, slow_level=False), 'FIFO': RigelConfig(budget_mib=8, selection='fifo')}
    cell = {'E@8': 'T1_dinov2_mpdd_a2711', 'U': 'T1_dinov2_mpdd_a2711', 'F': 'A4_dinov2_mpdd_a2711',
            'FIFO': 'A4_dinov2_mpdd_a2711'}
    models = {a: RigelCAD(c, keep_history=True) for a, c in arms.items()}
    for t in order:
        x16, _ = RN.fit_tokens(G, t, order=FN.train_rows_sorted(G, t))
        for m in models.values():
            m.fit_task(x16, t)
        del x16
    ex = PC.extract_rec(G, 'mpdd')
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = G.load_test('mpdd', c, ex)
        for a, m in models.items():
            ref = torch.load(G3E / cell[a] / a / f'{c}.pt', weights_only=True)
            assert ref['rows'] == rows and ref['stages'] == list(range(j, len(order)))
            mine = torch.stack([m.patch_energy_grid(q, m.state_at(l)) for l in ref['stages']])
            assert torch.equal(mine, ref['grid']), (a, c)


CC01 = Path('/mnt/k/NLCAD/cache/rigel_cc01')
need_q4 = pytest.mark.skipif(not (torch.cuda.is_available() and (CC01 / 'grids' / 'mpdd_a2711' / 'Q4T').exists()
                                  and METRICS.exists()), reason='needs CUDA and the RIGEL-CC-01 Q4T caches')


@need_q4
def test_int4_triangular_bitwise_research():
    """storage='int4_triangular' == the research Q4T implementation (exps/rigel_q8/d1.py::ArmQ4 via
    exps/rigel_cc01/cc.Stream, fed the cached v1 fits): int4 codes, scales, fp32 triangle, count and whitener after
    every task, every stage's grid (cached RIGEL-CC-01 grids) and the persisted state, MPDD order a seed 2711."""
    FN, RN, PC, G = _ctx()
    from exps.rigel_cc01 import cc
    from exps.rigel_gg01 import gg
    from rigel_cad import RigelCAD, RigelConfig
    order = FN.bench_order('mpdd', 'a')
    m = RigelCAD(RigelConfig(budget_mib=8, storage='int4_triangular'), keep_history=True)
    ref = cc.Stream(['Q4T'], 2711)
    arm = ref.arms['Q4T']
    for l, t in enumerate(order):
        x16, _ = RN.fit_tokens(G, t, order=FN.train_rows_sorted(G, t))
        m.fit_task(x16, t)
        del x16
        d = torch.load(CC01 / 'fit' / 'mpdd' / f'{t.split("/")[1]}_s2711.pt', weights_only=True)
        ref.task(l, gg.Fit(kc=d['kc'], S=d['S'], n=d['n'], n_tokens=d['n_tokens'], keys16=d['keys16']))
        assert torch.equal(m.fast.q, arm.q) and torch.equal(m.fast.scale, arm.scale), t
        assert torch.equal(m.slow.S_tri.cpu(), arm.S_tri.cpu()) and m.slow.n == ref.n, t
        assert torch.equal(m.history[l].W, ref.W_tri['Q4T'][l]), t
    ex = PC.extract_rec(G, 'mpdd')
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = G.load_test('mpdd', c, ex)
        r = torch.load(CC01 / 'grids' / 'mpdd_a2711' / 'Q4T' / f'{c}.pt', weights_only=True)
        assert list(r['rows']) == list(rows) and r['stages'] == list(range(j, len(order)))
        mine = torch.stack([m.patch_energy_grid(q, m.state_at(l)) for l in r['stages']])
        assert torch.equal(mine, r['grid']), c
    st = torch.load(CC01 / 'states' / 'mpdd_a2711' / 'Q4T.pt', weights_only=True)
    q4, scale = m.fast.packed()
    assert torch.equal(q4, st['q4']) and torch.equal(scale, st['scale']) and torch.equal(m.slow.S_tri, st['S_tri'])


@need
def test_metrics_bitwise():
    FN, RN, PC, G = _ctx()
    from rigel_cad import data, metrics
    me = json.loads(METRICS.read_text())
    boot = np.load(me['boot_file'])
    order = FN.bench_order('mpdd', 'a')
    for j, t in enumerate(order):
        c = t.split('/')[1]
        _, test = data.category_split('mpdd', G.data_root('mpdd'), c)
        cat = metrics.Category(G.data_root('mpdd'), test)
        for arm in ('U', 'E'):
            d = torch.load(MRB / 'mpdd_a2711' / arm / f'{c}.pt', weights_only=True)
            assert d['rows'] == cat.rows
            res, b = metrics.evaluate_category(cat, d['grid'], d['stages'], metrics.boot_weights(len(cat.y), j))
            for l, v in res.items():
                assert v == me['cells'][t][arm][str(l)], (t, arm, l)
            assert np.array_equal(b['ap'], boot[f'{t}||{arm}||ap'], equal_nan=True)
            assert np.array_equal(b['auroc'], boot[f'{t}||{arm}||auroc'], equal_nan=True)


AD2C = Path('/mnt/k/NLCAD/cache/nl03_multirate_01/ad2')
AD2O = ROOT / 'results/hope_cad/nl03_multirate_01/ad2'
need_ad2 = pytest.mark.skipif(not (AD2C.exists() and (AD2O / 'AD2_AUPRO.json').exists()), reason='needs AD 2 caches')


def _legacy_stem_mask_of(root, rel):
    """The pre-2026-11-26 stem-only mask lookup, kept HERE ONLY to reproduce the locked AD2_AUPRO.json numbers, which
    were computed with it. It is wrong for good images (they share file names with bad ones); the package uses the
    corrected rigel_cad.data.ad2_mask_of."""
    gt = Path(root) / Path(rel).parts[0] / 'test_public' / 'ground_truth'
    stem = Path(rel).stem
    c = [q for q in gt.rglob('*') if q.is_file() and q.stem in (stem, stem + '_mask')]
    return c[0] if c else None


@need_ad2
def test_ad2_threshold_aupro_segf1():
    """D-THR thresholds (all scenarios, arm E) and test_public AU-PRO / SegF1 (scenario 'can', final stage) equal the
    locked AD2_SCORE.json / AD2_AUPRO.json values when fed the same (legacy, defective) masks: this checks that the
    package's metric code is the research metric code. The corrected numbers are in results/rigel_cad/ad2/."""
    from rigel_cad import ad2, data, metrics, readout
    sc = json.loads((AD2O / 'AD2_SCORE.json').read_text())
    au = json.loads((AD2O / 'AD2_AUPRO.json').read_text())
    for s in sc['scenarios']:
        d = torch.load(AD2C / 'grids_val' / 'E' / f'{s}__validation.pt', weights_only=False)
        th, _ = ad2.validation_threshold(d['grid'], d['hws'])
        assert th == sc['theta_dthr']['E'][s], s
    root = Path(json.loads((AD2O / 'AD2_MANIFEST.json').read_text())['root'])
    s, T = 'can', len(sc['scenarios'])
    d = torch.load(AD2C / 'grids' / 'E' / f'{s}__test_public.pt', weights_only=False)
    maps = readout.anomaly_maps(d['grid'][-1])
    acc, f1 = metrics.ProAccumulator(), metrics.SegF1()
    for r, hw, m in zip(d['rows'], d['hws'], maps):
        gt = metrics.native_mask(_legacy_stem_mask_of(root, r), hw)
        nat = readout.to_native_cv2(m, hw)
        acc.add(nat.astype(np.float16), gt)
        f1.add(nat > sc['theta_dthr']['E'][s], gt)
    ref = au['per_stage'][s]['E'][str(T - 1)]
    assert acc.au_pro(0.05) == ref['au_pro_0.05'] and acc.au_pro(0.30) == ref['au_pro_0.3']
    assert [f1.tp, f1.fp, f1.fn] == ref['seg_tp_fp_fn'] and f1.value() == ref['seg_f1']


VB8 = Path('/mnt/k/NLCAD/cache/g3e_paper_01/vitb8/mvtec_bottle_test.pt')


@pytest.mark.skipif(not (torch.cuda.is_available() and VB8.exists()), reason='needs CUDA and the ViT-B/8 cache')
def test_vitb8_tokens_bitwise():
    from rigel_cad.encoders import RigelViTB8Encoder
    d = torch.load(VB8, weights_only=True)
    mine = RigelViTB8Encoder().encode_paths([ROOT / 'data/mvtec' / r for r in d['rows'][:32]], batch=32)
    assert torch.equal(mine, d['features'][:32])
