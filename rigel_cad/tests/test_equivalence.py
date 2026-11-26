"""Bitwise equivalence of RIGEL-CAD with the frozen research code that produced the paper numbers.

Needs CUDA, the repository's exps/ tree and the locked caches on K: (skipped otherwise). Checked on MPDD order a,
seed 2711 (the T1 cell of G3E-PAPER-01 and the NL03-MULTIRATE-01 benchmark cell):
  1. encoder tokens == cached canonical test tokens;
  2. RIGEL-CAD @ 8 MiB, unbounded, fast-only (F) and FIFO grids == locked grids at every stage;
  3. metrics on the locked grids == the locked METRICS json (per stage) and bootstrap replicates;
  4. task orders == the registered orders.
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


@need
def test_encoder_tokens_bitwise():
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


@need_ad2
def test_ad2_threshold_aupro_segf1():
    """D-THR thresholds (all scenarios, arm E) and test_public AU-PRO / SegF1 (scenario 'can', final stage) equal the
    locked AD2_SCORE.json / AD2_AUPRO.json values."""
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
        gt = metrics.native_mask(data.ad2_mask_of(root, r), hw)
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
