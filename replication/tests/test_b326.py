"""B326 driver contract tests (CPU). Torch-dependent tests skip when torch is absent."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b326_memory', ROOT / 'scripts' / 'b326_memory.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
RULES = M.config()['rules']


def test_config_pins_match_b325_runs():
    cfg = M.config()
    q = json.loads((ROOT / 'experiments/B325/runs/b325-q-20261119-03/execution.json').read_text(encoding='utf8'))
    r = json.loads((ROOT / 'experiments/B325/runs/b325-r-20261119-01/execution.json').read_text(encoding='utf8'))
    ck = cfg['checkpoints_F']
    assert ck['2711']['stage15_sha256'] == q['train']['F']['stage15_sha256']
    assert ck['2711']['base_sha256'] == q['inputs']['khang_base_sha256']
    for s in ('2712', '2713'):
        assert ck[s]['stage15_sha256'] == r['train'][f'{s}/F']['stage15_sha256']
        assert ck[s]['base_sha256'] == r['base'][s]['sha256']
    assert cfg['jobs']['M2']['seeds'] == [2711, 2712, 2713] and cfg['jobs']['M1']['seeds'] == [2712, 2713]


def _g(passed):
    return {'gate_passed': passed, 'gate_rel': .001 if passed else .02}


@pytest.mark.parametrize('deltas,gate,expected', [
    ([(.001, -.001)] * 3, True, 'INT8_LOSSLESS'),
    ([(.001, -.001), (.004, .0), (-.003, -.002)], True, 'INT8_WITHIN_0005'),
    ([(.010, .0)] * 3, True, 'INT8_WITHIN_0005'),              # a gain is not a loss (card s6.6)
    ([(.001, .0), (-.006, .0), (.0, .0)], True, 'INT8_LOSSY'),
    ([(.0, .0)] * 3, False, 'INT8_LOSSY'),                     # failed gate caps the label
])
def test_label_m2(deltas, gate, expected):
    per = {s: {'d_ap': d[0], 'd_auc': d[1], **_g(gate)} for s, d in zip((2711, 2712, 2713), deltas)}
    assert M.label_m2(per, RULES, [2711, 2712, 2713]) == expected


def test_label_m2_incomplete():
    ok = {'d_ap': 0, 'd_auc': 0, **_g(True)}
    assert M.label_m2({2711: None, 2712: ok}, RULES, [2711, 2712]) == 'NO_LABEL_INCOMPLETE'
    assert M.label_m2({}, RULES, [2711]) == 'NO_LABEL_INCOMPLETE'                  # empty set
    assert M.label_m2({}, RULES, []) == 'NO_LABEL_INCOMPLETE'
    assert M.label_m2({2711: ok}, RULES, [2711, 2712]) == 'NO_LABEL_INCOMPLETE'    # missing seed


def test_label_m2_flag_contradicts_measurement():
    bad = {'d_ap': 0, 'd_auc': 0, 'gate_passed': True, 'gate_rel': .05}
    assert M.label_m2({2711: bad}, RULES, [2711]) == 'NO_LABEL_INCONSISTENT'


def test_label_m1a():
    seeds = [2712, 2713]
    assert M.label_m1a({2712: ['a', 'a'], 2713: ['b', 'b']}, seeds) == 'BASE_SAME_MACHINE_ONLY'
    assert M.label_m1a({2712: ['a', 'x'], 2713: ['b', 'b']}, seeds) == 'BASE_NOT_REPRODUCIBLE'
    assert M.label_m1a({2712: ['a'], 2713: ['b', 'b']}, seeds) == 'NO_LABEL_INCOMPLETE'
    assert M.label_m1a({}, seeds) == 'NO_LABEL_INCOMPLETE'
    assert M.label_m1a({2712: ['a', 'a']}, seeds) == 'NO_LABEL_INCOMPLETE'


@pytest.mark.parametrize('deltas,expected', [
    ([.002, -.004], 'BASE_FROM_SEED_OK'),
    ([.012, -.012], 'BASE_FROM_SEED_MIXED'),
    ([-.011, -.020], 'BASE_FROM_SEED_WORSE'),
    ([.006, .006], 'BASE_FROM_SEED_MIXED'),
    ([.001, None], 'NO_LABEL_INCOMPLETE'),
])
def test_label_m1b(deltas, expected):
    assert M.label_m1b(deltas, RULES) == expected


def test_lanes_round_robin():
    assert M.lanes_round_robin([1, 2, 3], 2) == [[1, 3], [2]]
    assert M.lanes_round_robin([1], 2) == [[1]]


def test_quantizer_roundtrip_and_bytes():
    torch = pytest.importorskip('torch')
    g = torch.Generator().manual_seed(0)
    A = torch.randn(8, 768, generator=g) * 0.02
    B = torch.randn(3072, 8, generator=g) * 0.01
    B[:, 3] = 0                                             # zero rank component: scale guard
    q = M.quantize_archive([{'site': (A, B)}], torch)
    qa, sa, qb, sb = q[0]['site']
    assert qa.dtype == torch.int8 and qb.dtype == torch.int8 and sa.dtype == torch.float16
    assert sa.shape == (8,) and sb.shape == (8,) and float(sb[3]) == 1.0 and int(qb[:, 3].abs().max()) == 0
    assert int(qa.abs().max()) <= 127 and int(qb.abs().max()) <= 127
    (dA, dB), = M.dequantize_archive(q)[0].values()
    assert float((dA - A).abs().max()) <= float(sa.float().max()) / 2 + 1e-6
    assert M.stored_bytes(q) == 8 * 768 + 3072 * 8 + 2 * 8 * 2
    assert float(((dB @ dA) - (B @ A)).abs().max()) < 1e-3


def test_canonical_sha_ignores_file_format():
    torch = pytest.importorskip('torch')
    s1 = {'a': torch.ones(2, 3), 'b': torch.zeros(4)}
    s2 = {'a': torch.ones(2, 3).double(), 'b': torch.zeros(4)}
    assert M.canonical_sha(s1) == M.canonical_sha(s2)
    assert M.canonical_sha(s1) != M.canonical_sha({'b': torch.zeros(4), 'a': torch.ones(2, 3)})


def test_base_from_seed_deterministic_and_complete():
    torch = pytest.importorskip('torch')
    import inspect
    import torch.nn as nn
    assert 'generator' in inspect.signature(nn.init.trunc_normal_).parameters
    M.D.khang('unused', 'unused')
    from exps.vitb8_q2.common import SingleStreamDecoder
    b1, b2, b3 = M.base_from_seed(2712, torch, nn), M.base_from_seed(2712, torch, nn), M.base_from_seed(2713, torch, nn)
    assert list(b1) == list(SingleStreamDecoder(checkpoint=True).state_dict())
    assert M.canonical_sha(b1) == M.canonical_sha(b2) != M.canonical_sha(b3)
    w = [v for k, v in b1.items() if k.endswith('weight') and v.ndim == 2]
    assert all(float(x.abs().max()) <= 0.03 + 1e-7 for x in w)
    assert 0.007 < float(torch.cat([x.flatten() for x in w]).std()) < 0.011       # truncated N(0, 0.01)


def _dry(tmp_path, monkeypatch, job):
    calls = []

    def fake(commands, gpus=None, env_extra=None):
        for i, cmd in enumerate(commands):
            cmd = [str(x) for x in cmd]
            steps = json.loads(cmd[cmd.index('--steps') + 1]) if cmd[1] == 'lane' else [cmd]
            for st in steps:
                calls.append((Path(st[0]).name, st[1], None if gpus is None else gpus[i], st, dict(env_extra or {})))
                arg = lambda k: st[st.index(k) + 1]
                if st[1] == 'locate':
                    M.write_json(Path(arg('--out')) / 'INPUTS.json',
                                 {s: {'stage15': f'/in/{s}/stage15.pt', 'base': f'/in/{s}/BASE.pt', 'hist': f'/in/{s}'}
                                  for s in ('2711', '2712', '2713')})
                if st[1] == 'cpu-base':
                    out = Path(arg('--out'))
                    out.mkdir(parents=True, exist_ok=True)
                    (out / 'BASE.pt').write_bytes(arg('--seed').encode())
                if st[1] == 'fidelity':
                    M.write_json(Path(arg('--run')) / 'FIDELITY.json', {'passed': True})

    monkeypatch.setattr(M, 'run_parallel', fake)
    a = M.argparse.Namespace(job=job, data='d', weights='w', inputs='/in', khang=str(tmp_path / 'khang'),
                             work=str(tmp_path / 'work'), cache=str(tmp_path / 'cache'))
    M.job(a)
    return calls


def test_job_m2_orchestration(tmp_path, monkeypatch):
    calls = _dry(tmp_path, monkeypatch, 'M2')
    phases = [c[1] for c in calls]
    assert phases[:2] == ['gate', 'locate'] and phases[-1] == 'label'
    assert phases.count('quantize') == 3 and phases.count('int8-gate') == 3 and phases.count('score') == 6
    assert phases.count('metrics-plain') == 6 and phases.count('extract-test') == 1
    assert phases.index('extract-test') > max(i for i, p in enumerate(phases) if p in ('quantize', 'int8-gate'))
    gate = calls[0][3]
    assert '--base' in gate
    for c in calls:
        if c[1] == 'score':
            st = c[3]
            seed = st[st.index('--run') + 1].split('s27')[-1][:2]
            assert st[st.index('--hist') + 1] == f'/in/27{seed}' and st[st.index('--dtype') + 1] == 'bfloat16'


def test_job_m1_orchestration(tmp_path, monkeypatch):
    calls = _dry(tmp_path, monkeypatch, 'M1')
    phases = [c[1] for c in calls]
    assert phases[:3] == ['gate', 'locate', 'extract-train'] and phases[-1] == 'label'
    assert '--base' not in calls[0][3]
    cpu = [c for c in calls if c[1] == 'cpu-base']
    assert len(cpu) == 4 and all(c[4] == {'ATEN_CPU_CAPABILITY': 'default', 'CUDA_VISIBLE_DEVICES': ''} for c in cpu)
    trains = [c for c in calls if c[1] == 'train']
    got = [(t[3][t[3].index('--seed') + 1], Path(t[3][t[3].index('--out') + 1]).name, t[2]) for t in trains]
    assert got == [('2712', 'cpu', 0), ('2712', 't4', 0), ('2713', 'cpu', 1), ('2713', 't4', 1)]
    cfg = M.config()
    for t in trains:
        st = t[3]
        seed, arm = st[st.index('--seed') + 1], Path(st[st.index('--out') + 1]).name
        if arm == 't4':
            assert st[st.index('--hist') + 1] == f'/in/{seed}'
            assert st[st.index('--base-sha') + 1] == cfg['checkpoints_F'][seed]['base_sha256']
        else:
            assert st[st.index('--hist') + 1].endswith(f's{seed}{Path("/").as_posix()}cpu_base') or \
                Path(st[st.index('--hist') + 1]).name == 'cpu_base'
    assert phases.count('metrics') == 4 and phases.count('score') == 4


def _cfg_with(tmp_path, monkeypatch):
    """Synthetic mounted inputs matching a patched config (stdlib only)."""
    cfg = json.loads(json.dumps(M.config()))
    root = tmp_path / 'input'
    for seed, spec in cfg['checkpoints_F'].items():
        for key in ('stage15', 'base'):
            p = root / ('kernel-' + seed) / spec[key]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f'{seed}-{key}'.encode())
            spec[key + '_sha256'] = M.sha(p)
    decoy = root / 'kernel-q' / 'b325-q-20261119-03/base_probe/BASE.pt'
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_bytes(b'probe')                                   # must not match any pinned path
    monkeypatch.setattr(M, 'config', lambda: cfg)
    return cfg, root


def test_locate_finds_pinned_paths(tmp_path, monkeypatch):
    cfg, root = _cfg_with(tmp_path, monkeypatch)
    M.locate(M.argparse.Namespace(inputs=str(root), out=str(tmp_path / 'w')))
    got = json.loads((tmp_path / 'w' / 'INPUTS.json').read_text())
    assert set(got) == {'2711', '2712', '2713'}
    base = Path(got['2711']['base']).as_posix()
    assert base.endswith('khang-reference/BASE.pt') and 'base_probe' not in base
    assert Path(got['2712']['hist']).name == 's2712'


def test_locate_rejects_hash_mismatch(tmp_path, monkeypatch):
    cfg, root = _cfg_with(tmp_path, monkeypatch)
    cfg['checkpoints_F']['2712']['stage15_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='SHA-256 differs'):
        M.locate(M.argparse.Namespace(inputs=str(root), out=str(tmp_path / 'w')))


def test_locate_rejects_duplicate(tmp_path, monkeypatch):
    cfg, root = _cfg_with(tmp_path, monkeypatch)
    dup = root / 'copy' / cfg['checkpoints_F']['2713']['stage15']
    dup.parent.mkdir(parents=True)
    dup.write_bytes(b'2713-stage15')
    with pytest.raises(RuntimeError, match='expected one'):
        M.locate(M.argparse.Namespace(inputs=str(root), out=str(tmp_path / 'w')))


def _a1(ap, auc, cats=('bottle', 'cable')):
    return {'final_pixel_ap': ap, 'final_image_auroc': auc,
            'per_category': {c: {'final': {'pixel_ap': ap, 'image_auroc': auc}} for c in cats}}


def test_assemble_labels_m2(tmp_path):
    cfg = M.config()
    for s in cfg['jobs']['M2']['seeds']:
        M.write_json(tmp_path / f's{s}' / 'bf16' / 'RESULT.json', {'seed': s, 'variant': 'bf16', 'a1': _a1(.600, .970)})
        M.write_json(tmp_path / f's{s}' / 'int8' / 'RESULT.json', {'seed': s, 'variant': 'int8', 'a1': _a1(.599, .9705)})
        M.write_json(tmp_path / f's{s}' / 'int8' / 'INT8_GATE.json',
                     {'passed': True, 'relative_mean_abs_error_int8_vs_fp32_in_fp32': .001})
        M.write_json(tmp_path / f's{s}' / 'int8' / 'QUANT.json', {'stored_bytes_int8_plus_scales': 1})
    out = M.assemble_labels('M2', tmp_path, cfg, lambda a, b: {'point': a['final_pixel_ap'] - b['final_pixel_ap']})
    assert out['M2'] == 'INT8_LOSSLESS'
    assert abs(out['seeds'][2712]['ci_int8_minus_bf16']['point'] + .001) < 1e-12


def test_assemble_labels_m2_rejects_swapped_identity(tmp_path):
    cfg = M.config()
    s = cfg['jobs']['M2']['seeds'][0]
    M.write_json(tmp_path / f's{s}' / 'bf16' / 'RESULT.json', {'seed': s, 'variant': 'int8', 'a1': _a1(.6, .97)})
    M.write_json(tmp_path / f's{s}' / 'int8' / 'RESULT.json', {'seed': s, 'variant': 'int8', 'a1': _a1(.6, .97)})
    with pytest.raises(RuntimeError, match='identity'):
        M.assemble_labels('M2', tmp_path, cfg, lambda a, b: {})


def test_assemble_labels_m2_incomplete(tmp_path):
    out = M.assemble_labels('M2', tmp_path, M.config(), lambda a, b: {})
    assert out['M2'] == 'NO_LABEL_INCOMPLETE'


def _m1_tree(tmp_path, cfg, t4_base=None):
    for s, (c, t) in zip((2712, 2713), ((.610, .608), (.600, .603))):
        M.write_json(tmp_path / f's{s}' / 'cpu' / 'RESULT.json', {'seed': s, 'a1': _a1(c, .98)})
        M.write_json(tmp_path / f's{s}' / 't4' / 'RESULT.json', {'seed': s, 'a1': _a1(t, .98)})
        M.write_json(tmp_path / f's{s}' / 'cpu' / 'TRAIN_COMPLETE.json', {'base_sha256': f'file{s}'})
        M.write_json(tmp_path / f's{s}' / 't4' / 'TRAIN_COMPLETE.json',
                     {'base_sha256': t4_base or cfg['checkpoints_F'][str(s)]['base_sha256']})
        for rep in ('cpu_base', 'cpu_base_repeat'):
            M.write_json(tmp_path / f's{s}' / rep / 'BASE_SEED.json', {'canonical_sha256': f'h{s}', 'file_sha256': f'file{s}'})


def test_assemble_labels_m1_rejects_wrong_base(tmp_path):
    cfg = M.config()
    _m1_tree(tmp_path, cfg, t4_base='0' * 64)
    with pytest.raises(RuntimeError, match='trained on BASE'):
        M.assemble_labels('M1', tmp_path, cfg, lambda a, b: {})


def test_run_parallel_fails_fast(tmp_path):
    import time as _t
    bad, slow = tmp_path / 'bad.py', tmp_path / 'slow.py'
    bad.write_text('import sys; sys.exit(3)')
    slow.write_text('import time; time.sleep(120)')
    t0 = _t.monotonic()
    with pytest.raises(RuntimeError, match='child failure'):
        M.run_parallel([[slow], [bad]], poll=0.2)
    assert _t.monotonic() - t0 < 60


def test_assemble_labels_m1(tmp_path):
    cfg = M.config()
    _m1_tree(tmp_path, cfg)
    out = M.assemble_labels('M1', tmp_path, cfg, lambda a, b: {})
    assert out['M1a'] == 'BASE_SAME_MACHINE_ONLY' and out['M1b'] == 'BASE_FROM_SEED_OK'
    assert abs(out['seeds'][2712]['rerun_t4_minus_job_R'] - (.608 - cfg['job_R_F_a1']['2712']['final_pixel_ap'])) < 1e-12
