"""B325 driver contract tests (CPU, no torch, no data): run before any job."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]                     # replication/
spec = importlib.util.spec_from_file_location('hope_replicate', ROOT / 'scripts' / 'hope_replicate.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

MVTEC = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
         'tile', 'toothbrush', 'transistor', 'wood', 'zipper')


def test_khang_code_matches_pin():
    assert D.khang_code_intact().startswith('5217cb1')


def test_pin_covers_driver_imports():
    pinned = set(json.loads((ROOT / 'khang_source.json').read_text(encoding='utf8'))['files'])
    for f in ('exps/dinomaly/encoder.py', 'exps/hope_s_01/run.py', 'exps/nar_b14_01/run.py', 'exps/nar_lite_01/run.py',
              'exps/nar_patch_01/run.py', 'exps/nar_pilot_00/core.py', 'exps/nar_pilot_00/full.py',
              'exps/nar_pilot_00/detector.py', 'exps/nar_warm_01/run.py', 'exps/vitb8_q2/common.py'):
        assert f in pinned


def test_arms_match_card():
    assert D.ARMS == {'C': (0.0, False), 'F': (2.1, False), 'H': (0.0, True)}


def test_config_consistent():
    cfg = D.config()
    assert tuple(cfg['khang_train_manifests']) == MVTEC
    for name, job in cfg['jobs'].items():
        runs = [tuple(x) for lane in job['lanes'] for x in lane]
        assert len(job['lanes']) == len(job['gpus'])
        assert sorted(runs) == sorted((s, a) for s in job['seeds'] for a in job['arms'])
        assert set(job['arms']) <= set(D.ARMS)
    assert cfg['jobs']['Q']['base'] == 'khang' and cfg['jobs']['R']['base'] == 'generate'
    assert cfg['jobs']['R']['seeds'] == [2712, 2713]


def test_common_dtype_one_precision_per_seed():
    assert D.common_dtype([{'passed': True}, {'passed': True}]) == 'bfloat16'
    assert D.common_dtype([{'passed': True}, {'passed': False}]) == 'float32'
    assert D.common_dtype([{'passed': True}, {'failure': 'overflow', 'passed': False}]) == 'float32'
    with pytest.raises(RuntimeError):
        D.common_dtype([])


@pytest.mark.parametrize('deltas,lows,expected', [
    ([.012, .015], [.001, .002], 'NL_SIGNAL'),
    ([.012, .015], [-.001, .002], 'NL_MIXED'),
    ([-.011, -.020], [-.03, -.04], 'NL_NEGATIVE'),
    ([.002, -.004], [-.01, -.01], 'NL_NULL_REPLICATED'),
    ([.02, -.02], [0, -.03], 'NL_MIXED'),            # mean 0 but per-seed |delta| too large
    ([.008, .004], [-.001, -.003], 'NL_MIXED'),        # mean .006
])
def test_label_nl(deltas, lows, expected):
    assert D.label_nl(deltas, lows) == expected


@pytest.mark.parametrize('deltas,expected', [
    ([.006, .015], 'HARDPATCH_SUPPORTED'),
    ([.005, .005], 'HARDPATCH_MIXED'),                 # both >= .005 but mean < .010
    ([.002, -.001], 'HARDPATCH_NOT_SUPPORTED'),
    ([-.006, -.010], 'HARDPATCH_NEGATIVE'),
    ([.020, .001], 'HARDPATCH_MIXED'),
])
def test_label_hp(deltas, expected):
    assert D.label_hp(deltas) == expected


def test_labels_need_two_seeds():
    with pytest.raises(RuntimeError):
        D.label_nl([.01], [.0])
    with pytest.raises(RuntimeError):
        D.label_hp([.01, .01, .01])


def _port(f_ap, f_auc, h_ap, scale=1.0):
    ref = D.config()['khang_reference']
    gam = {k: v * scale for k, v in ref['H']['gamma_end'].items()}
    return D.label_port({'F': {'final_pixel_ap': f_ap, 'final_image_auroc': f_auc},
                         'H': {'final_pixel_ap': h_ap}, 'H_gamma_end': gam}, ref)


def test_port_rule():
    assert _port(.6013, .9783, .6006)['label'] == 'PORT_FAITHFUL'
    assert _port(.6013 + .0049, .9783 - .0049, .6006)['label'] == 'PORT_FAITHFUL'
    assert _port(.6013, .9783, .6006, scale=1.15)['label'] == 'PORT_FAITHFUL_GAMMA_DRIFT'
    assert _port(.5900, .9783, .6006)['label'] == 'PORT_DIVERGES'
    assert _port(.6013, .9700, .6006)['label'] == 'PORT_DIVERGES'
    assert _port(.6013, .9783, .6100, scale=1.15)['label'] == 'PORT_DIVERGES'


def test_write_json_never_overwrites(tmp_path):
    D.write_json(tmp_path / 'a.json', {'x': 1})
    with pytest.raises(FileExistsError):
        D.write_json(tmp_path / 'a.json', {'x': 2})
    assert json.loads((tmp_path / 'a.json').read_text()) == {'x': 1}


def test_cli_registers_every_phase():
    for phase in ('gate', 'extract-train', 'make-base', 'train', 'fidelity', 'extract-test', 'score', 'metrics',
                  'label', 'job'):
        with pytest.raises(SystemExit) as e:
            D.main([phase, '--help'])
        assert e.value.code == 0


def _dry_job(tmp_path, monkeypatch, job, fidelity_passed=True):
    """Walk D.job end to end with every child phase stubbed: orchestration order, paths and files."""
    calls = []

    def fake(commands, gpus=None):
        for i, cmd in enumerate(commands):
            cmd = [str(x) for x in cmd]
            steps = json.loads(cmd[cmd.index('--steps') + 1]) if cmd[0] == 'lane' else [cmd]
            for step in steps:
                calls.append((step[0], None if gpus is None else gpus[i], step))
                arg = lambda k: step[step.index(k) + 1]
                if step[0] == 'gate':
                    assert Path(arg('--cache')).is_dir()                  # disk_usage needs an existing directory
                if step[0] == 'make-base':
                    D.write_json(Path(arg('--out')) / 'BASE.json', {'sha256': 'b' * 64})
                if step[0] == 'train':
                    D.write_json(Path(arg('--out')) / 'TRAIN_COMPLETE.json', {})
                if step[0] == 'fidelity':
                    D.write_json(Path(arg('--run')) / 'FIDELITY.json', {'passed': fidelity_passed})

    monkeypatch.setattr(D, 'run_parallel', fake)
    a = D.argparse.Namespace(job=job, data='d', weights='w', khang=str(tmp_path / 'khang'),
                             work=str(tmp_path / 'work'), cache=str(tmp_path / 'cache'))
    D.job(a)
    return calls, json.loads((tmp_path / 'work' / 'PRECISION.json').read_text())


def test_job_q_orchestration(tmp_path, monkeypatch):
    calls, precision = _dry_job(tmp_path, monkeypatch, 'Q')
    phases = [c[0] for c in calls]
    assert phases == ['gate', 'extract-train', 'make-base', 'train', 'train', 'fidelity', 'fidelity',
                      'extract-test', 'score', 'score', 'metrics', 'metrics', 'label']
    gate = calls[0][2]
    assert '--base' in gate and gate[gate.index('--base') + 1].endswith('BASE.pt')
    trains = {c[2][c[2].index('--arm') + 1]: c for c in calls if c[0] == 'train'}
    assert trains['F'][1] == 0 and trains['H'][1] == 1                      # F on GPU 0, H on GPU 1
    for c in trains.values():
        s = c[2]
        assert s[s.index('--base-sha') + 1] == D.config()['khang_base_sha256']
        assert s[s.index('--hist') + 1] == str(tmp_path / 'khang')
    assert precision == {'2711': 'bfloat16'}
    test = next(c[2] for c in calls if c[0] == 'extract-test')
    assert test.count('--sealed') == 2


def test_job_r_orchestration_and_shared_precision(tmp_path, monkeypatch):
    cfg = D.config()                    # placeholder time limits for the dry run
    cfg['jobs']['R']['phase_seconds'] = {'train_arm': 1, 'fidelity_arm': 1, 'metrics_arm': 1}
    monkeypatch.setattr(D, 'config', lambda: cfg)
    calls, precision = _dry_job(tmp_path, monkeypatch, 'R', fidelity_passed=False)
    trains = [c for c in calls if c[0] == 'train']
    assert [(c[2][c[2].index('--seed') + 1], c[2][c[2].index('--arm') + 1], c[1]) for c in trains] == [
        ('2712', 'C', 0), ('2712', 'F', 0), ('2712', 'H', 0), ('2713', 'C', 1), ('2713', 'F', 1), ('2713', 'H', 1)]
    for c in trains:                                                        # one BASE per seed, shared by arms
        s = c[2]
        seed = s[s.index('--seed') + 1]
        assert s[s.index('--hist') + 1] == str(tmp_path / 'work' / f's{seed}')
    assert precision == {'2712': 'float32', '2713': 'float32'}
    scores = [c[2] for c in calls if c[0] == 'score']
    assert all(s[s.index('--dtype') + 1] == 'float32' for s in scores)
    assert sum(c[0] == 'metrics' for c in calls) == 6 and calls[-1][0] == 'label'


def test_label_c_path():
    ref = D.config()['khang_reference']['C']
    ok = D.label_c_path({'final_pixel_ap': ref['final_pixel_ap'] + .004, 'final_image_auroc': ref['final_image_auroc'] - .004}, ref)
    assert ok['label'] == 'C_PATH_FAITHFUL'
    assert D.label_c_path({'final_pixel_ap': ref['final_pixel_ap'] - .006, 'final_image_auroc': ref['final_image_auroc']},
                          ref)['label'] == 'C_PATH_DIVERGES'


def test_job_c_orchestration(tmp_path, monkeypatch):
    calls, precision = _dry_job(tmp_path, monkeypatch, 'C')
    assert [c[0] for c in calls] == ['gate', 'extract-train', 'make-base', 'train', 'fidelity', 'extract-test',
                                     'score', 'metrics', 'label']
    train = next(c for c in calls if c[0] == 'train')
    s = train[2]
    assert s[s.index('--arm') + 1] == 'C' and train[1] == 0
    assert s[s.index('--base-sha') + 1] == D.config()['khang_base_sha256']
    assert precision == {'2711': 'bfloat16'}


def test_job_r_single_gpu_runs_lanes_in_turn(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(D, 'run_parallel', lambda commands, gpus=None: seen.append((len(commands), gpus)))
    D.run_on_gpus([['a'], ['b']], [0])
    D.run_on_gpus([['a'], ['b']], [0, 1])
    assert seen == [(1, [0]), (1, [0]), (2, [0, 1])]
