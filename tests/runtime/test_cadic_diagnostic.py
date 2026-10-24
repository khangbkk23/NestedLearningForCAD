"""Small independent CPU fixtures; never presented as actual CADIC benchmarks."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hope_cad.baselines.cadic.bank import CADICPatchCoresetConfig, CADICPatchCoresetV1
from hope_cad.baselines.cadic.cache import CachedPairCoreset
from hope_cad.benchmark.engine import Pause
from hope_cad.benchmark.stream import _state_equal
from hope_cad.cli import parser
from hope_cad.runtime.backend import activation
from hope_cad.runtime.cadic_diagnostic import train_case, load_active, final_qa, run, evaluate_case, operator_parity


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def fixture(n=9):
    gen = torch.Generator().manual_seed(28)
    tensors = [torch.randn(1, 4, 3, generator=gen) for _ in range(n)]
    rows = [dict(relative_path=f'bottle/train/good/{i:03d}.png', task_name='bottle', split='train') for i in range(n)]

    class Provider:
        def get(self, row):
            i = rows.index(row)
            return tensors[i], dict(input_seconds=.001, features_seconds=.002)

    cfg = CADICPatchCoresetConfig(budget=7, dim=3, pair_chunk_size=3, query_chunk_size=3, chunk_size=4)
    return rows, tensors, Provider(), cfg


def test_direct_training_matches_independent_reference(tmp_path):
    rows, tensors, provider, cfg = fixture()
    fast, ref = CachedPairCoreset(cfg), CADICPatchCoresetV1(cfg)
    with activation(True):
        done = train_case(tmp_path, rows, provider, fast, dict(source='fixture'), checkpoint_every=2)
        for x in tensors:
            ref.update(x)
    assert _state_equal(fast.state_dict(), ref.state_dict())
    assert done['images'] == 9
    assert done['pipeline_seconds'] == pytest.approx(sum(done[k] for k in ('input_seconds','features_seconds','update_seconds')))
    assert done['per_category']['bottle']['images'] == 9
    assert done['stats']['seen_features'] == 36


def test_interrupted_resume_replays_only_saved_prefix(tmp_path):
    rows, tensors, provider, cfg = fixture()
    bank = CachedPairCoreset(cfg)
    calls = 0

    def stop(_):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise Pause()

    with patch('hope_cad.runtime.cadic_diagnostic.boundary', side_effect=stop):
        with pytest.raises(Pause):
            train_case(tmp_path, rows, provider, bank, dict(source='fixture'), checkpoint_every=2)
    saved = json.loads((tmp_path/'active.json').read_text())
    assert saved['cursor'] == 4
    resumed = CachedPairCoreset(cfg)
    train_case(tmp_path, rows, provider, resumed, dict(source='fixture'), checkpoint_every=2)
    ref = CADICPatchCoresetV1(cfg)
    for x in tensors:
        ref.update(x)
    assert _state_equal(ref.state_dict(), resumed.state_dict())


def test_failed_pointer_publication_keeps_previous_committed_state(tmp_path):
    rows, _, provider, cfg = fixture()
    from hope_cad.runtime.system import atomic_json

    def crash(value, path):
        if Path(path).name == 'active.json' and value['cursor'] == 4:
            raise OSError('fixture crash after inactive tensor slot write')
        atomic_json(value, path)

    with patch('hope_cad.runtime.cadic_diagnostic.atomic_json', side_effect=crash):
        with pytest.raises(OSError):
            train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), dict(source='fixture'), checkpoint_every=2)
    state = load_active(tmp_path, CachedPairCoreset(cfg), dict(source='fixture'), rows)
    assert state['cursor'] == 2
    train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), dict(source='fixture'), checkpoint_every=2)


def test_corrupted_checkpoint_rejected(tmp_path):
    rows, _, provider, cfg = fixture()
    train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), {}, checkpoint_every=2)
    pointer = json.loads((tmp_path/'active.json').read_text())
    (tmp_path/f"state_{pointer['slot']}.pt").write_bytes(b'bad fixture')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_active(tmp_path, CachedPairCoreset(cfg), {}, rows)


def test_source_or_prefix_change_rejected(tmp_path):
    rows, _, provider, cfg = fixture()
    train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), dict(source='first'))
    with pytest.raises(ValueError):
        train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), dict(source='changed'))
    changed = [dict(r) for r in rows]
    changed[0]['relative_path'] = 'bottle/train/good/future.png'
    with pytest.raises(ValueError, match='prefix mismatch'):
        load_active(tmp_path, CachedPairCoreset(cfg), dict(source='first'), changed)


def test_completed_train_is_idempotent_and_does_not_call_provider(tmp_path):
    rows, _, provider, cfg = fixture()
    identity = dict(source='fixture')
    first = train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), identity)
    with patch.object(provider, 'get', side_effect=AssertionError('must not retrain')):
        second = train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), identity)
    assert first == second


def test_final_qa_is_read_only_and_checks_original_nn(tmp_path):
    rows, tensors, provider, cfg = fixture()
    bank = CachedPairCoreset(cfg)
    with activation(True):
        train_case(tmp_path, rows, provider, bank, {})
        before = bank.state_dict()
        qa = final_qa(bank, tensors[0], tmp_path, {}, rows)
    assert qa['passed'] and qa['original_NN_Eq9_exact'] and qa['serialized_checkpoint_verified']
    assert _state_equal(before, bank.state_dict())


def test_nonfinite_feature_never_commits_success(tmp_path):
    rows, tensors, provider, cfg = fixture()
    tensors[0][0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        train_case(tmp_path, rows, provider, CachedPairCoreset(cfg), {})
    assert not (tmp_path/'train_complete.json').exists()


def test_final_qa_reloads_serialized_state(tmp_path):
    rows, tensors, provider, cfg = fixture()
    bank = CachedPairCoreset(cfg)
    train_case(tmp_path, rows, provider, bank, {})
    with patch('hope_cad.runtime.cadic_diagnostic.torch.load', side_effect=ValueError('unloadable fixture')):
        with pytest.raises(ValueError, match='unloadable fixture'):
            final_qa(bank, tensors[0], tmp_path, {}, rows)


def test_reference_update_checks_original_nn_when_replacement_is_required():
    cfg = CADICPatchCoresetConfig(budget=4, dim=3, pair_chunk_size=2, chunk_size=2, query_chunk_size=2)
    bank = CachedPairCoreset(cfg)
    bank.update(torch.tensor([[0.,0.,0.],[.01,0.,0.],[1.,0.,0.],[2.,0.,0.]]))
    raw = torch.tensor([[[10.,0.,0.],[20.,0.,0.],[30.,0.,0.],[40.,0.,0.]]])
    assert operator_parity(bank, raw)['passed']
    def faulty(self, query, vectors):
        return torch.zeros(len(query)), torch.zeros(len(query),dtype=torch.long)
    with patch.object(CachedPairCoreset, '_nearest', faulty):
        with pytest.raises(ValueError, match='update/score differs'):
            operator_parity(bank, raw)


def test_failure_exports_without_becoming_study_acceptance(tmp_path):
    c = dict(run_dir=str(tmp_path/'run'), max_hours=1., budgets=[2500,5000], phase='train',
             device='cpu', data_root=str(tmp_path/'data'), checkpoint=str(tmp_path/'weights'), expected_gpu=None)
    with (patch('hope_cad.runtime.cadic_diagnostic.check_remote_runtime', side_effect=ValueError('fixture runtime mismatch')),
          patch('hope_cad.runtime.cadic_diagnostic.output_directory', return_value=tmp_path/'run')):
        with pytest.raises(RuntimeError, match='evidence preserved'):
            run(c)
    handoff = json.loads((tmp_path/'run/handoff.json').read_text())
    assert handoff['status'] == 'failed_preserve_evidence'
    assert handoff['scientific_acceptance'] is False
    assert not (tmp_path/'run/preflight.json').exists()
    assert (tmp_path/'exports/session_01/evidence.zip').is_file()


def test_foreign_research_directory_rejected_without_mutation(tmp_path):
    work = tmp_path/'run'
    work.mkdir()
    (work/'suite_lock.json').write_text('{"research":true}')
    (work/'handoff.json').write_text('{"status":"pilot_complete"}')
    before = {p.name:p.read_bytes() for p in work.iterdir()}
    c = dict(run_dir=str(work), max_hours=1., budgets=[2500,5000], phase='train',
             device='cpu', data_root=str(tmp_path/'data'), checkpoint=str(tmp_path/'weights'))
    with patch('hope_cad.runtime.cadic_diagnostic.output_directory', return_value=work):
        with pytest.raises(ValueError, match='fresh or diagnostic-owned'):
            run(c)
    assert before == {p.name:p.read_bytes() for p in work.iterdir()}


def test_orchestration_runs_two_fresh_cases_and_exports_resumable_results(tmp_path):
    """Exercise orchestration with tiny CPU inputs, never a K2500/5000 benchmark."""
    rows, tensors, provider, cfg = fixture()
    work = tmp_path/'run'
    c = dict(run_dir=str(work), max_hours=1., budgets=[2500,5000], phase='train',
             device='cpu', data_root=str(tmp_path/'data'), checkpoint=str(tmp_path/'weights'))

    class Extractor:
        def __init__(self, *args):
            pass
        def close(self):
            pass

    manifest = dict(task_order=['bottle'], online=rows)
    with (patch('hope_cad.runtime.cadic_diagnostic.output_directory', return_value=work),
          patch('hope_cad.runtime.cadic_diagnostic.check_remote_runtime', return_value={'fixture':'CPU'}),
          patch('hope_cad.runtime.cadic_diagnostic.build_manifest', return_value=(manifest, 'fixture_rows')),
          patch('hope_cad.runtime.cadic_diagnostic.source_digest', return_value='fixture_source'),
          patch('hope_cad.runtime.cadic_diagnostic.verify_checkpoint', return_value='fixture_weights'),
          patch('hope_cad.runtime.cadic_diagnostic.CADICPatchCoresetConfig', return_value=cfg),
          patch('hope_cad.runtime.cadic_diagnostic.Inputs', return_value=provider),
          patch('hope_cad.models.features.CADICViTFeatureExtractor', Extractor),
          patch('hope_cad.runtime.cadic_diagnostic.torch.cuda.is_available', return_value=False),
          patch('hope_cad.runtime.cadic_diagnostic.torch.cuda.reset_peak_memory_stats'),
          patch('hope_cad.runtime.cadic_diagnostic.torch.cuda.empty_cache'),
          patch('hope_cad.runtime.cadic_diagnostic.resource_usage', return_value={'fixture':True})):
        first = run(c)
        with patch.object(provider, 'get', side_effect=AssertionError('completed cases must not retrain')):
            second = run(c)
    assert first['status'] == second['status'] == 'diagnostic_train_complete'
    assert first['cases'] == second['cases']
    for k in ('2500','5000'):
        assert first['cases'][k]['train']['images'] == len(tensors)
        assert first['cases'][k]['QA']['passed'] is True
    assert (tmp_path/'exports/session_01/evidence.zip').is_file()
    assert (tmp_path/'exports/session_02/timing.json').is_file()
    assert not (work/'preflight.json').exists()


@pytest.mark.parametrize('hours', [0., float('nan'), 5.])
def test_invalid_session_limit_before_outputs(hours):
    with pytest.raises(ValueError):
        run(dict(max_hours=hours))


def test_cli_budget_defaults_and_no_preflight_argument():
    v = parser().parse_args(['cadic-diagnostic','--run-dir','workspace/experiments/test/run'])
    assert v.budgets == [2500,5000] and v.phase == 'train'
    assert v.expected_gpu is None and not hasattr(v, 'preflight')


def test_optional_evaluation_is_read_only_and_resumable(tmp_path):
    cfg = CADICPatchCoresetConfig(budget=7, dim=3, pair_chunk_size=3, chunk_size=4, query_chunk_size=3)
    bank = CachedPairCoreset(cfg)
    gen = torch.Generator().manual_seed(5)
    bank.update(torch.randn(12,3,generator=gen))
    rows = [dict(relative_path=f'task/test/{i}.png',task_id=0,task_name='task',label=i) for i in (0,1)]

    class Provider:
        def get(self, row):
            import numpy as np
            return torch.randn(1,4,3,generator=gen), np.full((2,2),bool(row['label']))

    provider = Provider()
    before = bank.state_dict()
    first = evaluate_case(tmp_path, bank, rows, ['task'], provider, dict(locked='fixture'), None)
    with patch.object(provider,'get',side_effect=AssertionError('no repeated eval')):
        second = evaluate_case(tmp_path, bank, rows, ['task'], provider, dict(locked='fixture'), None)
    assert first == second
    assert _state_equal(before, bank.state_dict())
