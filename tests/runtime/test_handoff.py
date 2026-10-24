"""Independent CPU transport, actual tiny learner-state and readiness fixtures."""
import json
import stat
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hope_cad.contract import digest,file_sha,CONTRACT
from hope_cad.runtime import handoff as h
from hope_cad.runtime.system import atomic_json
from hope_cad.runtime.identity import source_digest,source_files


@pytest.fixture
def tmp_path():
    # Keep source/PDF names below Windows MAX_PATH in transport fixtures.
    import uuid
    from hope_cad.paths import WORKSPACE
    folder=WORKSPACE/'checks/handoff_tests'/uuid.uuid4().hex[:10]
    folder.mkdir(parents=True)
    return folder


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def archive(path, **updates):
    content={'experiment/run/opaque.pt':b'fixture checkpoint', 'records/environment.json':b'{}',
             'source/README.md':b'fixture source'}
    sf={'README.md':__import__('hashlib').sha256(content['source/README.md']).hexdigest()}
    manifest=dict(schema=h.SCHEMA,experiment_id='fixture_3090',source_commit='a'*40,
        source_digest=digest(sf),source_file_hashes=sf,suite_identity={'fixture':'identity'},
        files={n:dict(sha256=__import__('hashlib').sha256(v).hexdigest(),bytes=len(v)) for n,v in content.items()})
    manifest.update(updates)
    with zipfile.ZipFile(path,'w') as z:
        for n,v in content.items():z.writestr(n,v)
        z.writestr(h.HEADER,json.dumps(manifest))
    return manifest


def test_zip_verifies_every_payload_and_preserves_bytes(tmp_path):
    p=tmp_path/'a.zip';m=archive(p);target=tmp_path/'staging';target.mkdir()
    assert h.inspect_archive(p,extract=target)==m
    assert (target/'experiment/run/opaque.pt').read_bytes()==b'fixture checkpoint'
    with pytest.raises(ValueError,match='empty staging'):h.inspect_archive(p,extract=target)


@pytest.mark.parametrize('extra',['../escape','experiment/../escape','experiment/run/ACTIVE.pt','records/link','unknown/file'])
def test_unsafe_symlink_or_unlisted_payload_rejected(tmp_path,extra):
    p=tmp_path/'bad.zip';archive(p)
    with zipfile.ZipFile(p,'a') as z:
        entry=zipfile.ZipInfo(extra)
        if extra=='records/link':entry.external_attr=(stat.S_IFLNK|0o777)<<16
        z.writestr(entry,b'bad')
    with pytest.raises(ValueError):h.inspect_archive(p)
    assert not (tmp_path/'escape').exists()


def test_changed_payload_and_source_snapshot_rejected(tmp_path):
    p=tmp_path/'a.zip';archive(p)
    with zipfile.ZipFile(p) as z:payload={n:z.read(n) for n in z.namelist()}
    payload['experiment/run/opaque.pt']=b'changed'
    with zipfile.ZipFile(p,'w') as z:
        for n,v in payload.items():z.writestr(n,v)
    with pytest.raises(ValueError):h.inspect_archive(p)
    archive(p,source_digest='0'*64)
    with pytest.raises(ValueError,match='Source snapshot'):h.inspect_archive(p)


def test_case_collision_duplicate_json_and_truncated_zip_rejected(tmp_path):
    p=tmp_path/'a.zip';archive(p)
    with zipfile.ZipFile(p,'a') as z:z.writestr('experiment/run/OPAQUE.pt',b'collision')
    with pytest.raises(ValueError,match='case-colliding'):h.inspect_archive(p)
    with zipfile.ZipFile(p,'w') as z:z.writestr(h.HEADER,'{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError,match='Duplicate JSON'):h.inspect_archive(p)
    p.write_bytes(b'PK\x03\x04truncated')
    with pytest.raises(zipfile.BadZipFile):h.inspect_archive(p)


def test_scheduler_requires_reviews_screen_and_all_registered_jobs():
    _,main,extension=h.registered();c={}
    assert h.plan(c,False,False,None)['phase']=='preflight'
    assert h.plan(c,False,False,None,preflight_exists=True)['status']=='review_required'
    assert h.plan(c,True,False,None)['phase']=='pilot'
    assert h.plan(c,True,True,None)['job']==main[0]
    c={name:dict(matrix=True) for name in main}
    assert h.plan(c,True,True,False)['status']=='blocked_screen'
    assert h.plan(c,True,True,True)['job']==extension[0]
    c.update({name:dict(matrix=True) for name in extension})
    assert h.plan(c,True,True,True)['status']=='complete'


def test_budget_stage_transition_differs_from_partial_stage_rate_change(tmp_path):
    old=dict(stage='B',authorized=True,USD_per_hour=.2,USD_cap=3.)
    atomic_json(dict(binding=dict(stage='B',allocation=old,cap_seconds=43200),attempts=[]),tmp_path/'budget_B.json')
    assert h.prepare_allocation(tmp_path,'C',.4,5.)['stage']=='C'
    with pytest.raises(ValueError,match='never rewrites'):h.prepare_allocation(tmp_path,'B',.4,5.)
    assert h.prepare_allocation(tmp_path,'B',.2,3.)==old
    assert h.prepare_allocation(tmp_path,'D',.5,2.)['gpu_wall_seconds_cap']==14400


def test_budget_archive_sidecar_binding_and_corruption(tmp_path):
    ws,run,_,_=make_pack_fixture(tmp_path)
    h.ledger_check(run,run.parent)
    sidecar=run.parent/'exports/session_01/accounting.json'
    v=h.read(sidecar);v['attempt']['charged_seconds']+=1;atomic_json(v,sidecar)
    with pytest.raises(ValueError,match='accounting/archive'):h.ledger_check(run,run.parent)


def test_open_or_missing_ledger_is_not_a_new_budget(tmp_path):
    with pytest.raises(ValueError,match='missing origin ledger'):h.ledger_check(tmp_path,tmp_path)
    atomic_json(dict(binding=dict(stage='B'),attempts=[dict(id='uuid',status='reserved_incomplete')]),tmp_path/'budget_B.json')
    with pytest.raises(ValueError,match='open/duplicate'):h.ledger_check(tmp_path,tmp_path)


def tiny_job(folder,variant='CADIC',n=10):
    from hope_cad.models.research_session import ResearchSession
    from hope_cad.baselines.cadic.bank import CADICPatchCoresetConfig
    from hope_cad.benchmark.engine import train,evaluate
    from hope_cad.runtime.backend import activation
    half=n//2
    rows=[dict(task_id=i//half,task_name='bottle' if i<half else 'cable',split='train',
               relative_path=f't{i//half}/train/good/{i}.png',sha256='f'*64) for i in range(n)]
    tasks=['bottle','cable'];gen=torch.Generator().manual_seed(19)
    xs=[torch.randn(1,4,3,generator=gen)*(.01 if variant!='CADIC' else 1.)
        +(1.4**i if variant!='CADIC' else 0.) for i,_ in enumerate(rows)]
    def factory(name='CADIC_seed0',device='cpu'):
        return ResearchSession(variant,seed=0,device='cpu',bank_config=CADICPatchCoresetConfig(
            budget=6,dim=3,chunk_size=3,pair_chunk_size=3,query_chunk_size=3),
            model_config=dict(dim=3,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))
    class Provider:
        def get(self,row):
            value=xs[rows.index(row)]
            from hope_cad.models.primitives.state import tensor_digest
            atomic_json(dict(row=row,backbone_sha256=CONTRACT['backbone_sha256'],tensor_sha256=tensor_digest(({'patches':value},))),
                        folder/'feature_receipts'/(digest(row)+'.json'))
            return value
    expected=dict(job=variant+'_seed0',mode=variant,seed=0,suite_digest='fixture',
        method_digest=digest(h.method_identity(variant)),scope='fresh_fulltrain_3629')
    with activation(True):train(folder,rows,tasks,expected,Provider(),factory)
    return rows,tasks,expected,factory


def test_actual_h32_history_optimizer_rng_and_pending_state_transport(tmp_path):
    folder=tmp_path/'H32_EXACT_seed0'
    rows,tasks,expected,factory=tiny_job(folder,'H32_EXACT',48)
    before={p.relative_to(folder).as_posix():file_sha(p) for p in folder.rglob('*') if p.is_file()}
    state=torch.load(folder/'active.pt',map_location='cpu',weights_only=True)
    actor=factory();actor.load_state_dict(state['session'])
    assert actor.learner.labels>=32 and actor.learner.slow_updates>0
    assert len(actor.learner.history.buffer)>0
    with patch.object(h,'session_for',factory):
        assert h.validate_job(folder,rows,tasks,expected)['train'] is True
    assert before=={p.relative_to(folder).as_posix():file_sha(p) for p in folder.rglob('*') if p.is_file()}


def test_actual_cpu_checkpoint_events_tasks_validate_without_mutation(tmp_path):
    folder=tmp_path/'CADIC_seed0';rows,tasks,expected,factory=tiny_job(folder)
    before={p.relative_to(folder).as_posix():file_sha(p) for p in folder.rglob('*') if p.is_file()}
    with patch.object(h,'session_for',factory):
        result=h.validate_job(folder,rows,tasks,expected)
    assert result==dict(started=True,train=True,matrix=False,cursor=10)
    assert before=={p.relative_to(folder).as_posix():file_sha(p) for p in folder.rglob('*') if p.is_file()}
    event=folder/'events/00000.json';v=h.read(event);v['identity']='future';atomic_json(v,event)
    with patch.object(h,'session_for',factory),pytest.raises(ValueError,match='Event chronology'):
        h.validate_job(folder,rows,tasks,expected)


def test_state_loader_uses_qualified_device_instead_of_forcing_cpu(tmp_path):
    folder=tmp_path/'CADIC_seed0';rows,tasks,expected,factory=tiny_job(folder)
    devices=[]
    def recorded(name,device):
        devices.append(device)
        return factory(name,'cpu')  # Fixture data is CPU; inspect routing independently.
    with patch.object(h,'session_for',recorded):
        assert h.validate_job(folder,rows,tasks,expected,device='qualified-device')['train'] is True
    assert devices==['qualified-device']


def test_fabricated_completion_and_checkpoint_corruption_blocked(tmp_path):
    folder=tmp_path/'CADIC_seed0';rows,tasks,expected,factory=tiny_job(folder)
    v=h.read(folder/'train_complete.json');v['images']=9;atomic_json(v,folder/'train_complete.json')
    with patch.object(h,'session_for',factory),pytest.raises(ValueError,match='Fabricated'):
        h.validate_job(folder,rows,tasks,expected)
    (folder/'active.pt').write_bytes(b'broken')
    with pytest.raises(ValueError,match='corrupt active'):h.validate_job(folder,rows,tasks,expected)


def test_target_probe_checks_original_features_and_readonly_actual_state(tmp_path):
    run=tmp_path/'run';folder=run/'pilot/CADIC_seed0'
    rows,tasks,_,factory=tiny_job(folder)
    atomic_json(dict(online=rows,task_order=tasks),run/'manifest_train.json')
    raw=torch.randn(1,4,3,generator=torch.Generator().manual_seed(19))
    class Extractor:
        def __init__(self,*a,**k):pass
        def close(self):pass
    class Inputs:
        def __init__(self,*a,**k):pass
        def get(self,row):return raw,{}
    before=file_sha(folder/'active.pt')
    with (patch('hope_cad.models.features.CADICViTFeatureExtractor',Extractor),
          patch('hope_cad.runtime.cadic_diagnostic.Inputs',Inputs),
          patch.object(h,'session_for',factory),patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache')):
        report=h.target_probe(run,'fixture','fixture','cpu')
        assert report['passed'] and report['checks']==[dict(job='CADIC_seed0',index=0)]
        assert before==file_sha(folder/'active.pt')
        receipt=folder/'feature_receipts'/(digest(rows[0])+'.json');v=h.read(receipt)
        v['tensor_sha256']='0'*64;atomic_json(v,receipt)
        with pytest.raises(ValueError,match='raw feature parity'):
            h.target_probe(run,'fixture','fixture','cpu')


def test_actual_readonly_matrix_coverage_rejects_missing_cell_and_stale_metrics(tmp_path):
    import numpy as np
    from hope_cad.benchmark.engine import evaluate
    folder=tmp_path/'CADIC_seed0';rows,tasks,expected,factory=tiny_job(folder)
    test_rows=[dict(task_id=i,task_name=tasks[i],relative_path=f'{tasks[i]}/test/{label}.png',label=label)
               for i in range(2) for label in (0,1)]
    class Evaluation:
        def get(self,row):
            gen=torch.Generator().manual_seed(20+row['task_id']*2+row['label'])
            return torch.randn(1,4,3,generator=gen),np.full((4,4),bool(row['label']))
    tm=dict(rows=test_rows,digest='fixture_test')
    evaluate(folder,tasks,test_rows,factory,dict(provider=Evaluation(),test_digest=tm['digest']))
    evaluate(folder,tasks,test_rows,factory,dict(provider=Evaluation(),test_digest=tm['digest']),matrix=True)
    with patch.object(h,'session_for',factory):
        assert h.validate_job(folder,rows,tasks,expected,test=tm)['matrix'] is True
        metric=folder/'metrics/final_macro.json';original=metric.read_bytes();v=h.read(metric)
        v['macro_final_i_auroc']=.123;atomic_json(v,metric)
        with pytest.raises(ValueError,match='Final metrics differ'):h.validate_job(folder,rows,tasks,expected,test=tm)
        metric.write_bytes(original)
        (folder/'cells/00_00/completion.json').unlink()
        with pytest.raises(ValueError,match='Missing completed'):h.validate_job(folder,rows,tasks,expected,test=tm)


def test_gate_no_archive_and_changed_source_do_not_initialize_cuda(tmp_path):
    with patch.object(h,'WORKSPACE',tmp_path),patch('torch.cuda.get_device_name',side_effect=AssertionError('no GPU')):
        assert h.check()['status']=='fresh'
        p=tmp_path/'a.zip';archive(p)
        result=h.check(p,restore=True)
        assert result['status']=='blocked_source' and result['allowed'] is False
        assert not (tmp_path/'experiments').exists()


def test_review_bound_to_actual_source_receipt_not_filename(tmp_path):
    identity={'source_digest':'source'}
    v=dict(status='research_preflight_passed_pending_review',binding=identity,binding_digest=digest(identity))
    atomic_json(v,tmp_path/'preflight.json')
    assert h.accepted(tmp_path,'preflight',identity) is False
    atomic_json(dict(accepted=True,phase='preflight',source_digest='source',receipt_sha256='wrong'),tmp_path/'preflight_review.json')
    with pytest.raises(ValueError,match='Review/receipt'):h.accepted(tmp_path,'preflight',identity)


def test_cli_handoff_commands_are_explicit_not_research_bypass():
    from hope_cad.cli import parser
    value=parser().parse_args(['handoff-check','--archive','a.zip','--restore','--rate','.2','--cap','2'])
    assert value.restore and value.rate==.2
    assert parser().parse_args(['handoff-pack','--run-dir','run']).command=='handoff-pack'


def full_manifest():
    """Metadata-only canonical cardinalities; no actual image data or benchmark."""
    from hope_cad.data.protocol import CANONICAL_MVTEC_TASKS
    tasks=list(CANONICAL_MVTEC_TASKS);rows=[]
    counts=[209]+[245]*13+[235]
    for i,(task,n) in enumerate(zip(tasks,counts)):
        rows.extend(dict(task_id=i,task_name=task,split='train',relative_path=f'{task}/train/good/{j:04d}.png',sha256='e'*64) for j in range(n))
    assert len(rows)==3629
    value=dict(schema='hope_fulltrain_manifest_v1',task_order=tasks,source_identity=source_digest(),online=rows,excluded=[])
    value['digest']=digest(value)
    return value


def make_pack_fixture(tmp_path):
    from hope_cad.runtime.research import binding
    from hope_cad.benchmark.study import lock
    from hope_cad.runtime.budget import Budget
    from hope_cad.runtime.evidence import export_archive
    ws=tmp_path/'workspace'; run=ws/'experiments/fixture_3090/run';run.mkdir(parents=True)
    manifest=full_manifest();rt={'fixture':'CPU'}; cp='fixture_backbone'
    identity=binding(rt,digest(manifest['online']),cp)
    lock(run/'suite_lock.json',identity);atomic_json(manifest,run/'manifest_train.json')
    for p in (ws/'data/mvtec.asset.json',ws/'weights/vit_base_patch8_224.augreg_in21k.asset.json'):
        atomic_json({'fixture':'asset'},p)
    atomic_json(dict(fixture=True),ws/'records/environment.json')
    allocation=dict(stage='B',authorized=True,USD_per_hour=.2,USD_cap=2.)
    budget=Budget(run,'B',1,allocation=allocation);budget.begin_export()
    session=run.parent/'exports/session_01';session.mkdir(parents=True)
    exported=export_archive(run,session/'evidence.zip',schema='hope_hq_fulltrain_evidence_v1')
    budget.finish(exported,accounting_path=session/'accounting.json')
    return ws,run,manifest,identity


def test_pack_to_restore_real_zip_preserves_origin_and_does_not_start_training(tmp_path):
    ws,run,manifest,identity=make_pack_fixture(tmp_path)
    with patch.object(h,'WORKSPACE',ws),patch.object(h.subprocess if hasattr(h,'subprocess') else subprocess,'check_output',return_value='a'*40):
        # Explicitly emulate a committed checkout: never change real Git state.
        def git(args,**kwargs):
            return '' if 'status' in args else 'a'*40
        with patch('subprocess.check_output',side_effect=git):
            packed=h.pack(run)
    m=h.inspect_archive(packed['path'])
    assert m['source_file_hashes']==source_files()
    origin={p.relative_to(run).as_posix():p.read_bytes() for p in run.rglob('*') if p.is_file()}
    target=tmp_path/'target'
    with (patch.object(h,'WORKSPACE',target),
          patch('hope_cad.runtime.identity.check_remote_runtime',return_value=identity['runtime']),
          patch('hope_cad.data.fulltrain.build_manifest',return_value=(manifest,digest(manifest['online']))),
          patch('hope_cad.runtime.execution.verify_checkpoint',return_value='fixture_backbone'),
          patch.object(h,'target_probe',return_value={'passed':True,'scope':'mocked CPU transport fixture'})):
        result=h.check(packed['path'],restore=True,device='cpu',rate=.2,cap=2.)
        assert result['allowed'] and result['restored'] and result['next_phase']=='preflight'
        dest=target/'experiments/fixture_3090/run'
        assert origin=={p.relative_to(dest).as_posix():p.read_bytes() for p in dest.rglob('*') if p.is_file()}
        assert not (target/'records/environment.json').exists()
        assert (target/'experiments/fixture_3090/origin'/packed['sha256']/'records/environment.json').is_file()
        assert not (dest/'preflight_review.json').exists()
        repeated=h.check(packed['path'],restore=True,device='cpu',rate=.2,cap=2.)
        assert repeated['allowed'] is False and 'already exists' in repeated['reason']


def test_gate_blocks_new_driver_before_probe_or_restore(tmp_path):
    ws,run,manifest,identity=make_pack_fixture(tmp_path)
    def git(args,**kwargs):return '' if 'status' in args else 'a'*40
    with patch.object(h,'WORKSPACE',ws),patch('subprocess.check_output',side_effect=git):
        packed=h.pack(run)
    target=tmp_path/'newhost'
    with (patch.object(h,'WORKSPACE',target),
          patch('hope_cad.runtime.identity.check_remote_runtime',return_value={'fixture':'different_driver'}),
          patch('hope_cad.data.fulltrain.build_manifest',return_value=(manifest,digest(manifest['online']))),
          patch('hope_cad.runtime.execution.verify_checkpoint',return_value='fixture_backbone'),
          patch.object(h,'target_probe',side_effect=AssertionError('must not probe mismatched runtime'))):
        result=h.check(packed['path'],restore=True,device='cpu',rate=.2,cap=2.)
    assert result['status']=='blocked_identity' and result['different_fields']==['runtime']
    assert not (target/'experiments').exists()


def test_pack_refuses_dirty_source_and_active_writer(tmp_path):
    ws,run,_,_=make_pack_fixture(tmp_path)
    with patch.object(h,'WORKSPACE',ws),patch('subprocess.check_output',return_value=' M source'):
        with pytest.raises(ValueError,match='committed clean'):h.pack(run)
    (run/'.writer.lock').write_text('another writer')
    with pytest.raises(ValueError,match='Active writer'):h.inspect_run(run)
