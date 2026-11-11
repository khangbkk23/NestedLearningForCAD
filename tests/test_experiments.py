"""Evidence retention, design freezing, publication and complete metric matrices."""
import hashlib,json,os,re,subprocess,zipfile
from pathlib import Path
import pytest
from nested_replay_cad import provenance
import kaggle_run,record_run


def miniature(root):
    for name,data in {'experiments/B319/card.md':'retrospective fixture','configs/b319.json':'{}','src/m.py':'x=1','pixi.lock':'fixture lock','LICENSE':'fixture license'}.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
    return root


def test_bundle_retains_bytes_and_excludes_local_secrets(tmp_path):
    root=miniature(tmp_path/'repo');(root/'.local').mkdir();(root/'.local/secret').write_text('not source')
    origin={'git':{'commit':'fixture','dirty':True},'design_commit':'fixture'}
    bundle=tmp_path/'source.zip';manifest,_=provenance.source_bundle(root,bundle,origin)
    (root/'src/m.py').write_text('x=2')
    with zipfile.ZipFile(bundle) as z:
        assert z.read('src/m.py')==b'x=1' and 'SOURCE.json' in z.namelist()
        assert all(not name.startswith('.local/') for name in z.namelist())
        assert all(hashlib.sha256(z.read(name)).hexdigest()==sha for name,sha in manifest.items())


def test_design_requires_committed_unchanged_card_and_recipe(tmp_path):
    root=miniature(tmp_path/'repo')
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL,text=True).strip()
    git('init','--quiet')
    with pytest.raises(ValueError):provenance.check_design(root)
    git('add','experiments/B319/card.md','configs/b319.json')
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','test fixture')
    commit=provenance.check_design(root)
    (root/'src/m.py').write_text('dirty code is snapshotted')
    assert provenance.check_design(root)==commit and provenance.git_state(root)['dirty']
    (root/'configs/b319.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):provenance.check_design(root)


def test_extracted_design_binds_to_hashes(tmp_path):
    root=miniature(tmp_path/'repo');names=('experiments/B319/card.md','configs/b319.json')
    (root/'SOURCE.json').write_text(json.dumps({'design_commit':'fixture','design_hashes':{name:provenance.digest(root/name) for name in names}}))
    assert provenance.check_design(root)=='fixture'
    (root/names[0]).write_text('edited after packaging')
    with pytest.raises(ValueError):provenance.check_design(root)


def test_pack_rejects_uncommitted_design_without_leaving_package(tmp_path,monkeypatch):
    root=miniature(tmp_path/'repo');monkeypatch.setattr(kaggle_run,'ROOT',root)
    local=tmp_path/'artifacts';monkeypatch.setenv('NRCAD_LOCAL_ROOT',str(local))
    args=type('Args',(),{'run_id':'blocked','owner':'fixture'})()
    with pytest.raises(ValueError):kaggle_run.pack(args)
    assert not local.exists()


def test_failed_execution_retains_source_and_seals_receipt(tmp_path,monkeypatch):
    root=miniature(tmp_path/'repo');config={'seed':0,'full_manifest_sha256':'fixture'}
    monkeypatch.setattr(provenance,'source_identity',lambda *args,**kwargs:{'git':{'commit':'fixture','dirty':True},'design_commit':'fixture'})
    record=provenance.RunRecord(root,tmp_path/'run',tmp_path/'run/outputs',config,root/'configs/b319.json','fixture','all')
    runtime=tmp_path/'run/outputs/er_uniform/runtime.json';runtime.parent.mkdir(parents=True);runtime.write_text('{"GPU":"observed device","CUDA":"observed toolkit"}')
    record.finish('failed','worker exit 3');receipt=tmp_path/'run/receipts/receipt.json';before=receipt.read_bytes()
    assert (tmp_path/'run/package/source.zip').is_file()
    assert json.loads(before)['arm_runtime']['er_uniform']['GPU']=='observed device'
    data=json.loads(before);assert data['state']=='failed' and data['stop_reason']=='worker exit 3' and data['git']['dirty']
    record.finish('complete_review_pending');assert receipt.read_bytes()==before


def test_cancelled_and_failed_attempts_are_published_without_metrics(tmp_path):
    run=tmp_path/'run';p=run/'receipts/v1/provider-status.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'kernel':'fixture/run/1','status':'KernelWorkerStatus.CANCEL_ACKNOWLEDGED'}))
    target=record_run.export_run(run,tmp_path/'published','v1');before=(target/'receipt.json').read_bytes()
    assert not json.loads(before)['matrix_complete']
    assert record_run.export_run(run,tmp_path/'published','v1')==target
    p.write_text(json.dumps({'status':'KernelWorkerStatus.ERROR'}))
    with pytest.raises(FileExistsError):record_run.export_run(run,tmp_path/'published','v1')
    assert (target/'receipt.json').read_bytes()==before


def test_running_execution_cannot_be_published(tmp_path):
    run=tmp_path/'run';p=run/'receipts/receipt.json';p.parent.mkdir(parents=True);p.write_text('{"state":"running"}')
    with pytest.raises(ValueError):record_run.export_run(run,tmp_path/'published')


def test_full_matrices_and_producer_preservation(tmp_path):
    run=tmp_path/'run';receipt=run/'receipts/receipt.json';receipt.parent.mkdir(parents=True);receipt.write_text('{"state":"complete_review_pending"}')
    output=run/'outputs';producer=output/'summary.json';output.mkdir();producer.write_text('{"legacy":true}')
    for arm in ('er_uniform','a0_tail','a0_tail_ctx'):
        snapshots=[]
        for stage in range(15):
            metrics={str(i):{'pixel':{'AP':.5,'AUROC':.7},'image':{'AP':.6,'AUROC':.8}} for i in range(stage+1)}
            snapshots.append({'cursor':stage+1,'metrics':metrics})
        p=output/arm/'arms'/arm/'study.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':'COMPLETE_REVIEW_PENDING','committed':3629,'snapshots':snapshots}))
    target=record_run.export_run(run,tmp_path/'published');result=json.loads((target/'metrics.json').read_text())
    for row in result['arms'].values():
        matrix=row['matrix'];assert len(matrix['tasks'])==15 and matrix['boundaries']==list(range(1,16))
        assert set(matrix['values'])=={'pixel_AP','pixel_AUROC','image_AP','image_AUROC'}
        for values in matrix['values'].values():
            assert len(values)==15 and all(len(r)==15 for r in values)
            assert values[0][1] is None and all(v is not None for v in values[-1])
    assert producer.read_text()=='{"legacy":true}'
    assert (target/'results.md').read_text().startswith('# B319 results')


def test_light_fetch_keeps_actual_source_zip():
    assert re.search(kaggle_run.LIGHT_OUTPUT_PATTERN,'runs/b319/run/package/source.zip')
    assert re.search(kaggle_run.LIGHT_OUTPUT_PATTERN,'runs/b319/run/receipts/receipt.json')
    assert not re.search(kaggle_run.LIGHT_OUTPUT_PATTERN,'runs/b319/run/outputs/scores_part001.zip')


def test_closed_reports_cannot_be_overwritten(tmp_path):
    import summarize
    run=tmp_path/'run';output=run/'outputs';output.mkdir(parents=True)
    receipt=run/'receipts/receipt.json';receipt.parent.mkdir();receipt.write_text('{"state":"complete_review_pending"}')
    with pytest.raises(FileExistsError):summarize.write_results(output,{'arms':{}})
    assert not (output/'summary.json').exists()


def test_new_analysis_retains_previous_publication(tmp_path):
    run=tmp_path/'run';p=run/'receipts/receipt.json';p.parent.mkdir(parents=True);p.write_text('{"state":"failed"}')
    previous=record_run.export_run(run,tmp_path/'published');before=(previous/'receipt.json').read_bytes()
    current=record_run.export_run(run,tmp_path/'published',analysis_id='review-2')
    assert current!=previous and (previous/'receipt.json').read_bytes()==before
    sha=json.loads((current/'receipt.json').read_text())['analysis_source_sha256']
    assert provenance.digest(run/'audit/analysis'/sha/'summarize.py')==sha
