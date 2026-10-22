import hashlib,importlib.util,io,json,sys,tarfile,tempfile,time,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import torch
from continuation_runtime_v1 import common
from continuation_runtime_v1.entry import run_notebook
from continuation_runtime_v1.worker import continue_execution
from notebook_support.common.evidence import export_archive,restore_archive
from training.runtime import atomic_json
from training.support_benchmark_v5_1 import engine
from training.support_throughput_v5_1.contract import sha

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('migration_fixture',ROOT/'tests/support_migration_v5_1/test_migration.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

class ContinuationTests(unittest.TestCase):
    def archive(self,root,status='paused_at_atomic_boundary',lock=None):
        source=Path(root)/'source';source.mkdir();atomic_json(dict(status=status),source/'handoff.json')
        atomic_json(lock or {'op':'fixture'},source/'continuation_lock.json')
        (source/'benchmark').mkdir();(source/'benchmark/payload').write_bytes(b'checkpointfixture')
        out=Path(root)/'run.zip';export_archive(source,out,schema=common.SCHEMA);return source,out

    def test_same_manifest_zip_and_directory_are_one_identity(self):
        with tempfile.TemporaryDirectory() as t:
            source,out=self.archive(t)
            # Kaggle extracted representation includes raw original manifest bytes.
            with zipfile.ZipFile(out) as z:(source/'artifact_manifest.json').write_bytes(z.read('artifact_manifest.json'))
            selected=common.select_parent(t);self.assertEqual(selected[2],common.SCHEMA)
            self.assertIn(selected[1],[source,out])
    def test_ambiguous_versions_fail_and_explicit_override_selects(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a';a.mkdir();_,out=self.archive(a)
            b=Path(t)/'b';b.mkdir();self.archive(b,lock={'other':'different'})
            with self.assertRaisesRegex(ValueError,'Multiple parent'):common.select_parent(t)
            self.assertEqual(common.select_parent(t,str(out))[1],out)
    def test_failed_parent_never_automatically_resumes(self):
        with tempfile.TemporaryDirectory() as t:
            _,out=self.archive(t,status='failed_preserve_evidence')
            with self.assertRaisesRegex(ValueError,'no automatic retry'):common.select_parent(t)
            with self.assertRaises(ValueError):common.select_parent(t,str(out))
    def test_restore_all_payloads_and_operation_lock_guard(self):
        with tempfile.TemporaryDirectory() as t:
            _,out=self.archive(t);work=Path(t)/'work';work.mkdir()
            with self.assertRaisesRegex(ValueError,'orchestration identity'):common.restore_parent(out,common.SCHEMA,work,{'op':'changed'})
            self.assertEqual(common.restore_parent(out,common.SCHEMA,work,{'op':'fixture'}),3)
            self.assertEqual((work/'benchmark/payload').read_bytes(),b'checkpointfixture')
            with self.assertRaisesRegex(ValueError,'fresh kernel'):common.restore_parent(out,common.SCHEMA,work,{'op':'fixture'})
    def test_tampered_payload_cannot_restore_even_with_valid_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            _,out=self.archive(t);bad=Path(t)/'bad.zip'
            with zipfile.ZipFile(out) as z,zipfile.ZipFile(bad,'w') as dst:
                for n in z.namelist():dst.writestr(n,b'corrupt' if n=='benchmark/payload' else z.read(n))
            work=Path(t)/'work';work.mkdir()
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):common.restore_parent(bad,common.SCHEMA,work,{'op':'fixture'})
    def test_g4_immutable_publication_rebound_to_review_and_active_may_advance(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);files={}
            for n,raw in [('benchmark/bridge_checkpoint.pt',b'immutable'),('benchmark/runs/NESTED_seed0/active.pt',b'384'),('handoff.json',b'bridge')]:
                p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);files[n]=sha(p)
            accepted=dict(files=files);origin=root/'origin';origin.mkdir()
            raw=json.dumps(accepted).encode();(origin/'artifact_manifest.json').write_bytes(raw);(origin/'handoff.json').write_bytes(b'bridge')
            # Synthetic accepted map seam; production uses the pinned G4 manifest bytes.
            with patch('continuation_runtime_v1.common.read',return_value=accepted),patch('continuation_runtime_v1.common.BRIDGE_MANIFEST',hashlib.sha256(raw).hexdigest()):
                common.verify_publication(root)
                (root/'benchmark/runs/NESTED_seed0/active.pt').write_bytes(b'future')
                common.verify_publication(root)
                (root/'benchmark/bridge_checkpoint.pt').write_bytes(b'tamper')
                with self.assertRaisesRegex(ValueError,'publication changed'):common.verify_publication(root)
    def test_tar_train_staging_never_extracts_test_and_missing_is_fatal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);archive=root/'dataset.tar.gz';rows=[dict(relative_path='bottle/train/good/001.png',split='train',task_name='bottle')]
            with tarfile.open(archive,'w:gz') as z:
                for n in ['wrapper/bottle/test/bad/001.png','wrapper/bottle/ground_truth/bad/001_mask.png','wrapper/bottle/train/good/001.png']:
                    b=n.encode();m=tarfile.TarInfo(n);m.size=len(b);z.addfile(m,io.BytesIO(b))
            assets=root/'assets';self.assertEqual(common.stage_tar(archive,assets,normal_rows=rows),1)
            self.assertFalse((assets/'bottle/test').exists());self.assertFalse((assets/'bottle/ground_truth').exists())
            self.assertEqual(common.stage_tar(archive,assets,evaluation=True),2)
            with self.assertRaises(ValueError):common.stage_tar(archive,assets,normal_rows=[dict(rows[0],relative_path='bottle/train/good/missing.png')])
    def test_tar_traversal_and_duplicate_rejected(self):
        for names in [['../x'],['bottle/train/good/001.png']*2]:
            with tempfile.TemporaryDirectory() as t:
                archive=Path(t)/'bad.tar'
                with tarfile.open(archive,'w') as z:
                    for n in names:
                        m=tarfile.TarInfo(n);m.size=1;z.addfile(m,io.BytesIO(b'x'))
                rows=[dict(relative_path='bottle/train/good/001.png',split='train',task_name='bottle')]
                with self.assertRaises(ValueError):common.stage_tar(archive,Path(t)/'assets',normal_rows=rows)
    def test_session_budget_reserves_export_and_rejects_nan_or_bool(self):
        self.assertEqual(common.deadline(10,9),10+9*3600-600)
        for n in [True,float('nan'),float('inf'),0,.1,12,'9']:
            with self.assertRaises(ValueError):common.deadline(10,n)
    def test_failure_is_exported_before_error_and_dirty_work_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'failure.zip';c=dict(work=str(root/'work'),output=str(out),started=time.monotonic())
            with patch('continuation_runtime_v1.entry.prepare',side_effect=ValueError('fixture bootstrap error')):
                with self.assertRaises(RuntimeError):run_notebook(c)
            with zipfile.ZipFile(out) as z:self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
            before=sha(out)
            with self.assertRaises(ValueError):run_notebook(c)
            self.assertEqual(sha(out),before)
    def test_worker_failure_pause_complete_export_and_history(self):
        for state,exitcode,success in [('failed_preserve_evidence',1,False),('paused_at_atomic_boundary',0,True),('7.1_complete_pending_review',0,True)]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);work=root/'work';out=root/'out.zip'
                def prepare(c):
                    s=work/'session_history/0002';s.mkdir(parents=True)
                    old=work/'session_history/0001/worker.log';old.parent.mkdir();old.write_bytes(b'old')
                    atomic_json(dict(status=state),s/'worker_status.json')
                    if success and state.startswith('7.1'):
                        atomic_json(dict(cell_count=15),work/'benchmark/runs/NESTED_seed0/final_complete.json')
                        atomic_json({},work/'benchmark/runs/NESTED_seed0/train_complete.json')
                    return dict(session=str(s))
                with patch('continuation_runtime_v1.entry.prepare',side_effect=prepare),patch('continuation_runtime_v1.entry.stream_process',return_value=exitcode):
                    c=dict(work=str(work),output=str(out),started=time.monotonic())
                    if success:run_notebook(c)
                    else:
                        with self.assertRaises(RuntimeError):run_notebook(c)
                with zipfile.ZipFile(out) as z:
                    self.assertEqual(z.read('session_history/0001/worker.log'),b'old')
                    self.assertEqual(json.loads(z.read('handoff.json'))['status'],state)
    def test_actual_migration_train_pause_resume_final15_without_retraining(self):
        with tempfile.TemporaryDirectory() as t:
            f=fixture.Fixture(t);f.publish(f.bridge());e=f.execution();final_called=[]
            def evaluation():
                final_called.append(True)
                self.assertTrue((e.folder/'train_complete.json').exists())
                ep=fixture.EvalProvider()
                e.evaluate(fixture.eval_rows(),fixture.actor,dict(provider=ep,test_digest='fixture'),matrix=False)
            with self.assertRaises(engine.Pause):continue_execution(e,dict(deadline=time.monotonic()+100),f.provider(e.folder,stop=10),fixture.actor,evaluation)
            self.assertFalse(final_called)
            p=f.provider(e.folder);self.assertEqual(continue_execution(e,dict(deadline=time.monotonic()+100),p,fixture.actor,evaluation),'7.1_complete_pending_review')
            self.assertEqual(p.calls[0],8);self.assertEqual(len(list((e.folder/'cells').glob('*/completion.json'))),15)
            finals={p.relative_to(e.folder).as_posix():p.read_bytes() for p in (e.folder/'cells').rglob('*') if p.is_file()}
            p=f.provider(e.folder);continue_execution(e,dict(deadline=time.monotonic()+100),p,fixture.actor,evaluation)
            self.assertEqual(p.calls,[])
            for n,b in finals.items():self.assertEqual((e.folder/n).read_bytes(),b)
    def test_test_callback_never_invoked_after_budget_boundary(self):
        with tempfile.TemporaryDirectory() as t:
            f=fixture.Fixture(t);f.publish(f.bridge());e=f.execution()
            with self.assertRaises(engine.Pause):continue_execution(e,dict(deadline=-1),f.provider(e.folder),fixture.actor,lambda:self.fail('test accessed'))

if __name__=='__main__':unittest.main()
