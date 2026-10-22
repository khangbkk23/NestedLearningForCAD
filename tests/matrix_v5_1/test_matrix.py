import copy,hashlib,importlib.util,io,json,shutil,tarfile,tempfile,time,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import torch
from training.runtime import atomic_json
from training.support_benchmark_v5_1 import engine
from training.support_throughput_v5_1.contract import sha
from notebook_support.common.evidence import export_archive
from notebook_support.matrix_v5_1 import common
from notebook_support.matrix_v5_1.entry import run_notebook,prepare
from notebook_support.matrix_v5_1.worker import evaluate_matrix,readonly_preflight

ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
bench=module('matrix_benchmark_fixture','tests/support_benchmark_v5_1/test_benchmark_v5_1.py')
migrate=module('matrix_migration_fixture','tests/support_migration_v5_1/test_migration.py')

def files(root):return {p.relative_to(root).as_posix():sha(p) for p in Path(root).rglob('*') if p.is_file()}

class MatrixTests(unittest.TestCase):
    def parent(self,root,job='CADIC_seed0',status='7.1_complete_pending_review',schema='fixture_original'):
        root=Path(root);source=root/'source';source.mkdir();atomic_json(dict(status=status,job=job),source/'handoff.json')
        (source/'benchmark').mkdir();(source/'benchmark/state.bin').write_bytes(b'fixed train state')
        out=root/'own.zip';export_archive(source,out,schema=schema)
        with zipfile.ZipFile(out) as z:raw=z.read('artifact_manifest.json');m=json.loads(raw)
        (source/'artifact_manifest.json').write_bytes(raw)
        refs=dict(jobs={job:dict(schema=schema,manifest_sha256=hashlib.sha256(raw).hexdigest(),archive_sha256=sha(out))})
        return source,out,refs,m

    def test_zip_directory_duplicate_is_single_parent_and_wrong_job_fails(self):
        with tempfile.TemporaryDirectory() as t:
            source,out,ref,m=self.parent(t)
            self.assertIn(common.select_parent(t,'CADIC_seed0',ref)[1],(source,out))
            with self.assertRaises(ValueError):common.select_parent(t,'NESTED_seed0',dict(jobs={'NESTED_seed0':ref['jobs']['CADIC_seed0']}),str(out))
    def test_partial7_1_failed7_2_and_wrong_zip_bytes_rejected(self):
        for status,schema in [('paused_at_atomic_boundary','fixture_original'),('failed_preserve_evidence',common.SCHEMA)]:
            with tempfile.TemporaryDirectory() as t:
                _,out,ref,m=self.parent(t,status=status,schema=schema)
                with self.assertRaises(ValueError):common.select_parent(t,'CADIC_seed0',ref,str(out))
        with tempfile.TemporaryDirectory() as t:
            _,out,ref,m=self.parent(t);ref['jobs']['CADIC_seed0']['archive_sha256']='f'*64
            with self.assertRaisesRegex(ValueError,'ZIP bytes'):common.select_parent(t,'CADIC_seed0',ref,str(out))
    def test_latest_matrix_is_preferred_and_ambiguous_versions_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            first=Path(t)/'first';first.mkdir();_,_,ref,m=self.parent(first)
            second=Path(t)/'second';second.mkdir();_,out,_,_=self.parent(second,schema=common.SCHEMA,status='paused_at_atomic_boundary')
            self.assertEqual(common.select_parent(t,'CADIC_seed0',ref)[2],common.SCHEMA)
            third=Path(t)/'third';third.mkdir();source,out2,_,_=self.parent(third,schema=common.SCHEMA,status='7.2_complete_pending_review')
            with self.assertRaisesRegex(ValueError,'Multiple matrix'):common.select_parent(t,'CADIC_seed0',ref)
            self.assertEqual(common.select_parent(t,'CADIC_seed0',ref,str(out))[1],out)
    def test_restore_preserves_payloads_and_pins_original_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            source,out,refs,m=self.parent(t);work=Path(t)/'work';work.mkdir();op={'source':'test'}
            with patch.object(common,'parent_manifest',return_value=m):
                self.assertEqual(common.restore(out,'fixture_original',work,'CADIC_seed0',refs,op),2)
                self.assertEqual(common.verify_frozen(work,'CADIC_seed0',refs),1)
                (work/'handoff.json').write_text('new operational handoff')
                common.verify_frozen(work,'CADIC_seed0',refs)
                (work/'benchmark/state.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Frozen 7.1'):common.verify_frozen(work,'CADIC_seed0',refs)
    def test_resume_rejects_source_and_cross_job_lock(self):
        with tempfile.TemporaryDirectory() as t:
            source,out,refs,m=self.parent(t,schema=common.SCHEMA,status='paused_at_atomic_boundary')
            atomic_json(common.operation({'source':'old'},'CADIC_seed0'),source/'matrix_lock.json')
            (source/'artifact_manifest.json').unlink()
            out2=Path(t)/'resume.zip';export_archive(source,out2,schema=common.SCHEMA)
            with self.assertRaisesRegex(ValueError,'source/job'):common.restore(out2,common.SCHEMA,Path(t)/'work','CADIC_seed0',refs,{'source':'new'})
    def test_corrupt_parent_member_is_rejected_before_state_use(self):
        with tempfile.TemporaryDirectory() as t:
            _,out,refs,m=self.parent(t);bad=Path(t)/'bad.zip'
            with zipfile.ZipFile(out) as z,zipfile.ZipFile(bad,'w') as dst:
                for n in z.namelist():dst.writestr(n,b'bad' if n=='benchmark/state.bin' else z.read(n))
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):common.restore(bad,'fixture_original',Path(t)/'work','CADIC_seed0',refs,{})
    def test_runtime_mismatch_fails_before_import_or_dataset(self):
        with patch('notebook_support.matrix_v5_1.entry.verify_source',return_value=({},dict(runtime={'T4':True}),{})),patch('notebook_support.matrix_v5_1.entry.configure_runtime'),patch('notebook_support.matrix_v5_1.entry.runtime_identity',return_value={'CPU':True}),patch('notebook_support.matrix_v5_1.entry.select_parent',side_effect=AssertionError('must not import')):
            with self.assertRaisesRegex(ValueError,'runtime differs'):prepare(dict(job='CADIC_seed0',phase='7.2',started=time.monotonic(),session_hours=9))
    def test_failure_exports_before_raising(self):
        with tempfile.TemporaryDirectory() as t:
            work=Path(t)/'work';out=Path(t)/'failed.zip'
            with patch('notebook_support.matrix_v5_1.entry.prepare',side_effect=ValueError('fixture failure')):
                with self.assertRaises(RuntimeError):run_notebook(dict(work=str(work),output=str(out),job='CADIC_seed0',started=time.monotonic()))
            with zipfile.ZipFile(out) as z:self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
    def test_pause_exports_own_job_and_dirty_work_refused(self):
        with tempfile.TemporaryDirectory() as t:
            work=Path(t)/'work';out=Path(t)/'paused.zip';c=dict(work=str(work),output=str(out),job='NESTED_seed0',started=time.monotonic())
            with patch('notebook_support.matrix_v5_1.entry.prepare',side_effect=engine.Pause('budget')):run_notebook(c)
            with zipfile.ZipFile(out) as z:
                h=json.loads(z.read('handoff.json'));self.assertEqual(h['job'],'NESTED_seed0');self.assertEqual(h['status'],'paused_at_atomic_boundary')
            with self.assertRaisesRegex(ValueError,'fresh kernel'):run_notebook(c)
    def test_test_only_staging_excludes_train_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            archive=Path(t)/'dataset.tar.gz'
            with tarfile.open(archive,'w:gz') as z:
                for n in ['bottle/train/good/000.png','bottle/test/good/000.png','bottle/ground_truth/bad/000_mask.png']:
                    m=tarfile.TarInfo(n);m.size=1;z.addfile(m,io.BytesIO(b'x'))
            dest=Path(t)/'assets';self.assertEqual(common.stage_tar(archive,dest,evaluation=True),2);self.assertFalse((dest/'bottle/train').exists())
    def test_both_readonly_matrices_preserve_training_final_cells_and_resume(self):
        from models.hope_cad_v5_1.session import SupportSession
        from models.hope_cad_v5_1.learner import NestedUtilityLearner
        for mode in ('NESTED','CADIC'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                root=Path(t);rows,tasks,train=bench.fixture();factory=lambda:bench.actor(mode)
                engine.train(root,rows,tasks,dict(job=mode+'_seed0'),train,factory)
                test=[dict(relative_path=f'{task}/test/{label}.png',task_id=i,label=label) for i,task in enumerate(tasks) for label in (0,1)]
                p=bench.EvalProvider();engine.evaluate(root,tasks,test,factory,dict(provider=p,test_digest='fixture'))
                before=files(root);owner=object.__new__(common.CADICOwner);owner.folder=root;owner.tasks=tasks;owner.identities=dict(job=json.loads((root/'job_lock.json').read_text()))
                p=bench.EvalProvider();p.stop=3
                with patch.object(engine,'train',side_effect=AssertionError('no train')),patch.object(SupportSession,'observe',side_effect=AssertionError('no observe')),patch.object(NestedUtilityLearner,'observe',side_effect=AssertionError('no optimizer')):
                    with self.assertRaises(engine.Pause):evaluate_matrix(owner,test,factory,p,'fixture',None)
                    p.stop=None;matrix=evaluate_matrix(owner,test,factory,p,'fixture',None)
                    self.assertEqual(len(list((root/'cells').glob('*/completion.json'))),6)
                    count=p.calls;evaluate_matrix(owner,test,factory,p,'fixture',None);self.assertEqual(count,p.calls)
                for n,h in before.items():self.assertEqual(sha(root/n),h,n)
                for key in ('i_auroc','p_ap'):
                    self.assertEqual(len(matrix[key]['matrix']),3);self.assertIsNotNone(matrix[key]['bwt'])
    def test_migrated_NESTED_uses_legacy_task00_and_immutable_final_row(self):
        from models.hope_cad_v5_1.session import SupportSession
        with tempfile.TemporaryDirectory() as t:
            f=migrate.Fixture(t);f.publish(f.bridge());e=f.execution();e.train(f.provider(e.folder),migrate.actor)
            p=migrate.EvalProvider();e.evaluate(migrate.eval_rows(),migrate.actor,dict(provider=p,test_digest='fixture'))
            before=files(e.folder);legacy=sha(e.root/'legacy/runs/NESTED_seed0/states/task_00.pt')
            with patch.object(engine,'train',side_effect=AssertionError('no training')),patch.object(SupportSession,'observe',side_effect=AssertionError('no learning')):
                evaluate_matrix(e,migrate.eval_rows(),migrate.actor,p,'fixture',None)
            self.assertEqual(len(list((e.folder/'cells').glob('*/completion.json'))),120)
            self.assertEqual(sha(e.root/'legacy/runs/NESTED_seed0/states/task_00.pt'),legacy)
            for n,h in before.items():self.assertEqual(sha(e.folder/n),h,n)
            self.assertFalse((e.folder/'states/task_00.pt').exists())
    def test_preflight_score_is_readonly_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows,tasks,p=bench.fixture();engine.train(root,rows,tasks,dict(job='fixture'),p,bench.actor)
            owner=object.__new__(common.CADICOwner);owner.folder=root;owner.tasks=tasks;owner.identities=dict(job=json.loads((root/'job_lock.json').read_text()))
            tests=[dict(relative_path=f'{task}/test/{label}.png',task_id=i,label=label) for i,task in enumerate(tasks) for label in (0,1)]
            proof=readonly_preflight(owner,tests,bench.actor,bench.EvalProvider());self.assertEqual(proof['scored_images'],2)
            original=bench.actor
            def bad():
                a=original();score=a.score
                def mutate(x):v=score(x);a.cursor+=1;return v
                a.score=mutate;return a
            with self.assertRaisesRegex(ValueError,'Parity scalar mismatch'):readonly_preflight(owner,tests,bad,bench.EvalProvider())
    def test_unknown_job_and_phase_never_run_worker(self):
        for j in ['M_seed0','NESTED_seed1','../CADIC_seed0',None]:
            with self.assertRaises(ValueError):common.job_name(j)

if __name__=='__main__':unittest.main()
