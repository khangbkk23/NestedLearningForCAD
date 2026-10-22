import copy,hashlib,io,json,shutil,sys,tarfile,tempfile,time,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from models.hope_cad_v5_1.session import SupportSession
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from training.runtime import atomic_json
from training.support_benchmark_v5_1 import engine
from training.support_benchmark_v5_1.contract import lock
from training.support_throughput_v5_1.contract import Evidence,digest,sha
from training.support_throughput_v5_1.replay import assert_equal,state_digest
from training.support_migration_v5_1.contract import make_identity,prerequisites
from training.support_migration_v5_1.data import ArrivalProvider
from training.support_migration_v5_1.bridge import shadow,Incomplete
from training.support_migration_v5_1.lineage import publish,Execution,ORIGINAL_TASK_STATE
from notebook_support.common.evidence import export_archive

def actor():
    torch.set_num_threads(2)
    return SupportSession('NESTED',seed=11,bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))

class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.parent=self.root/'parent';self.parent.mkdir()
        self.rows=[dict(relative_path=f't{i//4:02d}/train/good/{i:03d}.png',task_id=i//4,task_name=f't{i//4:02d}',split='train',sha256='fixture') for i in range(60)]
        self.tasks=[f't{i:02d}' for i in range(15)];self.runtime={'device':'CPU fixture, not canonical T4'}
        self.oldmap={'files':{'original.py':'a'*64}};self.newmap={'files':{'original.py':'a'*64,'new.py':'b'*64}}
        self.source=dict(source_digest=digest(self.newmap['files']),files=2)
        suite=lock(self.parent/'suite_lock.json',dict(schema='fixture_suite',source_digest=digest(self.oldmap['files']),design_digest='method',study_digest='study',runtime=self.runtime))
        atomic_json(dict(online=self.rows,fixture=True),self.parent/'manifest_train.json');atomic_json(dict(fixture=True,official_test_for_learning=False),self.parent/'data_rights.json')
        identity=dict(job='NESTED_seed0',mode='NESTED',seed=0,fresh_initialization=True,suite_digest=suite['digest'],rows_digest=digest(self.rows))
        p=self.provider(self.parent/'runs/NESTED_seed0',stop=6)
        with unittest.TestCase().assertRaises(engine.Pause):engine.train(self.parent/'runs/NESTED_seed0',self.rows,self.tasks,identity,p,actor,checkpoint_every=1)
        self.active=torch.load(self.parent/'runs/NESTED_seed0/active.pt',weights_only=True)
        self.job=json.loads((self.parent/'runs/NESTED_seed0/job_lock.json').read_text())
        self.archive=self.root/'parent.zip';export_archive(self.parent,self.archive,schema='fixture_parent')
        with zipfile.ZipFile(self.archive) as z:self.manifest=json.loads(z.read('artifact_manifest.json'));mh=hashlib.sha256(z.read('artifact_manifest.json')).hexdigest()
        self.info=dict(rows=self.rows,suite=suite,job=self.job,manifest_sha256=mh)
        self.policy=dict(id='fixture_G4',start_cursor=6,end_cursor_exclusive=7,review_receipts_sha256={'G3':'review'},parent_archive_sha256=sha(self.archive),parent_active_sha256=sha(self.parent/'runs/NESTED_seed0/active.pt'))
        self.pins=dict(artifact_manifest_sha256=mh,suite_digest=suite['digest'],job_digest=self.job['digest'],archive_sha256=sha(self.archive),active_sha256=self.policy['parent_active_sha256'])
        self.identities=make_identity(self.info,self.source,self.runtime,self.policy)
        self.work=self.root/'work';self.work.mkdir()
    def provider(self,folder,stop=None):
        rows=self.rows;g=torch.Generator().manual_seed(80)
        xs=[torch.randn(1,4,4,generator=g)*.04+i*.15 for i in range(len(rows))]
        class P:
            def __init__(self):self.calls=[]
            def get(self,row):
                i=rows.index(row)
                if i==stop:raise engine.Pause('fixture interruption')
                self.calls.append(i)
                atomic_json(dict(row=row,fixture=True),Path(folder)/'feature_receipts'/(digest(row)+'.json'))
                return xs[i].clone()
        return P()
    def bridge(self,*,bad=None,deadline=None):
        a,b=actor(),actor();a.load_state_dict(self.active['session']);b.load_state_dict(self.active['session'])
        if bad:bad(b)
        p=ArrivalProvider(self.rows[6:7],self.provider(self.work))
        return shadow(a,b,p,self.rows,self.work/'shadow',start=6,end=7,chain=self.active['trace_digest'],deadline=deadline or time.monotonic()+100,device='cpu')
    def publish(self,result):
        with Evidence(self.archive) as e:
            return publish(self.work,e,self.info,self.identities,result,self.source,self.policy,self.active,source_map=self.newmap,base_source_map=self.oldmap,pins=self.pins)
    def execution(self):return Execution(self.work/'benchmark',self.source,self.runtime,self.policy,self.pins)

class EvalProvider:
    def __init__(self):self.calls=0;self.stop=None
    def get(self,row):
        if self.calls==self.stop:raise engine.Pause('evaluation interrupted')
        self.calls+=1;raw=torch.arange(16,dtype=torch.float32).reshape(1,4,4)*.03+row['label']+.1
        mask=np.zeros((2,2),bool);mask[0,0]=bool(row['label']);return raw,mask

def eval_rows():return [dict(relative_path=f't{j:02d}/test/{i}.png',task_id=j,label=i,mask_path=None) for j in range(15) for i in [0,1]]

class MigrationTests(unittest.TestCase):
    def test_prerequisite_reviews_and_source_gate(self):
        self.assertEqual(set(prerequisites()),{'G1','G2','G2local','G3','G3audit'})
    def test_identity_changes_execution_without_changing_science(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);v=f.identities
            self.assertNotEqual(v['job']['digest'],f.job['digest']);self.assertFalse(v['job']['identity']['fresh_initialization'])
            self.assertEqual(v['job']['identity']['rows'],f.rows);self.assertEqual(v['suite']['identity']['design_digest'],'method')
            with self.assertRaises(ValueError):make_identity(f.info,f.source,dict(device='other'),f.policy)
    def test_lazy_tar_reads_only_current_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=[dict(relative_path=f'a/train/good/{i}.png',split='train') for i in range(2)]
            tar=root/'data.tar'
            with tarfile.open(tar,'w') as z:
                for r in rows:
                    b=b'normal';m=tarfile.TarInfo(r['relative_path']);m.size=len(b);z.addfile(m,io.BytesIO(b))
            class P:
                def get(self,row):return (root/'assets'/row['relative_path']).read_bytes()
            p=ArrivalProvider(rows,P(),archive=tar,assets=root/'assets')
            try:
                self.assertFalse((root/'assets').exists());self.assertEqual(p.get(rows[0]),b'normal')
                self.assertFalse((root/'assets'/rows[1]['relative_path']).exists())
                with self.assertRaises(ValueError):p.get(rows[0])
                self.assertEqual(p.get(rows[1]),b'normal')
            finally:p.close()
    def test_future_row_and_GT_rejected_before_provider(self):
        class P:
            def get(self,row):raise AssertionError('must not read')
        rows=[dict(relative_path=f'a/train/good/{i}.png',split='train') for i in range(2)]
        p=ArrivalProvider(rows,P())
        with self.assertRaises(ValueError):p.get(rows[1])
        with self.assertRaises(ValueError):p.get(dict(relative_path='a/test/x.png',split='test'))
    def test_reverse_compressed_tar_metadata_does_not_preextract_future_pixels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=[dict(relative_path=f'a/train/good/{i}.png',split='train') for i in range(3)];tar=root/'reverse.tar.gz'
            with tarfile.open(tar,'w:gz') as z:
                for r in reversed(rows):
                    b=r['relative_path'].encode();m=tarfile.TarInfo('prefix/'+r['relative_path']);m.size=len(b);z.addfile(m,io.BytesIO(b))
            class P:
                def get(self,row):return (root/'assets'/row['relative_path']).read_bytes()
            p=ArrivalProvider(rows,P(),archive=tar,assets=root/'assets')
            try:
                for i,row in enumerate(rows):
                    self.assertEqual(p.get(row),row['relative_path'].encode())
                    for future in rows[i+1:]:self.assertFalse((root/'assets'/future['relative_path']).exists())
            finally:p.close()
    def test_bridge_exact_publication_keeps_parent_and_pending(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);before=sha(f.archive);result=f.bridge();receipt=f.publish(result);e=f.execution()
            self.assertEqual(sha(f.archive),before);self.assertEqual(e.receipt,receipt)
            state=torch.load(e.root/'bridge_checkpoint.pt',weights_only=True)
            self.assertAlmostEqual(state['elapsed_seconds'],f.active['elapsed_seconds']+result['ledger']['times']['provider']+result['ledger']['times']['candidate'],places=9)
            self.assertEqual(state['cursor'],7);self.assertIsNotNone(state['session']['pending']);self.assertFalse(state['session']['finalized'])
            self.assertEqual((e.root/'legacy/runs/NESTED_seed0/states/task_00.pt').read_bytes(),(f.parent/'runs/NESTED_seed0/states/task_00.pt').read_bytes())
    def test_mismatch_stops_before_next_arrival_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t)
            def bad(b):
                original=b.observe
                def observe(*a,**kw):v=original(*a,**kw);v['action']='incorrect';return v
                b.observe=observe
            with self.assertRaises(ValueError):f.bridge(bad=bad)
            self.assertFalse((f.work/'benchmark').exists())
            ledger=json.loads((f.work/'shadow/attempts.json').read_text());self.assertEqual(ledger['arrivals'],[6]);self.assertEqual(ledger['completed_pairs'],[])
    def test_worker_load_checks_fresh_hash_before_restoring_learned_state(self):
        from training.support_migration_v5_1.worker import load_owner
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);initial=json.loads((f.parent/'runs/NESTED_seed0/initialization.json').read_text())['identity']
            for opt in [False,True]:
                loaded=load_owner(actor,f.active['session'],initial,opt)
                self.assertNotEqual(loaded.initialization_hashes(),initial)
                assert_equal(loaded.state_dict(),f.active['session'])
            with self.assertRaises(ValueError):load_owner(actor,f.active['session'],dict(initial,theta='unknown'),True)
    def test_shared_raw_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t)
            def bad(b):
                original=b.observe
                def observe(raw,*a):v=original(raw,*a);raw.add_(1);return v
                b.observe=observe
            with self.assertRaisesRegex(ValueError,'mutated shared raw'):f.bridge(bad=bad)
    def test_budget_and_partial_cannot_publish(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t)
            with self.assertRaises(Incomplete):f.bridge(deadline=-1)
            self.assertFalse((f.work/'benchmark').exists())
            ledger=json.loads((f.work/'shadow/attempts.json').read_text());self.assertEqual(ledger['arrivals'],[])
    def test_partial_publication_rename_failure_does_not_create_accepted_root(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);result=f.bridge()
            with patch('training.support_migration_v5_1.lineage.os.rename',side_effect=OSError('simulated crash')):
                with self.assertRaises(OSError):f.publish(result)
            self.assertFalse((f.work/'benchmark').exists());self.assertTrue((f.work/'.publication_pending').is_dir())
            with self.assertRaises(ValueError):f.publish(result)
    def test_unknown_source_runtime_and_parent_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);f.publish(f.bridge())
            for source,runtime,pins in [(dict(source_digest='wrong',files=2),f.runtime,f.pins),(f.source,{'device':'other'},f.pins),(f.source,f.runtime,dict(f.pins,job_digest='wrong'))]:
                with self.assertRaises(ValueError):Execution(f.work/'benchmark',source,runtime,f.policy,pins)
    def test_inherited_hash_and_checkpoint_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);f.publish(f.bridge());e=f.execution();p=e.root/'legacy/runs/NESTED_seed0/states/task_00.pt';p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):f.execution()
    def test_train_resume_exact_old_trajectory_and_new_task_ownership(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);f.publish(f.bridge());e=f.execution();p=f.provider(e.folder)
            a=e.train(p,actor);self.assertEqual(p.calls,list(range(7,60)))
            original=f.provider(f.parent/'runs/NESTED_seed0');identity={k:v for k,v in f.job['identity'].items() if k not in ('rows','task_order')}
            b=engine.train(f.parent/'runs/NESTED_seed0',f.rows,f.tasks,identity,original,actor)
            assert_equal(a.state_dict(),b.state_dict());self.assertEqual(e.task_state(e.folder,0,actor(),e.identities['job']['digest']),f.manifest['files']['runs/NESTED_seed0/states/task_00.pt'])
            task=torch.load(e.folder/'states/task_01.pt',weights_only=True);self.assertEqual(task['job_digest'],e.identities['job']['digest'])
            for i in range(1,15):
                new=torch.load(e.folder/'states'/f'task_{i:02d}.pt',weights_only=True)
                old=torch.load(f.parent/'runs/NESTED_seed0/states'/f'task_{i:02d}.pt',weights_only=True)
                assert_equal(new['session'],old['session'],'intermediate_taskstate')
            # A legacy wrapper copied under the new owner must not be silently re-labelled.
            for suffix in ('.pt','.json'):shutil.copy2(e.root/'legacy/runs/NESTED_seed0/states'/('task_00'+suffix),e.folder/'states'/('task_00'+suffix))
            with self.assertRaises(ValueError):ORIGINAL_TASK_STATE(e.folder,0,actor(),e.identities['job']['digest'])
    def test_active_receipt_interruption_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);f.publish(f.bridge());e=f.execution();(e.folder/'active.json').unlink()
            p=f.provider(e.folder,stop=10)
            with self.assertRaises(engine.Pause):e.train(p,actor)
            # Saved cursor8: replay9 is safe and immutable records must match.
            e=f.execution();e.train(f.provider(e.folder),actor)
            value=torch.load(e.folder/'active.pt',weights_only=True);value['job_digest']='unknown';torch.save(value,e.folder/'active.pt')
            with self.assertRaises(ValueError):e.train(f.provider(e.folder),actor)
    def test_final15_matrix120_cells_resume_no_retraining_or_metric_change(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);f.publish(f.bridge());e=f.execution();e.train(f.provider(e.folder),actor)
            ep=EvalProvider();ident=dict(provider=ep,test_digest='synthetic')
            first=e.evaluate(eval_rows(),actor,ident);self.assertEqual(ep.calls,30)
            saved={p.relative_to(e.folder).as_posix():p.read_bytes() for p in (e.folder/'cells').rglob('*') if p.is_file()}
            final=e.evaluate(eval_rows(),actor,ident,matrix=True);self.assertEqual(first,final);self.assertEqual(ep.calls,240)
            self.assertEqual(len(list((e.folder/'cells').glob('*/completion.json'))),120)
            for n,b in saved.items():self.assertEqual((e.folder/n).read_bytes(),b)
            self.assertEqual(json.loads((e.folder/'final_complete.json').read_text())['cell_count'],15)
            self.assertEqual(json.loads((e.folder/'matrix_complete.json').read_text())['cell_count'],120)
            e.evaluate(eval_rows(),actor,ident,matrix=True);self.assertEqual(ep.calls,240)
            self.assertIs(engine.task_state,ORIGINAL_TASK_STATE)
    def test_task_resolver_restored_after_failure_and_nested_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);f.publish(f.bridge());e=f.execution()
            with self.assertRaises(RuntimeError):
                with e.installed():raise RuntimeError('fixture')
            self.assertIs(engine.task_state,ORIGINAL_TASK_STATE)
            with e.installed():
                with self.assertRaises(ValueError):
                    with e.installed():pass
    def test_failure_exports_before_raise(self):
        from notebook_support.support_migration_v5_1.entry import run_notebook
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'evidence.zip'
            with patch('notebook_support.support_migration_v5_1.entry.prepare',side_effect=ValueError('fixture source mismatch')):
                with self.assertRaises(RuntimeError):run_notebook(dict(phase='bridge',work=str(root/'work'),output=str(out),started=time.monotonic()))
            with zipfile.ZipFile(out) as z:self.assertEqual(json.loads(z.read('handoff.json'))['status'],'bridge_failed_preserve_parent')

if __name__=='__main__':unittest.main()
