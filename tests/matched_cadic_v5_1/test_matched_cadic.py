import copy,hashlib,io,json,tempfile,time,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from models.hope_cad_v5_1.session import SupportSession
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from training.runtime import atomic_json
from training.support_benchmark_v5_1 import engine
from training.support_throughput_v5_1.replay import assert_equal
from training.support_throughput_v5_1.contract import digest,sha
from training.support_nn_write_v5_1.candidate import activation,ORIGINAL_NEAREST
from notebook_support.common.evidence import export_archive
from notebook_support.matched_cadic_v5_1 import common
from notebook_support.matched_cadic_v5_1.worker import preflight,run_job,MatchedFeatures
from notebook_support.matched_cadic_v5_1.entry import run_notebook,prepare

def actor():
    torch.set_num_threads(2)
    return SupportSession('CADIC',seed=0,bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,query_chunk_size=4,pair_chunk_size=4),model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))

def rows():return [dict(task_id=i//4,task_name=f't{i//4:02d}',relative_path=f't{i//4:02d}/train/good/{i:03d}.png',split='train',sha256='fixture') for i in range(60)]

class Provider:
    def __init__(self,stop=None):self.calls=[];self.stop=stop;g=torch.Generator().manual_seed(29);self.xs=[torch.randn(1,4,4,generator=g)*.05+i*.07 for i in range(60)]
    def get(self,row):
        i=rows().index(row)
        if i==self.stop:raise engine.Pause('fixture atomic interruption')
        self.calls.append(i);return self.xs[i].clone()

class EvalProvider:
    def __init__(self):self.calls=0
    def get(self,row):
        self.calls+=1;raw=torch.arange(16,dtype=torch.float32).reshape(1,4,4)*.04+row['label']+.1
        mask=np.zeros((2,2),bool);mask[0,0]=bool(row['label']);return raw,mask

def testrows():return [dict(relative_path=f't{i:02d}/test/{label}.png',task_id=i,label=label,mask_path=None) for i in range(15) for label in [0,1]]
def job_identity():return dict(job='CADIC_seed0',mode='CADIC',seed=0,suite_digest='fixture',rows_digest=digest(rows()),fresh_initialization=True,learned_state_imported=False)

class MatchedCADICTests(unittest.TestCase):
    def archive(self,root,status='paused_at_atomic_boundary',op=None):
        source=Path(root)/'source';source.mkdir();atomic_json(dict(status=status),source/'handoff.json');atomic_json(op or {'op':'fixture'},source/'operation_lock.json')
        (source/'payload').write_bytes(b'ownCADICfixture');out=Path(root)/'out.zip';export_archive(source,out,schema=common.SCHEMA);return source,out
    def test_fresh_control_has_no_learner_and_same_geometry(self):
        from training.support_benchmark_v5_1.contract import session_for
        c=session_for('CADIC_seed0');self.assertEqual(c.cursor,0);self.assertIsNone(c.learner);self.assertIsNone(c.projection);self.assertIsNone(c.pending)
        self.assertEqual((c.bank_config.budget,c.bank_config.dim,c.image_tokens,c.bank_config.image_neighbors),(10000,768,784,9))
        self.assertEqual(c.bank_config.query_chunk_size,256);self.assertFalse(c.finalized)
    def test_discarded_preflight_exact_readonly_and_no_initialization_reuse(self):
        with tempfile.TemporaryDirectory() as t:
            p=Provider();a=preflight(t,rows(),p,actor,'op',time.monotonic()+100)
            self.assertEqual(p.calls,[0,1]);self.assertEqual(a['images'],2);self.assertTrue(a['score_readonly']);self.assertFalse(a['official_test_accessed'])
            self.assertEqual(actor().cursor,0);self.assertEqual(actor().bank.features.shape[0],0)
            p=Provider();self.assertEqual(preflight(t,rows(),p,actor,'op',time.monotonic()+100),a);self.assertEqual(p.calls,[])
            with self.assertRaises(ValueError):preflight(t,rows(),p,actor,'changed',time.monotonic()+100)
    def test_preflight_reference_candidate_mismatch_cannot_publish(self):
        with tempfile.TemporaryDirectory() as t:
            original=SupportSession.observe
            def bad(a,*args,**kwargs):
                result=original(a,*args,**kwargs)
                if a.bank._nearest.__func__ is not ORIGINAL_NEAREST:result['action']='fake'
                return result
            with patch.object(SupportSession,'observe',bad):
                with self.assertRaises(ValueError):preflight(t,rows(),Provider(),actor,'op',time.monotonic()+100)
            self.assertFalse((Path(t)/'preflight.json').exists())
    def test_preflight_pause_preserves_no_benchmark_owner(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(engine.Pause):preflight(t,rows(),Provider(),actor,'op',-1)
            self.assertFalse((Path(t)/'preflight.json').exists());self.assertFalse((Path(t)/'runs').exists())
    def test_features_hash_and_normal_rights_checked_before_learning(self):
        with tempfile.TemporaryDirectory() as t:
            row=rows()[0];key=digest(row);p=Provider()
            class Base:
                def get(self,r):
                    value=p.get(r);atomic_json(dict(row=r,tensor_sha256='expected'),Path(t)/(digest(r)+'.json'));return value
            provider=MatchedFeatures(Base(),t,{key:'expected'})
            self.assertEqual(provider.get(row).shape,(1,4,4))
            provider.expected[key]='wrong'
            with self.assertRaisesRegex(ValueError,'differ from NESTED'):provider.get(row)
            with self.assertRaises(ValueError):provider.get(dict(row,split='test'))
            with self.assertRaises(ValueError):provider.get(rows()[1])
    def test_exact_resume_and_final15_all_taskstates_and_no_retraining(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'resumed';tasks=[f't{i:02d}' for i in range(15)];called=[];ep=EvalProvider()
            def evaluate():
                self.assertTrue((f/'train_complete.json').exists());called.append(True)
                with activation(True):engine.evaluate(f,tasks,testrows(),actor,dict(provider=ep,test_digest='fixture'),matrix=False)
            with self.assertRaises(engine.Pause):run_job(f,rows(),tasks,job_identity(),Provider(stop=10),actor,evaluate,time.monotonic()+100)
            self.assertFalse(called);self.assertEqual(torch.load(f/'active.pt',weights_only=True)['cursor'],8)
            p=Provider();self.assertEqual(run_job(f,rows(),tasks,job_identity(),p,actor,evaluate,time.monotonic()+100),'7.1_complete_pending_review')
            self.assertEqual(p.calls,list(range(8,60)));self.assertEqual(ep.calls,30)
            completed=torch.load(f/'active.pt',weights_only=True);self.assertTrue(completed['session']['finalized']);self.assertIsNone(completed['session']['learner'])
            self.assertEqual(len(list((f/'states').glob('*.pt'))),15);self.assertEqual(len(list((f/'cells').glob('*/completion.json'))),15)
            g=Path(t)/'uninterrupted'
            with activation(True):engine.train(g,rows(),tasks,job_identity(),Provider(),actor)
            reference=torch.load(g/'active.pt',weights_only=True);assert_equal(reference['session'],completed['session']);self.assertEqual(reference['trace_digest'],completed['trace_digest'])
            saved={p.relative_to(f).as_posix():p.read_bytes() for p in (f/'cells').rglob('*') if p.is_file()}
            p=Provider();run_job(f,rows(),tasks,job_identity(),p,actor,evaluate,time.monotonic()+100)
            self.assertEqual(p.calls,[]);self.assertEqual(ep.calls,30)
            for name,raw in saved.items():self.assertEqual((f/name).read_bytes(),raw)
    def test_other_owner_checkpoint_refused_before_evaluation(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t);tasks=[f't{i:02d}' for i in range(15)]
            with self.assertRaises(engine.Pause):run_job(f,rows(),tasks,job_identity(),Provider(stop=10),actor,lambda:self.fail('must not evaluate'),time.monotonic()+100)
            state=torch.load(f/'active.pt',weights_only=True);state['job_digest']='NESTEDowner';torch.save(state,f/'active.pt')
            with self.assertRaises(ValueError):run_job(f,rows(),tasks,job_identity(),Provider(),actor,lambda:self.fail('must not evaluate'),time.monotonic()+100)
    def test_resume_zip_directory_duplicates_and_wrong_schema(self):
        with tempfile.TemporaryDirectory() as t:
            source,out=self.archive(t)
            with zipfile.ZipFile(out) as z:(source/'artifact_manifest.json').write_bytes(z.read('artifact_manifest.json'))
            self.assertIn(common.select_resume(t),[source,out]);work=Path(t)/'work';work.mkdir()
            self.assertEqual(common.restore(out,work,{'op':'fixture'}),3)
            wrong=Path(t)/'wrong';wrong.mkdir();(wrong/'payload').write_bytes(b'otherjob')
            other=Path(t)/'other.zip';export_archive(wrong,other,schema='NESTEDwrong')
            with self.assertRaises(ValueError):common.select_resume(t,str(other))
    def test_resume_failed_ambiguous_and_source_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a=root/'a';a.mkdir();_,out=self.archive(a);work=root/'work';work.mkdir()
            with self.assertRaises(ValueError):common.restore(out,work,{'op':'changed'})
            b=root/'b';b.mkdir();self.archive(b,op={'op':'different'})
            with self.assertRaises(ValueError):common.select_resume(root)
            c=root/'c';c.mkdir();_,bad=self.archive(c,status='failed_preserve_evidence')
            with self.assertRaises(ValueError):common.select_resume(root,str(bad))
    def test_runtime_mismatch_stops_before_assets_or_worker(self):
        ref=dict(runtime={'device':'T4'},study_digest='study',design_digest='design',normal_rows_digest='rows',backbone_sha256='backbone')
        with patch('notebook_support.matched_cadic_v5_1.entry.verify_source',return_value=({},ref,{})),patch('notebook_support.matched_cadic_v5_1.entry.configure_runtime'),patch('notebook_support.matched_cadic_v5_1.entry.runtime_identity',return_value={'device':'CPU'}),patch('notebook_support.matched_cadic_v5_1.entry.stage_tar',side_effect=AssertionError('must not stage')):
            with self.assertRaisesRegex(ValueError,'Runtime differs'):prepare(dict(phase='7.1',job='CADIC_seed0',started=time.monotonic(),session_hours=9))
    def test_identity_binds_source_geometry_rows_runtime_and_job(self):
        ref=dict(runtime={'device':'T4'},study_digest='study',design_digest='design',normal_rows_digest='rows',backbone_sha256='backbone')
        suite,ji=common.identity({'source':'fixture'},ref,ref['runtime']);self.assertEqual(suite['digest'],digest(suite['identity']));self.assertTrue(ji['fresh_initialization']);self.assertFalse(ji['learned_state_imported']);self.assertEqual(ji['job'],'CADIC_seed0')
        other,_=common.identity({'source':'other'},ref,ref['runtime']);self.assertNotEqual(suite['digest'],other['digest'])
        with self.assertRaises(ValueError):common.identity({},ref,{'device':'5090'})
    def test_bootstrap_failure_exports_before_raise(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'failed.zip'
            with patch('notebook_support.matched_cadic_v5_1.entry.prepare',side_effect=ValueError('fixture error')):
                with self.assertRaises(RuntimeError):run_notebook(dict(work=str(root/'work'),output=str(out),started=time.monotonic()))
            with zipfile.ZipFile(out) as z:self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
    def test_pause_complete_and_worker_failure_export_correct_status(self):
        for state,code,success in [('paused_at_atomic_boundary',0,True),('7.1_complete_pending_review',0,True),('failed_preserve_evidence',1,False)]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);work=root/'work';out=root/'out.zip'
                def prep(c):
                    session=work/'session_history/0001';session.mkdir(parents=True);atomic_json(dict(status=state),session/'worker_status.json')
                    if state.startswith('7.1'):
                        b=work/'benchmark/runs/CADIC_seed0';atomic_json(dict(digest='job'),b/'job_lock.json');atomic_json(dict(job_digest='job',images=3599,finalized=True),b/'train_complete.json');atomic_json(dict(job_digest='job',cell_count=15),b/'final_complete.json')
                    return dict(session=str(session))
                with patch('notebook_support.matched_cadic_v5_1.entry.prepare',side_effect=prep),patch('notebook_support.matched_cadic_v5_1.entry.stream_process',return_value=code):
                    cfg=dict(work=str(work),output=str(out),started=time.monotonic())
                    if success:run_notebook(cfg)
                    else:
                        with self.assertRaises(RuntimeError):run_notebook(cfg)
                with zipfile.ZipFile(out) as z:self.assertEqual(json.loads(z.read('handoff.json'))['status'],state)

if __name__=='__main__':unittest.main()
