import copy,json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1,CADICPatchCoresetConfig
from models.cadic_paperfaithful.cache import CachedPairCoreset
from models.hope_cad_v5_1.session import SupportSession
from training.support_nn_write_v5_1.candidate import ORIGINAL_NEAREST,activation,nearest_fixed_shape_writes
from training.support_nn_write_v5_1.diagnostics import recompute,Segments
from training.support_nn_write_v5_1.replay import replay
from training.support_nn_write_v5_1.runner import gates,run,verify_complete
from training.support_nn_write_v5_1.contract import contract
from training.support_throughput_v5_1.replay import assert_equal,state_digest
from training.runtime import atomic_json,atomic_torch_save
from training.support_throughput_v5_1.contract import sha

def actor(mode='NESTED'):
    torch.set_num_threads(2)
    return SupportSession(mode,seed=9,
        bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),
        model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))

def stream(n=24):
    gen=torch.Generator().manual_seed(54)
    return [torch.randn(1,4,4,generator=gen)*.02+1.7**i for i in range(n)]

def bit_equal(a,b):
    if torch.is_tensor(a):
        return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
    return all(bit_equal(x,y) for x,y in zip(a,b))

class NNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def core(self,**kw):return CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=10,dim=4,chunk_size=4,query_chunk_size=4,pair_chunk_size=4,**kw))

    def test_random_partial_tiles_and_dtype_geometry_exact(self):
        gen=torch.Generator().manual_seed(1);b=self.core()
        for nq,nb in [(0,1),(1,1),(4,4),(7,9),(17,31)]:
            q=torch.randn(nq,4,generator=gen);bank=torch.randn(nb,4,generator=gen)
            self.assertTrue(bit_equal(ORIGINAL_NEAREST(b,q,bank),nearest_fixed_shape_writes(b,q,bank)))

    def test_zero_ties_cross_tile_index_and_signed_zero(self):
        b=self.core();q=torch.zeros(5,4);bank=torch.zeros(9,4);bank[0]=1
        a=ORIGINAL_NEAREST(b,q,bank);c=nearest_fixed_shape_writes(b,q,bank)
        self.assertTrue(bit_equal(a,c));self.assertEqual(c[1].tolist(),[1]*5)
        bank[0]=0;bank[1,0]=-0.;self.assertTrue(bit_equal(ORIGINAL_NEAREST(b,q,bank),nearest_fixed_shape_writes(b,q,bank)))

    def test_NaN_inf_distance_strict_mask_and_first_index(self):
        b=self.core();q=torch.zeros(1,4);bank=torch.zeros(6,4)
        for data in [[float('nan'),1,2,3,0,0],[float('inf')]*6,[1,1,1,1,1,1]]:
            def distance(x,y):return torch.tensor([data[:4] if len(y)==4 else data[4:]],dtype=torch.float32)
            with patch('models.cadic_paperfaithful.cadic_patch_coreset.euclidean_distance_mm',side_effect=distance),patch('training.support_nn_write_v5_1.candidate.euclidean_distance_mm',side_effect=distance):
                self.assertTrue(bit_equal(ORIGINAL_NEAREST(b,q,bank),nearest_fixed_shape_writes(b,q,bank)))

    def test_same_distance_tile_shapes_call_order_and_metadata(self):
        b=self.core();q=torch.arange(28,dtype=torch.float32).reshape(7,4);bank=torch.arange(36,dtype=torch.float32).reshape(9,4)
        from models.cadic_paperfaithful.cadic_patch_coreset import euclidean_distance_mm
        calls=[]
        def distance(x,y):calls.append((x.clone(),y.clone()));return euclidean_distance_mm(x,y)
        with patch('models.cadic_paperfaithful.cadic_patch_coreset.euclidean_distance_mm',side_effect=distance):a=ORIGINAL_NEAREST(b,q,bank)
        first=calls[:];calls.clear()
        with patch('training.support_nn_write_v5_1.candidate.euclidean_distance_mm',side_effect=distance):c=nearest_fixed_shape_writes(b,q,bank)
        self.assertTrue(bit_equal(a,c));assert_equal(first,calls)

    def test_full_784_by_10000_raw768_geometry(self):
        gen=torch.Generator().manual_seed(46);q=torch.randn(784,768,generator=gen);bank=torch.randn(10000,768,generator=gen)
        b=CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=10000,chunk_size=256,query_chunk_size=256,pair_chunk_size=256))
        self.assertTrue(bit_equal(ORIGINAL_NEAREST(b,q,bank),nearest_fixed_shape_writes(b,q,bank)))

    def test_invalid_geometry_matches_reference(self):
        b=self.core()
        for q,k in [(torch.zeros(1,2,4),torch.zeros(2,4)),(torch.zeros(1,4),torch.zeros(0,4)),(torch.zeros(1,3),torch.zeros(2,4))]:
            for fn in [ORIGINAL_NEAREST,nearest_fixed_shape_writes]:
                with self.assertRaises(ValueError):fn(b,q,k)

    def test_scope_exception_nested_unknown_operator_restore(self):
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with activation(True):
                self.assertIs(CADICPatchCoresetV1._nearest,nearest_fixed_shape_writes)
                with self.assertRaises(RuntimeError):
                    with activation(False):pass
                raise RuntimeError('injected')
        self.assertIs(CADICPatchCoresetV1._nearest,ORIGINAL_NEAREST)

    def test_cache_trajectory_fork_noop_load_and_Eq9(self):
        ref=actor('CADIC');opt=actor('CADIC');xs=stream(16)
        for i,x in enumerate(xs):
            a=ref.observe(x,str(i))
            with activation(True):b=opt.observe(x,str(i))
            assert_equal(a,b);assert_equal(ref.state_dict(),opt.state_dict())
        before=opt.state_dict()
        for q in [xs[0],xs[-1]]:
            a=ref.score(q)
            with activation(True):b=opt.score(q)
            self.assertTrue(bit_equal(a,b))
        assert_equal(before,opt.state_dict())
        for core in [ref.bank,opt.bank]:
            core._closest_pair();core.features[0]=xs[-1].reshape(-1,4)[0]
        with activation(True):o=opt.bank._closest_pair()
        r=ref.bank._closest_pair();assert_equal(r,o)
        fork=opt.bank.fork();fork.load_state_dict(opt.bank.state_dict())
        with activation(True):assert_equal(opt.bank._nearest(xs[0].reshape(-1,4),opt.bank.features),fork._nearest(xs[0].reshape(-1,4),fork.features))
        z=actor();w=actor()
        for i in range(5):
            a=z.observe(torch.zeros(1,4,4),str(i))
            with activation(True):b=w.observe(torch.zeros(1,4,4),str(i))
            assert_equal(a,b)

    def test_full_learning_capture_actual_meta_gradients_and_resume_exact(self):
        xs=stream();rows=[dict(relative_path=str(i)) for i in range(len(xs))]
        r=actor()
        for i in range(4):r.observe(xs[i],str(i))
        o=actor();o.load_state_dict(r.state_dict())
        class Provider:
            job_digest='fixture';prefix_chain='prefix'
            def get(self,row):return xs[int(row['relative_path'])]
        rm,rs,rc=replay(r,Provider(),rows,self.root/'R',deadline=float('inf'),attempt=lambda i:None,device='cpu',start=4,end=20,capture_events=(10,18))
        with activation(True):om,os,oc=replay(o,Provider(),rows,self.root/'O',deadline=float('inf'),attempt=lambda i:None,device='cpu',start=4,end=20,capture_events=(10,18))
        assert_equal(rs,os);self.assertEqual(rm['trace_digest'],om['trace_digest']);self.assertEqual(om['capture_intervals'],4)
        for i in [10,18]:
            assert_equal(rc[i],oc[i]);a=recompute(rc[i],actor,'cpu')
            with activation(True):b=recompute(oc[i],actor,'cpu')
            assert_equal(a,b);self.assertTrue(a['raw_slow_gradients']);self.assertEqual(len(a['support']),4)
        restored=actor();restored.load_state_dict(os)
        with activation(True):assert_equal(restored.observe(xs[20],'20'),o.observe(xs[20],'20'))

    def test_candidate_failure_atomic_rollback(self):
        a=actor();xs=stream(12)
        for i,x in enumerate(xs[:11]):a.observe(x,str(i))
        state=a.state_dict()
        original=a.learner.observe
        def fail(record):original(record);raise RuntimeError('after learner')
        with activation(True),patch.object(a.learner,'observe',side_effect=fail):
            with self.assertRaises(RuntimeError):a.observe(xs[11],'11')
        assert_equal(state,a.state_dict())

    def test_segments_exclusion_is_outside_primary(self):
        clock=iter([0.,2.,5.,8.]).__next__;s=Segments('cpu',clock=clock)
        s.start();self.assertEqual(s.exclude(lambda:42),42);s.finish()
        self.assertEqual(s.pipeline_seconds,5.);self.assertEqual(s.excluded_seconds,3.);self.assertEqual(s.exclusions,1)

    def values(self,times):
        return {n:dict(pipeline_seconds_excluding_diagnostics=t,final_state_digest='same',gradient_digest='same',gpu=dict(peak_allocated_bytes=100)) for n,t in times.items()}

    def test_speed_gate_does_not_promote_close_noisy_or_slower_times(self):
        cfg=contract()
        fast=gates(self.values(dict(R1=100,O1=60,O2=61,R2=101)),cfg)
        self.assertTrue(fast['all_engineering_gates_passed']);self.assertFalse(fast['benchmark_adoption'])
        for times in [dict(R1=100,O1=101,O2=100,R2=101),dict(R1=100,O1=99,O2=99,R2=102)]:
            self.assertFalse(gates(self.values(times),cfg)['speed_gate_passed'])
        v=self.values(dict(R1=100,O1=60,O2=61,R2=101));v['O1']['gpu']['peak_allocated_bytes']=2**30
        self.assertFalse(gates(v,cfg)['memory_gate_passed'])

    def test_nonfinite_or_mismatched_gradient_gate_rejected(self):
        for kind in ['time','gradient']:
            v=self.values(dict(R1=100,O1=60,O2=61,R2=101))
            if kind=='time':v['O1']['pipeline_seconds_excluding_diagnostics']=float('nan')
            else:v['O1']['gradient_digest']='changed'
            with self.assertRaises(ValueError):gates(v,contract())

    def test_budget_partial_never_retried_and64cap(self):
        c=dict(work=str(self.root),prepared={},started=100.,session_utc='fixture');calls=[]
        def process(*a,**k):calls.append(a);return 0
        self.assertEqual(run(c,process=process,now=lambda:101.),'validation_incomplete_budget')
        self.assertEqual(run(c,process=process,now=lambda:102.),'validation_incomplete_budget');self.assertEqual(len(calls),1)
        ledger=json.loads((self.root/'ledger.json').read_text());self.assertEqual(ledger['reserved_transitions'],16)
        ledger['reserved_transitions']=65;atomic_json(ledger,self.root/'ledger.json')
        with self.assertRaises(ValueError):run(c,process=process,now=lambda:103.)

    def test_failure_always_exports_evidence(self):
        from notebook_support.support_nn_write_v5_1.entry import run_notebook
        c=dict(work=str(self.root/'work'),output=str(self.root/'out.zip'),started=0.,phase='validate')
        with patch('notebook_support.support_nn_write_v5_1.entry.prepare',side_effect=ValueError('wrong input')):
            with self.assertRaises(RuntimeError):run_notebook(c)
        from training.support_throughput_v5_1.contract import Evidence
        with Evidence(self.root/'out.zip') as e:
            e.verify_all();self.assertEqual(e.json('handoff.json')['status'],'validation_failed_preserve_evidence')

    def test_four_pass_transport_seals_resume_and_tamper(self):
        c=dict(work=str(self.root),prepared={},started=100.,session_utc='fixture');calls=[]
        def process(cmd,*a,**kw):
            config=json.loads(Path(cmd[-1]).read_text());n=config['pass'];calls.append(n)
            p=self.root/'passes'/n;p.mkdir(parents=True)
            if n=='R1':
                atomic_torch_save({'fixture':torch.ones(1)},self.root/'reference_final_state.pt')
                atomic_torch_save({'fixture':torch.ones(1)},self.root/'reference_gradients.pt')
            value=dict(status='complete',start_cursor=179,end_cursor=195,completed_images=16,attempted_event_ids=list(range(179,195)),
                capture_intervals=4,finalized=False,global_rng_unchanged=True,role=contract()['roles'][n],
                final_state_digest=contract()['G1_final_state_digest'],gradient_digest='fixture',pipeline_seconds_excluding_diagnostics=100. if n.startswith('R') else 60.,gpu=dict(peak_allocated_bytes=100),
                parity=dict(parent_events_exact=True,feature_receipts_exact=True,full_final_state_exact=True,G1_endpoint_exact=True,raw_actual_meta_gradients_and_steps_exact=True,gradient_capture_events=[181,189]))
            atomic_json(value,p/'measurement.json');atomic_json(dict(attempted_event_ids=list(range(179,195))),p/'attempts.json')
            atomic_json(dict(schema='nn_write_pass_completion_v1',pass_id=n,role=value['role'],files={f:sha(p/f) for f in ['measurement.json','attempts.json']},
                reference_final_sha256=sha(self.root/'reference_final_state.pt'),reference_gradients_sha256=sha(self.root/'reference_gradients.pt'),final_state_digest=value['final_state_digest'],gradient_digest='fixture'),p/'complete.json')
            return 0
        self.assertEqual(run(c,process=process,now=lambda:101.),'validation_complete_pending_review')
        self.assertEqual(calls,['R1','O1','O2','R2']);self.assertEqual(json.loads((self.root/'ledger.json').read_text())['reserved_transitions'],64)
        self.assertTrue(json.loads((self.root/'summary.json').read_text())['all_engineering_gates_passed'])
        def forbidden(*a,**kw):raise AssertionError('Must skip complete passes')
        self.assertEqual(run(dict(c,started=200.),process=forbidden,now=lambda:201.),'validation_complete_pending_review')
        (self.root/'passes/O1/measurement.json').write_text('{}')
        with self.assertRaises(ValueError):verify_complete(self.root,'O1')

    def test_bootstrap_rejects_profiler_as_resume_before_assets_or_GPU(self):
        from notebook_support.support_nn_write_v5_1.bootstrap import prepare
        with patch('notebook_support.support_nn_write_v5_1.bootstrap.verify_source',return_value={}),patch('notebook_support.support_nn_write_v5_1.bootstrap.select_evidence',return_value=self.root):
            with self.assertRaisesRegex(ValueError,'Remove completed profiler'):prepare(dict(inputs=str(self.root),work=str(self.root)))

if __name__=='__main__':unittest.main()
