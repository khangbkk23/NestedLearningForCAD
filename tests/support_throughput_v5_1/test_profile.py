import copy, io, json, tarfile, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
import torch
from models.hope_cad_v5_1.session import SupportSession
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from models.hope_cad_v5_1.contract import digest
from training.runtime import atomic_json,atomic_torch_save
from training.support_throughput_v5_1.contract import Evidence, select_evidence, safe, sha, SCHEMA, verify_source
from training.support_throughput_v5_1.replay import bounded_replay, assert_equal, state_digest
from training.support_throughput_v5_1.instrumentation import Profiler
from training.support_throughput_v5_1.runner import run,verify_complete
from notebook_support.support_throughput_v5_1.bootstrap import extract_window, verify_window
from notebook_support.common.evidence import export_archive

def actor():
    torch.set_num_threads(2)
    return SupportSession('NESTED',seed=9,
        bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),
        model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))

def images(n=24):
    gen=torch.Generator().manual_seed(54)
    return [torch.randn(1,4,4,generator=gen)*.02+1.7**i for i in range(n)]

class Provider:
    job_digest='fixture'
    def __init__(self,x):self.x=x;self.observed=[]
    def get(self,row):
        self.observed.append(int(row['relative_path']));return self.x[int(row['relative_path'])]

class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_bounded_replay_has_exact_uninstrumented_trajectory_partial_and_clocks(self):
        xs=images();rows=[dict(relative_path=str(i)) for i in range(len(xs))]
        a=actor()
        for i in range(4):a.observe(xs[i],str(i))
        saved=a.state_dict();b=actor();b.load_state_dict(saved)
        attempts=[];pa=Provider(xs);pb=Provider(xs)
        m,s=bounded_replay(a,pa,rows,self.root/'U1',start=4,end=20,chain=digest('prefix'),deadline=float('inf'),attempt=attempts.append)
        with Profiler() as prof:
            q,t=bounded_replay(b,pb,rows,self.root/'P1',start=4,end=20,chain=digest('prefix'),deadline=float('inf'),attempt=lambda i:None,profiler=prof)
        assert_equal(s,t);self.assertEqual(m['trace_digest'],q['trace_digest'])
        self.assertEqual(attempts,list(range(4,20)));self.assertEqual(pa.observed,pb.observed)
        self.assertFalse(b.finalized);self.assertIsNotNone(b.pending)
        self.assertGreater(b.learner.slow_updates,0);self.assertTrue(b.learner.records)
        self.assertTrue(m['global_rng_unchanged'])
        self.assertGreater(q['profile']['counters']['bank_hash_calls'],0)
        self.assertGreater(q['profile']['counters']['distance_blocks_nearest'],0)
        self.assertGreater(q['profile']['counters']['pair_tiles_requiring_refresh'],0)
        active=torch.load(self.root/'P1/active.pt',weights_only=True)
        self.assertEqual(active['cursor'],16)

    def test_no_finalize_on_window_end_or_row_truncation(self):
        a=actor();xs=images(3);rows=[dict(relative_path=str(i)) for i in range(3)]
        with self.assertRaises(ValueError):bounded_replay(a,Provider(xs),rows,self.root,start=0,end=3,chain='x',deadline=float('inf'),attempt=lambda i:None)
        self.assertEqual(a.cursor,0);self.assertFalse(a.finalized)

    def test_deadline_records_incomplete_without_new_arrivals(self):
        a=actor();xs=images(4);rows=[dict(relative_path=str(i)) for i in range(4)];p=Provider(xs);attempts=[]
        m,_=bounded_replay(a,p,rows,self.root,start=0,end=2,chain='x',deadline=-1,attempt=attempts.append)
        self.assertEqual(m['status'],'incomplete');self.assertFalse(attempts);self.assertFalse(p.observed)

    def test_exception_restores_wrapped_methods_and_actor_transaction(self):
        a=actor();xs=images(4)
        for i in range(3):a.observe(xs[i],str(i))
        before=a.state_dict();original=SupportSession.observe
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with Profiler():
                with patch.object(a.learner,'observe',side_effect=RuntimeError('injected')):
                    a.observe(xs[3],'3')
        self.assertIs(SupportSession.observe,original);assert_equal(before,a.state_dict())

    def test_profiler_install_failure_restores_prior_patch(self):
        prof=Profiler();original=SupportSession.observe
        def install():
            prof.patch(SupportSession,'observe','probe');raise RuntimeError('install failure')
        with patch.object(prof,'install',side_effect=install):
            with self.assertRaises(RuntimeError):prof.__enter__()
        self.assertIs(SupportSession.observe,original)

    def test_exclusive_host_time_and_exception_cleanup(self):
        clock=iter([0.,1.,3.,5.]).__next__;p=Profiler(clock=clock)
        with p.region('outer'):
            with p.region('inner'):pass
        report=p.report(5.)
        self.assertEqual(report['attributed_exclusive_host_seconds'],5.)
        self.assertEqual(report['unattributed_host_seconds'],0.)
        with self.assertRaises(RuntimeError):
            with p.region('ignored') if False else Profiler().region('fault'):raise RuntimeError()

    def test_no_post_measurement_regions_when_disabled(self):
        p=Profiler();p.enabled=False
        with p.region('excluded'):pass
        self.assertFalse(p.regions)

    def test_typed_state_digest_and_value_equality(self):
        a=dict(x=torch.tensor([1.,2.]),r=(1,True),slots={0:torch.tensor(1.)})
        self.assertEqual(state_digest(a),state_digest(copy.deepcopy(a)))
        b=copy.deepcopy(a);b['x']=b['x'].double()
        self.assertNotEqual(state_digest(a),state_digest(b))
        with self.assertRaises(ValueError):assert_equal(a,b)
        b=copy.deepcopy(a);b['r']=[1,True]
        with self.assertRaises(ValueError):assert_equal(a,b)

    def test_wrong_seen_ids_and_duplicate_pass_output_rejected(self):
        a=actor();xs=images(3);rows=[dict(relative_path=str(i)) for i in range(3)]
        a.observe(xs[0],'wrong')
        with self.assertRaises(ValueError):bounded_replay(a,Provider(xs),rows,self.root,start=1,end=2,chain='x',deadline=100,attempt=lambda i:None)

    def test_safe_archive_paths(self):
        for p in ['../a','/a','C:/a','a\\b','a/../../b']:
            with self.assertRaises(ValueError):safe(self.root,p)

    def evidence(self,folder,value):
        folder.mkdir();(folder/'value.json').write_text(json.dumps(value));out=self.root/(folder.name+'.zip')
        export_archive(folder,out,schema=SCHEMA);return out

    def test_directory_and_zip_dedup_verified_by_content(self):
        z=self.evidence(self.root/'a',{'v':1})
        (self.root/'a/artifact_manifest.json').write_bytes(zipfile.ZipFile(z).read('artifact_manifest.json'))
        p=select_evidence(self.root,schema=SCHEMA)
        with Evidence(p) as e:self.assertEqual(e.verify_all(),1)

    def test_multiple_outputs_and_tampered_payload_rejected(self):
        self.evidence(self.root/'a',{'v':1});self.evidence(self.root/'b',{'v':2})
        with self.assertRaises(ValueError):select_evidence(self.root,schema=SCHEMA)
        with self.assertRaises(ValueError):select_evidence(self.root,schema=SCHEMA,override=self.root/'missing')

    def test_manifest_inventory_and_duplicate_zip_entries_rejected(self):
        path=self.root/'bad.zip'
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('artifact_manifest.json',json.dumps(dict(schema=SCHEMA,files={})))
            z.writestr('unregistered','x')
        with self.assertRaises(ValueError):Evidence(path)

    def test_tampered_extracted_payload_is_not_trusted_by_manifest(self):
        z=self.evidence(self.root/'a',{'v':1})
        with zipfile.ZipFile(z) as archive:(self.root/'a/artifact_manifest.json').write_bytes(archive.read('artifact_manifest.json'))
        (self.root/'a/value.json').write_text('changed')
        with Evidence(self.root/'a') as e:
            with self.assertRaises(ValueError):e.verify_all()

    def test_dataset_tar_extracts_only_exact_known_window(self):
        rows=[dict(relative_path='cable/train/good/000.png',split='train',sha256=__import__('hashlib').sha256(b'normal').hexdigest())]
        path=self.root/'mvtec.tar'
        with tarfile.open(path,'w') as z:
            for name,data in [('root/cable/train/good/000.png',b'normal'),('root/cable/test/bad/000.png',b'test'),('root/cable/train/good/001.png',b'future')]:
                m=tarfile.TarInfo(name);m.size=len(data);z.addfile(m,io.BytesIO(data))
        out=extract_window(path,self.root/'data',rows);verify_window(out,rows)
        self.assertEqual([p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()],[rows[0]['relative_path']])
        (out/rows[0]['relative_path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):verify_window(out,rows)

    def test_budget_partial_pass_never_retried_and_cost_cumulative(self):
        prepared=dict(parent='fixture',dataset='fixture',checkpoint='fixture',runtime={},source={})
        c=dict(work=str(self.root),prepared=prepared,started=100.,session_utc='fixture')
        calls=[]
        def process(cmd,logfile,**kw):
            calls.append(cmd);path=Path(cmd[-1]);cfg=json.loads(path.read_text());p=self.root/'passes'/cfg['pass'];p.mkdir(parents=True)
            atomic_json(dict(attempted_event_ids=[179],worker_elapsed_seconds=1),p/'attempts.json');return 0
        self.assertEqual(run(c,process=process,now=lambda:101.),'profile_incomplete_budget')
        ledger=json.loads((self.root/'ledger.json').read_text());self.assertEqual(ledger['reserved_transitions'],16)
        self.assertEqual(ledger['charged_seconds'],181.)
        self.assertEqual(run(dict(c,started=200.),process=process,now=lambda:201.),'profile_incomplete_budget')
        self.assertEqual(len(calls),1)

    def test_wrong_source_map_rejected_before_model_creation(self):
        (self.root/'docs/main_spec').mkdir(parents=True)
        from training.support_throughput_v5_1.contract import contract
        (self.root/'docs/main_spec/throughput_v5_1_contract.json').write_text(json.dumps(contract()))
        (self.root/'bundle_manifest.json').write_text(json.dumps(dict(files={'x':'0'*64})))
        with self.assertRaises(ValueError):verify_source(self.root)

    def mock_complete(self,cmd,logfile,**kw):
        cfg=json.loads(Path(cmd[-1]).read_text());name=cfg['pass'];p=self.root/'passes'/name;p.mkdir(parents=True)
        if name=='U1':atomic_torch_save({'fixture':torch.tensor([1.])},self.root/'reference_final_state.pt')
        value=dict(status='complete',start_cursor=179,end_cursor=195,completed_images=16,pipeline_seconds=2.,
            parity=dict(parent_events_exact=True),final_state_digest='fixture-state',trace_digest='fixture-trace')
        atomic_json(value,p/'measurement.json');atomic_json(dict(attempted_event_ids=list(range(179,195))),p/'attempts.json')
        atomic_json(dict(schema='throughput_pass_completion_v1',pass_id=name,
            files={n:sha(p/n) for n in ['measurement.json','attempts.json']},
            final_state_digest='fixture-state',reference_final_sha256=sha(self.root/'reference_final_state.pt')),p/'complete.json')
        return 0

    def test_transport_lifecycle_complete_three_passes_and_verified_noop_resume(self):
        c=dict(work=str(self.root),prepared=dict(parent='fixture',dataset='fixture',checkpoint='fixture',runtime={},source={}),started=100.,session_utc='fixture')
        self.assertEqual(run(c,process=self.mock_complete,now=lambda:101.),'profile_complete_pending_review')
        ledger=json.loads((self.root/'ledger.json').read_text());self.assertEqual(ledger['reserved_transitions'],48)
        def forbidden(*a,**kw):raise AssertionError('Must skip completed passes')
        self.assertEqual(run(dict(c,started=200.),process=forbidden,now=lambda:201.),'profile_complete_pending_review')
        self.assertEqual(json.loads((self.root/'ledger.json').read_text())['charged_seconds'],362.)
        (self.root/'passes/P1/measurement.json').write_text('{}')
        with self.assertRaises(ValueError):verify_complete(self.root,'P1')

    def test_worker_failure_is_preserved_and_never_retried(self):
        c=dict(work=str(self.root),prepared=dict(parent='fixture'),started=100.,session_utc='fixture')
        with self.assertRaises(RuntimeError):run(c,process=lambda *a,**k:1,now=lambda:101.)
        with self.assertRaises(ValueError):run(c,process=lambda *a,**k:0,now=lambda:102.)
        self.assertEqual(json.loads((self.root/'ledger.json').read_text())['reserved_transitions'],16)

    def test_budget_expired_before_pass_never_dispatches(self):
        c=dict(work=str(self.root),prepared={},started=100.,session_utc='fixture')
        def forbidden(*a,**kw):raise AssertionError('Expired budget dispatched work')
        self.assertEqual(run(c,process=forbidden,now=lambda:6000.),'profile_incomplete_budget')
        self.assertEqual(json.loads((self.root/'ledger.json').read_text())['reserved_transitions'],0)

    def test_nonfinite_or_forged_attempt_ledger_rejected(self):
        c=dict(work=str(self.root),prepared={},started=100.,session_utc='fixture')
        for charge,reserved,passes in [(float('nan'),0,{}),(0.,1,{}),(0.,16,{'P1':dict(status='complete',reserved_attempts=16)})]:
            # Corrupt external JSON can contain NaN even though our atomic writer forbids it.
            (self.root/'ledger.json').write_text(json.dumps(dict(schema='throughput_budget_ledger_v1',charged_seconds=charge,reserved_transitions=reserved,passes=passes,sessions=[])))
            with self.assertRaises(ValueError):run(c,process=lambda *a,**k:0,now=lambda:101.)

    def test_entry_exports_failure_without_dispatching_gpu_or_hiding_error(self):
        from notebook_support.support_throughput_v5_1.entry import run_notebook
        config=dict(work=str(self.root/'work'),output=str(self.root/'evidence.zip'),started=0.)
        with patch('notebook_support.support_throughput_v5_1.entry.prepare',side_effect=ValueError('bad parent')):
            with self.assertRaises(RuntimeError):run_notebook(config)
        with Evidence(self.root/'evidence.zip') as e:
            self.assertEqual(e.manifest['schema'],SCHEMA);e.verify_all()
            self.assertEqual(e.json('handoff.json')['status'],'profile_failed_preserve_evidence')

if __name__=='__main__':unittest.main()
