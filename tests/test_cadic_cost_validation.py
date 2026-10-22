import copy
import tempfile
import unittest
import contextlib
import io
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from training.cadic_cost_validation import summarize
from models.hope_cad_v5_1.session import SupportSession
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from training.support_nn_write_v5_1.candidate import activation
from training.support_throughput_v5_1.replay import assert_equal, state_digest, bounded_replay
from training.runtime import atomic_json
from training.support_throughput_v5_1.contract import sha, ROOT


def fixtures():
    return {n:dict(status='complete',start_cursor=179,end_cursor=195,completed_images=16,
        finalized=False,global_rng_unchanged=True,final_state_digest='x',trace_digest='y',
        pipeline_seconds=t,gpu=dict(peak_allocated_bytes=100))
        for n,t in zip(('R1','O1','O2','R2'),(100.,60.,61.,101.))}


class SummaryTests(unittest.TestCase):
    def test_ratios_and_nonclaim(self):
        v=summarize(fixtures(),dict(reference_mean_seconds=200,optimized_mean_seconds=120))
        self.assertTrue(v['speed_gate']);self.assertTrue(v['memory_gate'])
        self.assertAlmostEqual(v['speedup'],100.5/60.5)
        self.assertFalse(v['historical_nested']['paired_between_methods'])
        self.assertFalse(v['full_stream_ETA']);self.assertFalse(v['reportable'])

    def test_bad_or_missing_pair_fails(self):
        for key,value in [('final_state_digest','bad'),('trace_digest','bad'),('status','incomplete'),
                          ('completed_images',15),('pipeline_seconds',float('nan'))]:
            v=fixtures();v['O1'][key]=value
            with self.assertRaises(ValueError):summarize(v,dict(reference_mean_seconds=200,optimized_mean_seconds=120))
        v=fixtures();del v['O2']
        with self.assertRaises(ValueError):summarize(v,{})

    def test_slow_candidate_not_rescued(self):
        v=fixtures();v['O1']['pipeline_seconds']=110
        self.assertFalse(summarize(v,dict(reference_mean_seconds=200,optimized_mean_seconds=120))['speed_gate'])

    def test_memory_gate(self):
        v=fixtures();v['O1']['gpu']['peak_allocated_bytes']=100+65*1024**2
        self.assertFalse(summarize(v,dict(reference_mean_seconds=200,optimized_mean_seconds=120))['memory_gate'])


class CADICTests(unittest.TestCase):
    def test_own_predecessor_and_full_exact_continuation(self):
        cfg=CADICPatchCoresetConfig(budget=8,dim=768,chunk_size=4,query_chunk_size=4,pair_chunk_size=4)
        g=torch.Generator().manual_seed(3)
        xs=[torch.randn(1,4,768,generator=g) for _ in range(4)]
        kwargs=dict(device='cpu',seed=0,bank_config=cfg,model_config=dict(height=2,width=2))
        actor=SupportSession('CADIC',**kwargs)
        actor.observe(xs[0],'a');actor.observe(xs[1],'b')
        predecessor=copy.deepcopy(actor.state_dict());before=state_digest(predecessor)
        states=[];events=[]
        for optimized in (False,True,True,False):
            with activation(optimized):
                a=SupportSession('CADIC',**kwargs)
                a.load_state_dict(predecessor)
                events.append([a.observe(xs[2],'c'),a.observe(xs[3],'d')]);states.append(a.state_dict())
                self.assertIsNone(a.learner);self.assertIsNone(a.pending);self.assertFalse(a.finalized)
        for e,s in zip(events,states):assert_equal(e,events[0]);assert_equal(s,states[0])
        self.assertEqual(before,state_digest(predecessor))

    def test_bounded_replay_checkpoint_and_counters(self):
        cfg=CADICPatchCoresetConfig(budget=8,dim=768,chunk_size=4,query_chunk_size=4,pair_chunk_size=4)
        actor=SupportSession('CADIC',device='cpu',seed=0,bank_config=cfg,model_config=dict(height=2,width=2))
        g=torch.Generator().manual_seed(9)
        xs=[torch.randn(1,4,768,generator=g) for _ in range(10)]
        rows=[dict(relative_path=str(i)) for i in range(10)]
        for i in range(6):actor.observe(xs[i],str(i))
        class Provider:
            job_digest='synthetic'
            def get(self,row):return xs[int(row['relative_path'])]
        attempted=[]
        with tempfile.TemporaryDirectory() as d:
            m,state=bounded_replay(actor,Provider(),rows,d,start=6,end=9,chain='initial',
                deadline=float('inf'),attempt=attempted.append,device='cpu')
            self.assertEqual(attempted,[6,7,8]);self.assertEqual(state['cursor'],9)
            self.assertTrue((Path(d)/'active.pt').exists())
            self.assertEqual(torch.load(Path(d)/'active.pt',weights_only=True)['cursor'],8)
            self.assertEqual(m['work_counters']['committed_replacements'],m['work_counters']['proposal_replacements'])


class OrchestrationTests(unittest.TestCase):
    def test_prepare_failure_exports_and_no_false_complete(self):
        from notebook_support.cadic_cost_validation import run_notebook
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'failure.zip';work=root/'work'
            c=dict(work=str(work),output=str(out),inputs=str(root),assets=str(root/'assets'),started=time.monotonic())
            with patch('notebook_support.cadic_cost_validation.verify_delivery',return_value='source'), \
                 patch('notebook_support.cadic_cost_validation.prepare',side_effect=ValueError('runtime mismatch')), \
                 contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):run_notebook(c)
            self.assertTrue(out.is_file())
            self.assertEqual(json.loads((work/'handoff.json').read_text())['status'],'failed_preserve_evidence')
            self.assertEqual(json.loads((work/'attempt_ledger.json').read_text())['reserved'],[])

    def test_full_worker_order_budget_and_seals(self):
        from notebook_support.cadic_cost_validation import run_notebook
        class Evidence:
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
        runtime=json.loads((ROOT/'notebook_support/cadic_cost_nested_reference.json').read_text())['runtime']
        calls=[]
        def process(command,log,**kwargs):
            c=json.loads(Path(command[-1]).read_text());name=c['pass'];calls.append(name)
            work=Path(c['work']);folder=work/'passes'/name;folder.mkdir(parents=True)
            if name=='PREP':(work/'predecessor179.pt').write_bytes(b'synthetic-owner')
            m=fixtures().get(name,dict(fixtures()['R1']))
            m['attempted_event_ids']=list(range(179)) if name=='PREP' else list(range(179,195))
            atomic_json(m,folder/'measurement.json')
            atomic_json(dict(status='complete',features_exact_to_reviewed_parent=True,full_state_exact_to_R1=name!='PREP',
                measurement_sha256=sha(folder/'measurement.json'),source_digest='source',
                predecessor_sha256=sha(work/'predecessor179.pt')),folder/'complete.json')
            return 0
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work'
            c=dict(work=str(work),output=str(root/'done.zip'),inputs=str(root),assets=str(root/'assets'),started=time.monotonic())
            with patch('notebook_support.cadic_cost_validation.verify_delivery',return_value='source'), \
                 patch('notebook_support.cadic_cost_validation.prepare',return_value=dict(runtime=runtime,parent='parent',dataset=str(root),checkpoint='checkpoint')), \
                 patch('notebook_support.cadic_cost_validation.Evidence',Evidence), \
                 patch('notebook_support.cadic_cost_validation.verify_parent',return_value=dict(rows=[dict(relative_path=str(i)) for i in range(195)])), \
                 patch('notebook_support.cadic_cost_validation.verify_window'), \
                 patch('notebook_support.cadic_cost_validation.stream_process',side_effect=process), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_notebook(c)
            self.assertEqual(calls,['PREP','R1','O1','O2','R2'])
            ledger=json.loads((work/'attempt_ledger.json').read_text())
            self.assertEqual(sum(v['arrivals'] for v in ledger['reserved']),243)
            self.assertTrue(all(v['state']=='complete' for v in ledger['reserved']))
            self.assertEqual(json.loads((work/'handoff.json').read_text())['status'],'complete_pending_review')


if __name__=='__main__':unittest.main()
