import unittest,tempfile,json,time,copy,io,tarfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from test_support import session,stream
from training.hope_support_v5 import run_job,PilotPause,compare_jobs
from training.benchmark_engine_v1 import _state_equal
from notebook_support.support_v5.bootstrap import normal_members,prepare
from models.hope_cad_v5.benchmark_adapter import SupportBenchmarkAdapter

class Provider:
    device='cpu'
    def __init__(self,xs):self.xs=xs
    def get(self,row):return self.xs[int(Path(row['relative_path']).stem)]

def rows(n):return [dict(relative_path=f'fixture/train/good/{i}.png',task_id=0,task_name='fixture',split='train',sha256=str(i)) for i in range(n)]

class PipelineContracts(unittest.TestCase):
    def test_real_driver_resume_with_pending_and_partial_meta_window(self):
        xs=stream(24);rs=rows(len(xs));identity={'fixture':True}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run_job(rs,Provider(xs),'NESTED',root/'full',identity,session_factory=session,checkpoint_every=8)
            calls=0
            def stop(deadline):
                nonlocal calls
                calls+=1
                if calls==14:raise PilotPause('fixture pause')
            with patch('training.hope_support_v5.boundary',stop),patch('builtins.print'):
                with self.assertRaises(PilotPause):run_job(rs,Provider(xs),'NESTED',root/'resume',identity,session_factory=session,checkpoint_every=8)
            before=torch.load(root/'resume/state.pt',weights_only=False);self.assertEqual(before['session']['cursor'],13);self.assertIsNotNone(before['session']['pending'])
            run_job(rs,Provider(xs),'NESTED',root/'resume',identity,session_factory=session,checkpoint_every=8)
            a=torch.load(root/'full/state.pt',weights_only=False);b=torch.load(root/'resume/state.pt',weights_only=False)
            self.assertTrue(_state_equal(a['session'],b['session']))
            self.assertEqual(json.loads((root/'full/normal_score_probes.json').read_text()),json.loads((root/'resume/normal_score_probes.json').read_text()))
    def test_crash_replays_only_uncheckpointed_images_without_changing_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs=rows(20);provider=Provider(stream(20));calls=0
            def crash(deadline):
                nonlocal calls
                calls+=1
                if calls==13:raise RuntimeError('simulated platform kill between checkpoints')
            with patch('training.hope_support_v5.boundary',crash),patch('builtins.print'):
                with self.assertRaises(RuntimeError):run_job(rs,provider,'NESTED',Path(tmp)/'resume',{'fixture':True},session_factory=session)
            state=torch.load(Path(tmp)/'resume/state.pt',weights_only=False);self.assertEqual(state['session']['cursor'],8)
            run_job(rs,provider,'NESTED',Path(tmp)/'resume',{'fixture':True},session_factory=session)
            run_job(rs,provider,'NESTED',Path(tmp)/'full',{'fixture':True},session_factory=session)
            a=torch.load(Path(tmp)/'resume/state.pt',weights_only=False);b=torch.load(Path(tmp)/'full/state.pt',weights_only=False);self.assertTrue(_state_equal(a['session'],b['session']))
    def test_driver_rejects_official_test_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad=rows(2);bad[0]['split']='test'
            with self.assertRaises(ValueError):run_job(bad,Provider(stream(2)),'CADIC',tmp,{},session_factory=session)
            with self.assertRaises(ValueError):run_job([rows(1)[0]]*2,Provider(stream(2)),'CADIC',tmp,{},session_factory=session)
    def test_completed_job_rejects_changed_identity_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_job(rows(3),Provider(stream(3)),'CADIC',tmp,{'fixture':True},session_factory=session)
            with self.assertRaises(ValueError):run_job(rows(3),Provider(stream(3)),'CADIC',tmp,{'fixture':False},session_factory=session)
            p=Path(tmp)/'state.pt';p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaises(ValueError):run_job(rows(3),Provider(stream(3)),'CADIC',tmp,{'fixture':True},session_factory=session)
    def test_four_jobs_compare_truthful_scientific_gate_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ['CADIC','GEOMETRIC','FIRST_ORDER','NESTED']:run_job(rows(12),Provider(stream(12)),mode,Path(tmp)/'jobs'/mode,{'fixture':True},session_factory=session)
            result=compare_jobs(tmp);self.assertEqual(result['status'],'pilot_complete_scientific_gate_failed');self.assertFalse(result['gates']['completed_labels']);self.assertFalse(result['reportable'])
    def test_input_scan_rejects_cpu_before_parent_or_assets(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Enable GPU'):prepare({})
    def test_archive_extracts_only_normals_not_official_test(self):
        bio=io.BytesIO()
        with tarfile.open(fileobj=bio,mode='w') as z:
            for name in ['bottle/train/good/0.png','bottle/test/broken/0.png','bottle/ground_truth/broken/0.png']:z.addfile(tarfile.TarInfo(name),io.BytesIO())
        bio.seek(0)
        with tempfile.TemporaryDirectory() as tmp,tarfile.open(fileobj=bio) as z:self.assertEqual([m.name for m in normal_members(z,tmp)],['bottle/train/good/0.png'])
    def test_remaining_ablation_modes_core_executable(self):
        for mode in ['FAST_ONLY','NO_FAST_ADAPTATION','NO_SELF_MODIFICATION','NO_ANCHOR_OBJECTIVE']:
            s=session(mode)
            for i,x in enumerate(stream(12)):s.observe(x,str(i))
            state=s.state_dict();other=session(mode);other.load_state_dict(state);self.assertTrue(_state_equal(state,other.state_dict()))
            self.assertGreater(s.learner.labels,8)
            self.assertEqual(s.learner.slow_updates,0 if mode=='FAST_ONLY' else s.learner.labels//8)
    def test_RGB_adapter_batch_transport_and_read_only_score(self):
        from models.hope_cad_v5.session import SupportSession
        from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
        class Dummy(nn.Module):
            def extract_patch_features(self,x):
                values=x[:,0].reshape(len(x),-1)[:,:784]
                return values[:,:,None].expand(-1,-1,768).contiguous()
        s=SupportSession('CADIC',bank_config=CADICPatchCoresetConfig(budget=8,dim=768,chunk_size=4,query_chunk_size=4,pair_chunk_size=4))
        adapter=SupportBenchmarkAdapter(None,extractor=Dummy(),session=s)
        gen=torch.Generator().manual_seed(19);batch=dict(images=torch.rand(2,3,224,224,generator=gen),labels=torch.zeros(2,dtype=torch.long),relative_path=['normal-a','normal-b'])
        result=adapter.fit_task(99,'bookkeeping-only',[batch]);self.assertEqual(result['samples'],2);before=adapter.state_dict()
        out=adapter.score_batch(batch);self.assertEqual(out['image_scores'].shape,(2,));self.assertEqual(out['anomaly_maps'].shape,(2,224,224));self.assertTrue(_state_equal(before,adapter.state_dict()))
        bad=dict(batch,labels=torch.ones(2,dtype=torch.long))
        with self.assertRaises(ValueError):adapter.fit_task(0,'ignored',[bad])

if __name__=='__main__':unittest.main()
