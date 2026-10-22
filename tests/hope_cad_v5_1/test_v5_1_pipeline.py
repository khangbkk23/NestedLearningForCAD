"""Real driver orchestration with synthetic normals, no external assets."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
from test_v5_1_session import session, stream
from models.hope_cad_v2.state import tensor_digest
from models.hope_cad_v5_1.contract import canonical
from training.hope_support_v5_1 import run_job, compare_jobs, lock, PilotPause
from training.runtime import atomic_json
from training.benchmark_engine_v1 import _state_equal
from notebook_support.support_v5_1.bootstrap import prepare, resume_source, normal_members
from notebook_support.support_v5_1.common import SCHEMA
from notebook_support.common.evidence import export_archive


class Provider:
    device='cpu'
    def __init__(self,xs):self.xs=xs
    def get(self,row):return self.xs[int(Path(row['relative_path']).stem)]


def rows(n):
    return [dict(relative_path=f'fixture/train/good/{i}.png',task_id=0,task_name='fixture',
                 split='train',sha256=str(i)) for i in range(n)]


def probe_lock(root,rs,provider):
    atomic_json(rs,root/'pilot_rows.json')
    probes=[dict(identity=r['relative_path'],image_sha256=r['sha256'],
                 feature_sha256=tensor_digest(({'patches':provider.get(r)},))) for r in rs[:8]]
    lock(root/'probe_lock.json',dict(probes=probes))


class PipelineContracts(unittest.TestCase):
    def test_driver_pause_and_crash_resume_same_events(self):
        rs=rows(19);provider=Provider(stream(19))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            run_job(rs,provider,'NESTED',root/'full',{'fixture':True},session_factory=session)
            for kind in ('pause','crash'):
                calls=0
                def stop(deadline):
                    nonlocal calls
                    calls+=1
                    if calls==14:
                        if kind=='pause':raise PilotPause('controlled boundary')
                        raise RuntimeError('simulated hard crash')
                with patch('training.hope_support_v5_1.boundary',stop):
                    with self.assertRaises((PilotPause,RuntimeError)):
                        run_job(rs,provider,'NESTED',root/kind,{'fixture':True},session_factory=session)
                saved=torch.load(root/kind/'state.pt',weights_only=False)
                self.assertEqual(saved['session']['cursor'],13 if kind=='pause' else 8)
                run_job(rs,provider,'NESTED',root/kind,{'fixture':True},session_factory=session)
                end=torch.load(root/kind/'state.pt',weights_only=False)
                full=torch.load(root/'full/state.pt',weights_only=False)
                self.assertTrue(_state_equal(end['session'],full['session']))
                for i in range(19):self.assertEqual(json.loads((root/kind/'events'/f'{i:04d}.json').read_text()),json.loads((root/'full/events'/f'{i:04d}.json').read_text()))

    def test_gates_reject_probe_mismatch_and_do_not_rescue(self):
        rs=rows(12);provider=Provider(stream(12))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);probe_lock(root,rs,provider)
            for mode in ('CADIC','GEOMETRIC','FIRST_ORDER','NESTED'):
                run_job(rs,provider,mode,root/'jobs'/mode,{'fixture':True},session_factory=session)
            result=compare_jobs(root,canonical_run=False)
            self.assertEqual(result['status'],'pilot_complete_scientific_gate_failed')
            self.assertFalse(result['gates']['completed_labels']);self.assertFalse(result['reportable'])
            path=root/'jobs/NESTED/normal_score_probes.json';v=json.loads(path.read_text());v.pop();atomic_json(v,path)
            with self.assertRaises(ValueError):compare_jobs(root,canonical_run=False)

    def test_driver_rejects_test_duplicate_and_changed_completion(self):
        provider=Provider(stream(10));rs=rows(10)
        with tempfile.TemporaryDirectory() as t:
            bad=copy.deepcopy(rs);bad[0]['split']='test'
            with self.assertRaises(ValueError):run_job(bad,provider,'CADIC',t,{},session_factory=session)
            with self.assertRaises(ValueError):run_job([rs[0]]*8,provider,'CADIC',t,{},session_factory=session)
            run_job(rs,provider,'CADIC',t,{},session_factory=session)
            with self.assertRaises(ValueError):run_job(rs,provider,'CADIC',t,{'changed':True},session_factory=session)
            path=Path(t)/'state.pt';path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaises(ValueError):run_job(rs,provider,'CADIC',t,{},session_factory=session)

    def test_cpu_rejected_before_asset_loading(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Enable GPU'):prepare({})

    def test_predecessor_resume_rejected_and_same_manifest_duplicates_allowed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'payload').mkdir();atomic_json({'status':'paused'},root/'payload/handoff.json')
            export_archive(root/'payload',root/'old.zip',schema='hope_support_v5_normal_pilot_evidence_v1')
            with self.assertRaises(ValueError):resume_source(root,root/'old.zip')
            self.assertIsNone(resume_source(root))
            export_archive(root/'payload',root/'new.zip',schema=SCHEMA)
            self.assertEqual(resume_source(root),root/'new.zip')

    def test_json_lock_canonicalizes_tuple_without_ignoring_real_changes(self):
        with tempfile.TemporaryDirectory() as t:
            target=Path(t)/'lock.json'
            self.assertEqual(lock(target,dict(a=(1,2))),lock(target,dict(a=[1,2])))
            with self.assertRaises(ValueError):lock(target,dict(a=(1,3)))

    def test_frozen_feature_cache_checks_image_and_tensor_bytes(self):
        from PIL import Image
        from training.hope_support_v5_1 import NormalFeatureCache
        from models.hope_cad_v5_1.contract import file_sha
        class Extractor:
            def extract_patch_features(self,image):return torch.zeros(1,784,768)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'fixture/train/good/0.png';p.parent.mkdir(parents=True)
            Image.new('RGB',(8,8)).save(p)
            row=rows(1)[0];row['sha256']=file_sha(p)
            provider=NormalFeatureCache(root,Extractor(),root/'cache',root/'receipts','cpu')
            first=provider.get(row);self.assertTrue(torch.equal(first,provider.get(row)))
            payload=next((root/'cache').glob('*.pt'));torch.save(first+1,payload)
            with self.assertRaises(ValueError):provider.get(row)
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaises(ValueError):provider.get(row)

    def test_rgb_adapter_batch_transport_and_readonly_finalization(self):
        from torch import nn
        from models.hope_cad_v5_1.benchmark_adapter import SupportBenchmarkAdapter
        from models.hope_cad_v5_1.session import SupportSession
        from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
        class Extractor(nn.Module):
            def extract_patch_features(self,x):return x[:,0].reshape(len(x),-1)[:,:784,None].expand(-1,-1,768).contiguous()
        s=SupportSession('CADIC',bank_config=CADICPatchCoresetConfig(budget=8,dim=768,chunk_size=4,pair_chunk_size=4,query_chunk_size=4))
        adapter=SupportBenchmarkAdapter(None,extractor=Extractor(),session=s)
        batch=dict(images=torch.rand(2,3,224,224),labels=torch.zeros(2,dtype=torch.long),relative_path=['a','b'])
        adapter.fit_task(99,'not-model-input',[batch]);old=adapter.state_dict()
        result=adapter.score_batch(batch)
        self.assertEqual(result['anomaly_maps'].shape,(2,224,224))
        self.assertTrue(_state_equal(old,adapter.state_dict()))
        bad=dict(batch,relative_path=['truncated'])
        with self.assertRaises(ValueError):adapter.fit_task(0,'ignored',[bad])
        adapter.finalize_stream();self.assertTrue(adapter.session.finalized)

    def test_bootstrap_failure_exports_evidence_before_raising(self):
        from notebook_support.support_v5_1.entry import run_notebook
        import zipfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config=dict(work=str(root/'work'),output=str(root/'failure.zip'),
                                    started=time.monotonic(),phase='pilot',session_hours=3.)
            with patch('notebook_support.support_v5_1.entry.prepare',side_effect=ValueError('missing fixture input')):
                with self.assertRaises(RuntimeError):run_notebook(config)
            with zipfile.ZipFile(root/'failure.zip') as z:
                self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
                self.assertEqual(json.loads(z.read('artifact_manifest.json'))['schema'],SCHEMA)


if __name__=='__main__':unittest.main()
