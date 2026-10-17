"""Meaningful recovery regressions; synthetic fixtures are never research benchmarks."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from models.hope_cad_v2.benchmark_adapter_v4 import HOPEBenchmarkAdapterV4
from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
from models.hope_cad_v2.visual_block import VisualBlockConfig
from training.hope_outer_learning_v4 import export_initialization,capture_rng
from training.benchmark_engine_v1 import _state_equal,_clone_state
from training.hope_w7_engine import W7Engine,SessionLimit
from notebook_support.recovery_stage1.common import (ObservedAdapter,locked_json,sampled_rows,
    assert_legal_normal,read_manifest,find_reviewed,restore_recovery,export,source_identity,JOBS,PROTOCOL)
from notebook_support.recovery_stage1.diagnostics import measure_normal,residual_components,run_diagnostics

ROOT=Path(__file__).resolve().parents[2]

class TinyExtractor:
    def extract_patch_features(self,images):
        p=torch.nn.functional.avg_pool2d(images,8,8).flatten(2).transpose(1,2)
        return torch.cat((p,p,p[:,:,:2]),dim=-1)
    def named_parameters(self):return iter(())
    def close(self):pass


def adapter(method='M'):
    torch.manual_seed(41)
    model=HOPEFeaturePredictorV4(VisualBlockConfig(8,8,8,4,4,4),method=method,beta=.01)
    identity=dict(roles_digest='roles',schedule_digest='schedule',source_sha256={'fixture.py':'fixture'},
        spec_sha256={'fixture.md':'fixture'},feature_identity={'backbone_sha256':'a'*64})
    initial=export_initialization(model,identity)
    return HOPEBenchmarkAdapterV4(initial,checkpoint_path='fixture.safetensors',checkpoint_sha256='a'*64,
        checkpoint_identity='fixture',extractor=TinyExtractor(),protocol_identity={'fixture':True},allow_test_components=True)


class Data:
    task_names=['fixture_a','fixture_b']
    def __init__(self,root):
        self.root=Path(root);self.image_size=32;self.mean=[0.,0.,0.];self.std=[1.,1.,1.];rows=[]
        for task,name in enumerate(self.task_names):
            for i in range(8):
                p=self.root/name/'train/good'/f'{i:03d}.png';p.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.full((32,32,3),20+task*80+i,dtype=np.uint8)).save(p)
                rows.append(dict(relative_path=p.relative_to(self.root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    split='train',task_id=task,task_name=name))
        self.manifest={'online':rows}
    def training_paths(self,task):return deepcopy([r for r in self.manifest['online'] if r['task_id']==task])
    def build_train_loader(self,task):
        return DataLoader(HashedNormalDataset(self.root,self.training_paths(task),32,self.mean,self.std,True),
            batch_size=1,shuffle=False,generator=torch.Generator().manual_seed(0))


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        base=ROOT/'results/recovery_stage1/tmp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_locked_tuple_json_resumes_and_real_change_rejected(self):
        p=self.root/'lock.json';a=locked_json(p,{'periods':(1,8),'rate':.01});raw=p.read_bytes()
        self.assertEqual(a,locked_json(p,{'periods':(1,8),'rate':.01}));self.assertEqual(raw,p.read_bytes())
        with self.assertRaises(ValueError):locked_json(p,{'periods':(1,1),'rate':.01})
        with self.assertRaises(ValueError):locked_json(self.root/'nan.json',{'value':float('nan')})
    def test_frozen_source_and_scope_no_refit_or_extra_jobs(self):
        self.assertEqual(len(source_identity()['frozen']),119)
        self.assertEqual(set(JOBS),{'CADIC10000_seed0','C1_seed0','A1_seed0'})
        self.assertFalse(PROTOCOL['selection_or_refitting'])
        self.assertFalse(PROTOCOL['diagnostic']['selection_feedback'])
    def test_residual_components_and_norm_bound(self):
        target=torch.tensor([[[1.,0.],[1.,0.],[0.,0.]]])
        pred=torch.tensor([[[-3.,0.],[0.,1.],[0.,0.]]])
        c=residual_components(pred,target)
        self.assertTrue(torch.allclose(c['score'],c['radial']+c['angular']))
        self.assertTrue(torch.isfinite(c['cosine']).all())
        self.assertAlmostEqual(float(c['score'][0,0]),16)
        self.assertGreaterEqual(float(c['prediction_norm'][0,0]),float(c['score'][0,0].sqrt())-1)
    def test_real_adapter_diagnostic_matches_scoring_and_is_readonly(self):
        for method in ('M','C1','A1'):
            with self.subTest(method=method):
                a=adapter(method);image=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
                before=_clone_state(a.state_dict());rng=capture_rng()
                out=measure_normal(a,image)
                score=a.score_batch({'images':image})
                self.assertAlmostEqual(out['image_score'],float(score['image_scores'][0]),places=5)
                self.assertTrue(_state_equal(before,a.state_dict()));self.assertTrue(_state_equal(rng,capture_rng()))
                if method!='C1':self.assertEqual(out['discarded_local_proposal']['proposal_clock'],1)
                a.close()
    def test_observer_preserves_stream_state_and_logs_resume_without_duplicates(self):
        data=Data(self.root)
        plain=adapter();W7Engine(plain,data,self.root/'plain','fixture').train();expected=plain.state_dict();rng=capture_rng()
        first=adapter();observed=ObservedAdapter(first,self.root/'observed_logs')
        with self.assertRaises(SessionLimit):W7Engine(observed,data,self.root/'observed','fixture',checkpoint_every=2).train(stop_after_events=5)
        resumed=adapter();observed=ObservedAdapter(resumed,self.root/'observed_logs')
        W7Engine(observed,data,self.root/'observed','fixture',checkpoint_every=2).train()
        self.assertTrue(_state_equal(expected,resumed.state_dict()));self.assertTrue(_state_equal(rng,capture_rng()))
        logs=list((self.root/'observed_logs').glob('*.json'));self.assertEqual(len(logs),16)
        self.assertTrue(all('mean_level_losses' in json.loads(p.read_text())['observed_fit_return'] for p in logs))
        for a in (plain,first,resumed):a.close()
    def test_observer_rejects_test_or_anomaly_before_update(self):
        a=adapter();wrapped=ObservedAdapter(a,self.root/'logs');before=a.state_dict()
        batch=dict(images=torch.ones(1,3,32,32),labels=torch.tensor([1]),relative_path=['fixture_a/test/defect/0.png'])
        with self.assertRaises(ValueError):wrapped.fit_task(0,'bad',[batch])
        self.assertTrue(_state_equal(before,a.state_dict()));a.close()
    def test_samples_deterministic_and_future_or_nononline_rows_rejected(self):
        d=Data(self.root);rows=sampled_rows(d,0);self.assertEqual(rows,sampled_rows(d,0))
        allowed={r['relative_path']:r for r in d.manifest['online']}
        assert_legal_normal(rows[0],allowed,0)
        with self.assertRaises(ValueError):assert_legal_normal(d.training_paths(1)[0],allowed,0)
        changed=deepcopy(rows[0]);changed['split']='test'
        with self.assertRaises(ValueError):assert_legal_normal(changed,allowed,1)
        changed=deepcopy(rows[0]);changed['sha256']='different'
        with self.assertRaises(ValueError):assert_legal_normal(changed,allowed,1)
    def test_diagnostic_resume_uses_same_normals_and_rejects_identity_tampering(self):
        d=Data(self.root);a=adapter('C1');e=W7Engine(a,d,self.root/'run','fixture');e.train()
        folder=self.root/'diagnostics'
        report=run_diagnostics(a,d,e,folder,'C1_seed0','study')
        files={p.relative_to(folder).as_posix():p.read_bytes() for p in folder.rglob('*.json')}
        run_diagnostics(a,d,e,folder,'C1_seed0','study')
        self.assertEqual(files,{p.relative_to(folder).as_posix():p.read_bytes() for p in folder.rglob('*.json')})
        self.assertTrue(all(abs(r['final_minus_diagonal_normal_loss'])<1e-7 for r in report['category_summaries'].values()))
        p=next((folder/'samples').rglob('*.json'));v=json.loads(p.read_text());v['identity']['study_digest']='bad';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'resume identity'):run_diagnostics(a,d,e,folder,'C1_seed0','study')
        a.close()
    def test_export_restore_integrity_and_duplicate_parent_copies(self):
        root=self.root/'evidence';root.mkdir();(root/'x.json').write_text('{"ok":true}')
        out=self.root/'evidence.zip';export(root,out)
        restore_recovery(out,self.root/'restored')
        self.assertEqual((root/'x.json').read_bytes(),(self.root/'restored/x.json').read_bytes())
        manifest,h=read_manifest(out)
        for name in ('a','b'):
            p=self.root/name;p.mkdir();(p/'artifact_manifest.json').write_bytes(zipfile.ZipFile(out).read('artifact_manifest.json'))
            (p/'x.json').write_text('{"ok":true}')
        self.assertIn(find_reviewed(self.root,h),[self.root/'a',self.root/'b'])
        (self.root/'restored/x.json').write_text('bad')
        with self.assertRaises(ValueError):restore_recovery(out,self.root/'restored')
    def test_cadic_reference_full_eq9_and_compact_roundtrip(self):
        from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1,CADICPatchCoresetConfig
        cfg=CADICPatchCoresetConfig(budget=9,dim=2,image_neighbors=9)
        bank=CADICPatchCoresetV1(cfg);bank.update(torch.arange(18).reshape(9,2).float())
        query=torch.tensor([[[21.,22.],[3.,4.]]]);before=bank.state_dict();score=bank.score(query)
        copy=CADICPatchCoresetV1(cfg);copy.load_state_dict(before)
        self.assertTrue(_state_equal(score,copy.score(query)));self.assertTrue(_state_equal(before,bank.state_dict()))

if __name__=='__main__':unittest.main()