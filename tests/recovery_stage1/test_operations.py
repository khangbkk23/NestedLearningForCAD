"""Operational tests: export before failure, pause semantics, fixed final-only scope and resource watchdog."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/hope_cad_v2'))
from test_w7_engine import FixtureData,FixtureAdapter,Rows
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from torch.utils.data import DataLoader
from notebook_support.recovery_stage1 import entry,runner
from notebook_support.recovery_stage1.bootstrap import resume_input
from notebook_support.recovery_stage1.cadic_profile import bounded_probe
from notebook_support.recovery_stage1.common import atomic_json,export,read_manifest,import_reference


class PathsFixture(FixtureData):
    def __init__(self,root):
        super().__init__(root)
        for task,rows in enumerate(self.train):
            for i,row in enumerate(rows):
                old=self.root/row['relative_path'];new=self.root/f'fixture_{task}/train/good/{i}.png'
                new.parent.mkdir(parents=True,exist_ok=True);old.rename(new)
                row['relative_path']=new.relative_to(self.root).as_posix()
                import hashlib
                row['sha256']=hashlib.sha256(new.read_bytes()).hexdigest()
                row['split']='train'
        self.manifest={'online':[]}
    def build_train_loader(self,task):
        return DataLoader(HashedNormalDataset(self.root,self.train[task],2,[0,0,0],[1,1,1],True),batch_size=1)

class Adapter(FixtureAdapter):
    def close(self):pass
    def profile_metadata(self):return {'fixture':True}
    def memory_stats(self):return {'fixture':True}


class OperationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
    def setUp(self):
        base=ROOT/'results/recovery_stage1/tmp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def config(self):return dict(phase='preflight',inputs=self.root/'inputs',work=self.root/'work',
        w6=self.root/'w6',assets=self.root/'assets',output=self.root/'out.zip',
        started=time.monotonic(),session_hours=1,jobs=['C1_seed0'],device='cpu')
    def test_bootstrap_failure_exports_zip_before_raising(self):
        c=self.config()
        with patch.object(entry,'prepare',side_effect=ValueError('wrong parent')),redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            with self.assertRaisesRegex(RuntimeError,'ZIP exported'):entry.run_notebook(c)
        self.assertTrue(c['output'].exists())
        with zipfile.ZipFile(c['output']) as z:
            self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
    def test_successful_preflight_and_controls_export_and_restore(self):
        for phase,status in [('preflight','recovery_preflight_passed'),('controls','recovery_stage1_complete_pending_review')]:
            c=self.config();c['phase']=phase;c['work']=self.root/phase;c['output']=self.root/(phase+'.zip')
            def success(*a,**k):
                atomic_json({'status':status,'fixture':True},c['work']/'handoff.json')
                (c['work']/'result.txt').write_text('fixture-complete');return 0
            with patch.object(entry,'prepare'),patch.object(entry,'stream_process',side_effect=success),redirect_stdout(StringIO()):
                result=entry.run_notebook(c)
            from notebook_support.recovery_stage1.common import restore_recovery
            target=self.root/(phase+'_restored');restore_recovery(result['path'],target)
            self.assertEqual(json.loads((target/'handoff.json').read_text())['status'],status)
            self.assertEqual((target/'result.txt').read_text(),'fixture-complete')
    def test_nonzero_runner_cannot_reuse_stale_success_handoff(self):
        c=self.config();c['work'].mkdir();atomic_json({'status':'recovery_preflight_passed'},c['work']/'handoff.json')
        with patch.object(entry,'prepare'),patch.object(entry,'stream_process',return_value=7),redirect_stdout(StringIO()):
            with self.assertRaises(RuntimeError):entry.run_notebook(c)
        with zipfile.ZipFile(c['output']) as z:
            self.assertEqual(json.loads(z.read('handoff.json'))['runner_exit_code'],7)
    def test_pause_exports_and_is_not_reported_as_completion(self):
        c=self.config()
        def pause(*a,**k):atomic_json({'status':'paused_at_atomic_boundary'},c['work']/'handoff.json');return 0
        with patch.object(entry,'prepare'),patch.object(entry,'stream_process',side_effect=pause),redirect_stdout(StringIO()):
            out=entry.run_notebook(c)
        self.assertEqual(Path(out['path']),c['output'])
        self.assertEqual(json.loads((c['work']/'handoff.json').read_text())['status'],'paused_at_atomic_boundary')
    def test_checkout_normalization_accepts_only_exact_newline_variants(self):
        import hashlib
        from notebook_support.recovery_stage1.restore_checkout import variants
        expected=hashlib.sha256(b'a\r\nb\r\n').hexdigest()
        self.assertEqual(variants(b'a\nb\n',expected)[1],b'a\r\nb\r\n')
        with self.assertRaises(ValueError):variants(b'changed\n',expected)

    def test_resource_watchdog_stops_probe_and_preserves_failure(self):
        p=self.root/'resource.json'
        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError,'exceeded limit'):
                bounded_probe([sys.executable,'-c','import time; time.sleep(3)'],p,.1)
        self.assertEqual(json.loads(p.read_text())['status'],'resource_timeout')
        self.assertFalse(json.loads(p.read_text())['official_test_accessed'])
    def test_resume_rejects_multiple_different_outputs_but_accepts_identical_copies(self):
        a=self.root/'a';a.mkdir();(a/'x').write_text('a');z=self.root/'first.zip';export(a,z)
        with zipfile.ZipFile(z) as archive:b=archive.read('artifact_manifest.json')
        for name in ('c','d'):
            p=self.root/name;p.mkdir();(p/'artifact_manifest.json').write_bytes(b)
        self.assertIn(resume_input(self.root),[self.root/'c',self.root/'d'])
        p=self.root/'d/artifact_manifest.json';m=json.loads(p.read_text());m['different_version']=True;p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'DIFFERENT'):resume_input(self.root)
    def test_wrong_w7_preflight_cannot_be_used_as_completed_reference(self):
        p=self.root/'wrong';p.mkdir();(p/'artifact_manifest.json').write_text(json.dumps({'schema':'hope_w7_evidence_v1','files':{}}))
        with self.assertRaisesRegex(ValueError,'completed W7.2'):import_reference(p,self.root/'reference')
    def test_new_preflight_executes_real_cms_observer_resume_path_on_rgb_fixtures(self):
        import torch
        from test_recovery import adapter as tiny_adapter, TinyExtractor
        from models.hope_cad_v2.benchmark_adapter_v4 import HOPEBenchmarkAdapterV4
        from notebook_support.recovery_stage1.preflight import run_preflight
        from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
        from models.hope_cad_v2.visual_block import VisualBlockConfig
        from training.hope_outer_learning_v4 import export_initialization
        from PIL import Image
        import hashlib
        rows=[]
        for i in range(16):
            p=self.root/'fixture/train/good'/f'{i:03d}.png';p.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(32,32),(20+i,30+i,40+i)).save(p)
            rows.append(dict(relative_path=p.relative_to(self.root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),split='train',task_id=0,task_name='fixture'))
        context={'roles':{'roles':{'meta_support':rows}},'protocol':{'dataset':{'root':str(self.root)}},
            'method':{'preprocessing':{'image_size':32,'mean':[0,0,0],'std':[1,1,1]}}}
        args=type('Args',(),dict(work_dir=self.root/'work',w6_dir=self.root/'w6',device='cpu'))()
        args.work_dir.mkdir()
        for method in ('M','C1','A1'):
            model=HOPEFeaturePredictorV4(VisualBlockConfig(8,8,8,4,4,4),method=method,beta=.01)
            identity=dict(roles_digest='roles',schedule_digest='schedule',source_sha256={'fixture.py':'fixture'},spec_sha256={'fixture.md':'fixture'},feature_identity={'backbone_sha256':'a'*64})
            initial=export_initialization(model,identity)
            path=args.w6_dir/'selection'/method/'deployment_initialization.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(initial,path)
        def construct(initial,context,identity,device):
            return HOPEBenchmarkAdapterV4(initial,checkpoint_path='fixture.safetensors',checkpoint_sha256='a'*64,checkpoint_identity='fixture',
                extractor=TinyExtractor(),protocol_identity=identity,allow_test_components=True)
        with patch('notebook_support.recovery_stage1.preflight.subprocess.call',return_value=0),\
             patch('notebook_support.recovery_stage1.preflight.adapter_from_initial',side_effect=construct),\
             patch('notebook_support.recovery_stage1.cadic_profile.bounded_probe',return_value={'status':'passed','full_budget_exercised':True}),redirect_stdout(StringIO()):
            result=run_preflight(args,context,{'digest':'suite'},{'digest':'study'},None)
        self.assertEqual(result['status'],'recovery_preflight_passed')
        report=json.loads((args.work_dir/'recovery_preflight.json').read_text())
        self.assertFalse(report['official_test_accessed'])
        self.assertTrue(all(v['paired_state_rng_score_resume'] for v in report['methods'].values()))

    def test_pass_boolean_without_supporting_evidence_is_rejected(self):
        from notebook_support.recovery_stage1.preflight import validate_preflight
        atomic_json({'passed':True,'study_digest':'study'},self.root/'recovery_preflight.json')
        with self.assertRaisesRegex(ValueError,'supporting receipt'):validate_preflight(self.root,{'digest':'study'})

    def test_full_supporting_gate_passes_and_wrong_cadic_budget_is_rejected(self):
        from notebook_support.recovery_stage1.preflight import validate_preflight
        from notebook_support.recovery_stage1.common import digest
        source={'frozen':{'fixture.py':'sha'},'extras':{},'protocol':{}}
        study={'digest':'study','identity':{'source':source}}
        resource={'status':'passed','full_budget_exercised':True,'normal_train_images_completed':16,
            'official_test_accessed':False,'algorithm_changed':False,'execution_optimization':'cadic_exact_pair_tile_cache_v2',
            'reference_parity':dict(backend='cadic_exact_pair_tile_cache_v2',bank_count=10000,candidate_vectors=4,
                **{k:True for k in ('passed','pair_exact','decisions_and_counters_exact','features_exact','pixel_and_image_scores_exact','main_state_rng_unchanged')}),
            'binding':dict(fit_task_and_snapshot_shared=True,registry_wrapper=True,cache_backend='cadic_exact_pair_tile_cache_v2'),
            'cache_progress':dict(closest_pair_calls=1),
            'checkpoint_roundtrip_exact':True,'state_rng_readonly':True,'cleanup_complete':True,
            'coreset':{'count':10000,'budget':10000,'dim':768,'update_batches':16}}
        methods={name:{'paired_state_rng_score_resume':True,'diagnostic_state_rng_unchanged':True,'normal_events_per_history':9} for name in ('M','C1','A1')}
        gate={'schema':'recovery_stage1_preflight_v1','passed':True,'study_digest':'study','official_test_accessed':False,
            'calibration_refitted':False,'CADIC':resource,'methods':methods}
        atomic_json(gate,self.root/'recovery_preflight.json')
        atomic_json({'success':True,'tests_run':130,'source_sha256':source['frozen']},self.root/'verification_frozen.json')
        atomic_json({'success':True,'tests_run':45,'operations_source_digest':digest(source)},self.root/'verification_recovery.json')
        atomic_json(resource,self.root/'cadic_resource_probe.json')
        self.assertTrue(validate_preflight(self.root,study)['passed'])
        resource['coreset']['budget']=5000;gate['CADIC']=resource
        atomic_json(resource,self.root/'cadic_resource_probe.json');atomic_json(gate,self.root/'recovery_preflight.json')
        with self.assertRaisesRegex(ValueError,'geometry or update unit'):validate_preflight(self.root,study)
        resource['coreset']['budget']=10000;resource.pop('reference_parity');gate['CADIC']=resource
        atomic_json(resource,self.root/'cadic_resource_probe.json');atomic_json(gate,self.root/'recovery_preflight.json')
        with self.assertRaisesRegex(ValueError,'reference parity'):validate_preflight(self.root,study)
        resource['binding']['fit_task_and_snapshot_shared']=False;gate['CADIC']=resource
        atomic_json(resource,self.root/'cadic_resource_probe.json');atomic_json(gate,self.root/'recovery_preflight.json')
        with self.assertRaisesRegex(ValueError,'shared registered cache'):validate_preflight(self.root,study)

    def test_only_registered_recovery_jobs_allowed(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):runner.main(['--phase','controls','--jobs','M_seed1','--work-dir',str(self.root),'--w6-dir',str(self.root)])
    def test_final_only_runner_resumes_without_training_or_rescoring_or_matrix(self):
        d=PathsFixture(self.root);args=type('Args',(),dict(work_dir=self.root/'work',w6_dir=self.root/'w6',device='cpu'))()
        context={};suite={'digest':'suite'};study={'digest':'study'}
        manifest={'digest':'manifest'};protocol={'id':'fixture'};method={}
        built=[]
        def make(*a,**k):
            x=Adapter();built.append(x)
            return x,d,{'fixture':True},method,protocol,manifest
        with patch.object(runner,'job_protocol',return_value=(protocol,manifest)),patch.object(runner,'make_adapter',side_effect=make),\
             patch.object(runner.W7Engine,'matrix',side_effect=AssertionError('must never evaluate matrix')),redirect_stdout(StringIO()):
            runner.run_control('CADIC10000_seed0',context,suite,study,args,None)
            self.assertEqual(built[0].scores,12)
            runner.run_control('CADIC10000_seed0',context,suite,study,args,None)
            self.assertEqual(built[1].scores,0)
        final=args.work_dir/'runs/CADIC10000_seed0/recovery_completion.json'
        self.assertFalse(json.loads(final.read_text())['matrix_complete'])
        self.assertFalse(json.loads(final.read_text())['calibration_refitted'])

if __name__=='__main__':unittest.main()