import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, Dataset
from test_rgb_adapter_w5 import TinyExtractor
from test_outer_w4 import fixture
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.benchmark_adapter import HOPEBenchmarkAdapterV3
from models.hope_cad_v2.state import tensor_digest
from training.hope_outer_learning_v2 import export_initialization, capture_rng, restore_rng, make_schedule
from training.hope_benchmark_engine_v3 import HOPEBenchmarkEngineV3, AtomicBenchmarkArtifacts
from training.benchmark_engine_v1 import _state_equal
from training.hope_feature_contract_v3 import expected_feature_identity, validate_bundle
from training.hope_initialization_v3 import validate_initialization, selection_receipt
from training.hope_runtime_v3 import atomic_torch_save, source_identity
from training.hope_cms_stress_v3 import run_stress_gate
from training.hope_mechanism_v3 import mechanism_report
from training.hope_selection_v3 import search
from training.hope_search_journal_v3 import SearchJournal


def initialization(method='M'):
    torch.manual_seed(41)
    model = HOPEFeaturePredictor(VisualBlockConfig(8,8,8,4,4,4),method)
    return export_initialization(model,dict(roles_digest='roles',schedule_digest='schedule',
        source_sha256={'source':'hash'},spec_sha256={'spec':'hash'},feature_identity={'backbone_sha256':'a'*64}))


def adapter(init):
    return HOPEBenchmarkAdapterV3(init,checkpoint_path='fixture',checkpoint_sha256='a'*64,
        checkpoint_identity='fixture',extractor=TinyExtractor(),allow_test_components=True)


class Images(Dataset):
    def __init__(self,count): self.count=count
    def __len__(self): return self.count
    def __getitem__(self,i):
        return dict(images=torch.linspace(-1,1,3*32*32).reshape(3,32,32)+(i*0.001),labels=0)


class Stream:
    manifest={'digest':'fixture_manifest'}
    task_names=['first','second']
    def build_train_loader(self,i,cap=None):
        return DataLoader(Images(min((6,3)[i],cap or 100)),batch_size=2)


class IntegrationContracts(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_static_controls_never_adapt_across_tasks_and_resume(self):
        for method in ('C0','C1'):
            init=initialization(method)
            a=adapter(init)
            before=copy.deepcopy(a.session.state.checkpoint())
            batch=next(iter(DataLoader(Images(1),batch_size=1)))
            scores=a.score_batch(batch)
            for task in range(2): a.fit_task(task,str(task),DataLoader(Images(5),batch_size=2))
            self.assertEqual(a.session.cursor,10)
            self.assertTrue(_state_equal(before,a.session.state.checkpoint()))
            self.assertTrue(_state_equal(scores,a.score_batch(batch)))
            b=adapter(init)
            b.load_state_dict(a.state_dict())
            self.assertEqual(b.session.cursor,10)
            self.assertTrue(_state_equal(before,b.session.state.checkpoint()))

    def test_rgb_event_resume_matches_uninterrupted_through_slow_commit(self):
        for method in ('M','C0'):
            init=initialization(method)
            full,part=adapter(init),adapter(init)
            rng=capture_rng()
            with tempfile.TemporaryDirectory() as root:
                full_engine=HOPEBenchmarkEngineV3({},full,AtomicBenchmarkArtifacts(Path(root)/'full'))
                full_engine.train(Stream(),checkpoint_every=1)
                expected_rng=capture_rng()
                restore_rng(rng)
                partial_engine=HOPEBenchmarkEngineV3({},part,AtomicBenchmarkArtifacts(Path(root)/'resume'))
                partial_engine.train(Stream(),checkpoint_every=1,stop_after_events=5)
                resumed=adapter(init)
                engine=HOPEBenchmarkEngineV3({},resumed,AtomicBenchmarkArtifacts(Path(root)/'resume'))
                engine.train(Stream(),checkpoint_every=1,resume=True)
                self.assertTrue(_state_equal(full.session.checkpoint(),resumed.session.checkpoint()))
                self.assertTrue(_state_equal(expected_rng,capture_rng()))
                batch=next(iter(DataLoader(Images(1),batch_size=1)))
                self.assertTrue(_state_equal(full.score_batch(batch),resumed.score_batch(batch)))
                self.assertEqual(resumed.session.state.commits,(9,1) if method=='M' else (0,0))

    def test_atomic_save_failure_keeps_last_good_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint.pt'
            atomic_torch_save({'step':5},path)
            good=path.read_bytes()
            with patch('torch.save',side_effect=OSError('disk interrupted')):
                with self.assertRaises(OSError): atomic_torch_save({'step':6},path)
            self.assertEqual(path.read_bytes(),good)
            with patch.object(Path,'replace',side_effect=OSError('replace interrupted')):
                with self.assertRaises(OSError): atomic_torch_save({'step':6},path)
            self.assertEqual(path.read_bytes(),good)

    def test_cache_rejects_forged_contract_and_probe_access(self):
        data=fixture()
        context=dict(method={'extractor':{'checkpoint_sha256':'a'*64,'checkpoint_identity':'fixture'},'preprocessing':{'size':224}},
                     manifest={'digest':'manifest'},lock={'digest':'lock'},roles=data.manifest)
        target=torch.zeros(1,784,768); target[:,:,0]=1
        tensors={**{f'masked_{v}':target for v in range(4)},'teacher':target}
        row={'tensors':tensors,'sha256':tensor_digest((tensors,))}
        bundle=dict(schema='hope_frozen_prefix_features_v3',cache_role='calibration',role_manifest=data.manifest,
                    feature_identity=expected_feature_identity(context),images={i:row for r in ('meta_support','meta_query','normal_validation') for i in data.ids(r)})
        validate_bundle(bundle,context)
        for field in ('mask_sha256','source_sha256','preprocessing_sha256','roles_digest','extractor_sha256','baseline_lock_digest'):
            bad={**bundle,'feature_identity':{**bundle['feature_identity'],field:'WRONG'}}
            with self.subTest(field=field),self.assertRaises(ValueError): validate_bundle(bad,context)
        bad={**bundle,'images':{**bundle['images'],'probe.png':row}}
        with self.assertRaises(ValueError): validate_bundle(bad,context)

    def test_random_initialization_cannot_enter_scientific_run(self):
        data=fixture()
        context=dict(method={'extractor':{'checkpoint_sha256':'a'*64,'checkpoint_identity':'fixture'},'preprocessing':{}},
                     manifest={'digest':'manifest'},lock={'digest':'lock'},roles=data.manifest)
        init=initialization()
        init['provenance'].update(source_sha256=source_identity(),feature_identity=expected_feature_identity(context))
        with self.assertRaisesRegex(ValueError,'selected recipe'): validate_initialization(init,context)

    def test_stress_resume_matches_uninterrupted(self):
        torch.manual_seed(19)
        model=HOPEFeaturePredictor(VisualBlockConfig(4,3,4,2,4,4),beta=1e-4)
        snapshots=[]
        report=run_stress_gate(model,fixture(),checkpoint_callback=lambda p:snapshots.append(p) if p['state']['clock']==16 else None)
        resumed=run_stress_gate(model,fixture(),resume=snapshots[0])
        for key in ('trace','losses','commits','events_completed','passed','feature_reads'):
            self.assertEqual(report[key],resumed[key],key)

    def test_mechanism_controls_preserve_parameters_rng_and_never_fit_probe(self):
        data=fixture()
        config=VisualBlockConfig(4,3,4,2,4,4)
        models={name:HOPEFeaturePredictor(config,name,beta=1e-4) for name in ('M','C0','C1','C2')}
        schedule=make_schedule(data,0,steps=1)
        # Probe tensors are supplied separately; PrefixFeatures never grants probe access.
        x,y,_=data.get(data.ids('meta_support')[0],0,'meta_support')
        from dataset.hope_mask_views import patch_masks
        probe=[('held_out_fixture',[(x,y,mask) for mask in patch_masks(4,4)])]
        rng=capture_rng()
        from models.hope_cad_v2.visual_memory import stability_matched_step
        report=mechanism_report(models,data,schedule,probe)
        self.assertEqual(report['status'],'complete_pending_scientific_review')
        self.assertTrue(_state_equal(rng,capture_rng()))
        from models.hope_cad_v2.visual_memory import stability_matched_step as after
        self.assertIs(after,stability_matched_step)
        self.assertFalse(report['controls']['teacher_statistics']['probe_used_for_fitting'])
        for name in models:
            rows=report['methods'][name]
            self.assertEqual(rows['reset_history']['score_delta_from_initial_rms'],0)
            self.assertIn('prediction_position_variance',rows['after_16_support'])
            if name in ('C0','C1'):
                self.assertEqual(rows['after_support_query_stream']['clock'],0)
                self.assertEqual(rows['after_support_query_stream']['score_delta_from_initial_rms'],0)
        self.assertEqual(set(report['interventions']),{'visual_updates_off_same_learned_weights',
                         'single_scan_readout','no_spectral_cap','no_injection_cap'})

    def test_journal_recovers_search_without_repeating_completed_candidates(self):
        config=VisualBlockConfig(4,3,4,2,4,4)
        data=fixture(); schedule=make_schedule(data,0,steps=1)
        factory=lambda method,beta,parent:HOPEFeaturePredictor(config,method,beta,parent)
        with tempfile.TemporaryDirectory() as root:
            journal=SearchJournal(root,data)
            outcome=search(factory,data,schedule,method='C0',smoke=True,
                           fit_options={'max_steps':1,'checkpoints':(0,1)},journal=journal)
            self.assertEqual(outcome['status'],'selected')
            with patch('training.hope_selection_v3.fit',side_effect=AssertionError('must not rerun')):
                resumed=search(factory,data,schedule,method='C0',smoke=True,journal=journal)
            self.assertTrue(_state_equal(outcome['selected']['shortlist'],resumed['selected']['shortlist']))
            self.assertEqual(len(list(Path(root).glob('*diagnostics_*.json'))),6)
            for candidate in resumed['candidates']:
                self.assertLessEqual(candidate['training'][0]['post_clip_gradient_norm'],1.000001)

    def test_selected_label_without_registered_gate_cannot_certify_weights(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'selection.pt'
            candidate=dict(status='fit_complete',method='M',beta=1e-3,outer_lr=1e-4,role_digest='roles',
                           shortlist=dict(step=25,weights={'w':torch.ones(1)}),gate=dict(passed=True))
            outcome=dict(status='selected',smoke=False,selection_rule_id='prefix_cms_stress_v3',
                         common_seed=0,schedule_digest='schedule',selected=candidate)
            atomic_torch_save(outcome,path)
            with self.assertRaisesRegex(ValueError,'1024-event'): selection_receipt(outcome,path)
            candidate.update(method='C0',outer_lr=3e-5,gate=dict(passed=True,status='not_applicable_static'))
            atomic_torch_save(outcome,path)
            receipt=selection_receipt(outcome,path)
            self.assertEqual(receipt['method'],'C0')
            outcome['smoke']=True
            with self.assertRaisesRegex(ValueError,'real successful'): selection_receipt(outcome,path)


if __name__=='__main__': unittest.main()
