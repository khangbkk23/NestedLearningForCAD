import copy
import unittest
from unittest.mock import patch
import torch
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import ROLES
from dataset.hope_mask_views import patch_masks
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.teacher_head import frozen_teacher
from models.hope_cad_v2.state import tensor_digest
from training.hope_outer_learning_v2 import PrefixFeatures, make_schedule, fit, episode_loss, export_initialization, capture_rng, restore_rng
from training.hope_cms_stress_v3 import run_stress_gate


def fixture():
    roles, offset = {}, 0
    for role, count in zip(ROLES, (16, 6, 4, 6)):
        roles[role] = [dict(relative_path=f'bottle/train/good/{i:03d}.png', task_name='bottle',
                            task_id=0, split='train', sha256='synthetic') for i in range(offset, offset+count)]
        offset += count
    manifest = dict(schema='hope_prefix_roles_v3', first_task='bottle', roles=roles)
    manifest['digest'] = canonical_digest(manifest)
    generator = torch.Generator().manual_seed(17)
    # Raw masked features are independent from the clean target; this is not a CAD experiment.
    x = torch.randn(1, 16, 4, generator=generator)
    target = frozen_teacher(torch.randn(1, 16, 4, generator=generator))
    masks = patch_masks(4, 4)
    data = PrefixFeatures(manifest, lambda identity, view: (x, target, masks[view]))
    return data


class OuterContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.model = HOPEFeaturePredictor(VisualBlockConfig(4, 3, 4, 2, 4, 4))
        self.data = fixture()
        self.schedule = make_schedule(self.data, 0, steps=2)

    def test_exact_episode_reaches_all_active_parameter_groups(self):
        loss = episode_loss(self.model, self.data, self.schedule['episodes'][0])
        loss.backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        self.assertEqual(self.data.reads['meta_support'], 16)
        self.assertEqual(self.data.reads['meta_query'], 1)
        self.assertEqual(self.data.reads['normal_validation'], 0)

    def test_fit_and_optimizer_rng_resume_match(self):
        payloads = []
        full = fit(self.model, self.data, self.schedule, max_steps=2, checkpoints=(0,1,2),
                   smoke=True, checkpoint_callback=lambda p: payloads.append(p))
        self.assertEqual(full['status'], 'fit_complete', full['first_failure'])
        resumed_model = HOPEFeaturePredictor(VisualBlockConfig(4, 3, 4, 2, 4, 4))
        resumed = fit(resumed_model, fixture(), self.schedule, max_steps=2, checkpoints=(0,1,2),
                      smoke=True, resume=payloads[1])
        self.assertEqual(full['short_losses'], resumed['short_losses'])
        self.assertEqual(full['training'], resumed['training'])
        self.assertEqual(full['feature_reads'], resumed['feature_reads'])
        self.assertEqual(tensor_digest((dict(self.model.named_parameters()),)),
                         tensor_digest((dict(resumed_model.named_parameters()),)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required to reproduce device-mapped checkpoint RNG state')
    def test_restore_rng_accepts_checkpoint_loaded_onto_cuda(self):
        state = capture_rng()
        state['torch'] = state['torch'].to('cuda')
        state['cuda'] = [rng_state.to('cuda') for rng_state in state['cuda']]
        restore_rng(state)
        self.assertEqual(torch.get_rng_state().device.type, 'cpu')

    def test_nonfinite_fit_discards_earlier_shortlist(self):
        bad = torch.tensor(float('nan'), requires_grad=True)
        with patch('training.hope_outer_learning_v2.episode_loss', return_value=bad):
            result = fit(self.model, self.data, self.schedule, max_steps=1, checkpoints=(0,1), smoke=True)
        self.assertEqual(result['status'], 'fit_failed')
        self.assertIsNone(result['shortlist'])
        self.assertEqual(result['first_failure']['step'], 1)

    def test_static_control_uses_17_images_and_discloses_inactive_maps(self):
        model = HOPEFeaturePredictor(VisualBlockConfig(4,3,4,2,4,4), method='C0')
        loss = episode_loss(model, self.data, self.schedule['episodes'][0])
        loss.backward()
        self.assertEqual(self.data.reads['meta_support'], 16)
        self.assertEqual(self.data.reads['meta_query'], 1)
        for name, p in model.named_parameters():
            if p.requires_grad: self.assertIsNotNone(p.grad, name)
        self.assertLess(model.metadata()['active_parameter_count'], model.metadata()['parameter_count'])

    def test_full_1024_event_gate_counts_trace_and_initialization_isolation(self):
        self.model = HOPEFeaturePredictor(VisualBlockConfig(4,3,4,2,4,4), beta=1e-4)
        before = tensor_digest((dict(self.model.named_parameters()),))
        rng = torch.get_rng_state().clone()
        # Use actual FP32 CMS dynamics and loss; small stable fixture beta in frozen grid.
        report = run_stress_gate(self.model, self.data)
        self.assertTrue(report['passed'], report['first_failure'])
        self.assertEqual(report['commits'], [1024, 128])
        self.assertEqual(len(report['trace']), 1024)
        self.assertEqual(set(report['losses']), {0,16,64,128,256,512,1024})
        self.assertEqual(before, tensor_digest((dict(self.model.named_parameters()),)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertGreater(report['trace'][6]['levels'][1]['accumulator_norm'], 0)
        self.assertEqual(report['trace'][7]['levels'][1]['pending_before_commit'], 8)

    def test_long_gate_rejects_drift_and_does_not_retry(self):
        with patch('training.hope_cms_stress_v3.validation_loss', side_effect=[1., 1., 3.]):
            report = run_stress_gate(self.model, self.data)
        self.assertFalse(report['passed'])
        self.assertEqual(report['first_failure']['event'], 64)
        self.assertEqual(report['events_completed'], 64)

    def test_export_zero_clock_and_requires_identity(self):
        with self.assertRaises(ValueError): export_initialization(self.model, {})
        report = export_initialization(self.model, {k:'fixture' for k in
                      ('roles_digest','schedule_digest','source_sha256','spec_sha256','feature_identity')})
        self.assertEqual(report['cms']['clock'], 0)
        self.assertEqual(report['cms']['pending'], [0,0])
        self.assertEqual(report['cursor'], 0)

    def test_search_one_shortlist_per_beta_and_c4_inheritance(self):
        from training.hope_selection_v3 import search
        calls = []
        def factory(method,beta,selected):
            calls.append((method,beta,selected))
            return HOPEFeaturePredictor(VisualBlockConfig(4,3,4,2,4,4),method,beta,selected)
        def pretend_fit(model,data,schedule,**kwargs):
            return dict(status='fit_complete',method=model.dynamics.method,beta=model.cms.rates[0],
                        role_digest=data.manifest['digest'],
                        shortlist=dict(loss=1.,step=25,weights=copy.deepcopy(model.state_dict())))
        with patch('training.hope_selection_v3.fit',side_effect=pretend_fit), patch('training.hope_selection_v3.run_stress_gate',return_value={'passed':False}) as gate:
            outcome = search(factory,self.data,self.schedule,smoke=True)
            self.assertEqual(gate.call_count,3)
            self.assertEqual(outcome['status'],'selection_failed')
        with patch('training.hope_selection_v3.fit',side_effect=pretend_fit), patch('training.hope_selection_v3.run_stress_gate',return_value={'passed':True}) as gate:
            selected = search(factory,self.data,self.schedule,smoke=True)
            self.assertEqual(selected['selected']['beta'],1e-4)
            gate.reset_mock()
            calls.clear()
            c4 = search(factory,self.data,self.schedule,method='C4',selected_m=selected,smoke=True)
            self.assertEqual(calls,[('C4',1e-4,1e-4)])
            self.assertEqual(gate.call_count,1)
            self.assertEqual(c4['selected']['shortlist']['step'],25)
            with self.assertRaises(ValueError):
                search(factory,self.data,self.schedule,method='C4',selected_m=selected,smoke=False)

    def test_scientific_selection_requires_canonical_gpu_memory(self):
        from scripts.hope_cad_v2.calibrate_features import (
            MIN_SCIENTIFIC_GPU_BYTES,
            canonical_resource_block,
        )
        self.assertIsNotNone(canonical_resource_block('cpu', False, None))
        self.assertIsNotNone(canonical_resource_block('cuda', True, 4 * 1024**3))
        self.assertIsNone(canonical_resource_block('cuda', True, MIN_SCIENTIFIC_GPU_BYTES))


if __name__ == '__main__': unittest.main()
