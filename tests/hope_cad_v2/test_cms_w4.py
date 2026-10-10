import copy
import unittest
import torch
from torch.func import functional_call
from models.hope_cad_v2.cms import ContinuumMemory, masked_teacher_loss
from models.hope_cad_v2.teacher_head import TeacherDecoder, frozen_teacher
from models.hope_cad_v2.predictor import HOPEFeaturePredictor, dynamics
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.state import tensor_digest
from training.hope_cms_session_v2 import CMSSession
from training.hope_outer_learning_v2 import select_candidate
from dataset.hope_mask_views import patch_masks, mask_normalized_rgb


class CMSContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(107)
        self.cms = ContinuumMemory(3, (5, 5)).double()
        self.decoder = TeacherDecoder(3, 4).double()
        self.x = torch.randn(1, 4, 3, dtype=torch.float64)
        self.target = frozen_teacher(torch.randn(1, 4, 4, dtype=torch.float64))
        self.mask = torch.tensor([True, False, True, False])

    def run_events(self, meta, count=16):
        state = self.cms.initial_state(meta=meta)
        for _ in range(count):
            state = self.cms.event(self.x, self.target, self.mask, self.decoder, state, meta=meta).state
        return state

    def test_mean_accumulation_pre_event_oracle_and_clock(self):
        state = self.cms.initial_state(meta=True)
        for t in range(1, 17):
            h1 = functional_call(self.cms.levels[0], state.weights[0], (self.x,))
            h2 = functional_call(self.cms.levels[1], state.weights[1], (h1,))
            expected = []
            for i, h in enumerate((h1, h2)):
                grad = torch.autograd.grad(masked_teacher_loss(self.decoder(h), self.target, self.mask),
                                           tuple(state.weights[i].values()), retain_graph=True)
                acc = {k: state.accumulators[i][k]+g for k, g in zip(state.weights[i], grad)}
                expected.append({k: v-self.cms.rates[i]*acc[k]/(state.pending[i]+1)
                                 if t % self.cms.periods[i] == 0 else v for k, v in state.weights[i].items()})
            result = self.cms.event(self.x, self.target, self.mask, self.decoder, state, meta=True)
            for actual, wanted in zip(result.state.weights, expected):
                for name in actual:
                    torch.testing.assert_close(actual[name], wanted[name], atol=1e-12, rtol=1e-12)
            state = result.state
        self.assertEqual(state.commits, (16, 2))
        self.assertEqual(state.pending, (0, 0))

    def test_functional_deployment_numerical_parity_and_no_history(self):
        a, b = self.run_events(True), self.run_events(False)
        for x, y in zip(a.weights, b.weights):
            for k in x:
                torch.testing.assert_close(x[k], y[k], atol=1e-12, rtol=1e-12)
                self.assertIsNone(y[k].grad_fn)

    def test_exact_16_meta_gradient_matches_central_difference(self):
        parameter = self.cms.levels[1].mlp[0].weight
        def objective():
            state = self.run_events(True)
            return masked_teacher_loss(self.decoder(self.cms(self.x, state)[-1]), self.target, self.mask)
        loss = objective()
        analytic = torch.autograd.grad(loss, parameter)[0][0, 0].item()
        value, eps = parameter[0, 0].item(), 1e-5
        with torch.no_grad(): parameter[0, 0] = value + eps
        plus = objective().item()
        with torch.no_grad(): parameter[0, 0] = value - eps
        minus = objective().item()
        with torch.no_grad(): parameter[0, 0] = value
        self.assertAlmostEqual(analytic, (plus-minus)/(2*eps), delta=2e-7)

    def test_outer_dependencies_to_input_decoder_and_both_initials(self):
        x = self.x.clone().requires_grad_()
        state = self.cms.initial_state(meta=True)
        for _ in range(16):
            state = self.cms.event(x, self.target, self.mask, self.decoder, state, meta=True).state
        loss = masked_teacher_loss(self.decoder(self.cms(x, state)[-1]), self.target, self.mask)
        groups = (x, self.decoder[0].weight, self.cms.levels[0].mlp[0].weight, self.cms.levels[1].mlp[0].weight)
        grads = torch.autograd.grad(loss, groups, create_graph=True)
        self.assertTrue(all(bool(torch.isfinite(g).all()) and g.norm() > 0 for g in grads))
        second = torch.autograd.grad(sum(g.square().sum() for g in grads), groups)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in second))

    def test_resume_mid_period_and_reject_corrupt_clock(self):
        state = self.run_events(False, 5)
        restored = self.cms.load_state(state.checkpoint())
        self.assertEqual(restored.pending, (0, 5))
        for _ in range(11):
            restored = self.cms.event(self.x, self.target, self.mask, self.decoder, restored).state
        self.assertEqual(tensor_digest(restored.weights), tensor_digest(self.run_events(False).weights))
        bad = state.checkpoint()
        bad['pending'][1] = 0
        with self.assertRaises(ValueError): self.cms.load_state(bad)

    def test_atomic_failure_cannot_mutate_state_or_initials(self):
        state = self.run_events(False, 7)
        before = tensor_digest((*state.weights, *state.accumulators))
        initial = tensor_digest(self.cms.initial_state().weights)
        with self.assertRaises(FloatingPointError):
            self.cms.event(self.x, self.target * float('nan'), self.mask, self.decoder, state)
        self.assertEqual(before, tensor_digest((*state.weights, *state.accumulators)))
        self.assertEqual(initial, tensor_digest(self.cms.initial_state().weights))

    def test_loss_sums_channels_not_channel_mean(self):
        self.assertEqual(masked_teacher_loss(torch.ones(1, 4, 4), torch.zeros(1, 4, 4), self.mask).item(), 2)

    def test_c4_inherited_beta_and_distinct_commit_counts(self):
        with self.assertRaises(ValueError): dynamics('C4')
        d = dynamics('C4', beta=1e-2, selected_m_beta=1e-3, dim=3)
        self.assertEqual(d.rates, (1e-3, 1e-3/8))
        self.cms = ContinuumMemory(3, d.hidden, d.periods, d.rates).double()
        self.assertEqual(self.run_events(False).commits, (16, 16))


class SessionContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = HOPEFeaturePredictor(VisualBlockConfig(4, 3, 4, 2, 4, 4)).double()
        self.features = torch.randn(1, 16, 4, dtype=torch.float64)
        self.target = frozen_teacher(torch.randn_like(self.features))
        self.masks = patch_masks(4, 4)

    def factory(self, view):
        return self.features, self.target, self.masks[view]

    def test_batch_transport_resume_readonly_and_order(self):
        a, b = CMSSession(self.model, {'dataset': 'fixture'}), CMSSession(self.model, {'dataset': 'fixture'})
        a.update_transport([self.factory]*16)
        for _ in range(5): b.update(self.factory)
        payload = b.checkpoint()
        b = CMSSession(self.model, {'dataset': 'fixture'})
        b.restore(payload)
        for _ in range(11): b.update(self.factory)
        self.assertEqual(tensor_digest(a.state.weights), tensor_digest(b.state.weights))
        before = tensor_digest((*b.state.weights, *b.state.accumulators))
        rng = torch.get_rng_state().clone()
        first = b.predict(self.features)
        b.predict(self.features * 2)
        torch.testing.assert_close(first, b.predict(self.features), atol=0, rtol=0)
        self.assertEqual(before, tensor_digest((*b.state.weights, *b.state.accumulators)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(b.cursor, 16)

    def test_rng_cursor_state_rollback(self):
        session = CMSSession(self.model, {'fixture': True})
        before = torch.get_rng_state().clone()
        def broken(view):
            torch.rand(17)
            return self.features, self.target*float('nan'), self.masks[view]
        with self.assertRaises((FloatingPointError, ValueError)): session.update(broken)
        self.assertEqual(session.cursor, 0)
        self.assertEqual(session.state.clock, 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class MaskSelectionContracts(unittest.TestCase):
    def test_masks_cover_rgb_and_tokens_once(self):
        masks = patch_masks()
        self.assertEqual(masks.sum(1).tolist(), [196]*4)
        self.assertTrue(bool((masks.sum(0) == 1).all()))
        image = torch.ones(1, 3, 224, 224)
        masked = mask_normalized_rgb(image, 0)
        self.assertEqual((masked == 0).sum().item(), 196*64*3)
        self.assertTrue(bool(image.all()))

    def test_failed_fit_is_never_rescued_and_gate_does_not_rank(self):
        def candidate(loss, step, beta, passed, status='fit_complete'):
            return dict(status=status, gate={'passed': passed}, beta=beta, shortlist={'loss': loss, 'step': step})
        a = candidate(.2, 25, 1e-3, True)
        b = candidate(.1, 0, 1e-4, False)
        c = candidate(.01, 0, 1e-2, True, status='fit_failed')
        self.assertIs(select_candidate([a, b, c])['selected'], a)
        self.assertEqual(select_candidate([b, c])['status'], 'selection_failed')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
