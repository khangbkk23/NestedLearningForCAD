import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from test_outer_w4 import fixture
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.visual_memory import _finite
from scripts.hope_cad_v2.preflight_outer import verify_outer_resume
from training.hope_outer_learning_v2 import fit, make_schedule
from training.hope_selection_v3 import search
from training.hope_search_journal_v3 import SearchJournal


CONFIG = VisualBlockConfig(4, 3, 4, 2, 4, 4)


def numeric_failure(*args, **kwargs):
    # Exercise the actual exception emitted inside the visual operator.
    _finite('scan result', torch.tensor(float('nan')))


class OuterPreflightContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_real_fit_path_serializes_optimizer_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            rng = torch.get_rng_state().clone()
            report = verify_outer_resume(fixture(), root, config=CONFIG)
            self.assertTrue(report['passed'])
            self.assertTrue(report['exact_weights_optimizer_rng_resume'])
            self.assertFalse(report['selected_initialization_exported'])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse((Path(root)/'deployment_initialization.pt').exists())
            checkpoint = torch.load(Path(root)/'interrupted_step1.pt', weights_only=False)
            self.assertTrue(checkpoint['optimizer']['state'])
            self.assertEqual(len(checkpoint['schedule']['episodes']), 200)
            self.assertTrue(checkpoint['smoke'])

    def test_corrupt_optimizer_resume_is_rejected_by_preflight(self):
        original = torch.load
        def corrupt(path, **kwargs):
            payload = original(path, **kwargs)
            if Path(path).name == 'interrupted_step1.pt':
                for state in payload['optimizer']['state'].values():
                    state['exp_avg'].add_(0.25)
            return payload
        with tempfile.TemporaryDirectory() as root, patch('torch.load', side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, 'resume mismatch'):
                verify_outer_resume(fixture(), root, config=CONFIG)

    def test_visual_numeric_fit_failure_is_terminal_and_never_retried(self):
        self._assert_terminal_failure(gate=False)

    def test_visual_numeric_gate_failure_is_terminal_and_never_retried(self):
        self._assert_terminal_failure(gate=True)

    def _assert_terminal_failure(self, *, gate):
        data = fixture()
        schedule = make_schedule(data, 0, steps=1)
        calls = []
        def factory(method, beta, selected):
            calls.append(beta)
            return HOPEFeaturePredictor(CONFIG, method, beta, selected)
        target = ('training.hope_cms_stress_v3.validation_loss' if gate
                  else 'training.hope_outer_learning_v2.episode_loss')
        options = dict(max_steps=0, checkpoints=(0,)) if gate else dict(max_steps=1, checkpoints=(0, 1))
        with tempfile.TemporaryDirectory() as root:
            journal = SearchJournal(root, data)
            save = journal.save_candidate
            def save_then_interrupt(*args):
                save(*args)
                raise KeyboardInterrupt('interrupt after durable failure, before search return')
            with patch(target, side_effect=numeric_failure), patch.object(journal, 'save_candidate', side_effect=save_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    search(factory, data, schedule, smoke=True, fit_options=options, journal=journal)
            failed = journal.load_candidate('M', 1e-4, 1e-4)
            if gate:
                self.assertFalse(failed['gate']['passed'])
                self.assertEqual(failed['gate']['first_failure']['event'], 0)
            else:
                self.assertEqual(failed['status'], 'fit_failed')
                self.assertIsNone(failed['shortlist'])
                self.assertEqual(failed['first_failure']['step'], 1)
            calls.clear()
            with patch(target, side_effect=numeric_failure):
                outcome = search(factory, data, schedule, smoke=True, fit_options=options, journal=journal)
            self.assertEqual(calls, [1e-3, 1e-2])
            self.assertEqual(outcome['status'], 'selection_failed')
            with patch('training.hope_selection_v3.fit', side_effect=AssertionError('terminal fit retried')), \
                 patch('training.hope_selection_v3.run_stress_gate', side_effect=AssertionError('terminal gate retried')):
                self.assertEqual(search(factory, data, schedule, smoke=True, fit_options=options,
                                        journal=journal)['status'], 'selection_failed')

    def test_contract_errors_are_not_mislabeled_as_numerical_candidates(self):
        data = fixture()
        with patch('training.hope_outer_learning_v2.episode_loss', side_effect=ValueError('wrong shape')):
            with self.assertRaisesRegex(ValueError, 'wrong shape'):
                fit(HOPEFeaturePredictor(CONFIG), data, make_schedule(data, 0, steps=1),
                    max_steps=1, checkpoints=(0, 1), smoke=True)


if __name__ == '__main__':
    unittest.main()
