from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from dataset.benchmark_manifest_v1 import canonical_digest
from training.hope_initialization_v4 import weights_digest
from training.hope_runtime_v4 import atomic_torch_save
from training.hope_w7_contract import source_hashes
from training.hope_w7_refit import fixed_refit,validate_refit
from training.hope_outer_learning_v4 import capture_rng
from training.benchmark_engine_v1 import _state_equal


class W7RefitContracts(unittest.TestCase):
    def test_fixed_refit_exports_terminal_weights_not_validation_shortlist_and_reuses_completed_fit(self):
        class Data:
            manifest={'digest':'fixture_roles'}
            def ids(self,role):return tuple(f'{role}_{i}' for i in range(50 if role=='meta_support' else 20))
        recipe=dict(method='C1',seed=0,beta=.001,outer_lr=.0003,selected_step=400)
        recipe['digest']=canonical_digest(recipe)
        context={'roles':{'digest':'fixture_roles'},'w7_recipes':{'C1':recipe}}
        calls=[];terminal={}
        def fake_fit(model,data,schedule,**options):
            calls.append(options)
            with torch.no_grad():
                for p in model.parameters():p.add_(.01)
            terminal['digest']=weights_digest(model.state_dict())
            options['checkpoint_callback'](dict(step=400,weights=deepcopy(model.state_dict()),rng=capture_rng()))
            return dict(status='fit_complete',shortlist={'step':25,'weights':{}},first_failure=None)
        with tempfile.TemporaryDirectory() as temp, \
             patch('training.hope_w7_refit.expected_feature_identity',return_value={'fixture':True}), \
             patch('training.hope_w7_refit.fit',side_effect=fake_fit):
            initial=fixed_refit(Data(),context,recipe,Path(temp),seed=1,device='cpu',source=source_hashes())
            self.assertEqual(weights_digest(initial['slow_and_initial_weights']),terminal['digest'])
            self.assertEqual(initial['w7_refit_receipt']['seed'],1)
            self.assertFalse(initial['w7_refit_receipt']['selection_performed'])
            fixed_refit(Data(),context,recipe,Path(temp),seed=1,device='cpu',source=source_hashes())
            self.assertEqual(len(calls),1)
            # Crash after a complete fit/gate but before its initialization export.
            (Path(temp)/'deployment_initialization.pt').unlink()
            recovered=fixed_refit(Data(),context,recipe,Path(temp),seed=1,device='cpu',source=source_hashes())
            self.assertTrue(_state_equal(initial['rng'],recovered['rng']))
            self.assertEqual(len(calls),1)
            wrong=deepcopy(initial);wrong['w7_refit_receipt']['seed']=0
            with self.assertRaises(ValueError):validate_refit(wrong,context)
            wrong=deepcopy(initial);wrong['cms']['clock']=1
            with self.assertRaises(ValueError):validate_refit(wrong,context)
    def test_preserved_failure_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'failure.json').write_text('{}')
            recipe=dict(method='C1',beta=.001,outer_lr=.0003,selected_step=400)
            with patch('training.hope_w7_refit.expected_feature_identity',return_value={}), \
                 patch('training.hope_w7_refit.fit',side_effect=AssertionError('must not train')):
                with self.assertRaisesRegex(ValueError,'preserved'):
                    fixed_refit(None,{},recipe,root,seed=1,device='cpu',source={})


if __name__=='__main__':unittest.main()
