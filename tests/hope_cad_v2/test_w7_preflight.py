import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np
import torch

from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.benchmark_adapter_v4 import HOPEBenchmarkAdapterV4
from training.hope_outer_learning_v4 import export_initialization
from scripts.hope_cad_v2.preflight_w7 import run_preflight
from test_rgb_adapter_v4 import TinyExtractor
from scripts.hope_cad_v2.diagnose_w7 import run_interventions
from training.benchmark_engine_v1 import _clone_state,_state_equal
from training.hope_outer_learning_v4 import capture_rng


class W7PreflightContracts(unittest.TestCase):
    def test_actual_w7_preflight_resume_with_reduced_real_cms_and_rgb_fixture(self):
        torch.manual_seed(103)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';data.mkdir();rows=[]
            for i in range(9):
                name=f'{i:03d}.png'
                image=np.random.default_rng(i).integers(0,256,(32,32,3),dtype=np.uint8)
                Image.fromarray(image).save(data/name)
                rows.append(dict(relative_path=name,sha256=hashlib.sha256((data/name).read_bytes()).hexdigest()))
            model=HOPEFeaturePredictorV4(VisualBlockConfig(8,8,8,4,4,4),beta=.001)
            initial=export_initialization(model,dict(roles_digest='fixture',schedule_digest='fixture',
                source_sha256={'fixture':'fixture'},spec_sha256={'fixture':'fixture'},
                feature_identity={'backbone_sha256':'a'*64}))
            w6=root/'w6/selection/M';w6.mkdir(parents=True)
            torch.save(initial,w6/'deployment_initialization.pt')
            context=dict(roles={'roles':{'meta_support':rows}},
                protocol={'dataset':{'root':str(data)}},
                manifest={'online':rows},method={'preprocessing':dict(image_size=32,
                    mean=[.485,.456,.406],std=[.229,.224,.225])})
            args=SimpleNamespace(work_dir=root/'work',w6_dir=root/'w6',device='cpu')
            args.work_dir.mkdir()
            def adapter(value,context,identity,device):
                return HOPEBenchmarkAdapterV4(value,checkpoint_path='fixture',checkpoint_sha256='a'*64,
                    checkpoint_identity='fixture',extractor=TinyExtractor(),device='cpu',
                    protocol_identity=identity,allow_test_components=True)
            def verify(command):
                Path(command[-1]).write_text(json.dumps({'success':True}));return 0
            with patch('scripts.hope_cad_v2.run_w7.adapter_from_initial',side_effect=adapter), \
                 patch('scripts.hope_cad_v2.preflight_w7.subprocess.call',side_effect=verify):
                result=run_preflight(args,context,{'digest':'fixture_suite'})
            self.assertEqual(result['status'],'w7_preflight_passed')
            report=json.loads((args.work_dir/'preflight.json').read_text())
            self.assertTrue(report['exact_state_rng_score_resume'])
            self.assertEqual(report['simulated_task_boundary_after'],6)
            self.assertFalse(report['official_test_accessed'])
            live=adapter(initial,context,{'normal_only_preflight':True},'cpu')
            state=torch.load(args.work_dir/'preflight_engine/resumed/states/task_01.pt',weights_only=False)
            live.load_state_dict(state['adapter'])
            before=_clone_state(live.state_dict());rng=capture_rng()
            run_interventions(live,context,args.work_dir/'interventions.json',
                              past_checkpoint=args.work_dir/'preflight_engine/resumed/states/task_00.pt')
            self.assertTrue(_state_equal(before,_clone_state(live.state_dict())))
            self.assertTrue(_state_equal(rng,capture_rng()))
            self.assertEqual(len(json.loads((args.work_dir/'interventions.json').read_text())['interventions']),7)
            live.close()


if __name__=='__main__':unittest.main()
