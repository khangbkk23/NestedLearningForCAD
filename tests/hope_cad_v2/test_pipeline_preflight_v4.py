"""Orchestration contracts: short preflight, full continuation and saved pauses."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.hope_cad_v2 import run_normal_v4 as pipeline


class PipelinePreflightContracts(unittest.TestCase):
    def test_preflight_stops_before_selection_and_main_reuses_its_gates(self):
        calls=[]
        class Process:
            def __init__(self,cmd,**kwargs):
                name=Path(cmd[2]).name
                calls.append(name)
                self.stdout=io.StringIO('fixture subprocess\n')
                self.code=0
                def argument(key): return Path(cmd[cmd.index(key)+1])
                if name=='verify_v4.py': argument('--output').write_text(json.dumps({'success':True,'complete_suite':True,'design_id':'hope_cad_v4_warm_joint_cms_row_retention'}))
                elif name=='profile_v4.py': argument('--output').write_text(json.dumps({'status':'passed'}))
                elif name=='prepare_prefix_features_v4.py': torch.save({'fixture':True},argument('--output'))
                elif name=='preflight_rgb_v4.py':
                    folder=argument('--output-dir'); folder.mkdir()
                    (folder/'preflight.json').write_text(json.dumps({'passed':True}))
                elif name=='preflight_outer_v4.py':
                    folder=argument('--output-dir'); folder.mkdir()
                    (folder/'preflight_outer.json').write_text(json.dumps({'passed':True,'kaggle_gate_eligible':True}))
                elif name=='calibrate_features_v4.py': self.code=7  # deliberate scientific failure
                elif name!='run_matched_cadic.py': raise AssertionError(name)
            def wait(self,timeout=None): return self.code
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'evidence'
            argv=['runner','--work-dir',str(root),'--device','cpu']
            with patch.object(pipeline,'source_identity',return_value={'source':'fixture'}), \
                 patch.object(pipeline,'runtime_identity',return_value={'runtime':'fixture'}), \
                 patch.object(pipeline,'load_context',return_value={}), \
                 patch.object(pipeline,'validate_bundle',side_effect=lambda value,context:value), \
                 patch.object(pipeline.subprocess,'Popen',Process):
                with patch.object(sys,'argv',[*argv,'--preflight-only']):
                    self.assertEqual(pipeline.main(),0)
                self.assertEqual(json.loads((root/'handoff.json').read_text())['status'],'preflight_passed')
                self.assertEqual(calls,['verify_v4.py','profile_v4.py','run_matched_cadic.py',
                                        'prepare_prefix_features_v4.py','preflight_rgb_v4.py','preflight_outer_v4.py'])
                self.assertTrue((root.parent/'hope_cad_v4_normal_evidence.zip').is_file())
                calls.clear()
                with patch.object(sys,'argv',[*argv,'--resume']): self.assertEqual(pipeline.main(),1)
                self.assertEqual(calls,['run_matched_cadic.py','calibrate_features_v4.py'])
                summary=json.loads((root/'handoff.json').read_text())
                self.assertEqual(summary['status'],'stopped')
                self.assertFalse(summary['W7_authorized'])

    def test_session_budget_pause_exports_evidence_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'evidence'
            with patch.object(sys,'argv',['runner','--work-dir',str(root),'--session-hours','1e-12']), \
                 patch.object(pipeline,'source_identity',return_value={}), \
                 patch.object(pipeline,'runtime_identity',return_value={}), \
                 patch.object(pipeline.subprocess,'Popen',side_effect=AssertionError('must not start')):
                self.assertEqual(pipeline.main(),0)
            summary=json.loads((root/'handoff.json').read_text())
            self.assertEqual(summary['status'],'paused_for_session_limit')
            self.assertFalse(summary['engineering_gates_passed'])
            self.assertTrue((root/'artifact_manifest.json').is_file())
            self.assertTrue((root.parent/'hope_cad_v4_normal_evidence.zip').is_file())

    def test_notebook_bootstrap_time_is_subtracted_and_invalid_budgets_fail(self):
        with patch.object(pipeline.time,'monotonic',return_value=5000.):
            self.assertEqual(pipeline.session_deadline(10.5,1000.),38800.)
            self.assertEqual(pipeline.session_deadline(10.5),42800.)
            for hours,start in ((0,None),(-1,None),(float('nan'),None),(float('inf'),None),(1,5001.),(1,float('nan'))):
                with self.assertRaises(ValueError): pipeline.session_deadline(hours,start)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'evidence'
            with patch.object(sys,'argv',['runner','--work-dir',str(root),'--session-hours','1',
                                         '--session-start-monotonic','1']), \
                 patch.object(pipeline.time,'monotonic',return_value=4000.), \
                 patch.object(pipeline,'source_identity',return_value={}), \
                 patch.object(pipeline,'runtime_identity',return_value={}), \
                 patch.object(pipeline.subprocess,'Popen',side_effect=AssertionError('budget spent in bootstrap')):
                self.assertEqual(pipeline.main(),0)
            summary=json.loads((root/'handoff.json').read_text())
            self.assertEqual(summary['status'],'paused_for_session_limit')
            self.assertTrue(summary['budget_includes_notebook_bootstrap'])
            self.assertEqual(summary['elapsed_since_session_start_seconds'],3999.)


if __name__=='__main__': unittest.main()
