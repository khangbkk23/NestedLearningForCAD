"""Viewer classifications, including an actual local streamed fixture; no API."""
import contextlib,importlib.util,io,json,sys,unittest
from types import SimpleNamespace,ModuleType
from unittest import mock
from pathlib import Path
spec=importlib.util.spec_from_file_location('watch_kaggle',Path(__file__).resolve().parents[1]/'scripts/watch_kaggle.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class WatchKaggleTests(unittest.TestCase):
    def test_blocked_preflight_cannot_be_done(self):
        p=m.terminal_payload('246s B321_TERMINAL {"status":"BLOCKED_RESOURCE_PREFLIGHT","worker_status":"PREFLIGHT_INSUFFICIENT_HOST_MEMORY"}')
        title,msg,code=m.completion_notice(p)
        self.assertIn('[BLOCKED]',title);self.assertIn('NOT STARTED',title)
        self.assertIn('INSUFFICIENT_HOST_MEMORY',msg);self.assertEqual(code,2)
        self.assertNotIn('[DONE]',title)

    def test_unknown_does_not_claim_training_or_science_pass(self):
        title,msg,code=m.completion_notice(None)
        self.assertIn('AUDIT REQUIRED',title);self.assertIn('NOT verified',msg)

    def test_completed_workload_still_needs_audit(self):
        title,msg,code=m.completion_notice({'status':'COMPLETE_REVIEW_PENDING'})
        self.assertEqual(code,0);self.assertIn('AUDIT PENDING',title);self.assertIn('NOT confirmed',msg)

    def test_failure_cannot_be_complete(self):
        title,msg,code=m.completion_notice({'status':'FAILED'})
        self.assertIn('[FAILED]',title);self.assertEqual(code,1)

    def test_b322_marker_is_recognised(self):
        p=m.terminal_payload('9s B322_TERMINAL {"status":"COMPLETE_REVIEW_PENDING"}')
        self.assertEqual(p,{'status':'COMPLETE_REVIEW_PENDING'})
        title,msg,code=m.completion_notice(p)
        self.assertIn('AUDIT PENDING',title);self.assertEqual(code,0)
        self.assertIsNone(m.terminal_payload('B322_PROFILE {"status":"PROFILE_COMPLETE"}'))

    def test_b324_marker_is_recognised(self):
        p=m.terminal_payload('12s B324_TERMINAL {"status":"FAILED","error":"x"}')
        self.assertEqual(p,{'status':'FAILED','error':'x'})

    def test_bad_and_nonfinal_markers_are_ignored(self):
        for s in ('B321_TERMINAL broken','B321_TERMINAL []','B321_TERMINAL {"status":7}','B321_PARALLEL_TERMINAL {"status":"PASS"}','warning only'):
            self.assertIsNone(m.terminal_payload(s))

    def test_actual_main_complete_provider_blocked_workload(self):
        api=SimpleNamespace(authenticate=lambda:None,config_values={'user':'owner-b'},CONFIG_NAME_USER='user')
        module=ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=lambda:api
        result=SimpleNamespace(returncode=0,stdout='KernelWorkerStatus.COMPLETE',stderr='')
        out=io.StringIO()
        with mock.patch.dict(sys.modules,{'kaggle.api.kaggle_api_extended':module}),mock.patch.object(sys,'argv',['watch','--kernel','owner-b/fixture']),mock.patch.object(m,'follow_logs',return_value={'status':'BLOCKED_RESOURCE_PREFLIGHT','worker_status':'PREFLIGHT_INSUFFICIENT_CAP'}),mock.patch.object(m.subprocess,'run',return_value=result),contextlib.redirect_stdout(out):
            self.assertEqual(m.main(),2)
        self.assertIn('[BLOCKED]',out.getvalue());self.assertNotIn('[DONE]',out.getvalue())

    def test_keyboard_interrupt_stops_only_local_viewer(self):
        class Stream:
            def __iter__(self):raise KeyboardInterrupt()
            def close(self):pass
        p=mock.Mock(stdout=Stream());p.poll.return_value=None
        with mock.patch.object(m.subprocess,'Popen',return_value=p):
            with self.assertRaises(KeyboardInterrupt):m.follow_logs(['local-viewer-fixture'])
        p.terminate.assert_called_once();p.wait.assert_called_once_with(timeout=5)

    def test_actual_local_stream_for_vietnamese_blocked_log(self):
        script='import json;print("Kiểm tra tài nguyên",flush=True);print("B321_TERMINAL "+json.dumps({"status":"BLOCKED_RESOURCE_PREFLIGHT","worker_status":"PREFLIGHT_INSUFFICIENT_CAP"}),flush=True)'
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            p=m.follow_logs([sys.executable,'-I','-S','-B','-X','utf8','-c',script])
        self.assertIn('Kiểm tra tài nguyên',out.getvalue())
        self.assertEqual(m.completion_notice(p)[2],2)

if __name__=='__main__':unittest.main()
