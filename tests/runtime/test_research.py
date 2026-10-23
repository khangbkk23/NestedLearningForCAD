import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hope_cad.runtime.budget import Budget
from hope_cad.research import contract,job,method_identity,study_digest
from hope_cad.contract import design_digest,digest
from hope_cad.runtime.research import assert_preflight,binding


class ResearchContracts(unittest.TestCase):
    def test_new_identity_does_not_alias_parent_or_other_variants(self):
        values=[digest(method_identity(v)) for v in contract()['study']['variants']]
        self.assertEqual(len(set(values)),9);self.assertNotIn(design_digest(),values)
        with self.assertRaises(ValueError):job('H32_EXACT_seed2','main')
        with self.assertRaises(ValueError):job('R_FO_seed1','extension')
        self.assertEqual(job('R_EXACT_seed2','extension'),('R_EXACT',2))
    def test_budget_reservation_crash_resume_and_caps(self):
        with tempfile.TemporaryDirectory() as t:
            a=Budget(t,'B',4,external_seconds=10);a.begin_export();a.finish(dict(path=str(Path(t)/'run.zip'),sha256='fixture'));self.assertLess(a.remaining(),12*3600)
            b=Budget(t,'B',4) # simulate crash, reservation deliberately not released
            c=Budget(t,'B',4)
            self.assertLess(c.remaining(),4*3600)
            with self.assertRaises(ValueError):Budget(t,'B',5)
            with self.assertRaises(ValueError):Budget(t,'D',1)
        with tempfile.TemporaryDirectory() as t:
            b=Budget(t,'C',4,allocation=dict(stage='C',authorized=True,USD_cap=1,USD_per_hour=2))
            self.assertEqual(b.binding['cap_seconds'],1800)

    def test_long_export_charged_and_failed_export_keeps_full_reservation(self):
        with tempfile.TemporaryDirectory() as t,patch('hope_cad.runtime.budget.time.monotonic') as clock:
            clock.return_value=0.;b=Budget(t,'B',1,external_seconds=10)
            with self.assertRaises(ValueError):b.finish(dict(path=str(Path(t)/'run.zip')))
            clock.return_value=60.;b.begin_export()
            on_disk=json.loads(b.path.read_text())['attempts'][0]
            self.assertEqual(on_disk['status'],'export_incomplete');self.assertEqual(on_disk['charged_seconds'],3610.)
            # A kill or export failure leaves this conservative reservation intact.
            clock.return_value=660.;value=b.finish(dict(path=str(Path(t)/'run.zip'),sha256='verified_fixture'))
            self.assertEqual(value['attempt']['export_seconds'],600.)
            self.assertEqual(value['attempt']['charged_seconds'],850.)
            self.assertTrue((Path(t)/'run.accounting.json').exists())
            self.assertEqual(json.loads(b.path.read_text())['attempts'][0]['status'],'closed_after_verified_export')

    def test_runner_does_not_close_budget_when_export_fails(self):
        from hope_cad.runtime.research import run
        from hope_cad.runtime.identity import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache') as t:
            root=Path(t)/'run'
            c=dict(run_dir=str(root),data_root='fixture',checkpoint='fixture',phase='preflight',device='cpu',expected_gpu='fixture',session_hours=1.)
            with patch('hope_cad.runtime.research.output_directory',return_value=root),\
                 patch('hope_cad.runtime.research.check_remote_runtime',side_effect=ValueError('fixture')),\
                 patch('hope_cad.runtime.research.export_archive',side_effect=OSError('export failed')):
                with self.assertRaises(OSError):run(c)
            entry=json.loads((root/'budget_B.json').read_text())['attempts'][0]
            self.assertEqual(entry['status'],'export_incomplete');self.assertEqual(entry['charged_seconds'],3600.)
            self.assertFalse((Path(t)/'run.accounting.json').exists())
    def test_old_preflight_or_wrong_data_rejected(self):
        identity=binding(dict(device='fixture'), 'rows','checkpoint')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'preflight.json';p.write_text(json.dumps(dict(status='remote_preflight_passed_pending_review',binding=identity,binding_digest=digest(identity))))
            with self.assertRaises(ValueError):assert_preflight(t,identity)
            p.write_text(json.dumps(dict(status='research_preflight_passed_pending_review',binding=identity,binding_digest=digest(identity))))
            assert_preflight(t,identity)
            with self.assertRaises(ValueError):assert_preflight(t,dict(identity,normal_rows_digest='other'))

    def test_explicit_review_rejects_failed_gate_and_binds_receipt_and_source(self):
        from hope_cad.runtime.research import record_review
        from hope_cad.contract import file_sha
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'pilot.json';p.write_text(json.dumps(dict(passed=False)))
            with self.assertRaises(ValueError):record_review(t,'pilot','fixture review')
            p.write_text(json.dumps(dict(passed=True)))
            value=record_review(t,'pilot','fixture review')
            self.assertEqual(value['pilot_receipt_sha256'],file_sha(p));self.assertFalse(value['candidate_adopted'])
            self.assertTrue((Path(t)/'pilot_review.json').exists())

    def test_runtime_resource_failure_still_exports_original_failure(self):
        from hope_cad.runtime.research import run
        from hope_cad.runtime.identity import ROOT
        import zipfile
        with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache') as t:
            c=dict(run_dir=str(Path(t)/'run'),data_root='fixture',checkpoint='fixture',phase='preflight',device='cuda:0',expected_gpu='fixture',session_hours=1.)
            # Use an external temp output: output_directory rejects source/scratch paths by design.
            with patch('hope_cad.runtime.research.output_directory',lambda *a,**k:Path(t)/'run'),\
                 patch('hope_cad.runtime.research.check_remote_runtime',side_effect=ValueError('fixture runtime mismatch')),\
                 patch('hope_cad.runtime.research.resource_usage',side_effect=RuntimeError('fixture resource failure')):
                with self.assertRaises(RuntimeError):run(c)
            with zipfile.ZipFile(Path(t)/'run.zip') as z:
                self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
                self.assertFalse(any('/test/' in n for n in z.namelist()))
                self.assertEqual(json.loads(z.read('budget_B.json'))['attempts'][-1]['status'],'export_incomplete')
            accounting=json.loads((Path(t)/'run.accounting.json').read_text())
            self.assertEqual(accounting['archive']['sha256'],__import__('hope_cad.contract',fromlist=['file_sha']).file_sha(Path(t)/'run.zip'))
            self.assertEqual(accounting['attempt']['status'],'closed_after_verified_export')

    def test_real_successor_engine_hard_crash_replays_same_history_and_bank(self):
        from hope_cad.benchmark.engine import train
        from hope_cad.benchmark.stream import _state_equal
        from tests.models.test_consolidation import actor
        from tests.models.test_session import stream
        from hope_cad.runtime.system import atomic_json
        xs=stream(24);rows=[dict(relative_path=str(i),task_id=0) for i in range(24)]
        class Provider:
            def get(self,row):return xs[int(row['relative_path'])]
        identity=dict(job='H32_EXACT_seed0');tasks=['fixture']
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            reference=train(root/'reference',rows,tasks,identity,Provider(),actor)
            def crash(value,path):
                atomic_json(value,path)
                if Path(path).name=='00013.json':raise RuntimeError('process killed after durable event, before checkpoint')
            with patch('hope_cad.benchmark.engine.atomic_json',crash):
                with self.assertRaises(RuntimeError):train(root/'resumed',rows,tasks,identity,Provider(),actor)
            resumed=train(root/'resumed',rows,tasks,identity,Provider(),actor)
            self.assertTrue(_state_equal(reference.state_dict(),resumed.state_dict()))
            for i in range(24):
                self.assertEqual((root/'reference/events'/f'{i:05d}.json').read_bytes(),(root/'resumed/events'/f'{i:05d}.json').read_bytes())
