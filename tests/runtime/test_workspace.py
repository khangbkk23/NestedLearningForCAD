"""Workspace boundaries and immutable session exports; no GPU or real data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hope_cad.paths import ROOT, WORKSPACE, session_export_directory
from hope_cad.runtime.identity import output_directory
from hope_cad.data.download import asset_destination
from hope_cad.runtime.budget import Budget
from hope_cad.runtime.evidence import export_archive, restore_archive


class WorkspaceTests(unittest.TestCase):
    def test_inputs_outputs_have_separate_namespaces(self):
        run=WORKSPACE/'experiments/example/run'
        self.assertEqual(output_directory(run,data_root=WORKSPACE/'data/mvtec',checkpoint=WORKSPACE/'weights/vit'),run)
        self.assertEqual(output_directory(WORKSPACE/'checks/local'),WORKSPACE/'checks/local')
        for name in ('workspace','workspace/data','workspace/weights','workspace/records',
                     'workspace/experiments','workspace/experiments/example',
                     'workspace/experiments/example/exports','workspace/experiments/example/run/child','src'):
            with self.subTest(name=name),self.assertRaises(ValueError):output_directory(ROOT/name)
        for name in ('workspace/experiments/a/run/file','workspace/records/a','workspace/data','src/a'):
            with self.subTest(name=name),self.assertRaises(ValueError):asset_destination(ROOT/name)
        with self.assertRaises(ValueError):output_directory(run,data_root=run/'images')
        with self.assertRaises(ValueError):output_directory(run,checkpoint=run/'vit')

    def test_sessions_never_overwrite_and_restore_retains_reserved_ledger(self):
        with tempfile.TemporaryDirectory() as t:
            work=Path(t)/'run';work.mkdir()
            b=Budget(work,'B',1);b.begin_export()
            slot=session_export_directory(work)
            receipt=export_archive(work,slot/'evidence.zip',schema='fixture')
            b.finish(receipt,accounting_path=slot/'accounting.json')
            original={p.name:p.read_bytes() for p in slot.iterdir()}
            slot2=session_export_directory(work)
            self.assertEqual(slot2.name,'session_02')
            # Failed export leaves its slot allocated; it must never be reused.
            self.assertEqual(session_export_directory(work).name,'session_03')
            self.assertEqual(original,{p.name:p.read_bytes() for p in slot.iterdir()})
            restored=Path(t)/'restored'
            restore_archive(slot/'evidence.zip',restored,schema='fixture')
            ledger=json.loads((restored/'budget_B.json').read_text())
            self.assertEqual(ledger['attempts'][0]['status'],'export_incomplete')
            self.assertEqual(ledger['attempts'][0]['charged_seconds'],3600)
            sidecar=json.loads((slot/'accounting.json').read_text())
            self.assertEqual(sidecar['archive']['sha256'],receipt['sha256'])
            self.assertEqual(sidecar['attempt']['id'],b.entry['id'])
            self.assertEqual(json.loads(b.path.read_text())['attempts'][0]['status'],'closed_after_verified_export')

    def test_export_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            work=Path(t)/'run';work.mkdir()
            # Portable simulation: Windows symlink privilege is not assumed.
            with patch.object(Path,'is_symlink',return_value=True):
                with self.assertRaises(ValueError):session_export_directory(work)
