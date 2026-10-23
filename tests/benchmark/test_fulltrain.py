import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hope_cad.data.fulltrain import build_manifest
from hope_cad.data.prefix import split_prefix
from hope_cad.data.protocol import CANONICAL_MVTEC_TASKS
from hope_cad.paths import ROOT


class FulltrainContracts(unittest.TestCase):
    def test_manifest_exact_set_difference_thirty_bottle_and_order_no_test_discovery(self):
        # File contents are synthetic; test the real filesystem manifest builder, not images/backbone.
        counts=[209]+[244]*13+[248]
        self.assertEqual(sum(counts),3629)
        with tempfile.TemporaryDirectory(dir=ROOT/'.pytest_cache') as t:
            root=Path(t)
            for task,n in zip(CANONICAL_MVTEC_TASKS,counts):
                folder=root/task/'train/good';folder.mkdir(parents=True)
                for i in range(n):(folder/f'{i:03d}.png').write_bytes(b'fixture')
                (root/task/'test').mkdir();(root/task/'test/DO_NOT_READ.png').write_bytes(b'forbidden')
            m,d=build_manifest(root,'fixture_source')
            roles,old=split_prefix(m['online'],m['task_order'])
            newset={r['relative_path'] for r in m['online']};oldset={r['relative_path'] for r in old}
            self.assertEqual(newset-oldset,{f'bottle/train/good/{i:03d}.png' for i in range(70,100)})
            self.assertEqual(len(old),3599);self.assertEqual(m['excluded'],[])
            self.assertTrue(all('/train/good/' in r['relative_path'] for r in m['online']))
            again,dd=build_manifest(root,'fixture_source');self.assertEqual((m,d),(again,dd))
            (root/'bottle/train/good/208.png').unlink()
            with self.assertRaises(ValueError):build_manifest(root,'fixture_source')
