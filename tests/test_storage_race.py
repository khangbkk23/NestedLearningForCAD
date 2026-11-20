"""Actual concurrent ZIP/unlink workload and fail-closed disk error controls."""
import importlib.util,os,sys,tempfile,threading,unittest,zipfile
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nested_replay_cad.storage import census,DiskGuard
SCRATCH=Path(os.environ['NRCAD_LOCAL_ROOT']);SCRATCH.mkdir(parents=True,exist_ok=True)
class CensusTests(unittest.TestCase):
    def test_actual_archive_delete_concurrent_with_repeated_census(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp);errors=[];done=threading.Event()
            def writer():
                try:
                    for i in range(100):
                        folder=root/str(i);folder.mkdir()
                        for j in range(8):(folder/str(j)).write_bytes(b'x'*1024)
                        with zipfile.ZipFile(root/(str(i)+'.zip'),'x') as z:
                            for p in folder.iterdir():z.write(p,p.name)
                        for p in folder.iterdir():p.unlink()
                        folder.rmdir()
                except BaseException as e:errors.append(e)
                finally:done.set()
            worker=threading.Thread(target=writer);worker.start();reads=0
            while not done.is_set():self.assertGreaterEqual(census(root),0);reads+=1
            worker.join();self.assertEqual(errors,[]);self.assertGreater(reads,0)
            self.assertEqual(census(root),sum(p.stat().st_size for p in root.iterdir()))
    def test_checkpoint_publish_retention_and_census_concurrently(self):
        import json,types
        sys.path.insert(0,str(ROOT/'scripts'))
        import budget_checkpoint as cp
        class SyntheticSerializer:
            Tensor=type('NotATensor',(),{})
            @staticmethod
            def save(state,stream):
                with zipfile.ZipFile(stream,'w') as archive:archive.writestr('fixture.json',json.dumps(state))
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp)/'run';root.mkdir();checkpoint=root/'er_uniform/checkpoints';errors=[];done=threading.Event()
            def writer():
                try:
                    for update in (500,1000,1500,2000):cp.save(SyntheticSerializer,checkpoint,{'identity':{'fixture':True},'update':update,'stage':1,'iteration':update})
                except BaseException as error:errors.append(error)
                finally:done.set()
            thread=threading.Thread(target=writer);thread.start()
            while not done.is_set():census(root)
            thread.join();self.assertEqual(errors,[])
            self.assertEqual(sorted(p.name for p in checkpoint.glob('g[0-9]*')),['g01500','g02000'])
            self.assertEqual(len((checkpoint/'ledger.jsonl').read_text(encoding='utf8').splitlines()),4)
            for p in checkpoint.glob('g[0-9]*'):
                meta=json.loads((p/'COMMIT.json').read_text(encoding='utf8'));self.assertEqual(cp.digest(p/'state.pt'),meta['sha256'])
    def test_deterministic_vanished_entry_and_permission_error(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            entry=mock.Mock();entry.stat.side_effect=FileNotFoundError()
            cm=mock.MagicMock();cm.__enter__.return_value=iter([entry])
            with mock.patch('os.scandir',return_value=cm):self.assertEqual(census(temp),0)
            entry.stat.side_effect=PermissionError('denied');cm.__enter__.side_effect=lambda:iter([entry])
            with mock.patch('os.scandir',return_value=cm):
                with self.assertRaises(PermissionError):census(temp)
    def test_missing_root_and_over_cap_still_fail(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            with self.assertRaises(FileNotFoundError):census(Path(temp)/'absent')
            (Path(temp)/'file').write_bytes(b'12345')
            with mock.patch('nested_replay_cad.storage.OUTPUT_CAP',4):
                with self.assertRaises(RuntimeError):DiskGuard(temp).check(force=True)
    def test_outer_and_parent_no_unsafe_census(self):
        import ast,json
        for text in [(ROOT/'scripts/parallel_budget.py').read_text(encoding='utf8'),''.join(json.loads((ROOT/'notebooks/b321.ipynb').read_text(encoding='utf8'))['cells'][0]['source'])]:
            self.assertIn('census(',text);self.assertNotIn("sum(p.stat().st_size for p in",text);ast.parse(text)
if __name__=='__main__':unittest.main()
