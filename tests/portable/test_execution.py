"""CPU fixtures exercise the real CLI orchestration/engine, not reportable remote evidence."""
import copy,json,tempfile,unittest,zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from models.hope_cad_v5_1.session import SupportSession
from models.hope_cad_v5_1.contract import digest,design_digest
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from training.support_benchmark_v5_1.contract import study as original_study
from hope_cad.runtime import source_digest,ROOT
from hope_cad.execution import run

class Extractor:
    def __init__(self,*a,**k):pass
    def close(self):pass

class ExecutionContracts(unittest.TestCase):
    def temporary(self):
        parent=ROOT/'results/portable-fixtures';parent.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)
    def fixture(self,t):
        torch.set_num_threads(2);root=Path(t)
        rows=[dict(relative_path=f't{i//10}/train/good/{i}.png',task_id=i//10,split='train') for i in range(18)]
        tasks=['t0','t1'];manifest=dict(online=rows,task_order=tasks)
        rs=[]
        for task in range(2):
            for label in (0,1):rs.append(dict(relative_path=f't{task}/test/{label}/{task*2+label}.png',task_id=task,label=label))
        tm=dict(rows=rs,digest=digest(rs));gen=torch.Generator().manual_seed(54)
        xs=[torch.randn(1,4,4,generator=gen)*.02+1.7**i for i in range(18)]
        class Normal:
            def __init__(self,*a):pass
            def get(self,row):return xs[int(Path(row['relative_path']).stem)]
        class Evaluation:
            def __init__(self,*a):pass
            def get(self,row):
                i=int(Path(row['relative_path']).stem);m=np.zeros((4,4),dtype=bool)
                if row['label']:m[:2,:2]=True
                return xs[i],m
        def factory(name,device):
            return SupportSession(name.rsplit('_seed',1)[0],seed=9,device='cpu',
                bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),
                model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))
        runtime={'fixture_cpu':True};pre=root/'pre';pre.mkdir()
        binding=dict(source_digest=source_digest(),design_digest=design_digest(),runtime=runtime,
            normal_rows_digest=digest(rows),backbone_sha256='fixture')
        (pre/'preflight.json').write_text(json.dumps(dict(status='remote_preflight_passed_pending_review',remote_acceptance=True,
            binding=binding,binding_digest=digest(binding))))
        cfg=dict(original_study(),test_images=4)
        c=dict(command='train',run_dir=str(root/'run'),job='NESTED_seed0',data_root='synthetic',checkpoint='synthetic',preflight=str(pre),device='cuda:0',session_hours=1.)
        stack=ExitStack()
        for target,value in [
            ('hope_cad.execution.normal_manifest',lambda _: (manifest,digest(rows))),
            ('hope_cad.execution.verify_checkpoint',lambda _: 'fixture'),
            ('hope_cad.execution.runtime_identity',lambda _:runtime),
            ('torch.cuda.is_available',lambda:True),('torch.cuda.empty_cache',lambda:None),
            ('training.runtime.resource_usage',lambda _:dict(fixture=True)),
            ('models.feature_extractors.cadic_vit_v1.CADICViTFeatureExtractor',Extractor),
            ('training.support_benchmark_v5_1.data.NormalFeatures',Normal),
            ('training.support_benchmark_v5_1.data.EvaluationFeatures',Evaluation),
            ('training.support_benchmark_v5_1.data.test_manifest',lambda *args:tm),
            ('training.support_benchmark_v5_1.contract.session_for',factory),
            ('training.support_benchmark_v5_1.contract.study',lambda:cfg)]:
            stack.enter_context(patch(target,value))
        return c,stack,root/'run'

    def test_real_engine_train_then_final_then_matrix_never_retrains_or_changes_states(self):
        with self.temporary() as t:
            c,stack,folder=self.fixture(t)
            with stack:
                self.assertEqual(run(c)['status'],'train_complete_pending_review')
                self.assertFalse((folder/'test_manifest.json').exists())
                with patch('training.support_benchmark_v5_1.engine.train',side_effect=AssertionError('evaluation cannot train')):
                    c['command']='evaluate';self.assertEqual(run(c)['status'],'final_complete_pending_review')
                    owner=folder/'runs/NESTED_seed0'
                    protected={p:p.read_bytes() for p in (owner/'states').rglob('*') if p.is_file()}
                    protected.update({p:p.read_bytes() for p in (owner/'cells').rglob('*') if p.is_file()})
                    c['command']='matrix';self.assertEqual(run(c)['status'],'matrix_complete_pending_review')
                    for p,b in protected.items():self.assertEqual(p.read_bytes(),b)
                    self.assertEqual(len(list((owner/'cells').glob('*/completion.json'))),3)

    def test_evaluation_without_own_training_exports_failure_never_reads_test(self):
        with self.temporary() as t:
            c,stack,folder=self.fixture(t);c['command']='evaluate'
            with stack,patch('training.support_benchmark_v5_1.data.test_manifest',side_effect=AssertionError('early test access')):
                with self.assertRaises(RuntimeError):run(c)
            with zipfile.ZipFile(folder.with_suffix('.zip')) as z:
                self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
                self.assertNotIn('test_manifest.json',z.namelist())

    def test_resource_query_failure_still_exports_and_does_not_claim_success(self):
        with self.temporary() as t:
            c,stack,folder=self.fixture(t)
            with stack,patch('training.runtime.resource_usage',side_effect=RuntimeError('fixture CUDA query failure')):
                with self.assertRaises(RuntimeError):run(c)
            with zipfile.ZipFile(folder.with_suffix('.zip')) as z:
                self.assertEqual(json.loads(z.read('handoff.json'))['status'],'failed_preserve_evidence')
                self.assertIn('unavailable',json.loads(z.read('resources.json'))['resources'])

if __name__=='__main__':unittest.main()
