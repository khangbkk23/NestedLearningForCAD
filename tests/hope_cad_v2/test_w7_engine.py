"""Resume/reuse/data-integrity checks for the W7 orchestration, without AD test data."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader

from training.benchmark_engine_v1 import _state_equal
from training.hope_outer_learning_v4 import capture_rng
from training.hope_w7_contract import jobs,job_by_id,MVTEC_COUNTS
from training.hope_w7_engine import W7Engine,SessionLimit,read_cell


class Rows(Dataset):
    def __init__(self,root,rows): self.root,self.rows=root,rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        value=torch.tensor(np.asarray(Image.open(self.root/row['relative_path'])).copy()).float()/255
        return dict(images=value[None].expand(3,-1,-1),labels=row['label'],
                    masks=torch.tensor([[0,1],[0,0]]) if row['label'] else torch.zeros(2,2,dtype=torch.long))


class FixtureData:
    task_names=['a','b','c']
    def __init__(self,root):
        self.root=Path(root);self.train=[];self.test=[]
        for task in range(3):
            train=[];test=[]
            for i in range(3):
                path=f'train_{task}_{i}.png';Image.new('L',(2,2),30+task+i).save(self.root/path)
                train.append(dict(relative_path=path,label=0))
            for i in range(4):
                path=f'test_{task}_{i}.png';Image.new('L',(2,2),40+task+100*(i%2)).save(self.root/path)
                mask=f'mask_{task}_{i}.png' if i%2 else None
                if mask: Image.new('L',(2,2),1).save(self.root/mask)
                test.append(dict(relative_path=path,label=i%2,mask_path=mask))
            self.train.append(train);self.test.append(test)
    def training_paths(self,task): return self.train[task]
    def test_paths(self,task): return self.test[task]
    def build_train_loader(self,task): return DataLoader(Rows(self.root,self.train[task]),batch_size=2)
    def build_test_loader(self,task): return DataLoader(Rows(self.root,self.test[task]),batch_size=2)


class FixtureAdapter:
    def __init__(self): self.value=torch.tensor(0.);self.events=0;self.scores=0;self.mutate=None
    def fit_task(self,task,name,batches):
        for batch in batches:
            for image in batch['images']:
                self.value+=image.mean();self.events+=1
    def state_dict(self): return dict(value=self.value.clone(),events=self.events)
    def load_state_dict(self,state): self.value=state['value'].clone();self.events=state['events']
    def score_batch(self,batch):
        self.scores+=len(batch['images'])
        maps=batch['images'][:,0].clone()+self.value
        maps[:,0,1]+=0.5
        if self.mutate=='state': self.value+=1
        if self.mutate=='rng': torch.rand(1)
        return dict(image_scores=maps.amax((1,2)),anomaly_maps=maps)


class W7EngineContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.data=FixtureData(self.root)
    def tearDown(self): self.temp.cleanup()
    def engine(self,adapter,name): return W7Engine(adapter,self.data,self.root/name,'fixture-lock',checkpoint_every=2)
    def test_mid_task_resume_matches_uninterrupted_and_all_task_checkpoints(self):
        a=FixtureAdapter();self.engine(a,'full').train();expected=a.state_dict()
        b=FixtureAdapter()
        with self.assertRaises(SessionLimit): self.engine(b,'resumed').train(stop_after_events=5)
        c=FixtureAdapter();self.engine(c,'resumed').train()
        self.assertTrue(_state_equal(expected,c.state_dict()))
        self.assertEqual(len(list((self.root/'resumed/states').glob('task_*.pt'))),3)
        self.engine(c,'resumed').train();self.assertEqual(c.events,9)
    def test_batched_baseline_transport_keeps_image_updates_and_resume(self):
        a=FixtureAdapter();self.engine(a,'image_transport').train()
        b=FixtureAdapter();b.train_transport_batch_size=2
        e=self.engine(b,'batch_transport')
        with self.assertRaises(SessionLimit):e.train(stop_after_events=5)
        c=FixtureAdapter();c.train_transport_batch_size=2
        self.engine(c,'batch_transport').train()
        self.assertTrue(_state_equal(a.state_dict(),c.state_dict()))
    def test_final_row_is_reused_by_matrix_without_training_or_rescoring(self):
        a=FixtureAdapter();e=self.engine(a,'reuse');e.train();e.final()
        self.assertEqual(a.scores,12)
        final=(self.root/'reuse/scores/02_00/result.json').read_bytes()
        e.matrix();self.assertEqual(a.scores,24);self.assertEqual(a.events,9)
        self.assertEqual((self.root/'reuse/scores/02_00/result.json').read_bytes(),final)
        e.matrix();self.assertEqual(a.scores,24)
    def test_evaluation_shard_resume_and_future_task_rejection(self):
        a=FixtureAdapter();e=self.engine(a,'eval');e.train();e.load_checkpoint(2)
        with self.assertRaises(SessionLimit):e.evaluate_cell(2,0,stop_after_shards=1)
        self.assertEqual(a.scores,2)
        e.evaluate_cell(2,0);self.assertEqual(a.scores,4)
        arrays=read_cell(self.root/'eval/scores/02_00');self.assertEqual(len(arrays['identities']),4)
        with self.assertRaisesRegex(ValueError,'future'):e.evaluate_cell(0,1)
    def test_state_and_rng_mutation_are_detected_before_committing_scores(self):
        for mode in ('state','rng'):
            with self.subTest(mode=mode):
                a=FixtureAdapter();e=self.engine(a,mode);e.train();e.load_checkpoint(2)
                rng=capture_rng();a.mutate=mode
                with self.assertRaises(RuntimeError):e.evaluate_cell(2,0)
                self.assertTrue(_state_equal(rng,capture_rng()))
                self.assertFalse(list((self.root/mode/'scores/02_00').glob('*.npz')))
    def test_changed_raw_shard_and_training_cursor_fail_closed(self):
        a=FixtureAdapter();e=self.engine(a,'bad');e.train();e.final()
        p=self.root/'bad/scores/02_00/000000.npz';p.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'bytes changed'):e.evaluate_cell(2,0)
        p=self.root/'bad/states/stream_resume.pt';payload=torch.load(p,weights_only=False)
        payload['cursor']+=1;torch.save(payload,p)
        with self.assertRaisesRegex(ValueError,'cursor'):e.train()
    def test_suite_has_all_seeds_budgets_and_orders_without_duplicate_jobs(self):
        plan=jobs();self.assertEqual(len(plan),35)
        self.assertEqual(len({j['id'] for j in plan}),35)
        self.assertEqual(job_by_id('A1_seed2')['seed'],2)
        self.assertEqual(sum(j['method']=='CADIC' for j in plan),10)
        self.assertEqual(tuple(sum(v[i] for v in MVTEC_COUNTS.values()) for i in range(3)),(3629,467,1258))
        with self.assertRaises(ValueError):job_by_id('unregistered')


if __name__=='__main__': unittest.main()
