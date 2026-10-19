"""Synthetic-only gate; scalar oracle is deliberately independent of torch."""
import io
import unittest
import torch
from . import oracle
from .contcore import CausalContCore, farthest_indices, consolidate_indices
from .distance import Euclidean, configure, global_nn
from .controls import ARMS, MemoryArm, byte_capacities, charged_bytes


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        configure()

    def test_empty_nonempty_base_index_domains(self):
        for x,b in [([[10.],[0.]],[[9.]]),([[1.],[7.]],[[0.],[2.],[3.],[4.],[10.]]),
                    ([[0.],[2.],[8.],[10.]],[]),([[1.],[2.],[5.]],[[3.]])]:
            xx=torch.tensor(x);bb=torch.tensor(b).reshape(-1,1);ids=torch.arange(len(x))
            for n in range(len(x)+1):
                actual=farthest_indices(xx,ids,n,bb).tolist()
                self.assertEqual(actual,oracle.greedy(x,ids.tolist(),n,b))
        self.assertEqual(farthest_indices(torch.tensor([[10.],[0.]]),torch.arange(2),1,torch.tensor([[9.]])).tolist(),[1])

    def test_known_farthest(self):
        x=torch.tensor([[0.],[2.],[8.],[10.]])
        self.assertEqual(farthest_indices(x,torch.arange(4),3,x[:0]).tolist(),[0,3,1])

    def test_duplicates_exact_ties_serial_order(self):
        for x in [torch.zeros(7,3),torch.tensor([[-1.],[1.],[-1.],[1.],[0.]])]:
            ids=torch.arange(len(x)).flip(0)
            actual=farthest_indices(x,ids,len(x),x[:0]).tolist()
            self.assertEqual(actual,oracle.greedy(x.tolist(),ids.tolist(),len(x),[]))
            self.assertEqual(len(set(actual)),len(x))
        self.assertEqual(consolidate_indices(torch.zeros(8,2),torch.arange(8),4,.75).tolist(),[0,1,2,3])

    def test_consolidation_q_zero_and_three_quarters(self):
        g=torch.Generator().manual_seed(3101)
        for count,k in [(4,6),(6,6),(12,5),(16,8)]:
            x=torch.randint(-9,10,(count,3),generator=g).float()
            ids=torch.randperm(count,generator=g)
            for q in [0,.75,1]:
                self.assertEqual(consolidate_indices(x,ids,k,q).tolist(),oracle.consolidate(x.tolist(),ids.tolist(),k,q))

    def test_fill_saturation_replacement_scalar_trajectory(self):
        for q in [0,.75]:
            m=CausalContCore(7,2,patches=6,candidates=3,q=q)
            b,s=[],[]
            g=torch.Generator().manual_seed(42)
            for event in range(9):
                x=(torch.randint(-50,50,(6,2),generator=g)+event*40).float()
                m.update(x,event=event)
                b,s=oracle.update(b,s,x.tolist(),event,7,3,q)
                self.assertEqual(m.serials.tolist(),s)
                self.assertEqual(m.bank.tolist(),b)
                self.assertEqual(len(m.bank),min((event+1)*3,7))
            self.assertTrue(any(i>=6*3 for i in s))

    def test_full_grid_78_admissions(self):
        m=CausalContCore(dim=3)
        stats=m.update(torch.randn(784,3,generator=torch.Generator().manual_seed(5)),event=0)
        self.assertEqual(stats['observed'],784);self.assertEqual(len(m.bank),78)
        with self.assertRaises(ValueError):m.update(torch.zeros(78,3),event=1)

    def test_chunk_numerical_parity_direct_scalar(self):
        g=torch.Generator().manual_seed(17)
        x=torch.randn(19,9,generator=g);y=torch.randn(23,9,generator=g)
        direct=torch.tensor([[oracle.distance(a,b) for b in y.tolist()] for a in x.tolist()])
        for chunk in [1,3,16,100]:
            d=Euclidean(chunk).matrix(x,y)
            torch.testing.assert_close(d,direct,rtol=2e-6,atol=2e-6)
        x=torch.tensor([[1e6,1e6],[1e6+.125,1e6],[1e6,1e6]])
        for chunk in [1,2,20]:
            d=Euclidean(chunk).matrix(x,x)
            torch.testing.assert_close(d,torch.tensor([[0,.125,0],[.125,0,.125],[0,.125,0]]),rtol=0,atol=0)

    def test_chunk_selection_parity(self):
        g=torch.Generator().manual_seed(77)
        stream=[torch.randint(-20,20,(12,3),generator=g).float() for _ in range(7)]
        models=[CausalContCore(13,3,12,5,chunk=c) for c in [1,4,256]]
        for t,x in enumerate(stream):
            for m in models:m.update(x,event=t)
            for m in models[1:]:self.assertTrue(torch.equal(m.serials,models[0].serials))

    def test_full_dimension_distance_and_close_vectors(self):
        g=torch.Generator().manual_seed(280)
        x=torch.randn(37,768,generator=g);y=torch.cat((x[:5],x[:5]+1e-4,torch.randn(41,768,generator=g)))
        direct=(x[:,None,:].double()-y[None,:,:].double()).square().sum(2).sqrt().float()
        for chunk in [7,256]:
            d=Euclidean(chunk).matrix(x,y)
            torch.testing.assert_close(d,direct,rtol=2e-6,atol=2e-6)

    def test_no_replay_future_or_self_witness(self):
        m=CausalContCore(4,1,patches=4,candidates=2)
        m.update(torch.tensor([[0.],[4.],[7.],[10.]]),event=0)
        state=m.state_dict()
        for event in [0,2,10,-1]:
            with self.assertRaises(ValueError):m.update(torch.ones(4,1),event=event)
            self.assertTrue(torch.equal(m.bank,state['bank']))
        self.assertEqual(set(vars(m)),{'capacity','dim','patches','candidates','q','chunk','bank','serials','next_event'})
        self.assertEqual(farthest_indices(torch.tensor([[0.],[9.]]),torch.tensor([2,3]),1,torch.tensor([[0.]])).tolist(),[1])

    def test_score_read_only_and_input_ownership(self):
        x=torch.randn(8,3,generator=torch.Generator().manual_seed(11))
        m=CausalContCore(5,3,8,3);m.update(x,event=0)
        before=m.state_dict();d=m.score(x);x.fill_(999)
        torch.testing.assert_close(m.bank,before['bank'],rtol=0,atol=0)
        torch.testing.assert_close(m.score(before['bank']),torch.zeros(3),rtol=0,atol=0)
        self.assertEqual(m.next_event,before['next_event']);self.assertTrue(torch.isfinite(d).all())
        before['bank'].zero_();self.assertFalse(torch.equal(before['bank'],m.bank))

    def test_serialization_deterministic_resume(self):
        g=torch.Generator().manual_seed(60)
        stream=[torch.randint(-100,100,(8,4),generator=g).float() for _ in range(6)]
        a=CausalContCore(11,4,8,4)
        for t,x in enumerate(stream[:3]):a.update(x,event=t)
        f=io.BytesIO();torch.save(a.state_dict(),f);f.seek(0)
        b=CausalContCore.from_state_dict(torch.load(f,weights_only=True))
        for t in range(3,6):
            a.update(stream[t],event=t);b.update(stream[t],event=t)
            self.assertTrue(torch.equal(a.bank,b.bank));self.assertTrue(torch.equal(a.serials,b.serials))
        bad=a.state_dict();bad['serials'][0]=-1
        with self.assertRaises(ValueError):CausalContCore.from_state_dict(bad)

    def test_controls_common_nn_readonly_resume_and_cache_ledger(self):
        g=torch.Generator().manual_seed(63)
        stream=[torch.randn(784,2,generator=g) for _ in range(2)]
        for name in ARMS:
            with self.subTest(arm=name):
                a=MemoryArm(name,capacity=32,dim=2)
                a.update(stream[0],event=0)
                before=a.snapshot();stream_file=io.BytesIO();torch.save(before,stream_file);stream_file.seek(0)
                b=MemoryArm.restore(torch.load(stream_file,weights_only=True))
                for m in [a,b]:m.update(stream[1],event=1)
                self.assertTrue(torch.equal(a.bank,b.bank))
                state=a.snapshot()
                torch.testing.assert_close(a.score(stream[0]),global_nn(stream[0],a.bank),rtol=0,atol=0)
                self.assertStateEqual(state,a.snapshot())
                self.assertTrue(torch.equal(a.bank,b.bank))
                ledger=a.ledger()
                self.assertGreater(ledger['checkpoint_bytes'],0)
                if name in ('cadic','s4','r4'):
                    self.assertTrue(any('_pair_matrix' in t['path'] for t in ledger['storages']))

    def assertStateEqual(self,a,b):
        if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for k in a:self.assertStateEqual(a[k],b[k])
        elif isinstance(a,list):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assertStateEqual(x,y)
        else:self.assertEqual(a,b)

    def test_common_scorer_identical_bank_across_arms(self):
        g=torch.Generator().manual_seed(89)
        x=torch.randn(784,2,generator=g);bank=torch.randn(32,2,generator=g)
        expected=global_nn(x,bank)
        for name in ARMS:
            m=MemoryArm(name,32,2)
            if name=='contcore':m.model.bank=bank.clone()
            elif name=='p0':m.model.bank=bank.clone();m.model.n=32
            elif name=='reservoir':m.rows=bank.clone();m.n=len(bank)
            else:
                off=0
                for b in m.banks:
                    b.features=bank[off:off+b.config.budget].clone();off+=b.config.budget
            torch.testing.assert_close(m.score(x),expected,rtol=0,atol=0)

    def test_byte_capacity_maximal(self):
        ceiling=charged_bytes('contcore',2500)
        for name,k in byte_capacities().items():
            self.assertLessEqual(charged_bytes(name,k),ceiling)
            self.assertGreater(charged_bytes(name,k+1),ceiling)

    def test_existing_p0_and_cadic_exact_integer_port_parity(self):
        from exps.task_p.p0_normal_support import Support
        from exps.ad01.hope_cad_ad01_fast_coreset import FastCADICPatchCoresetV1
        from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig
        from .p0_matched import MatchedP0
        from .controls import MatchedCADIC
        old_p=Support('WITNESS_PROTECTED',capacity=32,dim=2)
        new_p=MatchedP0(32,2)
        old_g=FastCADICPatchCoresetV1(CADICPatchCoresetConfig(budget=32,dim=2))
        new_g=MatchedCADIC(32,2,'cpu',256)
        old_r=Support('RESERVOIR',capacity=32,dim=2)
        new_r=MemoryArm('reservoir',32,2)
        for event in range(2):
            x=torch.stack((torch.arange(784).float()+1000*event,torch.zeros(784)),1)
            ids=torch.arange(event*784,(event+1)*784)
            a=old_p.update(x,ids);b=new_p.update_matched(x,ids)
            for key in ['inserted','replaced','rejected','guard_rejected_proposals']:
                self.assertEqual(a[key],b[key])
            for key in ['bank','ids','nn_distance','nn_index','witnesses','witness_ids']:
                self.assertTrue(torch.equal(getattr(old_p,key),getattr(new_p,key)))
            old_g.update(x);new_g.update(x)
            self.assertTrue(torch.equal(old_g.features,new_g.features))
            old_r.update(x,ids);new_r.update(x,event=event)
            self.assertTrue(torch.equal(old_r.bank,new_r.bank))
            self.assertTrue(torch.equal(old_r.ids,new_r.serials))
            self.assertTrue(torch.equal(old_r.rng.get_state(),new_r.rng.get_state()))

    def test_nonfinite_rejected_before_mutation(self):
        m=CausalContCore(4,2,4,2)
        with self.assertRaises(ValueError):m.update(torch.full((4,2),float('nan')),event=0)
        self.assertEqual(m.next_event,0);self.assertEqual(len(m.bank),0)


if __name__=='__main__':unittest.main(verbosity=2)
