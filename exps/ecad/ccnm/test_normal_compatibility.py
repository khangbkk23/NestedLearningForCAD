"""Normal-only numerical/compatibility counterexamples, no dataset reader."""
import io
import unittest
import torch
from exps.ecad.r1.distance import Euclidean,configure
from exps.ecad.r1.contcore import CausalContCore
from .precise_distance import PreciseEuclidean
from .normal_compatibility import NormalCompatibility,merge_bounds
from .test_mechanism import equal_state


def numerical_gate(device='cpu'):
    g=torch.Generator().manual_seed(50501)
    x=(torch.randn(19,768,generator=g)*5+100).float()
    y=torch.cat((x[:7]+torch.randn(7,768,generator=g),x[:3],x[:3]+.001))
    ref=(x[:,None].double()-y[None].double()).square().sum(2).sqrt().float()
    maximum=0.
    for chunk in (1,7,256):
        actual=PreciseEuclidean(chunk).matrix(x.to(device),y.to(device)).cpu()
        torch.testing.assert_close(actual,ref,atol=2e-6,rtol=2e-6)
        maximum=max(maximum,float((actual-ref).abs().max()))
    old=Euclidean().matrix(x.to(device),y.to(device)).cpu()
    assert bool(((old-ref).abs()>2e-6+2e-6*ref.abs()).any()),'regression fixture must expose old error'
    close=torch.tensor([[1e6,1e6],[1e6+.125,1e6],[1e6,1e6]],device=device)
    torch.testing.assert_close(PreciseEuclidean(1).matrix(close,close),
        torch.tensor([[0.,.125,0.],[.125,0.,.125],[0.,.125,0.]],device=device),rtol=0,atol=0)
    return {'status':'PASS','device':device,'maximum_new_absolute_error':maximum,
            'old_error_reproduced':True,'chunks':[1,7,256],'dimension':768}


def synthetic_stream(device='cpu'):
    g=torch.Generator().manual_seed(50502)
    m=NormalCompatibility(width=3,capacity=11,dim=6,patches=8,candidates=4,device=device)
    ref=CausalContCore(capacity=11,dim=6,patches=8,candidates=4,device=device)
    resumed=None
    learned=transported=0
    for event in range(8):
        x=torch.randn(8,6,generator=g).to(device)+event
        old_ids=m.base.model.serials.clone();old_bank=m.base.model.bank.clone()
        lo,hi=m.lower.clone(),m.upper.clone();c=m.base.context(x)
        if len(old_bank):
            pair=(old_bank[:,None].double()-old_bank[None].double()).square().sum(2).sqrt()
            pair.fill_diagonal_(torch.inf);radius=pair.min(1).values*.5
            distance=(x[:,None].double()-old_bank[None].double()).square().sum(2).sqrt()
            val,ix=distance.min(1)
            for row in torch.unique(ix[val<=radius[ix]]).tolist():
                lo[row]=torch.minimum(lo[row],c);hi[row]=torch.maximum(hi[row],c)
        stats=m.update(x,event=event);ref.update(x,event=event)
        learned+=stats['normal_prototypes_observed'];transported+=stats['evicted_profiles_transported']
        expected_lo=c.expand(len(ref.bank),-1).clone();expected_hi=expected_lo.clone()
        newids=ref.serials.tolist()
        for row,serial in enumerate(old_ids.tolist()):
            if serial in newids:
                target=newids.index(serial);expected_lo[target]=lo[row];expected_hi[target]=hi[row]
        for row,serial in enumerate(old_ids.tolist()):
            if serial not in newids:
                d=(ref.bank.double()-old_bank[row].double()).square().sum(1).sqrt()
                target=int(d.argmin())
                if d[target]<=radius[row]:
                    expected_lo[target]=torch.minimum(expected_lo[target],lo[row])
                    expected_hi[target]=torch.maximum(expected_hi[target],hi[row])
        assert torch.equal(m.lower,expected_lo) and torch.equal(m.upper,expected_hi),'independent profile transport oracle'
        assert torch.equal(m.base.model.bank,ref.bank) and torch.equal(m.base.model.serials,ref.serials)
        assert bool(((m.lower<=m.base.contexts)&(m.base.contexts<=m.upper)).all())
        if event==3:
            stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
            resumed=NormalCompatibility.from_state_dict(torch.load(stream,weights_only=True),device=device)
        elif event>3:
            resumed.update(x,event=event)
            assert equal_state(m.state_dict(),resumed.state_dict())
    before=m.state_dict()
    for mode in ('raw','legacy','boxes','shuffled'):
        out=m.score(x,mode=mode)
        assert bool(torch.isfinite(out['scores']).all()) and equal_state(before,m.state_dict())
        assert equal_state(out,resumed.score(x,mode=mode))
    assert equal_state(m.score(x,weight=0),m.score(x,mode='raw'))
    assert m.ledger()['tensor_total_bytes']==11*(6*4+8+3*4*3)+6*3*4+4
    return {'status':'PASS','device':device,'allocator_parity':True,'bounds_association':True,
            'independent_profile_oracle':True,'learned_profiles':learned,'transported_profiles':transported,
            'exact_resume':True,'read_only':True,'zero_weight_raw_parity':True}


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure();torch.set_num_threads(2)

    def test_precision_regression_high_norm_close_and_chunks(self):
        self.assertEqual(numerical_gate()['status'],'PASS')

    def test_stream_association_eviction_allocator_resume_readonly_and_storage(self):
        self.assertEqual(synthetic_stream()['status'],'PASS')

    def test_shared_normal_protection_unseen_collision_and_shuffle(self):
        m=NormalCompatibility(width=2,capacity=2,dim=2,patches=2,candidates=1)
        b=m.base
        b.model.bank=torch.tensor([[0.,0.],[3.,0.]])
        b.model.serials=torch.tensor([0,2]);b.model.next_event=2
        b.contexts=torch.eye(2);b.alpha.fill_(4.)
        m.lower=b.contexts.clone();m.upper=b.contexts.clone()
        x=torch.tensor([[3.,0.],[3.,0.]]);a=torch.tensor([1.,0.])
        self.assertEqual(m.score(x,query_context=a)['scores'].tolist(),[3.,3.])
        self.assertEqual(m.score(x,mode='shuffled',query_context=a)['scores'].tolist(),[0.,0.])
        merge_bounds(m.lower,m.upper,torch.tensor([1]),a[None],a[None])
        self.assertEqual(m.score(x,query_context=a)['scores'].tolist(),[0.,0.])
        self.assertTrue(bool((m.score(x,query_context=torch.tensor([-1.,0.]))['scores']>0).all()))

    def test_box_overgeneralization_is_exposed(self):
        lower=torch.tensor([[0.,0.]]);upper=torch.tensor([[1.,1.]])
        unseen=torch.tensor([2**-.5,2**-.5])
        self.assertEqual(float(((lower-unseen).clamp_min(0)+(unseen-upper).clamp_min(0)).norm()),0.)

    def test_duplicate_target_merge_matches_scalar_oracle(self):
        lo=torch.tensor([[1.,2.],[3.,4.]]);hi=lo.clone()
        source_lo=torch.tensor([[0.,1.],[-2.,3.],[2.,0.]])
        source_hi=source_lo+4;targets=torch.tensor([0,0,1])
        expected_lo,expected_hi=lo.tolist(),hi.tolist()
        for t,l,h in zip(targets.tolist(),source_lo.tolist(),source_hi.tolist()):
            expected_lo[t]=[min(a,b) for a,b in zip(expected_lo[t],l)]
            expected_hi[t]=[max(a,b) for a,b in zip(expected_hi[t],h)]
        merge_bounds(lo,hi,targets,source_lo,source_hi)
        self.assertEqual(lo.tolist(),expected_lo);self.assertEqual(hi.tolist(),expected_hi)

    def test_noncausal_nonfinite_and_corrupt_bounds_rejected(self):
        m=NormalCompatibility(width=2,capacity=4,dim=3,patches=4,candidates=2)
        m.update(torch.ones(4,3),event=0);before=m.state_dict()
        for event in (0,2):
            with self.assertRaises(ValueError):m.update(torch.ones(4,3),event=event)
        with self.assertRaises(ValueError):m.update(torch.full((4,3),float('nan')),event=1)
        self.assertTrue(equal_state(before,m.state_dict()))
        bad=m.state_dict();bad['lower'].fill_(99)
        with self.assertRaises(ValueError):NormalCompatibility.from_state_dict(bad)


if __name__=='__main__':unittest.main(verbosity=2)
