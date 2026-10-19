"""Synthetic-only correctness for the new evaluator; no data access."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
from exps.ecad.r1.distance import configure
from .mechanism import CCNM
from .test_mechanism import equal_state
from .probe_context import region_indices,replace_region,SharedReadout,contrasts,shuffled_contexts,validate_shared,write


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        torch.set_num_threads(2)

    def test_regions_nested_central_and_exact_cardinality(self):
        small,big=region_indices(4),region_indices(8)
        self.assertEqual(len(small),16);self.assertEqual(len(big),64)
        self.assertEqual(len(set(small.tolist())),16)
        self.assertTrue(set(small.tolist())<set(big.tolist()))
        self.assertEqual((int(small[0]),int(small[-1])),(348,435))

    def test_replacement_no_input_mutation_and_deterministic_donor_order(self):
        a=torch.arange(784*3).reshape(784,3).float()
        b=-a-1
        olda,oldb=a.clone(),b.clone();r=region_indices(4)
        out=replace_region(a,b,r)
        self.assertTrue(torch.equal(out[r],b[r]))
        bg=torch.ones(784,dtype=torch.bool);bg[r]=False
        self.assertTrue(torch.equal(out[bg],a[bg]))
        cycled=replace_region(a,b[:3],r,cycle=True)
        self.assertTrue(torch.equal(cycled[r],b[torch.arange(16)%3]))
        self.assertTrue(torch.equal(a,olda));self.assertTrue(torch.equal(b,oldb))

    def test_shared_exact_parity_direct_oracle_chunks_and_no_mutation(self):
        self.assertEqual(validate_shared()['status'],'PASS')

    def test_default_shuffle_matches_frozen_and_all_seeds_preserve_association(self):
        m=CCNM(width=2,capacity=9,dim=3,patches=4,candidates=2)
        g=torch.Generator().manual_seed(20302)
        for t in range(7):m.update(torch.randn(4,3,generator=g),event=t)
        self.assertTrue(torch.equal(shuffled_contexts(m,52011),m.shuffled_contexts()))
        for seed in (52011,52012,52018):
            x=shuffled_contexts(m,seed)
            for origin in torch.unique(m.model.serials//4):
                xx=x[m.model.serials//4==origin]
                self.assertTrue(torch.equal(xx,xx[:1].expand_as(xx)))

    def test_inferred_context_changes_and_frozen_clean_reference_is_separate(self):
        m=CCNM(width=3,capacity=7,dim=4,patches=16,candidates=4)
        g=torch.Generator().manual_seed(20303)
        for t in range(4):m.update(torch.randn(16,4,generator=g)+t,event=t)
        q=torch.randn(16,4,generator=g)
        altered=replace_region(q,torch.ones_like(q)*5,region_indices(2,4))
        reader=SharedReadout(m);before=m.state_dict();c=m.context(q)
        a,_,ca=reader.score(altered,clean_context=c)
        self.assertFalse(torch.equal(c,ca))
        self.assertTrue(torch.equal(ca,m.context(altered)))
        torch.testing.assert_close(a[1],m.score(altered)['scores'],rtol=0,atol=0)
        torch.testing.assert_close(a[-1],m.score(altered,query_context=c)['scores'],rtol=0,atol=0)
        self.assertTrue(equal_state(before,m.state_dict()))

    def test_paired_contrast_does_not_require_clean_self_match(self):
        clean=torch.tensor([[10.,12.,14.],[11.,13.,15.]])
        altered=torch.tensor([[2.,4.,14.],[12.,14.,15.]])
        self.assertEqual(contrasts(clean,altered,torch.tensor([0,1])).tolist(),[-8.,1.])

    def test_diagnostic_numpy_scalars_serialize_without_partial_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'summary.json'
            write(path,{'check':np.float64(1)<np.float64(2),'count':np.int64(3)})
            self.assertEqual(json.loads(path.read_text()),{'check':True,'count':3})

    def test_complete_fields_recover_from_preserved_truncated_summary(self):
        from .audit_probe import recover_prefix
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'partial.json'
            p.write_text('{\n "a": 3,\n "b": {"complete": true},\n "c": {"incomplete": ')
            self.assertEqual(recover_prefix(p),{'a':3,'b':{'complete':True}})


if __name__=='__main__':unittest.main(verbosity=2)
