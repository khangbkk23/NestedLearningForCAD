"""Validate intervention bookkeeping without changing the production CMS."""
import unittest
import torch
from models.hope_cad_v2.cms import ContinuumMemory
from scripts.hope_cad_v2.diagnose_normal_forensics import retained_level


class ForensicInterventionContracts(unittest.TestCase):
    def test_freeze_retains_weights_without_corrupting_clock_or_original_state(self):
        torch.manual_seed(17)
        cms=ContinuumMemory(4,(8,8),(1,8),(0.01,0.01))
        decoder=torch.nn.Linear(4,6)
        x=torch.randn(1,4,4)
        target=torch.nn.functional.normalize(torch.randn(1,4,6),dim=-1)
        mask=torch.tensor([True,False,True,False])
        initial=cms.initial_state()
        state=initial
        for t in range(8):
            previous=state
            result=cms.event(x,target,mask,decoder,state)
            state=retained_level(previous,result.state,0)
            cms.validate_state(state)
            self.assertEqual(state.clock,t+1)
            for k,value in initial.weights[0].items():
                torch.testing.assert_close(state.weights[0][k],value,rtol=0,atol=0)
        self.assertEqual(state.commits,(8,1))
        self.assertEqual(initial.clock,0)
        self.assertTrue(any(not torch.equal(state.weights[1][k],v) for k,v in initial.weights[1].items()))

    def test_freeze_second_preserves_first_update(self):
        torch.manual_seed(23)
        cms=ContinuumMemory(4,(8,8),(1,1),(0.01,0.00125))
        decoder=torch.nn.Linear(4,6)
        x=torch.randn(1,4,4)
        target=torch.nn.functional.normalize(torch.randn(1,4,6),dim=-1)
        mask=torch.tensor([True,True,False,False])
        original=cms.initial_state()
        result=cms.event(x,target,mask,decoder,original)
        frozen=retained_level(original,result.state,1)
        for k,v in original.weights[1].items():
            torch.testing.assert_close(frozen.weights[1][k],v,rtol=0,atol=0)
        for k,v in result.state.weights[0].items():
            torch.testing.assert_close(frozen.weights[0][k],v,rtol=0,atol=0)
        self.assertEqual(original.clock,0)
        cms.validate_state(frozen)


if __name__=='__main__':
    unittest.main()
