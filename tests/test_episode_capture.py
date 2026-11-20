"""Tiny CPU regression of live policy-gradient capture; no real episodes/data/GPU."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import torch as t
from nested_replay_cad.core.contracts import PhiState
from nested_replay_cad.core.policy.layout import phi_shapes
from nested_replay_cad.core.policy.policy import ReplayPolicy
from episode_diagnostic import EpisodeAudit
import nested_replay_cad.core.learning.adam as slow_adam

class CaptureRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):t.set_num_threads(4)
    def fixture(self):
        gen=t.Generator().manual_seed(32046)
        for _ in range(2):
            raw={n:t.randn(s,generator=gen)*.2 for n,s in phi_shapes().items()}
            context=t.rand((4,4),generator=gen);upstream=t.randn(4,generator=gen,dtype=t.float64)*.001
        return raw,context,upstream
    def gradient(self,raw,context,upstream,kl_first):
        p={n:v.clone().requires_grad_() for n,v in raw.items()};w=ReplayPolicy().weights(PhiState(p),context)
        if kl_first:
            kl=(w*t.log(4*w)).sum();objective=(w.double()*upstream).sum()+.001*kl
        else:
            query=(w.double()*upstream).sum();objective=query+.001*(w*t.log(4*w)).sum()
        return {n:v.detach() for n,v in zip(p,t.autograd.grad(objective,tuple(p.values())))}
    def audit(self,corrupt=False):
        raw,c,u=self.fixture();g=self.gradient(raw,c,u,True)
        if corrupt:g['hidden.weight'][0,0]+=1
        model=t.nn.Linear(1,1,bias=False)
        opt=SimpleNamespace(zero_grad=lambda *a,**k:None);ctrl=SimpleNamespace(policy=ReplayPolicy())
        original=slow_adam.adam_propose
        with patch.object(slow_adam,'adam_propose',return_value=('next','state')):
            with EpisodeAudit(t,model,opt,ctrl) as audit:
                ctrl.policy.weights(PhiState({n:v.clone().requires_grad_() for n,v in raw.items()}),c)
                slow_adam.adam_propose(None,None,g)
            self.assertIs(slow_adam.adam_propose,audit.original_adam)
        self.assertIs(slow_adam.adam_propose,original)
        audit.expected={n:p.detach().clone() for n,p in model.named_parameters()}
        audit.reference_alignments=u.tolist();audit.branch_report={'synthetic_fixture':True}
        event={'slow':{'alignments':u.tolist(),'gradient':{n:v.flatten().tolist() for n,v in g.items()}}}
        return audit,event
    def test_graph_order_can_break_bitwise_check_without_meaningful_gradient_error(self):
        raw,c,u=self.fixture();a=self.gradient(raw,c,u,True);b=self.gradient(raw,c,u,False)
        self.assertFalse(all(t.equal(a[n],b[n]) for n in a))
        self.assertLess(max(float((a[n]-b[n]).abs().max()) for n in a),2e-8)
    def test_valid_live_capture_passes_unchanged_reference_gate(self):
        audit,event=self.audit();report=audit.finish(event)
        self.assertEqual(report['status'],'PASS_SCOPED')
        self.assertTrue(report['controls']['actual_gradient_record_matches_live_capture'])
        self.assertFalse(report['recomputation_diagnostic']['acceptance_gate'])
    def test_wrong_live_gradient_is_rejected_by_unchanged_directional_gate(self):
        audit,event=self.audit(corrupt=True);report=audit.finish(event)
        self.assertEqual(report['status'],'FAIL_MISMATCH')
        self.assertTrue(any(not r['passed'] for r in report['total']))
    def test_record_tampering_is_still_rejected_exactly(self):
        audit,event=self.audit();event['slow']['gradient']['hidden.bias'][0]+=1
        with self.assertRaisesRegex(ValueError,'differs from live capture'):audit.finish(event)

if __name__=='__main__':unittest.main()
