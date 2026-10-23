import copy
import random
import unittest
from unittest.mock import patch
import torch
from hope_cad.models.consolidation import HistoricalUtilityLearner,query_objective
from hope_cad.models.history import CompletedHistory,payload_digest
from hope_cad.models.research_session import ResearchSession
from hope_cad.models.learner import NestedUtilityLearner,meta_loss
from hope_cad.models.learner import prediction,fast_step_from_prediction
from hope_cad.benchmark.stream import _state_equal
from tests.models.test_learning import model,records,factual
from tests.models.test_session import stream
from hope_cad.baselines.cadic.bank import CADICPatchCoresetConfig


def learner(variant='H32_EXACT'):
    return HistoricalUtilityLearner(model(torch.float32),variant,3,'fixture')


def long_records(m,n):
    rs=records(m,n)
    for r in rs:
        r['Jminus']+=100.;r['Jplus']+=100.
    return rs


def actor(variant='H32_EXACT'):
    torch.set_num_threads(2)
    return ResearchSession(variant,seed=9,bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),
        model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))


class ConsolidationContracts(unittest.TestCase):
    def test_shared_forward_exact_events_state_and_resume_across_variants(self):
        for variant in ('R_EXACT','R_FO','H32_EXACT','H32_FO','H32_DIRECT','FAST_ONLY'):
            with self.subTest(variant=variant):
                a=learner(variant);b=learner(variant)
                rs=long_records(a.model,48)
                for index,r in enumerate(rs):
                    factual_record=factual(a,r)
                    # Independent legacy path: validation forward followed by a new forward for the step.
                    def duplicate(phi,target,pred,denom,*,exact):
                        again,d=prediction(b.model,phi,factual_record)
                        return fast_step_from_prediction(phi,target,again,d,exact=exact)
                    da=a.observe(factual_record)
                    with patch('hope_cad.models.consolidation.fast_step_from_prediction',duplicate):db=b.observe(factual_record)
                    self.assertEqual(da,db);self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
                    if index==40:
                        restored=learner(variant);restored.load_state_dict(a.state_dict());a=restored
                # Second credit is neither response probe nor slow commit: one encoder forward.
                x=learner(variant);x.observe(factual(x,rs[0]));r=factual(x,rs[1])
                with patch.object(x.model,'encode',wraps=x.model.encode) as encoder:x.observe(r)
                self.assertEqual(encoder.call_count,1)

    def test_recent_equations_startup_and_full_block_parity(self):
        a=learner('R_EXACT');b=NestedUtilityLearner(copy.deepcopy(a.model))
        for r in long_records(a.model,24):
            a.observe(factual(a,r));b.observe(factual(b,r))
            self.assertTrue(_state_equal(a.phi,b.phi));self.assertTrue(_state_equal(a.model.state_dict(),b.model.state_dict()))
        h=learner();r=learner('R_EXACT')
        for item in records(h.model):h.observe(factual(h,item));r.observe(factual(r,item))
        self.assertTrue(_state_equal(h.phi,r.phi));self.assertTrue(_state_equal(h.model.state_dict(),r.model.state_dict()))

    def test_exact_historical_gradient_and_FO_and_DIRECT(self):
        m=model();rs=records(m,12);support=rs[:4];queries=rs[6:8]+rs[8:10];phi=m.initial_fast()
        e,_=query_objective(m,phi,support,queries,'H32_EXACT');f,_=query_objective(m,phi,support,queries,'H32_FO')
        self.assertEqual(float(e.detach()),float(f.detach()))
        params=dict(m.slow_parameters());ge=torch.autograd.grad(e,tuple(params.values()));gf=torch.autograd.grad(f,tuple(params.values()))
        self.assertGreater(sum(float((x-y).square().sum()) for x,y in zip(ge,gf)),1e-12)
        for name in ('geometry.weight','slow.mlp.0.weight','visual.projection.weight','visual.memory.initial_content','visual.memory.initial_value'):
            p=params[name];g=ge[list(params).index(name)];idx=int(g.abs().argmax());old=float(p.detach().reshape(-1)[idx]);eps=1e-5;vals=[]
            try:
                for sign in (1,-1):
                    with torch.no_grad():p.reshape(-1)[idx]=old+sign*eps
                    vals.append(float(query_objective(m,phi,support,queries,'H32_EXACT')[0].detach()))
            finally:
                with torch.no_grad():p.reshape(-1)[idx]=old
            self.assertAlmostEqual(float(g.reshape(-1)[idx]),(vals[0]-vals[1])/(2*eps),delta=2e-6)
        direct=learner('H32_DIRECT')
        for r in records(direct.model):direct.observe(factual(direct,r))
        self.assertEqual(direct.last_slow['gradient_norm'],0.);self.assertEqual(direct.slow_updates,1)

    def test_reservoir_private_rng_oracle_no_global_draws(self):
        h=learner('FAST_ONLY');state=random.getstate();oracle=[];rng=random.Random(3+96136);qr=random.Random(3+96137)
        for i,r in enumerate(long_records(h.model,48)):
            if i%8==0 and i:
                slots=qr.sample(range(len(oracle)),2)
            h.observe(factual(h,r))
            if (i+1)%8==0:
                start=i-7
                for j in range(start,i+1):
                    n=j+1
                    if n<=32:oracle.append(str(j))
                    else:
                        k=rng.randrange(n)
                        if k<32:oracle[k]=str(j)
                self.assertEqual([x['record']['decision_identity'] for x in h.history.buffer],oracle)
                self.assertEqual(h.history.query_rng.getstate(),qr.getstate())
        self.assertEqual(random.getstate(),state);self.assertEqual(h.slow_updates,0)

    def test_partial_resume_all_lengths_and_identity_corruption(self):
        a=learner();rs=long_records(a.model,24)
        for i,r in enumerate(rs):
            a.observe(factual(a,r));b=learner();b.load_state_dict(a.state_dict())
            self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
            if i+1<len(rs):
                c=learner();c.load_state_dict(a.state_dict())
                self.assertTrue(_state_equal(b.observe(factual(b,rs[i+1])),c.observe(factual(c,rs[i+1]))))
        bad=copy.deepcopy(a.state_dict());bad['history']['buffer'][0]['record']['target']+=1
        with self.assertRaises(ValueError):a.load_state_dict(bad)
        bad=copy.deepcopy(a.state_dict());bad['schema']='hope_support_learner_v5_1'
        with self.assertRaises(ValueError):a.load_state_dict(bad)

    def test_session_rollback_history_rng_and_readonly_resume(self):
        a=actor();xs=stream(14)
        for i,x in enumerate(xs):
            a.observe(x,str(i));b=actor();b.load_state_dict(a.state_dict())
            self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
        before=a.state_dict()
        with patch.object(a.learner.history,'commit',side_effect=RuntimeError('insert failure')):
            # Move to just before commit, preserving snapshot at the event that fails.
            i=14
            while (a.learner.labels+1)%8:
                a.observe(stream(22)[i],str(i));i+=1
            before=a.state_dict()
            with self.assertRaises(RuntimeError):a.observe(stream(22)[i],str(i))
        self.assertTrue(_state_equal(a.state_dict(),before))
        a.score(xs[-1]);self.assertTrue(_state_equal(a.state_dict(),before))
        a.finalize_stream();final=a.state_dict();a.finalize_stream();self.assertTrue(_state_equal(a.state_dict(),final))

    def test_scalar_controls_rng_resume_no_neural_state(self):
        for variant in ('BIAS_VETO','PILOT_RATE_RANDOM_VETO'):
            a=actor(variant)
            for i,x in enumerate(stream(40)):
                a.observe(x,str(i));b=actor(variant);b.load_state_dict(a.state_dict())
                self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
            self.assertEqual(a.learner.slow_updates,0);self.assertIsNone(a.projection)
            self.assertEqual(a.learner.fast_updates,a.learner.labels if variant=='BIAS_VETO' else 0)

if __name__=='__main__':unittest.main()
