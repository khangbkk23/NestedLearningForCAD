import unittest,copy,math,torch
from unittest.mock import patch
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig,CADICPatchCoresetV1
from models.hope_cad_v5.geometry import ForkableCoreset,risk,advantage,AnchorReservoir,FixedContextProjection
from models.hope_cad_v5.learner import UtilityModel,NestedUtilityLearner,meta_loss
from models.hope_cad_v5.session import SupportSession
from models.hope_cad_v5.contract import CONTRACT
from training.benchmark_engine_v1 import _state_equal

def model():
    torch.manual_seed(81);torch.set_num_threads(2)
    return UtilityModel(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=4,width=4).double().eval()
def records():
    gen=torch.Generator().manual_seed(12)
    return [dict(tokens=torch.randn(1,16,4,generator=gen,dtype=torch.float64),stats=torch.randn(16,generator=gen,dtype=torch.float64),target=(-1)**i*.4,decision_id=str(i)) for i in range(8)]
def session(mode='NESTED'):
    torch.set_num_threads(2)
    return SupportSession(mode,bank_config=CADICPatchCoresetConfig(budget=12,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=4,width=4),seed=9)
def stream(n=35):
    g=torch.Generator().manual_seed(54)
    return [torch.randn(1,16,4,generator=g)+i*.03 for i in range(n)]

class GeometryContracts(unittest.TestCase):
    def test_fork_cache_reference_and_parent_unchanged(self):
        cfg=CADICPatchCoresetConfig(budget=12,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4)
        ref=CADICPatchCoresetV1(cfg);fast=ForkableCoreset(cfg)
        for x in stream(5):
            before=fast.state_dict();child=fast.fork();self.assertTrue(_state_equal(before,fast.state_dict()))
            self.assertEqual(ref.update(x),child.update(x));self.assertTrue(_state_equal(ref.state_dict(),child.state_dict()));fast=child
        q=stream(1)[0];self.assertTrue(_state_equal(ref.pixel_scores(q),fast.pixel_scores(q)))
    def test_ties_duplicates_and_partial_fill_match(self):
        cfg=CADICPatchCoresetConfig(budget=5,dim=2,chunk_size=2,pair_chunk_size=2,query_chunk_size=2)
        ref=CADICPatchCoresetV1(cfg);bank=ForkableCoreset(cfg)
        for x in [torch.zeros(3,2),torch.tensor([[0.,0.],[1,1],[3,3],[2,2]]),torch.tensor([[6.,6.],[7,7]])]:
            bank=bank.fork();self.assertEqual(ref.update(x),bank.update(x));self.assertTrue(_state_equal(ref.state_dict(),bank.state_dict()))
    def test_coverage_and_advantage(self):
        bank=ForkableCoreset(CADICPatchCoresetConfig(budget=2,dim=1));bank.update(torch.tensor([[0.],[2.]]))
        self.assertAlmostEqual(risk(torch.tensor([[1.],[5.]]),bank),7.)
        for before,after in [(0,0),(0,3),(2,0),(3,8)]:self.assertLessEqual(abs(advantage(before,after)),1.)
    def test_projection_private_rng_and_deterministic(self):
        rng=torch.get_rng_state().clone();p=FixedContextProjection(4,4);q=FixedContextProjection(4,4)
        self.assertTrue(torch.equal(p.matrix,q.matrix));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_anchor_resume_uniform_scope(self):
        a=AnchorReservoir(seed=3,capacity=3,patches=2)
        for i,x in enumerate(stream(8)):a.update(x,str(i))
        b=AnchorReservoir(seed=3,capacity=3,patches=2);b.load_state_dict(a.state_dict(),'cpu')
        for i,x in enumerate(stream(4)):a.update(x,'r'+str(i));b.update(x,'r'+str(i))
        self.assertTrue(_state_equal(a.state_dict(),b.state_dict()));self.assertEqual(len(a.images),3)

class LearningContracts(unittest.TestCase):
    def test_full_and_first_order_same_forward_different_meta_gradient(self):
        m=model();r=records();initial=m.initial_fast();names,params=zip(*m.slow_parameters())
        full=meta_loss(m,initial,r,exact=True);gf=torch.autograd.grad(full,params)
        first=meta_loss(m,initial,r,exact=False);g1=torch.autograd.grad(first,params)
        self.assertAlmostEqual(float(full.detach()),float(first.detach()),12)
        difference=sum(float((a-b).square().sum()) for a,b in zip(gf,g1));self.assertGreater(difference,1e-12)
    def test_nested_meta_gradient_central_difference(self):
        m=model();r=records();initial=m.initial_fast();loss=meta_loss(m,initial,r);names,params=zip(*m.slow_parameters());grads=torch.autograd.grad(loss,params)
        for wanted in ['head.weight','slow.mlp.0.weight','geometry.weight','visual.projection.weight','visual.memory.initial_value']:
            i=names.index(wanted);p=params[i];g=grads[i];index=int(g.abs().argmax());old=float(p.detach().flatten()[index]);eps=1e-5
            with torch.no_grad():p.flatten()[index]=old+eps
            plus=float(meta_loss(m,initial,r).detach())
            with torch.no_grad():p.flatten()[index]=old-eps
            minus=float(meta_loss(m,initial,r).detach())
            with torch.no_grad():p.flatten()[index]=old
            numeric=(plus-minus)/(2*eps);analytic=float(g.flatten()[index]);self.assertAlmostEqual(numeric,analytic,delta=2e-7+abs(analytic)*2e-4,msg=wanted)
    def test_window_clock_real_fast_slow_updates_and_resume(self):
        m=model();a=NestedUtilityLearner(m);b=NestedUtilityLearner(copy.deepcopy(m));r=records();initial=copy.deepcopy(m.state_dict())
        for item in r[:5]:a.observe(item['tokens'],item['stats'],item['target'],item['decision_id'])
        b.load_state_dict(a.state_dict())
        for item in r[5:]:a.observe(item['tokens'],item['stats'],item['target'],item['decision_id']);b.observe(item['tokens'],item['stats'],item['target'],item['decision_id'])
        self.assertTrue(_state_equal(a.state_dict(),b.state_dict()));self.assertEqual(a.slow_updates,1);self.assertEqual(a.labels,8)
        self.assertFalse(_state_equal(initial,a.model.state_dict()));self.assertTrue(all(math.isfinite(v) for v in a.last_slow['group_norms'].values()))
        self.assertTrue(all(v>0 for v in a.last_slow['group_norms'].values()))

class SessionContracts(unittest.TestCase):
    def test_always_pass_exact_CADIC_bank_counters_and_Eq9(self):
        s=session('CADIC');ref=CADICPatchCoresetV1(s.bank_config)
        for i,x in enumerate(stream(12)):
            s.observe(x,str(i));ref.update(x)
            self.assertTrue(_state_equal(s.bank.state_dict(),ref.state_dict()))
        self.assertTrue(_state_equal(s.score(stream(1)[0]),ref.pixel_scores(stream(1)[0])))
    def test_causal_one_image_credit_and_terminal_censor(self):
        s=session();xs=stream(4);s.observe(xs[0],'0');o=s.observe(xs[1],'1');self.assertIsNone(o['credit']);self.assertEqual(s.learner.labels,0)
        self.assertIsNotNone(s.pending);minus=s.pending['other'] if s.pending['accepted'] else s.bank.features;plus=s.bank.features if s.pending['accepted'] else s.pending['other']
        expected=advantage(risk(xs[2],s._bank_view(minus))+s.pending['anchor0'],risk(xs[2],s._bank_view(plus))+s.pending['anchor1'])
        o=s.observe(xs[2],'2');self.assertAlmostEqual(o['credit']['target'],expected);self.assertEqual(o['credit']['reward_observed_at'],2);self.assertEqual(o['credit']['decision_cursor'],1)
        before=s.learner.labels;s.finalize_stream();self.assertEqual(before,s.learner.labels);self.assertEqual(s.counters['censored'],1)
    def test_scoring_is_read_only_no_delayed_label_resolution(self):
        s=session()
        for i,x in enumerate(stream(3)):s.observe(x,str(i))
        before=s.state_dict();rng=torch.get_rng_state().clone();s.score(stream(1)[0]);self.assertTrue(_state_equal(before,s.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_interrupted_resume_partial_window_and_pending_matches(self):
        a=session();b=session();xs=stream(24)
        for i,x in enumerate(xs[:13]):a.observe(x,str(i))
        self.assertIsNotNone(a.pending);b.load_state_dict(a.state_dict())
        for i,x in enumerate(xs[13:],13):
            oa=a.observe(x,str(i));ob=b.observe(x,str(i));self.assertTrue(_state_equal(oa,ob))
        self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
    def test_atomic_failure_rolls_back_learner_bank_pending_anchors(self):
        s=session()
        for i,x in enumerate(stream(3)):s.observe(x,str(i))
        before=s.state_dict();original=s.learner.observe
        def fail(*args,**kwargs):original(*args,**kwargs);raise RuntimeError('after utility update')
        with patch.object(s.learner,'observe',fail):
            with self.assertRaises(RuntimeError):s.observe(stream(4)[3],'3')
        self.assertTrue(_state_equal(before,s.state_dict()))
    def test_corrupt_identity_hash_cursor_rejected_without_mutation(self):
        s=session();s.observe(stream(1)[0],'a');before=s.state_dict()
        for field in ['hash','identity','cursor']:
            bad=copy.deepcopy(before)
            if field=='hash':bad['bank']['features'][0,0]+=1
            if field=='identity':bad['identity']['seed']=99
            if field=='cursor':bad['cursor']+=1
            with self.assertRaises(ValueError):s.load_state_dict(bad)
            self.assertTrue(_state_equal(before,s.state_dict()))
    def test_utility_changes_actual_bank_and_raw_scores(self):
        a=session();b=session();xs=stream(3);a.observe(xs[0],'0');b.observe(xs[0],'0')
        with patch.dict(CONTRACT['learning'],warmup_completed_labels=0),patch.object(a.learner,'predict',return_value=1.),patch.object(b.learner,'predict',return_value=-1.):
            a.observe(xs[1],'1');b.observe(xs[1],'1')
        self.assertFalse(torch.equal(a.bank.features,b.bank.features));self.assertFalse(torch.equal(a.score(xs[2])[0],b.score(xs[2])[0]))
    def test_duplicate_nonfinite_inputs_rejected(self):
        s=session();x=stream(1)[0];s.observe(x,'a');before=s.state_dict()
        with self.assertRaises(ValueError):s.observe(x,'a')
        with self.assertRaises(FloatingPointError):s.observe(x*float('nan'),'b')
        self.assertTrue(_state_equal(before,s.state_dict()))

if __name__=='__main__':unittest.main()
