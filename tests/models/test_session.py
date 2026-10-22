import copy
import unittest
from unittest.mock import patch
import torch
from hope_cad.models.session import SupportSession
from hope_cad.models.geometry import risk, causal_scale
from hope_cad.baselines.cadic.bank import CADICPatchCoresetConfig, CADICPatchCoresetV1
from hope_cad.benchmark.stream import _state_equal


def session(mode='NESTED'):
    torch.set_num_threads(2)
    return SupportSession(mode,seed=9,
        bank_config=CADICPatchCoresetConfig(budget=6,dim=4,chunk_size=4,pair_chunk_size=4,query_chunk_size=4),
        model_config=dict(dim=4,context_dim=4,memory_dim=4,head_dim=2,height=2,width=2))


def stream(n=24):
    gen=torch.Generator().manual_seed(54)
    # Expanding support forces genuine proposals; a stationary tiny-bank fixture
    # can legitimately stop producing labels and cannot test partial replay.
    return [torch.randn(1,4,4,generator=gen)*.02+1.7**i for i in range(n)]


class SessionContracts(unittest.TestCase):
    def test_private_initialization_matching_and_rng(self):
        rng=torch.get_rng_state().clone();a=session();b=session('FIRST_ORDER')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(a.initialization_hashes(),b.initialization_hashes())

    def test_cadic_full_pass_bank_counters_remainder_and_Eq9(self):
        a=session('CADIC');ref=CADICPatchCoresetV1(a.bank_config)
        for i,x in enumerate(stream(12)):
            a.observe(x,str(i));ref.update(x)
            self.assertTrue(_state_equal(a.bank.state_dict(),ref.state_dict()))
        self.assertTrue(_state_equal(a.score(stream(1)[0]),ref.pixel_scores(stream(1)[0])))

    def test_learned_warmup_forced_pass_matches_cadic(self):
        a=session();b=session('CADIC')
        for i,x in enumerate(stream(18)):
            a.observe(x,str(i));b.observe(x,str(i))
            self.assertTrue(_state_equal(a.bank.state_dict(),b.bank.state_dict()))
        self.assertGreater(a.learner.slow_updates,0)

    def test_causal_fixed_banks_cached_anchors_and_scale(self):
        a=session();xs=stream(4)
        for i,x in enumerate(xs[:3]):a.observe(x,str(i))
        p=copy.deepcopy(a.pending);self.assertIsNotNone(p)
        minus=p['other'] if p['accepted'] else a.bank.features
        plus=a.bank.features if p['accepted'] else p['other']
        j0=risk(xs[3],a._bank_view(minus))+p['anchor0'];j1=risk(xs[3],a._bank_view(plus))+p['anchor1']
        credit=a.observe(xs[3],'3')['credit']
        self.assertEqual(credit['target'],(j0-j1)/p['c'])
        self.assertEqual(credit['reward_observed_at'],p['decision_cursor']+1)
        self.assertEqual((credit['anchor_minus'],credit['anchor_plus']),(p['anchor0'],p['anchor1']))

    def test_all_partial_windows_resume_and_true_terminal(self):
        a=session();xs=stream(19)
        for i,x in enumerate(xs):
            a.observe(x,str(i))
            b=session();b.load_state_dict(a.state_dict())
            self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
            if i+1<len(xs):
                c=session();c.load_state_dict(a.state_dict())
                self.assertTrue(_state_equal(c.observe(xs[i+1],'next-check'),b.observe(xs[i+1],'next-check')))
        old=a.learner.labels;a.finalize_stream();a.finalize_stream()
        self.assertEqual(old,a.learner.labels);self.assertEqual(a.counters['censored'],1)
        self.assertEqual(len(a.learner.records),a.learner.labels%8)

    def test_pending_resume_score_read_only_and_rollback(self):
        a=session()
        for i,x in enumerate(stream(11)):a.observe(x,str(i))
        before=a.state_dict();rng=torch.get_rng_state().clone();a.score(stream(1)[0])
        self.assertTrue(_state_equal(before,a.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        original=a.learner.observe
        def fail(record):original(record);raise RuntimeError('injected after fast/slow learner update')
        with patch.object(a.learner,'observe',fail):
            with self.assertRaises(RuntimeError):a.observe(stream(12)[-1],'11')
        self.assertTrue(_state_equal(before,a.state_dict()))

    def test_malformed_full_session_atomic_load(self):
        a=session()
        for i,x in enumerate(stream(13)):a.observe(x,str(i))
        old=a.state_dict()
        self.assertIsNotNone(old['pending']);self.assertTrue(old['learner']['records'])
        def schema(s):s['schema']='hope_support_session_v5'
        def shape(s):s['pending']['tokens']=torch.zeros(1,3,4)
        def rng(s):s['anchors']['rng']=('invalid',)
        def target(s):s['learner']['records'][0]['target']=float('nan')
        def dtype(s):s['bank']['features']=s['bank']['features'].double()
        def counter(s):s['counters']['labels_completed']+=1
        def denominator(s):s['pending']['decision_d']+=1
        for mutate in (schema,shape,rng,target,dtype,counter,denominator):
            bad=copy.deepcopy(old);mutate(bad)
            with self.assertRaises((ValueError,FloatingPointError)):a.load_state_dict(bad)
            self.assertTrue(_state_equal(old,a.state_dict()))

    def test_noop_does_not_create_synthetic_labels(self):
        a=session();x=torch.zeros(1,4,4)
        for i in range(7):a.observe(x,str(i))
        self.assertEqual(a.learner.labels,0);self.assertIsNone(a.pending)
        self.assertEqual(a.counters['noop_proposals'],5)

    def test_current_coefficients_match_fp32_risk_before_fp64_promotion(self):
        from hope_cad.models.geometry import context_statistics
        for scale in (1e-22,1.,1e12):
            a=session('CADIC');a.observe(stream(1)[0]*scale,'a')
            a.observe(stream(2)[1]*scale,'b')
            old=a.bank;new=old.fork();x=stream(3)[2].reshape(-1,4)*scale
            result=new.update(x)
            self.assertEqual(old.features.shape,new.features.shape)
            _,known=context_statistics(x,old,new,result,0,1,0.,0.)
            self.assertEqual(known['current0'],risk(x,old))
            self.assertEqual(known['current1'],risk(x,new))

    def test_bad_inputs_do_not_mutate(self):
        a=session();x=stream(1)[0];a.observe(x,'0');old=a.state_dict()
        for invalid in (x.double(),x*float('nan'),x[:,:3]):
            with self.assertRaises((ValueError,FloatingPointError)):a.observe(invalid,'1')
            self.assertTrue(_state_equal(old,a.state_dict()))
        with self.assertRaises(ValueError):a.observe(x,'0')

    def test_warmup_boundary_and_saved_actual_episode(self):
        from hope_cad.models.learner import UtilityModel,meta_loss
        a=session();boundaries={}
        for i,x in enumerate(stream(65)):
            record=a.observe(x,str(i))
            if 'warmup' in record and record['learner_labels'] in (31,32):
                boundaries[record['learner_labels']]=record['warmup']
            if a.learner.labels==40:break
        self.assertEqual(boundaries,{31:True,32:False})
        self.assertEqual(a.learner.labels,40)
        episode=a.learner.saved_meta_episode
        self.assertEqual(episode['labels_inclusive'],[33,40])
        m=UtilityModel(**a.model_config).eval();m.load_state_dict(episode['model'])
        self.assertEqual(float(meta_loss(m,episode['phi_start'],episode['records']).detach()),a.learner.last_slow['loss'])
        b=session();b.load_state_dict(a.state_dict());self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))


if __name__=='__main__':unittest.main()
