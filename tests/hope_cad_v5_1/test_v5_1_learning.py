"""Independent algebra/finite-difference and functional topology contracts."""
import copy
import math
import unittest
import torch
from models.hope_cad_v5_1.learner import UtilityModel, NestedUtilityLearner, fast_step, meta_loss
from models.hope_cad_v5_1.geometry import causal_scale
from training.benchmark_engine_v1 import _state_equal


def model(dtype=torch.float64):
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(81)
        return UtilityModel(dim=4, context_dim=4, memory_dim=4, head_dim=2, height=2, width=2).to(dtype).eval()


def records(m, n=8):
    gen = torch.Generator().manual_seed(52)
    dtype = next(m.parameters()).dtype
    return [dict(tokens=torch.randn(1, 4, 4, generator=gen, dtype=dtype),
                 stats=torch.randn(16, generator=gen, dtype=dtype), c=2. + i * .3,
                 gamma=.03 * math.sin(i), target=(-1)**i * .7, decision_d=1.,
                 decision_identity=str(i), decision_cursor=i,
                 Jminus=5. + (-1)**i * .7 * (2. + i * .3), Jplus=5.) for i in range(n)]


def factual(learner, record):
    r = copy.deepcopy(record)
    r['target'] = (r['Jminus'] - r['Jplus']) / r['c']
    _, r['decision_d'] = learner.predict(r['tokens'], r['stats'], r['c'], r['gamma'])
    return r


class LearningContracts(unittest.TestCase):
    def test_known_scale_target_sign_counterexample(self):
        # Outcome-dependent ratio can have opposite mean sign to advantage.
        delta, scale = torch.tensor([1., -9.]), torch.tensor([1., 100.])
        self.assertGreater(float((delta / scale).mean()), 0)
        self.assertLess(float((delta / 3.).mean()), 0)
        c, prior = causal_scale(4., 2., 1., 3.)
        self.assertEqual(c, 11.)
        self.assertEqual(prior, 0.)

    def test_zero_head_response_and_no_initial_cms_gradient(self):
        m = model(); r = records(m)[0]; r['target'] = 2.4
        phi = m.initial_fast(); p, d = m.evaluate(r['tokens'], r['stats'], phi, r['c'], r['gamma'])
        new, _, denominator, grads = fast_step(m, phi, r, exact=True)
        after, _ = m.evaluate(r['tokens'], r['stats'], new, r['c'], r['gamma'])
        self.assertAlmostEqual(float(((after-r['target'])/(p-r['target'])).detach()), .95, 12)
        self.assertEqual(float(d.detach()), float(denominator.detach()))
        for (key, _), g in zip(phi.items(), grads):
            if key.startswith('cms.'): self.assertEqual(float(g.detach().abs().max()), 0.)
        self.assertNotIn('head.weight', dict(m.slow_parameters()))
        self.assertEqual(phi['readout.weight'].shape, (1, 5))

    def test_gradient_before_division_and_exact_denominator_dependency(self):
        x = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
        loss = .5 * (x * 0. - .6).square(); d = x.square() + 1.
        self.assertEqual(float(torch.autograd.grad(loss, x, retain_graph=True)[0]), 0.)
        self.assertNotEqual(float(torch.autograd.grad(loss/d, x)[0]), 0.)
        m = model(); r = records(m)[0]; phi = m.initial_fast()
        new, _, _, _ = fast_step(m, phi, r, exact=True)
        # Same-context cancellation checks the derivative of the denominator.
        after, _ = m.evaluate(r['tokens'], r['stats'], new, r['c'], r['gamma'])
        grad = torch.autograd.grad(after, m.geometry.weight)[0]
        self.assertLess(float(grad.detach().abs().max()), 1e-12)

    def test_full_4plus4_finite_difference_and_first_order(self):
        m = model(); rs = records(m); phi = m.initial_fast()
        params = dict(m.slow_parameters())
        exact = meta_loss(m, phi, rs, exact=True)
        gx = torch.autograd.grad(exact, tuple(params.values()))
        first = meta_loss(m, phi, rs, exact=False)
        gf = torch.autograd.grad(first, tuple(params.values()))
        self.assertEqual(float(exact.detach()), float(first.detach()))
        self.assertGreater(sum(float((a-b).square().sum()) for a,b in zip(gx,gf)), 1e-12)
        wanted = ('geometry.weight', 'slow.mlp.0.weight', 'visual.projection.weight',
                  'visual.memory.initial_content', 'visual.memory.initial_value')
        for name in wanted:
            p = params[name]; g = gx[list(params).index(name)]
            idx = int(g.detach().abs().argmax()); old = float(p.detach().reshape(-1)[idx]); eps = 1e-5
            values = []
            try:
                for sign in (1., -1.):
                    with torch.no_grad(): p.reshape(-1)[idx] = old + sign*eps
                    values.append(float(meta_loss(m, phi, rs).detach()))
            finally:
                with torch.no_grad(): p.reshape(-1)[idx] = old
            numeric = (values[0]-values[1])/(2*eps)
            analytic = float(g.detach().reshape(-1)[idx])
            self.assertAlmostEqual(numeric, analytic, delta=2e-7+abs(analytic)*2e-4, msg=name)

    def test_nonzero_head_step_matches_independent_loss_gradient_over_context_energy(self):
        m=model();r=records(m)[0];phi=m.initial_fast()
        with torch.no_grad():phi['readout.weight'].fill_(.2)
        p,d=m.evaluate(r['tokens'],r['stats'],phi,r['c'],r['gamma'])
        loss=.5*(p-r['target']).square()
        gradients=torch.autograd.grad(loss,tuple(phi.values()))
        expected={k:v-.05*g/d.detach() for (k,v),g in zip(phi.items(),gradients)}
        actual,_,_,_=fast_step(m,phi,r,exact=False)
        for k in phi:torch.testing.assert_close(expected[k],actual[k],rtol=1e-13,atol=1e-15,msg=k)

    def test_actual_all_eight_fast_steps_retained_and_partial_resume(self):
        a = NestedUtilityLearner(model()); rs = records(a.model)
        for r in rs[:5]: a.observe(factual(a, r))
        b = NestedUtilityLearner(model()); b.load_state_dict(a.state_dict())
        manual_phi = {k: v.detach().clone().requires_grad_(True) for k,v in a.phi.items()}
        for r in rs[5:]:
            item = factual(a, r)
            manual_phi, _, _, _ = fast_step(a.model, manual_phi, item, exact=False)
            manual_phi = {k: v.detach().requires_grad_(True) for k,v in manual_phi.items()}
            a.observe(item); b.observe(item)
        self.assertTrue(_state_equal(a.state_dict(), b.state_dict()))
        self.assertTrue(_state_equal(a.phi, manual_phi))
        self.assertEqual((a.labels,a.fast_updates,a.slow_updates),(8,8,1))

    def test_modes_counters_and_unbounded_label(self):
        for mode in ('NESTED','FIRST_ORDER','FAST_ONLY','NO_FAST_ADAPTATION'):
            a = NestedUtilityLearner(model(), mode)
            for r in records(a.model):
                if r['decision_cursor'] == 0: r.update(Jminus=205., Jplus=5.)
                a.observe(factual(a, r))
            self.assertEqual(a.fast_updates,0 if mode=='NO_FAST_ADAPTATION' else 8)
            self.assertEqual(a.slow_updates,1 if mode in ('NESTED','FIRST_ORDER') else 0)
            b = NestedUtilityLearner(model(),mode);b.load_state_dict(a.state_dict())
            self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))

    def test_malformed_learner_is_rejected_without_mutation(self):
        a=NestedUtilityLearner(model())
        for r in records(a.model): a.observe(factual(a,r))
        baseline=a.state_dict()
        def bad_shape(s): s['phi']['readout.weight']=torch.zeros(1,6,dtype=torch.float64)
        def bad_dtype(s): s['model']['geometry.weight']=s['model']['geometry.weight'].float()
        def bad_adam(s): next(iter(s['optimizer']['state'].values()))['exp_avg']=torch.zeros(1)
        def bad_clock(s): s['slow_updates']=True
        def bad_template(s): s['model']['fast.readout.bias']+=1
        for mutate in (bad_shape,bad_dtype,bad_adam,bad_clock,bad_template):
            bad=copy.deepcopy(baseline);mutate(bad)
            with self.assertRaises(ValueError):a.load_state_dict(bad)
            self.assertTrue(_state_equal(baseline,a.state_dict()))

    def test_canonical_fp32_28grid_full4plus4_cpu(self):
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(45)
            m=UtilityModel().eval()
        gen=torch.Generator().manual_seed(88)
        rs=[dict(tokens=torch.randn(1,784,32,generator=gen),stats=torch.rand(16,generator=gen),
                 c=3.,gamma=.02,target=(-1)**i*.3) for i in range(8)]
        loss=meta_loss(m,m.initial_fast(),rs,exact=True);loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _,p in m.slow_parameters()))
        self.assertTrue(all(p.dtype==torch.float32 for p in m.parameters()))
        self.assertEqual(m.initial_fast()['readout.weight'].shape,(1,33))
        self.assertEqual(loss.dtype,torch.float64)


if __name__=='__main__': unittest.main()
