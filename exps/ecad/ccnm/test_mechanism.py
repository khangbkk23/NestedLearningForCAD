"""Required synthetic invariants; no real or anomalous dataset access."""
import io
import unittest
import torch
from exps.ecad.r1.contcore import CausalContCore
from exps.ecad.r1.distance import configure, global_nn
from .mechanism import CCNM


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    return a == b


def collision_fixture(device='cpu'):
    m = CCNM(width=2, capacity=2, dim=2, patches=2, candidates=1, device=device)
    m.model.bank = torch.tensor([[0., 0.], [3., 0.]], device=device)
    m.model.serials = torch.tensor([0, 2], device=device)
    m.model.next_event = 2
    m.contexts = torch.eye(2, device=device)
    m.alpha.fill_(4.)
    return m


class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        torch.set_num_threads(2)

    def test_exact_association_admission_eviction_consolidation_same_ids(self):
        m = CCNM(width=3, capacity=11, dim=6, patches=8, candidates=4)
        raw = CausalContCore(capacity=11, dim=6, patches=8, candidates=4)
        expected = {}
        g = torch.Generator().manual_seed(921)
        previous = set()
        evicted = False
        for t in range(12):
            x = torch.randn(8, 6, generator=g) + t
            expected[t] = m.context(x)
            m.update(x, event=t)
            raw.update(x, event=t)
            self.assertTrue(torch.equal(m.model.serials, raw.serials))
            self.assertTrue(torch.equal(m.model.bank, raw.bank))
            self.assertTrue(torch.equal(m.contexts, torch.stack([expected[int(s)//8] for s in raw.serials])))
            evicted |= bool(previous - set(raw.serials.tolist()))
            previous = set(raw.serials.tolist())
        self.assertTrue(evicted)

    def test_stream_access_and_rejected_input_no_mutation(self):
        m = CCNM(width=2, capacity=4, dim=3, patches=4, candidates=2)
        m.update(torch.ones(4, 3), event=0)
        before = m.state_dict()
        for event in (0, 2, -1):
            with self.assertRaises(ValueError):
                m.update(torch.ones(4, 3), event=event)
            self.assertTrue(equal_state(before, m.state_dict()))
        with self.assertRaises(ValueError):
            m.update(torch.full((4, 3), float('nan')), event=1)
        self.assertTrue(equal_state(before, m.state_dict()))

    def test_exact_deterministic_replay_and_checkpoint_resume(self):
        g = torch.Generator().manual_seed(922)
        stream = [torch.randn(8, 6, generator=g) for _ in range(8)]
        a = CCNM(width=3, capacity=11, dim=6, patches=8, candidates=4)
        replay = CCNM(width=3, capacity=11, dim=6, patches=8, candidates=4)
        for t, x in enumerate(stream[:4]):
            a.update(x, event=t)
        f = io.BytesIO()
        torch.save(a.state_dict(), f)
        f.seek(0)
        b = CCNM.from_state_dict(torch.load(f, weights_only=True))
        for t, x in enumerate(stream):
            replay.update(x, event=t)
            if t >= 4:
                a.update(x, event=t)
                b.update(x, event=t)
        self.assertTrue(equal_state(a.state_dict(), b.state_dict()))
        self.assertTrue(equal_state(a.state_dict(), replay.state_dict()))
        for mode in ('raw', 'context', 'shuffled', 'routing'):
            self.assertTrue(equal_state(a.score(stream[0], mode=mode), b.score(stream[0], mode=mode)))

    def test_read_only_scoring_and_zero_weight_exact_raw(self):
        m = CCNM(width=3, capacity=11, dim=6, patches=8, candidates=4)
        g = torch.Generator().manual_seed(923)
        for t in range(4):
            m.update(torch.randn(8, 6, generator=g), event=t)
        x = torch.randn(8, 6, generator=g)
        before = m.state_dict()
        for mode in ('raw', 'context', 'shuffled', 'routing'):
            self.assertTrue(torch.isfinite(m.score(x, mode=mode)['scores']).all())
            self.assertTrue(equal_state(before, m.state_dict()))
        self.assertTrue(equal_state(m.score(x, mode='raw'), m.score(x, weight=0)))
        self.assertTrue(torch.equal(m.score(x, mode='raw')['scores'], global_nn(x, m.model.bank)))

    def test_known_wrong_and_shuffled_context_collision_independent_values(self):
        m = collision_fixture()
        x = torch.tensor([[3., 0.], [3., 0.]])
        a, b = torch.eye(2)
        raw = m.score(x, mode='raw', query_context=a)
        known = m.score(x, query_context=a)
        wrong = m.score(x, query_context=b)
        shuffled = m.score(x, mode='shuffled', query_context=a)
        route = m.score(x, mode='routing', query_context=a)
        self.assertEqual(raw['scores'].tolist(), [0., 0.])
        self.assertEqual(known['scores'].tolist(), [3., 3.])
        self.assertEqual(known['ids'].tolist(), [0, 0])
        self.assertEqual(wrong['scores'].tolist(), [0., 0.])
        self.assertEqual(shuffled['scores'].tolist(), [0., 0.])
        self.assertEqual(route['scores'].tolist(), [3., 3.])
        normal = m.score(torch.zeros(2, 2), query_context=a)
        self.assertEqual(normal['scores'].tolist(), [0., 0.])

    def test_context_alias_is_explicit_failure_case(self):
        m = collision_fixture()
        m.contexts[1] = m.contexts[0]
        x = torch.tensor([[3., 0.], [3., 0.]])
        self.assertEqual(m.score(x, query_context=m.contexts[0])['scores'].tolist(), [0., 0.])

    def test_finite_chunk_parity_full_dimension_and_independent_augmented_oracle(self):
        m = CCNM(capacity=23, patches=11, candidates=5)
        g = torch.Generator().manual_seed(924)
        for t in range(5):
            m.update(torch.randn(11, 768, generator=g), event=t)
        x = torch.randn(11, 768, generator=g)
        c = m.context(x)
        d = ((x[:, None].double()-m.model.bank[None].double()).square().sum(2)
             + m.alpha.double().square()*(c.double()-m.contexts.double()).square().sum(1)).sqrt()
        for chunk in (1, 7, 256):
            out = m.score(x, chunk=chunk)
            torch.testing.assert_close(out['scores'], d.min(1).values.float(), atol=2e-6, rtol=2e-6)
            self.assertTrue(torch.equal(out['ids'], m.model.serials[d.argmin(1)]))
        m.alpha.zero_()
        self.assertTrue(equal_state(m.score(x), m.score(x, mode='raw')))

    def test_complete_persistent_storage_and_corruption_rejection(self):
        m = CCNM(width=3, capacity=11, dim=6, patches=8, candidates=4)
        m.update(torch.ones(8, 6), event=0)
        ledger = m.ledger()
        self.assertEqual(ledger['tensor_total_bytes'], 4*6*4 + 4*8 + 6*3*4 + 4*3*4 + 4)
        self.assertEqual(set(vars(m)), {'model', 'width', 'projection', 'contexts', 'alpha'})
        s = m.state_dict()
        s['projection'][0, 0] += 1
        with self.assertRaises(ValueError):
            CCNM.from_state_dict(s)


if __name__ == '__main__':
    unittest.main(verbosity=2)
