"""Force nonempty context learning and profile transport, beyond random fixtures."""
import io
import unittest
import torch
from exps.ecad.r1.distance import configure
from .normal_compatibility import NormalCompatibility
from .test_mechanism import equal_state


class TransportTests(unittest.TestCase):
    def test_observed_context_follows_retained_and_evicted_patch(self):
        configure();torch.set_num_threads(2)
        m=NormalCompatibility(width=2,capacity=2,dim=2,patches=2,candidates=1)
        grids=[torch.tensor(x) for x in (
            [[0.,0.],[0.,10.]],[[10.,0.],[10.,0.]],
            [[0.,0.],[20.,0.]],[[0.,0.],[20.,1.]])]
        contexts=[m.base.context(x) for x in grids]
        for event,x in enumerate(grids):
            stats=m.update(x,event=event)
            if event==2:
                self.assertEqual(stats['normal_prototypes_observed'],1)
                self.assertEqual(stats['evicted_profiles_transported'],0)
                self.assertEqual(m.base.model.bank.tolist(),[[0.,0.],[20.,0.]])
                self.assertTrue(torch.equal(m.lower[0],torch.minimum(contexts[0],contexts[2])))
                self.assertTrue(torch.equal(m.upper[0],torch.maximum(contexts[0],contexts[2])))
        self.assertEqual(stats['normal_prototypes_observed'],2)
        self.assertEqual(stats['evicted_profiles_transported'],1)
        self.assertEqual(m.base.model.bank.tolist(),[[0.,0.],[20.,1.]])
        self.assertEqual(m.base.model.serials.tolist(),[0,7])
        self.assertTrue(torch.equal(m.lower[1],torch.minimum(contexts[2],contexts[3])))
        self.assertTrue(torch.equal(m.upper[1],torch.maximum(contexts[2],contexts[3])))
        self.assertTrue(torch.equal(m.lower[0],torch.stack([contexts[i] for i in (0,2,3)]).min(0).values))
        self.assertTrue(torch.equal(m.upper[0],torch.stack([contexts[i] for i in (0,2,3)]).max(0).values))
        # Resume after profiles really expanded and moved, then update both clones.
        f=io.BytesIO();torch.save(m.state_dict(),f);f.seek(0)
        resumed=NormalCompatibility.from_state_dict(torch.load(f,weights_only=True))
        following=torch.tensor([[0.,0.],[20.,2.]])
        m.update(following,event=4);resumed.update(following,event=4)
        self.assertTrue(equal_state(m.state_dict(),resumed.state_dict()))
        self.assertTrue(equal_state(m.score(following),resumed.score(following)))


if __name__=='__main__':unittest.main(verbosity=2)
