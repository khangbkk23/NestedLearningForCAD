import unittest,torch
from models.hope_cad_v5.learner import UtilityModel,meta_loss
from models.hope_cad_v2.state import finite_tensors

class CanonicalUtilityContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable for local canonical learner check')
    def test_full_28_grid_32_channel_exact_four_step_meta_backward(self):
        torch.set_num_threads(2);torch.manual_seed(45);device='cuda'
        model=UtilityModel().to(device).eval();gen=torch.Generator().manual_seed(88)
        records=[dict(tokens=torch.randn(1,784,32,generator=gen).to(device),stats=torch.rand(16,generator=gen).to(device),target=(-1)**i*.3,decision_id=str(i)) for i in range(8)]
        loss=meta_loss(model,model.initial_fast(),records,exact=True);loss.backward()
        grads=[p.grad for _,p in model.slow_parameters()];self.assertTrue(all(g is not None for g in grads));finite_tensors(grads,'canonical utility meta')
        self.assertGreater(sum(float(g.detach().double().square().sum()) for g in grads),0.)
        self.assertTrue(all(p.dtype==torch.float32 for p in model.parameters()))
        self.assertEqual(model.visual.config.height,28);self.assertEqual(model.visual.config.width,28)

if __name__=='__main__':unittest.main()
