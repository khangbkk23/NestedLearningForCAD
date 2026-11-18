"""Focused detector/readout/forgetting checks; no dataset access."""
import copy
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from exps.nar_pilot_00 import core as K
from exps.nar_pilot_00.detector import Detector,prefix_grids,fm_eq8


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,3)
    def forward(self,x):
        y=self.linear(x)
        return [x.transpose(1,2).reshape(len(x),3,2,2)],[y.transpose(1,2).reshape(len(x),3,2,2)]


def test_prefix_reuse_exact_and_checkpoint(tmp_path):
    torch.manual_seed(2)
    m=ToyDecoder()
    lo=K.attach(m,['linear'],2)
    a=K.init_state(lo,3)
    b={n:(x+0.1,y+0.2) for n,(x,y) in a.items()}
    archive=[a,b]
    x=torch.randn(2,4,3)
    detector=Detector(m,lo,archive)
    all_grids=detector.expert_grids(x)
    reused=prefix_grids(all_grids)
    for i in range(2):
        direct,_=Detector(m,lo,archive[:i+1]).predict(x)
        assert torch.equal(direct,reused[i])
    native,image=detector.predict(x,(31,29))
    assert native.shape==(2,31,29) and image.shape==(2,)
    assert torch.isfinite(native).all() and native.min()>=0
    torch.save({'model':m.state_dict(),'archive':archive},tmp_path/'s.pt')
    state=torch.load(tmp_path/'s.pt',weights_only=True)
    m2=ToyDecoder(); lo2=K.attach(m2,['linear'],2)
    m2.load_state_dict(state['model'])
    n2,i2=Detector(m2,lo2,state['archive']).predict(x,(31,29))
    assert torch.equal(native,n2) and torch.equal(image,i2)


def test_eq8_uses_intermediate_best_and_excludes_last():
    order=['a','b','c']
    cells={(0,'a'):{'pixel_ap':.8},(1,'a'):{'pixel_ap':.9},(2,'a'):{'pixel_ap':.7},
           (1,'b'):{'pixel_ap':.6},(2,'b'):{'pixel_ap':.65},(2,'c'):{'pixel_ap':.9}}
    fm=fm_eq8(cells,order,'pixel_ap')
    assert abs(fm['macro']-.075)<1e-12
    assert abs(fm['positive_part_macro']-.1)<1e-12


def test_native_metric_matches_sklearn_without_quantization():
    from exps.calm02.native import native_ap
    from exps.task_p.p1_readout import native_map
    import numpy as np
    rng=np.random.default_rng(9)
    patches=rng.random((3,784)).astype('float32')
    masks=[rng.random((35,31))>.8 for _ in range(3)]
    maps=[native_map(torch.from_numpy(p),(35,31)) for p in patches]
    reference=average_precision_score(np.concatenate([m.ravel() for m in masks]),
                                    np.concatenate([m.ravel() for m in maps]))
    assert abs(native_ap(patches,[(35,31)]*3,masks)['ap']-reference)<1e-12
