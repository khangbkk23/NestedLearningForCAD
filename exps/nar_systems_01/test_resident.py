import copy
import pytest
import torch
from exps.nar_pilot_00 import core as K
from exps.nar_pilot_00.test_detector import ToyDecoder
from exps.nar_pilot_00.detector import Detector
from exps.nar_systems_01.resident import ResidentDetector


@pytest.mark.parametrize('defer',[False,True])
def test_storage_binding_exact_and_archive_immutable(defer,tmp_path):
    torch.manual_seed(3); m=ToyDecoder(); base=copy.deepcopy(m.state_dict())
    lo=K.attach(m,['linear'],2); a=K.init_state(lo,4)
    b={n:(x+.1,y+.2) for n,(x,y) in a.items()}; archive=[a,b]
    snapshot=copy.deepcopy(archive); x=torch.randn(2,4,3)
    reference=Detector(m,lo,archive).predict(x,(31,29))
    m2=ToyDecoder(); m2.load_state_dict(base); lo2=K.attach(m2,['linear'],2)
    opt=ResidentDetector(m2,lo2,archive,defer)
    for _ in range(3):
        out=opt.predict(x,(31,29))
        assert all(torch.equal(u,v) for u,v in zip(reference,out))
    assert all(torch.equal(u,v) for s,t in zip(archive,snapshot) for n in s for u,v in zip(s[n],t[n]))
    assert opt.runtime_factor_bytes()==sum(K.tensor_state_bytes(s) for s in archive)
    assert not any(p.requires_grad for p in m2.parameters())
    torch.save({'base':base,'archive':archive},tmp_path/'state.pt')
    payload=torch.load(tmp_path/'state.pt',weights_only=True)
    m3=ToyDecoder(); m3.load_state_dict(payload['base']); lo3=K.attach(m3,['linear'],2)
    reopened=ResidentDetector(m3,lo3,payload['archive'],defer).predict(x,(31,29))
    assert all(torch.equal(u,v) for u,v in zip(reference,reopened))


def test_training_mode_is_rejected():
    m=ToyDecoder(); lo=K.attach(m,['linear'],2); d=ResidentDetector(m,lo,[K.init_state(lo,4)])
    m.train()
    with pytest.raises(RuntimeError,match='inference-only'): d.bind(d.resident[0])


def test_precomposition_is_not_bitwise_arithmetic_equivalence():
    torch.manual_seed(3)
    x=torch.randn(31,64); w=torch.randn(64,64); a=torch.randn(8,64); b=torch.randn(64,8)
    separate=torch.nn.functional.linear(x,w)+torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)
    merged=torch.nn.functional.linear(x,w+b@a)
    assert not torch.equal(separate,merged)
    assert torch.allclose(separate,merged,atol=5e-5,rtol=1e-4)
