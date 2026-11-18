"""Focused warm policy and frozen gate checks; no dataset/GPU access."""
import copy
import torch
import pytest
from exps.nar_warm_01 import run as W
from exps.nar_pilot_00 import core as K


def test_last_endpoint_not_average_and_no_mutation():
    fixed={'x':(torch.zeros(2,3),torch.zeros(3,2))}
    one={'x':(torch.ones(2,3),torch.ones(3,2))}
    two={'x':(torch.full((2,3),4.),torch.full((3,2),7.))}
    snapshot=copy.deepcopy(two)
    assert W.initial_state(fixed,[]) is fixed
    assert W.initial_state(fixed,[one,two]) is two
    lo=K.attach(torch.nn.Sequential(torch.nn.Linear(3,3)),['0'],2)
    K.set_state(lo,{'0':two['x']})
    with torch.no_grad():lo['0'].A.add_(1)
    assert W.digest(two)==W.digest(snapshot)


def summary(ap=.5,image=.9,acq=.5,fm=.01):
    return {'final_pixel_ap':ap,'final_image_auroc':image,'acquisition_pixel_ap':acq,'pixel_fm_eq8':{'macro':fm}}


def test_original_gate_boundaries():
    n=summary(); w=summary()
    assert W.gate(n,w,{'point':.01,'lo':1e-9})['passed']
    assert not W.gate(n,w,{'point':.009999,'lo':.001})['passed']
    assert not W.gate(n,w,{'point':.02,'lo':0})['passed']
    assert not W.gate(summary(image=.88),w,{'point':.02,'lo':.001})['passed']
    assert not W.gate(summary(acq=.48),w,{'point':.02,'lo':.001})['passed']
    assert not W.gate(summary(fm=.021),w,{'point':.02,'lo':.001})['passed']


def test_no_test_cache_before_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'OUT',tmp_path)
    with pytest.raises(FileNotFoundError):W.load_cache({},'bottle','test')


def test_disk_cap_prevents_write(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'OUT',tmp_path)
    class B:
        def check(self):pass
    with pytest.raises(RuntimeError,match='disk cap'):
        W.bounded_save(tmp_path/'large.pt',{},W.CAPS['artifact_bytes'],B())
    assert not (tmp_path/'large.pt').exists()
