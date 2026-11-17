import json
from pathlib import Path
import torch
from exps.nl_sgm_01a import am03_eval as A

def test_lock_and_protocol_immutable():
    d=A.lock(); assert d['device']=='cpu'; assert d['cuda_must_not_initialize'] is True

def test_support_decode_and_exact_smoke():
    x,b=A.load_support('C1_FPS_FP32',0); assert 'bottle' in x and b==2499*768*4
    q=torch.zeros(784,768); d,i=A.X.nn(q,x['bottle']); assert d.shape==(784,) and i.shape==(784,)
