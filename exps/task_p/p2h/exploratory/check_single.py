"""Supplemental saturated clip/cap test and actual single-map resume."""
import math
import torch
from collections import OrderedDict
from .learner import write_single
from .test_controls import run
from .design import OUT,save


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    m=OrderedDict(block=torch.zeros(3,16,dtype=torch.float64))
    g=torch.ones_like(m['block'])*100
    written=write_single(m,[g])
    assert abs(float(written['block'].norm())-.01*math.sqrt(2))<1e-12
    m['block'].fill_(100)
    capped=write_single(m,[torch.zeros_like(g)])
    assert abs(float(capped['block'].norm())-math.sqrt(2))<1e-12
    details=run('single_memory_16')
    save(OUT/'single_clip_check.json',dict(passed=True,gradient_clip=math.sqrt(2),norm_cap=math.sqrt(2),details=details))
    print('SINGLE_CLIP_AND_RESUME_PASS',flush=True)


if __name__=='__main__': main()
