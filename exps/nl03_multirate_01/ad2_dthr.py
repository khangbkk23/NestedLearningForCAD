"""NL03-MULTIRATE-01 deviation D-THR (owner task 2026-11-25, agents/tasks/task_ad2_dthr_g3e_paper_01.md, PART 1.1).

Arms U and E only. Binary-map threshold replaced by the MVTec AD 2 paper protocol, declared before any SegF1 or
server result:
    theta_s = mean + 3 * std of the per-pixel scores over validation/ images of scenario s,
    with the arm's state at that scenario's acquisition stage (same stage as the locked rule),
    per-pixel scores = the native-resolution float32 A1 map (R2.maps_of -> final._native), population std (fp64);
    binary map = final-stage float32 native map > theta_s.
Continuous maps are unchanged. Old thresholded maps are MOVED (not deleted) to AD2_SUBMISSION_OLDTHR/<arm>/.
The U/E states are rebuilt with the locked final.Learner; the stored locked theta (max validation image score) must
be reproduced bitwise, otherwise STOP.

    python -B -m exps.nl03_multirate_01.ad2_dthr theta | rethreshold
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np
import torch

from exps.nl03_multirate_01 import final as FN
from exps.nl03_multirate_01 import run as RN
from exps.nl03_multirate_01 import spec as SP

ARMS = ('U', 'E')
RULE = ('D-THR: theta_s = mean + 3*std (population, fp64) of native-resolution float32 A1 pixel scores over all '
        'validation/ images of scenario s, arm state at the acquisition stage of s; binary = final-stage float32 '
        'native map > theta_s')
VAL_GRIDS = FN.AD2_CACHE / 'grids_val'


def _free_gib(p):
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 2**30


@torch.no_grad()
def theta():
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    rec = FN.verify_final()
    sc = RN.rj(FN.AD2_OUT / 'AD2_SCORE.json')
    if 'theta_dthr' in sc:
        FN.log('D-THR theta exists'); return
    scens = sc['scenarios']
    counts = [len(FN.ad2_load(s, 'train')['rows']) for s in scens]
    assert counts == sc['counts']
    L = FN.Learner(list(ARMS), rec['CB'], counts)
    out, check = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    stats = {a: {} for a in ARMS}
    t0 = time.time()
    try:
        for l, s in enumerate(scens):
            tr = FN.ad2_load(s, 'train')
            x16 = tr['tokens'].reshape(-1, SP.D).cuda()
            L.task(x16, l)
            del x16, tr; torch.cuda.empty_cache()
            va = FN.ad2_load(s, 'validation')
            for a in ARMS:
                K, W = L.keys_at(a, l)
                g = FN.CO_energy(va['tokens'].reshape(-1, SP.D), K, W).reshape(len(va['rows']), 28, 28)
                p = VAL_GRIDS / a / f'{s}__validation.pt'
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'rows': va['rows'], 'hws': va['hws'], 'grid': g, 'stage': l}, p)
                maps = R2.maps_of(g)
                old = float(A.image_scores(maps).max())
                check[a][s] = {'old_theta_reproduced': old == sc['theta'][a][s], 'old': old, 'stored': sc['theta'][a][s]}
                if old != sc['theta'][a][s]:
                    raise SystemExit(f'STOP: locked theta not reproduced {a} {s}: {old} vs {sc["theta"][a][s]}')
                s1 = s2 = 0.0; n = 0
                for m, hw in zip(maps, va['hws']):
                    v = FN._native(m, hw).astype(np.float64)
                    s1 += v.sum(); s2 += (v * v).sum(); n += v.size
                mu = s1 / n; sd = float(np.sqrt(max(0.0, s2 / n - mu * mu)))
                out[a][s] = float(mu + 3 * sd)
                stats[a][s] = {'mean': mu, 'std': sd, 'n_pixels': n, 'n_images': len(va['rows']), 'stage': l}
            FN.log('D-THR', l, s, {a: round(out[a][s], 4) for a in ARMS}, {a: round(sc['theta'][a][s], 4) for a in ARMS})
    finally:
        RN.ledger_add({'phase': 'dthr_ad2_theta', 'seconds': time.time() - t0, 'gpu': True})
    sc['theta_dthr'] = out
    sc['theta_dthr_stats'] = stats
    sc['theta_dthr_rule'] = RULE
    sc['theta_dthr_arms'] = list(ARMS)
    sc['theta_dthr_old_theta_check'] = check
    sc['theta_dthr_time'] = time.strftime('%FT%T')
    RN.wj(FN.AD2_OUT / 'AD2_SCORE.json', sc)
    FN.log('D-THR theta written', out)


README_ADD = """
## Deviation D-THR (owner task 2026-11-25), arms U and E only

`anomaly_images_thresholded/` was regenerated with the MVTec AD 2 paper threshold protocol:
theta_s = mean + 3 * std of the native-resolution per-pixel scores over all validation/ images of scenario s,
computed with this arm's state at the scenario's acquisition stage; binary = final-stage float32 map > theta_s.
theta_s per scenario: AD2_SCORE.json -> theta_dthr. The old maps (theta = max validation image score) were moved to
AD2_SUBMISSION_OLDTHR/{arm}/anomaly_images_thresholded/ (not deleted). Continuous maps (anomaly_images/) unchanged.
"""


def rethreshold():
    import cv2
    from exps.nanm_r2 import run as R2
    FN.verify_final()
    sc = RN.rj(FN.AD2_OUT / 'AD2_SCORE.json')
    th = sc['theta_dthr']
    for a in ARMS:
        base = FN.AD2_OUT / 'AD2_SUBMISSION' / a
        if (base / 'DTHR_DONE').exists():
            continue
        old = FN.AD2_OUT / 'AD2_SUBMISSION_OLDTHR' / a / 'anomaly_images_thresholded'
        if not old.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            os.rename(base / 'anomaly_images_thresholded', old)          # move, never delete
        n, pos = 0, {}
        for s in sc['scenarios']:
            pos[s] = {}
            for split in ('test_private', 'test_private_mixed'):
                d = torch.load(FN.AD2_CACHE / 'grids' / a / f'{s}__{split}.pt', weights_only=False)
                maps = R2.maps_of(d['grid'])
                fr = []
                for r, hw, m in zip(d['rows'], d['hws'], maps):
                    nat = FN._native(m, hw)
                    from pathlib import Path
                    pt = base / 'anomaly_images_thresholded' / s / split / f'{Path(r).stem}.png'
                    pt.parent.mkdir(parents=True, exist_ok=True)
                    b = nat > th[a][s]
                    cv2.imwrite(str(pt), (b * 255).astype(np.uint8))
                    fr.append(float(b.mean())); n += 1
                pos[s][split] = float(np.mean(fr))
            assert _free_gib('/mnt/e') >= 30, 'STOP: E: below 30 GiB'
            FN.log('D-THR PNG', a, s, pos[s])
        (base / 'README.md').write_text((base / 'README.md').read_text() + README_ADD.format(arm=a))
        RN.wj(base / 'DTHR_DONE', {'maps': n, 'theta': th[a], 'positive_fraction': pos, 'time': time.strftime('%FT%T')})
        FN.log('D-THR DONE', a, n)


if __name__ == '__main__':
    {'theta': theta, 'rethreshold': rethreshold}[sys.argv[1]]()
