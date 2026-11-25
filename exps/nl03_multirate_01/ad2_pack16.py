"""NL03-MULTIRATE-01 engineering deviation D-PKG16 (post-FINAL_LOCK, format only, no method/threshold change).

final.ad2_package wrote uncompressed float32 TIFFs (~20 MB/map, 164 GB for 2.7 arms) and stopped on the E: free-space
guard. This rewrites AD2_SUBMISSION/<arm>/ from the SAME locked grids with the SAME readout (R2.maps_of -> final._native)
and the SAME thresholded PNG rule (float32 native map > theta[arm][scenario]); only the anomaly TIFF encoding changes to
float16 + zlib (lossless deflate; float16 is the dtype the MVTec AD 2 tooling is documented to expect, to be checked by
the owner with the official utilities). Files are overwritten one by one, so disk use only shrinks.
tifffile is imported from an existing local conda env (no download).

    python -B -m exps.nl03_multirate_01.ad2_pack16
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.nl03_multirate_01 import final as FN
from exps.nl03_multirate_01 import run as RN

_TF = sorted(glob.glob(os.path.expanduser('~/miniconda3/envs/pytorch_env/lib/python3*/site-packages')))
sys.path.append(_TF[0])
import tifffile  # noqa: E402

README16 = """# MVTec AD 2 server submission: arm {arm} (NL03-MULTIRATE-01, HELD_OUT_CONFIRMATION)

- Arm: {arm} (implementation `{impl}`); locked at FINAL_LOCK; stream = 8 scenarios alphabetical; seed 2711; fit on
  train/ only.{dup}
- anomaly_images/<scenario>/<split>/<image>.tiff: float16, zlib-compressed, final-stage anomaly map at native
  resolution (A1 readout, 392 map bilinearly resized). Max relative float16 rounding error ~3e-4 vs the float32 map.
- anomaly_images_thresholded/<scenario>/<split>/<image>.png: binary (0/255), float32 native map > theta, theta = max
  A1 image score over that scenario's validation/ images at its acquisition stage (registered before any test access).
- Official code utils: CODE_UTILS_MISSING (not on K:). Before upload, run the official MVTec AD 2 check/prepare script
  on this folder (it validates the layout and dtype and compresses the submission).
- Upload: the owner uploads to benchmark.mvtec.com. Claude never submits.
- Deviation D-PKG16: first packaging wrote float32 uncompressed TIFF (164 GB) and stopped on the disk guard; this
  folder was regenerated from the same locked grids with float16 + zlib encoding only.
"""


def main():
    rec = FN.verify_final()
    sc = RN.rj(FN.AD2_OUT / 'AD2_SCORE.json')
    from exps.nanm_r2 import run as R2
    order = ['P'] + [a for a in sc['arms'] if a != 'P']
    status = {'format': 'float16 TIFF zlib + binary PNG', 'deviation': 'D-PKG16', 'code_utils': 'CODE_UTILS_MISSING',
              'arms': {}}
    t0 = time.time()
    for a in order:
        base = FN.AD2_OUT / 'AD2_SUBMISSION' / a
        if (base / 'DONE16').exists():
            status['arms'][a] = RN.rj(base / 'DONE16'); continue
        n, nbytes, maxrel = 0, 0, 0.0
        for s in sc['scenarios']:
            th = sc['theta'][a][s]
            for split in ('test_private', 'test_private_mixed'):
                d = torch.load(FN.AD2_CACHE / 'grids' / a / f'{s}__{split}.pt', weights_only=False)
                maps = R2.maps_of(d['grid'])
                for r, hw, m in zip(d['rows'], d['hws'], maps):
                    nat = FN._native(m, hw)
                    name = Path(r).stem
                    pa = base / 'anomaly_images' / s / split / f'{name}.tiff'
                    pt = base / 'anomaly_images_thresholded' / s / split / f'{name}.png'
                    pa.parent.mkdir(parents=True, exist_ok=True); pt.parent.mkdir(parents=True, exist_ok=True)
                    h16 = nat.astype(np.float16)
                    tmp = pa.with_suffix('.tmp16')
                    tifffile.imwrite(tmp, h16, compression='zlib')
                    os.replace(tmp, pa)
                    import cv2
                    cv2.imwrite(str(pt), ((nat > th) * 255).astype(np.uint8))
                    maxrel = max(maxrel, float(np.abs(h16.astype(np.float32) - nat).max() / max(1e-12, float(nat.max()))))
                    n += 1; nbytes += pa.stat().st_size + pt.stat().st_size
                del d, maps
            FN.log('AD2 PACK16', a, s, n)
        dup = ''
        if a in ('CB', 'F') and rec.get('CB', sc.get('CB')) == 'C5':
            dup = '\n- NOTE: CB selected C5 at Phase I, so CB and F are the same implementation (identical maps).'
        (base / 'README.md').write_text(README16.format(arm=a, impl=FN.arm_impl(a, sc.get('CB', 'C5')), dup=dup))
        st = {'maps': n, 'bytes': nbytes, 'max_rel_f16_err': maxrel, 'time': time.strftime('%FT%T')}
        RN.wj(base / 'DONE16', st)
        (base / 'DONE').write_text(time.strftime('%FT%T'))   # lets final.ad2_package skip this arm
        status['arms'][a] = st
        RN.wj(FN.AD2_OUT / 'AD2_PACKAGE.json', status)
    status['seconds'] = time.time() - t0
    RN.wj(FN.AD2_OUT / 'AD2_PACKAGE.json', status)
    FN.log('AD2 PACK16 DONE', {a: v['maps'] for a, v in status['arms'].items()})


if __name__ == '__main__':
    main()
