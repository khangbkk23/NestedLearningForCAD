"""NL03-MULTIRATE-01 PART 1.2 (owner task 2026-11-25): official upload archives for arms U and E only.

Per arm: real copies (no symlinks) of anomaly_images/ + anomaly_images_thresholded/ into a clean staging dir, the
official check (check_submission), the official compression (compress_submission, cwd = UPLOAD/ so the archive is
UPLOAD/<arm>.tar.gz), then an archive verification (full gzip read; member names and sizes == staging). The staging
copy is deleted only after the check and the verification pass. Keeps >= 30 GiB free on E:.
Stdlib + the official utils only; run with an env that has tifffile/Pillow/tqdm (pytorch_env):

    ~/miniconda3/envs/pytorch_env/bin/python -I exps/nl03_multirate_01/ad2_upload.py <repo_root>
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

UTILS = '/mnt/k/mvtec_ad2_code_utils/MVTecAD2_public_code_utils'
ARMS = ('U', 'E')
MIN_FREE = 30 * 2**30


def free(p):
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize


def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main(root):
    sys.path.insert(0, UTILS)
    import check_and_prepare_data_for_upload as C
    from utils import compress_submission
    ad2 = Path(root) / 'results/hope_cad/nl03_multirate_01/ad2'
    up = ad2 / 'UPLOAD'; up.mkdir(exist_ok=True)
    stage_root = ad2 / '_staging'
    manp = up / 'UPLOAD_MANIFEST.json'
    man = json.loads(manp.read_text()) if manp.exists() else {'arms': {}}
    for a in ARMS:
        if a in man['arms'] and man['arms'][a].get('verified'):
            log('SKIP', a); continue
        src = ad2 / 'AD2_SUBMISSION' / a
        assert (src / 'DTHR_DONE').exists(), f'STOP: D-THR maps missing for {a}'
        need = du(src / 'anomaly_images') + du(src / 'anomaly_images_thresholded')
        if free('/mnt/e') - 2.1 * need < MIN_FREE:
            raise SystemExit(f'STOP: E: would drop below 30 GiB (need ~{2.1 * need / 2**30:.1f} GiB)')
        st = stage_root / a
        if st.exists():
            raise SystemExit(f'STOP: staging dir exists {st} (inspect manually)')
        st.mkdir(parents=True)
        t0 = time.time()
        for d in ('anomaly_images', 'anomaly_images_thresholded'):
            shutil.copytree(src / d, st / d, symlinks=False)
        files = sorted(str(f.relative_to(st)) for f in st.rglob('*') if f.is_file())
        assert not any(f.is_symlink() for f in st.rglob('*')), 'symlink in staging'
        log('STAGED', a, len(files), '%.1f GiB' % (need / 2**30), '%.0fs' % (time.time() - t0))
        C.check_submission(str(st))                                   # raises SubmissionException on failure
        log('CHECK OK', a)
        cwd = os.getcwd(); os.chdir(up)
        try:
            compress_submission(str(st))
        finally:
            os.chdir(cwd)
        arc = up / f'{a}.tar.gz'
        sizes = {f: (st / f).stat().st_size for f in files}
        seen = {}
        with tarfile.open(arc, 'r:gz') as tf:                            # full read: verifies the gzip CRC
            for m in tf:
                if m.isfile():
                    assert m.name.startswith(f'{a}/'), m.name
                    seen[m.name[len(a) + 1:]] = m.size
                    fo = tf.extractfile(m); fo.read(); fo.close()
        ok = seen == sizes
        rec = {'archive': str(arc.relative_to(root)), 'bytes': arc.stat().st_size, 'sha256': fsha(arc),
               'n_files': len(files), 'official_check': 'All checks successful', 'archive_members_match_staging': ok,
               'verified': ok, 'source': str(src.relative_to(root)), 'time': time.strftime('%FT%T'),
               'seconds': time.time() - t0}
        man['arms'][a] = rec
        manp.write_text(json.dumps(man, indent=1))
        if not ok:
            raise SystemExit(f'STOP: archive members differ from staging for {a}; staging kept')
        shutil.rmtree(st)                                               # only after check + verification
        log('ARCHIVE OK', a, rec['bytes'], rec['sha256'][:16], 'free E: %.0f GiB' % (free('/mnt/e') / 2**30))
    if stage_root.exists() and not any(stage_root.iterdir()):
        stage_root.rmdir()
    man['code_utils'] = UTILS
    man['time'] = time.strftime('%FT%T')
    manp.write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
