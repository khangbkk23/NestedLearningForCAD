"""NL03-MULTIRATE-01 Phase M: MVTec AD 2 custody monitor (custody only; never decodes an image or reads a split).

Polls /mnt/k at depth 1 every 15 min. Accepts a folder with 8 scenario subfolders, each containing
train/ validation/ test_public/ test_private/ test_private_mixed/. If only a finished archive is present, waits
until its size is unchanged for >= 15 min, integrity-tests it, checks disk headroom (K: >= 30 GiB free after
extraction) and extracts it under /mnt/k (never downloads anything). Ready = structure OK, no partial files, and
recursive file counts unchanged across two checks 15 min apart; then it byte-hashes every file (custody hash, no
decoding) and writes AD2_MANIFEST.json.

    python3 -B exps/nl03_multirate_01/ad2_monitor.py            # detached via setsid nohup
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

K = Path('/mnt/k')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/nl03_multirate_01/ad2'
LOG = OUT / 'AD2_MONITOR.jsonl'
MANIFEST = OUT / 'AD2_MANIFEST.json'
POLL = 900
MAX_HOURS = 96
SPLITS = ('train', 'validation', 'test_public', 'test_private', 'test_private_mixed')
KNOWN = {'$RECYCLE.BIN', '.disk', 'EFI', 'NLCAD', 'ProgramData', 'System Volume Information', 'boot', 'boot.catalog',
         'casper', 'dists', 'install', 'md5sum.txt', 'pool', 'preseed'}
PARTIAL = ('.part', '.crdownload', '.tmp')
ARCH = ('.tar', '.tar.gz', '.tgz', '.tar.xz', '.txz', '.zip', '.7z')
MIN_FREE = 30 * 2**30


def log(ev, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'time': time.strftime('%FT%T'), 'event': ev, **kw}
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def top():
    out = []
    for p in K.iterdir():
        if p.name in KNOWN:
            continue
        try:
            out.append((p, p.is_dir(), 0 if p.is_dir() else p.stat().st_size))
        except OSError:
            pass
    return out


def scen_root(d: Path):
    """Return the folder holding the 8 scenarios (d itself) or None."""
    try:
        subs = [s for s in d.iterdir() if s.is_dir()]
    except OSError:
        return None
    ok = [s for s in subs if all((s / x).is_dir() for x in SPLITS)]
    return d if len(ok) == 8 else None


def count(d: Path):
    n, partial = 0, []
    for dp, _, fs in os.walk(d):
        for f in fs:
            n += 1
            if f.endswith(PARTIAL):
                partial.append(str(Path(dp) / f))
    return n, partial


def is_arch(p: Path):
    return any(p.name.lower().endswith(a) for a in ARCH)


def integrity(p: Path):
    nm = p.name.lower()
    if nm.endswith('.zip'):
        r = subprocess.run(['unzip', '-tq', str(p)], capture_output=True, text=True)
        if r.returncode:
            return False, r.stdout[-500:] + r.stderr[-500:], None, None
        r2 = subprocess.run(['unzip', '-Z1', str(p)], capture_output=True, text=True)
        names = r2.stdout.split('\n')
        r3 = subprocess.run(['unzip', '-l', str(p)], capture_output=True, text=True)
        tot = int(r3.stdout.strip().split('\n')[-1].split()[0])
        return True, 'unzip -t ok', names, tot
    if nm.endswith('.7z'):
        if not shutil.which('7z'):
            return False, '7z not installed', None, None
        r = subprocess.run(['7z', 't', str(p)], capture_output=True, text=True)
        return r.returncode == 0, r.stdout[-500:], None, None
    r = subprocess.run(['tar', '-tvf', str(p)], capture_output=True, text=True)
    if r.returncode:
        return False, r.stderr[-800:], None, None
    names, tot = [], 0
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 5)
        if len(parts) >= 6:
            try:
                tot += int(parts[2])
            except ValueError:
                pass
            names.append(parts[5])
    return True, 'tar -tvf ok', names, tot


def extract(p: Path, names, tot):
    free = shutil.disk_usage(K).free
    if free - tot < MIN_FREE:
        log('EXTRACT_REFUSED_DISK', archive=p.name, free=free, need=tot)
        return False
    tops = {n.strip('./').split('/')[0] for n in names if n.strip('./')}
    dest = K if len(tops) == 1 else K / p.name.split('.')[0]
    dest.mkdir(exist_ok=True)
    log('EXTRACT_START', archive=p.name, dest=str(dest), top_entries=sorted(tops)[:10], uncompressed=tot)
    nm = p.name.lower()
    if nm.endswith('.zip'):
        cmd = ['unzip', '-q', '-n', str(p), '-d', str(dest)]
    elif nm.endswith('.7z'):
        cmd = ['7z', 'x', '-aos', f'-o{dest}', str(p)]
    else:
        cmd = ['tar', '--skip-old-files', '-xf', str(p), '-C', str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log('EXTRACT_END', archive=p.name, rc=r.returncode, err=r.stderr[-500:])
    return r.returncode == 0


def sha(p: Path):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def manifest(root: Path, n):
    files, per = {}, {}
    for s in sorted(x for x in root.iterdir() if x.is_dir()):
        for sp in SPLITS:
            c = 0
            for dp, _, fs in os.walk(s / sp):
                for f in sorted(fs):
                    q = Path(dp) / f
                    files[str(q.relative_to(root))] = {'sha256': sha(q), 'bytes': q.stat().st_size}
                    c += 1
            per.setdefault(s.name, {})[sp] = c
    tmp = MANIFEST.with_suffix('.tmp_writing')
    tmp.write_text(json.dumps({'root': str(root), 'n_files': len(files), 'n_files_walk': n,
                               'per_scenario_split_counts': per, 'files': files,
                               'note': 'custody hashes only; no image decoded; no split read for modelling',
                               'time': time.strftime('%FT%T')}, indent=1))
    tmp.rename(MANIFEST)
    log('READY', root=str(root), n_files=len(files), per_scenario=per)


def main():
    if MANIFEST.exists():
        log('ALREADY_READY'); return
    log('START', pid=os.getpid(), poll_s=POLL)
    t0, prev_sizes, prev_counts, tested = time.time(), {}, {}, set()
    while time.time() - t0 < MAX_HOURS * 3600:
        ents = top()
        partial_top = [p.name for p, d, _ in ents if not d and p.name.endswith(PARTIAL)]
        cands = [p for p, d, _ in ents if d and scen_root(p)]
        state = {'partial_top': {n: s for (p, d, s) in ents for n in [p.name] if n in partial_top},
                 'dirs': [p.name for p, d, _ in ents if d], 'archives': {p.name: s for p, d, s in ents if not d and is_arch(p)},
                 'k_free_gib': round(shutil.disk_usage(K).free / 2**30, 1)}
        log('POLL', **state, candidates=[c.name for c in cands])
        for c in cands:
            n, part = count(c)
            if not part and not partial_top and prev_counts.get(c.name) == n:
                manifest(c, n); return
            prev_counts[c.name] = n
            log('CANDIDATE', root=c.name, n_files=n, partial_inside=len(part))
        if not cands:
            for p, d, s in ents:
                if d or not is_arch(p) or p.name in tested:
                    continue
                if prev_sizes.get(p.name) == s and not partial_top:
                    ok, msg, names, tot = integrity(p)
                    log('INTEGRITY', archive=p.name, ok=ok, msg=msg[-300:] if msg else msg)
                    tested.add(p.name)
                    if ok:
                        extract(p, names or [], tot or 0)
                prev_sizes[p.name] = s
        time.sleep(POLL)
    log('TIMEOUT', hours=MAX_HOURS)


if __name__ == '__main__':
    sys.exit(main())
