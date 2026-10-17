"""Restore ONLY exact newline variants of historical frozen bytes after a fresh Git checkout.
Default is check-only; --apply saves every changed preimage before writing matching bytes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import time
ROOT=Path(__file__).resolve().parents[2]


def variants(raw,expected):
    lf=raw.replace(b'\r\n',b'\n')
    found=[(kind,value) for kind,value in [('LF',lf),('CRLF',lf.replace(b'\n',b'\r\n'))]
           if hashlib.sha256(value).hexdigest()==expected]
    if len(found)!=1:raise ValueError('difference is NOT a verified pure newline variant')
    return found[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');args=p.parse_args()
    ref=json.loads((ROOT/'notebook_support/w7_resume_fix/frozen_reference.json').read_text())
    changes=[]
    for name,expected in ref.items():
        raw=(ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:
            kind,value=variants(raw,expected);changes.append((name,raw,value,kind))
    print('Exact newline-only mismatches:',len(changes))
    if not args.apply or not changes:return 1 if changes else 0
    folder=ROOT/'results/recovery_stage1';folder.mkdir(parents=True,exist_ok=True)
    stamp=str(time.time_ns());backup=folder/f'checkout_before_{stamp}.zip'
    with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw,_,_ in changes:z.writestr(name,raw)
    for name,_,value,_ in changes:(ROOT/name).write_bytes(value)
    receipt=dict(schema='recovery_checkout_newline_normalization_v1',semantic_change=False,
        backup=backup.relative_to(ROOT).as_posix(),files=[dict(path=name,before_sha256=hashlib.sha256(raw).hexdigest(),
        after_sha256=hashlib.sha256(value).hexdigest(),line_endings=kind) for name,raw,value,kind in changes])
    (folder/f'checkout_normalization_{stamp}.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    return 0

if __name__=='__main__':raise SystemExit(main())