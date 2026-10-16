"""Audited JSON-transport fix around the immutable W7 runner; no model changes.

The frozen runner compares decoded JSON lists to live metadata tuples. Normalize
only the proposed identity, then use its ORIGINAL full equality/digest check.
Both this file and its receipt travel in every output. No old lock is rewritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
REFERENCE_DIGEST='86c941b4aa1b2e82682c2d0ed31e97325b3e02442f6f88efa333511884db25ad'
FIX_ID='w7_json_lock_transport_v1'


def normalize_json_identity(identity):
    return json.loads(json.dumps(identity,sort_keys=True,ensure_ascii=False,allow_nan=False))


def verify_frozen_source():
    from dataset.benchmark_manifest_v1 import canonical_digest
    from training.hope_w7_contract import source_hashes
    reference=json.loads((Path(__file__).parent/'frozen_reference.json').read_text(encoding='utf8'))
    if canonical_digest(reference)!=REFERENCE_DIGEST:
        raise ValueError('transport fix reference was changed')
    current=source_hashes()
    if current!=reference:
        changed=sorted(k for k in set(reference)|set(current) if reference.get(k)!=current.get(k))
        raise ValueError('transport fix rejects frozen source drift: '+', '.join(changed))
    return reference


def install_json_lock_fix():
    from training import hope_w7_contract as contract
    original=contract.lock_json
    if getattr(original,'transport_fix_id',None)==FIX_ID:
        return original
    def canonical_lock(path,identity):
        return original(path,normalize_json_identity(identity))
    canonical_lock.transport_fix_id=FIX_ID
    contract.lock_json=canonical_lock
    return canonical_lock


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if '--help' in argv or '-h' in argv:
        sys.argv=[str(ROOT/'scripts/hope_cad_v2/run_w7.py'),*argv]
        runpy.run_path(sys.argv[0],run_name='__main__')
        return 0
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--phase',choices=('preflight','7.1','7.2','7.3'),required=True)
    p.add_argument('--job',required=True)
    args,_=p.parse_known_args(argv)
    reference=verify_frozen_source()
    from training.hope_w7_contract import job_by_id
    from training.hope_runtime_v4 import atomic_json
    job_by_id(args.job)
    folder=args.work_dir/'runs'/args.job/'transport_fixes'/FIX_ID
    folder.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).read_bytes()
    launcher_sha=hashlib.sha256(source).hexdigest()
    archived=folder/(launcher_sha+'.py')
    if archived.exists() and archived.read_bytes()!=source:
        raise ValueError('existing launcher provenance differs')
    archived.write_bytes(source)
    receipt=dict(schema='hope_w7_transport_fix_v1',fix_id=FIX_ID,phase=args.phase,job=args.job,
        launcher_sha256=launcher_sha,source_reference_digest=REFERENCE_DIGEST,
        verified_frozen_files=len(reference),frozen_source_unchanged=True,
        operation='JSON-normalize proposed identity before ORIGINAL strict lock check',
        existing_locks_rewritten=False,checkpoint_weights_rewritten=False,
        model_or_scoring_changed=False,runner_exit_code=None,reportable=False)
    path=folder/(args.phase+'_'+uuid.uuid4().hex+'.json')
    atomic_json(receipt,path)
    install_json_lock_fix()
    print('[W7 LOCK FIX] '+FIX_ID+': '+str(len(reference))+' frozen files verified; existing locks preserved',flush=True)
    code=0
    previous=sys.argv
    try:
        sys.argv=[str(ROOT/'scripts/hope_cad_v2/run_w7.py'),*argv]
        runpy.run_path(sys.argv[0],run_name='__main__')
    except SystemExit as error:
        code=error.code if isinstance(error.code,int) else (0 if error.code is None else 1)
    except BaseException:
        code=1
        raise
    finally:
        sys.argv=previous
        receipt['runner_exit_code']=code
        atomic_json(receipt,path)
    return code


if __name__=='__main__':
    raise SystemExit(main())
