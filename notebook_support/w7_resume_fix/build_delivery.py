"""Rebuild ONLY notebook/transport files around the pinned original W7 bundle."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from notebook_support.w7_resume_fix.launcher import verify_frozen_source,FIX_ID

PARENT_SHA='1b402113ae91aa2ee9f07ff976d3ad73d1fc173327bd8bc6d93c4c438473394a'
W6_SELECTOR="""# Select only the reviewed W6; equivalent extracted copies are harmless.
from training.hope_w7_contract import W6_SHA256, W6_MANIFEST_SHA256
if W6_OVERRIDE:
    w6source=Path(W6_OVERRIDE)
else:
    reviewed=[p.parent for p in INPUT.rglob('artifact_manifest.json')
              if sha(p)==W6_MANIFEST_SHA256]
    if not reviewed:
        reviewed=[p for p in INPUT.rglob('hope_cad_v4_normal_evidence.zip')
                  if sha(p)==W6_SHA256]
    if not reviewed:
        raise RuntimeError('No reviewed completed W6 found; check the input or W6_OVERRIDE')
    w6source=sorted(set(reviewed))[0]
print('Selected reviewed W6:',w6source,flush=True)
"""
OLD_SELECTOR="""w6s=list(INPUT.rglob('hope_cad_v4_normal_evidence.zip'))
if not w6s:
    w6s=[p.parent for p in INPUT.rglob('artifact_manifest.json')
         if json.loads(p.read_text()).get('schema')=='hope_normal_evidence_v4']
w6source=Path(W6_OVERRIDE) if W6_OVERRIDE else unique(w6s,'Completed reviewed W6')
"""
FINAL_GUARD="""
# Export is complete BEFORE failure is raised; never show a failed runner as success.
expected_status={'preflight':'w7_preflight_passed','7.1':'7.1_complete_pending_review',
                 '7.2':'7.2_complete_pending_review','7.3':'7.3_job_complete_pending_suite'}[PHASE]
status=json.loads((WORK/'handoff.json').read_text()).get('status') if (WORK/'handoff.json').exists() else None
if EXIT_CODE!=0 or status not in (expected_status,'paused_at_atomic_boundary'):
    raise RuntimeError(f'W7 did NOT complete: exit={EXIT_CODE}, status={status}. Evidence ZIP was exported; keep it for review.')
print('VERIFIED RUNNER STATUS:',status,flush=True)
"""


def patched_notebook(nb):
    # Always patch from the immutable PARENT bundle, not an edited user notebook.
    import copy
    nb=copy.deepcopy(nb)
    boot=nb['cells'][1]['source']
    if isinstance(boot,list):boot=''.join(boot)
    if OLD_SELECTOR not in boot:raise ValueError('unexpected original W6 bootstrap')
    boot=boot.replace(OLD_SELECTOR,W6_SELECTOR)
    boot=boot.replace("print('Verified source files:',len(manifest['files']),'origin:',manifest['origin'])",
                      "print('Verified source files:',len(manifest['files']),'origin:',manifest['origin'],flush=True)")
    boot += "\nif PHASE=='7.2' and not (WORK/'runs'/JOB/'metrics/final_macro.json').is_file():\n    raise RuntimeError('7.2 needs completed 7.1 evidence, not preflight evidence')\n"
    nb['cells'][1]['source']=boot
    command=nb['cells'][2]['source']
    if isinstance(command,list):command=''.join(command)
    command=command.replace("'scripts/hope_cad_v2/run_w7.py'","'notebook_support/w7_resume_fix/launcher.py'")
    nb['cells'][2]['source']=command
    final=nb['cells'][3]['source']
    if isinstance(final,list):final=''.join(final)
    nb['cells'][3]['source']=final+FINAL_GUARD
    title=nb['cells'][0]['source']
    if isinstance(title,list):title=''.join(title)
    nb['cells'][0]['source']=title+'\nTransport fix v1: frozen Python dependencies unchanged; tuple/list JSON lock comparison corrected by a hash-guarded launcher. Runtime receipt is exported. Failed runners raise AFTER ZIP export.\n'
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            compile(cell['source'],'w7_transport_notebook','exec')
            cell['outputs']=[];cell['execution_count']=None
    return nb


def main():
    verify_frozen_source()
    parent=ROOT/'ultils/hope_cad_w7_source_bundle.zip'
    with parent.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=PARENT_SHA:raise ValueError('original source bundle differs')
    destination=ROOT/'ultils/w7_resume_fix/hope_cad_w7_source_bundle.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(parent) as z:
        old=json.loads(z.read('bundle_manifest.json'))
        files={n:z.read(n) for n in old['files']}
        for name in sorted(files):
            if name.startswith('notebooks/hope_cad_w7_') and 'report_cpu' not in name:
                patched=patched_notebook(json.loads(files[name]))
                raw=(json.dumps(patched,ensure_ascii=False,indent=1)+'\n').encode()
                target=ROOT/name
                # Never silently overwrite notebook edits outside this exact fix.
                if target.exists() and target.read_bytes() not in (files[name],raw):
                    raise ValueError('local notebook has unrelated edits: '+name)
                target.write_bytes(raw);files[name]=raw
        for path in (ROOT/'notebook_support/w7_resume_fix').glob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.md'):
                files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
        note='docs/main_spec/transport_fixes/w7_json_lock_fix_v1.md'
        files[note]=(ROOT/note).read_bytes()
        manifest=dict(schema='hope_w7_source_v1',origin=dict(old['origin'],
            transport_fix_id=FIX_ID,parent_source_bundle_sha256=PARENT_SHA),
            files={n:hashlib.sha256(raw).hexdigest() for n,raw in sorted(files.items())})
        tmp=destination.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as out:
            for n,raw in sorted(files.items()):out.writestr(n,raw)
            out.writestr('bundle_manifest.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(tmp) as out:
        for n,h in manifest['files'].items():
            if hashlib.sha256(out.read(n)).hexdigest()!=h:raise RuntimeError('bundle checksum mismatch')
    tmp.replace(destination)
    print(json.dumps(dict(path=str(destination),files=len(files),bytes=destination.stat().st_size,
                         sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                         frozen_source_unchanged=True),indent=2))


if __name__=='__main__':main()
