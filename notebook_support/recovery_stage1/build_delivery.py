"""Build two Kaggle notebooks and a self-verifying source bundle; never edits frozen files."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from notebook_support.recovery_stage1.common import SOURCE_SCHEMA,source_identity,PROTOCOL,digest

BOOT=r'''
import os,sys,time,json,hashlib,zipfile,shutil
from pathlib import Path,PurePosixPath
STARTED=time.monotonic()
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
INPUT=Path('/kaggle/input')
REPO=Path('/tmp/hope_recovery_stage1_source')
WORK=Path('/kaggle/working/hope_recovery_stage1')
W6=Path('/tmp/hope_recovery_parent_w6')
ASSETS=Path('/tmp/hope_recovery_assets')
OUTPUT=Path('/kaggle/working/hope_cad_recovery_stage1_evidence.zip')
SOURCE_OVERRIDE=None
W6_OVERRIDE=None
W72_OVERRIDE=None
RESUME_OVERRIDE=None
MVTEC_OVERRIDE=None
CHECKPOINT_OVERRIDE=None
SESSION_HOURS=10.5
JOBS=['C1_seed0','A1_seed0','CADIC10000_seed0']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def safe(root,name):
    rel=PurePosixPath(name)
    path=(root/name).resolve()
    if rel.is_absolute() or '..' in rel.parts or rel.as_posix()!=name or chr(92) in name or ':' in name or not path.is_relative_to(root.resolve()):
        raise ValueError('Unsafe source path: '+name)
    return path
print('[BOOTSTRAP] locating recovery source; elapsed',round(time.monotonic()-STARTED,1),'s',flush=True)
sources=[]
for p in INPUT.rglob('bundle_manifest.json'):
    try:
        if json.loads(p.read_text()).get('schema')=='hope_recovery_stage1_source_v1':sources.append(p.parent)
    except ValueError:pass
if not sources:sources=list(INPUT.rglob('hope_cad_recovery_stage1_source*.zip'))
if SOURCE_OVERRIDE:source=Path(SOURCE_OVERRIDE)
else:
    if not sources:raise ValueError('Attach the NEW hope_cad_recovery_stage1_source bundle')
    manifests=[]
    for p in sorted(set(sources)):
        if p.is_dir():raw=(p/'bundle_manifest.json').read_bytes()
        else:
            with zipfile.ZipFile(p) as z:raw=z.read('bundle_manifest.json')
        manifests.append((hashlib.sha256(raw).hexdigest(),p))
    if len({h for h,p in manifests})!=1:raise ValueError('Multiple different source versions; remove old inputs or set SOURCE_OVERRIDE')
    source=manifests[0][1]
if source.is_dir():
    manifest=json.loads((source/'bundle_manifest.json').read_text());archive=None
else:
    archive=zipfile.ZipFile(source);names=archive.namelist()
    if len(names)!=len(set(names)):raise ValueError('Duplicate source entries')
    manifest=json.loads(archive.read('bundle_manifest.json'))
    if set(names)!=set(manifest['files'])|{'bundle_manifest.json'}:raise ValueError('Source inventory mismatch')
try:
    if manifest['schema']!='hope_recovery_stage1_source_v1':raise ValueError('Wrong source schema')
    if manifest.get('origin',{}).get('delivery_revision')!='lifecycle-r4':raise ValueError('Attach NEW r4 source; old timeout bundle is not compatible')
    for name,expected in manifest['files'].items():
        target=safe(REPO,name)
        if archive is None:
            original=safe(source,name)
            if original.is_symlink():raise ValueError('Source symlink')
            raw=original.read_bytes()
        else:raw=archive.read(name)
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Source checksum mismatch: '+name)
        if target.exists() and sha(target)!=expected:raise ValueError('Working source differs; restart session: '+name)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
finally:
    if archive is not None:archive.close()
(REPO/'bundle_manifest.json').write_text(json.dumps(manifest))
os.chdir(REPO);sys.path.insert(0,str(REPO))
print('[BOOTSTRAP] verified',len(manifest['files']),'source files; origin:',manifest['origin'],flush=True)
print('Phase:',PHASE,'requested controls:',JOBS,'session budget:',SESSION_HOURS,'hours',flush=True)
'''


def notebook(phase):
    title=f'''# Nested Learning for CAD -- recovery stage 1: {phase}

Worktree `hope_cad_v5`; algorithms remain **frozen v4**. No W6 refit, learner revision, full matrix, or VisA. CADIC uses an explicitly identified exact tile-min cache with reference parity.

**r4: do not attach the FAILED recovery output; start a fresh preflight with these 5 inputs.**

**Preflight inputs (5):** NEW recovery source bundle; completed W6 v4; completed W7.2 from `w7_kaggle_7_2_1`; MVTec AD; ViT-B/8 checkpoint.
**Controls/resume inputs (5):** same source/W6/MVTec/ViT; replace W7.2 input with the LATEST recovery Output/ZIP (preflight first).
You may leave reviewed W7.2 attached; recovered parent copies are verified. Detach older recovery outputs; keep only latest.

Use **GPU T4 x2**, Internet not required, **Save Version -> Save & Run All**. Runner uses one GPU.
Preflight runs regressions, normal RGB parity, and a **600-second bounded** cached K=10000 CADIC probe including full-bank reference parity, explicit cleanup and a zero worker exit. The cache is extra compute memory, disclosed separately. If it times out, preserve ZIP and stop for engineering review; do not disable the gate.
Controls run C1 -> A1 -> CADIC10k (operational order only), with checkpoints and final metrics. Normal diagnostics read M's saved states and new C1/A1 states. No full W7.3.
A paused output resumes the SAME study from saved boundaries. Completion needs artifact review. A hard platform kill can prevent final export; normal pauses/failures export before raising.
'''
    cells=[]
    for kind,source in [('markdown',title),('code',f'PHASE={phase!r}\n'+BOOT),('code',r'''
from notebook_support.recovery_stage1.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,inputs=INPUT,work=WORK,w6=W6,assets=ASSETS,output=OUTPUT,
    started=STARTED,session_hours=SESSION_HOURS,jobs=JOBS,device='cuda',
    W6_OVERRIDE=W6_OVERRIDE,W72_OVERRIDE=W72_OVERRIDE,RESUME_OVERRIDE=RESUME_OVERRIDE,
    MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''),('markdown','''## Keep the saved version output

Download `hope_cad_recovery_stage1_evidence.zip` and log, or attach this version's Output to the next run.
Check `handoff.json`: `recovery_preflight_passed`, `paused_at_atomic_boundary`, or `recovery_stage1_complete_pending_review`.
A failure also exports a ZIP before the cell raises. Full scope/instructions: `docs/main_spec/recovery_stage1_handoff.md`.
''')]:
        c=dict(cell_type=kind,id=f'{kind}-{len(cells):02d}',metadata={},source=source.strip()+'\n')
        if kind=='code':c.update(execution_count=None,outputs=[]);compile(c['source'],'recovery_notebook','exec')
        cells.append(c)
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=5)


def build():
    frozen=source_identity()['frozen']
    for phase in ('preflight','controls'):
        p=ROOT/'notebooks'/f'hope_cad_recovery_stage1_{phase}_kaggle.ipynb'
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(notebook(phase),ensure_ascii=False,indent=1)+'\n',encoding='utf8')
    paths=set()
    folders=['conf','dataset','models','training','scripts/hope_cad_v2','tests/hope_cad_v2','docs/main_spec',
             'notebook_support/w7_resume_fix','notebook_support/recovery_stage1','tests/recovery_stage1']
    for folder in folders:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json','.md') and '__pycache__' not in p.parts)
    paths.update((ROOT/'notebooks').glob('hope_cad*.ipynb'))
    paths.update(ROOT/name for name in ('README.md','pixi.toml','pixi.lock','requirements.txt','LICENSE'))
    paths.update(ROOT/n for n in frozen)
    files={p.relative_to(ROOT).as_posix():p.read_bytes() for p in paths if p.is_file()}
    for n,h in frozen.items():
        if hashlib.sha256(files[n]).hexdigest()!=h:raise ValueError('frozen dependency changed: '+n)
    def git(*args):
        r=subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,text=True)
        return r.stdout.strip() if r.returncode==0 else None
    manifest=dict(schema=SOURCE_SCHEMA,origin=dict(branch='hope_cad_v5',delivery_revision='lifecycle-r4',source_commit=git('rev-parse','HEAD'),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        identity_basis='all packaged file SHA256s; commit alone does not represent dirty checkout',
        original_frozen_source_digest='86c941b4aa1b2e82682c2d0ed31e97325b3e02442f6f88efa333511884db25ad'),
        protocol=PROTOCOL,files={n:hashlib.sha256(b).hexdigest() for n,b in sorted(files.items())})
    destination=ROOT/'ultils/recovery_stage1/hope_cad_recovery_stage1_source_r4.zip';destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):z.writestr(n,b)
        z.writestr('bundle_manifest.json',json.dumps(manifest,sort_keys=True,ensure_ascii=False))
    with zipfile.ZipFile(tmp) as z:
        for n,h in manifest['files'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('bundle integrity failure')
    tmp.replace(destination)
    report=dict(schema='recovery_delivery_v1',source_bundle=str(destination.relative_to(ROOT)),files=len(files),
        bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        frozen_dependencies_verified=len(frozen),operations_source=source_identity(),operations_source_digest=digest(source_identity()),
        Kaggle_preflight_pending=True,scientific_validation_pending=True)
    out=ROOT/'results/recovery_stage1/delivery.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='operations_source'},indent=2))
    return destination

if __name__=='__main__':build()