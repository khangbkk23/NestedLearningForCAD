"""Build thin Kaggle handoffs for normal-only preflight and the three W7 parts."""
from pathlib import Path
import json
import textwrap
ROOT=Path(__file__).resolve().parents[2]

BOOTSTRAP=r'''
import os, sys, time, json, hashlib, zipfile, tarfile, shutil, subprocess
from pathlib import Path
if 'STARTED' not in globals():STARTED=time.monotonic()
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
INPUT=Path('/kaggle/input')
SOURCE_OVERRIDE=None  # Source ZIP or Kaggle-extracted source directory.
W6_OVERRIDE=None      # Reviewed completed W6 v4 ZIP (NOT a preflight/paused W6).
RESUME_OVERRIDE=None  # W7 preflight or latest W7 output for this job.
C1_EVIDENCE_OVERRIDE=None  # A1 seed1/2: C1 output of that same seed.
MVTEC_OVERRIDE=None
CHECKPOINT_OVERRIDE=None
SESSION_HOURS=10.5
REPO=Path('/tmp/hope_w7_source')
W6=Path('/tmp/hope_w7_parent_w6')
WORK=Path('/kaggle/working/hope_w7')
ASSETS=Path('/tmp/hope_w7_assets')
EXPECTED_BACKBONE='2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def unique(values,label):
    values=sorted(set(Path(p) for p in values))
    if len(values)!=1:raise RuntimeError(f'{label}: expected ONE input, got {values}; set its override')
    return values[0]
def safe(root,name):
    path=(root/name).resolve()
    if Path(name).is_absolute() or chr(92) in name or ':' in name or not path.is_relative_to(root.resolve()):
        raise RuntimeError('Unsafe archive path')
    return path
sources=[p.parent for p in INPUT.rglob('bundle_manifest.json')
         if json.loads(p.read_text()).get('schema')=='hope_w7_source_v1']
sources=sources or list(INPUT.rglob('hope_cad_w7_source_bundle.zip'))
source=Path(SOURCE_OVERRIDE) if SOURCE_OVERRIDE else unique(sources,'W7 source')
if source.is_dir():
    manifest=json.loads((source/'bundle_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        original=source/name
        if sha(original)!=expected:raise RuntimeError('Source checksum mismatch: '+name)
        path=safe(REPO,name);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and sha(path)!=expected:raise RuntimeError('Working source changed: '+name)
        shutil.copyfile(original,path)
else:
    with zipfile.ZipFile(source) as archive:
        manifest=json.loads(archive.read('bundle_manifest.json'))
        if len(archive.namelist())!=len(set(archive.namelist())):raise RuntimeError('Duplicate source entries')
        for name,expected in manifest['files'].items():
            raw=archive.read(name)
            if hashlib.sha256(raw).hexdigest()!=expected:raise RuntimeError('Source checksum mismatch: '+name)
            path=safe(REPO,name);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and sha(path)!=expected:raise RuntimeError('Working source changed: '+name)
            path.write_bytes(raw)
if manifest['schema']!='hope_w7_source_v1':raise RuntimeError('Wrong source bundle')
(REPO/'bundle_manifest.json').write_text(json.dumps(manifest))
os.chdir(REPO);sys.path.insert(0,str(REPO))
print('Verified source files:',len(manifest['files']),'origin:',manifest['origin'])
# Keep the W6-tested runtime. Fail before a long run if Kaggle's image has changed.
from training.hope_runtime_v4 import configure_runtime,runtime_identity
configure_runtime();print(json.dumps(runtime_identity(),indent=2))
w6s=list(INPUT.rglob('hope_cad_v4_normal_evidence.zip'))
if not w6s:
    w6s=[p.parent for p in INPUT.rglob('artifact_manifest.json')
         if json.loads(p.read_text()).get('schema')=='hope_normal_evidence_v4']
w6source=Path(W6_OVERRIDE) if W6_OVERRIDE else unique(w6s,'Completed reviewed W6')
from training.hope_w7_io import import_w6,restore_archive
import_w6(w6source,W6,include_features=PHASE=='7.3' and not JOB.startswith('CADIC') and
          not JOB.endswith('seed0') and '_seed0_' not in JOB)
parent_runtime=json.loads((W6/'pipeline_lock.json').read_text())['runtime']
if runtime_identity()!=parent_runtime:raise RuntimeError('Runtime differs from W6; stop and request parity review')
checkpoints=list(INPUT.rglob('vit_base_patch8_224.augreg_in21k.safetensors'))
checkpoint=Path(CHECKPOINT_OVERRIDE) if CHECKPOINT_OVERRIDE else unique(checkpoints,'ViT checkpoint')
if sha(checkpoint)!=EXPECTED_BACKBONE:raise RuntimeError('Wrong ViT checkpoint bytes')
roots=[p.parent for p in INPUT.rglob('bottle') if (p/'train/good').is_dir()]
if MVTEC_OVERRIDE:dataset=Path(MVTEC_OVERRIDE)
elif roots:dataset=unique(roots,'MVTec root')
else:
    archives=list(INPUT.rglob('*mvtec*.tar.xz'))+list(INPUT.rglob('*mvtec*.tar.gz'))
    archive=unique(archives,'MVTec archive');ASSETS.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            safe(ASSETS,member.name)
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise RuntimeError('Unsafe dataset tar member')
        stream.extractall(ASSETS,filter='data')
    dataset=unique([p.parent for p in ASSETS.rglob('bottle') if (p/'train/good').is_dir()],'Extracted MVTec')
os.environ.update(MVTEC_ROOT=str(dataset),VIT_B8_CHECKPOINT=str(checkpoint),VIT_B8_SHA256=EXPECTED_BACKBONE)
previous=list(INPUT.rglob('hope_cad_w7_evidence.zip'))
if not previous:
    previous=[p.parent for p in INPUT.rglob('artifact_manifest.json')
              if json.loads(p.read_text()).get('schema')=='hope_w7_evidence_v1']
previous=[p for p in previous if str(p)!=str(C1_EVIDENCE_OVERRIDE)]
resume=Path(RESUME_OVERRIDE) if RESUME_OVERRIDE else (unique(previous,'Previous W7 output') if previous else None)
if resume:
    # Carry shared locks/preflight, plus ONLY this job. A new 7.3 job starts fresh.
    shared={'suite_lock.json','preflight.json','verification_w7.json','handoff.json'}
    restore_archive(resume,WORK,schema='hope_w7_evidence_v1',
                    select=lambda n:n in shared or n.startswith('runs/'+JOB+'/'))
print('Phase:',PHASE,'job:',JOB,'resume:',resume,'W6:',w6source)
'''


def notebook(phase,job):
    cells=[]
    def cell(kind,source):
        value=dict(cell_type=kind,id=f'{kind}-{len(cells):02d}',metadata={},source=textwrap.dedent(source).strip()+'\n')
        if kind=='code':value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell('markdown',f'''# Nested Learning for CAD — W7 {phase}

    Frozen v4 learner. Read `docs/main_spec/w7_implementation.md` first.
    Inputs: NEW W7 source, completed reviewed W6 v4, MVTec AD, agreed ViT-B/8.
    After preflight, also attach the previous W7 saved output (or ZIP).
    Use **Save Version → Save & Run All**. Download/preserve evidence on PASS, pause or failure.
    `7.1` and `7.2` always stop for review; `7.3` runs ONE registered job per invocation.
    A1 seed1/2 additionally needs the accepted C1 W7 output of the same seed.
    ''')
    cell('code',f"PHASE={phase!r}\nJOB={job!r}\n"+BOOTSTRAP)
    cell('code',r'''
    command=[sys.executable,'-B','scripts/hope_cad_v2/run_w7.py','--phase',PHASE,'--job',JOB,
             '--w6-dir',str(W6),'--work-dir',str(WORK),'--device','cuda',
             '--session-hours',str(SESSION_HOURS),'--session-start-monotonic',str(STARTED),'--skip-export']
    if C1_EVIDENCE_OVERRIDE:command+=['--c1-evidence',str(C1_EVIDENCE_OVERRIDE)]
    print(' '.join(command),flush=True)
    logdir=WORK/'runs'/JOB/'logs';logdir.mkdir(parents=True,exist_ok=True)
    with (logdir/f'{PHASE}_{int(time.time())}.log').open('w') as log:
        process=subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        EXIT_CODE=process.wait()
    print('Runner exit code:',EXIT_CODE)
    # Preserve failure/partial artifacts too; do not raise before the export cell.
    ''')
    cell('code',r'''
    from training.hope_w7_io import export_archive
    from IPython.display import display,FileLink
    result=export_archive(WORK,Path('/kaggle/working/hope_cad_w7_evidence.zip'))
    print(json.dumps(result,indent=2))
    if (WORK/'handoff.json').exists():print((WORK/'handoff.json').read_text())
    display(FileLink(result['path']))
    print('Check the saved version Output. Completion needs artifact review; a pause resumes the SAME job.')
    ''')
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=5)


def main():
    for phase,job,name in [('preflight','M_seed0','preflight'),('7.1','M_seed0','7_1'),
                           ('7.2','M_seed0','7_2'),('7.3','M_seed1','7_3')]:
        path=ROOT/'notebooks'/f'hope_cad_w7_{name}_kaggle.ipynb'
        path.write_text(json.dumps(notebook(phase,job),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(path)
    report=notebook('7.3','M_seed1')
    report['cells'][0]['source']='# Nested Learning for CAD — W7 CPU report\n\nAccelerator: None. Inputs: W7 source and all saved W7 evidence ZIPs. No dataset/ViT/W6 input needed.\n'
    report['cells'][1]['source']=BOOTSTRAP.split('# Keep the W6-tested runtime.')[0]+r'''
EVIDENCE=sorted(INPUT.rglob('hope_cad_w7_evidence.zip'))
REPORT=Path('/kaggle/working/hope_w7_report')
INVENTORY_ONLY=True  # First inspect missing jobs; set False for complete paired uncertainty.
EXACT_PIXELS=False   # Only histogram bootstrap is approximate, with strict exact validation.
from training.hope_w7_io import restore_archive
old=list(INPUT.rglob('hope_cad_w7_analysis.zip'))
if old:restore_archive(unique(old,'Previous CPU analysis'),REPORT,schema='hope_w7_analysis_v1')
'''
    report['cells'][2]['source']=r'''
command=[sys.executable,'-B','scripts/hope_cad_v2/report_w7.py','--evidence',*map(str,EVIDENCE),
         '--output-dir',str(REPORT),'--session-hours',str(SESSION_HOURS)]
if INVENTORY_ONLY:command+=['--inventory-only']
if EXACT_PIXELS:command+=['--exact-pixels']
print(' '.join(command));EXIT_CODE=subprocess.call(command,cwd=REPO)
print('Report exit code:',EXIT_CODE)
'''
    report['cells'][3]['source']=r'''
from training.hope_w7_io import export_archive
from IPython.display import display,FileLink
result=export_archive(REPORT,Path('/kaggle/working/hope_cad_w7_analysis.zip'),schema='hope_w7_analysis_v1')
print(result)
if (REPORT/'handoff.json').exists():print((REPORT/'handoff.json').read_text())
display(FileLink(result['path']))
'''
    path=ROOT/'notebooks/hope_cad_w7_report_cpu.ipynb'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(path)

if __name__=='__main__':main()
