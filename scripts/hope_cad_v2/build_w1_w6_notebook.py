"""Generate the thin Kaggle bootstrap; learning and acceptance live in versioned CLIs."""
import json
from pathlib import Path
from textwrap import dedent
ROOT=Path(__file__).resolve().parents[2]


def build(preflight_only=False):
    cells=[]
    def cell(kind,text):
        result=dict(cell_type=kind,id=f'hope-v3-{len(cells):02d}',metadata={},source=dedent(text).strip().splitlines(keepends=True))
        if kind=='code': result.update(execution_count=None,outputs=[])
        cells.append(result)
    cell('markdown','''
    # Nested Learning for CAD — W1–W6, spec v3

    Attach the **new** `hope_cad_v2_2_source_bundle`, MVTec AD and the agreed ViT-B/8 ImageNet-21k checkpoint.
    The old `mvtec_prefix_features_final.pt` input is optional: incompatible caches are rejected and rebuilt.
    Enable Internet for the pinned Python packages and select a T4 GPU. The stream uses one GPU.

    This notebook verifies W1–W5, performs prefix-only selection and normal-probe diagnostics for W6.
    It never runs W7 or uses anomaly-test metrics. Selection failure stops later scientific stages and preserves evidence.
    Success means **ready for artifact review**, not a claim of CAD performance.
    The 10.5-hour budget starts at the first code cell, including bootstrap; completion exports immediately.
    A planned pause leaves time before Kaggle's 12-hour GPU limit for export and saving outputs.

    Start with **Save Version → Save & Run All**, with output saving enabled, to run in the background.
    Inspect the saved version's Output after completion; the final ZIP must be present there.
    For another session, attach the previous notebook's saved output or evidence ZIP as an input. The same source,
    runtime and methods are required; completed fits and partial fit/gate checkpoints resume automatically.
    If the session is terminated before export, save/download the working output directory, including current checkpoints.
    ''')
    if preflight_only:
        cells[0]['source']=dedent('''
        # Nested Learning for CAD — Kaggle preflight, spec v3

        Attach the new source bundle, MVTec AD and the agreed ViT checkpoint. Select T4 and enable Internet.
        Start with **Save Version → Save & Run All**, with output saving enabled.

        This short notebook checks the actual setup, correctness suite, canonical exact-16 resource profile,
        frozen-feature cache, real RGB parity and checkpoint/resume, plus two real-prefix exact-16 Adam steps
        and a disk checkpoint/optimizer/RNG resume comparison. Smoke weights never enter scientific selection.
        It stops before the full 200-step calibration/search/probe. The session budget includes bootstrap time.
        Allow roughly 15–30 minutes, or 30–45 with data/cache preparation; these are estimates, not measured limits.

        After **preflight_passed**, open the saved version's Output and verify that the evidence ZIP is present.
        Attach this preflight output to the main notebook using the same source/runtime; it resumes after these gates.
        A planned session pause means incomplete preflight and requires resume, not permission to start W6.
        ''').strip().splitlines(keepends=True)
    cell('code',r'''
    from pathlib import Path
    import hashlib, json, os, shutil, subprocess, sys, tarfile, zipfile, time
    if 'NOTEBOOK_STARTED_MONOTONIC' not in globals():
        NOTEBOOK_STARTED_MONOTONIC = time.monotonic()
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    INPUT = Path('/kaggle/input')
    SOURCE_OVERRIDE = None      # Optional: extracted source directory or source ZIP.
    MVTEC_OVERRIDE = None       # Optional: dataset root containing bottle/train/good.
    CHECKPOINT_OVERRIDE = None  # Optional: agreed .safetensors file.
    RESUME_OVERRIDE = None      # Optional: evidence ZIP or extracted evidence directory.
    METHODS = ['M', 'C0', 'C1', 'C2', 'C3', 'C4', 'A1']
    SESSION_HOURS = 10.5        # From the first code cell, including bootstrap; reserve time for export.
    REPO = Path('/tmp/hope_cad_source_v3')
    WORK = Path('/kaggle/working/hope_w1_w6')
    ASSETS = Path('/tmp/hope_assets')
    EXPECTED_BACKBONE = '2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec'
    def sha(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    def unique(items, label):
        items = sorted(set(Path(x) for x in items))
        if len(items) != 1: raise RuntimeError(f'{label}: expected one match, found {items}; set its override.')
        return items[0]
    def safe_relative(root, name):
        path = (root / name).resolve()
        if Path(name).is_absolute() or chr(92) in name or not path.is_relative_to(root.resolve()):
            raise RuntimeError(f'Unsafe archive path: {name}')
        return path
    def import_manifest_tree(source, destination, manifest_name):
        if source.is_file():
            with zipfile.ZipFile(source) as archive:
                manifest = json.loads(archive.read(manifest_name))
                for name, expected in manifest['files'].items():
                    data = archive.read(name)
                    if hashlib.sha256(data).hexdigest() != expected: raise RuntimeError(f'Corrupt archive entry: {name}')
                    target = safe_relative(destination,name); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
                (destination/manifest_name).write_bytes(archive.read(manifest_name))
        else:
            manifest = json.loads((source/manifest_name).read_text())
            for name, expected in manifest['files'].items():
                original = safe_relative(source,name)
                if sha(original) != expected: raise RuntimeError(f'Corrupt input: {name}')
                target = safe_relative(destination,name); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(original,target)
            shutil.copyfile(source/manifest_name,destination/manifest_name)
        return manifest
    source = Path(SOURCE_OVERRIDE) if SOURCE_OVERRIDE else unique(
        [p.parent for p in INPUT.rglob('bundle_manifest.json')] or list(INPUT.rglob('hope_cad_v2_2_source_bundle.zip')), 'Source bundle')
    if REPO.exists() and any(REPO.iterdir()):
        existing = json.loads((REPO/'bundle_manifest.json').read_text())
        for name, expected in existing['files'].items():
            if sha(REPO/name) != expected: raise RuntimeError('Working source changed; start a clean session with the intended bundle.')
        incoming = json.loads((source/'bundle_manifest.json').read_text()) if source.is_dir() else json.loads(zipfile.ZipFile(source).read('bundle_manifest.json'))
        if incoming['files'] != existing['files']: raise RuntimeError('Source changed; use a clean source directory/session.')
        manifest = existing
    else:
        manifest = import_manifest_tree(source,REPO,'bundle_manifest.json')
    if 'scripts/hope_cad_v2/run_w1_w6.py' not in manifest['files']:
        raise RuntimeError('This is the old source bundle. Attach the new W1–W6 bundle.')
    os.chdir(REPO); sys.path.insert(0,str(REPO))
    print('Verified source files:',len(manifest['files']))
    print('Source origin:',manifest['origin'])
    ''')
    cells[-1]['source'].insert(0, f'PREFLIGHT_ONLY = {preflight_only!r}\n')
    cell('code',r'''
    import importlib.metadata
    pins = {'timm':'1.0.29', 'safetensors':'0.8.0'}
    for name, expected in pins.items():
        try: actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: actual = None
        if actual != expected:
            subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-deps',f'{name}=={expected}'])
        assert importlib.metadata.version(name) == expected
    import torch
    from training.hope_runtime_v3 import configure_runtime, runtime_identity
    configure_runtime()
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 12*1024**3:
        raise RuntimeError('Select a CUDA GPU with at least 12 GiB; canonical calibration is not reduced to fit smaller devices.')
    print(json.dumps(runtime_identity(),indent=2))
    ''')
    cell('code',r'''
    checkpoint = Path(CHECKPOINT_OVERRIDE) if CHECKPOINT_OVERRIDE else unique(INPUT.rglob('*.safetensors'),'ViT checkpoint')
    if sha(checkpoint) != EXPECTED_BACKBONE: raise RuntimeError('Wrong frozen backbone SHA256.')
    roots = [p.parent for p in INPUT.rglob('bottle') if (p/'train/good').is_dir()]
    roots += [p.parent for p in ASSETS.rglob('bottle') if (p/'train/good').is_dir()] if ASSETS.exists() else []
    if MVTEC_OVERRIDE:
        dataset = Path(MVTEC_OVERRIDE)
    elif roots:
        dataset = unique(roots,'MVTec root')
    else:
        archive = unique(list(INPUT.rglob('*mvtec*.tar.xz')) + list(INPUT.rglob('*mvtec*.tar.gz')),'MVTec archive')
        ASSETS.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive) as stream:
            for entry in stream.getmembers():
                safe_relative(ASSETS,entry.name)
                if entry.issym() or entry.islnk() or not (entry.isfile() or entry.isdir()):
                    raise RuntimeError('Links/devices are not accepted in the data archive.')
            stream.extractall(ASSETS,filter='data')
        dataset = unique([p.parent for p in ASSETS.rglob('bottle') if (p/'train/good').is_dir()],'Extracted MVTec root')
    if not (dataset/'bottle/train/good').is_dir(): raise RuntimeError('MVTec root does not contain bottle/train/good.')
    os.environ.update(MVTEC_ROOT=str(dataset),VIT_B8_CHECKPOINT=str(checkpoint),VIT_B8_SHA256=EXPECTED_BACKBONE)
    optional_cache = list(INPUT.rglob('mvtec_prefix_features_final.pt'))
    FEATURE_CACHE = unique(optional_cache,'Optional legacy cache') if optional_cache else None
    print('MVTec:',dataset,'\nBackbone:',checkpoint,'\nOptional cache:',FEATURE_CACHE)
    ''')
    cell('code',r'''
    resume_candidates = [p.parent for p in INPUT.rglob('artifact_manifest.json')
                         if json.loads(p.read_text()).get('schema') == 'hope_w1_w6_evidence_v3']
    resume_candidates = resume_candidates or list(INPUT.rglob('hope_cad_v2_2_w1_w6_evidence.zip'))
    resume_source = Path(RESUME_OVERRIDE) if RESUME_OVERRIDE else (unique(resume_candidates,'Resume evidence') if resume_candidates else None)
    if resume_source and not WORK.exists():
        recovered = import_manifest_tree(resume_source,WORK,'artifact_manifest.json')
        if recovered.get('schema') != 'hope_w1_w6_evidence_v3': raise RuntimeError('Wrong resume artifact schema.')
        print('Restored',len(recovered['files']),'verified files')
    RESUME = (WORK/'pipeline_lock.json').is_file()
    print('Resume:',RESUME,'\nEvidence directory:',WORK)
    ''')
    cell('code',r'''
    command = [sys.executable,'-B','scripts/hope_cad_v2/run_w1_w6.py',
               '--work-dir',str(WORK),'--device','cuda','--session-hours',str(SESSION_HOURS),
               '--session-start-monotonic',str(NOTEBOOK_STARTED_MONOTONIC),'--methods',*METHODS]
    if FEATURE_CACHE: command += ['--feature-cache',str(FEATURE_CACHE)]
    if RESUME: command += ['--resume']
    if PREFLIGHT_ONLY: command += ['--preflight-only']
    print(' '.join(command),flush=True)
    EXIT_CODE = subprocess.call(command,cwd=REPO)
    print('Pipeline exit code:',EXIT_CODE)
    # Do not abort this cell: the next cell exposes evidence even after a failed gate.
    ''')
    cell('code',r'''
    from IPython.display import display, FileLink
    # Refresh the archive after a pause too; never serve an older mid-run export.
    from scripts.hope_cad_v2.run_w1_w6 import export_evidence, notebook_outcome
    evidence = export_evidence(WORK)
    summary = json.loads((WORK/'handoff.json').read_text()) if (WORK/'handoff.json').exists() else {'status':'interrupted_before_summary'}
    print(json.dumps(summary,indent=2))
    print('Evidence bytes:',evidence.stat().st_size,'\nSHA256:',sha(evidence))
    display(FileLink(str(evidence)))
    print('For Save & Run All, inspect this saved version\'s Output after completion. W7 requires artifact review.')
    print(notebook_outcome(EXIT_CODE, summary, preflight_only=PREFLIGHT_ONLY))
    ''')
    target=ROOT/'notebooks'/('hope_cad_v2_2_preflight_kaggle.ipynb' if preflight_only else 'hope_cad_v2_2_kaggle.ipynb')
    target.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for index, entry in enumerate(cells):
        if entry['cell_type']=='code': compile(''.join(entry['source']),f'cell_{index}','exec')
    print(target)


if __name__=='__main__':
    build(preflight_only=True)
    build()
