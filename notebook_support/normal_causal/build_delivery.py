"""Build a new, self-contained diagnostic source without touching any r4 file."""
from pathlib import Path
import hashlib,json,zipfile,datetime,ast
from .contract import ROOT,SOURCE_SCHEMA,R4_ZIP_SHA,sha,PROTOCOL,digest

BOOT = r"""PHASE='run'  # 'run': preflight tích hợp + gói hữu hạn; 'preflight': chỉ kiểm trước
SESSION_HOURS=3.0  # soft stop; lưu/resume ở atomic boundary; không phải ETA
SOURCE_OVERRIDE=None
RECOVERY_OVERRIDE=None  # completed recovery main, KHÔNG dùng preflight recovery
W6_OVERRIDE=None
RESUME_OVERRIDE=None   # chỉ latest normal_causal output; lần đầu để None
MVTEC_OVERRIDE=None
CHECKPOINT_OVERRIDE=None

from pathlib import Path,PurePosixPath
import hashlib,json,zipfile,os,sys,time,shutil
STARTED=time.monotonic()
INPUT=Path('/kaggle/input');ROOT=Path('/tmp/hope_normal_causal_source_v1')
print('[BOOTSTRAP] Tìm source chẩn đoán',flush=True)
def hash_file(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def source_manifest(p):
    if p.is_dir():return json.loads((p/'bundle_manifest.json').read_text())
    with zipfile.ZipFile(p) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate source ZIP paths')
        m=json.loads(z.read('bundle_manifest.json'))
        if set(z.namelist())!=set(m['files'])|{'bundle_manifest.json'}:raise ValueError('Source inventory mismatch')
        return m
def target(name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or rel.as_posix()!=name or '\\' in name or ':' in name:raise ValueError('Unsafe source path')
    p=ROOT/name
    if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Escaping source path')
    return p
if SOURCE_OVERRIDE:choices=[Path(SOURCE_OVERRIDE)]
else:
    choices=[]
    for p in INPUT.rglob('bundle_manifest.json'):
        if json.loads(p.read_text()).get('schema')=='hope_normal_causal_source_v1':choices.append(p.parent)
    if not choices:
        for p in INPUT.rglob('*.zip'):
            try:
                if source_manifest(p).get('schema')=='hope_normal_causal_source_v1':choices.append(p)
            except (KeyError,ValueError,zipfile.BadZipFile):pass
if len(choices)!=1:raise RuntimeError(f'Expected ONE new diagnostic source, found {choices}; set SOURCE_OVERRIDE')
source=choices[0];manifest=source_manifest(source)
if manifest['schema']!='hope_normal_causal_source_v1':raise ValueError('Wrong source schema')
ROOT.mkdir(parents=True,exist_ok=True)
if source.is_dir():
    for name,expected in manifest['files'].items():
        original=source/name;dest=target(name)
        if not original.is_file() or original.is_symlink() or hash_file(original)!=expected:raise ValueError('Source checksum mismatch: '+name)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
else:
    with zipfile.ZipFile(source) as z:
        for name,expected in manifest['files'].items():
            dest=target(name);dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as reader,dest.open('wb') as writer:shutil.copyfileobj(reader,writer)
            if hash_file(dest)!=expected:raise ValueError('Source checksum mismatch: '+name)
(ROOT/'bundle_manifest.json').write_text(json.dumps(manifest,sort_keys=True))
os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
print('[BOOTSTRAP] Verified source files:',len(manifest['files']),manifest['origin'],flush=True)
WORK=Path('/kaggle/working/hope_normal_causal')
W6=Path('/tmp/hope_normal_causal_w6')
ASSETS=Path('/tmp/hope_normal_causal_assets')
OUTPUT=Path('/kaggle/working/hope_cad_normal_causal_evidence.zip')
"""
RUN = """from notebook_support.normal_causal.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,session_hours=SESSION_HOURS,started=STARTED,
    inputs=str(INPUT),work=str(WORK),w6=str(W6),assets=str(ASSETS),output=str(OUTPUT),
    RECOVERY_OVERRIDE=RECOVERY_OVERRIDE,W6_OVERRIDE=W6_OVERRIDE,RESUME_OVERRIDE=RESUME_OVERRIDE,
    MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
"""
INTRO="""# Nested Learning for CAD — chẩn đoán normal-only có budget hữu hạn

**Đây chưa phải method revision và không chạy lại W6/full W7.** Đọc `docs/main_spec/normal_causal_diagnostic_v1.md`.

Attach đúng 5 inputs:
1. Source mới `hope_cad_normal_causal_source_v1` (ZIP hoặc Kaggle tự giải nén).
2. **Completed recovery main** `hope_cad_recovery_stage1_evidence.zip` đã nghiệm thu; không dùng recovery preflight.
3. Completed W6 v4 `hope_cad_v4_normal_evidence.zip` (có original prefix cache và M fit-resume 400).
4. MVTec AD.
5. `vit_base_patch8_224.augreg_in21k.safetensors`.

Bật **GPU T4 x2**, dùng **Save Version → Save & Run All**. Giữ `PHASE='run'` cho một lượt có preflight tích hợp.
Không attach source recovery cũ hoặc W7 preflight làm resume. Hai GPU không cộng VRAM, worker dùng một T4.
Nếu cần kiểm riêng, đặt `PHASE='preflight'`; sau PASS attach output đó và chạy `PHASE='run'`.
Khi pause, attach **latest diagnostic output** làm input thứ sáu; tháo bản diagnostic cũ hơn. 5 parent inputs giữ nguyên.

Notebook luôn cố xuất ZIP khi thành công/pause/lỗi; hard kill của nền tảng có thể ngăn export.
Chỉ `complete_pending_review` là xong phần chạy, vẫn cần kiểm artifacts; loss không cải thiện vẫn là kết quả hợp lệ.
Không có benchmark anomaly/VisA, không xuất learned weights để triển khai.
"""

def build():
    notebook=ROOT/'notebooks/hope_cad_normal_causal_kaggle.ipynb'
    cells=[]
    for i,(kind,text) in enumerate([('markdown',INTRO),('code',BOOT),('code',RUN)]):
        cell=dict(cell_type=kind,id=f'normal-causal-v1-{i}',metadata={},source=text.splitlines(True))
        if kind=='code':ast.parse(text);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    notebook.parent.mkdir(parents=True,exist_ok=True)
    notebook.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    base=ROOT/'ultils/recovery_stage1/hope_cad_recovery_stage1_source_r4.zip'
    if sha(base)!=R4_ZIP_SHA:raise ValueError('immutable r4 template checksum failed')
    added=list((ROOT/'notebook_support/normal_causal').glob('*.py'))+list((ROOT/'tests/normal_causal').glob('*.py'))+list((ROOT/'docs/main_spec').glob('normal_causal_diagnostic_v1.*'))+[notebook]
    output=ROOT/'ultils/normal_causal/hope_cad_normal_causal_source_v1.zip';output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(base) as z:
        prior=json.loads(z.read('bundle_manifest.json'));files=dict(prior['files'])
        if any(sha(ROOT/n)!=h for n,h in files.items()):raise ValueError('live frozen r4 template files changed')
        if any(p.relative_to(ROOT).as_posix() in files for p in added):raise ValueError('new namespace clobbers frozen dependency')
        files.update({p.relative_to(ROOT).as_posix():sha(p) for p in added})
        manifest=dict(schema=SOURCE_SCHEMA,origin=dict(branch='hope_cad_v5',source_commit='4dd6f36fe093b2ae153d196a9333aeda1d196a06',delivery_revision='normal-causal-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity_basis='all packaged SHA256s; no commit-only claim',frozen_base_zip_sha256=R4_ZIP_SHA),files=dict(sorted(files.items())))
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
            for name in prior['files']:out.writestr(name,z.read(name))
            for path in added:out.write(path,path.relative_to(ROOT).as_posix())
            out.writestr('bundle_manifest.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(output) as z:
        for name,expected in files.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('built bundle checksum failed')
    receipt=dict(schema='normal_causal_delivery_receipt_v1',source_zip=str(output),sha256=sha(output),bytes=output.stat().st_size,files=len(files),source_digest=digest(files),protocol_digest=digest(PROTOCOL),frozen_r4_files_unchanged=len(prior['files']),gpu_experiment_pending=True)
    (output.parent/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':build()
