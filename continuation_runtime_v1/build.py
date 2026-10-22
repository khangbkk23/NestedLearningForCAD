"""Add operational files without changing any entry of the accepted r2 archive."""
import argparse,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='244e49b892fa8bc691290b7f97fbc4cfcb8ad370fc62ed312ca20c7e39b3cb59'
SOURCE_SCHEMA='hope_support_v5_1_continuation_7_1_source_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def notebook():
    intro='''# Nested Learning for CAD — continuation7.1 sau bridge384

Giữ nguyên method v5.1 và core source r2; chỉ thêm lớp vận hành khóa riêng. Không chạy lại W6/pilot/G3/bridge. Chỉ NESTED_seed0; chưa chạy CADIC hoặc matrix trong notebook này.

**Bốn inputs:** (1) source mới `hope_cad_support_v5_1_continuation_source_r1.zip` (hoặc Kaggle đã giải nén), (2) **reviewed bridge384 output** lần đầu; lần sau **latest continuation output** (tháo bridge/outputs cũ), (3) MVTec AD, (4) `vit_base_patch8_224.augreg_in21k.safetensors`. Không cần attach source bridge r2 riêng vì nó đã được giữ byte-exact trong source mới. Không attach paused376 hoặc pilot/W6.

GPU T4×2, kernel mới; Save Version → Save & Run All. SESSION_HOURS=9 là soft budget, trừ600s để export, không là ETA. Deadline chỉ kiểm ở atomic boundary; platform hard kill vẫn có thể mất output. Không tự đổi package/runtime. Có heartbeat30s và progress mỗi8ảnh; bootstrap/staging có log trước worker. Lần đầu384→3599, rồi15final evaluation cells; pause tiếp từ checkpoint mới, không train lại từ384. Không dùng notebook7.1 cũ để resume source này.

Chỉ `7.1_complete_pending_review` là chạy xong, vẫn cần review artifacts. `paused_at_atomic_boundary`: tải ZIP/log, input latest output rồi chạy notebook này tiếp. `failed_preserve_evidence`: giữ ZIP/log để review, không tự retry. Điểm AD không được dùng để sửa method/threshold hoặc chọn category. Chưa cho phép VisA.

Lưu output và Kaggle log vào `D:/src/DATN/.agents/artifacts/runs/support_v5_1_benchmark_NESTED_seed0_continuation/session_01/`; phiên tiếp theo session_02,03… File output `hope_cad_support_v5_1_continuation_evidence.zip`. Final metrics ở `benchmark/runs/NESTED_seed0/metrics/`. Đọc `docs/main_spec/continuation_7_1_v5_1.md` và implementation trước chạy.
'''
    boot='''PHASE='7.1'
JOB='NESTED_seed0'
SESSION_HOURS=9.0
SOURCE_OVERRIDE=None
PARENT_OVERRIDE=None
MVTEC_OVERRIDE=None
CHECKPOINT_OVERRIDE=None

import os,sys,time,json,hashlib,zipfile,shutil
from pathlib import Path,PurePosixPath
STARTED=time.monotonic()
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
INPUT=Path('/kaggle/input');ROOT=Path('/tmp/hope_continuation_7_1_source_r1')
WORK=Path('/kaggle/working/hope_continuation_7_1')
ASSETS=Path('/tmp/hope_continuation_assets')
OUTPUT=Path('/kaggle/working/hope_cad_support_v5_1_continuation_evidence.zip')
print('[CONTINUATION BOOTSTRAP] finding hash-bound source',flush=True)
def safe(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or p.as_posix()!=name or '..' in p.parts or '\\\\' in name or ':' in name:raise ValueError('Unsafe source path')
    dest=root/name
    if not dest.resolve().is_relative_to(root.resolve()):raise ValueError('Source path escapes target')
    return dest
def source_manifest(p):
    if p.is_dir():return json.loads((p/'continuation_manifest.json').read_text())
    with zipfile.ZipFile(p) as z:
        ns=z.namelist()
        if len(ns)!=len(set(ns)):raise ValueError('Duplicate source entries')
        m=json.loads(z.read('continuation_manifest.json'))
        if set(ns)!=set(m['files'])|{'continuation_manifest.json'}:raise ValueError('Source inventory changed')
        return m
choices=[]
paths=[Path(SOURCE_OVERRIDE)] if SOURCE_OVERRIDE else [p.parent for p in INPUT.rglob('continuation_manifest.json')]+list(INPUT.rglob('*source*.zip'))
for p in paths:
    try:
        m=source_manifest(p)
        if m.get('schema')=='hope_support_v5_1_continuation_7_1_source_v1':choices.append((p,m))
    except (KeyError,zipfile.BadZipFile):
        if SOURCE_OVERRIDE:raise
if not choices or len({json.dumps(m,sort_keys=True) for p,m in choices})!=1:raise ValueError('Expected ONE continuation source identity; set SOURCE_OVERRIDE')
src,m=sorted(choices,key=lambda x:str(x[0]))[0]
if ROOT.exists():raise ValueError('Source root already exists; start a fresh kernel/session')
ROOT.mkdir(parents=True)
archive=zipfile.ZipFile(src) if src.is_file() else None
try:
    for name,h in m['files'].items():
        original=safe(src,name) if not archive else None
        if original and original.is_symlink():raise ValueError('Source symlink')
        raw=archive.read(name) if archive else original.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Source SHA mismatch: '+name)
        dst=safe(ROOT,name);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
finally:
    if archive:archive.close()
(ROOT/'continuation_manifest.json').write_text(json.dumps(m,sort_keys=True))
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
print('[CONTINUATION BOOTSTRAP] verified',len(m['files']),'files; frozen r2 plus separate operational extension',flush=True)
from continuation_runtime_v1.common import verify_delivery
CORE_SOURCE,ORCHESTRATION=verify_delivery()
print('[CONTINUATION IDENTITY]',json.dumps(ORCHESTRATION),flush=True)
'''
    run='''from continuation_runtime_v1.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,job=JOB,session_hours=SESSION_HOURS,started=STARTED,
    inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),
    PARENT_OVERRIDE=PARENT_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''
    cells=[]
    for i,(kind,text) in enumerate([('markdown',intro),('code',boot),('code',run)]):
        cell=dict(cell_type=kind,id='continuation-'+str(i),metadata={},source=text.splitlines(keepends=True))
        if kind=='code':compile(text,'continuation_cell','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    value=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    p=ROOT/'notebooks/hope_cad_v5_1_continuation_7_1_kaggle.ipynb';p.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return p

def build(output):
    nb=notebook();shared=ROOT.parent/'.agents'
    base=shared/'artifacts/deliveries/hope_support_v5_1_nn_write_bridge_r2/hope_cad_support_v5_1_nn_write_bridge_source_r2.zip'
    if sha(base)!=BASE_SHA:raise ValueError('Wrong accepted r2 source ZIP')
    with zipfile.ZipFile(base) as z:payload={n:z.read(n) for n in z.namelist()}
    assert len(payload)==222
    review=shared/'evidence/reviews/support-v5-1-nn-write-bridge-session1-20261021'
    receipts=ROOT/'continuation_runtime_v1/receipts';receipts.mkdir(exist_ok=True)
    for old,new in [('review.json','G4_review.json'),('audit.json','G4_audit.json')]:
        raw=(review/old).read_bytes();p=receipts/new
        if p.exists() and p.read_bytes()!=raw:raise ValueError('Existing G4 receipt differs')
        p.write_bytes(raw)
    parent=shared/'artifacts/runs/support_v5_1_nn_write_bridge_01/session_01/hope_cad_support_v5_1_nn_write_bridge_evidence.zip'
    with zipfile.ZipFile(parent) as z:raw=z.read('artifact_manifest.json')
    if hashlib.sha256(raw).hexdigest()!='05350f6f35cc2cbfa4e79f3fd906889229898f9d451e792564621569fa7a0993':raise ValueError('Wrong reviewed bridge manifest')
    p=receipts/'G4_manifest.json'
    if p.exists() and p.read_bytes()!=raw:raise ValueError('G4 manifest changed')
    p.write_bytes(raw)
    added=[p for p in (ROOT/'continuation_runtime_v1').rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    added += [p for p in (ROOT/'tests/continuation_v5_1').rglob('*.py')]
    added += [nb,ROOT/'docs/main_spec/continuation_7_1_v5_1.md',ROOT/'docs/main_spec/continuation_7_1_v5_1_implementation.md']
    for p in added:
        n=p.relative_to(ROOT).as_posix()
        if n in payload:raise ValueError('Operational extension replaces core')
        payload[n]=p.read_bytes()
    files={n:hashlib.sha256(b).hexdigest() for n,b in sorted(payload.items())}
    manifest=dict(schema=SOURCE_SCHEMA,files=files,core_archive_sha256=BASE_SHA,core_entries=222,method_changed=False)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(payload.items()):z.writestr(n,b)
        z.writestr('continuation_manifest.json',json.dumps(manifest,sort_keys=True))
    receipt=dict(path=str(output),sha256=sha(output),files=len(files),core_byte_exact_entries=222,method_changed=False,canonical_continuation='not_run')
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
