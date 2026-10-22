"""Build additive profiler delivery from the immutable166 benchmark ZIP, never live substitutions."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
from training.support_throughput_v5_1.contract import ROOT,contract,sha,digest,SOURCE_SCHEMA,safe

NOTEBOOK='notebooks/hope_cad_v5_1_throughput_profile_kaggle.ipynb'
DELIVERY='hope_support_v5_1_throughput_profile_r1'
ZIP_NAME='hope_cad_support_v5_1_throughput_source_r1.zip'

def write_notebook():
    pin=sha(ROOT/'docs/main_spec/throughput_v5_1_contract.json')
    intro='''# Nested Learning for CAD — profiling throughput v5.1

Chỉ đo trên clone bị discard; giữ nguyên toàn bộ method. Không tiếp tục benchmark hoặc tối ưu model trong notebook này.

**4 inputs bắt buộc:** source `hope_cad_support_v5_1_throughput_source_r1.zip` (hoặc Dataset đã giải nén), reviewed **NESTED_seed0 session01 ở376/3599** evidence ZIP/output, MVTec AD, ViT-B/8 `vit_base_patch8_224.augreg_in21k.safetensors`. Không attach W6/pilot/preflight output thay parent376. Các output khác schema benchmark nên tháo nếu không dùng; parent luôn được chọn bằng manifest hash chính xác.

Chọn **GPU T4×2**, Internet không bắt buộc khi packages đã đúng. Dùng notebook mới trong kernel mới. **Save Version → Save & Run All**. Không pip upgrade để vượt runtime lock. Không đổi ngân sách/window/method. Code chỉ dùng cuda:0.

U1→P1→U2, mỗi pass fresh process từ task00cursor179,16normals179–194 đã được thấy. Tổng tối đa48attempts, soft90phút cumulative (gồm bootstrap/audit);180s dành export. Đây là budget, không phải ETA. Heartbeat30s; không finalize ở195. Chỉ resume giữa pass complete; partial pass không tự chạy lại. Nếu `profile_incomplete_budget`/failure hãy giữ evidence và yêu cầu review, không tăng sốảnh hoặc đổi source tùy ý.

Chỉ khi được hướng dẫn resume giữa pass: attach **latest profiler output** cùng4inputs. Tháo profiler output cũ hơn. Explicit overrides cho ZIP hoặc thư mục evidence/source khi cần.

Lưu ZIP `hope_cad_support_v5_1_throughput_evidence.zip` và log vào `.agents/artifacts/runs/support_v5_1_throughput_profile_01/session_01/`; phiên tiếp tăngsession. `profile_complete_pending_review` cần review transparency/timing; không là speedup hoặc ADgain. Đọc `docs/main_spec/throughput_v5_1_implementation.md`.
'''
    boot=r'''PHASE='profile'
SOURCE_OVERRIDE=None
PARENT_OVERRIDE=None  # exact reviewed benchmark376 ZIP or extracted evidence root
RESUME_OVERRIDE=None  # latest profiler output only; first run None
MVTEC_OVERRIDE=None
CHECKPOINT_OVERRIDE=None

from pathlib import Path,PurePosixPath
import hashlib,json,zipfile,time,os,sys,shutil
STARTED=time.monotonic()
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.dont_write_bytecode=True
INPUT=Path('/kaggle/input');ROOT=Path('/tmp/hope_throughput_source_r1')
WORK=Path('/kaggle/working/hope_throughput_v5_1')
ASSETS=Path('/tmp/hope_throughput_assets')
OUTPUT=Path('/kaggle/working/hope_cad_support_v5_1_throughput_evidence.zip')
PINNED_CONTRACT='__CONTRACT_SHA__'
print('[PROFILE BOOTSTRAP] locating source; elapsed',round(time.monotonic()-STARTED,1),'s',flush=True)
def safe(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or p.as_posix()!=name or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe source path')
    out=root/name
    if not out.resolve().is_relative_to(root.resolve()):raise ValueError('Escaping source path')
    return out
def manifest(p):
    if p.is_dir():return json.loads((p/'throughput_manifest.json').read_text())
    with zipfile.ZipFile(p) as z:return json.loads(z.read('throughput_manifest.json'))
choices=[Path(SOURCE_OVERRIDE)] if SOURCE_OVERRIDE else [p.parent for p in INPUT.rglob('throughput_manifest.json')]+list(INPUT.rglob('*.zip'))
valid=[]
for p in choices:
    try:
        m=manifest(p)
        if m.get('schema')=='hope_support_v5_1_throughput_source_v1':valid.append((p,m))
    except (OSError,KeyError,ValueError,zipfile.BadZipFile):
        if SOURCE_OVERRIDE:raise
identities={json.dumps(m,sort_keys=True) for p,m in valid}
if len(identities)!=1:raise ValueError('Expected ONE profiler source identity; set SOURCE_OVERRIDE')
source,m=sorted(valid,key=lambda v:str(v[0]))[0]
if m['contract_sha256']!=PINNED_CONTRACT:raise ValueError('Notebook/source contract version mismatch')
ROOT.mkdir(parents=True,exist_ok=True)
z=None if source.is_dir() else zipfile.ZipFile(source)
try:
    if z and (len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(m['files'])|{'throughput_manifest.json'}):raise ValueError('Source ZIP inventory mismatch')
    for name,expected in m['files'].items():
        original=safe(source,name) if source.is_dir() else None
        if original is not None and original.is_symlink():raise ValueError('Source symlink')
        raw=original.read_bytes() if original is not None else z.read(name)
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Source payload SHA mismatch: '+name)
        dest=safe(ROOT,name);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=raw:raise ValueError('Existing source differs; use a fresh kernel/session')
        dest.write_bytes(raw)
    target=ROOT/'throughput_manifest.json';raw=json.dumps(m,sort_keys=True).encode()
    if target.exists() and json.loads(target.read_bytes())!=m:raise ValueError('Existing delivery differs')
    target.write_bytes(raw)
finally:
    if z:z.close()
for name in ('training','models','dataset','notebook_support'):
    module=sys.modules.get(name)
    if module:
        paths=list(getattr(module,'__path__',[]))
        if not paths or any(not Path(p).resolve().is_relative_to(ROOT) for p in paths):raise ValueError('Mixed source imports; restart kernel')
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
print('[PROFILE BOOTSTRAP] verified',len(m['files']),'files;',m['origin'],flush=True)
print('Fixed budget90min; U1/P1/U2 on16known normals; no benchmark continuation',flush=True)
'''.replace('__CONTRACT_SHA__',pin)
    run='''from notebook_support.support_throughput_v5_1.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),started=STARTED,
    PARENT_OVERRIDE=PARENT_OVERRIDE,RESUME_OVERRIDE=RESUME_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''
    cells=[]
    for i,(typ,text) in enumerate([('markdown',intro),('code',boot),('code',run)]):
        cell=dict(cell_type=typ,id='throughput-v5-1-'+str(i),metadata={},source=text.splitlines(keepends=True))
        if typ=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.13.15')),nbformat=4,nbformat_minor=5)
    p=ROOT/NOTEBOOK;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return notebook

def build(output=None):
    cfg=contract();shared=ROOT.parent/'.agents';source=shared/cfg['frozen_reference']['source_zip_from_shared_root']
    if sha(source)!=cfg['frozen_reference']['source_zip_sha256']:raise ValueError('Original benchmark ZIP differs')
    write_notebook()
    extension=[]
    for folder in ['training/support_throughput_v5_1','notebook_support/support_throughput_v5_1','tests/support_throughput_v5_1']:
        extension += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*.py')]
    extension += [NOTEBOOK,'docs/main_spec/throughput_v5_1_spec.md','docs/main_spec/throughput_v5_1_plan.md',
        'docs/main_spec/throughput_v5_1_contract.json','docs/main_spec/throughput_v5_1_implementation.md']
    with zipfile.ZipFile(source) as z:
        base=json.loads(z.read('bundle_manifest.json'))
        payload={name:z.read(name) for name in z.namelist()}
    if len(base['files'])!=166 or digest(base['files'])!=cfg['frozen_reference']['source_digest']:raise ValueError('Base map differs')
    for name,h in base['files'].items():
        if sha(ROOT/name)!=h or hashlib.sha256(payload[name]).hexdigest()!=h:raise ValueError('Frozen base changed: '+name)
    if set(extension)&set(payload):raise ValueError('Profiler extension overwrites frozen payload')
    payload.update({name:(ROOT/name).read_bytes() for name in sorted(set(extension))})
    files={name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(payload.items())}
    origin=dict(delivery_revision='throughput-profile-r1',frozen_reference_source_digest=cfg['frozen_reference']['source_digest'],
        scientific_method_unchanged=True,identity_basis='all packaged file SHA256s; additive extension; not commit alone')
    manifest=dict(schema=SOURCE_SCHEMA,contract_sha256=sha(ROOT/'docs/main_spec/throughput_v5_1_contract.json'),origin=origin,files=files)
    output=Path(output) if output else shared/'artifacts/deliveries'/DELIVERY/ZIP_NAME
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Delivery exists; never overwrite a handed-off source archive')
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,raw in payload.items():z.writestr(n,raw)
        z.writestr('throughput_manifest.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(output) as z:
        for n,h in files.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Source export verification failed')
    receipt=dict(schema='throughput_delivery_receipt_v1',path=str(output),bytes=output.stat().st_size,sha256=sha(output),source_digest=digest(files),
        files=len(files),frozen_base_payloads=166,extension_payloads=len(extension),canonical_T4_profile='not_run',optimization_implemented=False,
        notebook=NOTEBOOK,contract_sha256=manifest['contract_sha256'])
    (output.parent/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2));return receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output');ap.add_argument('--notebook-only',action='store_true');a=ap.parse_args()
    if a.notebook_only:write_notebook()
    else:build(a.output)
if __name__=='__main__':main()
