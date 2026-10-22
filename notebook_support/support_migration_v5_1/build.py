"""Additive G4 source package over the exact reviewed200-payload G3 delivery."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path
from training.support_throughput_v5_1.contract import ROOT,sha,digest
from training.support_migration_v5_1.contract import contract,CONTRACT_PATH,SOURCE_SCHEMA

NOTEBOOK='notebooks/hope_cad_v5_1_nn_write_bridge_kaggle.ipynb'
ZIP_NAME='hope_cad_support_v5_1_nn_write_bridge_source_r2.zip'
DELIVERY='hope_support_v5_1_nn_write_bridge_r2'

def write_notebook():
    original=json.loads((ROOT/'notebooks/hope_cad_v5_1_nn_write_validation_kaggle.ipynb').read_text(encoding='utf-8'))
    boot=''.join(original['cells'][1]['source'])
    for a,b in {"PHASE='validate'":"PHASE='bridge'",'nn_write_manifest.json':'bridge_manifest.json',
        'hope_support_v5_1_nn_write_source_v1':SOURCE_SCHEMA,'/tmp/hope_nn_write_source_r1':'/tmp/hope_nn_write_bridge_source_r2',
        '/kaggle/working/hope_nn_write_validation':'/kaggle/working/hope_nn_write_bridge',
        '/tmp/hope_throughput_assets':'/tmp/hope_nn_write_bridge_assets',
        'hope_cad_support_v5_1_nn_write_evidence.zip':'hope_cad_support_v5_1_nn_write_bridge_evidence.zip',
        '[NN BOOTSTRAP]':'[BRIDGE BOOTSTRAP]','Expected ONE profiler source identity':'Expected ONE bridge source identity',
        'Fixed90min; R1/O1/O2/R2 on16known normals; exact gradient diagnostics; no benchmark adoption':'Fixed60min budget;8newarrivals376-383 lockstep; export384 only after exact; no auto benchmark continuation'}.items():boot=boot.replace(a,b)
    boot=''.join(line for line in boot.splitlines(keepends=True) if not line.startswith('RESUME_OVERRIDE='))
    boot=re.sub(r"PINNED_CONTRACT='[0-9a-f]{64}'","PINNED_CONTRACT='"+sha(CONTRACT_PATH)+"'",boot)
    intro='''# Nested Learning for CAD — bridge chuyển source NN-write v5.1

G3 đã review: giảm44% pipeline time trên cửa sổ16 ảnh, exact gradients/state. **Notebook này chỉ làm G4 bridge**, không thay method hoặc tự chạy hết benchmark.

**Bốn inputs:** source bridge mới `hope_cad_support_v5_1_nn_write_bridge_source_r2.zip` (ZIP hoặc Kaggle giải nén); **original reviewed benchmark NESTED_seed0 paused376/3599** evidence; MVTec AD raw; locked `vit_base_patch8_224.augreg_in21k.safetensors`. Tháo source/output NN-write validation và profiler. Không attach W6/pilot hoặc clone195. Delivery r2 hỗ trợ metadata tar khác thứ tự; không thay phương pháp hoặc bridge budget.

GPU **T4×2**, kernel mới. Giữ PHASE='bridge'; các overrides=None nếu discovery rõ. **Save Version → Save & Run All.** Có heartbeat30 giây. Source verify trước; runtime/backbone/parent checks fail closed. Tar được đọc theo từng arrival, không extract sẵn future normals. Budget60 phút gồm bootstrap và180s export reserve, không phải ETA; không tự tăng budget hoặc retry partial bridge.

Hai owners cùng strict-load active376 và revalidate pending trên T4. Mỗi ảnh376–383 được xử lý ở cả reference/candidate rồi so full state/event trước ảnh kế tiếp. Đủ8 exact mới publish candidate384 cùng new execution lineage; không finalize pending/partial. Không dùng ADtest/GT/VisA.

`bridge_complete_pending_review`: giữ ZIP/log để review trước khi tiếp benchmark. `bridge_incomplete_preserve_parent` hoặc failure: giữ mọi evidence, parent376 vẫn nguyên; không tự resume/rerun hoặc dùng shadow reference. Notebook không expose continuation sau384.

Output `hope_cad_support_v5_1_nn_write_bridge_evidence.zip` và log lưu tại `D:/src/DATN/.agents/artifacts/runs/support_v5_1_nn_write_bridge_01/session_01/`. Đọc `docs/main_spec/migration_nn_write_v5_1.md` và `migration_nn_write_v5_1_implementation.md` trước chạy.
'''
    run='''from notebook_support.support_migration_v5_1.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),started=STARTED,
    PARENT_OVERRIDE=PARENT_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''
    cells=[]
    for i,(kind,text) in enumerate([('markdown',intro),('code',boot),('code',run)]):
        v=dict(cell_type=kind,id='nn-bridge-'+str(i),metadata={},source=text.splitlines(keepends=True))
        if kind=='code':compile(text,'notebook_cell','exec');v.update(outputs=[],execution_count=None)
        cells.append(v)
    nb=dict(cells=cells,metadata=original['metadata'],nbformat=4,nbformat_minor=5)
    (ROOT/NOTEBOOK).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');return nb

def build(output=None):
    write_notebook();c=contract();shared=ROOT.parent/'.agents'
    base=shared/'artifacts/deliveries/hope_support_v5_1_nn_write_r1/hope_cad_support_v5_1_nn_write_source_r1.zip'
    if sha(base)!=c['G3_source_zip_sha256']:raise ValueError('Wrong reviewed G3 ZIP')
    with zipfile.ZipFile(base) as z:payload={n:z.read(n) for n in z.namelist()};old=json.loads(z.read('nn_write_manifest.json'))
    if digest(old['files'])!=c['G3_source_digest']:raise ValueError('G3 source map differs')
    for n,h in old['files'].items():
        if hashlib.sha256(payload[n]).hexdigest()!=h or (n not in ('bundle_manifest.json','throughput_manifest.json') and sha(ROOT/n)!=h):raise ValueError('Reviewed G3 source drift: '+n)
    extension=[]
    for folder in ['training/support_migration_v5_1','notebook_support/support_migration_v5_1','tests/support_migration_v5_1']:
        extension += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    extension += [NOTEBOOK,'docs/main_spec/migration_nn_write_v5_1.md','docs/main_spec/migration_nn_write_v5_1_contract.json','docs/main_spec/migration_nn_write_v5_1_implementation.md']
    if set(extension)&set(payload):raise ValueError('Bridge extension overwrites reviewed source')
    payload.update({n:(ROOT/n).read_bytes() for n in sorted(extension)})
    files={n:hashlib.sha256(b).hexdigest() for n,b in sorted(payload.items())}
    m=dict(schema=SOURCE_SCHEMA,contract_sha256=sha(CONTRACT_PATH),origin=dict(G3_source_digest=c['G3_source_digest'],method_changed=False,migration='G4_user_run_pending',identity_basis='all packaged payload hashes'),files=files)
    output=Path(output) if output else shared/'artifacts/deliveries'/DELIVERY/ZIP_NAME;output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('bridge_manifest.json',json.dumps(m,sort_keys=True))
    with zipfile.ZipFile(output) as z:
        if any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in files.items()):raise ValueError('Packaged SHA verification failed')
    receipt=dict(schema='nn_write_bridge_delivery_v1',path=str(output),sha256=sha(output),bytes=output.stat().st_size,files=len(files),source_digest=digest(files),
        immutable_G3_payloads=200,extension_payloads=len(extension),contract_sha256=sha(CONTRACT_PATH),canonical_T4_bridge='not_run',notebook=NOTEBOOK)
    (output.parent/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2));return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--notebook-only',action='store_true');a=p.parse_args()
    if a.notebook_only:write_notebook()
    else:build(a.output)
if __name__=='__main__':main()
