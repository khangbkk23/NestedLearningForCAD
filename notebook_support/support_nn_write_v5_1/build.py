"""Add candidate source to exact reviewed profiler delivery; never overwrite an artifact."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path
from training.support_throughput_v5_1.contract import ROOT,sha,digest
from training.support_nn_write_v5_1.contract import contract,SOURCE_SCHEMA,CONTRACT_PATH

NOTEBOOK='notebooks/hope_cad_v5_1_nn_write_validation_kaggle.ipynb'
ZIP_NAME='hope_cad_support_v5_1_nn_write_source_r1.zip'
DELIVERY='hope_support_v5_1_nn_write_r1'

def write_notebook():
    original=json.loads((ROOT/'notebooks/hope_cad_v5_1_throughput_profile_kaggle.ipynb').read_text(encoding='utf-8'))
    boot=''.join(original['cells'][1]['source'])
    replacements={"PHASE='profile'":"PHASE='validate'",'throughput_manifest.json':'nn_write_manifest.json',
        'hope_support_v5_1_throughput_source_v1':SOURCE_SCHEMA,'/tmp/hope_throughput_source_r1':'/tmp/hope_nn_write_source_r1',
        '/kaggle/working/hope_throughput_v5_1':'/kaggle/working/hope_nn_write_validation',
        'hope_cad_support_v5_1_throughput_evidence.zip':'hope_cad_support_v5_1_nn_write_evidence.zip',
        '[PROFILE BOOTSTRAP]':'[NN BOOTSTRAP]',
        'Fixed budget90min; U1/P1/U2 on16known normals; no benchmark continuation':'Fixed90min; R1/O1/O2/R2 on16known normals; exact gradient diagnostics; no benchmark adoption'}
    for a,b in replacements.items():boot=boot.replace(a,b)
    boot=re.sub(r"PINNED_CONTRACT='[0-9a-f]{64}'","PINNED_CONTRACT='"+sha(CONTRACT_PATH)+"'",boot)
    intro='''# Nested Learning for CAD — kiểm chứng tối ưu NN-write v5.1

Patch execution riêng, giữ đầy đủ Nested Learning. **Chưa biết speedup T4; chưa được migrate benchmark.** Đọc `docs/main_spec/throughput_nn_write_v5_1.md` và `nn_write_v5_1_implementation.md`.

**4 inputs:** newsource `hope_cad_support_v5_1_nn_write_source_r1.zip` (Kaggle giải nén cũng được), **original reviewed NESTED_seed0 benchmark376/3599** evidence (ZIP/output), MVTec raw, lockedViT-B/8 safetensors. Tháo source profiler cũ và completed profiler evidence: G1receipt đã bind trong source; parent vẫn là376, không lấy state195 của clone. Không attach W6/pilot learned states.

**GPU T4×2; Save Version → Save & Run All; kernel mới.** Giữ PHASE='validate'. R1→O1→O2→R2, mỗi pass từ179 qua16ảnh179–194.64attempts tối đa/90min cumulative,180s exportreserve. Budget không phải ETA; hai referencepasses riêng trước đây ~34phút, tổng phụ thuộc candidate/diagnostics. Không tự tăng budget hoặc retry partialpass.

Primarytiming loại các khoảng capture giống nhau ở R/O. Sau đó recompute actual/meta gradients/steps trên clone để kiểm exact; không thêmimagearrivals hoặc ADtest. Heartbeat30s và progress mỗi8arrivals. Check `summary.json`: complete không tự là speedgatePASS, không là approval adopt/migrate.

Output `hope_cad_support_v5_1_nn_write_evidence.zip` và log lưu vào `.agents/artifacts/runs/support_v5_1_nn_write_validation_01/session_01/`. Giữ mọi failure/incomplete evidence. Chỉ resume latest **NN-write validation** output khi được hướng dẫn giữa completedpasses; tháo output cũ. Sau run cần review trước khi đổi source benchmark376.
'''
    run='''from notebook_support.support_nn_write_v5_1.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),started=STARTED,
    PARENT_OVERRIDE=PARENT_OVERRIDE,RESUME_OVERRIDE=RESUME_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''
    cells=[]
    for i,(kind,text) in enumerate([('markdown',intro),('code',boot),('code',run)]):
        v=dict(cell_type=kind,id='nn-write-'+str(i),metadata={},source=text.splitlines(keepends=True))
        if kind=='code':compile(text,'notebook_cell','exec');v.update(outputs=[],execution_count=None)
        cells.append(v)
    nb=dict(cells=cells,metadata=original['metadata'],nbformat=4,nbformat_minor=5)
    (ROOT/NOTEBOOK).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return nb

def build(output=None):
    write_notebook();cfg=contract();shared=ROOT.parent/'.agents'
    base=shared/'artifacts/deliveries/hope_support_v5_1_throughput_profile_r1/hope_cad_support_v5_1_throughput_source_r1.zip'
    if sha(base)!=cfg['base_profile_source_zip_sha256']:raise ValueError('Wrong reviewed profiler ZIP')
    with zipfile.ZipFile(base) as z:
        payload={n:z.read(n) for n in z.namelist()};old=json.loads(z.read('throughput_manifest.json'))
    if digest(old['files'])!=cfg['base_profile_source_digest']:raise ValueError('Base183source differs')
    for n,h in old['files'].items():
        if hashlib.sha256(payload[n]).hexdigest()!=h or (n!='bundle_manifest.json' and sha(ROOT/n)!=h):raise ValueError('Frozen183source drift: '+n)
    extension=[]
    for folder in ['training/support_nn_write_v5_1','notebook_support/support_nn_write_v5_1','tests/support_nn_write_v5_1']:
        extension += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*.py')]
    extension += [NOTEBOOK,'docs/main_spec/throughput_nn_write_v5_1.md','docs/main_spec/throughput_nn_write_v5_1_contract.json','docs/main_spec/nn_write_v5_1_implementation.md']
    if set(extension)&set(payload):raise ValueError('Extension replaces immutable base')
    payload.update({n:(ROOT/n).read_bytes() for n in sorted(extension)})
    files={n:hashlib.sha256(b).hexdigest() for n,b in sorted(payload.items())}
    manifest=dict(schema=SOURCE_SCHEMA,contract_sha256=sha(CONTRACT_PATH),origin=dict(candidate=cfg['candidate_id'],frozen_base_profile_source_digest=cfg['base_profile_source_digest'],method_unchanged=True,identity_basis='all packaged payload hashes'),files=files)
    output=Path(output) if output else shared/'artifacts/deliveries'/DELIVERY/ZIP_NAME
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('nn_write_manifest.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(output) as z:
        if any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in files.items()):raise ValueError('Delivery SHA verification failed')
    receipt=dict(schema='nn_write_delivery_receipt_v1',path=str(output),sha256=sha(output),bytes=output.stat().st_size,files=len(files),source_digest=digest(files),frozen_profile_payloads=183,extension_payloads=len(extension),canonical_T4='not_run',migration_implemented=False,contract_sha256=sha(CONTRACT_PATH),notebook=NOTEBOOK)
    (output.parent/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2));return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--notebook-only',action='store_true');a=p.parse_args()
    if a.notebook_only:write_notebook()
    else:build(a.output)
if __name__=='__main__':main()
