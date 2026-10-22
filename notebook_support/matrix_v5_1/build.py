"""Add readonly matrix launchers to the frozen matched-CADIC base; metadata only."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE_SHA='d5af61547fb1b77000d5781b6dead5821b618d0172f1046816790ca870113f3f'
PINS={
 'NESTED_seed0':('support_v5_1_benchmark_NESTED_seed0_continuation','hope_cad_support_v5_1_continuation_evidence.zip','039e91861f27864e9d85b31761dc9794d1a1acf412faa97d3b95f734b4ca21f1','support-v5-1-nested-seed0-final-session1-20261021'),
 'CADIC_seed0':('support_v5_1_matched_CADIC_seed0','hope_cad_matched_cadic_seed0_evidence.zip','bf1e0590d95dea68ddf2742bcfd29b5c5a74175114b47bc21925e4481d773b82','support-v5-1-matched-cadic-seed0-session1-20261021')}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write_once(p,raw):
    if p.exists() and p.read_bytes()!=raw:raise ValueError('Frozen receipt changed; new revision required')
    p.write_bytes(raw)

def notebooks():
    old=json.loads((ROOT/'notebooks/hope_cad_v5_1_matched_cadic_seed0_kaggle.ipynb').read_text(encoding='utf-8'))
    paths=[]
    for job in PINS:
        boot=''.join(old['cells'][1]['source'])
        changes={"PHASE='7.1'":"PHASE='7.2'","JOB='CADIC_seed0'":f"JOB='{job}'",'RESUME_OVERRIDE':'PARENT_OVERRIDE','matched_cadic_manifest.json':'matrix_manifest.json','hope_matched_cadic_seed0_source_v1':'hope_support_v5_1_matrix_7_2_source_v1','/tmp/hope_matched_cadic_seed0_source_r1':'/tmp/hope_matrix_7_2_source_r1','/kaggle/working/hope_matched_cadic_seed0':f'/kaggle/working/hope_matrix_{job}','/tmp/hope_matched_cadic_assets':f'/tmp/hope_matrix_assets_{job}','hope_cad_matched_cadic_seed0_evidence.zip':f'hope_cad_matrix_{job}_evidence.zip','[MATCHED CADIC BOOTSTRAP]':'[MATRIX BOOTSTRAP]','[MATCHED CADIC IDENTITY]':'[MATRIX IDENTITY]','Expected ONE matched CADIC source identity':'Expected ONE matrix source identity','frozen core plus fresh CADIC orchestration':'frozen core plus readonly matrix orchestration','from notebook_support.matched_cadic_v5_1.common import verify_source':'from notebook_support.matrix_v5_1.common import verify_source','ORCHESTRATION,REFERENCE,TRAIN_MANIFEST=verify_source()':'ORCHESTRATION,REFERENCE,CORE_SOURCE=verify_source()'}
        for a,b in changes.items():boot=boot.replace(a,b)
        parent=PINS[job][1]
        intro=f'''# Nested Learning for CAD — 7.2 {job}

Readonly task matrix từ completed7.1: **không train lại**. Đọc `docs/main_spec/matrix_7_2_v5_1.md` và implementation. Hai method chạy notebook SavedVersion riêng; không chia sharedstate.

**Đúng4inputs:** source mới `hope_cad_matrix_7_2_source_r1.zip` (hoặc Kaggle giải nén), completedown7.1 **`{parent}`**, MVTec AD và lockedViT `vit_base_patch8_224.augreg_in21k.safetensors`. Tháo source7.1, evidence methodkhác, bridge/pilot/cost/W6. Mỗi notebook chỉ một parentversion; duplicate ZIP/directory cùngmanifest được chấpnhận.

GPU T4×2, freshkernel; giữ `PHASE='7.2'`, `JOB='{job}'`, `SESSION_HOURS=9.0`. Save Version → Save & Run All. Budget9h là softpause có600s exportreserve, khôngETA. Khôngpipupgrade hoặc đổiGPU để bypass runtime. Cảhai versions cóthể chạy song song nếu Kaggle cho phép/quota đủ; quota cộngdồn. Không cầnDrive/W&B hoặc preflightnotebookriêng.

Bootstrap tựverifyparent/source/runtime/ViTSHA, khôngloadtrain, chỉstage officialtest/GT; preflightscore2images trên discardedactors rồi thêm105cells cònthiếu. 15finalcells vàtoàntrainingstate giữbytes; NESTEDtask00 dùng resolverlegacy đãreview. Cóheartbeat30s và `[BENCH CELL] i j complete`. Không có `[BENCH TRAIN]` mới.

`paused_at_atomic_boundary`: giữZIP/log; phiênsau **thay7.1input bằng latestown7.2output**, vẫn4inputs/cùngsource/runtime, không giữnhiềuphiêncũ. `7.2_complete_pending_review`: đủ120cells, gửiZIP/log. Failure phải export rồiraise; giữevidence đểreview, khôngtựretry hoặc sửa phươngpháp.

Output **`hope_cad_matrix_{job}_evidence.zip`**; lưu ZIP+Kagglelog tại `D:/src/DATN/.agents/artifacts/runs/support_v5_1_matrix_{job}/session_01/`; resume tăngsession. Metrics `benchmark/runs/{job}/metrics/task_matrix.json`. Không chạyfullstudy/VisA hoặc retune từmatrix.
'''
        run='''from notebook_support.matrix_v5_1.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,job=JOB,session_hours=SESSION_HOURS,started=STARTED,
    inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),
    PARENT_OVERRIDE=PARENT_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''
        cells=[]
        for i,(kind,t) in enumerate([('markdown',intro),('code',boot),('code',run)]):
            cell=dict(cell_type=kind,id='matrix-'+job+'-'+str(i),metadata={},source=t.splitlines(keepends=True))
            if kind=='code':compile(t,'matrix_cell','exec');cell.update(outputs=[],execution_count=None)
            cells.append(cell)
        path=ROOT/f'notebooks/hope_cad_v5_1_matrix_{job}_kaggle.ipynb';path.write_text(json.dumps(dict(cells=cells,metadata=old['metadata'],nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=1)+'\n',encoding='utf-8');paths.append(path)
    return paths

def build(output):
    shared=ROOT.parent/'.agents';base=shared/'artifacts/deliveries/hope_support_v5_1_matched_cadic_seed0_r1/hope_cad_matched_cadic_seed0_source_r1.zip'
    if sha(base)!=BASE_SHA:raise ValueError('Wrong frozen matched-CADIC base')
    with zipfile.ZipFile(base) as z:payload={n:z.read(n) for n in z.namelist()}
    if len(payload)!=249:raise ValueError('Wrong base inventory')
    for n,raw in payload.items():
        if (ROOT/n).is_file() and (ROOT/n).read_bytes()!=raw:raise ValueError('Frozen live base drift: '+n)
    paths=notebooks();receipt=ROOT/'notebook_support/matrix_v5_1/receipts';receipt.mkdir(exist_ok=True)
    reference=json.loads(payload['notebook_support/matched_cadic_v5_1/receipts/reference.json'])
    refs=dict(schema='matrix_reviewed_parents_v1',runtime=reference['runtime'],backbone_sha256=reference['backbone_sha256'],train_manifest_sha256=reference['train_manifest_sha256'],test_manifest_sha256=reference['test_manifest_sha256'],jobs={})
    for job,(run,name,h,reviewid) in PINS.items():
        parent=shared/'artifacts/runs'/run/'session_01'/name
        if sha(parent)!=h:raise ValueError('Wrong accepted parent ZIP')
        review=shared/'evidence/reviews'/reviewid/'review.json'
        if json.loads(review.read_text(encoding='utf-8'))['status']!='finalized':raise ValueError('Missing finalized parent review')
        with zipfile.ZipFile(parent) as z:
            raw=z.read('artifact_manifest.json');m=json.loads(raw);handoff=json.loads(z.read('handoff.json'))
            if handoff['job']!=job or handoff['status']!='7.1_complete_pending_review':raise ValueError('Parent is not completed own7.1')
            for n in ['benchmark/manifest_train.json','benchmark/test_manifest.json']:
                expected=refs['train_manifest_sha256'] if n.endswith('manifest_train.json') else refs['test_manifest_sha256']
                if m['files'][n]!=expected or hashlib.sha256(z.read(n)).hexdigest()!=expected:raise ValueError('Parents dataset differs')
            refs['jobs'][job]=dict(archive_sha256=h,manifest_sha256=hashlib.sha256(raw).hexdigest(),schema=m['schema'],review_sha256=sha(review))
            write_once(receipt/(job+'_manifest.json'),raw)
    write_once(receipt/'parents.json',(json.dumps(refs,sort_keys=True,indent=2)+'\n').encode())
    added=[p for p in (ROOT/'notebook_support/matrix_v5_1').rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    added+=list((ROOT/'tests/matrix_v5_1').rglob('*.py'))+paths+[ROOT/'docs/main_spec/matrix_7_2_v5_1.md',ROOT/'docs/main_spec/matrix_7_2_v5_1_implementation.md']
    for p in added:
        n=p.relative_to(ROOT).as_posix()
        if n in payload:raise ValueError('Extension overwrites frozen source')
        payload[n]=p.read_bytes()
    if any(Path(n).suffix in ('.pt','.pth','.safetensors','.npz') for n in payload):raise ValueError('No weights/data in source')
    files={n:hashlib.sha256(raw).hexdigest() for n,raw in sorted(payload.items())}
    manifest=dict(schema='hope_support_v5_1_matrix_7_2_source_v1',files=files,frozen_base_entries=249,base_zip_sha256=BASE_SHA,phase='7.2',jobs=list(PINS),science_changed=False)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,raw in sorted(payload.items()):z.writestr(n,raw)
        z.writestr('matrix_manifest.json',json.dumps(manifest,sort_keys=True))
    result=dict(path=str(output),sha256=sha(output),source_digest=digest(files),files=len(files),frozen_base_byte_exact_entries=249,method_changed=False,canonical_T4='not_run')
    output.with_suffix('.receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
