"""Add fresh CADIC launcher and metadata to exact frozen continuation source; never package learned state."""
import argparse,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE_SHA='8452323e50cbbe1730492ea8525f99e84f7f7a84c121dfcfe59ad634278e6d2b'
NESTED_SHA='039e91861f27864e9d85b31761dc9794d1a1acf412faa97d3b95f734b4ca21f1'
NOTEBOOK='notebooks/hope_cad_v5_1_matched_cadic_seed0_kaggle.ipynb'
SOURCE_SCHEMA='hope_matched_cadic_seed0_source_v1'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def save_receipt(path,raw):
    if path.exists() and path.read_bytes()!=raw:raise ValueError('Receipt changed; new delivery revision required')
    path.write_bytes(raw)

def write_notebook():
    old=json.loads((ROOT/'notebooks/hope_cad_v5_1_continuation_7_1_kaggle.ipynb').read_text(encoding='utf-8'))
    boot=''.join(old['cells'][1]['source'])
    substitutions={"JOB='NESTED_seed0'":"JOB='CADIC_seed0'",'PARENT_OVERRIDE':'RESUME_OVERRIDE','continuation_manifest.json':'matched_cadic_manifest.json','hope_support_v5_1_continuation_7_1_source_v1':SOURCE_SCHEMA,'/tmp/hope_continuation_7_1_source_r1':'/tmp/hope_matched_cadic_seed0_source_r1','/kaggle/working/hope_continuation_7_1':'/kaggle/working/hope_matched_cadic_seed0','/tmp/hope_continuation_assets':'/tmp/hope_matched_cadic_assets','hope_cad_support_v5_1_continuation_evidence.zip':'hope_cad_matched_cadic_seed0_evidence.zip','[CONTINUATION BOOTSTRAP]':'[MATCHED CADIC BOOTSTRAP]','[CONTINUATION IDENTITY]':'[MATCHED CADIC IDENTITY]','Expected ONE continuation source identity':'Expected ONE matched CADIC source identity','frozen r2 plus separate operational extension':'frozen core plus fresh CADIC orchestration','from continuation_runtime_v1.common import verify_delivery':'from notebook_support.matched_cadic_v5_1.common import verify_source','CORE_SOURCE,ORCHESTRATION=verify_delivery()':'ORCHESTRATION,REFERENCE,TRAIN_MANIFEST=verify_source()'}
    for a,b in substitutions.items():boot=boot.replace(a,b)
    intro='''# Nested Learning for CAD — matched CADIC seed0

Đối chứng cùng study/protocol với NESTED v5.1 đã chạy. Bank mới K10000, rawViTgeometry/fullEq9, cùng backendNN tốiưu. Không dùng learned bank NESTED hoặc clone của góiCADICcost. Đọc `docs/main_spec/matched_cadic_v5_1.md` và implementation.

**Lần đầu attach đúng3inputs:** source mới `hope_cad_matched_cadic_seed0_source_r1.zip` (ZIP hoặc Kaggle giải nén), MVTec AD cũ, `vit_base_patch8_224.augreg_in21k.safetensors` cũ. Tháo source/evidence NESTED, bridge, CADICcost và pilot/W6. Source đã đóng kèm metadata manifest/runtime; không cần output/preflight khác.

GPU T4×2; kernel/session mới; `PHASE='7.1'`, `JOB='CADIC_seed0'`, `SESSION_HOURS=9.0`. Save Version → Save & Run All. Budget9h là softlimit gồm600s exportreserve, khôngETA; atomic event dài/hardplatformkill vẫn có thể làm mấtlatestoutput. Runtime mismatch thì giữlog đểreview, không tự pipupgrade. Cóheartbeat30s/progress8ảnh.

Notebook tự stage/hash trainmanifest, kiểm discarded2normalowners reference/optimized/readonlyscore, sau đó **fresh CADIC0→3599** và15finalmetrics. Từng rawfeaturehash phải khớp NESTED reference. Test/GT chỉ evaluator sau traincomplete. Không giảmảnh/K hoặc đổiscore để khớp kếtquả mongmuốn.

`paused_at_atomic_boundary`: tảiZIP/log; phiênsau thêm **latest CADIC output** thànhinputthứ4, giữsource/dataset/checkpoint và notebook, khôngattach cácphiêncũ khác. `7.1_complete_pending_review`: đã xong, gửiZIP/log đểreview. `failed_preserve_evidence`: giữevidence, không tựretry. Không dùngCADICcostclone195 đểresume.

Output `hope_cad_matched_cadic_seed0_evidence.zip` vàKagglelog lưu tại `D:/src/DATN/.agents/artifacts/runs/support_v5_1_matched_CADIC_seed0/session_01/`; cácphiêntiếp theo session_02… Metrics ở `benchmark/runs/CADIC_seed0/metrics/`. Đủ15taskstates sẽdùng cho7.2 sau này; notebooknày chưa exposematrix/seeds/VisA. Kếtquảseed0 descriptive, chưafullstudy/CI.
'''
    run='''from notebook_support.matched_cadic_v5_1.entry import run_notebook
RESULT=run_notebook(dict(phase=PHASE,job=JOB,session_hours=SESSION_HOURS,started=STARTED,
    inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),
    RESUME_OVERRIDE=RESUME_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''
    cells=[]
    for i,(kind,text) in enumerate([('markdown',intro),('code',boot),('code',run)]):
        cell=dict(cell_type=kind,id='matched-cadic-'+str(i),metadata={},source=text.splitlines(keepends=True))
        if kind=='code':compile(text,'matched_cadic_cell','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    nb=dict(cells=cells,metadata=old['metadata'],nbformat=4,nbformat_minor=5)
    (ROOT/NOTEBOOK).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')

def build(output):
    write_notebook();shared=ROOT.parent/'.agents'
    base=shared/'artifacts/deliveries/hope_support_v5_1_continuation_7_1_r1/hope_cad_support_v5_1_continuation_source_r1.zip'
    if sha(base)!=BASE_SHA:raise ValueError('Wrong frozen base ZIP')
    with zipfile.ZipFile(base) as z:payload={n:z.read(n) for n in z.namelist()}
    if len(payload)!=235:raise ValueError('Unexpected base inventory')
    # Every old source entry that has a live counterpart must remain byte-exact.
    for n,raw in payload.items():
        p=ROOT/n
        if p.exists() and p.read_bytes()!=raw:raise ValueError('Frozen live base drift: '+n)
    parent=shared/'artifacts/runs/support_v5_1_benchmark_NESTED_seed0_continuation/session_01/hope_cad_support_v5_1_continuation_evidence.zip'
    if sha(parent)!=NESTED_SHA:raise ValueError('Wrong reviewed reference evidence')
    receipts=ROOT/'notebook_support/matched_cadic_v5_1/receipts';receipts.mkdir(exist_ok=True)
    review=shared/'evidence/reviews/support-v5-1-nested-seed0-final-session1-20261021/review.json';r=json.loads(review.read_text())
    if r['status']!='finalized' or r['scientific_status']!='canonical_T4_seed0_final_endpoint_reviewed_matched_CADIC_pending':raise ValueError('Reference not reviewed')
    cost=shared/'evidence/reviews/support-v5-1-cadic-cost-session1-20261021/review.json';save_receipt(receipts/'cadic_cost_review.json',cost.read_bytes())
    with zipfile.ZipFile(parent) as z:
        manifest=json.loads(z.read('artifact_manifest.json'))
        def verified(name):
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=manifest['files'][name]:raise ValueError('Reference payload drift')
            return raw
        trainraw=verified('benchmark/manifest_train.json');train=json.loads(trainraw);save_receipt(receipts/'manifest_train.json',trainraw)
        suite=json.loads(verified('benchmark/suite_lock.json'))['identity'];test=json.loads(verified('benchmark/test_manifest.json'))
        config=json.loads(verified('session_history/0001/worker_config.json'))
        features={}
        for row in train['online']:
            key=digest(row);receipt=json.loads(verified('benchmark/runs/NESTED_seed0/feature_receipts/'+key+'.json'))
            if receipt['row']!=row:raise ValueError('Reference feature row changed')
            features[key]=receipt['tensor_sha256']
        save_receipt(receipts/'feature_hashes.json',(json.dumps(features,sort_keys=True)+'\n').encode())
        ref=dict(schema='matched_cadic_reference_v1',reference_evidence_sha256=NESTED_SHA,reference_review_sha256=sha(review),reference_manifest_sha256=hashlib.sha256(z.read('artifact_manifest.json')).hexdigest(),original_scientific_source_digest='ece9cfcaa66f63f0ddea18124667f22636a99a9ffc1051947b80f413a80ff5e5',core_r2_source_digest=suite['source_digest'],study_digest=suite['study_digest'],design_digest=suite['design_digest'],normal_rows_digest=digest(train['online']),train_manifest_sha256=manifest['files']['benchmark/manifest_train.json'],test_manifest_sha256=manifest['files']['benchmark/test_manifest.json'],test_manifest_digest=test['digest'],runtime=config['runtime'],backbone_sha256=suite['backbone_sha256'],cadic_cost_review_sha256=sha(cost),feature_hashes_sha256=sha(receipts/'feature_hashes.json'),learned_tensors_packaged=False,metrics_packaged=False)
        save_receipt(receipts/'reference.json',(json.dumps(ref,sort_keys=True,indent=2)+'\n').encode())
    added=[p for p in (ROOT/'notebook_support/matched_cadic_v5_1').rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    added+=[p for p in (ROOT/'tests/matched_cadic_v5_1').rglob('*.py')]
    added+=[ROOT/NOTEBOOK,ROOT/'docs/main_spec/matched_cadic_v5_1.md',ROOT/'docs/main_spec/matched_cadic_v5_1_implementation.md']
    for p in added:
        name=p.relative_to(ROOT).as_posix()
        if name in payload:raise ValueError('New launcher overwrites frozen entry')
        payload[name]=p.read_bytes()
    if any(Path(n).suffix in ('.pt','.pth','.safetensors','.npz') for n in payload):raise ValueError('No learned tensors/data in source')
    files={n:hashlib.sha256(raw).hexdigest() for n,raw in sorted(payload.items())}
    m=dict(schema=SOURCE_SCHEMA,files=files,base_zip_sha256=BASE_SHA,base_entries=235,science_changed=False,job='CADIC_seed0')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,raw in sorted(payload.items()):z.writestr(n,raw)
        z.writestr('matched_cadic_manifest.json',json.dumps(m,sort_keys=True))
    receipt=dict(path=str(output),sha256=sha(output),files=len(files),source_digest=digest(files),frozen_base_byte_exact_entries=235,method_changed=False,canonical_run='not_run')
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
