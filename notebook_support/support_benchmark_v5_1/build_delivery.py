"""Immutable new delivery; preserve all144 accepted pilot payloads byte-for-byte."""
import ast,datetime,json,os,subprocess,zipfile
from pathlib import Path
from models.hope_cad_v5_1.contract import ROOT,digest,file_sha,design_digest
from training.support_benchmark_v5_1.contract import SOURCE_SCHEMA,source_files,study_digest,study
from notebook_support.support_v5_1.build_delivery import BOOT as PILOT_BOOT

def delivery_directory():return Path(os.environ.get('HOPE_BENCHMARK_DELIVERY_DIR',str(ROOT.parent/'.agents/artifacts/deliveries/hope_support_v5_1_benchmark_r1'))).resolve()

def notebook(phase,default_job):
    boot=PILOT_BOOT[PILOT_BOOT.index('from pathlib'):]
    boot=boot.replace('hope_support_v5_1_source','hope_support_v5_1_benchmark_source').replace('hope_support_v5_1_pilot','hope_support_v5_1_benchmark').replace('hope_cad_support_v5_1_pilot_evidence','hope_cad_support_v5_1_benchmark_evidence')
    boot=boot.replace('hope_support_v5_1_source_v1',SOURCE_SCHEMA).replace('HOPE-CAD v5','HOPE-CAD v5.1 benchmark')
    boot=f"PHASE={phase!r}\nJOB={default_job!r}\nSESSION_HOURS=9.0  # soft budget, không phải ETA\nSOURCE_OVERRIDE=None\nRESUME_OVERRIDE=None  # latest output cùng job, hoặc preflight PASS khi bắt đầu job mới\nMVTEC_OVERRIDE=None\nCHECKPOINT_OVERRIDE=None\n\n"+boot
    intro=f'''# Nested Learning for CAD — v5.1 benchmark {phase}

Đọc `docs/main_spec/benchmark_v5_1.md` và study JSON. Method đã qua normal pilot; **AD gain chưa được xác minh**. MVTec development, không VisA. Giữ source này cho mọi phases/jobs.

Ba inputs: source `hope_cad_support_v5_1_benchmark_source_r1.zip` (Kaggle có thể giải nén), MVTec AD và locked ViT-B/8 safetensors. Không attach pilot/v4/v5 learned weights. GPU T4×2; **Save Version → Save & Run All**. Không pip upgrade để bypass runtime lock.

Chạy `preflight` trước, giữ ZIP/log. Sau đó import notebook7.1 và thêm preflight PASS output. Job mặc định NESTED_seed0. CADIC_seed0 và FIRST_ORDER_seed0 chạy separate versions để matched quick view. **7.2** cần completed7.1 của đúng JOB, chỉ thêm105cells, không train. **7.3** một registered job mỗiversion, seed1/2 và các controls; defaultsNESTED_seed1. Fullstudy27jobs, không chỉ3NESTEDseeds. Mỗi outputZIP chỉ một job +commonlocks; khi quay lại job cũ phải attach latestoutput đúngjob đó. Tháo older/preflight outputs khi đã có latestjob output, tránh ambiguity.

`SESSION_HOURS=9.0` chỉ softpause với180s export reserve, không trainingbudget/ETA. Pause: attach latest benchmark output với cùngsource/assets/job/phase. Check handoff; `paused_at_atomic_boundary` chưa xong, `*_complete_pending_review` cần review. Technicalerror phải giữfailureZIP/log và stop; không sửa learning/scoring từtestmetrics. Heartbeat30s.

Giữ ZIP `hope_cad_support_v5_1_benchmark_evidence.zip` và log tại `.agents/artifacts/runs/support_v5_1_benchmark_<JOB>/session_01/` (preflight: `support_v5_1_benchmark_preflight_01`). Resume tăngsession. Không ghi đè evidence. Không hứa ETA; full3599normal learning có thể cần nhiều sessions.
'''
    run="from notebook_support.support_benchmark_v5_1.entry import run_notebook\nRESULT=run_notebook(dict(phase=PHASE,job=JOB,session_hours=SESSION_HOURS,started=STARTED,inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),RESUME_OVERRIDE=RESUME_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))\n"
    cells=[]
    for i,(typ,body) in enumerate([('markdown',intro),('code',boot),('code',run)]):
        c=dict(cell_type=typ,id=f'benchmark-v5-1-{phase}-{i}',metadata={},source=body.splitlines(True))
        if typ=='code':ast.parse(body);c.update(execution_count=None,outputs=[])
        cells.append(c)
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)

def build():
    for phase in ('preflight','7.1','7.2','7.3'):
        name='hope_cad_v5_1_benchmark_'+phase.replace('.','_')+'_kaggle.ipynb'
        (ROOT/'notebooks'/name).write_text(json.dumps(notebook(phase,'NESTED_seed1' if phase=='7.3' else 'NESTED_seed0'),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    baseline=json.loads((ROOT/'conf/benchmarks/support_v5_1_pilot_source.json').read_text())
    if digest(baseline)!=study()['pilot_source_digest']:raise ValueError('Pilot preservation map identity changed')
    for n,h in baseline.items():
        if file_sha(ROOT/n)!=h:raise ValueError('Accepted core/pilot bytes changed: '+n)
    files=source_files();out=delivery_directory();out.mkdir(parents=True,exist_ok=True)
    target=out/'hope_cad_support_v5_1_benchmark_source_r1.zip';receipt_path=out/'delivery_receipt.json'
    if target.exists():
        receipt=json.loads(receipt_path.read_text())
        with zipfile.ZipFile(target) as z:previous=json.loads(z.read('bundle_manifest.json'))
        if previous['files']!=files or file_sha(target)!=receipt['sha256']:raise ValueError('Immutable delivery already exists with different bytes; create new revision')
        return receipt
    origin=dict(branch='hope_cad_v6',source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),delivery_revision='support-v5.1-benchmark-r1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity_basis='all packaged SHA256s; no HEAD-only claim',pilot_payloads_preserved=len(baseline))
    manifest=dict(schema=SOURCE_SCHEMA,origin=origin,files=files)
    tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in files:z.write(ROOT/n,n)
        z.writestr('bundle_manifest.json',json.dumps(manifest,sort_keys=True))
    tmp.replace(target)
    value=dict(schema='support_v5_1_benchmark_delivery_v1',source_zip=str(target),sha256=file_sha(target),source_digest=digest(files),design_digest=design_digest(),study_digest=study_digest(),files=len(files),preserved_pilot_files=len(baseline),method_unchanged=True,benchmark_T4_status='pending',AD_performance='not_run')
    receipt_path.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2));return value
if __name__=='__main__':build()
