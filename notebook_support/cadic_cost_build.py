"""Build an additive CADIC timing source; original G3 source stays byte-exact."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(destination):
    shared = ROOT.parent/'.agents'
    base = shared/'artifacts/deliveries/hope_support_v5_1_nn_write_r1/hope_cad_support_v5_1_nn_write_source_r1.zip'
    if sha(base) != '9e401eb432003add4025125c5c89d48d11183b8e53f157514f15bf0a2cba3800':
        raise ValueError('Wrong G3 base source')
    with zipfile.ZipFile(base) as z:
        payload = {n:z.read(n) for n in z.namelist()}
    original = json.loads(payload['nn_write_manifest.json'])
    for n,h in original['files'].items():
        if hashlib.sha256(payload[n]).hexdigest()!=h or (n not in ('bundle_manifest.json','throughput_manifest.json') and sha(ROOT/n)!=h):
            raise ValueError('Frozen G3 source drift: '+n)
    evidence = shared/'artifacts/runs/support_v5_1_nn_write_validation_01/session_01/hope_cad_support_v5_1_nn_write_evidence.zip'
    if sha(evidence) != 'c4490b3091523d6a5b6b2f6b0d6569e71d0cb8163be45aa48292d7cfdde94587':
        raise ValueError('Wrong historical G3 evidence')
    with zipfile.ZipFile(evidence) as z:
        summary = json.loads(z.read('summary.json'))
        lock = json.loads(z.read('nn_write_lock.json'))
        times = summary['times']
        events = [json.loads(z.read('passes/R1/events/'+f'{i:05d}.json')) for i in range(179,195)]
        reference = dict(reference_mean_seconds=(times['R1']+times['R2'])/2,
            optimized_mean_seconds=(times['O1']+times['O2'])/2, times=times,
            evidence_sha256=sha(evidence), runtime=lock['runtime'],
            summary_sha256=hashlib.sha256(z.read('summary.json')).hexdigest(),
            measurement_sha256={n:hashlib.sha256(z.read('passes/'+n+'/measurement.json')).hexdigest() for n in times},
            review='support-v5-1-nn-write-session1-20261021', window=[179,195],
            source_sha256=sha(base), paired_with_future_CADIC=False)
        reference['work_counters'] = dict(proposal_replacements=sum(e['proposal']['replaced'] for e in events),
            committed_replacements=sum(e['proposal']['replaced'] for e in events if e['accepted']),
            accepted=sum(e['accepted'] for e in events), rejected=sum(not e['accepted'] for e in events))
    ref = ROOT/'notebook_support/cadic_cost_nested_reference.json'
    ref.write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    nb = json.loads(payload['notebooks/hope_cad_v5_1_nn_write_validation_kaggle.ipynb'])
    boot = ''.join(nb['cells'][1]['source'])
    for a,b in {
        "PHASE='validate'":"PHASE='cadic_cost'",
        'nn_write_manifest.json':'cadic_cost_manifest.json',
        'hope_support_v5_1_nn_write_source_v1':'hope_cadic_cost_source_v1',
        '/tmp/hope_nn_write_source_r1':'/tmp/hope_cadic_cost_source_r1',
        '/kaggle/working/hope_nn_write_validation':'/kaggle/working/hope_cadic_cost',
        'hope_cad_support_v5_1_nn_write_evidence.zip':'hope_cad_cadic_cost_evidence.zip',
        'latest profiler output only; first run None':'unsupported: preserve failures for review; keep None',
        '[NN BOOTSTRAP]':'[CADIC COST BOOTSTRAP]',
        'Expected ONE profiler source identity':'Expected ONE CADIC cost source identity',
        'Fixed90min; R1/O1/O2/R2 on16known normals; exact gradient diagnostics; no benchmark adoption':
            'Soft180min including export reserve; PREP179 separately then CADIC R1/O1/O2/R2 on16 known normals; no anomaly evaluation or adoption'
    }.items(): boot=boot.replace(a,b)
    intro='''# Nested Learning for CAD — CADIC gốc/tối ưu, đối chiếu chi phí T4

Đọc `docs/main_spec/cadic_cost_validation.md`. Không thay method/protocol, không anomaly test hay VisA, không tiếp tục benchmark. CADIC dựng predecessor riêng từ179 normals rồi R1/O1/O2/R2 trên16 ảnh179–194; PREP báo riêng. So historical NESTED là cross-session descriptive, bank trajectories khác; không là whole-stream ETA hay accuracy gain.

**4 inputs:** source `hope_cad_cadic_cost_source_r1.zip`; original reviewed NESTED_seed0 paused376 evidence; MVTec AD raw; locked ViT-B/8 safetensors. Tháo source/output profiler, NN-validation và bridge khác. Không cần attach G3 output, các số tham chiếu đã pin SHA trong source.

**GPU T4×2, kernel mới, Save Version → Save & Run All.** `RESUME_OVERRIDE=None`; soft180min gồm180s export reserve, không ETA. Không tự retry khi partial/fail. Runtime phải khớp canonical T4; không pip upgrade. Một GPU cuda:0 được sử dụng. Heartbeat subprocess30s; progress mỗi8ảnh. Chi phí179ảnh chuẩn bị có thể đáng kể.

Lưu ZIP `hope_cad_cadic_cost_evidence.zip` và log tại `D:/src/DATN/.agents/artifacts/runs/support_v5_1_cadic_cost_01/session_01/`. Complete cần review source/runtime/parity/timing. Không thay output bridge hoặc resume benchmark bằng clone này.
'''
    run='''from notebook_support.cadic_cost_validation import run_notebook
RESULT=run_notebook(dict(inputs=str(INPUT),work=str(WORK),assets=str(ASSETS),output=str(OUTPUT),started=STARTED,
    PARENT_OVERRIDE=PARENT_OVERRIDE,RESUME_OVERRIDE=RESUME_OVERRIDE,MVTEC_OVERRIDE=MVTEC_OVERRIDE,CHECKPOINT_OVERRIDE=CHECKPOINT_OVERRIDE))
'''
    for cell,text in zip(nb['cells'],(intro,boot,run)):
        cell['source']=text.splitlines(keepends=True)
        if cell['cell_type']=='code':
            compile(text,'cadic_cost_notebook','exec');cell['outputs']=[];cell['execution_count']=None
    notebook='notebooks/hope_cad_v5_1_cadic_cost_kaggle.ipynb'
    (ROOT/notebook).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    names=['training/cadic_cost_validation.py','notebook_support/cadic_cost_validation.py',
        'notebook_support/cadic_cost_build.py','notebook_support/cadic_cost_nested_reference.json',
        'docs/main_spec/cadic_cost_validation.md','tests/test_cadic_cost_validation.py',notebook]
    for n in names:
        if n in payload:raise ValueError('Refuse overwrite frozen source')
        payload[n]=(ROOT/n).read_bytes()
    files={n:hashlib.sha256(v).hexdigest() for n,v in sorted(payload.items())}
    manifest=dict(schema='hope_cadic_cost_source_v1',files=files,
        contract_sha256=original['contract_sha256'],
        origin=dict(delivery='cadic-cost-r1',base_G3_source_sha256=sha(base),
            method_changed=False,protocol_changed=False,canonical_run='not_run'))
    output=Path(destination);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,v in payload.items():z.writestr(n,v)
        z.writestr('cadic_cost_manifest.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(output) as z:
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
    receipt=dict(path=str(output),sha256=sha(output),files=len(files),frozen_G3_files=200,
        source_digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        canonical_T4='not_run',method_changed=False,protocol_changed=False)
    (output.parent/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    build(p.parse_args().output)
