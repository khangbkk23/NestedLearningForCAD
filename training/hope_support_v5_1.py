"""Finite normal-only v5 pilot and content-locked job resume. No test discovery."""
from pathlib import Path
import json,time,math,gc,torch
from models.hope_cad_v5_1.contract import CONTRACT,DESIGN_ID,design_digest,digest,canonical,file_sha,PILOT_JOBS
from models.hope_cad_v5_1.session import SupportSession
from models.hope_cad_v5_1.geometry import risk
from models.hope_cad_v2.state import tensor_digest,finite_tensors
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from training.runtime import atomic_json,atomic_torch_save,resource_usage
from training.benchmark_engine_v1 import _state_equal

class PilotPause(Exception):pass

def boundary(deadline):
    if deadline is not None and time.monotonic()>=deadline:raise PilotPause('saved image boundary; attach latest v5 pilot output to resume')

def lock(path,identity):
    from training.runtime import lock_json
    return lock_json(path,canonical(identity))

class NormalFeatureCache:
    def __init__(self,root,extractor,folder,receipts,device):
        self.root,self.extractor,self.folder,self.receipts,self.device=Path(root),extractor,Path(folder),Path(receipts),device;self.reads=[]
    def get(self,row):
        if row['split']!='train' or '/train/good/' not in row['relative_path']:raise ValueError('forbidden pilot feature identity')
        image=HashedNormalDataset(self.root,[row],224,[.485,.456,.406],[.229,.224,.225],True)[0]['images'].unsqueeze(0)
        key=digest(dict(row=row,backbone=CONTRACT['backbone_sha256'],producer=file_sha(Path(__file__).resolve().parents[1]/'models/feature_extractors/cadic_vit_v1.py')))
        path=self.folder/(key+'.pt');receipt=self.receipts/(key+'.json')
        if path.exists():
            value=torch.load(path,map_location='cpu',weights_only=True)
            if not receipt.exists() or json.loads(receipt.read_text())['tensor_sha256']!=tensor_digest(({'patches':value},)):raise ValueError('raw feature cache mismatch')
        else:
            with torch.no_grad():value=self.extractor.extract_patch_features(image.to(self.device)).detach().cpu().contiguous()
            info=dict(row=row,backbone_sha256=CONTRACT['backbone_sha256'],tensor_sha256=tensor_digest(({'patches':value},)))
            if receipt.exists() and json.loads(receipt.read_text())!=info:raise ValueError('regenerated raw features differ')
            atomic_json(info,receipt);atomic_torch_save(value,path)
        if value.shape!=(1,784,768) or value.dtype!=torch.float32 or value.requires_grad:raise ValueError('canonical raw feature geometry violated')
        expected=dict(row=row,backbone_sha256=CONTRACT['backbone_sha256'],tensor_sha256=tensor_digest(({'patches':value},)))
        if json.loads(receipt.read_text())!=expected:raise ValueError('raw cache row/backbone/tensor receipt changed')
        finite_tensors((value,),'raw cached patches');self.reads.append(row['relative_path']);return value.to(self.device)

def run_job(rows,provider,mode,folder,identity,*,deadline=None,session_factory=None,checkpoint_every=8):
    if mode not in PILOT_JOBS:raise ValueError('unregistered normal pilot job')
    if any(row.get('split')!='train' or '/train/good/' not in row.get('relative_path','') for row in rows):raise ValueError('pilot driver accepts train/good normals only')
    if len({row['relative_path'] for row in rows})!=len(rows):raise ValueError('duplicate pilot identities')
    if session_factory is None and len(rows)!=256:raise ValueError('canonical pilot requires exactly256images')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);active=folder/'state.pt';completion=folder/'completion.json'
    locked=lock(folder/'job_lock.json',dict(**identity,mode=mode,rows=rows))
    if completion.exists():
        result=json.loads(completion.read_text())
        if result['job_digest']!=locked['digest'] or file_sha(active)!=result['state_sha256'] or file_sha(folder/'normal_score_probes.json')!=result['probe_sha256']:raise ValueError('completed pilot checkpoint/probes changed')
        return result
    session=session_factory(mode) if session_factory else SupportSession(mode,device=provider.device,seed=0)
    initial=session.initialization_hashes()
    lock(folder/'initialization.json',initial)
    elapsed=0.;peak_bytes=0
    if active.exists():
        old=torch.load(active,map_location='cpu',weights_only=False)
        if old['job_digest']!=locked['digest']:raise ValueError('pilot checkpoint lock changed')
        session.load_state_dict(old['session']);elapsed=old['elapsed_seconds'];peak_bytes=old['peak_persistent_tensor_bytes']
    started=time.monotonic()
    def save():
        nonlocal peak_bytes
        profile=session.memory_profile();peak_bytes=max(peak_bytes,profile['persistent_tensor_bytes'])
        atomic_torch_save(dict(job_digest=locked['digest'],session=session.state_dict(),elapsed_seconds=elapsed+time.monotonic()-started,peak_persistent_tensor_bytes=peak_bytes),active)
    if not active.exists():save()
    for i in range(session.cursor,len(rows)):
        try:boundary(deadline)
        except PilotPause:save();raise
        row=rows[i];raw=provider.get(row);record=session.observe(raw,row['relative_path']);target=folder/'events'/f'{i:04d}.json'
        if target.exists() and json.loads(target.read_text())!=canonical(record):raise ValueError('replayed event differs from pre-crash trace')
        atomic_json(record,target)
        peak_bytes=max(peak_bytes,session.memory_profile()['persistent_tensor_bytes'])
        if session.cursor%checkpoint_every==0:save()
        if session.cursor%8==0:print('[PILOT]',mode,session.cursor,'/',len(rows),'labels',session.counters['labels_completed'],'slow',session.learner.slow_updates if session.learner else 0,flush=True)
    session.finalize_stream();save()
    if session.learner and session.learner.saved_meta_episode is not None:
        atomic_torch_save(session.learner.saved_meta_episode,folder/'meta_episode_33_40.pt')
    probes=[]
    # Read-only, reused past normals: solely verify that policy can affect actual scores.
    before=session.state_dict()
    for row in rows[:8]:
        raw=provider.get(row);pixel,_,image=session.score(raw)
        probes.append(dict(identity=row['relative_path'],image_sha256=row['sha256'],
            feature_sha256=tensor_digest(({'patches':raw},)),pixel=pixel.cpu().tolist(),image=image.cpu().tolist()))
    if not _state_equal(before,session.state_dict()):raise RuntimeError('probe evaluation mutated session')
    atomic_json(probes,folder/'normal_score_probes.json')
    events=[json.loads((folder/'events'/f'{i:04d}.json').read_text()) for i in range(len(rows))]
    risks=[v['prequential_risk'] for v in events if v['prequential_risk'] is not None]
    result=dict(status='complete_pending_review',mode=mode,job_digest=locked['digest'],state_sha256=file_sha(active),images=session.cursor,counters=session.counters,
        slow_updates=session.learner.slow_updates if session.learner else 0,prequential_mean_risk=sum(risks)/len(risks),last128_mean_risk=sum(risks[-128:])/len(risks[-128:]),
        final_bank_sha256=tensor_digest(({'features':session.bank.features},)),initialization=initial,
        probe_sha256=file_sha(folder/'normal_score_probes.json'),memory=session.memory_profile(),resource=resource_usage(provider.device),peak_persistent_tensor_bytes=peak_bytes,elapsed_seconds=elapsed+time.monotonic()-started,official_test_accessed=False,probe_is_independent=False,timing_scope='includes raw-cache reads/extraction and checkpoint IO; unequal cache warming across jobs, not a fair compute benchmark',reportable=False)
    atomic_json(result,completion);return result

def compare_jobs(work,*,canonical_run=True):
    work=Path(work);results={m:json.loads((work/'jobs'/m/'completion.json').read_text()) for m in PILOT_JOBS}
    nested,base=results['NESTED'],results['CADIC'];n=nested['counters']['learned_decisions'];rejected=nested['counters']['rejected_learned']/n if n else 0.
    probes={m:json.loads((work/'jobs'/m/'normal_score_probes.json').read_text()) for m in PILOT_JOBS}
    if results['FIRST_ORDER']['initialization']!=nested['initialization']:raise ValueError('learned initialization hashes differ')
    expected_rows=json.loads((work/'pilot_rows.json').read_text())[:8]
    probe_lock=json.loads((work/'probe_lock.json').read_text())['identity']
    expected_manifest=[]
    for mode in PILOT_JOBS:
        values=probes[mode]
        if len(values)!=8:raise ValueError('exactly eight fixed probes required')
        if file_sha(work/'jobs'/mode/'normal_score_probes.json')!=results[mode]['probe_sha256']:raise ValueError('probe file changed')
        manifest=[]
        for i,value in enumerate(values):
            if value['identity']!=expected_rows[i]['relative_path'] or value['image_sha256']!=expected_rows[i]['sha256']:raise ValueError('ordered probe identity changed')
            pixels=torch.tensor(value['pixel'],dtype=torch.float64)
            if pixels.ndim!=2 or pixels.shape[0]!=1 or (canonical_run and pixels.shape[1]!=784) or not bool(torch.isfinite(pixels).all()):raise ValueError('invalid raw patch probe geometry')
            manifest.append({k:value[k] for k in ('identity','image_sha256','feature_sha256')})
        if not expected_manifest:expected_manifest=manifest
        if manifest!=expected_manifest or manifest!=probe_lock['probes']:raise ValueError('probe manifests/features differ across controls')
    a=torch.tensor([v['pixel'][0] for v in probes['NESTED']],dtype=torch.float64)
    b=torch.tensor([v['pixel'][0] for v in probes['CADIC']],dtype=torch.float64)
    if a.shape!=b.shape:raise ValueError('probe lengths must match, no truncation')
    delta=float((a-b).abs().max())
    gates=dict(completed_labels=nested['counters']['labels_completed']>=64,slow_updates=nested['slow_updates']>=8,nontrivial_admission=.01<=rejected<=.99,
        bank_differs=nested['final_bank_sha256']!=base['final_bank_sha256'],score_differs=delta>1e-6,
        bounded_prequential_risk=nested['last128_mean_risk']<=1.1*base['last128_mean_risk'])
    result=dict(status='pilot_complete_pending_review' if all(gates.values()) else 'pilot_complete_scientific_gate_failed',gates=gates,results=results,reject_fraction_postwarmup=rejected,max_abs_reused_normal_patch_score_difference=delta,last128_risk_ratio=nested['last128_mean_risk']/max(base['last128_mean_risk'],1e-12),claim='normal development pilot only; no AD superiority or automatic benchmark authorization',reportable=False)
    atomic_json(result,work/'pilot_review_inputs.json');return result
