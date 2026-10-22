"""Canonical preflight and four-job finite normal pilot. No anomaly metrics/imports."""
from pathlib import Path
import argparse,json,time,gc,unittest
import torch
from models.hope_cad_v5_1.contract import CONTRACT,design_digest,canonical,digest,file_sha,ROOT,PILOT_JOBS,executable_contract
from models.hope_cad_v5_1.session import SupportSession
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from training.benchmark_config_v1 import load_configs
from dataset.benchmark_protocol_v2 import build_prefix_manifest
from training.hope_support_v5_1 import NormalFeatureCache,run_job,compare_jobs,lock,boundary,PilotPause
from training.runtime import atomic_json,resource_usage
from training.benchmark_engine_v1 import _state_equal
from models.cadic_paperfaithful.cache import reference_parity
from .common import configure_runtime,source_identity,runtime_identity,source_revision
from .numerical import canonical_meta_gate


def preflight(work,rows,provider,deadline):
    target=work/'preflight.json'
    if target.exists():
        if json.loads(target.read_text())['status']!='passed':raise ValueError('unreviewed failed preflight')
        return
    boundary(deadline);started=time.monotonic()
    tests=unittest.defaultTestLoader.discover(str(ROOT/'tests/hope_cad_v5_1'))
    result=unittest.TextTestRunner(verbosity=1).run(tests)
    if not result.wasSuccessful() or result.testsRun<29 or result.skipped:raise RuntimeError('v5.1 local contracts failed/missing/skipped')
    torch.cuda.reset_peak_memory_stats(provider.device)
    numerical=canonical_meta_gate(provider.device)
    actor=SupportSession('NESTED',device=provider.device,seed=0);middle=None
    observed=[]
    for i,row in enumerate(rows[:24]):
        boundary(deadline);actor.observe(provider.get(row),row['relative_path']);observed.append(row['relative_path'])
        if i==19:middle=actor.state_dict()
        if (i+1)%4==0:print('[V5.1 PREFLIGHT]',i+1,'/24 real normals',flush=True)
    if not actor.bank.is_full or actor.bank.count!=10000 or actor.learner.slow_updates<1:raise RuntimeError('canonical preflight did not reach full bank and real nested update')
    clone=SupportSession('NESTED',device=provider.device,seed=0);clone.load_state_dict(middle)
    for row in rows[20:24]:clone.observe(provider.get(row),row['relative_path'])
    if not _state_equal(actor.state_dict(),clone.state_dict()):raise RuntimeError('canonical GPU pending/partial-window resume parity failed')
    raw=provider.get(rows[23]);parity=reference_parity(actor.bank,raw[0,:4],raw[:,:7])
    last=actor.learner.last_slow
    if not last or not all(v>=0 and __import__('math').isfinite(v) for v in last['group_norms'].values()) or not any(v>0 for v in last['group_norms'].values()):raise RuntimeError('real slow meta gradients missing/nonfinite')
    out=dict(status='passed',canonical_v5_1=True,numerical=numerical,tests=result.testsRun,images=24,extra_resume_replay_images=4,extra_frozen_cache_read_for_parity=1,full_bank_count=actor.bank.count,nested_updates=actor.learner.slow_updates,last_slow=last,actual_full_budget_CADIC_parity=parity,pending_and_window_resume_exact=True,resource=resource_usage(provider.device),memory=actor.memory_profile(),elapsed_seconds=time.monotonic()-started,learned_clone_discarded=True,official_test_accessed=False,reportable=False)
    atomic_json(out,target);print('[V5.1 PREFLIGHT PASS]',json.dumps(out),flush=True)
    del actor,clone,middle;gc.collect();torch.cuda.empty_cache()


def run(config):
    configure_runtime();work=Path(config['work']);prepared=config['prepared'];device='cuda'
    protocol,method=load_configs(ROOT/'conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml',ROOT/'conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    # Manifest reads train/good only. No evaluator/test discovery object is constructed.
    manifest=build_prefix_manifest(protocol,seed=0,source_commit=source_revision())
    if len(manifest['online'])!=3599:raise ValueError('matched online partition must contain3599identities')
    rows=manifest['online'][:CONTRACT['pilot']['images_per_fresh_job']]
    if len(rows)!=256:raise ValueError('pilot requires exactly first256 canonical online normals')
    identity=dict(schema='hope_support_pilot_identity_v5_1',design_digest=design_digest(),source_digest=source_identity(),runtime=runtime_identity(),manifest_digest=manifest['digest'],rows_digest=digest(rows),backbone_sha256=CONTRACT['backbone_sha256'],trial_budget=1)
    locked=lock(work/'pipeline_lock.json',identity)
    rights=dict(offline_calibration=False,search=False,task_id_to_learner=False,one_image_causal_delivery=True,observed_projected_context_replay=True,official_test_accessed=False)
    for name,value in [('manifest_train.json',manifest),('pilot_rows.json',rows),('protocol.json',executable_contract()),('data_rights.json',rights)]:
        path=work/name
        if path.exists() and json.loads(path.read_text())!=canonical(value):raise ValueError('resume manifest/protocol bytes changed')
        atomic_json(value,path)
    deadline=config['started']+config['session_hours']*3600-180
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=prepared['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),device)
    provider=NormalFeatureCache(prepared['dataset'],extractor,Path(config['assets'])/'raw_cache',work/'feature_receipts',device)
    try:
        preflight(work,rows,provider,deadline)
        probes=[]
        for row in rows[:8]:
            raw=provider.get(row)
            from models.hope_cad_v2.state import tensor_digest
            probes.append(dict(identity=row['relative_path'],image_sha256=row['sha256'],feature_sha256=tensor_digest(({'patches':raw},))))
        probe_lock=lock(work/'probe_lock.json',dict(probes=probes))
        if config['phase']=='preflight':return 'preflight_passed'
        initials={}
        for mode in ('FIRST_ORDER','NESTED'):
            fresh=SupportSession(mode,device=device,seed=0)
            initials[mode]=fresh.initialization_hashes();del fresh
        if initials['FIRST_ORDER']!=initials['NESTED']:raise ValueError('initial theta/phi/P/private RNG are not matched')
        lock(work/'matched_initialization.json',initials)
        jobidentity=dict(pipeline_digest=locked['digest'],design_digest=design_digest(),probe_manifest_digest=probe_lock['digest'],scientific_scope='normal-only256images; not anomaly benchmark')
        for mode in PILOT_JOBS:
            boundary(deadline)
            run_job(rows,provider,mode,work/'jobs'/mode,jobidentity,deadline=deadline)
            print('[V5.1 JOB COMPLETE]',mode,flush=True);gc.collect();torch.cuda.empty_cache()
        result=compare_jobs(work);return result['status']
    finally:
        extractor.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();config=json.loads(Path(args.config).read_text());work=Path(config['work'])
    try:status=run(config);atomic_json(dict(status=status,reportable=False,anomaly_benchmark_authorized=False),work/'handoff.json')
    except PilotPause as e:atomic_json(dict(status='paused_at_image_boundary',reason=str(e),reportable=False),work/'handoff.json')
    except BaseException as e:atomic_json(dict(status='failed_preserve_evidence',error=repr(e),reportable=False),work/'handoff.json');raise
if __name__=='__main__':main()
