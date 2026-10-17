"""Canonical preflight and four-job finite normal pilot. No anomaly metrics/imports."""
from pathlib import Path
import argparse,json,time,gc,unittest
import torch
from models.hope_cad_v5.contract import CONTRACT,design_digest,canonical,digest,file_sha,ROOT,PILOT_JOBS
from models.hope_cad_v5.session import SupportSession
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from training.benchmark_config_v1 import load_configs
from dataset.benchmark_protocol_v2 import build_prefix_manifest
from training.hope_support_v5 import NormalFeatureCache,run_job,compare_jobs,lock,boundary,PilotPause
from training.runtime import atomic_json,resource_usage
from training.benchmark_engine_v1 import _state_equal
from models.cadic_paperfaithful.cache import reference_parity
from .common import configure_runtime,source_identity,runtime_identity,source_revision


def preflight(work,rows,provider,deadline):
    target=work/'preflight.json'
    if target.exists():
        if json.loads(target.read_text())['status']!='passed':raise ValueError('unreviewed failed preflight')
        return
    boundary(deadline);started=time.monotonic()
    tests=unittest.defaultTestLoader.discover(str(ROOT/'tests/hope_cad_v5'))
    result=unittest.TextTestRunner(verbosity=1).run(tests)
    if not result.wasSuccessful() or result.testsRun<16:raise RuntimeError('v5 local contracts failed/missing')
    torch.cuda.reset_peak_memory_stats(provider.device)
    actor=SupportSession('NESTED',device=provider.device,seed=0);middle=None
    observed=[]
    for i,row in enumerate(rows[:24]):
        boundary(deadline);actor.observe(provider.get(row),row['relative_path']);observed.append(row['relative_path'])
        if i==19:middle=actor.state_dict()
        if (i+1)%4==0:print('[V5 PREFLIGHT]',i+1,'/24 real normals',flush=True)
    if not actor.bank.is_full or actor.bank.count!=10000 or actor.learner.slow_updates<1:raise RuntimeError('canonical preflight did not reach full bank and real nested update')
    clone=SupportSession('NESTED',device=provider.device,seed=0);clone.load_state_dict(middle)
    for row in rows[20:24]:clone.observe(provider.get(row),row['relative_path'])
    if not _state_equal(actor.state_dict(),clone.state_dict()):raise RuntimeError('canonical GPU pending/partial-window resume parity failed')
    raw=provider.get(rows[23]);parity=reference_parity(actor.bank,raw[0,:4],raw[:,:7])
    last=actor.learner.last_slow
    if not last or not all(v>=0 and __import__('math').isfinite(v) for v in last['group_norms'].values()) or not any(v>0 for v in last['group_norms'].values()):raise RuntimeError('real slow meta gradients missing/nonfinite')
    out=dict(status='passed',canonical_v5=True,tests=result.testsRun,images=24,extra_resume_replay_images=4,extra_frozen_cache_read_for_parity=1,full_bank_count=actor.bank.count,nested_updates=actor.learner.slow_updates,last_slow=last,actual_full_budget_CADIC_parity=parity,pending_and_window_resume_exact=True,resource=resource_usage(provider.device),memory=actor.memory_profile(),elapsed_seconds=time.monotonic()-started,learned_clone_discarded=True,official_test_accessed=False,reportable=False)
    atomic_json(out,target);print('[V5 PREFLIGHT PASS]',json.dumps(out),flush=True)
    del actor,clone,middle;gc.collect();torch.cuda.empty_cache()


def run(config):
    configure_runtime();work=Path(config['work']);prepared=config['prepared'];device='cuda'
    protocol,method=load_configs(ROOT/'conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml',ROOT/'conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    # Manifest reads train/good only. No evaluator/test discovery object is constructed.
    manifest=build_prefix_manifest(protocol,seed=0,source_commit=source_revision())
    rows=manifest['online'][:CONTRACT['pilot']['first_online_images']]
    if len(rows)!=256:raise ValueError('pilot requires exactly first256 canonical online normals')
    identity=dict(schema='hope_support_pilot_identity_v5',design_digest=design_digest(),source_digest=source_identity(),runtime=runtime_identity(),manifest_digest=manifest['digest'],rows_digest=digest(rows),backbone_sha256=CONTRACT['backbone_sha256'],trial_budget=1)
    locked=lock(work/'pipeline_lock.json',identity)
    for name,value in [('manifest_train.json',manifest),('pilot_rows.json',rows),('protocol.json',CONTRACT)]:
        path=work/name
        if path.exists() and json.loads(path.read_text())!=canonical(value):raise ValueError('resume manifest/protocol bytes changed')
        atomic_json(value,path)
    deadline=config['started']+config['session_hours']*3600-180
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=prepared['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),device)
    provider=NormalFeatureCache(prepared['dataset'],extractor,Path(config['assets'])/'raw_cache',work/'feature_receipts',device)
    try:
        preflight(work,rows,provider,deadline)
        if config['phase']=='preflight':return 'preflight_passed'
        jobidentity=dict(pipeline_digest=locked['digest'],design_digest=design_digest(),scientific_scope='normal-only256images; not anomaly benchmark')
        for mode in PILOT_JOBS:
            boundary(deadline)
            run_job(rows,provider,mode,work/'jobs'/mode,jobidentity,deadline=deadline)
            print('[V5 JOB COMPLETE]',mode,flush=True);gc.collect();torch.cuda.empty_cache()
        result=compare_jobs(work);return result['status']
    finally:
        extractor.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();config=json.loads(Path(args.config).read_text());work=Path(config['work'])
    try:status=run(config);atomic_json(dict(status=status,reportable=False,anomaly_benchmark_authorized=False),work/'handoff.json')
    except PilotPause as e:atomic_json(dict(status='paused_at_image_boundary',reason=str(e),reportable=False),work/'handoff.json')
    except BaseException as e:atomic_json(dict(status='failed_preserve_evidence',error=repr(e),reportable=False),work/'handoff.json');raise
if __name__=='__main__':main()
