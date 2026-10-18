"""Engineering lifecycle, Execution Lock, one-shot complete matrix, no retries."""
import argparse
import json
import shutil
import time
import traceback
import torch
from ..design import ROOT,sha,save
from .design import OUT,ORIGINAL,PACKAGE,MATRIX,source_hashes,validate_execution,budget_elapsed
from .runner import run,configure
from .analysis import summarize,known_signs
from .verify import verify_stage


def dryrun():
    assert json.loads((OUT/'extension_tests.json').read_text())['passed']
    assert json.loads((OUT/'profile.json').read_text())['passed']
    assert json.loads((OUT/'single_clip_check.json').read_text())['passed']
    save(OUT/'dryrun_started.json',dict(utc=time.time(),seed=1901,G3_untouched=True))
    rows=[]; failure=None
    try:
        for regime,arms in MATRIX.items():
            for arm in arms: rows.append(run(1901,regime,arm))
        assert known_signs()
        for regime in ['D8','D16']:
            # Verify full128-cycle port against original engineering evidence, not just tiny tests.
            old=torch.load(ORIGINAL/'engineering'/f'{regime}_1901_candidate'/'state.pt',weights_only=False,map_location='cpu')
            new=torch.load(OUT/'dryrun'/f'{regime}_1901_candidate'/'final.pt',weights_only=False,map_location='cpu')
            for n in old['live']['phi']: assert torch.equal(old['live']['phi'][n],new['live']['phi'][n])
            for m in ['M4','M16']:
                for n in old[m]: assert torch.equal(old[m][n],new[m][n]),('fullshape port',regime,m,n)
        verification=verify_stage('dryrun',[1901])
    except Exception:
        failure=traceback.format_exc(); verification=None
    save(OUT/'dryrun.json',dict(passed=failure is None,failure=failure,
        lifecycle=['prepare','train','witness_evaluate','analyze','verify'],
        completed_runs=len(rows),verification=verification,known_signs=True,
        analysis_scope='n1 engineering noninferential; known-values statistics tested separately',
        G3_untouched=True))
    if failure: print(failure,flush=True); raise SystemExit(1)


def execution_lock():
    assert json.loads((OUT/'extension_tests.json').read_text())['passed']
    assert json.loads((OUT/'control_tests.json').read_text())['passed']
    assert json.loads((OUT/'single_clip_check.json').read_text())['passed']
    assert json.loads((OUT/'profile.json').read_text())['passed']
    assert json.loads((OUT/'dryrun.json').read_text())['passed']
    reuse=json.loads((OUT/'reusecheck_validation.json').read_text()); assert reuse['passed']
    assert json.loads((ORIGINAL/'G1b.json').read_text())['passed']
    assert not (OUT/'evaluation_started.json').exists()
    assert known_signs()
    rows=[json.loads(p.read_text()) for p in (OUT/'dryrun').glob('*/trace.json')]
    assert len(rows)==sum(map(len,MATRIX.values()))
    projected=budget_elapsed()+12*reuse['projected_seconds_per_seed']
    storage=sum(p.stat().st_size for p in (OUT/'dryrun').rglob('*') if p.is_file())
    free=shutil.disk_usage(OUT).free
    assert projected<=21600,('RESOURCE_INFEASIBLE: projected budget',projected)
    assert reuse['projected_storage_bytes']+25*1024**3<free,('RESOURCE_INFEASIBLE: projected disk',reuse['projected_storage_bytes'],free)
    lock=dict(schema='p2h_x_EL_v1',passed=True,utc=time.time(),G3_untouched_at_lock=True,
        protocol_sha256=sha(PACKAGE/'PROTOCOL.md'),design_lock_sha256=sha(OUT/'design_lock.json'),
        source_sha256=source_hashes(),matrix=MATRIX,seeds=list(range(3101,3111)),
        numerical_tolerances=dict(eps_abs=1e-7,eps_rel=1e-5,FD_relative=1e-3,
            random_norm_relative=1e-5,witness_CPU_absolute=1e-7,witness_CPU_relative=1e-5),
        budget_projection_seconds=projected,consumed_seconds_before_lock=budget_elapsed(),
        storage_projection_bytes=reuse['projected_storage_bytes'],free_disk_bytes=free,
        resource_profile_sha256=sha(OUT/'profile.json'),dryrun_sha256=sha(OUT/'dryrun.json'),
        statistics='positive reference-candidate, paired t95%, fixed12 secondary Holm family')
    lock.update(prefix_reuse_sha256=sha(PACKAGE/'PREFIX_REUSE.md'),reusecheck_sha256=sha(OUT/'reusecheck.json'),
        reusecheck_validation_sha256=sha(OUT/'reusecheck_validation.json'),
        storage_projection_convention=reuse['storage_convention'],
        reuse_savings_seconds=reuse['conservative_savings_per_arm'],
        projection_multiplier=12.,projection_convention='10 seeds plus20% headroom; all prior extension CUDA time')
    save(OUT/'execution_lock.json',lock)
    print(json.dumps(lock),flush=True)


def reusecheck():
    from .prefix import path as prefix_path
    from .data import SyntheticStream,Replay,partition
    from ..core import Observation
    assert json.loads((OUT/'dryrun.json').read_text())['passed']
    save(OUT/'reusecheck_started.json',dict(utc=time.time(),seed=1901,G3_untouched=True))
    failure=None; completed=[]; savings={}
    try:
        for arm in MATRIX['D8']:
            run(1901,'D8',arm,stage='reusecheck',cycles=64,reuse_prefix=True)
        for arm in MATRIX['D16']:
            new=run(1901,'D16',arm,stage='reusecheck',reuse_prefix=True)
            old=json.loads((OUT/'dryrun'/f'D16_1901_{arm}'/'trace.json').read_text())
            assert [s['live_sha256'] for s in new['traces']]==[s['live_sha256'] for s in old['traces']],('prefix port',arm)
            assert new['causal_replay']==old['causal_replay']
            assert new['acquisition']==old['acquisition'] and new['retained']==old['retained'] and new['new_normal']==old['new_normal']
            assert new['meter']==old['meter']
            old_state=torch.load(OUT/'dryrun'/f'D16_1901_{arm}'/'final.pt',weights_only=False,map_location='cpu')
            new_state=torch.load(OUT/'reusecheck'/f'D16_1901_{arm}'/'final.pt',weights_only=False,map_location='cpu')
            for m in ['M4','M16']:
                assert old_state[m].keys()==new_state[m].keys()
                assert all(torch.equal(old_state[m][n],new_state[m][n]) for n in old_state[m])
            prefix=torch.load(prefix_path('reusecheck',1901,arm),weights_only=False,map_location='cpu')
            savings[arm]=max(0.,min(prefix['prefix_seconds'],old['seconds']-new['seconds']))
            completed.append(arm)
        saved=torch.load(prefix_path('reusecheck',1901,'candidate'),weights_only=False,map_location='cpu')
        generator_checks={}
        for regime in MATRIX:
            saved=torch.load(prefix_path('reusecheck',1901,'candidate'),weights_only=False,map_location='cpu')
            fresh=SyntheticStream(1901,regime); replay=Replay(1901)
            restored=SyntheticStream.restore(saved['stream']); restored.regime=regime
            restored_replay=Replay.restore(saved['replay'])
            for event in range(32):
                x=fresh.arrive(event)
                for local in [0,1]: replay.sample(event)
                replay.insert(Observation(event,x,partition(event,1,784)[1]))
            assert torch.equal(replay.rng.get_state(),restored_replay.rng.get_state())
            for event in range(32,80 if regime=='R' else 64):
                a=fresh.arrive(event); b=restored.arrive(event); assert torch.equal(a,b)
                for local in [0,1]:
                    x=replay.sample(event); y=restored_replay.sample(event)
                    assert x.identity==y.identity and torch.equal(x.features,y.features)
                replay.insert(Observation(event,a,partition(event,1,784)[1]))
                restored_replay.insert(Observation(event,b,partition(event,1,784)[1]))
            generator_checks[regime]=True
        dry=[json.loads(p.read_text()) for p in (OUT/'dryrun').glob('*/trace.json')]
        projected=sum(r['seconds']-(savings[r['arm']] if r['regime']!='D8' else 0.) for r in dry)
    except Exception:
        failure=traceback.format_exc(); projected=None; generator_checks=None
    save(OUT/'reusecheck.json',dict(passed=failure is None,failure=failure,
        completed_arms=completed,generator_checks=generator_checks,conservative_savings_per_arm=savings,
        projected_seconds_per_seed=projected,G3_untouched=True))
    if failure: print(failure,flush=True); raise SystemExit(1)


def evaluate():
    validate_execution()
    save(OUT/'evaluation_started.json',dict(utc=time.time(),seeds=list(range(3101,3111)),one_shot=True))
    failure=None; completed=[]
    try:
        for seed in range(3101,3111):
            for regime,arms in MATRIX.items():
                for arm in arms:
                    row=run(seed,regime,arm,stage='evaluation',reuse_prefix=True); completed.append((seed,regime,arm))
                    print('MATRIX_PROGRESS',len(completed),'/590',flush=True)
        analysis=summarize(OUT/'evaluation',list(range(3101,3111)))
        assert analysis['complete']
        save(OUT/'analysis.json',analysis)
        verification=verify_stage('evaluation',list(range(3101,3111)))
        save(OUT/'verification.json',verification)
    except Exception:
        failure=traceback.format_exc(); print(failure,flush=True)
    save(OUT/'evaluation_finished.json',dict(failure=failure,completed=completed,
        completed_runs=len(completed),budget_elapsed=budget_elapsed(),
        verdict=('RESOURCE_INFEASIBLE' if failure and 'RESOURCE_INFEASIBLE' in failure else 'IMPLEMENTATION_INVALID') if failure else analysis['verdict']))
    if failure: raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['dryrun','reusecheck','lock','evaluate'])
    args=parser.parse_args(); configure()
    {'dryrun':dryrun,'reusecheck':reusecheck,'lock':execution_lock,'evaluate':evaluate}[args.stage]()


if __name__=='__main__': main()
