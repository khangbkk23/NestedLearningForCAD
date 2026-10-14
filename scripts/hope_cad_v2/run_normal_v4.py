"""Resumable normal-only W1-W6 handoff; never launches W7 or anomaly-test evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from dataset.benchmark_protocol_v2 import file_hash
from training.hope_runtime_v4 import configure_runtime,atomic_json,source_identity,runtime_identity
from training.hope_feature_contract_v3 import load_context,validate_bundle


class SessionBudgetReached(RuntimeError):
    """Operational pause, not a failed scientific candidate."""


def session_deadline(hours, started_monotonic=None):
    now = time.monotonic()
    start = now if started_monotonic is None else started_monotonic
    if not math.isfinite(hours) or hours <= 0 or not math.isfinite(start) or start > now:
        raise ValueError('session hours must be finite/positive and start must not be in the future')
    return start + hours * 3600


def notebook_outcome(exit_code, summary, *, preflight_only=False):
    """A planned pause is a saved incomplete run; failures remain failures."""
    status = summary.get('status')
    if exit_code == 0 and status == 'paused_for_session_limit':
        return 'PAUSED: checkpoints exported. Attach this saved output to resume; W6 is incomplete.'
    expected = 'preflight_passed' if preflight_only else 'ready_for_artifact_review'
    if exit_code == 0 and status == expected and summary.get('engineering_gates_passed'):
        return ('PREFLIGHT PASSED: inspect the saved output, then attach it to the main notebook.'
                if preflight_only else 'READY FOR ARTIFACT REVIEW: W6 scientific acceptance is still pending.')
    raise RuntimeError(f'Run has not passed ({status}, exit={exit_code}). Inspect the saved failure evidence.')


def export_evidence(root):
    root=Path(root)
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and p.suffix!='.tmp')
    manifest=dict(schema='hope_normal_evidence_v4',files={p.relative_to(root).as_posix():file_hash(p) for p in files})
    atomic_json(manifest,root/'artifact_manifest.json')
    target=root.parent/'hope_cad_v4_normal_evidence.zip'
    temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in [*files,root/'artifact_manifest.json']:
            archive.write(path,path.relative_to(root).as_posix())
    temporary.replace(target)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--feature-cache',type=Path)
    p.add_argument('--parent-evidence',type=Path,
                   help='optional v3 evidence used ONLY for verified frozen calibration features, never learned state')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true',
                   help='stop after correctness/resource/cache/RGB gates; never start calibration')
    p.add_argument('--session-hours',type=float,default=10.5,
                   help='pause and export before the GPU session expires; does not change the recipe')
    p.add_argument('--session-start-monotonic',type=float,
                   help='same-host monotonic clock from the first notebook code cell; includes bootstrap time')
    p.add_argument('--methods',nargs='+',default=['M','C0','C1','C2','C3','C4','A1'])
    p.add_argument('--protocol',default='conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml')
    p.add_argument('--baseline-method',default='conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    args=p.parse_args()
    try:
        deadline=session_deadline(args.session_hours,args.session_start_monotonic)
    except ValueError as error:
        p.error(str(error))
    configure_runtime()
    if args.methods != [m for m in ('M','C0','C1','C2','C3','C4','A1') if m in args.methods] or not set(('M','C0','C1','C2'))<=set(args.methods):
        p.error('methods must include M C0 C1 C2, then optional C3 C4 A1 in registered order')
    root=args.work_dir.resolve(); root.mkdir(parents=True,exist_ok=True)
    lock_path=root/'pipeline_lock.json'
    lock=dict(design_id='hope_cad_v4_warm_joint_cms_row_retention',
              selection_rule_id='prefix_warm_joint_stress_v4',
              source_sha256=source_identity(),runtime=runtime_identity(),methods=args.methods,
              protocol_path=args.protocol,baseline_method_path=args.baseline_method)
    if lock_path.exists():
        if not args.resume or json.loads(lock_path.read_text())!=lock:
            raise ValueError('use --resume with identical source/runtime/methods, or a new work directory')
    elif any(root.iterdir()):
        raise ValueError('refuse an unlocked nonempty work directory')
    else: atomic_json(lock,lock_path)
    log_path=root/'commands.json'
    commands=json.loads(log_path.read_text()) if log_path.exists() else []
    summary=dict(status='running',reportable=False,W7_authorized=False,review_required=True,
                 design_id='hope_cad_v4_warm_joint_cms_row_retention',
                 normal_probe_status='development_exposed_not_independent_confirmation',
                 official_test_accessed=False,methods=args.methods,completed=[],
                 session_hours=args.session_hours,
                 budget_includes_notebook_bootstrap=args.session_start_monotonic is not None)
    common=['--protocol',args.protocol,'--baseline-method',args.baseline_method,'--lock-dir',str(root/'baseline_lock')]
    def run(name,*arguments):
        if time.monotonic() >= deadline:
            raise SessionBudgetReached('session budget reached before the next command')
        cmd=[sys.executable,'-B',f'scripts/hope_cad_v2/{name}',*map(str,arguments)]
        print(' '.join(cmd),flush=True)
        started=time.time()
        log=root/'logs'/f'{len(commands):03d}_{Path(name).stem}.log'; log.parent.mkdir(exist_ok=True)
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            def relay():
                for line in process.stdout:
                    print(line,end='',flush=True); stream.write(line); stream.flush()
            reader=threading.Thread(target=relay,daemon=True)
            reader.start()
            paused=False
            try:
                result=process.wait(timeout=max(0.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                paused=True
                process.terminate()
                try: result=process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); result=process.wait()
            except BaseException:
                process.terminate(); process.wait()
                raise
            finally:
                reader.join()
                process.stdout.close()
        commands.append(dict(command=cmd,exit_code=result,session_budget_pause=paused,started_unix=started,elapsed_seconds=time.time()-started,log=log.relative_to(root).as_posix()))
        atomic_json(commands,log_path)
        if paused:
            raise SessionBudgetReached(f'{name} paused; resume from the last atomic checkpoint with the same source/runtime')
        if result: raise RuntimeError(f'{name} stopped with code {result}; see {log.name}')
    try:
        tests=root/'verification.json'
        if not tests.exists(): run('verify_v4.py','--output',tests)
        verification=json.loads(tests.read_text())
        if (not verification.get('success') or not verification.get('complete_suite')
                or verification.get('design_id')!='hope_cad_v4_warm_joint_cms_row_retention'):
            raise RuntimeError('complete v4 correctness gate failed or wrong evidence')
        profile=root/'canonical_resource_profile.json'
        if not profile.exists(): run('profile_v4.py','--device',args.device,'--output',profile)
        if json.loads(profile.read_text())['status']!='passed':
            raise RuntimeError('canonical exact-16 resource gate has not passed')
        run('run_matched_cadic.py','--protocol',args.protocol,'--method',args.baseline_method,
            '--run-dir',root/'baseline_lock','--phase','prepare')
        context=load_context(args.protocol,args.baseline_method,root/'baseline_lock')
        cache=root/'prefix_features.pt'
        if args.parent_evidence and not cache.exists():
            run('import_frozen_cache_v4.py','--parent',args.parent_evidence,
                '--output',cache,'--receipt',root/'parent_cache_receipt.json')
        chosen=cache if cache.exists() else args.feature_cache
        accepted=False
        if chosen is not None and chosen.exists():
            try:
                value=validate_bundle(torch.load(chosen,map_location='cpu',weights_only=False),context)
                if chosen!=cache:
                    from training.hope_runtime_v4 import atomic_torch_save
                    atomic_torch_save(value,cache)
                accepted=True
                atomic_json(dict(status='verified',source=str(chosen)),root/'cache_decision.json')
                del value
            except (ValueError,KeyError) as error:
                if cache.exists(): raise
                atomic_json(dict(status='rebuild_required',reason=str(error),source=str(chosen)),root/'cache_decision.json')
        if not accepted:
            run('prepare_prefix_features_v4.py','--protocol',args.protocol,'--method',args.baseline_method,
                '--run-dir',root/'baseline_lock','--output',cache,'--device',args.device,'--batch-size','1')
        preflight=root/'preflight'
        if not (preflight/'preflight.json').exists() or not json.loads((preflight/'preflight.json').read_text())['passed']:
            # Keep every failed attempt; never overwrite its stream checkpoints.
            attempt=root/f'preflight_attempt_{len(commands):03d}'
            run('preflight_rgb_v4.py',*common,'--bundle',cache,'--output-dir',attempt,'--device',args.device)
            preflight.mkdir(exist_ok=True)
            atomic_json(json.loads((attempt/'preflight.json').read_text()),preflight/'preflight.json')
        summary['completed']+=['protocol_lock','verified_cache','RGB_exact_resume']
        outer=root/'preflight_outer'/'preflight_outer.json'
        if not outer.exists():
            run('preflight_outer_v4.py',*common,'--bundle',cache,'--output-dir',outer.parent,'--device',args.device)
        outer_report=json.loads(outer.read_text())
        if not outer_report.get('passed') or not outer_report.get('kaggle_gate_eligible'):
            raise RuntimeError('real-prefix exact-meta/optimizer resume gate has not passed')
        summary['completed'].append('real_prefix_outer_exact_resume')
        if args.preflight_only:
            summary.update(status='preflight_passed',engineering_gates_passed=True,
                           selection_started=False,scientific_utility='not_evaluated',
                           next_action='attach this saved evidence to the main notebook and resume')
            return 0
        for method in args.methods:
            folder=root/'selection'/method
            if method=='A1':
                run('derive_a1_v4.py',*common,'--bundle',cache,'--selection-root',root/'selection','--device',args.device)
            else:
                extra=['--selected-m',root/'selection/M/selection.pt'] if method=='C4' else []
                resume=['--resume'] if folder.exists() else []
                run('calibrate_features_v4.py',*common,'--bundle',cache,'--output-dir',folder,
                    '--method',method,'--device',args.device,*extra,*resume)
            summary['completed'].append(f'selection_{method}')
            atomic_json(summary,root/'handoff.json')
            export_evidence(root)
        probe=root/'probe_features.pt'
        if not probe.exists():
            run('prepare_prefix_features_v4.py','--protocol',args.protocol,'--method',args.baseline_method,
                '--run-dir',root/'baseline_lock','--output',probe,'--device',args.device,'--batch-size','1',
                '--role','probe','--selected-initialization',root/'selection/M/deployment_initialization.pt')
        run('run_mechanism_v4.py',*common,'--bundle',cache,'--probe-bundle',probe,
            '--selection-root',root/'selection','--methods',*args.methods,'--output',root/'mechanism.json','--device',args.device)
        summary.update(status='ready_for_artifact_review',engineering_gates_passed=True,
                       scientific_utility='pending_review; no CAD performance measured')
        summary['completed'].append('normal_probe_mechanism_diagnostics')
    except SessionBudgetReached as error:
        summary.update(status='paused_for_session_limit',engineering_gates_passed=False,error=str(error))
        print(str(error),flush=True)
    except KeyboardInterrupt:
        summary.update(status='paused_by_operator',engineering_gates_passed=False)
    except Exception as error:
        summary.update(status='stopped',engineering_gates_passed=False,error=str(error))
        print(str(error),flush=True)
    finally:
        summary['elapsed_since_session_start_seconds']=time.monotonic()-(deadline-args.session_hours*3600)
        atomic_json(summary,root/'handoff.json')
        target=export_evidence(root)
        print(f'Evidence archive: {target}',flush=True)
    return 0 if summary['status'] in ('ready_for_artifact_review','paused_for_session_limit') else 1


if __name__=='__main__': raise SystemExit(main())
