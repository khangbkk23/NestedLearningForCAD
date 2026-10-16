"""CPU-only W7 suite audit, paired uncertainty and claim rules; no model fitting."""
import argparse
from pathlib import Path
import sys
import json
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import file_hash
from training.hope_runtime_v4 import atomic_json,runtime_identity
from training.hope_w7_contract import jobs,source_hashes,lock_json
from training.hope_w7_engine import atomic_npz
from training.hope_w7_statistics import (EvidenceView,image_counts,cell_bootstrap,
                                       continual_bootstrap,paired_interval)


def collect(paths):
    views=[EvidenceView(path) for path in paths];index={}
    if not views:raise ValueError('no W7 evidence inputs')
    suite=views[0].suite
    for view in views:
        if view.suite!=suite:raise ValueError('cannot combine different W7 suites')
        for name in view.completions():
            completion=view.json(name);job=completion['job']['id']
            if completion['status']!='complete':continue
            if completion['suite_digest']!=suite['digest']:raise ValueError('completion suite mismatch')
            if job in index:
                previous=index[job]
                shared=set(previous.manifest['files'])&set(view.manifest['files'])
                # Later 7.2 adds cells to 7.1. Shared committed score files must remain identical.
                if any(previous.manifest['files'][n]!=view.manifest['files'][n]
                       for n in shared if n.startswith(f'runs/{job}/scores/') and not n.endswith('progress.json')):
                    raise ValueError('conflicting duplicate run outputs')
                old=previous.json(f'runs/{job}/completion.json')
                if old['matrix_complete'] and not completion['matrix_complete']:continue
            index[job]=view
    return views,index,suite


def summarize(paths,output,*,inventory_only=False,exact_pixels=False,deadline=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    views,index,suite=collect(paths)
    try:
        expected={j['id'] for j in jobs()}
        missing=sorted(expected-set(index))
        incomplete=sorted(job for job,v in index.items() if not v.json(f'runs/{job}/completion.json')['matrix_complete'])
        inventory=dict(expected_jobs=len(expected),present_jobs=len(index),missing=missing,
                       incomplete_matrices=incomplete,suite_digest=suite['digest'],reportable=False)
        atomic_json(inventory,output/'inventory.json')
        if inventory_only or missing or incomplete:return inventory
        if suite['identity']['source_sha256']!=source_hashes():
            raise ValueError('analysis source differs from the frozen suite; do not silently rebind')
        analysis_runtime=runtime_identity()
        lock_json(output/'analysis_lock.json',dict(suite_digest=suite['digest'],exact_pixels=exact_pixels,
            python=analysis_runtime['python'],
            packages={k:analysis_runtime['packages'][k] for k in ('numpy','scikit-learn')},
            rng_policy='numpy.default_rng(SeedSequence([20261009, category_index]))'))
        table={job:dict(final=v.json(f'runs/{job}/metrics/final_macro.json'),
                       continual=v.json(f'runs/{job}/metrics/continual_summary.json')) for job,v in index.items()}
        atomic_json(table,output/'all_jobs_point_metrics.json')
        main_summary={}
        for method in ('M','C0','C1','C2','C3','C4','A1','CADIC2500','CADIC10000'):
            main_summary[method]={}
            for metric in ('i_auroc','p_aupr','i_ap','p_auroc'):
                main_summary[method][metric]={}
                for endpoint in ('final_macro','bwt','fm'):
                    values=[table[f'{method}_seed{s}']['continual'][metric][endpoint] for s in range(3)]
                    main_summary[method][metric][endpoint]=dict(mean=float(np.mean(values)),
                        std=float(np.std(values,ddof=1)),seeds=values)
        atomic_json(main_summary,output/'main_seed_mean_std.json')
        orders={}
        for order in (100,200):
            for method in ('M','C4','CADIC2500','CADIC10000'):
                key=f'{method}_seed0_order{order}'
                orders[key]=dict(task_order=index[key].json(f'runs/{key}/manifest_train.json')['task_order'],
                    final=table[key]['final'],
                    final_delta_from_canonical={metric:table[key]['final'][metric]-table[f'{method}_seed0']['final'][metric]
                                               for metric in ('i_auroc','p_aupr','i_ap','p_auroc')},
                    continual=table[key]['continual'])
        atomic_json(dict(scope='two first-task-fixed orders; separate from trained-seed uncertainty',
                         orders=orders),output/'order_studies.json')
        bootstrap={};reference={};accuracy={}
        for seed in range(3):
            for method in ('M','C0','C1','C2','C3','C4','A1','CADIC2500','CADIC10000'):
                job=f'{method}_seed{seed}';view=index[job]
                matrices={k:np.full((15,15,2000),np.nan) for k in ('i_auroc','p_aupr')}
                for checkpoint in range(15):
                    for task in range(checkpoint+1):
                        key=f'{job}_{checkpoint:02d}_{task:02d}'
                        saved=output/'bootstrap_cells'/f'{key}.npz'
                        meta_path=saved.with_suffix('.json')
                        identity=dict(suite_digest=suite['digest'],job=job,checkpoint=checkpoint,task=task,
                                      exact_pixels=exact_pixels,replicates=2000,seed=20261009)
                        if deadline is not None and time.monotonic()>=deadline:
                            return dict(status='analysis_paused_at_cell_boundary',reportable=False,
                                        suite_digest=suite['digest'],next_cell=key)
                        arrays=view.cell(job,checkpoint,task)
                        identities=arrays['identities'].tolist();labels=arrays['labels'].tolist()
                        # Pair masks/identities across all methods, seeds and checkpoints.
                        signature=canonical_digest(dict(identities=identities,labels=labels,
                            masks_sha256=__import__('hashlib').sha256(arrays['masks'].tobytes()).hexdigest()))
                        if task in reference and reference[task]!=signature:
                            raise ValueError('paired bootstrap image identities/labels/masks differ')
                        reference[task]=signature
                        identity['paired_inventory_digest']=signature
                        if saved.exists() and meta_path.exists():
                            meta=json.loads(meta_path.read_text())
                            if meta['identity']!=identity or meta['sha256']!=file_hash(saved):
                                raise ValueError('bootstrap resume identity/checksum mismatch')
                            with np.load(saved,allow_pickle=False) as raw:values={k:raw[k] for k in matrices}
                            validation=meta['accuracy']
                        else:
                            values,validation=cell_bootstrap(arrays,image_counts(labels,task),exact_pixels=exact_pixels)
                            atomic_npz(saved,**values)
                            atomic_json(dict(identity=identity,accuracy=validation,sha256=file_hash(saved)),meta_path)
                        accuracy[key]=validation
                        for metric in matrices:matrices[metric][checkpoint,task]=values[metric]
                        del arrays
                bootstrap[job]={metric:continual_bootstrap(matrix) for metric,matrix in matrices.items()}
                print(f'[W7 CPU REPORT] {job} complete',flush=True)
        comparisons={}
        for other in ('CADIC10000','CADIC2500','C0','C1','C2','C3','C4','A1'):
            primary=other=='CADIC10000';entry={}
            for metric in ('i_auroc','p_aupr'):
                entry[metric]={}
                for endpoint in ('final','bwt','forgetting'):
                    deltas=np.stack([bootstrap[f'M_seed{s}'][metric][endpoint]-bootstrap[f'{other}_seed{s}'][metric][endpoint]
                                     for s in range(3)])
                    interval=paired_interval(deltas.mean(axis=0),primary=primary and endpoint=='final')
                    field='final_macro' if endpoint=='final' else endpoint if endpoint=='bwt' else 'fm'
                    points=[table[f'M_seed{s}']['continual'][metric][field]-table[f'{other}_seed{s}']['continual'][metric][field] for s in range(3)]
                    entry[metric][endpoint]=dict(paired_interval=interval,paired_seed_deltas=points,
                        mean_delta=float(np.mean(points)),seed_delta_std=float(np.std(points,ddof=1)),
                        better_direction='negative' if endpoint=='forgetting' else 'positive')
            comparisons[f'M_vs_{other}']=entry
        main=comparisons['M_vs_CADIC10000']
        report=dict(status='complete_pending_research_review',reportable=False,suite_digest=suite['digest'],
            comparisons=comparisons,primary_simultaneous_improvement_supported=all(
                main[k]['final']['paired_interval']['lower']>0 for k in ('i_auroc','p_aupr')),
            interpretation='CIs conditional on fixed categories/trained seeds; order results separate; disclose prior exposure',
            bootstrap=dict(replicates=2000,seed=20261009,categories='fixed',
                resampling='whole images, paired and stratified by category/image label',
                trained_seeds='fixed, average metric deltas, never ensemble scores',
                checkpoint_weights='same per-image multiplicities across every checkpoint',accuracy=accuracy),
            calibration_cost_once=views[0].json('w6_compute_ledger.json'),analysis_runtime=analysis_runtime,
            CADIC_replicates='report observed variation; identical deterministic repeats are not independent evidence')
        atomic_json(report,output/'suite_report.json')
        return report
    finally:
        for view in views:view.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--exact-pixels',action='store_true')
    p.add_argument('--session-hours',type=float,default=10.5)
    args=p.parse_args()
    if not 0.05<=args.session_hours<=11:p.error('session hours must be within 0.05..11')
    result=summarize(args.evidence,args.output_dir,inventory_only=args.inventory_only,exact_pixels=args.exact_pixels,
                     deadline=time.monotonic()+args.session_hours*3600-180)
    atomic_json({k:v for k,v in result.items() if k not in ('comparisons','bootstrap')},args.output_dir/'handoff.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','bootstrap')},indent=2))
    return 0 if result.get('status') in ('complete_pending_research_review','analysis_paused_at_cell_boundary') or args.inventory_only else 2


if __name__=='__main__':raise SystemExit(main())
