"""Analyze saved engineering witness scalars only; never creates new scores/data."""
import json
import math
from pathlib import Path
import numpy as np
from .design import OUT,ARMS,save
from .statistics import paired_benefit,primary,mde


def main():
    gate=json.loads((OUT/'G1b.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'engineering').glob('*/trace.json'))]
    summary={}
    for regime in ['D8','D16']:
        for arm in ['candidate','relaxed_cap']:
            selected=[r for r in rows if r['regime']==regime and r['arm']==arm and r.get('excess')]
            if len(selected)!=10: continue
            E=[r['excess']['E'] for r in selected]; candidate=[r['excess']['candidate'] for r in selected]
            new_E=[r['new_normal']['E'] for r in selected]; new_candidate=[r['new_normal']['candidate'] for r in selected]
            contrast=primary(E,candidate,new_E,new_candidate)
            # Independent arithmetic and known df9 critical value, not the statistics implementation.
            differences=[a-b for a,b in zip(E,candidate)]
            mean=sum(differences)/10
            sd=math.sqrt(sum((v-mean)**2 for v in differences)/9)
            independent=(mean-2.2621571628540993*sd/math.sqrt(10),mean+2.2621571628540993*sd/math.sqrt(10))
            assert abs(contrast['benefit']-mean)<1e-12
            assert max(abs(contrast['lower']-independent[0]),abs(contrast['upper']-independent[1]))<1e-12
            # Diagnostic local-E, standalone E and evaluator forwards were intentionally
            # outside the candidate meter. Charge them explicitly to study resources.
            extra={'normal_gradient_calls':0,'loss_forwards':0,'image_forward_accesses':0,
                   'input_patches':0,'queried_patches':0}
            for row in selected:
                cycles=row['completed_cycles']
                witness_events=sum(bool(row[k]) for k in ['acquisition','retained','new_normal'])
                witness_forwards=32*witness_events
                # Two E paths, each: support1 + current/replay2 images per cycle;
                # the first event's two cycles have no replay.
                extra['normal_gradient_calls']+=4*cycles
                extra['loss_forwards']+=4*cycles+witness_forwards
                accesses=2*(3*cycles-2)+witness_forwards
                extra['image_forward_accesses']+=accesses
                extra['input_patches']+=accesses*784
                extra['queried_patches']+=accesses*392
            summary[f'{regime}/{arm}']=dict(
                seeds=[r['seed'] for r in selected],
                witness_contrast=contrast,MDE=mde(E,candidate),
                relative_benefit=None if np.mean(E)<=1e-8 else contrast['benefit']/float(np.mean(E)),
                acquisition_A={n:float(np.mean([r['acquisition'][n] for r in selected])) for n in ['E','candidate']},
                endpoint_old_loss={n:float(np.mean([r['retained'][n] for r in selected])) for n in ['E','candidate']},
                acquisition_B={n:float(np.mean([r['new_normal'][n] for r in selected])) for n in ['E','candidate']},
                median_correction_ratio=float(np.median([r['median_correction_base_ratio'] for r in selected])),
                accepted_effective_cycles=sum(r['effective_after16'] for r in selected),
                gradient_diagnostics=[dict(seed=r['seed'],parts=r['decompositions']) for r in selected],
                total_seconds=sum(r['seconds'] for r in selected),
                peak_reserved=max(r['cuda_peak_reserved'] for r in selected),
                peak_allocated=max(r['cuda_peak_allocated'] for r in selected),
                max_host_RSS=max(r['cpu_maxrss_bytes'] for r in selected),
                extra_study_counts_reconstructed_from_call_contract=extra,
                aggregate_counts={key:sum(r['meter'].get(key,0) for r in selected)
                                  for key in set().union(*(r['meter'] for r in selected))})
    G2=None if not (OUT/'G2.json').exists() else json.loads((OUT/'G2.json').read_text())
    if not gate['passed']:
        verdict=gate['verdict']; first=dict(gate='G1b',detail=gate['first_failure'])
    elif G2 is not None and not G2['passed']:
        verdict=G2['verdict']; first=dict(gate='G2',detail=G2['first_failure'])
    else:
        verdict=None; first=None
    matrix=[dict(regime=regime,arm=arm,status='NOT_STARTED',reason='No valid Execution Lock')
            for regime in ['D8','D16','R','C','D32'] for arm in ARMS if arm!='relaxed_cap']
    save(OUT/'engineering_analysis.json',dict(schema='p2h_engineering_analysis_v1',
         scope='nonconfirmatory engineering seeds only; no G3 endpoint',summary=summary,
         G1b_passed=gate['passed'],G2=G2,first_failed_gate=first,verdict=verdict,
         G3_failure_matrix=matrix,G3_untouched=True,independent_CI_arithmetic=True))
    print(json.dumps(dict(verdict=verdict,first_failed_gate=first,
          contrasts={key:dict(benefit=r['witness_contrast']['benefit'],CI=[r['witness_contrast']['lower'],r['witness_contrast']['upper']],
                             relative=r['relative_benefit'],acquisition_regression=r['witness_contrast']['acquisition_regression'],
                             MDE=r['MDE']) for key,r in summary.items()})),flush=True)


if __name__=='__main__': main()
