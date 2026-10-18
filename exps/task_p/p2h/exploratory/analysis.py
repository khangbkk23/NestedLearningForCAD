"""Frozen statistical orientation, fixed families and complete failure matrix."""
import json
import numpy as np
from ..statistics import paired_benefit,primary,holm
from .design import OUT,MATRIX,save

CONTROLS=['immediate_protection','extra_replay','short_credit','reset_memory',
          'collapsed_clock','single_memory_16','M4_only_16','M16_only_16','zero_memory']


def summarize(root,expected_seeds):
    rows={}
    for p in root.glob('*/trace.json'):
        row=json.loads(p.read_text()); rows[(row['regime'],row['arm'],row['seed'])]=row
    complete=all((r,a,s) in rows for r,arms in MATRIX.items() for a in arms for s in expected_seeds)
    if not complete:
        return dict(passed=False,complete=False,missing=[dict(regime=r,arm=a,seed=s)
            for r,arms in MATRIX.items() for a in arms for s in expected_seeds if (r,a,s) not in rows])
    def values(r,a,key): return [rows[(r,a,s)][key] for s in expected_seeds]
    def contrast(r,a,b='candidate'): return paired_benefit(values(r,a,'excess'),values(r,b,'excess'))
    c1=primary(values('D8','E','excess'),values('D8','candidate','excess'),
               values('D8','E','new_normal'),values('D8','candidate','new_normal'))
    acquisition=np.array(values('D8','candidate','new_normal'))/np.maximum(values('D8','E','new_normal'),1e-8)-1
    acquisition_CI=paired_benefit(acquisition,np.zeros(len(expected_seeds)))
    comparisons={a:contrast('D8',a) for a in CONTROLS}
    pooled_control=np.mean([values(r,'post_onset_only','excess') for r in ['D16','R']],axis=0)
    pooled_candidate=np.mean([values(r,'candidate','excess') for r in ['D16','R']],axis=0)
    pooled_E=np.mean([values(r,'E','excess') for r in ['D16','R']],axis=0)
    comparisons['pre_onset_credit']=paired_benefit(pooled_control,pooled_candidate)
    comparisons['random_direction']=contrast('D8','norm_matched_random')
    comparisons['deployed_correction']=contrast('D8','deployed_ablation')
    adjusted=holm({a:r['paired_p'] for a,r in comparisons.items()})
    for a,r in comparisons.items():
        threshold=.5*(float(np.mean(pooled_E-pooled_candidate)) if a=='pre_onset_credit' else c1['benefit'])
        r.update(required_benefit=threshold,holm_p=adjusted[a],
                 passed=r['benefit']>=threshold and r['lower']>0 and adjusted[a]<.05)
    invariant={}
    for regime,arm,base in [('D32','post_onset_only','candidate'),('D8','post_onset_only','candidate')]:
        equal=[]
        for seed in expected_seeds:
            a=rows[(regime,arm,seed)]; b=rows[(regime,base,seed)]
            limit=128 if regime=='D32' else 80
            equal.append(all(x['live_sha256']==y['live_sha256'] for x,y in zip(a['traces'][:limit],b['traces'][:limit])))
        invariant[f'{regime}_mask_forward_identity']=all(equal)
    E_invariants={}
    for regime in MATRIX:
        E_invariants[regime]=all(
            all(x['live_sha256']==y['live_sha256'] for x,y in zip(rows[(regime,'E',s)]['traces'],rows[(regime,'zero_memory',s)]['traces']))
            for s in expected_seeds)
        if regime in ['D8','D16','R']:
            E_invariants[regime+'_deployed_equals_E']=all(
                all(x['live_sha256']==y['live_sha256'] for x,y in zip(rows[(regime,'E',s)]['traces'],rows[(regime,'deployed_ablation',s)]['traces']))
                for s in expected_seeds)
    descriptive={f'{r}/{a}':contrast(r,a) for r,arms in MATRIX.items() for a in arms if a!='candidate'}
    valid=all(invariant.values()) and all(E_invariants.values())
    supported=c1['passed'] and acquisition_CI['upper']<=.01 and all(r['passed'] for r in comparisons.values()) and valid
    return dict(complete=True,passed=valid,primary=c1,acquisition_regression_CI=acquisition_CI,
        C2_C3= comparisons,C4_invariants=invariant,E_zero_deployment_invariants=E_invariants,
        guard_utility=contrast('D8','disabled_guard'),descriptive=descriptive,
        verdict='IMPLEMENTATION_INVALID' if not valid else ('MECHANISM_SUPPORTED' if supported else 'MECHANISM_NOT_SUPPORTED'),
        secondary_family_size=12,scope='exploratory; original P2h remains RESOURCE_INFEASIBLE',
        failure_matrix=[dict(regime=r,arm=a,seed=s,status='COMPLETE') for r,arms in MATRIX.items() for a in arms for s in expected_seeds],
        aggregate_resources=dict(seconds=sum(r['seconds'] for r in rows.values()),
            logical_seconds=sum(r.get('logical_seconds',r['seconds']) for r in rows.values()),
            peak_reserved=max(r['peak_reserved'] for r in rows.values()),peak_allocated=max(r['peak_allocated'] for r in rows.values()),
            host_maxrss=max(r['host_maxrss'] for r in rows.values()),
            ordinary_gradient_calls=sum(r['meter'].get('normal_gradient_calls',0) for r in rows.values()),
            physical_ordinary_gradient_calls=sum(r.get('physical_meter',r['meter']).get('normal_gradient_calls',0) for r in rows.values()),
            meta_gradient_calls=sum(r['meter'].get('meta_gradient_calls',0) for r in rows.values()),
            adjoint_sites=sum(r['meter'].get('inner_gradient_adjoint_sites',0) for r in rows.values()),
            witness_forwards=sum(r['witness_loss_forwards'] for r in rows.values()),
            physical_witness_forwards=sum(r.get('physical_witness_forwards',r['witness_loss_forwards']) for r in rows.values())))


def known_signs():
    a=paired_benefit([1.]*10,[.8]*10)
    assert a['benefit']>0 and a['lower']>0
    for name in ['C1','C2','C3','C4']:
        assert paired_benefit([.8]*10,[1.]*10)['upper']<0,name
    # Difference of paired D16/R differences is averaged within seed, never n20.
    pooled=paired_benefit(np.mean([[1.]*10,[2.]*10],axis=0),np.mean([[.8]*10,[1.8]*10],axis=0))
    assert abs(pooled['benefit']-.2)<1e-12 and pooled['n']==10
    return True
