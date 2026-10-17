"""Independent read-only P0 analysis from saved losses. No feature reads or model fitting."""
import json
from pathlib import Path
import numpy as np
import torch

from exps.task_p.p0_protocol import ROOT, OUT, CODE, CATEGORIES, save, sha
from exps.task_p.p0_normal_support import ARMS, BUDGET, fingerprint, require


def quantiles(x):return [float(y) for y in np.quantile(x,[.025,.975])]


def values(records,event,cat,arm,metric='tail_normalized'):
    rows=sorted([r for r in records if r['event']==event and r['category']==cat],key=lambda r:r['path'])
    require(len(rows)==20,'endpoint coverage differs')
    return np.array([r['arms'][arm][metric] for r in rows])


def ranks(x):
    x=np.asarray(x);order=np.argsort(x,kind='stable');out=np.empty(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out


def correlation(a,b):
    x,y=ranks(a),ranks(b);x-=x.mean();y-=y.mean()
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denominator) if denominator>0 else None


def analyze(records,protocol,trace):
    final=np.array([[values(records,350,c,a) for c in CATEGORIES] for a in ARMS])
    rng=np.random.default_rng(protocol['bootstrap']['seed'])
    ix=rng.integers(0,20,size=(10000,3,20))
    boot=np.array([[final[j,k][ix[:,k]].mean(1) for k in range(3)] for j in range(4)])
    category=final.mean(-1);macro=category.mean(-1);bm=boot.mean(1);eps=protocol['gates']['near_zero_normalized_risk']
    valid=bool(np.isfinite(final).all() and np.all(final>=0) and macro[2]>eps and np.all(category[2]>eps) and np.all(bm[2]>eps))
    effect=float(1-macro[3]/macro[2]) if valid else None
    ci=quantiles(1-bm[3]/bm[2]) if valid else None
    retention={};acquisition={}
    for cat,start,end in protocol['gates']['retention_contrasts']:
        k=CATEGORIES.index(cat);old=values(records,start,cat,ARMS[3]);new=values(records,end,cat,ARMS[3])
        valid=valid and bool(old.mean()>eps)
        ratio=float(new.mean()/old.mean()-1) if old.mean()>eps else None
        retention[f'{cat}_{start}_{end}']={'relative_deterioration':ratio,
            'ci95':quantiles(new[ix[:,k]].mean(1)/old[ix[:,k]].mean(1)-1) if old.min()>eps else None}
    for cat,event in [('carpet',200),('hazelnut',300)]:
        k=CATEGORIES.index(cat);a=values(records,event,cat,ARMS[0]);d=values(records,event,cat,ARMS[3])
        valid=valid and bool(a.mean()>eps)
        acquisition[cat]={'event':event,'relative_gain':float(1-d.mean()/a.mean()) if a.mean()>eps else None,
            'ci95':quantiles(1-d[ix[:,k]].mean(1)/a[ix[:,k]].mean(1)) if a.min()>eps else None}
    dtrace=[r for r in trace if r['arm']==ARMS[3]]
    effect_count=sum(r['guard_effect_attempts'] for r in dtrace)
    cost_count=sum(r['nonzero_witness_cost_proposals'] for r in dtrace)
    gates={'valid_execution':True,'final_gain_at_least_5pct':bool(effect is not None and effect>=.05),
           'gain_ci_lower_above_zero':bool(ci is not None and ci[0]>0),
           'retention_within_20pct':all(v['relative_deterioration'] is not None and v['relative_deterioration']<=.2 for v in retention.values()),
           'acquisition_at_least_10pct_each':all(v['relative_gain'] is not None and v['relative_gain']>=.1 for v in acquisition.values()),
           'per_category_within_10pct_greedy':bool(np.all(category[3]<=1.1*category[2])),
           'actual_witness_constraint_effect':bool(effect_count>0 and cost_count>0),
           'synthetic_guard_off_and_causal_tests':True}
    status='P0_INCONCLUSIVE' if not valid else 'P0_GO' if all(gates.values()) else 'P0_NO_GO'
    summary={'status':status,'macro':{a:{'normalized_risk':float(macro[j]),'ci95':quantiles(bm[j])} for j,a in enumerate(ARMS)},
       'per_category':{c:{a:{'raw_risk':float(values(records,350,c,a,'tail_raw').mean()),
             'normalized_risk':float(category[j,k]),'ci95':quantiles(boot[j,k])} for j,a in enumerate(ARMS)} for k,c in enumerate(CATEGORIES)},
       'primary_relative_gain':effect,'primary_gain_ci95':ci,'retention':retention,'acquisition':acquisition,'gates':gates,
       'valid_denominators':bool(valid),'guard_effect_attempts':effect_count,'nonzero_witness_cost_proposals':cost_count,
       'per_category_final_gain':{c:{'relative_gain':float(1-category[3,k]/category[2,k]),
            'ci95':quantiles(1-boot[3,k]/boot[2,k])} for k,c in enumerate(CATEGORIES)} if valid else None}
    return summary,{'seed':7201,'replicates':10000,'paired_image_stratification':True,'primary_gain_samples':(1-bm[3]/bm[2]).tolist() if valid else []}


def main():
    p=json.loads((OUT/'protocol_resolved.json').read_text());require(sha(OUT/'protocol_resolved.json')==(OUT/'protocol.sha256').read_text().strip(),'protocol mismatch')
    require(all(sha(ROOT/n)==h for n,h in p['source_code_sha256'].items()),'frozen source mismatch')
    require(all(sha(ROOT/n)==h for n,h in p['preserved_sha256'].items()),'protected source mismatch')
    read=lambda name:json.loads((OUT/name).read_text())
    records=read('per_image_results.json');trace=read('update_trace.json');chain=read('state_chain.json');access=read('access_log.json');boundaries=read('boundaries.json')
    require(len(records)==180 and len(trace)==1400 and len(chain)==350,'incomplete experiment')
    require(all(x['after']==y['before'] for x,y in zip(chain,chain[1:])),'state discontinuity')
    require([r['path'] for r in access if r['role']=='update']==[r['relative_path'] for r in p['updates']],'update chronology differs')
    expected=[];admitted=[]
    for row in p['updates']:
        if row['category'] not in admitted:admitted.append(row['category'])
        expected.append(('update',row['relative_path'],row['event']))
        if row['event'] in p['arrival_boundaries']:
            expected.extend(('probe',q['relative_path'],row['event']) for q in p['probes'] if q['category'] in admitted)
    require(expected==[(r['role'],r['path'],r['event']) for r in access],'exact access order differs')
    for event in range(1,351):
        block=[r for r in trace if r['event']==event]
        require(len(block)==4 and len({r['input_sha256'] for r in block})==1,'unmatched proposals')
        require(all(r['ledger']['charged_persistent_bytes']<=BUDGET for r in block),'budget failed')
        d=[r for r in block if r['arm']==ARMS[3]][0]
        require(d['accepted_readout_changes']==d['inserted']+d['replaced'] and d['rejected_bank_unchanged'],'causal check failed')
    frozen=[r for r in trace if r['arm']==ARMS[0] and r['event']>=100]
    require(len({r['score_bank_sha256'] for r in frozen})==1,'frozen score bank changed')
    require(all(r['no_write_state_unchanged'] for r in read('causal_checks.json')),'no-write failed')
    for entry in records:
        require(len(entry['arms'])==4,'missing arm')
        for arm,metrics in entry['arms'].items():
            d=np.array(metrics['distances']);tail=float(np.sort(d)[-7:].mean())
            require(len(d)==64 and abs(tail-metrics['tail_raw'])<1e-10,'independent tail recomputation differs')
            require(abs(tail/entry['scale']-metrics['tail_normalized'])<1e-12,'normalization differs')
    summary,bootstrap=analyze(records,p,trace)
    for boundary in boundaries:
        for cat,metrics in boundary['categories'].items():
            for arm,v in metrics.items():
                require(abs(values(records,boundary['event'],cat,arm).mean()-v['normalized_risk'])<1e-12,'independent boundary mean differs')
    final=torch.load(OUT/'states_t350.pt',weights_only=True,map_location='cpu')
    require(final['protocol_sha256']==sha(OUT/'protocol_resolved.json'),'state protocol identity mismatch')
    require(all(fingerprint(final['models'][a])==chain[-1]['after'][a] for a in ARMS),'final state mismatch')
    geometry={}
    for cat in CATEGORIES:
        rows=[r for r in records if r['event']==350 and r['category']==cat]
        corr=[correlation(r['arms'][ARMS[2]]['distances'],r['arms'][ARMS[3]]['distances']) for r in rows]
        matches=[float(np.mean(np.array(r['arms'][ARMS[2]]['nearest_serials'])==np.array(r['arms'][ARMS[3]]['nearest_serials']))) for r in rows]
        geometry[cat]={'mean_D_C_distance_spearman':float(np.mean([v for v in corr if v is not None])),
                       'mean_D_C_same_nearest_descriptor_fraction':float(np.mean(matches))}
    diagnostics={'geometry':geometry,'method_update_counts':{a:{k:sum(r[k] for r in trace if r['arm']==a) for k in ('inserted','replaced','rejected')} for a in ARMS},
       'witness':{k:sum(r[k] for r in trace if r['arm']==ARMS[3]) for k in ('evaluated_replacements','guard_effect_attempts','guard_rejected_proposals','nonzero_witness_cost_proposals','acquisition_gain_sum','witness_cost_sum')},
       'integrity':{'independent_tail_recompute':True,'independent_boundary_recompute':True,'exact_access_order':True,'state_continuity':True,
                    'input_parity':True,'byte_cap':True,'no_write':True,'frozen_bank':True,'final_state_reopened':True},
       'analysis_scope':'saved per-image distances, logs and final states only; no caches, raw images, fitting or inference'}
    save('summary.json',summary);save('bootstrap.json',bootstrap);save('forensic_analysis.json',diagnostics)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
