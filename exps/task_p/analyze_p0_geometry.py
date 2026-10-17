import json
from pathlib import Path
import numpy as np
import torch
root=Path('/mnt/d/code_for_fun/project/nestedlearningforcad/nestedlearningforcad');out=root/'results/hope_cad/task_p/p0'
p=json.loads((out/'protocol_resolved.json').read_text());records=json.loads((out/'per_image_results.json').read_text());trace=json.loads((out/'update_trace.json').read_text())
models=torch.load(out/'states_t350.pt',weights_only=True,map_location='cpu')['models']
C='STREAM_GREEDY';D='WITNESS_PROTECTED'
result={'scope':'POSTHOC DESCRIPTIVE ONLY; saved predictions/traces/states, no feature cache, inference, fitting, threshold change or gate modification','categories':{},'state_geometry':{}}
for cat in ('bottle','carpet','hazelnut'):
 rows=[r for r in records if r['event']==350 and r['category']==cat]
 tailids=[];equal=[];meancontrast=[];sourcecounts={a:{c:0 for c in ('bottle','carpet','hazelnut')} for a in (C,D)}
 positioncounts={a:np.zeros(64,dtype=int) for a in (C,D)}
 for r in rows:
  c,d=(np.array(r['arms'][a]['distances']) for a in (C,D))
  it=np.argsort(c,kind='stable')[-7:]
  tailids.append(float(np.mean(np.array(r['arms'][C]['nearest_serials'])[it]==np.array(r['arms'][D]['nearest_serials'])[it])))
  equal.append(bool(np.array_equal(np.sort(c)[-7:],np.sort(d)[-7:])))
  meancontrast.append(float(c.mean()-d.mean()))
  for a in (C,D):
   top=np.argsort(r['arms'][a]['distances'],kind='stable')[-7:];positioncounts[a][top]+=1
   for j in top:sourcecounts[a][p['updates'][r['arms'][a]['nearest_serials'][j]//16]['category']]+=1
 result['categories'][cat]={'all_images_top7_distance_multiset_bitwise_equal':all(equal),'images_equal_top7':sum(equal),
  'C_top7_same_nearest_serial_fraction_C_vs_D':float(np.mean(tailids)),
  'mean_patch_distance_C_minus_D':float(np.mean(meancontrast)),'tail_nearest_source_counts':sourcecounts,
  'top7_position_counts':{a:positioncounts[a].tolist() for a in (C,D)},
  'most_frequent_top7_positions':{a:[{'row':p['probe_grid'][j]//28,'col':p['probe_grid'][j]%28,'count':int(positioncounts[a][j])} for j in np.argsort(-positioncounts[a],kind='stable')[:8]] for a in (C,D)}}
for a,s in models.items():
 n=s['n'];geom={'capacity':s['capacity'],'occupied':n,'read_only_allocated_bank_and_serial_bytes':s['bank'].numel()*4+s['ids'].numel()*8,
               'inference_minimal_occupied_descriptor_bytes':n*768*4,'state_tensor_byte_ledger':{k:v.numel()*v.element_size() for k,v in s.items() if isinstance(v,torch.Tensor)}}
 if 'nn_distance' in s:
  v=s['nn_distance'][:n].numpy();geom['stored_exact_bank_nearest_other_distance']={'minimum':float(v.min()),'mean':float(v.mean()),'median':float(np.median(v)),'p90':float(np.quantile(v,.9))}
 result['state_geometry'][a]=geom
cset=set(models[C]['ids'][:models[C]['n']].tolist());dset=set(models[D]['ids'][:models[D]['n']].tolist())
result['score_bank_overlap']={'common_descriptors':len(cset&dset),'C_only':len(cset-dset),'D_only':len(dset-cset)}
dtrace=[r for r in trace if r['arm']==D];decisions=[z for r in dtrace for z in r['decisions'] if z['selected_slot'] is not None]
slack=[z['gain']-z['witness_cost'] for z in decisions];tails=[z['tail_cost'] for z in decisions]
result['witness_sensitivity_descriptive']={'accepted_replacements':len(decisions),'zero_accepted_witness_cost_count':sum(z['witness_cost']==0 for z in decisions),
 'zero_accepted_tail_increment_count':sum(x==0 for x in tails),'accepted_gain_minus_cost_min_median_max':[float(np.min(slack)),float(np.median(slack)),float(np.max(slack))],
 'accepted_tail_cost_min_median_max':[float(np.min(tails)),float(np.median(tails)),float(np.max(tails))],
 'accepted_cost_to_gain_sum_ratio':sum(z['witness_cost'] for z in decisions)/sum(z['gain'] for z in decisions),
 'guard_changes_eviction_not_admission':sum(r['rejected'] for r in dtrace)==0,
 'candidate_no_update_image_count':sum(r['inserted']+r['replaced']==0 for r in dtrace)}
(out/'posthoc_geometry_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='categories'},indent=2))
for cat,v in result['categories'].items():print(cat,{k:v for k,v in v.items() if k not in ('top7_position_counts',)})

# Exploratory display of saved measurements only; no fitting, inference or new endpoint.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary=json.loads((out/'summary.json').read_text());boundaries=json.loads((out/'boundaries.json').read_text())
colors={'FROZEN_FIRST':'#777777','RESERVOIR':'#e69f00','STREAM_GREEDY':'#0072b2','WITNESS_PROTECTED':'#cc79a7'}
fig,axes=plt.subplots(1,4,figsize=(15,3.7),layout='constrained')
for ax,cat in zip(axes[:3],('bottle','carpet','hazelnut')):
 for a in colors:
  series=[(b['event'],b['categories'][cat][a]['normalized_risk']) for b in boundaries if cat in b['categories']]
  ax.plot([x for x,y in series],[y for x,y in series],marker='o',markersize=4,color=colors[a],label=a,linewidth=1.6)
 ax.axvline(300,color='#aaaaaa',linestyle=':',linewidth=1)
 ax.set(title=cat,xlabel='Arrived normal images',ylabel='Normalized support tail risk')
 ax.set_xticks([100,200,300,350]);ax.grid(alpha=.2)
ax=axes[-1];names=['bottle','carpet','hazelnut','macro'];points=[];errors=[]
for cat in names:
 if cat=='macro':v=summary['primary_relative_gain'];lo,hi=summary['primary_gain_ci95']
 else:v=summary['per_category_final_gain'][cat]['relative_gain'];lo,hi=summary['per_category_final_gain'][cat]['ci95']
 points.append(100*v);errors.append([100*(v-lo),100*(hi-v)])
ax.errorbar(points,range(4),xerr=np.array(errors).T,fmt='o',color=colors[D],capsize=3)
ax.axvline(0,color='black',linewidth=.8);ax.axvline(5,color='#999999',linestyle='--',linewidth=.8)
ax.set(yticks=range(4),yticklabels=names,xlabel='D improvement over C (%)',title='Final paired image bootstrap 95% CI')
ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False,fontsize=8)
fig.suptitle('P0 normal support only: acquisition/retention pass, witness advantage fails',fontsize=12)
fig.savefig(out/'p0_normal_support_results.png',dpi=180);fig.savefig(out/'p0_normal_support_results.pdf');plt.close(fig)
