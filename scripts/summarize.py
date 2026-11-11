"""Summarize the three B319 arms from their recorded stage metrics."""
import argparse,json
from pathlib import Path
from statistics import mean
METRIC_VERSION="B319-rank-v1-prior14-v1"

def summarize(root):
    result={'metric_version':METRIC_VERSION,'status':'COMPLETE_REVIEW_PENDING','arms':{},'scientific_acceptance':False,
        'forgetting_definition':{'tasks':'14 categories with at least one observation before final',
            'Fmax':'max(AP before final)-AP final; signed',
            'Facq':'AP at acquisition-AP final; signed',
            'legacy_final_inclusive_Fmax':'max(AP including final)-AP final, averaged over all 15 categories'}}
    for arm in ('er_uniform','a0_tail','a0_tail_ctx'):
        study=json.loads((root/arm/'arms'/arm/'study.json').read_text())
        if study['status']!='COMPLETE_REVIEW_PENDING' or study['committed']!=3629:
            raise ValueError('Incomplete arm '+arm)
        snapshots=study['snapshots']
        if len(snapshots)!=15 or len(snapshots[-1]['metrics'])!=15:
            raise ValueError('Expected full15 stage/category metrics')
        categories=list(snapshots[-1]['metrics'])
        matrix={scope+'_'+metric:[[stage['metrics'].get(category,{}).get(scope,{}).get(metric) for category in categories] for stage in snapshots] for scope in ('pixel','image') for metric in ('AP','AUROC')}
        per={}
        for category,cell in snapshots[-1]['metrics'].items():
            history=[s['metrics'][category]['pixel']['AP'] for s in snapshots if category in s['metrics']]
            per[category]={'pixel_AP':cell['pixel']['AP'],'image_AUROC':cell['image']['AUROC'],
                'Fmax':max(history[:-1])-history[-1] if len(history)>1 else None,
                'Facq':history[0]-history[-1] if len(history)>1 else None,
                'legacy_final_inclusive_Fmax':max(history)-history[-1]}
        result['arms'][arm]={'final_macro_pixel_AP':mean(x['pixel_AP'] for x in per.values()),
            'final_macro_image_AUROC':mean(x['image_AUROC'] for x in per.values()),
            'macro_Fmax':mean(x['Fmax'] for x in per.values() if x['Fmax'] is not None),
            'macro_Facq':mean(x['Facq'] for x in per.values() if x['Facq'] is not None),
            'legacy_macro_final_inclusive_Fmax_15':mean(x['legacy_final_inclusive_Fmax'] for x in per.values()),
            'legacy_macro_Facq_15':sum(x['Facq'] or 0 for x in per.values())/len(per),
            'per_category':per,'matrix':{'boundaries':[s.get('cursor') for s in snapshots],'tasks':categories,'values':matrix}}
    er=result['arms']['er_uniform']['final_macro_pixel_AP']
    for arm in ('a0_tail','a0_tail_ctx'):
        result['arms'][arm]['pixel_AP_vs_ER']=result['arms'][arm]['final_macro_pixel_AP']-er
    return result

def render_results(result):
    lines=['# B319 results','',f'Generated from recorded stage metrics; metric version: `{result["metric_version"]}`.','',
           '| Arm | Pixel AP | Image AUROC | Prior-max forgetting | Acquisition forgetting |',
           '| --- | ---: | ---: | ---: | ---: |']
    for arm,row in result['arms'].items():
        lines.append(f'| {arm} | {row["final_macro_pixel_AP"]:.8f} | {row["final_macro_image_AUROC"]:.8f} | {row["macro_Fmax"]:.8f} | {row["macro_Facq"]:.8f} |')
    lines+=['','Forgetting is signed and averages the 14 tasks observed before the final boundary.','The JSON retains full precision and the boundary-by-task matrices.','Execution/reproduction does not establish scientific acceptance.','']
    return '\n'.join(lines)

def write_results(root,result):
    root=Path(root)
    if (root/'receipts/receipt.json').exists() or (root.parent/'receipts/receipt.json').exists():
        raise FileExistsError('Closed run: publish a separate analysis instead of editing producer outputs')
    if any((root/name).exists() for name in ('summary.json','metrics.json','results.md')):
        raise FileExistsError('Preserve existing producer reports')
    for name in ('summary.json','metrics.json'):(root/name).write_text(json.dumps(result,indent=2)+'\n')
    (root/'results.md').write_text(render_results(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    result=summarize(a.run);write_results(a.run,result)
    print(json.dumps(result,indent=2),flush=True)
