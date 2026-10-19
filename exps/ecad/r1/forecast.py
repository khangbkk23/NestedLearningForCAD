"""Cost-only admission arithmetic. Missing measurements never become zero cost."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    profile=json.loads(args.profile.read_text())
    result={'updates':25200,'boundary_witness_784_query_reads':10368,
            'conservative_total_784_query_reads':35568,'fixed_reservations_seconds':1920,
            'saturated_safety_multiplier':1.25,'forecast_admission_seconds':6480,
            'hard_total_seconds':7200,'measurement_status':profile['status'],
            'forecast_seconds':None,'normal_screen_admitted':False,
            'source_profile':str(args.profile),'missing_cells':[]}
    total=1920.
    for ledger in ('equal_descriptor','equal_byte'):
        for arm in ('contcore','reservoir','cadic','p0','s4','r4'):
            key=ledger+'_'+arm;cell=profile['arms'].get(key,{})
            summary=args.profile.parent/key/'summary.json'
            if cell.get('status')!='COMPLETE' or not summary.is_file():
                result['missing_cells'].append(key);continue
            data=json.loads(summary.read_text());rows=[r for r in data['records'] if r['label'].startswith('saturated')]
            u=max(r['cached_image_wall_seconds'] for r in rows)
            nn=max(r['global_784_nn_seconds']+r['inference_bank_preparation_seconds']+r['result_transfer_seconds'] for r in rows)
            total+=1.25*(2100*u+864*nn)
    if not result['missing_cells']:
        result['forecast_seconds']=total
        result['synthetic_cost_below_admission_ceiling']=total<=6480
        result['status']='PROVISIONAL_REQUIRES_CALIBRATION_IO_AND_PEAK_GATES'
    else:result['status']='BLOCKED_MISSING_SATURATED_COSTS'
    result['illustrative_unmeasured_scenarios']=[
        {'mean_selection_seconds':u,'mean_nn_seconds':.02,
         'full_minutes_with_reservations_and_safety':(1920+1.25*(25200*u+35568*.02))/60}
        for u in (.05,.1,.2,.3)]
    with args.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
