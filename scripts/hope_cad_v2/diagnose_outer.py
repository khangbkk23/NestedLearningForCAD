"""Post-outer visual diagnostics at 0/1/5/25 on an explicit synthetic fixture."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from training.hope_outer_learning_v2 import fit, make_schedule
from scripts.hope_cad_v2.synthetic import feature_fixture
from scripts.hope_cad_v2.diagnostics import visual_diagnostics
from scripts.hope_cad_v2.artifacts import provenance, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--reduced',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(103)
    config = VisualBlockConfig(8,8,8,4,4,4) if args.reduced else VisualBlockConfig()
    model = HOPEFeaturePredictor(config).to(args.device)
    data = feature_fixture(config,args.device)
    features,_,_ = data.get(data.ids('meta_support')[0],0,'meta_support')
    report = dict(**provenance(),scope='synthetic 25-step diagnostics, not a scientific candidate search',
                  canonical=config.canonical,reportable=False,checkpoints={})
    def record(step,model):
        report['checkpoints'][step] = visual_diagnostics(model,features)
        write_json(args.output,report)
    result = fit(model,data,make_schedule(data,0,steps=25),max_steps=25,
                 checkpoints=(0,25),smoke=True,diagnostic_callback=record)
    report['fit'] = {k:v for k,v in result.items() if k != 'shortlist'}
    report['status'] = result['status']
    write_json(args.output,report)
    print(result['status'])
    return 0 if result['status']=='fit_complete' else 1


if __name__ == '__main__': raise SystemExit(main())
