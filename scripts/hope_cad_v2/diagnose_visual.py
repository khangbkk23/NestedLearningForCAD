"""Canonical W3 initialization diagnostics; no external assets are read."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from scripts.hope_cad_v2.diagnostics import visual_diagnostics
from scripts.hope_cad_v2.artifacts import provenance, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(103)
    model = HOPEFeaturePredictor().to(args.device)
    features = torch.randn(1,784,768,device=args.device)
    result = visual_diagnostics(model, features)
    write_json(args.output, dict(**provenance(), fixture_seed=103, outer_step=0,
                                features='synthetic Gaussian raw features', reportable=False, diagnostics=result))
    print('Contraction diagnostics saved:', args.output)


if __name__ == '__main__': main()
