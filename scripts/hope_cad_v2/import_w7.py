"""Import the reviewed W6 prefix/initializations without fitting or rerunning W6."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from training.hope_w7_io import import_w6

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--w6-evidence',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--features',action='store_true')
    args=p.parse_args()
    import_w6(args.w6_evidence,args.output_dir,include_features=args.features)
    print('Reviewed W6 imported; no learned weights fitted or selected again')

if __name__=='__main__':main()
