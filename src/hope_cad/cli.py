"""One SSH-friendly interface; no automatic rentals, experiments or external services."""
import argparse,json,sys
from hope_cad.paths import ROOT, DATA, WEIGHTS

DATA_ROOT=str(DATA/'mvtec')
CHECKPOINT=str(WEIGHTS/'vit_base_patch8_224.augreg_in21k.safetensors')

def parser():
    p=argparse.ArgumentParser(prog='python -m hope_cad')
    sub=p.add_subparsers(dest='command',required=True)
    pf=sub.add_parser('preflight',help='Discarded local smoke or real Linux/GPU acceptance')
    pf.add_argument('--run-dir',required=True);pf.add_argument('--device',default='cuda:0')
    pf.add_argument('--local-smoke',action='store_true')
    pf.add_argument('--expected-gpu',default='NVIDIA GeForce RTX 5090')
    pf.add_argument('--data-root',default=DATA_ROOT);pf.add_argument('--checkpoint',default=CHECKPOINT)
    for name in ('train','evaluate','matrix'):
        q=sub.add_parser(name)
        q.add_argument('--run-dir',required=True);q.add_argument('--job',required=True)
        q.add_argument('--data-root',default=DATA_ROOT);q.add_argument('--checkpoint',default=CHECKPOINT)
        q.add_argument('--preflight',required=True,help='Passed remote preflight directory')
        q.add_argument('--device',default='cuda:0');q.add_argument('--session-hours',type=float,default=8.)
    q=sub.add_parser('aggregate');q.add_argument('--inputs',nargs='+',required=True)
    q.add_argument('--output',required=True);q.add_argument('--statistics',action='store_true')
    q.add_argument('--compare',default='CADIC')
    q=sub.add_parser('export');q.add_argument('--run-dir',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('source',help='Build self-contained source ZIP, no artifacts/runtime')
    q.add_argument('--output',required=True)
    q=sub.add_parser('attribution',help='CPU normal-event D0 analysis; requires the two SHA-locked historical ZIPs')
    q.add_argument('--input-root',required=True);q.add_argument('--output',required=True)
    q.add_argument('--nested-archive');q.add_argument('--cadic-archive')
    q=sub.add_parser('research',help='Fresh full-train/H32 study on an explicitly provisioned remote CUDA machine')
    q.add_argument('--phase',choices=('preflight','pilot','main','extension'),required=True)
    q.add_argument('--run-dir',required=True);q.add_argument('--data-root',default=DATA_ROOT);q.add_argument('--checkpoint',default=CHECKPOINT)
    q.add_argument('--job',default='H32_EXACT_seed0');q.add_argument('--device',default='cuda:0')
    q.add_argument('--expected-gpu',default='NVIDIA GeForce RTX 5090');q.add_argument('--session-hours',type=float,default=4.)
    q.add_argument('--external-billed-seconds',type=float,default=0.,help='Unrecorded setup/idle billed time, counted once in stage ledger')
    q.add_argument('--allocation',help='Stage-D user allocation JSON; never auto-rents a GPU')
    q=sub.add_parser('research-aggregate',help='CPU summary/paired finite-seed uncertainty for successor cohort')
    q.add_argument('--inputs',nargs='+',required=True);q.add_argument('--output',required=True)
    q.add_argument('--statistics',action='store_true');q.add_argument('--compare',default='CADIC')
    q=sub.add_parser('research-review',help='Explicit operator acceptance AFTER reviewing a complete preflight/pilot')
    q.add_argument('--run-dir',required=True);q.add_argument('--phase',choices=('preflight','pilot'),required=True)
    q.add_argument('--accept',action='store_true',required=True);q.add_argument('--note',required=True)
    q=sub.add_parser('cadic-diagnostic',help='Standalone full-train CADIC K2500/K5000 host timing/QA before study preflight')
    q.add_argument('--phase',choices=('train','evaluate'),default='train')
    q.add_argument('--run-dir',required=True)
    q.add_argument('--data-root',default=DATA_ROOT);q.add_argument('--checkpoint',default=CHECKPOINT)
    q.add_argument('--budgets',nargs='+',type=int,choices=(2500,5000),default=[2500,5000])
    q.add_argument('--device',default='cuda:0');q.add_argument('--expected-gpu')
    q.add_argument('--max-hours',type=float,default=1.)
    q=sub.add_parser('handoff-pack',help='One verified research experiment ZIP for download; no training')
    q.add_argument('--run-dir',required=True);q.add_argument('--output')
    q=sub.add_parser('handoff-check',help='Exact-identity gate/optional restore; never launches a stage')
    q.add_argument('--archive');q.add_argument('--restore',action='store_true')
    q.add_argument('--device',default='cuda:0');q.add_argument('--expected-gpu')
    q.add_argument('--data-root',default=DATA_ROOT);q.add_argument('--checkpoint',default=CHECKPOINT)
    q.add_argument('--rate',type=float);q.add_argument('--cap',type=float)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='handoff-pack':
            from hope_cad.runtime.handoff import pack
            result=pack(args.run_dir,args.output)
        elif args.command=='handoff-check':
            from hope_cad.runtime.handoff import check
            result=check(args.archive,restore=args.restore,device=args.device,expected_gpu=args.expected_gpu,
                data_root=args.data_root,checkpoint=args.checkpoint,rate=args.rate,cap=args.cap)
        elif args.command=='cadic-diagnostic':
            from hope_cad.runtime.cadic_diagnostic import run
            result=run(vars(args))
        elif args.command=='attribution':
            import os
            os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
            from hope_cad.benchmark.attribution import run
            result=run(args.input_root,args.output,nested_archive=args.nested_archive,cadic_archive=args.cadic_archive)
        elif args.command=='research-review':
            from hope_cad.runtime.research import record_review
            result=record_review(args.run_dir,args.phase,args.note)
        elif args.command=='research':
            from hope_cad.runtime.research import run
            result=run(vars(args))
        elif args.command=='research-aggregate':
            from hope_cad.benchmark.statistics import aggregate
            result=aggregate(args.inputs,args.output,statistics=args.statistics,compare=args.compare,research=True)
        elif args.command=='preflight':
            from hope_cad.runtime.preflight import run
            result=run(vars(args))
        elif args.command in ('train','evaluate','matrix'):
            from hope_cad.runtime.execution import run
            result=run(vars(args))
        elif args.command=='aggregate':
            from hope_cad.benchmark.statistics import aggregate
            result=aggregate(args.inputs,args.output,statistics=args.statistics,compare=args.compare)
        elif args.command=='source':
            from hope_cad.runtime.shipping import package
            result=package(args.output)
        else:
            from hope_cad.runtime.evidence import export_run
            result=export_run(args.run_dir,args.output)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return 0
    except KeyboardInterrupt:
        print('Interrupted: preserve run directory; inspect last committed checkpoint before resume.',file=sys.stderr,flush=True)
        return 130
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        return 1
