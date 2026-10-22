"""One SSH-friendly interface; no automatic rentals, experiments or external services."""
import argparse,json,sys

def parser():
    p=argparse.ArgumentParser(prog='python -m hope_cad')
    sub=p.add_subparsers(dest='command',required=True)
    pf=sub.add_parser('preflight',help='Discarded local smoke or real Linux/GPU acceptance')
    pf.add_argument('--run-dir',required=True);pf.add_argument('--device',default='cuda:0')
    pf.add_argument('--local-smoke',action='store_true')
    pf.add_argument('--expected-gpu',default='NVIDIA GeForce RTX 5090')
    pf.add_argument('--data-root');pf.add_argument('--checkpoint')
    for name in ('train','evaluate','matrix'):
        q=sub.add_parser(name)
        q.add_argument('--run-dir',required=True);q.add_argument('--job',required=True)
        q.add_argument('--data-root',required=True);q.add_argument('--checkpoint',required=True)
        q.add_argument('--preflight',required=True,help='Passed remote preflight directory')
        q.add_argument('--device',default='cuda:0');q.add_argument('--session-hours',type=float,default=8.)
    q=sub.add_parser('aggregate');q.add_argument('--inputs',nargs='+',required=True)
    q.add_argument('--output',required=True);q.add_argument('--statistics',action='store_true')
    q.add_argument('--compare',default='CADIC')
    q=sub.add_parser('export');q.add_argument('--run-dir',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('source',help='Build self-contained source ZIP, no artifacts/runtime')
    q.add_argument('--output',required=True)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='preflight':
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
