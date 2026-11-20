"""Synthetic B321 native CUDA checkpoint qualification. No dataset/weights/AD."""
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import parallel_budget as parallel

def fingerprint(t,value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,t.Tensor):
            x=x.detach().cpu().contiguous();h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def worker(args):
    parallel.validate_slot(os.environ,args.arm)
    from nested_replay_cad.trainer import configure
    import budget_train as train
    import budget_checkpoint as cp
    t=configure();binding=parallel.gpu_binding(t,os.environ,args.arm)
    args.output.mkdir(parents=True,exist_ok=False)
    deadline=float(os.environ['NRCAD_RUN_DEADLINE_MONOTONIC']);guard=train.resource_guard(t,args.output,deadline)
    model,opt=train.new_model(t,0);learner=train.Learner(t,model,opt,0,4);schedule=train.Schedule(0)
    ids=[f'{i:064x}' for i in range(40)];bank=t.randn((40,2,789,768),generator=t.Generator().manual_seed(9251))
    schedule.admit(ids,0);store={k:bank[int(k,16)].clone() for k in schedule.retained()};batches=train.Batches(ids,8,813)
    study={'arm':args.arm,'snapshots':[],'data_selection_sha256':hashlib.sha256().hexdigest()};rows=[]
    identity={'fixture':'B321-native-resume-01','arm':args.arm,'source_sha256':{p:train.digest(ROOT/p) for p in ['scripts/budget_train.py','scripts/budget_checkpoint.py','scripts/qualify_budget_resume.py']}}
    start=1
    if args.phase=='resume':
        state=cp.load(t,args.parent/args.arm/'checkpoints/g00015',identity)
        store=cp.restore(t,state,model,opt,learner,schedule,batches,identity);rows=state['selection_rows'];study=state['study'];start=16;del state;gc.collect()
    saves=[]
    for update in range(start,18):
        current=batches.next();replay,recent,history,eligible=schedule.reserve(current,2)
        if not eligible:raise ValueError('Synthetic query fixture not eligible')
        learner.step([bank[int(k,16)] for k in current],replay,recent,history,store,schedule.metas,schedule.serial,update,True,args.arm=='a0_tail_ctx',guard)
        rows.append({'u':update,'stage':1,'current':current,'replay':replay,'recent':recent,'history':history});study['data_selection_sha256']=cp.selection_digest(rows).hexdigest()
        if update==15:
            state=cp.capture(t,model,opt,learner,schedule,batches,store,study,rows,1,15,identity)
            saves.append(cp.save(t,args.output/'checkpoints',state,guard));del state;gc.collect()
        print('QUALIFICATION_UPDATE',args.phase,args.arm,update,flush=True)
    state=cp.capture(t,model,opt,learner,schedule,batches,store,study,rows,1,17,identity)
    hashes={key:fingerprint(t,state[key]) for key in ('model','optimizer','controller','counts','schedule','batches','store','selection_rows','rng','runtime')}
    del state;gc.collect();model.eval()
    with t.no_grad():prediction=model(bank[:1,0].cuda(),bank[:1,1].cuda())
    hashes['prediction']=fingerprint(t,prediction)
    result={'status':'COMPLETE_REVIEW_PENDING','phase':args.phase,'arm':args.arm,'binding':binding,'hashes':hashes,'slow_steps':learner.ctrl.slow_steps if learner.ctrl else 0,'checkpoint':saves,'peak_allocated_VRAM_bytes':t.cuda.max_memory_allocated(),'scope':'Synthetic17updates; split15/16due-slow/17. Cadence500 qualified separately onCPU. NoADdata.'}
    parallel.write(args.output/'result.json',result)
    return 0

def parent(args):
    parallel.require_owned_process_group();args.output.mkdir(parents=True,exist_ok=False)
    deadline=min(time.monotonic()+1200,float(os.environ['NRCAD_RUN_DEADLINE_MONOTONIC']))
    monitor=parallel.GroupMonitor(args.output,deadline);devices=parallel.physical_devices();result={'status':'FAILED','scientific_acceptance':False,'data_access':False}
    try:
        for phase in ('original','resume'):
            commands={};environments={}
            for arm in parallel.ARMS:
                env=parallel.child_environment(os.environ,arm,'profile' if phase=='original' else 'train',deadline,args.output/'inputs',args.output/'unused',devices[parallel.SLOTS[arm]]['uuid'])
                runtime=Path(env['NRCAD_LOCAL_ROOT']);(runtime/'tmp').mkdir(parents=True,exist_ok=False)
                environments[arm]=env
                commands[arm]=[sys.executable,'-B',str(Path(__file__).resolve()),'--phase',phase,'--arm',arm,'--parent',str(args.output/'original'),'--output',str(args.output/phase/arm)]
            result[phase]=parallel.run_phase(commands,environments,args.output/'logs'/phase,deadline,monitor)
        pairs={}
        for arm in parallel.ARMS:
            a=parallel.read(args.output/'original'/arm/'result.json');b=parallel.read(args.output/'resume'/arm/'result.json')
            same=a['hashes']==b['hashes'] and a['binding']['physical_uuid']==b['binding']['physical_uuid']
            if arm=='a0_tail_ctx':same=same and a['slow_steps']==b['slow_steps']==1
            pairs[arm]={'bitwise_match':same,'original':a,'resume':b}
        result.update(pairs=pairs,status='PASS_SCOPED' if all(v['bitwise_match'] for v in pairs.values()) else 'FAIL_MISMATCH')
        if result['status']!='PASS_SCOPED':raise ValueError('Native resume mismatch; do not launch longrun')
        return 0
    except BaseException as error:result['error']=repr(error);traceback.print_exc();return 1
    finally:
        result['resources']=monitor.latest;parallel.write(args.output/'result.json',result);print('NATIVE_QUALIFICATION_TERMINAL',result['status'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('original','resume'));p.add_argument('--arm',choices=parallel.ARMS);p.add_argument('--parent',type=Path);a=p.parse_args()
    if bool(a.phase)!=bool(a.arm):p.error('phase and arm required together')
    return worker(a) if a.phase else parent(a)
if __name__=='__main__':raise SystemExit(main())
