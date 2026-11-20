"""CPU synthetic state-equivalence tests; not native CUDA/workload qualification."""
import copy,importlib.util,json,os,random,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch as t
import torch._dynamo, torch._inductor # Initialize native registrations before sys.modules fixtures.
import numpy as np
import safetensors.torch # Load native extension before temporary sys.modules fixtures.
import budget_checkpoint as cp
import budget_train as train
from nested_replay_cad.model.decoder import StableAdamW
from nested_replay_cad.replay import NestedReplay
from nested_replay_cad.core.learning.adam import adam_propose
SCRATCH=Path(os.environ['NRCAD_LOCAL_ROOT']);SCRATCH.mkdir(parents=True,exist_ok=True)
t.set_num_threads(1)

def fixture():
    t.use_deterministic_algorithms(True)
    t.backends.cuda.matmul.allow_tf32=False
    t.backends.cudnn.allow_tf32=False
    t.manual_seed(57);random.seed(71);np.random.seed(89)
    model=t.nn.Sequential(t.nn.Linear(4,4),t.nn.Dropout(.2),t.nn.Linear(4,1))
    opt=StableAdamW(model.parameters(),lr=.002,amsgrad=True)
    learner=train.Learner(t,model,opt,0);learner.ctrl=NestedReplay(0,'tail',device='cpu',normalize_context=True)
    schedule=train.Schedule(0);batches=train.Batches([f'{i:064x}' for i in range(40)],16,11)
    return model,opt,learner,schedule,batches,{},dict(arm='er_uniform',snapshots=[],data_selection_sha256=''),[]

def step(state,update):
    model,opt,learner,schedule,batches,store,study,rows=state
    ids=batches.next();replay,recent,historical,eligible=schedule.reserve([],2) if schedule.serial>=24 else ([],[],[],False)
    opt.zero_grad(set_to_none=True)
    x=t.rand(4,4)+float(np.random.random())+random.random()
    model(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
    ctrl=learner.ctrl;ctrl.eligible_commits+=1
    if update%16==0:
        grad={n:t.full_like(p,float(x.mean())*.001) for n,p in ctrl.phi.parameters.items()}
        phi,adam=adam_propose(ctrl.phi,ctrl.adam,grad)
        from nested_replay_cad.core.contracts import PhiState
        ctrl.phi=PhiState({n:v.detach().requires_grad_() for n,v in phi.parameters.items()});ctrl.adam=adam;ctrl.slow_steps+=1
    learner.counts['optimizer']=update;learner.counts['support_forwards']+=16
    schedule.admit(ids,update)
    for k in schedule.retained():
        if k not in store:store[k]=t.tensor([float(int(k,16))])
    for k in list(store):
        if k not in schedule.retained():del store[k]
    rows.append(dict(u=update,stage=1,current=ids,replay=replay,recent=recent,history=historical))
    study['data_selection_sha256']=cp.selection_digest(rows).hexdigest()

def capture(state,update):return cp.capture(t,*state[:5],state[5],state[6],state[7],1,update,{'fixture':'synthetic CPU only'})

def equal(test,a,b):
    if isinstance(a,t.Tensor):test.assertTrue(t.equal(a,b));return
    test.assertEqual(type(a),type(b))
    if isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal(test,x,y)
    else:test.assertEqual(a,b)

class CheckpointTests(unittest.TestCase):
    def test_cadence_is500_per_arm_and_final10005(self):
        self.assertEqual([n for n in range(1,10006) if cp.due(n)],list(range(500,10001,500))+[10005])
    def test_interrupted500_then_resume_matches_continuous520_bitwise(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            a=fixture()
            for n in range(1,501):step(a,n)
            saved=capture(a,500);rec=cp.save(t,Path(temp)/'run/er_uniform/checkpoints',saved)
            for n in range(501,521):step(a,n)
            expected=capture(a,520)
            # Fresh objects emulate a new process; restore all independent RNGs.
            b=fixture();loaded=cp.load(t,rec['path'],{'fixture':'synthetic CPU only'})
            store=cp.restore(t,loaded,*b[:5],{'fixture':'synthetic CPU only'})
            b=(*b[:5],store,loaded['study'],loaded['selection_rows'])
            for n in range(501,521):step(b,n)
            equal(self,expected,capture(b,520))
    def test_resume_in_fresh_cpu_process(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp);a=fixture()
            for n in range(1,501):step(a,n)
            record=cp.save(t,root/'run/er_uniform/checkpoints',capture(a,500))
            for n in range(501,521):step(a,n)
            expected=capture(a,520);out=root/'child.pt'
            code="""import sys;from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
import test_budget_checkpoint as f
b=f.fixture();loaded=f.cp.load(f.t,sys.argv[1],{'fixture':'synthetic CPU only'})
store=f.cp.restore(f.t,loaded,*b[:5],{'fixture':'synthetic CPU only'})
b=(*b[:5],store,loaded['study'],loaded['selection_rows'])
for n in range(501,521):f.step(b,n)
f.t.save(f.capture(b,520),sys.argv[2])
"""
            result=subprocess.run([sys.executable,'-I','-B','-c',code,record['path'],str(out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,encoding='utf8',timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            equal(self,expected,t.load(out,weights_only=True,map_location='cpu'))
    def test_corruption_identity_and_cursor_fail(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            a=fixture();step(a,1);saved=capture(a,1);rec=cp.save(t,Path(temp)/'run/er_uniform/checkpoints',saved)
            with self.assertRaises(ValueError):cp.load(t,rec['path'],{'wrong':True})
            bad_runtime=copy.deepcopy(saved);bad_runtime['runtime']['matmul_tf32']=not bad_runtime['runtime']['matmul_tf32']
            with self.assertRaisesRegex(ValueError,'runtime mismatch'):cp.restore(t,bad_runtime,*a[:5],saved['identity'])
            bad=copy.deepcopy(saved);bad['iteration']=2
            with self.assertRaisesRegex(ValueError,'cursor'):cp.restore(t,bad,*a[:5],{'fixture':'synthetic CPU only'})
            p=Path(rec['path'])/'state.pt'
            with p.open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash'):cp.load(t,rec['path'],{'fixture':'synthetic CPU only'})
    def test_bad_batch_and_memory_state_are_rejected(self):
        a=fixture();step(a,1);state=capture(a,1)
        before={n:p.detach().clone() for n,p in a[0].named_parameters()}
        for key in ('batch','memory','rng'):
            bad=copy.deepcopy(state)
            if key=='batch':bad['batches']['ids'][0]='wrong'
            elif key=='memory':bad['store'].pop(next(iter(bad['store'])))
            else:bad['schedule']['rng'].pop('recent')
            with self.assertRaises(ValueError):cp.restore(t,bad,*a[:5],state['identity'])
            for n,p in a[0].named_parameters():self.assertTrue(t.equal(p,before[n]))
    def test_commit_marker_cursor_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            a=fixture();step(a,1);state=capture(a,1);record=cp.save(t,Path(temp)/'run/er_uniform/checkpoints',state)
            marker=Path(record['path'])/'COMMIT.json';meta=json.loads(marker.read_text(encoding='utf8'));meta['update']=2;marker.write_text(json.dumps(meta),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'marker/payload'):cp.load(t,record['path'],state['identity'])
    def test_failed_publish_keeps_previous_checkpoint(self):
        for point in ('payload_written','before_publish'):
            with self.subTest(point=point),tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
                a=fixture();step(a,1);saved=capture(a,1);root=Path(temp)/'run/er_uniform/checkpoints';first=cp.save(t,root,saved)
                step(a,2)
                def fail(where):
                    if where==point:raise RuntimeError('injected interruption')
                with self.assertRaisesRegex(RuntimeError,'injected'):cp.save(t,root,capture(a,2),fault=fail)
                self.assertEqual(cp.load(t,first['path'],saved['identity'])['update'],1)
                with self.assertRaises(ValueError):cp.load(t,root/'g00002.partial',saved['identity'])
    def test_retention_two_complete_and_append_only_ledger(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            a=fixture();root=Path(temp)/'run/er_uniform/checkpoints'
            for n in (1,2,3):step(a,n);cp.save(t,root,capture(a,n))
            self.assertFalse((root/'g00001').exists());self.assertTrue((root/'g00002/COMMIT.json').is_file());self.assertTrue((root/'g00003/COMMIT.json').is_file())
            self.assertEqual(len((root/'ledger.jsonl').read_text(encoding='utf8').splitlines()),3)
            with self.assertRaises(FileExistsError):cp.save(t,root,capture(a,3))
    def test_actual_run_arm_checkpoint_and_resume_across_task_boundary(self):
        import contextlib,io,time
        from unittest import mock
        from nested_replay_cad.model import encoder,scoring
        from nested_replay_cad.data import ORDER
        original_learner=train.Learner;original_save=cp.save;original_write=train.write
        def fixture_write(path,value):
            # Windows scanners can transiently lock this synthetic progress JSON.
            # The production Linux writer and checkpoint atomic writer are unmodified.
            for attempt in range(10):
                try:return original_write(path,value)
                except PermissionError:
                    if sys.platform!='win32' or attempt==9:raise
                    time.sleep(.02)
        saved=[]
        class TinyLearner(original_learner):
            def step(self,current,replay,recent,history,store,metas,serial,update,eligible,learn,guard):
                self.model.train();self.opt.zero_grad(set_to_none=True)
                loss=self.model(t.stack(current)).square().mean();loss.backward();self.opt.step();self.opt.zero_grad(set_to_none=True)
                self.counts['optimizer']+=1;self.counts['support_forwards']+=len(current)+len(replay);self.counts['current_exposures']+=len(current);self.counts['replay_exposures']+=len(replay)
                return {'loss':float(loss.detach()),'slow':None}
        def new_model(_,seed):
            t.manual_seed(seed);model=t.nn.Sequential(t.nn.Linear(4,4),t.nn.Dropout(.2),t.nn.Linear(4,1));return model,StableAdamW(model.parameters(),lr=.002,amsgrad=True)
        def save_and_interrupt(*args,**kwargs):
            result=original_save(*args,**kwargs)
            if args[2]['update']==1000:
                saved.append(result);raise RuntimeError('injected stop after published1000')
            return result
        rows=[{'opaque_id':f'{c*40+i:064x}','file_sha256':'fixture','task_audit_only':category} for c,category in enumerate(ORDER) for i in range(40)]
        def observe(_,model,opt,learner,store,service,dataset,cohort,stage,update,output,deadline,guard,access):
            return {'stage':stage,'cursor':update,'metrics':{c:{} for c in ORDER[:stage]},'score_archives':[],'model_sha256':train.model_hash(model)}
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp);weights=root/'fixture.weights';weights.write_bytes(b'no model weights')
            with contextlib.ExitStack() as stack:
                if sys.platform=='win32':
                    from types import SimpleNamespace
                    stack.enter_context(mock.patch.dict(sys.modules,{'resource':SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:SimpleNamespace(ru_maxrss=0))}))
                for target,kwargs in [(mock.patch.object(train,'write',side_effect=fixture_write),{}),(mock.patch.object(train,'new_model',side_effect=new_model),{}),(mock.patch.object(train,'Learner',TinyLearner),{}),(mock.patch.object(train,'extract',side_effect=lambda _,service,data,row,access:t.full((4,),int(row['opaque_id'],16)/1000)),{}),(mock.patch.object(train,'observe',side_effect=observe),{}),(mock.patch.object(train,'resource_guard',return_value=lambda:None),{}),(mock.patch.object(encoder,'load',return_value=None),{}),(mock.patch.object(scoring,'gauss',return_value=None),{}),(mock.patch.object(t.cuda,'get_rng_state',side_effect=lambda *_:t.get_rng_state()),{}),(mock.patch.object(t.cuda,'set_rng_state',side_effect=lambda state,*_:t.set_rng_state(state)),{}),(mock.patch.object(t.cuda,'empty_cache'),{}),(mock.patch.object(t.cuda,'synchronize'),{}),(mock.patch.object(cp,'save',side_effect=save_and_interrupt),{})]:stack.enter_context(target)
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'published1000'):train.run_arm(t,'er_uniform',{'microbatch':4},rows,[],root,weights,root/'continuous/er_uniform',time.monotonic()+600)
                    with self.assertRaisesRegex(RuntimeError,'published1000'):train.run_arm(t,'er_uniform',{'microbatch':4},rows,[],root,weights,root/'resumed/er_uniform',time.monotonic()+600,resume_checkpoint=root/'continuous/er_uniform/checkpoints/g00500')
            self.assertEqual(len(saved),2)
            identity=json.loads((Path(saved[0]['path'])/'COMMIT.json').read_text(encoding='utf8'))['identity']
            a,b=[cp.load(t,r['path'],identity) for r in saved]
            for key in ('model','optimizer','controller','counts','schedule','batches','store','selection_rows','rng'):equal(self,a[key],b[key])
            self.assertEqual(a['stage'],2);self.assertEqual(a['iteration'],333)
            self.assertEqual(a['study']['snapshots'],b['study']['snapshots'])
            # A full cursor/cadence/observer/final path run with a tiny CPU model,
            # real optimizer and full3629 metadata. This is not the CAD workload.
            import gc
            rows_full=json.loads((ROOT/'configs/train_manifest.json').read_text(encoding='utf8'))['records']
            def safe_extract(_,service,data,row,access):return t.full((4,),int(row['opaque_id'][:4],16)/65536)
            with contextlib.ExitStack() as stack:
                if sys.platform=='win32':
                    from types import SimpleNamespace
                    stack.enter_context(mock.patch.dict(sys.modules,{'resource':SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:SimpleNamespace(ru_maxrss=0))}))
                for target in [mock.patch.object(train,'write',side_effect=fixture_write),mock.patch.object(train,'new_model',side_effect=new_model),mock.patch.object(train,'Learner',TinyLearner),mock.patch.object(train,'extract',side_effect=safe_extract),mock.patch.object(train,'observe',side_effect=observe),mock.patch.object(train,'resource_guard',return_value=lambda:None),mock.patch.object(encoder,'load',return_value=None),mock.patch.object(scoring,'gauss',return_value=None),mock.patch.object(t.cuda,'get_rng_state',side_effect=lambda *_:t.get_rng_state()),mock.patch.object(t.cuda,'set_rng_state',side_effect=lambda state,*_:t.set_rng_state(state)),mock.patch.object(t.cuda,'empty_cache'),mock.patch.object(t.cuda,'synchronize'),mock.patch.object(t.cuda,'max_memory_allocated',return_value=0),mock.patch.object(t.cuda,'max_memory_reserved',return_value=0),mock.patch.object(train.gc,'collect',return_value=0)]:stack.enter_context(target)
                with contextlib.redirect_stdout(io.StringIO()):
                    full=train.run_arm(t,'er_uniform',{'microbatch':4},rows_full,[],root,weights,root/'full/er_uniform',time.monotonic()+600)
                # Resume a fully evaluated10005 checkpoint: no repeated train/eval.
                with mock.patch.object(train,'observe',side_effect=AssertionError('Final boundary evaluated twice')),contextlib.redirect_stdout(io.StringIO()):
                    final_resume=train.run_arm(t,'er_uniform',{'microbatch':4},rows_full,[],root,weights,root/'final-resumed/er_uniform',time.monotonic()+600,resume_checkpoint=root/'full/er_uniform/checkpoints/g10005')
            # Real run_arm soft-stop and continuation, tiny CPU model only.
            # Force one pause mid-task and another inside evaluation, then compare
            # final model/optimizer/controller/sampler/RNG with uninterrupted run.
            identity=json.loads((root/'full/er_uniform/checkpoints/g10005/COMMIT.json').read_text(encoding='utf8'))['identity']
            expected=cp.load(t,root/'full/er_uniform/checkpoints/g10005',identity)
            for mode in ('step507','evaluation667'):
                switch={'stop':False,'injected':False};real_clock=time.monotonic
                class PausingLearner(TinyLearner):
                    def step(self,*args,**kwargs):
                        result=super().step(*args,**kwargs)
                        if mode=='step507' and self.counts['optimizer']==507:switch['stop']=True
                        return result
                def clock():return real_clock()+ (2000000 if switch['stop'] else 0)
                def pausing_observe(*args,**kwargs):
                    if mode=='evaluation667' and not switch['injected']:
                        switch['injected']=True
                        dest=args[10]/'scores'/('g'+str(args[9]));dest.mkdir(parents=True)
                        (dest/'partial.fixture').write_text('unfinished',encoding='utf-8')
                        t.rand(1) # Must be rolled back by transactional evaluation.
                        raise train.SessionPause('synthetic evaluation pause')
                    return observe(*args,**kwargs)
                # These fixtures use the same patch set as the full CPU run.
                with contextlib.ExitStack() as stack:
                    if sys.platform=='win32':
                        from types import SimpleNamespace
                        stack.enter_context(mock.patch.dict(sys.modules,{'resource':SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:SimpleNamespace(ru_maxrss=0))}))
                    for target in [mock.patch.object(train,'write',side_effect=fixture_write),mock.patch.object(train,'new_model',side_effect=new_model),mock.patch.object(train,'Learner',PausingLearner),mock.patch.object(train,'extract',side_effect=safe_extract),mock.patch.object(train,'observe',side_effect=pausing_observe),mock.patch.object(train,'resource_guard',return_value=lambda:None),mock.patch.object(encoder,'load',return_value=None),mock.patch.object(scoring,'gauss',return_value=None),mock.patch.object(t.cuda,'get_rng_state',side_effect=lambda *_:t.get_rng_state()),mock.patch.object(t.cuda,'set_rng_state',side_effect=lambda state,*_:t.set_rng_state(state)),mock.patch.object(t.cuda,'empty_cache'),mock.patch.object(t.cuda,'synchronize'),mock.patch.object(t.cuda,'max_memory_allocated',return_value=0),mock.patch.object(t.cuda,'max_memory_reserved',return_value=0),mock.patch.object(train.gc,'collect',return_value=0),mock.patch.dict(os.environ,{'NRCAD_SESSION_SOFT_STOP_MONOTONIC':str(real_clock()+1000000)}),mock.patch.object(train.time,'monotonic',side_effect=clock)]:stack.enter_context(target)
                    with contextlib.redirect_stdout(io.StringIO()):
                        paused=train.run_arm(t,'er_uniform',{'microbatch':4},rows_full,[],root,weights,root/mode/'er_uniform',real_clock()+3000000)
                        self.assertEqual(paused['status'],'PAUSED_RESUMABLE')
                        self.assertEqual(paused['counts']['optimizer'],507 if mode=='step507' else 667)
                        self.assertFalse((root/mode/'er_uniform/scores/g667/partial.fixture').exists())
                        switch['stop']=False
                        resumed=train.run_arm(t,'er_uniform',{'microbatch':4},rows_full,[],root,weights,root/(mode+'-resume')/'er_uniform',real_clock()+3000000,resume_checkpoint=Path(paused['resume_checkpoint']['path']))
                    self.assertEqual(resumed['status'],'COMPLETE_REVIEW_PENDING')
                    actual=cp.load(t,Path(resumed['resume_checkpoint']['path']),identity)
                    for k in ('model','optimizer','controller','counts','schedule','batches','store','selection_rows','rng'):equal(self,expected[k],actual[k])
                    self.assertEqual(actual['study']['snapshots'],expected['study']['snapshots'])
            self.assertEqual(final_resume['status'],'COMPLETE_REVIEW_PENDING')
            self.assertEqual(final_resume['counts'],full['counts'])
            self.assertEqual(final_resume['snapshots'],full['snapshots'])
            self.assertEqual(full['status'],'COMPLETE_REVIEW_PENDING');self.assertEqual(full['counts']['optimizer'],10005)
            self.assertEqual(full['counts']['support_forwards'],160080);self.assertEqual(len(full['snapshots']),15)
            ledger=[json.loads(line) for line in (root/'full/er_uniform/checkpoints/ledger.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([x['update'] for x in ledger],list(range(500,10001,500))+[10005])
            self.assertEqual(sorted(p.name for p in (root/'full/er_uniform/checkpoints').glob('g[0-9]*')),['g10000','g10005'])

    def test_copy_parent_evidence_hash_and_no_parent_mutation(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp);old=root/'old/er_uniform';new=root/'new/er_uniform';old.mkdir(parents=True);new.mkdir(parents=True)
            archive=old.parent/'er_uniform_g667_scores_part001.zip';archive.write_bytes(b'synthetic fixture')
            record={'path':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
            study={'arm':'er_uniform','snapshots':[{'score_archives':[record]}]}
            cp.copy_evidence(old,new,study);self.assertEqual(archive.read_bytes(),(new.parent/archive.name).read_bytes())
            with self.assertRaises(FileExistsError):cp.copy_evidence(old,new,study)
            record['sha256']='0'*64;fresh=root/'bad/er_uniform';fresh.mkdir(parents=True)
            with self.assertRaises(ValueError):cp.copy_evidence(old,fresh,study)
if __name__=='__main__':unittest.main()
