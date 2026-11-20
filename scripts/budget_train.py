"""B321 grouped replay: prospective budget variant, never changes B319 source."""
import argparse,gc,hashlib,json,math,os,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n');q.replace(p)


def append(path,value):
    with Path(path).open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


class Batches:
    """Seeded without-replacement epochs; incomplete tails are dropped."""
    def __init__(self,ids,size,seed):
        self.ids=tuple(ids);self.size=size;self.rng=random.Random(seed);self.queue=[]
        if len(self.ids)<size:raise ValueError('Batch larger than current task')
    def next(self):
        if len(self.queue)<self.size:
            self.queue=list(self.ids);self.rng.shuffle(self.queue)
            self.queue=self.queue[:len(self.queue)//self.size*self.size]
        batch,self.queue=self.queue[:self.size],self.queue[self.size:]
        if len(set(batch))!=self.size:raise ValueError('Padded or duplicated current batch')
        return batch


class Schedule:
    def __init__(self,seed):
        from nested_replay_cad.core.memory.rng import seed_sampling
        self.admission=seed_sampling(seed).admission;self.reservoir=[];self.fifo=[]
        self.seen=set();self.metas={};self.serial=0
        self.rng={n:random.Random(seed+v) for n,v in [('recent',301),('historical',307),('replay',311),('partition',313)]}
    def reserve(self,current,stage):
        if stage==1:return [],[],[],False
        excluded=set(current)
        recent_pool=[k for k in self.fifo if k not in excluded]
        recent=self.rng['recent'].sample(recent_pool,min(2,len(recent_pool)))
        old=[k for k in self.reservoir if k not in set(self.fifo)|excluded]
        history=self.rng['historical'].sample(old,min(2,len(old)))
        blocked=excluded|set(recent)|set(history)
        pool=[k for k in self.reservoir if k not in blocked]
        if len(pool)<8:raise ValueError('Cannot reserve eight distinct replay supports without padding')
        replay=self.rng['replay'].sample(pool,8);self.rng['partition'].shuffle(replay)
        roles=list(current)+recent+history+replay
        if len(set(roles))!=len(roles):raise ValueError('Role overlap')
        return replay,recent,history,len(recent)==len(history)==2
    def admit(self,ids,update):
        from nested_replay_cad.core.memory.sampler import admission_slot
        for k in ids:
            if k in self.seen:continue
            self.seen.add(k);self.serial+=1
            slot,self.admission=admission_slot(self.admission,self.serial)
            if slot is not None:
                if self.serial<=24:self.reservoir.append(k)
                else:self.reservoir[slot]=k
            self.fifo=(self.fifo+[k])[-8:]
            self.metas[k]={'admission_serial':self.serial,'first_update':update}
    def retained(self):return set(self.reservoir)|set(self.fifo)


def patch_loss(t,target,prediction):
    return t.stack([1-t.nn.functional.cosine_similarity(a,b,dim=1) for a,b in zip(target,prediction)]).mean(0)


def sample_losses(t,official,target,prediction,p):
    return [official.global_cosine_hm_percent([x[i:i+1] for x in target],[x[i:i+1] for x in prediction],p=p,factor=.1) for i in range(target[0].shape[0])]


def alignment_value(adjoint_gradient_pairs):
    return .25*math.fsum(float((adjoint*gradient.detach().double()).sum()) for adjoint,gradient in adjoint_gradient_pairs)


def grouped_objective(t,weights,alignments,include_kl=True):
    query=sum((w.double()*t.tensor(alignments[g*4:g*4+4],dtype=t.float64)).sum() for g,w in enumerate(weights))
    kl=sum((w*t.log(4*w)).sum() for w in weights)/2
    return query+(.001*kl if include_kl else 0)


class Learner:
    def __init__(self,t,model,opt,seed,micro=4):
        self.t=t;self.model=model;self.opt=opt;self.micro=micro;self.seed=seed
        self.ctrl=None
        self.counts={n:0 for n in ('optimizer','support_forwards','current_exposures','replay_exposures','context_forwards','query_forwards','alignment_forwards')}
    def step(self,current,replay,recent,history,store,metas,serial,update,eligible,learn,guard):
        t=self.t
        from nested_replay_cad.model import decoder as official
        from nested_replay_cad.context import normalize_contexts
        from nested_replay_cad.fast_update import functional_step,conditional_vjp
        from nested_replay_cad.core.contracts import PhiState
        from nested_replay_cad.core.learning.adam import adam_propose
        if learn and self.ctrl is None:
            from nested_replay_cad.replay import NestedReplay
            self.ctrl=NestedReplay(self.seed,'tail',normalize_context=True)
        model,opt=self.model,self.opt;params=dict(model.named_parameters());model.train();opt.zero_grad(set_to_none=True)
        p=.9*min(update/26,1.);loss_total=0.
        current_n=len(current)
        for start in range(0,current_n,self.micro):
            guard();x=t.stack(current[start:start+self.micro]).to('cuda:0')
            a,b=model(x[:,0],x[:,1]);ls=sample_losses(t,official,a,b,p)
            coefficient=(.5/current_n) if replay else 1/current_n
            value=sum(ls)*coefficient;value.backward();loss_total+=float(value.detach())
            del a,b,ls,value,x
        self.counts['current_exposures']+=current_n
        contexts=[];draws=[];weights=[]
        if replay:
            if learn:
                current_mean=t.stack([x[:,5:].mean((0,1)) for x in current]).mean(0).to('cuda:0')
                with t.no_grad():
                    for group in range(2):
                        guard();draws.append(t.cuda.get_rng_state(0).clone())
                        ids=replay[group*4:group*4+4];x=t.stack([store[k] for k in ids]).to('cuda:0')
                        a,b=model(x[:,0],x[:,1]);patch=patch_loss(t,a,b);rows=[]
                        for i,k in enumerate(ids):
                            age=serial-metas[k]['admission_serial']
                            if age<0 or metas[k]['first_update']>=update:raise ValueError('Replay causality')
                            old=x[i,:,5:].mean((0,1));sim=t.nn.functional.cosine_similarity(old[None],current_mean[None],dim=1)[0]
                            z=patch[i].flatten();rows.append(t.stack((z.mean()/2,t.topk(z,79).values.mean()/2,z.new_tensor(age/(age+32)),(1+sim)/2)))
                        contexts.append(normalize_contexts(t.stack(rows).detach().cpu().contiguous()))
                        del a,b,patch,rows,x
                self.counts['context_forwards']+=8
                weights=[self.ctrl.policy.weights(self.ctrl.phi,c) if eligible else t.full((4,),.25) for c in contexts]
            else:weights=[t.full((4,),.25),t.full((4,),.25)]
            for group in range(2):
                guard()
                if learn:t.cuda.set_rng_state(draws[group],0)
                x=t.stack([store[k] for k in replay[group*4:group*4+4]]).to('cuda:0')
                a,b=model(x[:,0],x[:,1]);ls=sample_losses(t,official,a,b,p)
                value=sum(.25*weights[group][i].detach().to('cuda:0')*ls[i] for i in range(4))
                value.backward();loss_total+=float(value.detach());del a,b,ls,value,x
            self.counts['replay_exposures']+=8
        self.counts['support_forwards']+=current_n+len(replay)
        if any(v.grad is None or not t.isfinite(v.grad).all() for v in params.values()):raise ValueError('Missing/nonfinite batch gradient')
        due=learn and eligible and (self.ctrl.eligible_commits+1)%16==0
        slow=None;next_phi,next_adam=(self.ctrl.phi,self.ctrl.adam) if self.ctrl is not None else (None,None);expected=None
        final_rng=t.cuda.get_rng_state(0).clone()
        if due:
            gradient={n:v.grad.detach() for n,v in params.items()};opt.zero_grad(set_to_none=True)
            parent={n:v.detach().clone() for n,v in params.items()}
            if any(opt.state[v]['step']!=update-1 for v in params.values()):raise ValueError('Fast optimizer clock mismatch')
            moments={n:{k:opt.state[v][k] for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq')} for n,v in params.items()}
            with t.no_grad():functional_step(parent,gradient,moments,update-1,destination=params)
            expected={n:v.detach().cpu().clone() for n,v in params.items()}
            model.eval();model.training=True;query_values=[]
            for i,k in enumerate(recent+history):
                guard();x=store[k].unsqueeze(0).to('cuda:0');a,b=model(x[:,0],x[:,1]);z=patch_loss(t,a,b).flatten()
                q=t.sort(z,descending=True,stable=True).values[:79].mean() if i>=2 else z.mean()
                query_values.append(float(q.detach()));(.25*q).backward();del a,b,z,q,x
            self.counts['query_forwards']+=4
            if not t.equal(final_rng,t.cuda.get_rng_state(0)):raise ValueError('Query advanced dropout RNG')
            query={n:v.grad.detach() for n,v in params.items()};opt.zero_grad(set_to_none=True)
            adjoint,branches=conditional_vjp(gradient,moments,update-1,query);del query
            with t.no_grad():
                for n,v in params.items():v.copy_(parent[n])
            model.train();align=[]
            for group in range(2):
                guard();t.cuda.set_rng_state(draws[group],0)
                x=t.stack([store[k] for k in replay[group*4:group*4+4]]).to('cuda:0')
                a,b=model(x[:,0],x[:,1]);ls=sample_losses(t,official,a,b,p)
                for i in range(4):
                    gi=t.autograd.grad(ls[i],tuple(params.values()),retain_graph=i<3)
                    align.append(alignment_value((adjoint[n],v) for n,v in zip(params,gi)))
                    del gi
                del a,b,ls,x
            self.counts['alignment_forwards']+=8
            if not t.equal(final_rng,t.cuda.get_rng_state(0)):raise ValueError('Replay restore advanced dropout RNG')
            objective=grouped_objective(t,weights,align)
            deriv=t.autograd.grad(objective,tuple(self.ctrl.phi.parameters.values()))
            pg={n:v.detach().clone() for n,v in zip(self.ctrl.phi.parameters,deriv)}
            next_phi,next_adam=adam_propose(self.ctrl.phi,self.ctrl.adam,pg)
            slow={'query_values':query_values,'alignments':align,'phi_gradient_norm':math.sqrt(math.fsum(float(v.double().square().sum()) for v in pg.values())),'global_clip_active':branches['global_clip_active']}
            for n,v in params.items():v.grad=gradient[n]
            del adjoint,parent,moments
        norm=t.nn.utils.clip_grad_norm_(model.parameters(),.1)
        if not t.isfinite(norm):raise ValueError('Nonfinite clipped norm')
        opt.step();self.counts['optimizer']+=1
        if any(not t.isfinite(v).all() for v in params.values()):raise ValueError('Nonfinite committed parameters')
        if due:
            for n,v in params.items():t.testing.assert_close(v.detach().cpu(),expected[n],atol=1e-6,rtol=1e-5)
            self.ctrl.phi=PhiState({n:v.detach().clone().requires_grad_() for n,v in next_phi.parameters.items()});self.ctrl.adam=next_adam
            self.ctrl.slow_steps+=1;slow['step']=self.ctrl.slow_steps
            del expected,gradient
        if learn and eligible:self.ctrl.eligible_commits+=1
        opt.zero_grad(set_to_none=True)
        return {'loss':loss_total,'gradient_norm':float(norm),'slow':slow,'weights':[w.detach().tolist() for w in weights]}


def model_hash(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def new_model(t,seed):
    from nested_replay_cad.model import decoder as official
    t.manual_seed(seed);model=official.Decoder(checkpoint=True);official.init_trainable(model);model.to('cuda:0')
    return model,official.make_optimizer(model)


def resource_guard(t,output,deadline):
    def guard():
        import resource
        if time.monotonic()>deadline-120:raise TimeoutError('B321 worker/whole-job cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise MemoryError('RSS8GiB cap')
        if t.cuda.max_memory_allocated(0)>12*1024**3:raise MemoryError('VRAM12GiB cap')
        from nested_replay_cad.storage import DiskGuard
        guard.disk.check()
    from nested_replay_cad.storage import DiskGuard
    guard.disk=DiskGuard(output)
    return guard


def extract(t,service,dataset,row,access,*,test=False):
    from nested_replay_cad.core.data.manifest import read_verified_bytes,checked_path
    from nested_replay_cad.model import encoder
    relative=row['path'] if test else row['relative_path'];expected=row['image_sha256'] if test else row['file_sha256']
    raw=read_verified_bytes(dataset,relative);actual=hashlib.sha256(raw).hexdigest()
    append(access,{'operation':'read','kind':'MVTec_test' if test else 'train_good','path':relative,'sha256':actual,'bytes':len(raw)})
    if actual!=expected or len(raw)!=row['bytes']:raise ValueError('Image identity mismatch')
    with t.no_grad():
        x=encoder.preprocess(checked_path(dataset,relative)).unsqueeze(0).to('cuda:0');feature=service.groups(x).detach()[0].cpu().contiguous()
    if feature.shape!=(2,789,768) or feature.dtype!=t.float32 or not t.isfinite(feature).all():raise ValueError('Frozen feature contract')
    return feature


def observe(t,model,opt,learner,store,service,dataset,cohort,stage,update,output,deadline,guard,access):
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from safetensors.torch import save_file
    from nested_replay_cad.core.data.manifest import read_verified_bytes
    from nested_replay_cad.model import scoring
    from nested_replay_cad.metrics import category_metrics,memory_report
    from nested_replay_cad.data import ORDER
    from nested_replay_cad.storage import archive_tensors
    rows=[r for r in cohort if r['category'] in ORDER[:stage]];dest=output/'scores'/f'g{update}';dest.mkdir(parents=True,exist_ok=False)
    before=model_hash(model);cpu=t.get_rng_state().clone();cuda=t.cuda.get_rng_state(0).clone();model.eval();scores=[]
    started=time.monotonic()
    for index,row in enumerate(rows):
        guard();f=extract(t,service,dataset,row,access,test=True).unsqueeze(0).to('cuda:0')
        with t.no_grad():a,b=model(f[:,0],f[:,1]);pixel,image=scoring.dinomaly(a,b)
        pixel,image=pixel[0].cpu().contiguous(),image[0].cpu().contiguous()
        if row['image_label']==0:gt=t.zeros((256,256),dtype=t.uint8);original=0
        else:
            raw=read_verified_bytes(dataset,row['mask_path']);h=hashlib.sha256(raw).hexdigest()
            append(access,{'operation':'read','kind':'MVTec_mask','path':row['mask_path'],'sha256':h,'bytes':len(raw)})
            if h!=row['mask_sha256'] or len(raw)!=row['mask_bytes']:raise ValueError('Mask identity mismatch')
            with Image.open(BytesIO(raw)) as mask:
                original=int(np.count_nonzero(np.asarray(mask)));gt=t.from_numpy((np.asarray(mask.resize((256,256),Image.Resampling.NEAREST))!=0).astype(np.uint8).copy())
            if not original or bool(gt.any())!=row['mask_survives_nearest256']:raise ValueError('GT mask resize contract')
        if pixel.shape!=(256,256) or image.ndim or not t.isfinite(pixel).all() or not t.isfinite(image):raise ValueError('Observer output contract')
        p=dest/(row['image_id']+'.safetensors');save_file({'pixel':pixel,'image':image,'gt':gt},str(p));scores.append({'image_id':row['image_id'],'path':p.relative_to(output).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size,'original_positive':original,'resized_positive':int(gt.sum())})
        if index%32==0:print('B321_EVAL',output.name,stage,index,len(rows),flush=True)
        del f,a,b,pixel,image,gt
    write(dest/'score_manifest.json',{'cursor':update,'scores':scores})
    metrics={c:category_metrics([r for r in rows if r['category']==c],dest) for c in ORDER[:stage]}
    if before!=model_hash(model) or not t.equal(cpu,t.get_rng_state()) or not t.equal(cuda,t.cuda.get_rng_state(0)):raise ValueError('Observer mutated model/RNG')
    model.train();snapshot={'stage':stage,'cursor':update,'metrics':metrics,'model_sha256':before,'state_unchanged':True,'cohort_images':len(rows),'seconds':time.monotonic()-started,'memory':memory_report(model,opt,store,learner.ctrl,t,service)}
    chunks=archive_tensors(dest,output,output.parent,output.name+'_g'+str(update)+'_scores',deadline)
    snapshot['score_archives']=chunks['archives'];write(output/('archives_g'+str(update)+'.json'),chunks)
    print('B321_SNAPSHOT',output.name,stage,json.dumps(metrics),flush=True)
    return snapshot


def summarize(studies):
    from nested_replay_cad.data import ORDER
    result={'experiment':'B321','metric_version':'signed-prior-max-14-v1','arms':{},'scientific_acceptance':False}
    for arm,study in studies.items():
        if study['status']!='COMPLETE_REVIEW_PENDING' or study['counts']['optimizer']!=10005 or study['counts']['support_forwards']!=160080:raise ValueError('Incomplete count-matched arm')
        snaps=study['snapshots'];per={}
        for c in ORDER:
            cells=[s['metrics'][c] for s in snaps if c in s['metrics']];ap=[r['pixel']['AP'] for r in cells]
            per[c]={'pixel_AP':ap[-1],'image_AUROC':cells[-1]['image']['AUROC'],'Fmax':max(ap[:-1])-ap[-1] if len(ap)>1 else None,'Facq':ap[0]-ap[-1] if len(ap)>1 else None}
        mean=lambda x:sum(x)/len(x)
        result['arms'][arm]={'final_macro_pixel_AP':mean([v['pixel_AP'] for v in per.values()]),'final_macro_image_AUROC':mean([v['image_AUROC'] for v in per.values()]),'macro_Fmax':mean([v['Fmax'] for v in per.values() if v['Fmax'] is not None]),'macro_Facq':mean([v['Facq'] for v in per.values() if v['Facq'] is not None]),'per_category':per,'matrix':{'boundaries':[s['cursor'] for s in snaps],'tasks':list(ORDER),'values':{scope+'_'+metric:[[s['metrics'].get(c,{}).get(scope,{}).get(metric) for c in ORDER] for s in snaps] for scope in ('pixel','image') for metric in ('AP','AUROC')}}}
    er=result['arms']['er_uniform'];ours=result['arms']['a0_tail_ctx'];ours['pixel_AP_vs_ER']=ours['final_macro_pixel_AP']-er['final_macro_pixel_AP']
    passed=ours['pixel_AP_vs_ER']>=.005 and all(ours['per_category'][c]['pixel_AP']-er['per_category'][c]['pixel_AP']>=-.01 and ours['per_category'][c]['image_AUROC']-er['per_category'][c]['image_AUROC']>=-.005 for c in ORDER)
    result['screen']='MET_EXPLORATORY' if passed else 'SCREEN_NOT_MET';return result


class SessionPause(Exception):
    """Operational pause at a committed boundary, not a scientific failure."""


def run_arm(t,arm,cfg,rows,cohort,dataset,weights,output,deadline,resume_checkpoint=None):
    from nested_replay_cad.data import ORDER
    from nested_replay_cad.model import encoder,scoring
    from safetensors.torch import save_file
    output.mkdir(parents=True,exist_ok=False);access=output/'access.jsonl'
    model,opt=new_model(t,0);learner=Learner(t,model,opt,0,cfg['microbatch']);schedule=Schedule(0)
    initial=model_hash(model);selection_hash=hashlib.sha256();arm_started=time.monotonic();timing={'training_seconds':0.,'cache_encoder_seconds':0.,'observer_seconds':0.,'checkpoint_seconds':0.}
    scoring.gauss(t.device('cuda:0'));t.manual_seed(1000000)
    study={'experiment':'B321','arm':arm,'status':'RUNNING','initial_model_sha256':initial,'snapshots':[],'counts':learner.counts,'cache':[],'seed':0,'scientific_acceptance':False,'timing':timing,'data_selection_sha256':selection_hash.hexdigest()}
    guard=resource_guard(t,output.parent,deadline);store={};update=0
    import budget_checkpoint as recovery
    identity={'arm':arm,'recipe':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'weights':digest(weights),'train':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'cohort':hashlib.sha256(json.dumps(cohort,sort_keys=True).encode()).hexdigest(),'source':{p.relative_to(ROOT).as_posix():digest(p) for p in sorted([*ROOT.joinpath('src').rglob('*.py'),ROOT/'scripts/budget_train.py',ROOT/'scripts/budget_checkpoint.py'])}}
    pending=recovery.load(t,resume_checkpoint,identity) if resume_checkpoint is not None else None
    selection_rows=[]
    soft=float(os.environ.get('NRCAD_SESSION_SOFT_STOP_MONOTONIC','inf'))
    iteration_start=0;iteration=-1;stage=1
    def pause_guard():
        guard()
        if time.monotonic()>=soft:raise SessionPause('11h session work boundary')
    if pending is not None:
        # A new output root is mandatory; never roll back or alter the parent run.
        recovery.copy_evidence(Path(resume_checkpoint).parent.parent,output,pending['study'])
        study=pending['study'];study['status']='RUNNING';timing=study['timing'];study['resume_parent']={'checkpoint':str(resume_checkpoint),'update':pending['update']}
    try:
        for stage,category in enumerate(ORDER,1):
            if pending is not None and stage<pending['stage']:continue
            guard();cpu=t.get_rng_state().clone();cuda=t.cuda.get_rng_state(0).clone()
            append(access,{'operation':'read','kind':'weights_load','path':str(weights),'sha256':digest(weights)})
            service=encoder.load(weights,device='cuda:0');t.set_rng_state(cpu);t.cuda.set_rng_state(cuda,0)
            members=[r for r in rows if r['task_audit_only']==category];features={}
            key=hashlib.sha256(json.dumps({'inputs':[(r['opaque_id'],r['file_sha256']) for r in members],'weights':digest(weights),'preprocess':digest(ROOT/'src/nested_replay_cad/model/encoder.py')},sort_keys=True).encode()).hexdigest()
            extraction_started=time.monotonic()
            for r in members:guard();features[r['opaque_id']]=extract(t,service,dataset,r,access)
            timing['cache_encoder_seconds']+=time.monotonic()-extraction_started
            study['cache'].append({'stage':stage,'key':key,'fresh':True,'records':len(features),'bytes':sum(x.numel()*x.element_size() for x in features.values()),'storage':'bounded current-task CPU features, no persistent cache'})
            service=None;gc.collect();t.cuda.empty_cache()
            batches=Batches([r['opaque_id'] for r in members],16 if stage==1 else 8,1000+stage)
            iteration_start=0
            if pending is not None:
                store=recovery.restore(t,pending,model,opt,learner,schedule,batches,identity)
                update=pending['update'];iteration_start=pending['iteration'];selection_rows=pending['selection_rows'];selection_hash=recovery.selection_digest(selection_rows)
                study['counts']=learner.counts;pending=None
            for iteration in range(iteration_start,667):
                pause_guard();update+=1;ids=batches.next();replay,recent,history,eligible=schedule.reserve(ids,stage)
                append(access,{'operation':'read','kind':'feature_cache','cache_id':key,'current':ids,'replay':replay,'recent':recent,'history':history})
                selection_row={'u':update,'stage':stage,'current':ids,'replay':replay,'recent':recent,'history':history};selection_rows.append(selection_row)
                selection_hash.update((json.dumps(selection_row,sort_keys=True,separators=(',',':'))+'\n').encode('utf8'));study['data_selection_sha256']=selection_hash.hexdigest()
                t.cuda.synchronize();event_started=time.monotonic()
                event=learner.step([features[k] for k in ids],replay,recent,history,store,schedule.metas,schedule.serial,update,eligible,arm=='a0_tail_ctx',guard)
                t.cuda.synchronize();timing['training_seconds']+=time.monotonic()-event_started
                schedule.admit(ids,update);retained=schedule.retained()
                for k in retained:
                    if k not in store:
                        if k not in features:raise ValueError('Retained feature unavailable')
                        store[k]=features[k].clone() # Do not retain an entire old task cache through a view.
                store={k:v for k,v in store.items() if k in retained}
                append(output/'events.jsonl',{'u':update,'stage':stage,'current':ids,'replay':replay,'recent':recent,'history':history,'eligible':eligible,'admission_serial':schedule.serial,**event})
                if recovery.due(update):
                    if update!=10005:
                        checkpoint_started=time.monotonic()
                        state=recovery.capture(t,model,opt,learner,schedule,batches,store,study,selection_rows,stage,iteration+1,identity)
                        rec=recovery.save(t,output/'checkpoints',state,guard);del state
                        timing['checkpoint_seconds']+=time.monotonic()-checkpoint_started
                        print('B321_CHECKPOINT',arm,update,json.dumps(rec),flush=True)
                if update%16==0:
                    print('B321_TRAIN',arm,stage,iteration+1,update,'/10005',json.dumps(learner.counts),flush=True);write(output/'study.json',study)
            features.clear();gc.collect();t.cuda.empty_cache()
            cpu=t.get_rng_state().clone();cuda=t.cuda.get_rng_state(0).clone();append(access,{'operation':'read','kind':'weights_load','path':str(weights),'sha256':digest(weights)})
            service=encoder.load(weights,device='cuda:0');t.set_rng_state(cpu);t.cuda.set_rng_state(cuda,0)
            if len(study['snapshots'])<stage:
                observer_started=time.monotonic()
                # Evaluation is transactional: a soft stop restores RNG/model mode
                # and discards only this output's incomplete score directory.
                eval_cpu=t.get_rng_state().clone();eval_cuda=t.cuda.get_rng_state(0).clone()
                try:
                    study['snapshots'].append(observe(t,model,opt,learner,store,service,dataset,cohort,stage,update,output,deadline,pause_guard,access))
                except SessionPause:
                    t.set_rng_state(eval_cpu);t.cuda.set_rng_state(eval_cuda,0);model.train()
                    import shutil
                    partial=output/'scores'/f'g{update}'
                    if partial.exists():shutil.rmtree(partial)
                    raise
                timing['observer_seconds']+=time.monotonic()-observer_started
                service=None;gc.collect();t.cuda.empty_cache();write(output/'study.json',study)
        if learner.counts['optimizer']!=10005 or learner.counts['support_forwards']!=160080 or schedule.serial!=3629:raise ValueError('Final update/exposure/admission ledger mismatch')
        if len(study['snapshots'])!=15 or sum(len(s['metrics']) for s in study['snapshots'])!=120:raise ValueError('Expected15 snapshots and120 task-boundary cells')
        checkpoint_started=time.monotonic()
        state=recovery.capture(t,model,opt,learner,schedule,batches,store,study,selection_rows,15,667,identity)
        rec=recovery.save(t,output/'checkpoints',state,guard);del state
        study['resume_checkpoint']=rec;print('B321_CHECKPOINT',arm,10005,json.dumps(rec),flush=True)
        tensors={'model.'+n:v.detach().cpu().contiguous() for n,v in model.state_dict().items()}
        for n,p in model.named_parameters():
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):tensors['optimizer.'+n+'.'+key]=opt.state[p][key].detach().cpu().contiguous()
        tensors.update({'memory.'+k:v.contiguous() for k,v in store.items()})
        if learner.ctrl is not None:learner.ctrl.preserve(tensors)
        tensors['rng.torch']=t.get_rng_state();tensors['rng.cuda0']=t.cuda.get_rng_state(0)
        checkpoint=output/'final-state.safetensors';save_file(tensors,str(checkpoint));study['checkpoint']={'path':checkpoint.name,'sha256':digest(checkpoint),'bytes':checkpoint.stat().st_size,'scope':'audit checkpoint, not qualified resume','global_optimizer_step':10005,'reservoir':schedule.reservoir,'fifo':schedule.fifo,'admission_serial':schedule.serial}
        timing['checkpoint_seconds']+=time.monotonic()-checkpoint_started
        study['slow_steps']=learner.ctrl.slow_steps if learner.ctrl is not None else 0;study['eligible_batches']=learner.ctrl.eligible_commits if learner.ctrl is not None else 0
        study['controller_active']=learner.ctrl is not None
        study['peak_VRAM_allocated']=t.cuda.max_memory_allocated(0);study['peak_VRAM_reserved']=t.cuda.max_memory_reserved(0)
        study['status']='COMPLETE_REVIEW_PENDING'
    except SessionPause as pause:
        # iteration is the next uncommitted zero-based step when stopping before
        # step admission, or 666 after the completed task's evaluation pause.
        cursor_iteration=update-(stage-1)*667
        state=recovery.capture(t,model,opt,learner,schedule,batches,store,study,selection_rows,stage,cursor_iteration,identity)
        target=output/'checkpoints'/('g'+str(update).zfill(5))
        if target.exists():
            loaded=recovery.load(t,target,identity)
            if loaded['stage']!=stage or loaded['iteration']!=cursor_iteration or len(loaded['study']['snapshots'])!=len(study['snapshots']) or any(not t.equal(loaded['model'][k],v) for k,v in state['model'].items()) or not t.equal(loaded['rng']['torch'],state['rng']['torch']):raise ValueError('Existing pause checkpoint is not the committed state')
            rec=json.loads((target/'COMMIT.json').read_text(encoding='utf8'));rec['path']=str(target)
        else:rec=recovery.save(t,output/'checkpoints',state,guard)
        study.update(status='PAUSED_RESUMABLE',resume_checkpoint=rec,pause_reason=str(pause),pause_cursor={'stage':stage,'iteration':cursor_iteration,'update':update})
        print('B321_PAUSED_RESUMABLE',arm,update,json.dumps(rec),flush=True)
    except BaseException as error:
        study.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        import resource
        timing['whole_arm_seconds']=time.monotonic()-arm_started
        study['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write(output/'study.json',study)
    return study


def preflight(t,cfg,weights,cohort,output,deadline,dataset,normal_row):
    from nested_replay_cad.model import encoder,scoring
    from budget_checks import composition_checks,batch_model_check
    report={'status':'FAILED','scope':'synthetic decoder/composition/ranking plus fixed train/good encoder throughput, no AD quality'}
    guard=resource_guard(t,output,deadline);start=time.monotonic()
    report['composition']=composition_checks(t)
    model,opt=new_model(t,0)
    gen=t.Generator().manual_seed(32101);bank=t.randn((32,2,789,768),generator=gen)
    report['batch_parity']=batch_model_check(t,model,bank[:4],cfg)
    learner=Learner(t,model,opt,0,cfg['microbatch']);store={f'{i:064x}':bank[i] for i in range(8,32)}
    metas={k:{'admission_serial':i+1,'first_update':0} for i,k in enumerate(store)}
    current=[x for x in bank[:8]];replay=list(store)[4:12];recent=list(store)[:2];history=list(store)[2:4]
    def measure(fn):
        t.cuda.synchronize();a=time.monotonic();fn();t.cuda.synchronize();return time.monotonic()-a
    learner.step(current,replay,recent,history,store,metas,32,1,True,False,guard)
    er=max(measure(lambda u=u:learner.step(current,replay,recent,history,store,metas,32,u,True,False,guard)) for u in (2,3))
    ordinary=max(measure(lambda u=u:learner.step(current,replay,recent,history,store,metas,32,u,True,True,guard)) for u in (4,5))
    learner.ctrl.eligible_commits=15
    slow=measure(lambda:learner.step(current,replay,recent,history,store,metas,32,6,True,True,guard))
    model.eval();x=bank[:1].to('cuda:0')
    def infer():
        with t.no_grad():a,b=model(x[:,0],x[:,1]);scoring.dinomaly(a,b)
    from profile_timing import warmed_timing
    readout_timing=warmed_timing(infer,t.cuda.synchronize)
    readout=readout_timing['maximum_seconds']
    from safetensors.torch import save_file
    scratch=output/'preflight-scratch';scratch.mkdir(exist_ok=False)
    checkpoint=scratch/'synthetic-checkpoint.safetensors'
    payload={'model.'+n:v.detach().cpu().contiguous() for n,v in model.state_dict().items()}
    for n,p in model.named_parameters():
        for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):payload['optimizer.'+n+'.'+k]=opt.state[p][k].detach().cpu().contiguous()
    before=time.monotonic();save_file(payload,str(checkpoint));checkpoint_hash=digest(checkpoint)
    if digest(checkpoint)!=checkpoint_hash:raise ValueError('Synthetic checkpoint readback mismatch')
    checkpoint_seconds=time.monotonic()-before;checkpoint_bytes=checkpoint.stat().st_size
    import budget_checkpoint as recovery
    recovery_started=time.monotonic()
    probe={'schema':'synthetic_resource_probe_only','identity':{'synthetic':True},'update':500,'stage':1,'iteration':500,'payload':payload,'store':store}
    recovery_record=recovery.save(t,scratch/'recovery',probe,guard)
    recovery_checkpoint_seconds=time.monotonic()-recovery_started;recovery_checkpoint_bytes=recovery_record['bytes']
    import shutil
    shutil.rmtree(scratch/'recovery') # Own synthetic preflight scratch only.
    append(output/'access.jsonl',{'operation':'write_hash_verify','kind':'preflight_synthetic_checkpoint','sha256':checkpoint_hash,'bytes':checkpoint_bytes,'path':str(checkpoint)})
    # This is this attempt's synthetic scratch, not scientific evidence/checkpoint.
    checkpoint.unlink();scratch.rmdir();del payload
    del learner,opt,model,store,current,x;gc.collect();t.cuda.empty_cache()
    append(output/'access.jsonl',{'operation':'read','kind':'weights_load','path':str(weights),'sha256':digest(weights)})
    service=encoder.load(weights,device='cuda:0')
    def enc():extract(t,service,dataset,normal_row,output/'access.jsonl')
    encoder_timing=warmed_timing(enc,t.cuda.synchronize)
    encoder_seconds=encoder_timing['maximum_seconds'];del service;gc.collect();t.cuda.empty_cache()
    # Profile representative sorting/serialization without real AD labels.
    import numpy as np,zlib
    from nested_replay_cad.metrics import exact_numpy
    n=167*65536;rs=np.random.default_rng(32101);a=rs.random(n,dtype=np.float32);labels=(np.arange(n)%97==0).astype(np.uint8)
    before=time.monotonic();exact_numpy(a,labels);rank_maxclass=time.monotonic()-before
    block=a[:65536].tobytes()+labels[:65536].tobytes();before=time.monotonic();zlib.compress(block,1);serial_seconds=time.monotonic()-before
    del a,labels
    observations=sum(sum(r['category']==c for r in cohort)*(15-i) for i,c in enumerate(__import__('nested_replay_cad.data',fromlist=['ORDER']).ORDER))
    cache=2*3629*encoder_seconds
    eval_seconds=2*observations*(encoder_seconds+readout+serial_seconds)+240*rank_maxclass
    train_seconds=10005*er+10005*ordinary+583*max(slow-ordinary,0)
    # Charge a cold readout per boundary, and an encoder start per cache/eval
    # phase, instead of extrapolating a cold first call to every image.
    phase_startup=15*max(readout_timing['warmup_seconds'])+30*max(encoder_timing['warmup_seconds'])
    remaining=cache+eval_seconds+train_seconds+4*checkpoint_seconds+42*recovery_checkpoint_seconds+2*phase_startup
    estimated=remaining*1.2+1800
    left=deadline-time.monotonic()
    report.update(status='PASS' if estimated<=left else 'PREFLIGHT_INSUFFICIENT_CAP',seconds=time.monotonic()-start,
                  timing_samples={'readout':readout_timing,'encoder':encoder_timing},
                  measurements={'ER_batch_seconds':er,'primary_ordinary_seconds':ordinary,'primary_due_seconds':slow,'encoder_image_seconds':encoder_seconds,'readout_image_seconds':readout,'serialization_image_seconds':serial_seconds,'rank_maxclass_seconds':rank_maxclass,'checkpoint_seconds':checkpoint_seconds,'checkpoint_bytes':checkpoint_bytes,'recovery_checkpoint_seconds':recovery_checkpoint_seconds,'recovery_checkpoint_bytes':recovery_checkpoint_bytes,'profiled_commits':6,'phase_startup_seconds':phase_startup},
                  projection={'training':train_seconds,'cache':cache,'evaluation':eval_seconds,'remaining_with_margin_export':estimated,'remaining_whole_cap':left,'observations_per_arm':observations,'margin_fraction':.2,'export_reserve_seconds':1800},
                  unverified='Synthetic throughput plus conservative representative rank cost; actual data I/O/thermals still uncertain. No design/precision/evaluation fallback.')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/b321.json');args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf8'))
    expected={'experiment':'B321','seed':0,'arms':['er_uniform','a0_tail_ctx'],'primary':'a0_tail_ctx','iterations_per_task':667,'batch_size':16,'microbatch':4,'replay_size':8,'reservoir_size':24,'fifo_size':8,'slow_interval':16,'hard_mining_final':.9,'hard_mining_ramp_updates':26}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('B321 prospective protocol differs')
    thresholds={'gradientcheck_relative_L2':2e-4,'loss_atol':1e-5,'loss_rtol':1e-4,'budget_margin_fraction':.2,'export_reserve_seconds':1800,'base_recipe':'configs/b319.json'}
    if any(cfg.get(k)!=v for k,v in thresholds.items()):raise ValueError('Changed prospective qualification thresholds')
    caps={'whole_seconds':43200,'worker_seconds':41400,'group_RSS_bytes':8*1024**3,'VRAM_bytes':12*1024**3,'payload_bytes':18*1024**3}
    if cfg.get('caps')!=caps:raise ValueError('Changed B321 resource caps')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=min(started+41400,float(os.environ.get('NRCAD_RUN_DEADLINE_MONOTONIC',started+43200)))
    result={'experiment':'B321','status':'SETUP','seed':0,'scientific_acceptance':False,'VisA_access':False,'config_sha256':digest(args.config),'studies':{}}
    try:
        from nested_replay_cad.trainer import configure
        t=configure();t.cuda.set_device(0);t.empty(1,device='cuda');t.cuda.reset_peak_memory_stats(0)
        recipe=json.loads((ROOT/'configs/b319.json').read_text(encoding='utf8'));manifest=ROOT/recipe['full_manifest_path']
        if digest(manifest)!=recipe['full_manifest_sha256'] or digest(args.weights)!=recipe['encoder']['sha256']:raise ValueError('Manifest/weights identity mismatch')
        full=json.loads(manifest.read_text(encoding='utf8'))
        from nested_replay_cad.data import prepare,selection
        append(args.output/'access.jsonl',{'operation':'hash','kind':'metadata','path':str(manifest),'sha256':digest(manifest)})
        append(args.output/'access.jsonl',{'operation':'hash','kind':'weights','path':str(args.weights),'sha256':digest(args.weights)})
        chosen=selection(full);prepare(args.data,full,args.output)
        append(args.output/'access.jsonl',{'operation':'read','kind':'cohort_preparation','paths':'MVTec test+ground_truth byte inventory during data.prepare; no AD quality evaluation','manifest_sha256':digest(args.output/'full15_cohort.json')})
        cohort=json.loads((args.output/'full15_cohort.json').read_text(encoding='utf8'))['records']
        result.update(runtime={'python':sys.version,'torch':t.__version__,'CUDA':t.version.cuda,'GPU':t.cuda.get_device_name(0)},manifest_sha256=digest(manifest),weights_sha256=digest(args.weights),source_sha256={p.relative_to(ROOT).as_posix():digest(p) for scope in ('src','scripts','tests') for p in (ROOT/scope).rglob('*.py')})
        result['preflight']=preflight(t,cfg,args.weights,cohort,args.output,deadline,args.data,chosen['records'][0]);write(args.output/'preflight.json',result['preflight'])
        print('B321_PREFLIGHT',json.dumps(result['preflight']),flush=True)
        if result['preflight']['status']!='PASS':result['status']=result['preflight']['status'];return 2
        initial=None;selection_digest=None
        for arm in cfg['arms']:
            result['status']='TRAINING';write(args.output/'result.json',result)
            study=run_arm(t,arm,cfg,chosen['records'],cohort,args.data,args.weights,args.output/arm,deadline)
            if initial is not None and initial!=study['initial_model_sha256']:raise ValueError('Paired initial decoder mismatch')
            initial=study['initial_model_sha256']
            if selection_digest is not None and selection_digest!=study['data_selection_sha256']:raise ValueError('Paired data-side selection schedule mismatch')
            selection_digest=study['data_selection_sha256'];result['studies'][arm]=study
            gc.collect();t.cuda.empty_cache()
        metrics=summarize(result['studies']);write(args.output/'metrics.json',metrics);result['status']='COMPLETE_REVIEW_PENDING';return 0
    except BaseException as error:
        result.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:
        result['seconds']=time.monotonic()-started;write(args.output/'result.json',result);print('B321_TERMINAL',json.dumps({'status':result['status'],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':raise SystemExit(main())
