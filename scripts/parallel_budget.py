"""Run the unchanged B321 arms on separate visible GPUs under one resource cap.

The parent imports no Torch. Concurrent synthetic profiles are a resource gate,
not extra experimental arms. A rejected gate never starts normal training.
"""
import argparse,codecs,csv,ctypes,hashlib,io,json,math,os,re,signal,subprocess,sys,time,traceback,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests')]
ARMS=('er_uniform','a0_tail_ctx')
SLOTS=dict(zip(ARMS,('0','1')))
FEATURE_BYTES=4847616
GROUP_RSS_BYTES=16*1024**3
WORKER_RSS_BYTES=8*1024**3
HOST_RESERVE_BYTES=1024**3
DEADLINE_GUARD_SECONDS=120


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n');temp.replace(path)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024**2):h.update(block)
    return h.hexdigest()


def validate_config(cfg):
    fixed={'experiment':'B321','seed':0,'arms':list(ARMS),'primary':'a0_tail_ctx','iterations_per_task':667,'batch_size':16,'microbatch':4,'replay_size':8,'reservoir_size':24,'fifo_size':8,'slow_interval':16,'hard_mining_final':.9,'hard_mining_ramp_updates':26,'gradientcheck_relative_L2':2e-4,'loss_atol':1e-5,'loss_rtol':1e-4,'budget_margin_fraction':.2,'export_reserve_seconds':1800,'base_recipe':'configs/b319.json'}
    if any(cfg.get(k)!=v for k,v in fixed.items()):raise ValueError('Changed B321 scientific/configuration contract')
    expected={'whole_seconds':43200,'worker_seconds':41400,'group_RSS_bytes':8*1024**3,'VRAM_bytes':12*1024**3,'payload_bytes':18*1024**3}
    if cfg.get('caps')!=expected:raise ValueError('Missing or changed B321 resource cap')


def child_environment(base,arm,phase,deadline,shared,receipt,physical_uuid):
    if arm not in SLOTS or phase not in ('profile','train'):raise ValueError('Unknown arm/phase')
    env=dict(base);runtime=Path(shared).parent/'runtime'/phase/arm
    env.update(NRCAD_LOCAL_ROOT=str(runtime),TEMP=str(runtime/'tmp'),TMP=str(runtime/'tmp'),TMPDIR=str(runtime/'tmp'),TORCH_HOME=str(runtime/'cache/torch'),HF_HOME=str(runtime/'cache/huggingface'),XDG_CACHE_HOME=str(runtime/'cache'),CUDA_CACHE_PATH=str(runtime/'cache/cuda'),CUDA_VISIBLE_DEVICES=SLOTS[arm],B321_PHYSICAL_GPU_SLOT=SLOTS[arm],B321_EXPECTED_GPU_UUID=physical_uuid,
               B321_SHARED_INPUTS_DIR=str(shared),B321_WORKER_RECEIPT=str(receipt),NRCAD_RUN_DEADLINE_MONOTONIC=str(deadline),
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def validate_slot(env,arm):
    if arm not in SLOTS:raise ValueError('Unknown worker arm')
    slot=SLOTS[arm]
    if env.get('CUDA_VISIBLE_DEVICES')!=slot or env.get('B321_PHYSICAL_GPU_SLOT')!=slot:
        raise ValueError('GPU slot must be isolated before Torch import')
    if not env.get('B321_EXPECTED_GPU_UUID'):raise ValueError('Missing physical GPU identity')
    return slot


def canonical_uuid(value):
    text=str(value).lower()
    match=re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',text)
    if not match:raise ValueError('GPU UUID unavailable or malformed')
    return 'GPU-'+match.group(0)


def visible_uuid(t):
    properties=t.cuda.get_device_properties(0)
    value=getattr(properties,'uuid',None)
    if value is not None:
        try:return canonical_uuid(value)
        except ValueError:pass
    # Query the CUDA context selected by Torch, not physical ordinal zero.
    driver=ctypes.CDLL('libcuda.so.1');device=ctypes.c_int()
    if driver.cuCtxGetDevice(ctypes.byref(device))!=0:raise RuntimeError('Cannot identify active CUDA context')
    raw=(ctypes.c_ubyte*16)();get_uuid=getattr(driver,'cuDeviceGetUuid_v2',None) or driver.cuDeviceGetUuid
    if get_uuid(ctypes.byref(raw),device)!=0:raise RuntimeError('Cannot identify active CUDA UUID')
    return 'GPU-'+str(uuid.UUID(bytes=bytes(raw)))


def gpu_binding(t,env,arm):
    slot=validate_slot(env,arm)
    if not t.cuda.is_available() or t.cuda.device_count()!=1:raise RuntimeError('Worker requires exactly one visible CUDA device')
    t.cuda.set_device(0);t.empty(1,device='cuda:0');actual=visible_uuid(t)
    expected=canonical_uuid(env['B321_EXPECTED_GPU_UUID'])
    if actual.lower()!=expected.lower():raise RuntimeError('Visible CUDA device does not match assigned physical GPU')
    return {'physical_slot':int(slot),'physical_uuid':actual,'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],'visible_device_count':1,'logical_device':'cuda:0','name':t.cuda.get_device_name(0),'torch':t.__version__,'CUDA':t.version.cuda,'python':sys.version}


def physical_devices():
    command=['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits']
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf8',timeout=30,check=True)
    devices={}
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row)!=4:raise ValueError('Unexpected nvidia-smi inventory')
        index=int(row[0].strip());devices[str(index)]={'slot':index,'uuid':canonical_uuid(row[1]),'name':row[2].strip(),'memory_MiB':int(row[3].strip())}
    if not all(s in devices for s in ('0','1')) or devices['0']['uuid']==devices['1']['uuid']:raise RuntimeError('Two distinct physical GPUs are required')
    return devices


def descendants(root,records):
    found={root};changed=True
    while changed:
        changed=False
        for pid,row in records.items():
            if pid not in found and row['ppid'] in found:found.add(pid);changed=True
    return found


def effective_memory(proc=Path('/proc'),cgroup=Path('/sys/fs/cgroup')):
    """Read host availability and every exposed finite cgroup ancestor limit.

    Only numeric resource metadata is read; no dataset/credential payloads.
    The exposed mount root is also checked when a container hides ancestors.
    Unexposed ancestor limits remain an explicit limitation.
    """
    available=None
    for line in (proc/'meminfo').read_text(encoding='utf8').splitlines():
        if line.startswith('MemAvailable:'):
            fields=line.split()
            if len(fields)!=3 or fields[2]!='kB':raise ValueError('Invalid MemAvailable units')
            available=int(fields[1])*1024
    if available is None or available<=0:raise ValueError('Host MemAvailable unavailable')
    membership=(proc/'self/cgroup').read_text(encoding='utf8').splitlines()
    roots=[]
    for line in membership:
        _,controllers,relative=line.split(':',2)
        if controllers=='' or 'memory' in controllers.split(','):
            base=cgroup if controllers=='' else cgroup/'memory'
            parts=Path(relative.lstrip('/')).parts
            roots.append(base)
            if '..' not in parts:roots.extend(base.joinpath(*parts[:i]) for i in range(1,len(parts)+1))
    limits=[];exposed=False
    for base in sorted(set(roots)):
        for cap_name,use_name in (('memory.max','memory.current'),('memory.limit_in_bytes','memory.usage_in_bytes')):
            cap_path=base/cap_name
            if not cap_path.is_file():continue
            exposed=True;raw=cap_path.read_text(encoding='utf8').strip()
            if raw=='max':continue
            cap=int(raw)
            if cap<=0:raise ValueError('Invalid cgroup memory limit')
            if cap>=2**60:continue # cgroup v1 unlimited sentinel
            used=int((base/use_name).read_text(encoding='utf8').strip())
            if used<0:raise ValueError('Invalid cgroup memory usage')
            limits.append({'limit_path':str(cap_path),'limit_bytes':cap,'used_bytes':used,'available_bytes':max(0,cap-used)})
    if not exposed:raise RuntimeError('No readable cgroup memory controller; cannot qualify effective RAM')
    return {'host_MemAvailable_bytes':available,'cgroup_limits':limits,'effective_available_bytes':min([available]+[v['available_bytes'] for v in limits]),'scope':'Host MemAvailable plus exposed cgroup ancestors; hidden ancestors cannot be verified'}


def parent_anon_bytes(proc=Path('/proc')):
    # Called only after both profiling workers exited; shared/file RSS receives
    # no credit because its unique cgroup/host charge is not established.
    for line in (proc/'self/status').read_text(encoding='utf8').splitlines():
        if line.startswith('RssAnon:'):
            fields=line.split()
            if len(fields)!=3 or fields[2]!='kB':raise ValueError('Invalid parent RssAnon units')
            value=int(fields[1])*1024
            if value<0:raise ValueError('Invalid parent RssAnon')
            return value
    raise ValueError('Parent RssAnon unavailable')


class GroupMonitor:
    """Detect aggregate parent+all-descendant limits using Linux /proc, no psutil."""
    def __init__(self,output,deadline):
        if sys.platform!='linux':raise RuntimeError('Production group monitor requires Linux /proc')
        self.root=os.getpid();self.output=Path(output);self.deadline=deadline
        self.page=os.sysconf('SC_PAGE_SIZE');self.ticks=os.sysconf('SC_CLK_TCK');self.cpu={};self.peak_rss=0;self.last_disk=0.;self.latest={};self.worker_peaks={a:0 for a in ARMS}
    def check(self,managed=()):
        now=time.monotonic()
        if now>self.deadline-120:raise TimeoutError('Parallel whole/worker deadline reached')
        rows={}
        for p in Path('/proc').iterdir():
            if not p.name.isdecimal():continue
            try:
                values=(p/'stat').read_text(encoding='utf8',errors='replace').rsplit(') ',1)[1].split()
                rows[int(p.name)]={'ppid':int(values[1]),'pgid':int(values[2]),'cpu':(int(values[11])+int(values[12]))/self.ticks,'started':int(values[19]),'rss':max(0,int(values[21]))*self.page}
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if self.root not in rows:raise RuntimeError('Cannot inspect parent process resources')
        for process in managed:
            if process.poll() is None and process.pid not in rows:raise RuntimeError('Cannot inspect active worker resources')
        pids=descendants(self.root,rows)
        if os.getpgrp()==self.root:pids|={pid for pid,row in rows.items() if row['pgid']==self.root}
        rss=sum(rows[p]['rss'] for p in pids if p in rows)
        for pid in pids:
            if pid in rows:self.cpu[(pid,rows[pid]['started'])]=max(self.cpu.get((pid,rows[pid]['started']),0),rows[pid]['cpu'])
        cpu=sum(self.cpu.values());self.peak_rss=max(self.peak_rss,rss)
        self.latest={'group_RSS_bytes':rss,'peak_group_RSS_bytes':self.peak_rss,'aggregate_CPU_seconds':cpu,'processes':len(pids),'sample_monotonic':now}
        if rss>GROUP_RSS_BYTES:raise MemoryError('Aggregate process-group RSS16GiB cap exceeded')
        for arm,process in zip(ARMS,managed):
            worker_rss=sum(rows[p]['rss'] for p in descendants(process.pid,rows) if p in rows)
            self.worker_peaks[arm]=max(self.worker_peaks[arm],worker_rss)
            if worker_rss>WORKER_RSS_BYTES:raise MemoryError(arm+' worker process-tree RSS8GiB cap exceeded')
        self.latest['peak_worker_RSS_bytes']=dict(self.worker_peaks)
        if cpu>172800:raise TimeoutError('Aggregate CPU172800s cap exceeded')
        if now-self.last_disk>=5:
            import shutil
            from nested_replay_cad.storage import census
            used=census(self.output)
            free=shutil.disk_usage(self.output).free
            self.latest.update(output_bytes=used,free_bytes=free);self.last_disk=now
            if used>18*1024**3 or free<2*1024**3:raise OSError('Aggregate output18GiB/free2GiB reserve cap')
        return self.latest


def cancel(process):
    if process.poll() is not None:return
    # Children inherit the launcher's process group. Never detach them: an
    # outer killpg must reach both GPUs even if this parent is forcibly killed.
    descendants_to_stop=[]
    if sys.platform=='linux':
        relations={}
        for p in Path('/proc').iterdir():
            if not p.name.isdecimal():continue
            try:
                values=(p/'stat').read_text(encoding='utf8',errors='replace').rsplit(') ',1)[1].split()
                relations[int(p.name)]={'ppid':int(values[1])}
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        descendants_to_stop=list(descendants(process.pid,relations)-{process.pid})
        for pid in reversed(descendants_to_stop):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    try:process.terminate()
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        for pid in reversed(descendants_to_stop):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        try:process.kill()
        except ProcessLookupError:pass
        process.wait(timeout=5)
    for pid in reversed(descendants_to_stop):
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass


def require_owned_process_group():
    if os.getpgrp()!=os.getpid():raise RuntimeError('Launcher must give the parallel parent its own process group')
    return True


def cancel_owned_group_peers():
    # A crashed child may leave an orphan whose PPID changed. Its inherited
    # launch group still proves ownership; never kill the user's shell group.
    if sys.platform!='linux' or not hasattr(os,'getpgrp') or os.getpgrp()!=os.getpid():return
    own=os.getpid()
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal() or int(p.name)==own:continue
        try:
            values=(p/'stat').read_text(encoding='utf8',errors='replace').rsplit(') ',1)[1].split()
            if int(values[2])==own:os.kill(int(p.name),signal.SIGKILL)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue


def run_phase(commands,environments,logs,deadline,monitor):
    """Start both children before polling; fail-fast and drain file-backed logs."""
    processes={};readers={};decoders={};starts={};ends={};buffers={}
    logs=Path(logs);logs.mkdir(parents=True,exist_ok=False)
    def tail(arm,final=False):
        while True:
            block=readers[arm].read(128*1024)
            if not block:break
            text=decoders[arm].decode(block);buffers[arm]+=text
            lines=buffers[arm].splitlines(keepends=True);buffers[arm]=''
            for line in lines:
                if line.endswith(('\n','\r')):print('['+arm+'] '+line,end='',flush=True)
                else:buffers[arm]=line
            if not final:break
        if final:
            buffers[arm]+=decoders[arm].decode(b'',final=True)
            if buffers[arm]:print('['+arm+'] '+buffers[arm],flush=True);buffers[arm]=''
    try:
        if set(commands)!=set(ARMS) or set(environments)!=set(ARMS):raise ValueError('Exactly the paired arms are required')
        for arm in ARMS:
            path=logs/(arm+'.log')
            with path.open('xb') as stream:
                kwargs={'cwd':str(ROOT),'env':environments[arm],'stdout':stream,'stderr':subprocess.STDOUT}
                processes[arm]=subprocess.Popen(commands[arm],**kwargs)
            starts[arm]=time.monotonic();readers[arm]=path.open('rb');decoders[arm]=codecs.getincrementaldecoder('utf8')(errors='replace');buffers[arm]=''
        while True:
            monitor.check(list(processes.values()))
            finished=True
            for arm,proc in processes.items():
                tail(arm);code=proc.poll()
                if code is None:finished=False
                elif arm not in ends:ends[arm]=time.monotonic()
                if code not in (None,0):raise RuntimeError(arm+' subprocess failed, exit='+str(code))
            if finished:break
            if time.monotonic()>deadline-120:raise TimeoutError('Phase deadline')
            time.sleep(.5)
        for arm in ARMS:tail(arm,True)
        return {'started':starts,'ended':ends,'overlap_seconds':max(0.,min(ends.values())-max(starts.values())),'exit_codes':{a:p.returncode for a,p in processes.items()}}
    finally:
        for proc in processes.values():cancel(proc)
        cancel_owned_group_peers()
        for arm in readers:
            try:tail(arm,True)
            finally:readers[arm].close()


def finite_positive(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Missing/nonpositive measurement: '+name)
    return float(value)


def parallel_forecast(profiles,remaining_worker,max_category_records,profile_group_rss,*,remaining_whole,host_memory,worker_profile_rss,current_parent_anon_bytes=0):
    if set(profiles)!=set(ARMS) or set(worker_profile_rss)!=set(ARMS):raise ValueError('Missing arm profile/RSS')
    finite_positive(remaining_worker,'remaining worker seconds');finite_positive(remaining_whole,'remaining whole seconds')
    estimates={}
    for arm,report in profiles.items():
        m=report['measurements'];get=lambda n:finite_positive(m[n],n)
        er=get('ER_batch_seconds');ordinary=get('primary_ordinary_seconds');slow=get('primary_due_seconds')
        encoder=get('encoder_image_seconds');readout=get('readout_image_seconds');serialization=get('serialization_image_seconds');ranking=get('rank_maxclass_seconds');checkpoint=get('checkpoint_seconds');recovery_checkpoint=get('recovery_checkpoint_seconds')
        if report.get('composition',{}).get('status')!='PASS_SCOPED' or report.get('batch_parity',{}).get('status')!='PASS_SCOPED':raise ValueError('Profile numerical composition was not qualified')
        observations=report['projection']['observations_per_arm']
        if observations!=13950:raise ValueError('Unexpected full-test observation schedule')
        training=10005*er if arm=='er_uniform' else 10005*ordinary+583*max(0,slow-ordinary)
        cache=3629*encoder;evaluation=observations*(encoder+readout+serialization)+120*ranking
        startup=get('phase_startup_seconds') if 'phase_startup_seconds' in m else 0. # Legacy receipts remain replayable without inventing samples.
        estimates[arm]={'training':training,'cache':cache,'evaluation':evaluation,'checkpoint_export':2*checkpoint+21*recovery_checkpoint,'phase_startup':startup,'total':training+cache+evaluation+2*checkpoint+21*recovery_checkpoint+startup}
    critical=max(r['total'] for r in estimates.values());worker_required=critical*1.2
    projected=worker_required+1800
    headroom=2*max_category_records*FEATURE_BYTES;memory=finite_positive(profile_group_rss,'profile group RSS')+headroom
    workers={a:finite_positive(worker_profile_rss[a],a+' RSS')+headroom/2 for a in ARMS}
    host_available=finite_positive(host_memory.get('effective_available_bytes'),'effective host RAM')
    # Availability already excludes this parent's resident anonymous pages.
    # Credit only those measured pages, never file cache or profile worker RSS.
    if isinstance(current_parent_anon_bytes,bool) or not isinstance(current_parent_anon_bytes,int) or not 0<=current_parent_anon_bytes<=profile_group_rss:raise ValueError('Invalid current parent anonymous RSS')
    gross_host_required=memory+HOST_RESERVE_BYTES
    host_required=gross_host_required-current_parent_anon_bytes
    blockers=[]
    if memory>GROUP_RSS_BYTES or max(workers.values())>WORKER_RSS_BYTES:blockers.append('PREFLIGHT_INSUFFICIENT_MEMORY')
    if host_required>host_available:blockers.append('PREFLIGHT_INSUFFICIENT_HOST_MEMORY')
    if worker_required>remaining_worker or projected>remaining_whole:blockers.append('PREFLIGHT_INSUFFICIENT_CAP')
    # One committed step + cold phase starts + full checkpoint + export must fit.
    # The complete experiment is allowed to span multiple persisted sessions.
    segment=max(max(r['measurements']['ER_batch_seconds'],r['measurements']['primary_ordinary_seconds'],r['measurements']['primary_due_seconds'])*1.2 + r['measurements']['recovery_checkpoint_seconds']*1.2 + r['measurements'].get('phase_startup_seconds',0) for r in profiles.values())
    total_blockers=list(blockers)
    blockers=[b for b in blockers if b!='PREFLIGHT_INSUFFICIENT_CAP']
    if segment>remaining_worker or segment+1800>remaining_whole:blockers.append('PREFLIGHT_INSUFFICIENT_CAP')
    status=blockers[0] if blockers else 'PASS'
    return {'status':status,'blocking_reasons':blockers,'full_experiment_diagnostic_blockers':total_blockers,'session_admission':{'required_work_seconds':segment,'work_remaining_seconds':remaining_worker,'required_with_export_seconds':segment+1800,'whole_remaining_seconds':remaining_whole},'per_arm':estimates,'projection':{'remaining_worker_with_margin':worker_required,'remaining_worker_cap':remaining_worker,'remaining_with_margin_export':projected,'remaining_whole_cap':remaining_whole,'parallel_critical_path_seconds':critical,'sequential_sum_diagnostic':sum(r['total'] for r in estimates.values()),'margin_fraction':.2,'export_reserve_seconds':1800,'observations_per_arm':13950},'memory':{'observed_group_profile_peak_RSS_bytes':profile_group_rss,'two_current_task_cache_headroom_bytes':headroom,'projected_group_RSS_bytes':memory,'limit_bytes':GROUP_RSS_BYTES,'projected_worker_RSS_bytes':workers,'worker_limit_bytes':WORKER_RSS_BYTES,'host':host_memory,'host_required_bytes':host_required,'gross_host_required_bytes':gross_host_required,'current_parent_anon_credit_bytes':current_parent_anon_bytes,'host_reserve_bytes':HOST_RESERVE_BYTES},'scope':'Concurrent profiles, separate worker and whole deadlines with one export reserve; forecasts do not guarantee runtime or scientific gain.'}


def training_admitted(forecast):
    if forecast.get('status')!='PASS' or forecast.get('blocking_reasons'):return False
    p=forecast.get('projection',{});m=forecast.get('memory',{})
    if m.get('limit_bytes')!=GROUP_RSS_BYTES or m.get('worker_limit_bytes')!=WORKER_RSS_BYTES:return False
    workers=m.get('projected_worker_RSS_bytes',{})
    if set(workers)!=set(ARMS):return False
    seg=forecast.get('session_admission',{})
    return (finite_positive(seg.get('required_work_seconds'),'segment required')<=finite_positive(seg.get('work_remaining_seconds'),'segment left')
        and finite_positive(seg.get('required_with_export_seconds'),'segment+export')<=finite_positive(seg.get('whole_remaining_seconds'),'whole left')
        and finite_positive(m.get('projected_group_RSS_bytes'),'group RSS')<=GROUP_RSS_BYTES
        and all(finite_positive(v,'worker RSS')<=WORKER_RSS_BYTES for v in workers.values())
        and finite_positive(m.get('host_required_bytes'),'host required')<=finite_positive(m.get('host',{}).get('effective_available_bytes'),'host available'))


def paired_studies(studies):
    if set(studies)!=set(ARMS):raise ValueError('Missing completed paired arm')
    for arm,study in studies.items():
        if study.get('status')!='COMPLETE_REVIEW_PENDING':raise ValueError('Incomplete arm '+arm)
        c=study['counts']
        if c.get('optimizer')!=10005 or c.get('support_forwards')!=160080:raise ValueError('Update/exposure ledger mismatch')
        snapshots=study['snapshots']
        if len(snapshots)!=15 or sum(len(s['metrics']) for s in snapshots)!=120:raise ValueError('Incomplete15-stage/120-cell evaluation')
    a,b=[studies[k] for k in ARMS]
    for key in ('initial_model_sha256','data_selection_sha256'):
        if not a.get(key) or a[key]!=b.get(key):raise ValueError('Paired identity mismatch: '+key)
    return True


def verify_inputs(args,shared):
    identity=read(shared/'identity.json');files=identity['files']
    for name,expected in files.items():
        path=Path(name)
        if digest(path)!=expected:raise ValueError('Changed shared input '+name)
    if identity['dataset']!=str(args.data.resolve()) or identity['weights']!=str(args.weights.resolve()):raise ValueError('Worker input path differs from parent')
    if digest(args.config)!=identity['config_sha256']:raise ValueError('Worker config identity differs')
    return identity,read(shared/'full15_selection.json'),read(shared/'full15_cohort.json')['records']


def worker(args):
    slot=validate_slot(os.environ,args.arm) # This must precede any Torch import.
    receipt=Path(os.environ['B321_WORKER_RECEIPT']);started=time.monotonic()
    result={'arm':args.arm,'phase':args.worker_phase,'status':'FAILED','started_monotonic':started,'physical_slot':int(slot),'scientific_acceptance':False}
    try:
        cfg=read(args.config);validate_config(cfg);shared=Path(os.environ['B321_SHARED_INPUTS_DIR'])
        identity,selection,cohort=verify_inputs(args,shared)
        runtime=Path(os.environ['NRCAD_LOCAL_ROOT']);expected_runtime=shared.parent/'runtime'/args.worker_phase/args.arm
        if runtime.resolve()!=expected_runtime.resolve():raise ValueError('Worker runtime path escaped its phase/arm scope')
        runtime.mkdir(parents=True,exist_ok=False);(runtime/'tmp').mkdir();(runtime/'cache').mkdir()
        result['input_identity']=identity;result['runtime_root']=str(runtime)
        import budget_train as budget
        from nested_replay_cad.trainer import configure
        t=configure();result['binding']=gpu_binding(t,os.environ,args.arm);t.cuda.reset_peak_memory_stats(0)
        write(receipt,result);print('B321_GPU_BINDING',json.dumps(result['binding']),flush=True)
        deadline=float(os.environ['NRCAD_RUN_DEADLINE_MONOTONIC'])
        if args.worker_phase=='profile':
            args.output.mkdir(parents=True,exist_ok=False)
            result['profile_started_monotonic']=time.monotonic()
            report=budget.preflight(t,cfg,args.weights,cohort,args.output,deadline,args.data,selection['records'][0])
            result['profile_ended_monotonic']=time.monotonic();result['profile']=report
            if report.get('status') not in ('PASS','PREFLIGHT_INSUFFICIENT_CAP'):raise RuntimeError('Unexpected raw profile verdict')
            result['status']='PROFILE_COMPLETE_REVIEW_PENDING';write(args.output/'preflight.json',report)
        else:
            study=budget.run_arm(t,args.arm,cfg,selection['records'],cohort,args.data,args.weights,args.output,deadline,resume_checkpoint=args.resume_checkpoint)
            result['status']=study['status'];result['study_path']=str(args.output/'study.json')
        return 0
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:
        result['ended_monotonic']=time.monotonic();result['seconds']=time.monotonic()-started;write(receipt,result)


def parent(args):
    cfg=read(args.config);validate_config(cfg)
    if sys.platform!='linux':raise RuntimeError('Parallel B321 requires Linux')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();monitor=None
    result={'experiment':'B321','execution':'two-GPU parallel arms','status':'SETUP','studies':{},'scientific_acceptance':False,'VisA_access':False,'attempts':1,'config_sha256':digest(args.config)}
    try:
        whole=float(os.environ['NRCAD_RUN_DEADLINE_MONOTONIC'])
        if not math.isfinite(whole) or whole<=started:raise ValueError('Missing/expired declared whole-job deadline')
        soft=float(os.environ.get('NRCAD_SESSION_SOFT_STOP_MONOTONIC',str(whole-3600)))
        if not math.isfinite(soft) or not started<soft<=whole-3600:raise ValueError('Invalid11hsoftstop/1hexport reserve')
        deadline=whole;require_owned_process_group();monitor=GroupMonitor(args.output,deadline)
        result['execution_caps']={'group_RSS_bytes':GROUP_RSS_BYTES,'worker_RSS_bytes':WORKER_RSS_BYTES,'whole_seconds':43200,'worker_seconds':43200,'soft_stop_monotonic':soft,'export_reserved_seconds':3600,'legacy_recipe_group_RSS_bytes':cfg['caps']['group_RSS_bytes']}
        result['host_memory_before_profiles']=effective_memory();print('B321_EFFECTIVE_MEMORY',json.dumps(result['host_memory_before_profiles']),flush=True)
        devices=physical_devices();result['physical_devices']=devices
        recipe=read(ROOT/'configs/b319.json');manifest=ROOT/recipe['full_manifest_path']
        if digest(manifest)!=recipe['full_manifest_sha256'] or digest(args.weights)!=recipe['encoder']['sha256']:raise ValueError('Frozen manifest/weights mismatch')
        full=read(manifest);shared=args.output/'inputs';shared.mkdir(exist_ok=False)
        from nested_replay_cad.data import prepare,selection
        prepare(args.data,full,shared)
        files=[args.config.resolve(),args.weights.resolve(),manifest.resolve(),(ROOT/'scripts/budget_train.py').resolve(),(ROOT/'scripts/parallel_budget.py').resolve(),(ROOT/'tests/budget_checks.py').resolve(),(shared/'full15_selection.json').resolve(),(shared/'full15_cohort.json').resolve()]
        identity={'dataset':str(args.data.resolve()),'weights':str(args.weights.resolve()),'config_sha256':digest(args.config),'files':{str(p):digest(p) for p in files},'scope':'Exact shared metadata/weights/source identity; cohort preparation decoded MVTec dev masks but no AD quality evaluation'}
        write(shared/'identity.json',identity);result['input_identity']=identity
        with (args.output/'access.jsonl').open('x',encoding='utf8') as log:log.write(json.dumps({'operation':'hash_and_prepare','identity':identity})+'\n')
        children=[]
        for phase in ('profile','train'):
            if phase=='train' and not training_admitted(result.get('preflight',{})):raise RuntimeError('Training cannot start without qualified parallel admission')
            commands={};environments={};receipts={}
            for arm in ARMS:
                out=args.output/'qualification'/arm if phase=='profile' else args.output/arm
                receipt=args.output/'workers'/phase/(arm+'.json');receipts[arm]=receipt
                environments[arm]=child_environment(os.environ,arm,phase,deadline,shared,receipt,devices[SLOTS[arm]]['uuid'])
                commands[arm]=[sys.executable,'-B',str(Path(__file__).resolve()),'--data',str(args.data.resolve()),'--weights',str(args.weights.resolve()),'--config',str(args.config.resolve()),'--output',str(out.resolve()),'--worker-phase',phase,'--arm',arm]
            if phase=='train':
                for arm in ARMS:
                    resume=getattr(args,'resume_er' if arm=='er_uniform' else 'resume_ctx',None)
                    if resume is not None:commands[arm].extend(['--resume-checkpoint',str(resume.resolve())])
            result['status']='PROFILING' if phase=='profile' else 'TRAINING';write(args.output/'result.json',result)
            result[phase+'_processes']=run_phase(commands,environments,args.output/'logs'/phase,deadline,monitor)
            records={arm:read(p) for arm,p in receipts.items()}
            bindings=[r['binding'] for r in records.values()]
            if bindings[0]['physical_uuid']==bindings[1]['physical_uuid']:raise ValueError('Paired workers used the same physical GPU')
            if phase=='profile':
                intervals=[(r['profile_started_monotonic'],r['profile_ended_monotonic']) for r in records.values()]
                overlap=min(v[1] for v in intervals)-max(v[0] for v in intervals)
                if overlap<=0:raise ValueError('Profiles did not overlap')
                counts={}
                for r in full['records']:counts[r['task_audit_only']]=counts.get(r['task_audit_only'],0)+1
                now=time.monotonic()
                forecast=parallel_forecast({a:r['profile'] for a,r in records.items()},soft-now,max(counts.values()),monitor.peak_rss,remaining_whole=whole-now-DEADLINE_GUARD_SECONDS,host_memory=effective_memory(),worker_profile_rss=monitor.worker_peaks,current_parent_anon_bytes=parent_anon_bytes())
                forecast['profile_overlap_seconds']=overlap;forecast['raw_sequential_forecasts']={a:r['profile']['projection'] for a,r in records.items()}
                result['preflight']=forecast;write(args.output/'preflight.json',forecast)
                print('B321_PARALLEL_PREFLIGHT',json.dumps(forecast),flush=True)
                if not training_admitted(forecast):result['status']=forecast['status'];return 2
            else:
                result['studies']={a:read(args.output/a/'study.json') for a in ARMS}
                if any(v.get('status')=='PAUSED_RESUMABLE' for v in result['studies'].values()):
                    if not all(v.get('status') in ('PAUSED_RESUMABLE','COMPLETE_REVIEW_PENDING') for v in result['studies'].values()):raise ValueError('Invalid mixed session outcomes')
                    result['status']='PAUSED_RESUMABLE';return 0
                paired_studies(result['studies'])
        import budget_train as budget
        write(args.output/'metrics.json',budget.summarize(result['studies']));result['status']='COMPLETE_REVIEW_PENDING';return 0
    except BaseException as error:
        result.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:
        result['seconds']=time.monotonic()-started;result['resources']=monitor.latest if monitor is not None else {};write(args.output/'result.json',result)
        print('B321_PARALLEL_TERMINAL',json.dumps({'status':result['status'],'seconds':result['seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/b321.json');p.add_argument('--worker-phase',choices=('profile','train'));p.add_argument('--arm',choices=ARMS);p.add_argument('--resume-checkpoint',type=Path);p.add_argument('--resume-er',type=Path);p.add_argument('--resume-ctx',type=Path);args=p.parse_args()
    if bool(args.worker_phase)!=bool(args.arm):p.error('--worker-phase and --arm must be used together')
    if args.resume_checkpoint is not None and args.worker_phase!='train':p.error('--resume-checkpoint requires a train worker; parent uses --resume-er/--resume-ctx')
    if args.worker_phase is not None and (args.resume_er is not None or args.resume_ctx is not None):p.error('Per-arm resume paths belong to parent only')
    if not args.worker_phase:
        def terminate_parent(signum,frame):raise TimeoutError('Parallel parent received SIGTERM; cancel workers and preserve outputs')
        signal.signal(signal.SIGTERM,terminate_parent)
    return worker(args) if args.worker_phase else parent(args)


if __name__=='__main__':raise SystemExit(main())
