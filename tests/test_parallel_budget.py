"""Standalone stdlib tests for parallel B321 orchestration, no Torch/GPU/data."""
import ast,copy,importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('parallel_budget',ROOT/'scripts/parallel_budget.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SCRATCH=Path(os.environ.get('NRCAD_LOCAL_ROOT',str(ROOT/'.local/cache/tests/b321-parallel-worker')));SCRATCH.mkdir(parents=True,exist_ok=True)
def forecast(profiles,remaining,records,rss,**kwargs):
    defaults={'remaining_whole':remaining,'host_memory':{'effective_available_bytes':64*1024**3},'worker_profile_rss':{a:rss/2 for a in m.ARMS}}
    defaults.update(kwargs)
    return m.parallel_forecast(profiles,remaining,records,rss,**defaults)

class Monitor:
    def check(self,managed=()):return {}
def profile(scale=1):
    return {'composition':{'status':'PASS_SCOPED'},'batch_parity':{'status':'PASS_SCOPED'},'measurements':{k:v*scale for k,v in {'ER_batch_seconds':.1,'primary_ordinary_seconds':.2,'primary_due_seconds':1.,'encoder_image_seconds':.001,'readout_image_seconds':.001,'serialization_image_seconds':.0001,'rank_maxclass_seconds':.01,'checkpoint_seconds':1.,'recovery_checkpoint_seconds':1.}.items()},'projection':{'observations_per_arm':13950}}
def study():
    return {'status':'COMPLETE_REVIEW_PENDING','counts':{'optimizer':10005,'support_forwards':160080},'snapshots':[{'metrics':{str(j):{} for j in range(i+1)}} for i in range(15)],'initial_model_sha256':'abc','data_selection_sha256':'def'}
class ParallelBudgetTests(unittest.TestCase):
    def test_parent_module_has_no_torch_import(self):
        tree=ast.parse((ROOT/'scripts/parallel_budget.py').read_text(encoding='utf8'))
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
                self.assertTrue(all(not x.startswith(('torch','budget_train','nested_replay_cad.trainer')) for x in names))
    def test_gpu_slots_isolated_before_import(self):
        for arm,slot in m.SLOTS.items():
            env=m.child_environment({'CUDA_VISIBLE_DEVICES':'9'},arm,'profile',123,'shared','receipt','GPU-00000000-0000-0000-0000-000000000001')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],slot);self.assertEqual(m.validate_slot(env,arm),slot)
            self.assertEqual(Path(env['NRCAD_LOCAL_ROOT']),Path('runtime')/'profile'/arm)
            self.assertEqual(Path(env['TMPDIR']),Path(env['NRCAD_LOCAL_ROOT'])/'tmp')
            bad=dict(env,CUDA_VISIBLE_DEVICES='0,1')
            with self.assertRaises(ValueError):m.validate_slot(bad,arm)
    def test_missing_gpu_is_reported(self):
        fake=type('T',(),{'cuda':type('C',(),{'is_available':staticmethod(lambda:False)})()})()
        env=m.child_environment({},'er_uniform','profile',123,'shared','receipt','GPU-00000000-0000-0000-0000-000000000001')
        with self.assertRaisesRegex(RuntimeError,'exactly one'):m.gpu_binding(fake,env,'er_uniform')
    def test_forecast_parallel_critical_path_not_sum(self):
        r=forecast({a:profile(1+i) for i,a in enumerate(m.ARMS)},40000,391,1024**3)
        self.assertEqual(r['status'],'PASS');p=r['projection']
        self.assertEqual(p['parallel_critical_path_seconds'],max(x['total'] for x in r['per_arm'].values()))
        self.assertLess(p['parallel_critical_path_seconds'],p['sequential_sum_diagnostic']);self.assertTrue(m.training_admitted(r))
    def test_time_and_memory_refusals_cannot_launch_training(self):
        profiles={a:profile() for a in m.ARMS}
        time_block=forecast(profiles,1,391,1024**3)
        self.assertEqual(time_block['status'],'PREFLIGHT_INSUFFICIENT_CAP');self.assertFalse(m.training_admitted(time_block))
        self.assertGreater(time_block['projection']['remaining_with_margin_export'],time_block['projection']['remaining_whole_cap'])
        memory_block=forecast(profiles,40000,391,15*1024**3)
        self.assertEqual(memory_block['status'],'PREFLIGHT_INSUFFICIENT_MEMORY');self.assertFalse(m.training_admitted(memory_block))
        self.assertGreater(memory_block['memory']['projected_group_RSS_bytes'],memory_block['memory']['limit_bytes'])
        parent=next(n for n in ast.parse((ROOT/'scripts/parallel_budget.py').read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='parent')
        self.assertIn('not training_admitted',ast.unparse(parent))
    def test_invalid_profile_measurement_fails_closed(self):
        profiles={a:profile() for a in m.ARMS};profiles['er_uniform']['measurements']['ER_batch_seconds']=0
        with self.assertRaises(ValueError):forecast(profiles,40000,391,1024**3)
    def test_paired_identity_and_ledger_negative_cases(self):
        good={a:study() for a in m.ARMS};self.assertTrue(m.paired_studies(good))
        for key,value in [('initial_model_sha256','wrong'),('data_selection_sha256','wrong')]:
            bad=copy.deepcopy(good);bad['a0_tail_ctx'][key]=value
            with self.assertRaises(ValueError):m.paired_studies(bad)
        bad=copy.deepcopy(good);bad['a0_tail_ctx']['counts']['optimizer']=10004
        with self.assertRaises(ValueError):m.paired_studies(bad)
    def test_all_descendants_count_once(self):
        rows={1:{'ppid':0},2:{'ppid':1},3:{'ppid':2},4:{'ppid':1},5:{'ppid':99}}
        self.assertEqual(m.descendants(1,rows),{1,2,3,4})
    def test_actual_parent_refuses_before_train_on_synthetic_profiles(self):
        from types import SimpleNamespace
        import nested_replay_cad.data as data
        actual_digest=m.digest;actual_read=m.read
        recipe=actual_read(ROOT/'configs/b319.json');manifest=ROOT/recipe['full_manifest_path']
        for refuse in ('time','memory'):
            with self.subTest(refuse=refuse), tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
                folder=Path(temp);weights=folder/'weights.fixture';weights.write_text('synthetic',encoding='utf8')
                args=SimpleNamespace(config=ROOT/'configs/b321.json',data=folder/'dataset',weights=weights,output=folder/'run')
                phases=[]
                class SyntheticMonitor:
                    def __init__(self,*_):
                        self.peak_rss=(15 if refuse=='memory' else 1)*1024**3;self.latest={};self.worker_peaks={a:self.peak_rss/2 for a in m.ARMS}
                    def check(self,*_):return {}
                def fake_digest(path):
                    if Path(path)==weights:return recipe['encoder']['sha256']
                    if Path(path)==manifest:return recipe['full_manifest_sha256']
                    return actual_digest(path)
                def fake_read(path):
                    if Path(path)==manifest:return {'records':[{'task_audit_only':'bottle'} for _ in range(391)]}
                    return actual_read(path)
                def prepare(dataset,full,out):
                    m.write(out/'full15_selection.json',{'records':[]});m.write(out/'full15_cohort.json',{'records':[]})
                def phase(commands,environments,logs,deadline,monitor):
                    self.assertTrue(all(cmd[cmd.index('--worker-phase')+1]=='profile' for cmd in commands.values()))
                    phases.append('profile')
                    for arm,env in environments.items():
                        m.write(env['B321_WORKER_RECEIPT'],{'binding':{'physical_uuid':env['B321_EXPECTED_GPU_UUID']},'profile_started_monotonic':1.,'profile_ended_monotonic':2.,'profile':profile(100000 if refuse=='time' else 1)})
                    return {'overlap_seconds':1.}
                devices={str(i):{'slot':i,'uuid':'GPU-00000000-0000-0000-0000-'+str(i+1).zfill(12)} for i in range(2)}
                with mock.patch.object(m.sys,'platform','linux'),mock.patch.object(m,'require_owned_process_group',return_value=True),mock.patch.object(m,'GroupMonitor',SyntheticMonitor),mock.patch.object(m,'effective_memory',return_value={'effective_available_bytes':64*1024**3}),mock.patch.object(m,'parent_anon_bytes',return_value=0),mock.patch.object(m,'physical_devices',return_value=devices),mock.patch.object(m,'digest',side_effect=fake_digest),mock.patch.object(m,'read',side_effect=fake_read),mock.patch.object(data,'prepare',side_effect=prepare),mock.patch.object(m,'run_phase',side_effect=phase),mock.patch.dict(os.environ,{'NRCAD_RUN_DEADLINE_MONOTONIC':str(time.monotonic()+43200)}):
                    self.assertEqual(m.parent(args),2)
                result=actual_read(args.output/'result.json')
                expected='PREFLIGHT_INSUFFICIENT_CAP' if refuse=='time' else 'PREFLIGHT_INSUFFICIENT_MEMORY'
                self.assertEqual(result['status'],expected);self.assertEqual(result['preflight']['status'],expected)
                self.assertEqual(result['studies'],{});self.assertEqual(phases,['profile'])
    def test_worker_and_whole_deadlines_are_separate(self):
        profiles={a:profile() for a in m.ARMS}
        reference=forecast(profiles,40000,391,1024**3)
        need=reference['session_admission']['required_work_seconds']
        # Remaining worker may be smaller than work+export without counting export twice.
        good=forecast(profiles,need+121,391,1024**3,remaining_whole=need+1800+121)
        self.assertTrue(m.training_admitted(good))
        for worker,whole in ((need-1,need+1800+121),(need+121,need+1799)):
            denied=forecast(profiles,worker,391,1024**3,remaining_whole=whole)
            self.assertEqual(denied['status'],'PREFLIGHT_INSUFFICIENT_CAP');self.assertFalse(m.training_admitted(denied))
    def test_host_and_worker_memory_refusals(self):
        profiles={a:profile() for a in m.ARMS}
        for kwargs,status in [({'host_memory':{'effective_available_bytes':1024**3}},'PREFLIGHT_INSUFFICIENT_HOST_MEMORY'),({'worker_profile_rss':{'er_uniform':7*1024**3,'a0_tail_ctx':1024**3}},'PREFLIGHT_INSUFFICIENT_MEMORY')]:
            report=forecast(profiles,40000,391,2*1024**3,**kwargs)
            self.assertEqual(report['status'],status);self.assertFalse(m.training_admitted(report))
        good=forecast(profiles,40000,391,7*1024**3)
        self.assertTrue(m.training_admitted(good));good['memory']['limit_bytes']=32*1024**3
        self.assertFalse(m.training_admitted(good))
    def test_effective_host_and_nested_cgroup_memory(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp);proc=root/'proc';cg=root/'cgroup';(proc/'self').mkdir(parents=True);(cg/'nested').mkdir(parents=True)
            (proc/'meminfo').write_text('MemAvailable: 25165824 kB\n',encoding='utf8')
            (proc/'self/cgroup').write_text('0::/nested\n',encoding='utf8')
            (cg/'memory.max').write_text(str(16*1024**3),encoding='utf8');(cg/'memory.current').write_text(str(2*1024**3),encoding='utf8')
            (cg/'nested/memory.max').write_text(str(12*1024**3),encoding='utf8');(cg/'nested/memory.current').write_text(str(3*1024**3),encoding='utf8')
            self.assertEqual(m.effective_memory(proc,cg)['effective_available_bytes'],9*1024**3)
            (cg/'nested/memory.max').write_text('max',encoding='utf8')
            self.assertEqual(m.effective_memory(proc,cg)['effective_available_bytes'],14*1024**3)
            (cg/'memory.max').write_text('invalid',encoding='utf8')
            with self.assertRaises(ValueError):m.effective_memory(proc,cg)
    def test_cold_phase_starts_count_even_when_steady_samples_fit(self):
        profiles={a:profile() for a in m.ARMS}
        for p in profiles.values():p['measurements']['phase_startup_seconds']=100.
        report=forecast(profiles,40000,391,1024**3)
        need=report['session_admission']['required_work_seconds']
        denied=forecast(profiles,need-1,391,1024**3,remaining_whole=need+1800+121)
        self.assertEqual(denied['status'],'PREFLIGHT_INSUFFICIENT_CAP')
        self.assertFalse(m.training_admitted(denied))
        self.assertEqual(report['per_arm']['a0_tail_ctx']['phase_startup'],100.)
        profiles['a0_tail_ctx']['measurements']['phase_startup_seconds']=-1
        with self.assertRaises(ValueError):forecast(profiles,40000,391,1024**3)

    def test_total_experiment_over12h_can_admit_safe_segment(self):
        profiles={a:profile(40) for a in m.ARMS}
        r=forecast(profiles,10000,391,1024**3,remaining_whole=14000)
        self.assertGreater(r['projection']['remaining_with_margin_export'],14000)
        self.assertIn('PREFLIGHT_INSUFFICIENT_CAP',r['full_experiment_diagnostic_blockers'])
        self.assertTrue(m.training_admitted(r))
        r['session_admission']['required_with_export_seconds']=14001
        self.assertFalse(m.training_admitted(r))

    def test_all_blockers_reported_and_no_gate_bypass(self):
        profiles={a:profile() for a in m.ARMS}
        report=forecast(profiles,1,391,15*1024**3,host_memory={'effective_available_bytes':1})
        self.assertEqual(report['blocking_reasons'],['PREFLIGHT_INSUFFICIENT_MEMORY','PREFLIGHT_INSUFFICIENT_HOST_MEMORY','PREFLIGHT_INSUFFICIENT_CAP'])
        self.assertFalse(m.training_admitted(report))
        report['status']='PASS'
        self.assertFalse(m.training_admitted(report))

    def test_only_measured_parent_anonymous_memory_gets_credit(self):
        profiles={a:profile() for a in m.ARMS}
        base=forecast(profiles,40000,391,1024**3)
        gross=base['memory']['host_required_bytes']
        credit=32*1024**2
        # The cgroup already charges this parent's anonymous pages. A new worker
        # only needs the incremental forecast, while retaining the 1 GiB reserve.
        r=forecast(profiles,40000,391,1024**3,host_memory={'effective_available_bytes':gross-credit},current_parent_anon_bytes=credit)
        self.assertTrue(m.training_admitted(r))
        self.assertEqual(r['memory']['host_required_bytes'],gross-credit)
        self.assertEqual(r['memory']['host_reserve_bytes'],1024**3)
        denied=forecast(profiles,40000,391,1024**3,host_memory={'effective_available_bytes':gross-credit-1},current_parent_anon_bytes=credit)
        self.assertFalse(m.training_admitted(denied))
        for value in (-1,True,float('nan'),2*1024**3):
            with self.assertRaises(ValueError):forecast(profiles,40000,391,1024**3,current_parent_anon_bytes=value)

    def test_parent_anon_source_is_not_total_or_file_rss(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            proc=Path(temp);(proc/'self').mkdir()
            p=proc/'self/status'
            p.write_text('VmRSS: 999 kB\nRssAnon: 32 kB\nRssFile: 967 kB\n',encoding='utf-8')
            self.assertEqual(m.parent_anon_bytes(proc),32768)
            for text in ('VmRSS: 999 kB\n','RssAnon: -1 kB\n','RssAnon: 1 MB\n'):
                p.write_text(text,encoding='utf-8')
                with self.assertRaises(ValueError):m.parent_anon_bytes(proc)

    def test_cgroup_unexposed_is_not_silently_host_only(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp);proc=root/'proc';(proc/'self').mkdir(parents=True);cg=root/'cgroup';cg.mkdir()
            (proc/'meminfo').write_text('MemAvailable: 25165824 kB\n',encoding='utf8');(proc/'self/cgroup').write_text('0::/../../host\n',encoding='utf8')
            with self.assertRaisesRegex(RuntimeError,'No readable'):m.effective_memory(proc,cg)
    def test_cgroup_v1_limit_and_usage(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root=Path(temp);proc=root/'proc';(proc/'self').mkdir(parents=True);cg=root/'cgroup';(cg/'memory').mkdir(parents=True)
            (proc/'meminfo').write_text('MemAvailable: 25165824 kB\n',encoding='utf8');(proc/'self/cgroup').write_text('5:cpu,memory:/\n',encoding='utf8')
            (cg/'memory/memory.limit_in_bytes').write_text(str(14*1024**3),encoding='utf8');(cg/'memory/memory.usage_in_bytes').write_text(str(2*1024**3),encoding='utf8')
            self.assertEqual(m.effective_memory(proc,cg)['effective_available_bytes'],12*1024**3)
    @unittest.skipUnless(sys.platform=='linux','Native /proc monitor requires Linux')
    def test_actual_monitor_rejects_group_and_worker_rss(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            monitor=m.GroupMonitor(Path(temp),time.monotonic()+600)
            with mock.patch.object(m,'GROUP_RSS_BYTES',1):
                with self.assertRaisesRegex(MemoryError,'Aggregate'):monitor.check([])
            with mock.patch.object(m,'WORKER_RSS_BYTES',1):
                with self.assertRaisesRegex(MemoryError,'worker process-tree'):monitor.check([SimpleNamespace(pid=m.os.getpid(),poll=lambda:None)])
    def test_nonowned_shell_group_is_rejected(self):
        with mock.patch.object(m.os,'getpgrp',return_value=123,create=True),mock.patch.object(m.os,'getpid',return_value=456):
            with self.assertRaisesRegex(RuntimeError,'own process group'):m.require_owned_process_group()
    def test_fresh_output_required(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            p=Path(temp)/'run';p.mkdir()
            with self.assertRaises(FileExistsError):p.mkdir(exist_ok=False)
    def test_toy_children_overlap_and_inherit_process_group(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            commands={a:[sys.executable,'-I','-S','-B','-c',"import os,time;print(os.environ['CUDA_VISIBLE_DEVICES'],flush=True);time.sleep(.8)"] for a in m.ARMS}
            environments={a:dict(os.environ,CUDA_VISIBLE_DEVICES=m.SLOTS[a]) for a in m.ARMS}
            original=m.subprocess.Popen;kwargs_seen=[]
            def spawn(*args,**kwargs):kwargs_seen.append(kwargs);return original(*args,**kwargs)
            with mock.patch.object(m.subprocess,'Popen',side_effect=spawn):result=m.run_phase(commands,environments,Path(temp)/'logs',time.monotonic()+180,Monitor())
            self.assertGreater(result['overlap_seconds'],0)
            self.assertTrue(all(not k.get('start_new_session',False) and not k.get('creationflags',0) for k in kwargs_seen))
            self.assertEqual((Path(temp)/'logs/er_uniform.log').read_text(encoding='utf8').strip(),'0')
            self.assertEqual((Path(temp)/'logs/a0_tail_ctx.log').read_text(encoding='utf8').strip(),'1')
    def test_one_child_failure_cancels_peer(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            commands={'er_uniform':[sys.executable,'-I','-S','-B','-c','import sys,time;time.sleep(.25);sys.exit(7)'],'a0_tail_ctx':[sys.executable,'-I','-S','-B','-c','import time;time.sleep(60)']}
            environments={a:dict(os.environ) for a in m.ARMS};original=m.subprocess.Popen;processes=[]
            def spawn(*args,**kwargs):p=original(*args,**kwargs);processes.append(p);return p
            started=time.monotonic()
            with mock.patch.object(m.subprocess,'Popen',side_effect=spawn):
                with self.assertRaisesRegex(RuntimeError,'exit=7'):m.run_phase(commands,environments,Path(temp)/'logs',time.monotonic()+180,Monitor())
            self.assertLess(time.monotonic()-started,10);self.assertTrue(all(p.poll() is not None for p in processes))
if __name__=='__main__':unittest.main()
