"""Stdlib-only regression for resource-stop classification and source whitelist."""
import ast,contextlib,hashlib,importlib.util,io,json,tempfile,unittest,zipfile,os,shutil,subprocess,sys
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('development_packager',ROOT/'scripts/package_development.py')
pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
code=''.join(json.loads((ROOT/'notebooks/b321.ipynb').read_text(encoding='utf8'))['cells'][0]['source'])
node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='registered_resource_stop')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<resource-stop>','exec'),ns)
class TransportRegression(unittest.TestCase):
    def payload(self):return {'experiment':'B321','status':'PREFLIGHT_INSUFFICIENT_CAP','studies':{},'preflight':{'status':'PREFLIGHT_INSUFFICIENT_CAP','projection':{'remaining_worker_with_margin':59200,'remaining_worker_cap':39200,'remaining_with_margin_export':61000,'remaining_whole_cap':41000}}}
    def test_registered_stop_is_recognized(self):self.assertTrue(ns['registered_resource_stop'](2,self.payload()))
    def test_execution_error_is_not_relabelled(self):self.assertFalse(ns['registered_resource_stop'](1,self.payload()))
    def test_started_training_is_not_a_preflight_stop(self):
        p=self.payload();p['studies']={'er_uniform':{'status':'FAILED'}}
        self.assertFalse(ns['registered_resource_stop'](2,p))
    def test_missing_or_inconsistent_payload_is_not_a_stop(self):
        self.assertFalse(ns['registered_resource_stop'](2,{}));p=self.payload();p['preflight']['projection']['remaining_with_margin_export']=100;p['preflight']['projection']['remaining_worker_with_margin']=100
        self.assertFalse(ns['registered_resource_stop'](2,p))
    def test_activation_files_are_explicitly_in_source_whitelist(self):
        tree=ast.parse((ROOT/'scripts/package_development.py').read_text(encoding='utf8'))
        values={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        for n in ('scripts/activate_source.sh','scripts/activate_source.bat'):
            self.assertIn(n,values);self.assertTrue((ROOT/n).is_file())
    def test_memory_projection_refusal_is_explicit(self):
        p=self.payload();p['status']=p['preflight']['status']='PREFLIGHT_INSUFFICIENT_MEMORY'
        p['preflight']['memory']={'projected_group_RSS_bytes':17*1024**3,'limit_bytes':16*1024**3,'worker_limit_bytes':8*1024**3,'projected_worker_RSS_bytes':{'er_uniform':7*1024**3,'a0_tail_ctx':7*1024**3}}
        self.assertTrue(ns['registered_resource_stop'](2,p))
    def test_memory_gate_cannot_be_used_to_raise_allowed_cap(self):
        p=self.payload();p['status']=p['preflight']['status']='PREFLIGHT_INSUFFICIENT_MEMORY'
        p['preflight']['memory']={'projected_group_RSS_bytes':33*1024**3,'limit_bytes':32*1024**3,'worker_limit_bytes':8*1024**3,'projected_worker_RSS_bytes':{'er_uniform':7*1024**3,'a0_tail_ctx':7*1024**3}}
        self.assertFalse(ns['registered_resource_stop'](2,p))
    def test_host_and_worker_refusal_payloads(self):
        p=self.payload();p['status']=p['preflight']['status']='PREFLIGHT_INSUFFICIENT_HOST_MEMORY'
        p['preflight']['memory']={'projected_group_RSS_bytes':10*1024**3,'limit_bytes':16*1024**3,'worker_limit_bytes':8*1024**3,'projected_worker_RSS_bytes':{'er_uniform':5*1024**3,'a0_tail_ctx':5*1024**3},'host_required_bytes':11*1024**3,'host':{'effective_available_bytes':9*1024**3}}
        self.assertTrue(ns['registered_resource_stop'](2,p))
        p['preflight']['memory']['host']['effective_available_bytes']=12*1024**3
        self.assertFalse(ns['registered_resource_stop'](2,p))
        p['status']=p['preflight']['status']='PREFLIGHT_INSUFFICIENT_MEMORY';p['preflight']['memory']['projected_worker_RSS_bytes']['a0_tail_ctx']=9*1024**3
        self.assertTrue(ns['registered_resource_stop'](2,p))
    def test_worker_time_block_is_recognized_when_whole_fits(self):
        p=self.payload();p['preflight']['projection'].update(remaining_worker_with_margin=40001,remaining_worker_cap=40000,remaining_with_margin_export=41801,remaining_whole_cap=42000)
        self.assertTrue(ns['registered_resource_stop'](2,p))
        p['preflight']['projection']['remaining_worker_with_margin']=float('nan')
        self.assertFalse(ns['registered_resource_stop'](2,p))
    def test_notebook_uses_parallel_parent_with_same_wall_cap(self):
        self.assertIn('scripts/parallel_budget.py',code)
        self.assertNotIn("'scripts/budget_train.py'",code)
        self.assertIn('START+43200',code)
    def test_real_minimal_bundle_preserves_plan_and_excludes_private_state(self):
        scratch=ROOT/'.local/cache/tests/b321-transport-a';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as d:
            root=Path(d);rid='b320-synthetic-package-check'
            plan='experiments/B320/attempts/'+rid+'.md'
            names=['src/demo.py','scripts/demo.py','tests/demo.py','scripts/activate_source.sh','scripts/activate_source.bat','pixi.toml','pixi.lock','pyproject.toml','LICENSE','.gitattributes','configs/b319.json','configs/b320.json','configs/train_manifest.json','experiments/B320/card.md',plan,'.private/credentials.json']
            for n in names:
                f=root/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture only\n',encoding='utf8')
            f=root/'notebooks/b320.ipynb';f.parent.mkdir();f.write_text(json.dumps({'cells':[{'source':["load_source(None,None,'"+'a'*64+"','"+'b'*64+"')\nRUN_ID='__RUN_ID__'\n"]}]}),encoding='utf8')
            def git_output(args,**kwargs):return '' if 'status' in args else '1'*40
            with patch.object(pack,'ROOT',root),patch.object(pack.subprocess,'check_output',side_effect=git_output),patch.object(pack.subprocess,'run'),contextlib.redirect_stdout(io.StringIO()):
                run=pack.package('B320','owner-a',rid)
                with self.assertRaises(FileExistsError):pack.package('B320','owner-a',rid)
            receipt=json.loads((run/'run.json').read_text(encoding='utf8'))
            with zipfile.ZipFile(run/'package/dataset/source.zip') as archive:
                self.assertIsNone(archive.testzip());self.assertIn(plan,archive.namelist())
                self.assertIn('scripts/activate_source.sh',archive.namelist())
                self.assertFalse(any(n.startswith('.') and n!='.gitattributes' for n in archive.namelist()))
                for n,h in receipt['files'].items():self.assertEqual(hashlib.sha256(archive.read(n)).hexdigest(),h)
                source=json.loads(archive.read('SOURCE.json'));self.assertIn(plan,source['design_hashes'])
            metadata=json.loads((run/'package/kernel/kernel-metadata.json').read_text(encoding='utf8'))
            self.assertTrue(metadata['is_private']);self.assertEqual(metadata['id'],'owner-a/'+rid)
            code=''.join(json.loads((run/'package/kernel/b320.ipynb').read_text(encoding='utf8'))['cells'][0]['source'])
            self.assertNotIn('__RUN_ID__',code);self.assertIn(receipt['source_zip_sha256'],code)
    def test_qualification_bundle_runs_filesystem_suite_without_checkout(self):
        scratch=ROOT/'.local/cache/tests/b321-transport-b';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as d:
            fixture=Path(d)/'checkout';fixture.mkdir();rid='b321-synthetic-package-review'
            names=['pixi.toml','pixi.lock','pyproject.toml','LICENSE','.gitattributes','scripts/activate_source.sh','scripts/activate_source.bat','configs/b319.json','configs/b321.json','configs/train_manifest.json','experiments/B321/card.md','notebooks/b321.ipynb','notebooks/b321_qualification.ipynb']
            names += [p.relative_to(ROOT).as_posix() for scope in ('src','scripts','tests') for p in (ROOT/scope).rglob('*.py') if '__pycache__' not in p.parts]
            for name in set(names):
                dest=fixture/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
            plan=fixture/('experiments/B321/attempts/'+rid+'.md');plan.parent.mkdir(parents=True);plan.write_text('Synthetic packaging fixture; never upload or executeGPU.\n',encoding='utf8')
            def git_output(args,**kwargs):return '' if 'status' in args else '1'*40
            # Metadata source is synthetic; execution below is stdlibfilesystem only.
            with patch.object(pack,'ROOT',fixture),patch.object(pack.subprocess,'check_output',side_effect=git_output),patch.object(pack.subprocess,'run'),contextlib.redirect_stdout(io.StringIO()):
                run=pack.package('B321','owner-b',rid,'notebooks/b321_qualification.ipynb')
            record=json.loads((run/'run.json').read_text(encoding='utf8'));self.assertEqual(record['notebook_template'],'notebooks/b321_qualification.ipynb')
            isolated=Path(d)/'isolated';isolated.mkdir()
            with zipfile.ZipFile(run/'package/dataset/source.zip') as archive:
                self.assertIsNone(archive.testzip())
                for name in ('notebooks/b321.ipynb','notebooks/b321_qualification.ipynb'):self.assertIn(name,archive.namelist())
                for name,digest in record['files'].items():self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),digest)
                archive.extractall(isolated)
            notebook=json.loads((run/'package/kernel/b321.ipynb').read_text(encoding='utf8'));text=''.join(notebook['cells'][0]['source'])
            self.assertIn('scripts/qualify_budget_resume.py',text);self.assertIn('START+1800',text);self.assertNotIn('scripts/prepare_assets.py',text);self.assertNotIn('__RUN_ID__',text)
            env=dict(os.environ,NRCAD_LOCAL_ROOT=str(Path(d)/'local'),PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
            cmd=[sys.executable,'-I','-S','-B',str(isolated/'tests/test_storage_race.py')]
            result=subprocess.run(cmd,cwd=isolated,env=env,capture_output=True,text=True,encoding='utf8',timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            # Demonstrate the original missing member fails; never skip this gate.
            (isolated/'notebooks/b321.ipynb').unlink()
            bad=subprocess.run(cmd+['CensusTests.test_outer_and_parent_no_unsafe_census'],cwd=isolated,env=env,capture_output=True,text=True,encoding='utf8',timeout=30)
            self.assertNotEqual(bad.returncode,0);self.assertIn('FileNotFoundError',bad.stderr)
    def test_wrong_experiment_template_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unregistered notebook'):pack.package('B320','owner-a','b320-synthetic-review','notebooks/b321_qualification.ipynb')
    def test_notebook_and_python_compile(self):
        compile(code,'b321-notebook','exec');compile((ROOT/'scripts/episode_diagnostic.py').read_text(encoding='utf8'),'diagnostic','exec')
if __name__=='__main__':unittest.main()
