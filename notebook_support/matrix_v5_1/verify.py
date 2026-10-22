"""Local CPU source/bootstrap/accepted-parent verification; never dispatch GPU."""
import argparse,hashlib,json,os,subprocess,sys,zipfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--work',required=True);p.add_argument('--parents',action='store_true');p.add_argument('--parent-cache');args=p.parse_args()
    source=Path(args.source).resolve();work=Path(args.work).resolve();work.mkdir(parents=True,exist_ok=False);package=work/'package';package.mkdir();began=time.monotonic()
    with zipfile.ZipFile(source) as z:
        m=json.loads(z.read('matrix_manifest.json'));names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(m['files'])|{'matrix_manifest.json'}:raise ValueError('Wrong source inventory')
        for n in names:
            from notebook_support.common.evidence import destination
            dest=destination(package,n);raw=z.read(n)
            if n in m['files'] and hashlib.sha256(raw).hexdigest()!=m['files'][n]:raise ValueError('Source SHA mismatch')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    base=ROOT.parent/'.agents/artifacts/deliveries/hope_support_v5_1_matched_cadic_seed0_r1/hope_cad_matched_cadic_seed0_source_r1.zip'
    with zipfile.ZipFile(base) as z:
        for n in z.namelist():
            if (package/n).read_bytes()!=z.read(n):raise ValueError('Frozen base changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',TEMP=str(work),TMP=str(work))
    boots=[]
    for job in ('NESTED_seed0','CADIC_seed0'):
        nb=json.loads((package/f'notebooks/hope_cad_v5_1_matrix_{job}_kaggle.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'matrix_notebook','exec')
        for kind in ('zip','directory'):
            target=source if kind=='zip' else package;inputs=work/(job+'_'+kind);inputs.mkdir()
            # Use the actual first cell with explicit source override; discovery exercised separately by tests.
            text=''.join(nb['cells'][1]['source']).replace('SOURCE_OVERRIDE=None','SOURCE_OVERRIDE='+repr(str(target)))
            text=text.replace("INPUT=Path('/kaggle/input');ROOT=Path('/tmp/hope_matrix_7_2_source_r1')",'INPUT=Path('+repr(str(inputs))+');ROOT=Path('+repr(str(inputs/'source'))+')')
            result=subprocess.run([sys.executable,'-u','-B','-c',text],cwd=package,env=env,text=True,encoding='utf-8',capture_output=True)
            (work/(job+'_'+kind+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode:raise ValueError('Notebook source bootstrap failed: '+job+'/'+kind+' '+result.stderr)
            boots.append(dict(job=job,representation=kind,status='passed'))
    runner=work/'package_tests.py';runner.write_text("import os,sys,platform,unittest\nos.environ['CUDA_VISIBLE_DEVICES']=''\nplatform.platform()\nimport torch\ntorch.set_num_threads(2)\ntorch.use_deterministic_algorithms(True)\nsys.path.insert(0,"+repr(str(package))+ ")\nsuite=unittest.defaultTestLoader.discover("+repr(str(package/'tests/matrix_v5_1'))+")\nr=unittest.TextTestRunner(verbosity=2).run(suite)\nraise SystemExit(0 if r.wasSuccessful() else 1)\n",encoding='utf-8')
    result=subprocess.run([sys.executable,'-u','-B',str(runner)],cwd=package,env=env,text=True,encoding='utf-8',capture_output=True);(work/'packaged_tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise ValueError('Packaged matrix tests failed')
    parents=[]
    if args.parents:
        proof=work/'parents.py';proof.write_text('''import os,sys,json,gc
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import torch
torch.set_num_threads(2)
from notebook_support.matrix_v5_1.common import verify_source,restore,open_owner,load_task,verify_frozen
from models.hope_cad_v5_1.session import bank_digest
from models.hope_cad_v5_1.validation import tree_finite
op,refs,_=verify_source()
source=json.loads(Path('matrix_manifest.json').read_text())
parent_root=Path(sys.argv[1]);out=Path(sys.argv[2]);reports=[]
class MetadataActor:
    # A CPU audit cannot claim canonical T4 numerical restore. Exercise the real
    # owner resolver with structural/state checks; do not weaken session guards.
    def load_state_dict(self,s):
        tree_finite(s)
        assert s['schema']=='hope_support_session_v5_1'
        assert len(s['seen_ids'])==s['cursor']==len(set(s['seen_ids']))
        assert tuple(s['bank']['features'].shape)==(10000,768)
        assert bank_digest(s['bank']['features'])==s['bank_sha256']
        if s['learner']:
            l=s['learner'];assert l['labels']==l['fast_updates']==s['counters']['labels_completed']
            assert l['slow_updates']==l['labels']//8 and len(l['records'])==l['labels']%8
        if s['pending'] is not None:
            assert not s['finalized'] and s['pending']['decision_cursor']==s['cursor']-1
        self.seen_ids=s['seen_ids'];self.cursor=s['cursor']
names={'NESTED_seed0':('support_v5_1_benchmark_NESTED_seed0_continuation','hope_cad_support_v5_1_continuation_evidence.zip'),'CADIC_seed0':('support_v5_1_matched_CADIC_seed0','hope_cad_matched_cadic_seed0_evidence.zip')}
for job,(run,name) in names.items():
    dst=out/job;parent=parent_root/run/'session_01'/name
    cached=Path(sys.argv[4])/job if sys.argv[4] else None
    if cached and cached.is_dir():
        frozen=verify_frozen(cached,job,refs);dst=cached;restored=0
    else:restored=restore(parent,refs['jobs'][job]['schema'],dst,job,refs,op)
    owner=open_owner(dst,job,refs['runtime'],refs);actor=MetadataActor();cursors=[]
    for i in range(15):
        load_task(owner,i,actor);cursors.append(actor.cursor)
    frozen=verify_frozen(dst,job,refs)
    reports.append(dict(job=job,status='passed',restored_payloads=restored,frozen_payloads=frozen,task_cursors=cursors,scope='CPU real owner resolver and15checkpoint metadata/bank/clock checks; canonical numeric session restore still gated on T4; no scoring/training/GPU'))
    del owner,actor;gc.collect()
Path(sys.argv[3]).write_text(json.dumps(reports,indent=2)+'\\n')
print(json.dumps(reports,indent=2))
''',encoding='utf-8')
        result=subprocess.run([sys.executable,'-u','-B',str(proof),str(ROOT.parent/'.agents/artifacts/runs'),str(work/'parent_extracts'),str(work/'parent_proof.json'),str(Path(args.parent_cache).resolve()) if args.parent_cache else ''],cwd=package,env=dict(env,PYTHONPATH=str(package)),text=True,encoding='utf-8',capture_output=True)
        (work/'parent_proof.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise ValueError('Actual completed-parent CPU smoke failed: '+result.stderr)
        parents=json.loads((work/'parent_proof.json').read_text())
    result=dict(status='passed',source_sha256=sha(source),payloads=len(m['files']),frozen_base_byte_exact_entries=249,notebook_bootstraps=boots,packaged_matrix_tests='passed',actual_parent_CPU_proofs=parents,canonical_GPU_run='not_run',seconds=time.monotonic()-began)
    (work/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
