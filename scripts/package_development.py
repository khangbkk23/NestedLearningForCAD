"""Package a prospectively registered development run; never uploads or launches."""
import argparse,ast,hashlib,json,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()
def write(p,value):
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def package(experiment,owner,run_id,notebook_template=None):
    if experiment not in ('B320','B321','B322','B323','B324'):raise ValueError('Unregistered experiment')
    # owner is the Kaggle account that will hold the private dataset and kernel; it is not stored in the source bundle.
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,49}',owner) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{4,60}',run_id):raise ValueError('Invalid owner/run_id')
    card=f'experiments/{experiment}/card.md';recipe=f'configs/{experiment.lower()}.json'
    default_template=f'notebooks/{experiment.lower()}.ipynb'
    extra_templates={'B321':['notebooks/b321_qualification.ipynb'],'B322':['notebooks/b322_profile.ipynb'],'B323':['notebooks/b323_qualify.ipynb'],'B324':[]}.get(experiment,[])
    allowed_templates=[default_template]+extra_templates
    notebook_template=notebook_template or default_template
    if notebook_template not in allowed_templates:raise ValueError('Unregistered notebook template')
    execution_plan=f'experiments/{experiment}/attempts/{run_id}.md'
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    for name in (card,recipe,execution_plan,*allowed_templates):
        subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch',name],check=True,stdout=subprocess.DEVNULL)
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    if dirty:raise ValueError('Commit scoped design and implementation before packaging')
    run=ROOT/'.local/runs'/experiment.lower()/run_id
    if run.exists():raise FileExistsError(run)
    dataset=run/'package/dataset';kernel=run/'package/kernel'
    dataset.mkdir(parents=True);kernel.mkdir()
    files=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
    files += [ROOT/n for n in ('pixi.toml','pixi.lock','pyproject.toml','LICENSE','.gitattributes','scripts/activate_source.sh','scripts/activate_source.bat','configs/b319.json','configs/train_manifest.json',card,recipe,execution_plan)]
    files += [ROOT/n for n in allowed_templates]
    origin={'schema':'development_source_v1','experiment':experiment,'git':{'commit':commit,'dirty':False},'design_commit':commit,'design_hashes':{n:digest(ROOT/n) for n in (card,recipe,execution_plan)},'execution_plan':execution_plan,'notebook_template':notebook_template,'template_sha256':digest(ROOT/notebook_template)}
    payload={p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(set(files))}
    payload['SOURCE.json']=(json.dumps(origin,indent=2)+'\n').encode('utf8')
    if any(n.startswith('.') and n!='.gitattributes' or '..' in Path(n).parts for n in payload):raise ValueError('Unsafe source allowlist')
    manifest={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}
    raw=(json.dumps(manifest,indent=2)+'\n').encode('utf8');manifest_sha=hashlib.sha256(raw).hexdigest()
    with zipfile.ZipFile(dataset/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('BUNDLE.json',raw)
    sha=digest(dataset/'source.zip')
    nb=json.loads((ROOT/notebook_template).read_text(encoding='utf8'));code=''.join(nb['cells'][0]['source'])
    tree=ast.parse(code);call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_source')
    oldsha,oldmanifest=call.args[2].value,call.args[3].value
    code=code.replace(oldsha,sha).replace(oldmanifest,manifest_sha).replace('__RUN_ID__',run_id)
    compile(code,'packaged-notebook','exec');nb['cells'][0]['source']=code.splitlines(True)
    write(kernel/f'{experiment.lower()}.ipynb',nb)
    dataset_id=owner+'/'+run_id+'-source';kernel_id=owner+'/'+run_id
    write(dataset/'dataset-metadata.json',{'title':run_id+' source','id':dataset_id,'licenses':[{'name':'other'}]})
    write(kernel/'kernel-metadata.json',{'id':kernel_id,'title':run_id,'code_file':experiment.lower()+'.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[dataset_id],'competition_sources':[],'kernel_sources':[]})
    record={**origin,'run_id':run_id,'kernel':kernel_id,'dataset':dataset_id,'source_zip_sha256':sha,'source_zip_bytes':(dataset/'source.zip').stat().st_size,'manifest_sha256':manifest_sha,'files':manifest,'is_private':True}
    write(run/'run.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('files','design_hashes')},indent=2))
    return run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--owner',required=True);p.add_argument('--run-id',required=True);p.add_argument('--notebook-template',choices=('notebooks/b320.ipynb','notebooks/b321.ipynb','notebooks/b321_qualification.ipynb','notebooks/b322.ipynb','notebooks/b322_profile.ipynb','notebooks/b323.ipynb','notebooks/b323_qualify.ipynb','notebooks/b324.ipynb'));a=p.parse_args();package(a.experiment,a.owner,a.run_id,a.notebook_template)
