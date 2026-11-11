"""Build packages and import Kaggle evidence into the canonical local run layout."""
import argparse,hashlib,json,os,re,shutil,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nested_replay_cad import paths, provenance
LIGHT_OUTPUT_PATTERN=r'(?i)(\.(json|jsonl|log|txt|ipynb)$|(?:^|/)package/source\.zip$)'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def destination(run,version,name):
    if name.startswith('/') or '\\' in name or ':' in name:raise ValueError('Unsafe provider output path')
    if not re.fullmatch(r'v[0-9]+',version):raise ValueError('Unsafe version label')
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe provider output path')
    parts=list(relative.parts)
    if len(parts)>=4 and parts[:2]==['runs','b319']:parts=parts[3:]
    if parts and parts[0].startswith('.'):raise ValueError('Hidden provider output path')
    if not parts:raise ValueError('Empty provider path')
    if parts[0] in ('logs','outputs','receipts','audit','package'):
        return run/parts[0]/version/Path(*parts[1:])
    if parts[0]=='setup':return run/'logs'/version/relative
    if relative.suffix=='.log':return run/'logs'/version/relative
    if len(parts)==1 and relative.suffix=='.json':return run/'receipts'/version/relative
    return run/'outputs'/version/relative

def pack(args):
    origin=provenance.source_identity(ROOT)
    run=paths.run_root(args.run_id);package=run/'package'
    if package.exists():raise FileExistsError(package)
    package.mkdir(parents=True);dataset=package/'dataset';kernel=package/'kernel';dataset.mkdir();kernel.mkdir()
    manifest,manifest_sha=provenance.source_bundle(ROOT,dataset/'source.zip',origin)
    sha=digest(dataset/'source.zip');nb=json.loads((ROOT/'notebooks/b319.ipynb').read_text());code=''.join(nb['cells'][0]['source'])
    import ast
    tree=ast.parse(code);call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_source')
    code=code.replace(call.args[2].value,sha).replace(call.args[3].value,manifest_sha).replace('__RUN_ID__',run.name)
    compile(code,'packaged-notebook','exec');nb['cells'][0]['source']=code.splitlines(True)
    (kernel/'b319.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    slug=re.sub('[^a-z0-9-]+','-',run.name.lower()).strip('-');dataset_id=args.owner+'/'+slug+'-source';kernel_id=args.owner+'/'+slug
    (dataset/'dataset-metadata.json').write_text(json.dumps({'title':slug+' source','id':dataset_id,'licenses':[{'name':'other'}]},indent=2)+'\n')
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':kernel_id,'title':slug,'code_file':'b319.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[dataset_id],'competition_sources':[],'kernel_sources':[]},indent=2)+'\n')
    record={'schema':'local_run_v1','source_origin':origin,'run_id':run.name,'method':'b319','package_sha256':sha,'manifest_sha256':manifest_sha,'files':manifest,'kernel':kernel_id,'dataset':dataset_id}
    (run/'run.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record|{'files':len(manifest),'run':str(run)},indent=2));return run

def publish_terminal(run,version):
    destination=ROOT/'experiments/B319/runs'
    existing=destination/(run.name+'-'+version)/'receipt.json'
    if existing.exists():
        print('PRESERVED_CLOSED_RECEIPT',existing,flush=True)
        return existing.parent
    from record_run import export_run
    return export_run(run,destination,version=version)

def fetch(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[0-9]+)?',args.kernel):raise ValueError('Expected owner/kernel[/version]')
    if args.token_file:os.environ['KAGGLE_API_TOKEN']=str(args.token_file.resolve(strict=True))
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    api=KaggleApi();api.authenticate();owner,slug,*version=args.kernel.split('/')
    if api.config_values.get(api.CONFIG_NAME_USER,'').casefold()!=owner.casefold():raise ValueError('Authenticated owner mismatch')
    if not version:
        with api.build_kaggle_client() as client:
            req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug;meta=client.kernels.kernels_api_client.get_kernel(req).metadata
        version=[str(meta.current_version_number)]
    ref=owner+'/'+slug+'/'+version[0];label='v'+version[0];run=paths.run_root(args.run_id);receipts=run/'receipts'/label;receipts.mkdir(parents=True,exist_ok=True)
    metadata=run/'run.json'
    info=json.loads(metadata.read_text()) if metadata.exists() else {'schema':'local_run_v1','run_id':run.name,'method':'b319'}
    if info.get('kernel','').split('/')[:2] not in ([''],[owner,slug]):raise ValueError('Run ID belongs to another kernel')
    state=api.kernels_status(ref);record={'kernel':ref,'status':str(state.status),'failure_message':state.failure_message}
    info.update(kernel=owner+'/'+slug,latest_fetched_version=version[0],provider_status=str(state.status))
    metadata.write_text(json.dumps(info,indent=2)+'\n')
    (receipts/'provider-status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    logs=run/'logs'/label;logs.mkdir(parents=True,exist_ok=True);raw=api.kernels_logs(ref);(logs/'execution-raw.log').write_text(raw,encoding='utf8')
    try:readable=''.join(x.get('data','') for x in json.loads(raw))
    except Exception:readable=raw
    (logs/'execution.log').write_text(readable,encoding='utf8')
    if not str(state.status).endswith(('.COMPLETE','.ERROR','.CANCEL_ACKNOWLEDGED')):return run
    paths.temporary().mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='kaggle-fetch-',dir=paths.temporary()))
    pattern=None if args.raw else LIGHT_OUTPUT_PATTERN
    files,_=api.kernels_output(ref,path=str(staging),file_pattern=pattern,quiet=True);inventory=[]
    for source in files:
        source=Path(source);name=source.relative_to(staging).as_posix();target=destination(run,label,name);target.parent.mkdir(parents=True,exist_ok=True);sha=digest(source)
        if target.exists():
            if digest(target)!=sha:raise FileExistsError('Different evidence at '+str(target))
            source.unlink()
        else:source.rename(target)
        inventory.append({'provider_path':name,'local_path':target.relative_to(run).as_posix(),'bytes':target.stat().st_size,'sha256':sha})
    (receipts/'download-manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    shutil.rmtree(staging)
    publish_terminal(run,label)
    print('FETCHED',len(inventory),'files into',run,flush=True);return run

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('pack');p.add_argument('--run-id');p.add_argument('--owner',required=True)
    p=sub.add_parser('fetch');p.add_argument('--kernel',required=True);p.add_argument('--run-id',required=True);p.add_argument('--token-file',type=Path);p.add_argument('--raw',action='store_true')
    args=parser.parse_args();return pack(args) if args.action=='pack' else fetch(args)

if __name__=='__main__':main()
