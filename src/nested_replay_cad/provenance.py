"""Exact source bytes and lightweight execution receipts; no training imports."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE_DIRS=('src','scripts','configs','tests','experiments','notebooks')
SOURCE_FILES=('pixi.toml','pixi.lock','pyproject.toml','LICENSE','.gitattributes')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    staged=path.with_suffix(path.suffix+'.partial')
    staged.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');staged.replace(path)

def source_files(root):
    return sorted([p for folder in SOURCE_DIRS if (root/folder).exists()
        for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        +[root/name for name in SOURCE_FILES if (root/name).is_file()])

def git_state(root):
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL,text=True).strip()
    try:
        if Path(git('rev-parse','--show-toplevel')).resolve()!=root.resolve():raise ValueError('Ancestor repository')
        return {'commit':git('rev-parse','HEAD'),'dirty':bool(git('status','--porcelain'))}
    except (OSError,ValueError,subprocess.CalledProcessError):
        origin=root/'SOURCE.json'
        return json.loads(origin.read_text()).get('git',{'commit':None,'dirty':None}) if origin.exists() else {'commit':None,'dirty':None}

def check_design(root,experiment='B319',recipe='configs/b319.json'):
    """Freeze card+recipe before a new execution; allow dirty code when bundled."""
    card=f'experiments/{experiment}/card.md'
    try:
        subprocess.run(['git','-C',str(root),'ls-files','--error-unmatch',card,recipe],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        changed=subprocess.check_output(['git','-C',str(root),'diff','HEAD','--name-only','--',card,recipe],text=True).strip()
        if changed:raise ValueError('Commit the experiment card and recipe before running')
        return git_state(root)['commit']
    except (OSError,subprocess.CalledProcessError):
        origin=root/'SOURCE.json'
        if origin.exists():
            info=json.loads(origin.read_text());commit=info.get('design_commit')
            if commit and info.get('design_hashes')=={name:digest(root/name) for name in (card,recipe)}:return commit
        raise ValueError('Commit the experiment card and recipe before packaging/running') from None

def source_identity(root,experiment='B319',recipe='configs/b319.json'):
    return {'schema':'source_origin_v1','git':git_state(root),'design_commit':check_design(root,experiment,recipe),
        'design_hashes':{name:digest(root/name) for name in (f'experiments/{experiment}/card.md',recipe)}}

def source_bundle(root,destination,origin):
    payload={p.relative_to(root).as_posix():p.read_bytes() for p in source_files(root)}
    payload['SOURCE.json']=(json.dumps(origin,indent=2)+'\n').encode()
    manifest={name:hashlib.sha256(raw).hexdigest() for name,raw in payload.items()}
    raw=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,data in payload.items():archive.writestr(name,data)
        archive.writestr('BUNDLE.json',raw)
    return manifest,hashlib.sha256(raw).hexdigest()

def runtime_info():
    versions={}
    for name in ('torch','torchvision','numpy','timm','Pillow','safetensors'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    value={'python':platform.python_version(),'system':platform.system(),'packages':versions,'GPU':None,'CUDA':None}
    torch=sys.modules.get('torch')
    if torch is not None:
        value['CUDA']=torch.version.cuda
        if torch.cuda.is_initialized():
            try:value['GPU']=torch.cuda.get_device_name(torch.cuda.current_device())
            except RuntimeError as error:value['GPU_probe_error']=repr(error)
    return value

class RunRecord:
    """run.json is live; receipt.json is sealed once, including failed executions."""
    def __init__(self,root,folder,output,config,config_path,run_id,arm):
        self.root=Path(root);self.folder=Path(folder);self.output=Path(output);self.started=time.monotonic()
        self.path=self.folder/'run.json';self.closed=False
        origin=source_identity(self.root,recipe=Path(config_path).resolve().relative_to(self.root.resolve()).as_posix())
        previous=json.loads(self.path.read_text()) if self.path.exists() else {}
        if (self.folder/'receipts/receipt.json').exists():raise FileExistsError('Run already closed')
        bundle=self.folder/'package/source.zip';source_bundle(self.root,bundle,origin)
        self.value=previous|{'schema':'execution_run_v1','run_id':run_id,'experiment':'B319','arm':arm,'seed':config['seed'],
            'state':'running','started_at':datetime.now(timezone.utc).isoformat(),'git':origin['git'],'design_commit':origin['design_commit'],
            'command':[sys.executable,*sys.argv],'config_sha256':digest(config_path),'manifest_sha256':config['full_manifest_sha256'],
            'pixi_lock_sha256':digest(self.root/'pixi.lock'),'source_bundle':{'path':'package/source.zip','sha256':digest(bundle),'bytes':bundle.stat().st_size},
            'runtime':runtime_info(),'parent_run_id':None,'parent_checkpoint':None,'scientific_acceptance':False}
        write_json(self.path,self.value)
        write_json(self.folder/'receipts/config.resolved.json',config)

    def finish(self,state,reason=None):
        if self.closed:return
        self.value.update(state=state,stop_reason=reason,ended_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-self.started,runtime=runtime_info())
        arm_runtime=sorted(self.output.glob('*/runtime.json'))
        if (self.output/'runtime.json').exists():arm_runtime.append(self.output/'runtime.json')
        self.value['arm_runtime']={p.parent.name:json.loads(p.read_text()) for p in arm_runtime}
        candidates=[self.output/'metrics.json',self.output/'summary.json',*arm_runtime,
                    *self.output.glob('*/full15_cohort.json'),*self.output.glob('*/full15_selection.json'),
                    *self.output.glob('*/config.json'),*self.output.glob('*/arms/*/study.json')]
        self.value['files']={p.relative_to(self.folder).as_posix():digest(p) for p in candidates if p.is_file() and p.is_relative_to(self.folder)}
        write_json(self.path,self.value);write_json(self.folder/'receipts/receipt.json',self.value);self.closed=True
