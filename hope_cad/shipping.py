"""Content-bound portable source ZIP; never imports old bundles or shared .agents."""
import json,zipfile,hashlib,subprocess
from pathlib import Path
from .runtime import ROOT,source_files

def package(output):
    output=Path(output).resolve();files=source_files()
    if output.exists():raise ValueError('Source delivery already exists; choose a new revision/path')
    if output in [(ROOT/n).resolve() for n in files]:raise ValueError('Output overlaps source')
    output.parent.mkdir(parents=True,exist_ok=True)
    origin={}
    try:
        origin['source_commit']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        origin['branch']=subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True,stderr=subprocess.DEVNULL).strip()
        origin['git_dirty']=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],stderr=subprocess.DEVNULL))
    except (OSError,subprocess.SubprocessError):origin['source_commit']=None
    manifest=dict(schema='hope_cad_portable_source_bundle_v1',files=files,origin=origin,
        identity_basis='all packaged file SHA256s; no commit-only claim',method='hope_cad_v5_1_causal_scaled_utility')
    tmp=output.with_name(output.name+'.tmp')
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for n in files:z.write(ROOT/n,n)
            z.writestr('bundle_manifest.json',json.dumps(manifest,sort_keys=True))
        with zipfile.ZipFile(tmp) as z:
            for n,h in files.items():
                if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Source changed during shipping: '+n)
        tmp.replace(output)
    finally:tmp.unlink(missing_ok=True)
    with output.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(output),files=len(files),sha256=checksum,method_changed=False,remote_validated=False)
