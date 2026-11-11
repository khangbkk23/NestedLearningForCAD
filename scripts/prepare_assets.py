"""Download and verify the exact MVTec archive and DINOv2 weights used by B319."""
import argparse,hashlib,json,tarfile,time,urllib.request
from pathlib import Path
import sys,uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from nested_replay_cad import paths

def fetch(url,path,expected,expected_bytes=None):
    if path.exists():
        if checksum(path)!=expected:raise ValueError('Existing asset SHA mismatch: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    paths.temporary().mkdir(parents=True,exist_ok=True)
    partial=paths.temporary()/('download-'+expected[:12]+'-'+path.name+'.part')
    h=hashlib.sha256();count=0;last=time.monotonic()
    with urllib.request.urlopen(url,timeout=90) as response,partial.open('xb') as target:
        while chunk:=response.read(1024*1024):
            target.write(chunk);h.update(chunk);count+=len(chunk)
            if time.monotonic()-last>30:
                print('ASSET_DOWNLOAD',path.name,count,flush=True);last=time.monotonic()
    if h.hexdigest()!=expected or expected_bytes is not None and count!=expected_bytes:
        raise ValueError('Downloaded asset identity mismatch')
    partial.rename(path)

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,help='Local artifact root; defaults to NRCAD_LOCAL_ROOT or repo/.local');a=p.parse_args();root=paths.local_root(a.root);paths.process_environment(root);paths.inputs(root).mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/b319.json').read_text())
    weights=paths.weights(root)
    fetch(cfg['encoder']['url'],weights,cfg['encoder']['sha256'],346393545)
    dataset=paths.dataset(root)
    if not dataset.exists():
        archive=paths.cache(root)/'downloads'/'mvtec.tar.xz'
        fetch('https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/eb692b36147a3b3f12c561b879a062104c1f20b8/mvtec_anomaly_detection.tar.xz',archive,'cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d',5264982680)
        staging=paths.temporary(root)/("mvtec-extract-"+uuid.uuid4().hex);staging.mkdir()
        with tarfile.open(archive) as tar:
            members=tar.getmembers()
            if len(members)>30000 or sum(x.size for x in members)>12*1024**3:raise ValueError('Archive size/member cap')
            if any(x.issym() or x.islnk() or not (x.isfile() or x.isdir()) for x in members):raise ValueError('Archive nonregular member')
            tar.extractall(staging,filter='data')
        staging.rename(dataset)
        archive.unlink()
    (paths.inputs(root)/'assets.json').write_text(json.dumps({'dataset':str(dataset),'weights':str(weights),'weights_sha256':cfg['encoder']['sha256'],'archive_sha256':'cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d'},indent=2)+'\n')
    print(json.dumps({'dataset':str(dataset),'weights':str(weights),'status':'ASSETS_READY_FOR_MANIFEST_CHECK'}),flush=True)
