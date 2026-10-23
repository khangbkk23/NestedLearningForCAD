"""Provision pinned MVTec AD into ignored repo data/, or an explicit directory."""
import argparse,json,stat,tarfile,tempfile,zipfile
from pathlib import Path,PurePosixPath
from hope_cad.paths import ROOT
from hope_cad.data.transfer import download,checked_file,sha256,write_json,writer_lock

REVISION='eb692b36147a3b3f12c561b879a062104c1f20b8'
DOWNLOAD_URL=f'https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/{REVISION}/mvtec_anomaly_detection.tar.xz'
ARCHIVE_SHA256='cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d'
ARCHIVE_BYTES=5264982680
CATEGORIES=('bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper')

def asset_destination(path):
    path=Path(path).absolute()
    if path.is_symlink():raise ValueError('Asset destination cannot be a symlink')
    resolved=path.resolve()
    if ROOT.is_relative_to(resolved):raise ValueError('Assets cannot contain source checkout')
    if resolved.is_relative_to(ROOT) and resolved.relative_to(ROOT).parts[0] not in ('data','weights'):
        raise ValueError('Use ignored data/weights or an external asset directory')
    return resolved

def validate_tree(root):
    return all((root/c/'train/good').is_dir() and (root/c/'test').is_dir() and (root/c/'ground_truth').is_dir() for c in CATEGORIES)

def tree_manifest(root):
    if not validate_tree(root):raise ValueError('Expected complete15-category MVTec AD tree')
    files={};train=test=bottle=0
    for c in CATEGORIES:
        for p in sorted((root/c).rglob('*')):
            if p.is_symlink():raise ValueError('Symlink in dataset')
            if p.is_file():
                rel=p.relative_to(root).as_posix();files[rel]=sha256(p)
                if p.suffix.lower()=='.png':
                    if '/train/good/' in rel:train+=1;bottle+=int(c=='bottle')
                    if '/test/' in rel:test+=1
    if (train,bottle,test)!=(3629,209,1725):raise ValueError('MVTec counts differ from3629/209Bottle/1725test')
    return dict(files=files,train_normals=train,bottle_normals=bottle,test_images=test)

def unpack(archive,stage):
    def check(name):
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive path')
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate archive paths')
            for item in z.infolist():
                check(item.filename)
                if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Archive symlink')
            z.extractall(stage)
    else:
        with tarfile.open(archive,'r:*') as tf:
            members=tf.getmembers();names=[m.name for m in members]
            if len(names)!=len(set(names)):raise ValueError('Duplicate archive paths')
            for item in members:
                check(item.name)
                if not (item.isfile() or item.isdir()):raise ValueError('Archive links/devices forbidden')
            tf.extractall(stage,members=members,filter='data')

def prepare(output=ROOT/'data/mvtec',*,archive=None,archive_sha256=None):
    output=asset_destination(output);output.parent.mkdir(parents=True,exist_ok=True)
    receipt_path=output.parent/(output.name+'.asset.json')
    with writer_lock(output.parent/(output.name+'.prepare.lock')):
        if output.exists():
            if not receipt_path.exists():raise ValueError('Existing dataset has no receipt; inspect rather than overwrite')
            receipt=json.loads(receipt_path.read_text(encoding='utf8'))
            if receipt.get('schema')!='hope_mvtec_asset_v1' or tree_manifest(output)!=receipt['tree']:raise ValueError('Dataset/receipt changed')
            return receipt
        if archive is None:
            archive=output.parent/'archives/mvtec_anomaly_detection.tar.xz'
            evidence=download(DOWNLOAD_URL,archive,ARCHIVE_SHA256,ARCHIVE_BYTES)
            provenance=dict(source='third_party_mirror',url=DOWNLOAD_URL,revision=REVISION,
                scope='Pinned mirror bytes, not independently authenticated author distribution')
        else:
            archive=Path(archive).resolve()
            if not archive_sha256:raise ValueError('Explicit archive requires --archive-sha256')
            evidence=checked_file(archive,archive_sha256)
            provenance=dict(source='operator_supplied_archive',scope='Operator SHA records integrity, not author provenance')
        with tempfile.TemporaryDirectory(dir=output.parent,prefix='.mvtec-stage-') as t:
            stage=Path(t);unpack(archive,stage)
            roots=[p for p in (stage,*stage.iterdir()) if p.is_dir() and validate_tree(p)]
            if len(roots)!=1:raise ValueError('Expected one unambiguous MVTec root')
            tree=tree_manifest(roots[0])
            receipt=dict(schema='hope_mvtec_asset_v1',archive=evidence,provenance=provenance,tree=tree,
                scope='Asset integrity/counts only; no metrics/learning or independent authenticity claim')
            write_json(receipt_path,receipt);roots[0].rename(output)
        print('[ASSET READY]',output,flush=True);return receipt

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/mvtec')
    p.add_argument('--archive',type=Path);p.add_argument('--archive-sha256')
    a=p.parse_args(argv);result=prepare(a.output,archive=a.archive,archive_sha256=a.archive_sha256)
    print(json.dumps(dict(output=str(a.output),train=result['tree']['train_normals'],test=result['tree']['test_images']),indent=2))

if __name__=='__main__':main()
