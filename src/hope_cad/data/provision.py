"""Explicit CPU/network-only asset provisioning; never starts training."""
import argparse,json
from pathlib import Path
from hope_cad.paths import ROOT, DATA, WEIGHTS
from hope_cad.data.download import prepare,asset_destination
from hope_cad.data.transfer import download,write_json

VIT_REVISION='4747ebdcc1c6ba9275d50c702b59426911f686b8'
VIT_URL=f'https://huggingface.co/timm/vit_base_patch8_224.augreg_in21k/resolve/{VIT_REVISION}/model.safetensors'
VIT_SHA256='2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec'
VIT_BYTES=410434646
VIT_NAME='vit_base_patch8_224.augreg_in21k.safetensors'

def provision_vit(path=WEIGHTS/VIT_NAME):
    path=asset_destination(path);receipt=download(VIT_URL,path,VIT_SHA256,VIT_BYTES)
    write_json(path.with_suffix('.asset.json'),dict(schema='hope_vit_asset_v1',**receipt,revision=VIT_REVISION))
    print('[ASSET READY]',path,flush=True);return receipt

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only',choices=('all','mvtec','vit'),default='all')
    p.add_argument('--data-root',type=Path,default=DATA/'mvtec')
    p.add_argument('--checkpoint',type=Path,default=WEIGHTS/VIT_NAME)
    p.add_argument('--archive',type=Path);p.add_argument('--archive-sha256')
    a=p.parse_args(argv);out={}
    if a.only in ('all','mvtec'):out['mvtec']=prepare(a.data_root,archive=a.archive,archive_sha256=a.archive_sha256)['archive']
    if a.only in ('all','vit'):out['vit']=provision_vit(a.checkpoint)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
