"""Record installed metadata without installing packages or running ML."""
import argparse,json,platform,sys,time,subprocess
from importlib.metadata import distributions,version

def main():
    from hope_cad.paths import ROOT, RECORDS
    from hope_cad.data.transfer import write_json,sha256
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    pins=json.loads((ROOT/'src/hope_cad/resources/runtime.json').read_text())
    if sys.platform=='linux':
        if sys.version_info[:2]!=(3,13):raise ValueError('Linux runtime requires Python3.13')
        for name,expected in pins['packages'].items():
            if version(name)!=expected:raise ValueError(f'{name}: expected {expected}, found {version(name)}')
    value=dict(schema='hope_pixi_environment_v1',platform=platform.platform(),python=sys.version,
        executable=sys.executable,pixi_lock_sha256=sha256(ROOT/'pixi.lock'),
        packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in distributions()],key=lambda d:d['name'].lower()),
        scientific_acceptance=False,scope='Installed metadata; no GPU kernels, preflight or AD acceptance')
    value['pixi_version']=subprocess.check_output(['pixi','--version'],text=True,timeout=10).strip()
    if a.gpu:
        import torch
        value['GPU']=dict(available=torch.cuda.is_available(),cuda_wheel=torch.version.cuda,
            architectures=torch.cuda.get_arch_list(),name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    path=RECORDS/'environment'/f'{time.time_ns()}.json';write_json(path,value)
    print(json.dumps(dict(receipt=str(path),python=platform.python_version(),linux_pins_checked=sys.platform=='linux',scientific_acceptance=False),indent=2))

if __name__=='__main__':main()
