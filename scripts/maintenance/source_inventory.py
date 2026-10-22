"""Update explicit portable source inventory after reviewed source changes; no runtime/data scan."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
AREAS=('hope_cad','models','training','dataset','conf','scripts','tests','docs','refs','requirements')
SUFFIXES={'.py','.yaml','.yml','.json','.md','.toml','.txt','.sh','.pdf'}
SKIP={'.git','.pixi','.venv','__pycache__','.pytest_cache','.cache','data','datasets','results','logs','runs','dist','build'}
ROOT_FILES=('.gitattributes','.gitignore','AGENTS.md','README.md','LICENSE','pixi.toml','pixi.lock','pytest.ini','requirements.txt')

def paths():
    import os
    found={n for n in ROOT_FILES if (ROOT/n).is_file()}
    for area in AREAS:
        for folder,dirs,files in os.walk(ROOT/area,followlinks=False):
            dirs[:]=[n for n in dirs if n not in SKIP and not (Path(folder)/n).is_symlink()]
            for n in files:
                p=Path(folder)/n
                if p.suffix.lower() in SUFFIXES:
                    if p.is_symlink():raise ValueError('No source symlink: '+str(p))
                    found.add(p.relative_to(ROOT).as_posix())
    found.add('conf/runtime/source_inventory_v7.json')
    return sorted(found)

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    target=ROOT/'conf/runtime/source_inventory_v7.json'
    value=dict(schema='hope_cad_portable_source_v1',files=paths(),scope='Explicit self-contained source/config/spec/test/papers; no environment/data/artifacts')
    if a.write:target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    elif json.loads(target.read_text(encoding='utf-8'))!=value:raise ValueError('Source inventory differs; review then use --write')
    print(json.dumps(dict(files=len(value['files']),written=a.write)))

if __name__=='__main__':main()
