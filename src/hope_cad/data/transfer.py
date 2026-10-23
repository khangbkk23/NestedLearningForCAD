"""Pinned asset transport; independent of the ML environment and learned state."""
import hashlib,json,os,re,urllib.request
from pathlib import Path
from contextlib import contextmanager

def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@contextmanager
def writer_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
        yield
    finally:path.unlink(missing_ok=True)

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def checked_file(path,expected_sha256,size=None):
    path=Path(path)
    expected_sha256=str(expected_sha256).strip().lower()
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256):raise ValueError('Expected a SHA256 hex digest')
    if path.is_symlink() or not path.is_file():raise ValueError('Missing/symlinked asset: '+str(path))
    if size is not None and path.stat().st_size!=size:raise ValueError('Asset size differs: '+str(path))
    if sha256(path)!=expected_sha256:raise ValueError('Asset SHA256 differs: '+str(path))
    return dict(path=str(path.resolve()),sha256=expected_sha256,bytes=path.stat().st_size)

def download(url,destination,expected_sha256,size,*,opener=urllib.request.urlopen):
    """Resume a checked Range response; publish only after full size/SHA check."""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with writer_lock(destination.with_name(destination.name+'.download.lock')):
        if destination.exists():return checked_file(destination,expected_sha256,size)
        part=destination.with_name(destination.name+'.part')
        if part.is_symlink():raise ValueError('Symlinked partial asset')
        offset=part.stat().st_size if part.exists() else 0
        if offset>size:raise ValueError('Partial asset too large; preserve for inspection')
        if offset<size:
            req=urllib.request.Request(url,headers={'User-Agent':'hope-cad-asset-provisioner'})
            if offset:req.add_header('Range',f'bytes={offset}-')
            with opener(req,timeout=120) as response:
                if response.status==206:
                    value=response.headers.get('Content-Range','')
                    if not value.startswith(f'bytes {offset}-') or not value.endswith('/'+str(size)):raise ValueError('Range response does not match partial asset')
                    mode='ab'
                elif response.status==200:mode='wb'
                else:raise ValueError('Unexpected asset HTTP status')
                with part.open(mode) as out:
                    while block:=response.read(8*1024*1024):
                        if out.tell()+len(block)>size:raise ValueError('Response larger than pinned asset size')
                        out.write(block)
                    out.flush();os.fsync(out.fileno())
        checked_file(part,expected_sha256,size);part.replace(destination)
        return dict(**checked_file(destination,expected_sha256,size),source_url=url)
