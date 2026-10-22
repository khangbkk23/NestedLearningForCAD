"""Deliver each new bridge normal once; no pre-extraction of future image bytes."""
from pathlib import Path,PurePosixPath
import shutil,tarfile
from training.support_throughput_v5_1.contract import safe

class ArrivalProvider:
    def __init__(self,rows,provider,*,archive=None,assets=None):
        self.rows=list(rows);self.provider=provider;self.cursor=0;self.seen=[]
        self.assets=Path(assets) if assets else None
        self.tar=tarfile.open(archive) if archive else None
        self.members=iter(self.tar) if self.tar else None
        self.targets={r['relative_path'] for r in self.rows};self.index={};self.encountered=set()
        if len(self.targets)!=len(self.rows):raise ValueError('Duplicate registered arrival identity')
    def get(self,row):
        if self.cursor>=len(self.rows) or row!=self.rows[self.cursor] or row.get('split')!='train' or '/train/good/' not in row['relative_path']:raise ValueError('Out-of-order or forbidden bridge arrival')
        if self.tar:
            wanted=row['relative_path']
            if wanted not in self.index:
                for m in self.members:
                    p=PurePosixPath(m.name)
                    if p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name or m.issym() or m.islnk():raise ValueError('Unsafe dataset tar')
                    if not m.isfile():continue
                    matches=[n for n in self.targets if p.as_posix()==n or p.as_posix().endswith('/'+n)]
                    if not matches:continue
                    if len(matches)!=1 or matches[0] in self.encountered:raise ValueError('Ambiguous/duplicate registered member in tar')
                    name=matches[0];self.encountered.add(name);self.index[name]=m
                    if name==wanted:break
            if wanted not in self.index:raise ValueError('Dataset tar lacks current registered arrival')
            # Metadata for later permitted paths may have appeared earlier in the archive.
            # extractfile is called ONLY for this arrival, even for reverse-ordered compressed tar.
            member=self.index.pop(wanted);dest=safe(self.assets,wanted);dest.parent.mkdir(parents=True,exist_ok=True)
            with self.tar.extractfile(member) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
        value=self.provider.get(row);self.seen.append(row['relative_path']);self.cursor+=1
        return value
    def close(self):
        if self.tar:self.tar.close()
