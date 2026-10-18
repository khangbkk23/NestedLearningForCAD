"""Preserve executed engineering source bytes before any later code amendment."""
from datetime import datetime,timezone
from .design import ROOT,OUT,sha,save


def main():
    names=['__init__.py','design.py','core.py','data.py','learner.py','statistics.py',
           'preflight.py','test_mechanics.py','check_contracts.py','diagnose_guard.py']
    records={}
    for name in names:
        source=ROOT/'exps/task_p/p2h'/name
        target=OUT/'source_snapshots/stage1b'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(source.read_bytes())
        records[name]=sha(source)
    save(OUT/'stage1b_source_snapshot.json',dict(utc=datetime.now(timezone.utc).isoformat(),
         scope='executed unchanged engineering sources; captured after G1b before any later implementation amendment',files=records))
    print('STAGE1B_SOURCE_SNAPSHOT',len(records),flush=True)


if __name__=='__main__': main()
