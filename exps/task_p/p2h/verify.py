"""Read-only historical checks and append-only P2h artifact integrity ledger."""
import argparse
import hashlib
import json
from pathlib import Path
from .design import ROOT,OUT,save,sha


def stream_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    lock=json.loads((OUT/'design_lock.json').read_text())
    assert sha(OUT/'protocol_design.json')==lock['protocol_sha256']
    assert sha(OUT/'preservation.json')==lock['preservation_sha256']
    assert sha(ROOT/'exps/task_p/p2h/design.py')==lock['design_source_sha256']
    protocol=json.loads((OUT/'protocol_design.json').read_text())
    assert sha(ROOT/'exps/task_p/p2h/PROTOCOL.md')==protocol['mathematics_protocol_sha256']
    preserved=json.loads((OUT/'preservation.json').read_text())['files']
    bad=[n for n,r in preserved.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=r['sha256']]
    assert not bad,('historical changes',bad)
    for n,h in protocol['canonical_source_sha256'].items(): assert sha(ROOT/n)==h
    assert json.loads((OUT/'G1.json').read_text())['passed']
    contract_checks=json.loads((OUT/'G1_contract_checks.json').read_text())
    assert contract_checks['passed']
    analysis=json.loads((OUT/'engineering_analysis.json').read_text())
    assert analysis['G3_untouched']
    if analysis['verdict'] in ['RESOURCE_INFEASIBLE','INCONCLUSIVE_UNDERPOWERED','IMPLEMENTATION_INVALID']:
        assert not (OUT/'execution_lock.json').exists()
        assert not list(OUT.glob('g3*'))
    runs=[]; replay_draws=0; traces={}
    for p in sorted((OUT/'engineering').glob('*/trace.json')):
        row=json.loads(p.read_text()); assert row['seed'] in range(1901,1911)
        traces[(row['regime'],row['seed'],row['arm'])]=row
        assert row['G3_untouched']
        for draw in row['causal_replay']:
            assert draw['current']==(draw['clock']-1)//2
            assert draw['replay'] is None or 0<=draw['replay']<draw['current']
            replay_draws+=1
        assert all(d['memory_norm']==0 for d in row['decompositions'] if d['kind'].startswith('zero_'))
        assert all(d['memory_norm']>=0 for d in row['decompositions'] if d['kind'].startswith('actual_'))
        for step in row['traces']:
            assert step['lambda_value'] in [0.,.25,.5,1.]
            rho=1. if row['arm']=='relaxed_cap' else .25
            for n,norm in step['block_norms'].items():
                assert norm['correction']<=rho*norm['base']*(1+1e-6)+1e-12
        runs.append(dict(seed=row['seed'],regime=row['regime'],arm=row['arm'],cycles=row['completed_cycles']))
    for seed in range(1901,1911):
        a=traces.get(('D8',seed,'candidate')); b=traces.get(('D16',seed,'candidate'))
        if a and b:
            assert a['traces'][:72]==b['traces'][:72],('future-regime-dependent prefix',seed)
    for n in ['agents/tasks/current_task_p2h.md','agents/reports/task_p_p2h_mechanism_implementation.md']:
        text=(ROOT/n).read_text()
        for field in ['Task:','Status:','Last agent:','Next agent:','Branch / HEAD:',
                      'Completed work:','Changed files:','Verification / Tests:',
                      'Unresolved issues:','Exact next action:']: assert field in text,(n,field)
        assert all(line==line.rstrip() for line in text.splitlines()),n
    if args.check:
        manifest=json.loads((OUT/'artifact_manifest.json').read_text())
        for n,record in manifest['files'].items():
            p=ROOT/n
            assert p.stat().st_size==record['bytes'] and stream_sha(p)==record['sha256'],n
        print(json.dumps(dict(passed=True,historical_files=len(preserved),replay_draws=replay_draws,
                             runs=len(runs),manifest_files=len(manifest['files']),G3_untouched=True)),flush=True)
        return
    assert not (OUT/'artifact_manifest.json').exists(),'immutable manifest already exists'
    paths=[p for p in OUT.rglob('*') if p.is_file()]
    paths+=list((ROOT/'exps/task_p/p2h').glob('*.py'))+[ROOT/'exps/task_p/p2h/PROTOCOL.md']
    files={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':stream_sha(p)} for p in sorted(paths)}
    save(OUT/'artifact_manifest.json',dict(schema='p2h_artifacts_v1',files=files))
    result=dict(passed=True,historical_files=len(preserved),canonical_source_pins=len(protocol['canonical_source_sha256']),
                replay_draws=replay_draws,runs=runs,manifest_files=len(files),
                artifact_manifest_sha256=sha(OUT/'artifact_manifest.json'),
                G3_untouched=True,verdict=analysis['verdict'],first_failed_gate=analysis['first_failed_gate'])
    save(OUT/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
