"""CPU-only repair of a fixture isolation error; no additional training."""
import argparse
import json
import shutil
import torch
from ..design import save,sha
from ..core import Observation
from .design import OUT,PACKAGE,MATRIX,source_hashes
from .prefix import path as prefix_path
from .data import SyntheticStream,Replay,partition


def snapshot():
    target=OUT/'source_snapshots'/'reusecheck'
    target.mkdir(parents=True,exist_ok=False)
    for p in PACKAGE.glob('*.py'): shutil.copyfile(p,target/p.name)
    save(target/'source_hashes.json',source_hashes())


def validate():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    failed=json.loads((OUT/'reusecheck.json').read_text())
    assert set(failed['completed_arms'])==set(MATRIX['D16'])
    assert failed['failure'] and 'restored_replay.sample' in failed['failure']
    checks={}
    for regime in MATRIX:
        # Reload each immutable checkpoint independently: Replay.restore's list
        # is owned by its deserialized object, not shared with another branch.
        saved=torch.load(prefix_path('reusecheck',1901,'candidate'),weights_only=False,map_location='cpu')
        fresh=SyntheticStream(1901,regime); replay=Replay(1901)
        restored=SyntheticStream.restore(saved['stream']); restored.regime=regime
        restored_replay=Replay.restore(saved['replay'])
        for event in range(32):
            x=fresh.arrive(event)
            for local in [0,1]: replay.sample(event)
            replay.insert(Observation(event,x,partition(event,1,784)[1]))
        assert torch.equal(replay.rng.get_state(),restored_replay.rng.get_state())
        for event in range(32,80 if regime=='R' else 64):
            a=fresh.arrive(event); b=restored.arrive(event); assert torch.equal(a,b)
            for local in [0,1]:
                x=replay.sample(event); y=restored_replay.sample(event)
                assert x.identity==y.identity and torch.equal(x.features,y.features)
            replay.insert(Observation(event,a,partition(event,1,784)[1]))
            restored_replay.insert(Observation(event,b,partition(event,1,784)[1]))
        checks[regime]=True
    dry=[json.loads(p.read_text()) for p in (OUT/'dryrun').glob('*/trace.json')]
    savings=failed['conservative_savings_per_arm']
    projected=sum(r['seconds']-(savings[r['arm']] if r['regime']!='D8' else 0.) for r in dry)
    # Exact tensor shapes/pickle encoding: charge ten complete matrices, shared
    # immutable prefix artifacts once per seed, plus1GiB metadata allowance.
    shared_names={'endpoint_64.pt'}|{f'window_{i:02d}.pt' for i in range(4)}
    bytes_per_seed=0
    for r in dry:
        directory=OUT/'dryrun'/f"{r['regime']}_{r['seed']}_{r['arm']}"
        for p in directory.iterdir():
            if p.is_file() and not (r['regime']!='D8' and p.name in shared_names): bytes_per_seed+=p.stat().st_size
    bytes_per_seed+=sum(p.stat().st_size for p in (OUT/'reusecheck'/'prefixes').glob('*.pt'))
    result=dict(passed=True,previous_failure_sha256=sha(OUT/'reusecheck.json'),
        correction='fixture checkpoint reloaded separately per regime; actual training already loaded separately',
        completed_arms=failed['completed_arms'],generator_checks=checks,
        conservative_savings_per_arm=savings,projected_seconds_per_seed=projected,
        projected_storage_bytes=10*bytes_per_seed+1024**3,bytes_per_seed=bytes_per_seed,
        storage_convention='10 matrices, hardlinked prefix tensors once,1GiB metadata allowance',G3_untouched=True)
    save(OUT/'reusecheck_validation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['snapshot','validate']);a=p.parse_args()
    {'snapshot':snapshot,'validate':validate}[a.stage]()
