"""Artifact-level causal invariants and saved witness endpoint recomputation."""
import json
import torch
from ..core import Predictor,State,objective_loss
from ..design import ROOT,sha
from .design import OUT,ORIGINAL,MATRIX
from .runner import tensor_digest


def verify_stage(stage,seeds,recompute=True):
    root=OUT/stage; rows={}
    for r,arms in MATRIX.items():
        for s in seeds:
            for a in arms:
                directory=root/f'{r}_{s}_{a}'
                row=json.loads((directory/'trace.json').read_text()); rows[(r,s,a)]=row
                assert row['cycles']==(160 if r=='R' else 128)
                assert row['peak_reserved']<=4.5*1024**3
                for d in row['causal_replay']:
                    assert d['current']==(d['clock']-1)//2
                    assert d['replay'] is None or d['replay']<d['current']
                for step in row['traces']:
                    if a not in ['extra_replay']:
                        for norm in step['block_norms'].values():
                            assert norm['correction']<=.25*norm['base']*(1+1e-5)+1e-10
                    for norms in step.get('random_norms',{}).values():
                        assert abs(norms['learned']-norms['random'])<=1e-10+1e-5*norms['learned']
                hashes=json.loads((directory/'hashes.json').read_text())
                assert sha(directory/'trace.json')==hashes['trace.json']['sha256']
                witnesses_meta=json.loads((root/'witnesses'/f'{r}_{s}.json').read_text())
                assert row['witness_tensor_sha256']==witnesses_meta['tensor_sha256']
    identities={}
    for r,arms in MATRIX.items():
        for s in seeds:
            E=rows[(r,s,'E')]
            assert all(rows[(r,s,a)]['causal_replay']==E['causal_replay'] for a in arms)
            for a in ['zero_memory']+(['deployed_ablation'] if 'deployed_ablation' in arms else []):
                assert [d['live_sha256'] for d in rows[(r,s,a)]['traces']]==[d['live_sha256'] for d in E['traces']]
            if r in ['D8','D32']:
                n=80 if r=='D8' else 128
                assert [d['live_sha256'] for d in rows[(r,s,'candidate')]['traces'][:n]]==[d['live_sha256'] for d in rows[(r,s,'post_onset_only')]['traces'][:n]]
                identities[f'{r}/{s}']=True
    for s in seeds:
        prefix=[d['live_sha256'] for d in rows[('D8',s,'candidate')]['traces'][:72]]
        for regime in ['D16','R','C','D32']:
            assert prefix==[d['live_sha256'] for d in rows[(regime,s,'candidate')]['traces'][:72]],('future exposure',regime,s)
    recomputed=[]
    if recompute:
        # CPU independent reload; no new model training/data/scoring rule.
        model=Predictor(channels=768,latent=64,hidden=128,side=28)
        for r in ['D8','D16','R','C','D32']:
            s=seeds[0]
            samples=torch.load(root/'witnesses'/f'{r}_{s}.pt',weights_only=False,map_location='cpu')
            assert tensor_digest(samples)==json.loads((root/'witnesses'/f'{r}_{s}.json').read_text())['tensor_sha256']
            for arm in ['E','candidate']:
                directory=root/f'{r}_{s}_{arm}'; row=rows[(r,s,arm)]
                endpoint={'D8':80,'D16':86,'R':128,'C':80,'D32':104}[r]
                for clock,phase,key in [(64,'A','acquisition'),(endpoint,'A','retained'),(128,'B','new_normal')]:
                    state=State.restore(torch.load(directory/f'endpoint_{clock}.pt',weights_only=False,map_location='cpu'))
                    with torch.no_grad():
                        loss=sum(float(objective_loss(model,state.phi,x,q)) for x,q in samples[phase])/16
                    assert abs(loss-row[key])<=1e-7+1e-5*abs(row[key]),(r,arm,key,loss,row[key])
                    recomputed.append(dict(regime=r,arm=arm,key=key,absolute_error=abs(loss-row[key])))
    # Original sealed files are checked; new subdirectory is intentionally outside that manifest.
    original_manifest=json.loads((ORIGINAL/'artifact_manifest.json').read_text())
    for n,record in original_manifest['files'].items():
        if n.startswith('exps/') or n.endswith('.json'):
            assert sha(ROOT/n)==record['sha256'],n
    return dict(passed=True,runs=len(rows),mask_identities=identities,witness_recomputed=recomputed,
                original_evidence_preserved=True)
