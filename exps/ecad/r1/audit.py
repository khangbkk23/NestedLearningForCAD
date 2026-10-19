"""Metadata/source-only audit and explicitly synthetic saturated state ledger.

Never imports a data reader, loads a feature cache, or traverses a test split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import torch
from .controls import MemoryArm, ARMS, byte_capacities, charged_bytes
from .distance import configure

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def provenance():
    r=ROOT/'agents/reports/task_ecad_r0_evidence_architecture_decision.md'
    tail=r.read_text().split('## Appendix A',1)[1]
    inventory=max((json.loads(s) for s in re.findall(r'```json\s*(.*?)```',tail,re.S)),key=len)
    bad=[n for n,h in inventory.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    if bad:raise RuntimeError(f'historical preservation failure: {bad}')
    ref=Path(__file__).parent/'reference'
    pins=json.loads((ref/'SOURCE_PIN.json').read_text())
    for n,v in pins['files'].items():
        b=(ref/n).read_bytes()
        assert hashlib.sha256(b).hexdigest()==v['sha256']
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==v['git_blob_sha1']
    feature=ROOT/'results/hope_cad/frozen_features_v1'
    src=json.loads((feature/'source_manifest.json').read_text())
    assert all(Path(r['relative_path']).parts[1:3]==('train','good') for r in src['entries'])
    canonical=hashlib.sha256(json.dumps({k:v for k,v in src.items() if k!='manifest_digest'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert canonical==src['manifest_digest']=='060579457d344992eef159b1d4f1a86c34dd9283b97b70d3c3a386006b0f4dcc'
    counts={c:len({r['raw_png_sha256'] for r in src['entries'] if r['category']==c}) for c in ('bottle','carpet','hazelnut')}
    # Identity integrity only: byte-hash exactly the already admitted train/good
    # source paths; no image decoding, feature extraction or split discovery.
    data=(ROOT/'data/mvtec').resolve()
    raw_checked=0;raw_missing=0;raw_mismatched=0
    for row in src['entries']:
        path=data/row['relative_path']
        if not path.is_file():raw_missing+=1;continue
        if not path.resolve().is_relative_to(data) or path.resolve().parts[-3:-1]!=('train','good'):
            raise RuntimeError('normal source resolves outside allowed train/good')
        raw_checked+=1;raw_mismatched+=raw_sha(path)!=row['raw_png_sha256']
    contract=json.loads((feature/'extractor_contract.json').read_text())
    prov=json.loads((feature/'checkpoint_provenance.json').read_text())
    cp=ROOT/'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth'
    return {'historical_checked':len(inventory),'historical_mismatches':bad,'upstream':pins,
            'metadata_sha256':{n:sha(feature/n) for n in ('source_manifest.json','extractor_contract.json','checkpoint_provenance.json','cache_integrity.json','independent_verification.json')},
            'normal_unique_content_by_category':counts,'normal_global_unique':len({r['raw_png_sha256'] for r in src['entries']}),
            '206_requirement_met_in_manifest':all(n>=206 for n in counts.values()),
            'raw_normal_files_checked':raw_checked,'raw_normal_files_missing':raw_missing,
            'raw_normal_hash_mismatches':raw_mismatched,'raw_images_decoded':False,
            'checkpoint_present':cp.is_file(),'checkpoint_expected':contract['checkpoint_sha256'],
            'historical_checkpoint':prov['historical_checkpoint_sha256'],
            'historical_byte_identity_established':False,
            'clean_cache_files':{c:{'present':(feature/f'clean/{c}_train.pt').is_file(),'bytes':(feature/f'clean/{c}_train.pt').stat().st_size} for c in counts},
            'feature_payloads_loaded':False,'current_cache_bytes_rehashed':False}


def materialize(arm):
    """Ledger fixture only: not a timed update or a causal-fill claim."""
    g=torch.Generator().manual_seed(6211)
    k,d=arm.capacity,arm.dim
    if arm.name=='contcore':
        arm.model.bank=torch.randn(k,d,generator=g)
        # Valid serial pattern: at most 78 retained arrivals from each image.
        arm.model.serials=torch.tensor([i//78*784+i%78 for i in range(k)])
        arm.model.next_event=(k+77)//78;arm.event=arm.model.next_event
    elif arm.name=='reservoir':
        arm.rows=torch.randn(k,d,generator=g);arm.serials=torch.arange(k);arm.n=k;arm.event=(k+783)//784
    elif arm.name=='p0':
        m=arm.model;m.bank=torch.randn(k,d,generator=g);m.ids=torch.arange(k);m.n=k
        m.witnesses=torch.randn(128,d,generator=g);m.witness_ids=torch.arange(128)
        m.wn=128;m.seen=m.wseen=k;m.events=(k+783)//784;arm.event=m.events
    else:
        for b in arm.banks:
            b.features=torch.randn(b.config.budget,d,generator=g)
            b._pair_matrix=torch.zeros(b.count,b.count,dtype=torch.float32)
            b._pair_dirty=False
        arm.event=(k+783)//784


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    configure();torch.set_num_threads(2)
    result={'provenance':provenance(),'ledger_kind':'materialized synthetic storage fixture, not runtime or geometry measurement','ledgers':[]}
    result['equal_byte_ceiling']=charged_bytes('contcore',2500)
    result['equal_byte_capacities']=byte_capacities()
    for ledger in ['equal_descriptor','equal_byte']:
        for name in ARMS:
            k=2500 if ledger=='equal_descriptor' else result['equal_byte_capacities'][name]
            arm=MemoryArm(name,capacity=k);materialize(arm);entry=arm.ledger()
            assert entry['charged_online_bytes']==charged_bytes(name,k),(name,entry)
            entry['comparison']=ledger;result['ledgers'].append(entry);del arm
    result['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    with args.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps({'provenance':{k:v for k,v in result['provenance'].items() if k not in ('upstream','metadata_sha256')},'ledgers':[{k:v for k,v in x.items() if k!='storages'} for x in result['ledgers']],'process_peak_rss_bytes':result['process_peak_rss_bytes']},indent=2))


if __name__=='__main__':main()
