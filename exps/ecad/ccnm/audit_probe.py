"""Recover/audit saved CCNM-02 scores, without reopening any feature cache."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from exps.ecad.r1.train_normal import ROOT, file_sha
from .probe_context import PRIOR,MODES,SEEDS,aggregate,region_indices,contrasts,write


def recover_prefix(path):
    """Recover only complete top-level fields from the preserved truncated JSON."""
    text=path.read_text()
    decoder=json.JSONDecoder()
    pos=1
    out={}
    while pos<len(text):
        while pos<len(text) and text[pos] in ' \t\r\n,':pos+=1
        try:
            key,n=decoder.raw_decode(text,pos)
            pos=n
            while text[pos] in ' \t\r\n':pos+=1
            assert text[pos]==':'
            pos+=1
            while text[pos] in ' \t\r\n':pos+=1
            value,n=decoder.raw_decode(text,pos)
        except (ValueError,IndexError,AssertionError):
            break
        out[key]=value
        pos=n
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    contract=json.loads((args.run/'contract.json').read_text())
    records=[json.loads(line) for line in (args.run/'records.jsonl').read_text().splitlines()]
    assert len(records)==576
    source_matches={}
    for name,digest in contract['source_hashes'].items():
        current=file_sha(ROOT/name)
        if current!=digest:
            assert name in ('exps/ecad/ccnm/probe_context.py','exps/ecad/ccnm/test_probe_context.py')
            if name.endswith('/probe_context.py'):
                assert file_sha(args.run/'probe_context_executed.py')==digest
                source_matches[name]='exact executed source preserved; current source fixes JSON serialization only'
            else:
                source_matches[name]='added serialization regression test after completed execution'
        else:source_matches[name]='unchanged'
    cpids={}
    prior=json.loads((PRIOR/'summary.json').read_text())
    for boundary in (300,350):
        path=PRIOR/f'ccnm_{boundary:03}.pt'
        expected=next(c for c in prior['checkpoints'] if c['boundary']==boundary)
        assert file_sha(path)==expected['sha256']
        cpids[boundary]=set(torch.load(path,weights_only=True)['allocator']['serials'].tolist())
    audited=[]
    regional_nn=[]
    fresh_clean=[]
    clean_by_group={}
    for path in sorted(args.run.glob('probe_*.npz')):
        _,boundary,category,wi=path.stem.split('_')
        boundary,wi=int(boundary),int(wi)
        rows=[r for r in records if r['boundary']==boundary and r['category']==category and r['witness_index']==wi]
        assert len(rows)==6
        with np.load(path) as data:
            s,ids,c=data['scores'],data['ids'],data['contexts']
            assert s.shape==ids.shape==(7,12,784) and c.shape==(7,32)
            assert np.isfinite(s).all() and (s>=0).all() and np.isfinite(c).all()
            assert set(ids.flat)<=cpids[boundary]
            np.testing.assert_allclose(np.linalg.norm(c,axis=1),1,rtol=2e-6,atol=2e-6)
            np.testing.assert_array_equal(s[0,1],s[0,-1])
            with np.load(PRIOR/f'witness_{boundary}_{category}.npz') as old:
                for ix,mode in ((0,'raw'),(1,'context'),(2,'routing'),(3,'shuffled')):
                    np.testing.assert_array_equal(s[0,ix],old[f'{mode}_scores'][wi])
                    np.testing.assert_array_equal(ids[0,ix],old[f'{mode}_ids'][wi])
            clean_by_group.setdefault((boundary,category),[]).append(s[0,:3].copy())
            for i,record in enumerate(rows,1):
                r=region_indices(record['side']).numpy()
                bg=np.ones(784,dtype=bool);bg[r]=False
                np.testing.assert_array_equal(s[i,0,bg],s[0,0,bg])
                np.testing.assert_array_equal(s[i,-1,bg],s[0,-1,bg])
                if record['kind']=='retained_foreign':
                    np.testing.assert_array_equal(s[i,0,r],np.zeros(len(r),dtype=np.float32))
                delta=(s[i,:,r]-s[0,:,r]).mean(axis=0)
                # Advanced indexing places region before mode in the expression above.
                expected=np.array([record['contrast'][mode] for mode in MODES])
                np.testing.assert_allclose(delta,expected,rtol=2e-6,atol=2e-6)
                actualdrift=float(np.linalg.norm(c[i]-c[0]))
                assert abs(actualdrift-record['context_drift'])<2e-6
                regional_nn.append({'boundary':boundary,'category':category,'witness_index':wi,
                    'kind':record['kind'],'side':record['side'],
                    'context_vs_raw_changed_fraction':float(np.mean(ids[i,1,r]!=ids[i,0,r])),
                    'routing_vs_raw_changed_fraction':float(np.mean(ids[i,2,r]!=ids[i,0,r]))})
            audited.append({'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path)})
    assert len(audited)==96
    for (boundary,category),values in clean_by_group.items():
        s=np.stack(values);raw,ctx=s[:,0],s[:,1]
        q=lambda a:float(np.quantile(a,.99,method='inverted_cdf'))
        cr={'boundary':boundary,'category':category,'raw_mean':float(raw.mean()),'context_mean':float(ctx.mean()),
            'raw_q99':q(raw),'context_q99':q(ctx),'mean_ratio':float(ctx.mean()/raw.mean()),'q99_ratio':q(ctx)/q(raw)}
        assert cr['mean_ratio']<=1.25 and cr['q99_ratio']<=1.25
        fresh_clean.append(cr)
    summary=aggregate(records,fresh_clean)
    preserved=recover_prefix(args.run/'summary.json')
    # Worker completed all576 records but failed only on np.bool JSON encoding.
    # Keep original failed summary untouched; recover complete measured fields.
    for key in ('artifact_bytes_before_summary','cache_hashes_verified','cuda_peak_allocated_bytes',
                'cuda_peak_reserved_bytes','detector_ledger','device','evaluator_normal_grid_bytes',
                'evaluator_shuffle_cache_bytes','global_elapsed_seconds','process_peak_rss_bytes'):
        if key in preserved:summary[key]=preserved[key]
    summary.update({'status':'DONE_ARTIFACT_AUDIT_PASS','clean_grids':96,'counterfactual_grids':576,
        'regional_nn_diagnostics':regional_nn,
        'audited_npz_files':audited,'source_verification':source_matches,'audit_source_sha256':file_sha(__file__),
        'serialization_failure_preserved':str(args.run/'summary.json'),
        'scoring_worker_seconds':37.01171932200032,
        'scoring_worker_seconds_note':'last printed all576-probes progress; exact final worker field was not serialized',
        'current_artifact_bytes':sum(f.stat().st_size for f in args.run.iterdir() if f.is_file()),
        'audit_epoch':time.time()})
    write(args.out,summary)
    print(json.dumps({'status':summary['status'],'verdict':summary['verdict'],
        'audited_npz':len(audited),'counterfactuals':len(records),'progression':summary['progression']},
        indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


if __name__=='__main__':main()
