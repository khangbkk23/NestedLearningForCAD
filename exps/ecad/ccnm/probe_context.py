"""CCNM-02 frozen-checkpoint probes. No learner updates or anomaly-data reader."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from exps.ecad.r1.distance import Euclidean, configure
from exps.ecad.r1.train_normal import ROOT, FEATURES, SOURCE_SHA, CATEGORIES, ArrivalCache, file_sha, tensor_sha
from .mechanism import CCNM
from .test_mechanism import equal_state

PRIOR = ROOT/'results/hope_cad/ecad/ccnm/ccnm_01_8011_20261018'
SEEDS = tuple(range(52011, 52019))
MODES = ('raw', 'context', 'routing') + tuple(f'shuffle_{s}' for s in SEEDS) + ('clean_context_reference',)
KINDS = ('retained_foreign', 'heldout_foreign', 'heldout_same')


def write(path, obj):
    # Materialize before opening: NumPy diagnostic scalars must not leave a
    # truncated summary after a complete successful scoring run.
    def scalar(value):
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    data = json.dumps(obj, indent=2, sort_keys=True, default=scalar)
    with path.open('x') as f:
        f.write(data)


def region_indices(side, grid_side=28, device='cpu'):
    if not 0 < side <= grid_side:
        raise ValueError('invalid patch region')
    start = (grid_side-side)//2
    return torch.tensor([(start+r)*grid_side+start+c for r in range(side) for c in range(side)],
                        device=device, dtype=torch.int64)


def replace_region(recipient, donor, region, *, cycle=False):
    if recipient.dtype != torch.float32 or donor.dtype != torch.float32 or recipient.shape[1] != donor.shape[1]:
        raise ValueError('matching FP32 feature dimensions required')
    if len(donor) == 0 or recipient.device != donor.device:
        raise ValueError('nonempty donor on same device required')
    altered = recipient.clone()
    altered[region] = donor[torch.arange(len(region), device=region.device) % len(donor)] if cycle else donor[region]
    return altered


def shuffled_contexts(m, seed):
    _, inv, counts = torch.unique_consecutive(m.model.serials//m.model.patches,
                                             return_inverse=True, return_counts=True)
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    perm = torch.randperm(len(starts), generator=torch.Generator().manual_seed(seed)).to(starts.device)
    return m.contexts[starts][perm][inv]


class SharedReadout:
    """Evaluation workspace ONLY: amortize identical geometry over frozen controls."""
    def __init__(self, m, seeds=SEEDS):
        self.m = m
        self.shuffled = {s: shuffled_contexts(m, s) for s in seeds}
        self.modes = ('raw', 'context', 'routing') + tuple(f'shuffle_{s}' for s in seeds) + ('clean_context_reference',)

    @torch.no_grad()
    def score(self, grid, clean_context=None, chunk=256):
        m = self.m
        c = m.context(grid)
        clean_context = c if clean_context is None else clean_context
        d = Euclidean(chunk).matrix(grid, m.model.bank)
        scores, ids = [], []
        for mode in self.modes:
            if mode == 'raw':
                distances = d
            else:
                ctx = self.shuffled[int(mode.split('_')[1])] if mode.startswith('shuffle_') else m.contexts
                qc = clean_context if mode == 'clean_context_reference' else c
                penalty = (ctx-qc).square().sum(1)*m.alpha.square()
                if mode == 'routing':
                    # The frozen scorer computes GEMM on a selected sub-bank.
                    # Masking a full-bank GEMM can differ by one CUDA FP32 ULP.
                    # Preserve the original numerical execution for this control.
                    routed = m.score(grid,mode='routing',chunk=chunk)
                    scores.append(routed['scores'])
                    ids.append(routed['ids'])
                    continue
                else:
                    distances = (d.square()+penalty[None]).sqrt()
            values, ix = distances.min(1)
            if not bool(torch.isfinite(values).all()):
                raise ValueError('nonfinite score')
            scores.append(values)
            ids.append(m.model.serials[ix])
        return torch.stack(scores), torch.stack(ids), c

    def workspace_bytes(self):
        return sum(t.untyped_storage().nbytes() for t in self.shuffled.values())


def contrasts(clean_scores, altered_scores, region):
    """Paired within-recipient endpoint; no perfect self-match clean assumption."""
    return (altered_scores[:, region]-clean_scores[:, region]).mean(1)


def validate_shared(device='cpu'):
    m = CCNM(capacity=23, patches=11, candidates=5, device=device)
    g = torch.Generator().manual_seed(20301)
    for t in range(5):
        m.update(torch.randn(11, 768, generator=g).to(device), event=t)
    x = torch.randn(11, 768, generator=g).to(device)
    before = m.state_dict()
    reader = SharedReadout(m)
    s, ids, c = reader.score(x)
    for name, original in [('raw', 'raw'), ('context', 'context'), ('routing', 'routing'), ('shuffle_52011', 'shuffled')]:
        actual = m.score(x, mode=original)
        ix = reader.modes.index(name)
        torch.testing.assert_close(s[ix], actual['scores'], rtol=0, atol=0)
        assert torch.equal(ids[ix], actual['ids'])
    for chunk in (7, 256):
        other = reader.score(x, chunk=chunk)
        torch.testing.assert_close(s, other[0], atol=2e-6, rtol=2e-6)
        assert torch.equal(ids, other[1])
    reference = ((x[:, None].double()-m.model.bank[None].double()).square().sum(2)
        + m.alpha.double().square()*(c.double()-m.contexts.double()).square().sum(1)).sqrt().min(1).values.float()
    torch.testing.assert_close(s[1], reference, atol=2e-6, rtol=2e-6)
    assert equal_state(before, m.state_dict())
    assert equal_state(m.score(x, weight=0), m.score(x, mode='raw'))
    return {'status': 'PASS', 'device': device, 'exact_frozen_scorer_parity': True,
            'chunks': [7, 256], 'direct_fp64_oracle': True, 'read_only': True}


def aggregate(records, clean):
    groups = []
    for boundary in (300, 350):
        for category in CATEGORIES:
            for kind in KINDS:
                for side in (4, 8):
                    rows = [r for r in records if (r['boundary'],r['category'],r['kind'],r['side']) == (boundary,category,kind,side)]
                    if not rows:
                        continue
                    group = {'boundary': boundary, 'category': category, 'kind': kind, 'side': side, 'n': len(rows),
                             'context_drift_mean': float(np.mean([r['context_drift'] for r in rows])),
                             'modes': {}}
                    raw = np.array([r['contrast']['raw'] for r in rows])
                    for mode in MODES:
                        v = np.array([r['contrast'][mode] for r in rows])
                        group['modes'][mode] = {'mean_contrast': float(v.mean()), 'median_contrast': float(np.median(v)),
                            'mean_advantage_over_raw': float((v-raw).mean()), 'positive_contrast_fraction': float((v>0).mean()),
                            'background_mean_drift': float(np.mean([r['background_drift'][mode] for r in rows])),
                            'image_max_delta_mean': float(np.mean([r['image_max_delta'][mode] for r in rows]))}
                    collapsed = raw < -1e-6
                    group['raw_collapses'] = int(collapsed.sum())
                    group['context_rescues'] = int(sum(r['contrast']['context'] >= 0 and r['contrast']['raw'] < -1e-6 for r in rows))
                    groups.append(group)
    progression = []
    for side in (4, 8):
        selected = {k: [r for r in records if r['side']==side and r['kind']==k] for k in KINDS}
        if any(len(v)!=96 for v in selected.values()):
            continue
        def mean_delta(kind, mode):
            return float(np.mean([r['contrast'][mode] for r in selected[kind]]))
        foreign = mean_delta('heldout_foreign','context')
        advantage = foreign-mean_delta('heldout_foreign','raw')
        shuffle_advantages = [mean_delta('heldout_foreign',f'shuffle_{seed}')-mean_delta('heldout_foreign','raw') for seed in SEEDS]
        same_advantage = mean_delta('heldout_same','context')-mean_delta('heldout_same','raw')
        collapsed = [r for r in selected['retained_foreign'] if r['contrast']['raw'] < -1e-6]
        rescue = sum(r['contrast']['context']>=0 for r in collapsed)/len(collapsed) if collapsed else 0.
        checks = {'heldout_foreign_positive_contrast': foreign>0,
                  'heldout_foreign_positive_advantage': advantage>0,
                  'better_than_average_shuffle': advantage>np.mean(shuffle_advantages),
                  'foreign_more_than_same_control': advantage>same_advantage,
                  'retained_foreign_positive_contrast': mean_delta('retained_foreign','context')>0,
                  'retained_foreign_majority_rescued': rescue>.5}
        progression.append({'side': side, 'heldout_foreign_context_contrast': foreign,
            'heldout_foreign_raw_contrast': mean_delta('heldout_foreign','raw'), 'advantage': advantage,
            'shuffle_advantages': shuffle_advantages, 'same_advantage': same_advantage,
            'foreign_minus_same_advantage': advantage-same_advantage,
            'retained_foreign_context_contrast': mean_delta('retained_foreign','context'),
            'retained_raw_collapses': len(collapsed), 'retained_rescue_fraction': rescue, 'checks': checks})
    verdict = ('CONTEXT_EFFECT_SURVIVES_INFERRED_CONTEXT' if len(progression)==2 and all(all(r['checks'].values()) for r in progression)
               else 'NO_ROBUST_SYNTHETIC_DISCRIMINATION' if len(records)==576 else 'INCONCLUSIVE')
    return {'verdict': verdict, 'groups': groups, 'progression': progression, 'clean_safety': clean}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    p.add_argument('--window-start-epoch', type=float, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    configure()
    torch.set_num_threads(2)
    wall = time.monotonic()
    def check():
        if time.time() >= args.window_start_epoch+840:
            raise TimeoutError('900s global cap with60s reporting reserve')
    def timeout(*_):
        raise TimeoutError('worker hard deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1,int(args.window_start_epoch+840-time.time())))
    records, clean_records, loaded_hashes = [], [], {}
    diagnostics = {'status': 'IN_PROGRESS', 'window_start_epoch': args.window_start_epoch,
                   'clean_grids': 0, 'counterfactual_grids': 0, 'modes': MODES, 'seeds': SEEDS}
    rc = 1
    try:
        check()
        write(args.out/'device_gate.json', validate_shared(args.device))
        prior = json.loads((PRIOR/'summary.json').read_text())
        contract = json.loads((PRIOR/'contract.json').read_text())
        for name, digest in contract['source_hashes'].items():
            assert file_sha(ROOT/name) == digest
        source = json.loads((FEATURES/'source_manifest.json').read_text())
        digest = hashlib.sha256(json.dumps({k:v for k,v in source.items() if k!='manifest_digest'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert source['manifest_digest'] == digest == SOURCE_SHA
        witnesses = contract['witnesses']
        assert len(witnesses)==48 and len({r['raw_png_sha256'] for r in witnesses+contract['updates']})==398
        read = ArrivalCache(source,witnesses)
        grids = {}
        for i,row in enumerate(witnesses):
            check()
            grids[row['relative_path']] = read.next(i)
        loaded_hashes = read.checked
        by_category = {c:[r for r in witnesses if r['category']==c] for c in CATEGORIES}
        write(args.out/'contract.json', {'protocol': 'CCNM-02 inferred-context local feature replacement v1',
            'prior_contract_sha256': file_sha(PRIOR/'contract.json'), 'witnesses': witnesses,
            'source_manifest_sha': SOURCE_SHA, 'seeds': SEEDS, 'modes': MODES, 'sides': [4,8],
            'source_hashes': {str(f.relative_to(ROOT)):file_sha(f) for f in sorted((ROOT/'exps/ecad/ccnm').glob('*.py'))},
            'normal_only': True, 'learner_updates': 0, 'parameter_changes': 0,
            'context_override': 'clean-context reference only; production always inferred',
            'window_start_epoch':args.window_start_epoch, 'torch':torch.__version__, 'numpy':np.__version__})
        with (args.out/'records.jsonl').open('x') as log:
            for boundary in (300,350):
                check()
                cp = next(r for r in prior['checkpoints'] if r['boundary']==boundary)
                checkpoint = PRIOR/f'ccnm_{boundary:03}.pt'
                assert file_sha(checkpoint)==cp['sha256']
                m = CCNM.from_state_dict(torch.load(checkpoint,weights_only=True),device=args.device)
                before = m.state_dict()
                reader = SharedReadout(m)
                diagnostics['detector_ledger'] = m.ledger()
                diagnostics['evaluator_shuffle_cache_bytes'] = reader.workspace_bytes()
                diagnostics['evaluator_normal_grid_bytes'] = sum(t.untyped_storage().nbytes() for t in grids.values())
                for category in CATEGORIES:
                    category_clean = []
                    previous = np.load(PRIOR/f'witness_{boundary}_{category}.npz')
                    for wi,row in enumerate(by_category[category]):
                        check()
                        grid = grids[row['relative_path']].to(args.device)
                        scores0,ids0,c0 = reader.score(grid)
                        diagnostics['clean_grids'] += 1
                        for mode,oldmode in [('raw','raw'),('context','context'),('routing','routing'),('shuffle_52011','shuffled')]:
                            ix = MODES.index(mode)
                            torch.testing.assert_close(scores0[ix].cpu(),torch.from_numpy(previous[f'{oldmode}_scores'][wi]),rtol=0,atol=0)
                            assert np.array_equal(ids0[ix].cpu().numpy(),previous[f'{oldmode}_ids'][wi])
                        category_clean.append(scores0[:3].cpu().numpy())
                        far = (m.contexts-c0).square().sum(1).argmax()
                        origin = m.model.serials[far]//784
                        donor_bank = m.model.bank[m.model.serials//784==origin]
                        foreign_cat = CATEGORIES[(CATEGORIES.index(category)+1)%3]
                        donors = {'heldout_foreign':by_category[foreign_cat][(wi+1)%16],
                                  'heldout_same':by_category[category][(wi+1)%16]}
                        all_scores, all_ids, contexts = [scores0.cpu().numpy()],[ids0.cpu().numpy()],[c0.cpu().numpy()]
                        for side in (4,8):
                            region = region_indices(side,device=args.device)
                            background = torch.ones(784,dtype=torch.bool,device=args.device)
                            background[region] = False
                            for kind in KINDS:
                                check()
                                donor = donor_bank if kind=='retained_foreign' else grids[donors[kind]['relative_path']].to(args.device)
                                altered = replace_region(grid,donor,region,cycle=kind=='retained_foreign')
                                sc,ids,c = reader.score(altered,clean_context=c0)
                                if kind=='retained_foreign':
                                    assert bool((sc[0,region]==0).all())
                                torch.testing.assert_close(sc[0,background],scores0[0,background],rtol=0,atol=0)
                                delta = contrasts(scores0,sc,region)
                                if args.device=='cuda':
                                    torch.cuda.synchronize()
                                rec = {'boundary':boundary,'category':category,'witness_index':wi,
                                    'recipient':row['relative_path'],'kind':kind,'side':side,
                                    'donor':donors[kind]['relative_path'] if kind!='retained_foreign' else f'live_origin_{int(origin)}',
                                    'retained_donor_rows':len(donor_bank) if kind=='retained_foreign' else None,
                                    'context_drift':float((c-c0).norm()),
                                    'contrast':dict(zip(MODES,map(float,delta.cpu()))),
                                    'background_drift':dict(zip(MODES,map(float,(sc[:,background]-scores0[:,background]).mean(1).cpu()))),
                                    'image_max_delta':dict(zip(MODES,map(float,(sc.max(1).values-scores0.max(1).values).cpu()))),
                                    'region_nn_change_fraction':dict(zip(MODES,map(float,(ids[:,region]!=ids0[:,region]).float().mean(1).cpu())))}
                                records.append(rec)
                                log.write(json.dumps(rec)+'\n')
                                log.flush()
                                all_scores.append(sc.cpu().numpy());all_ids.append(ids.cpu().numpy());contexts.append(c.cpu().numpy())
                                diagnostics['counterfactual_grids'] += 1
                        np.savez_compressed(args.out/f'probe_{boundary}_{category}_{wi:02}.npz',
                            scores=np.stack(all_scores),ids=np.stack(all_ids),contexts=np.stack(contexts))
                        if (wi+1)%8==0:
                            print(json.dumps({'boundary':boundary,'category':category,'witnesses':wi+1,
                                'counterfactual_grids':len(records),'seconds':time.monotonic()-wall}),flush=True)
                    previous.close()
                    clean = np.stack(category_clean)
                    raw,ctx = clean[:,0],clean[:,1]
                    q = lambda x:float(np.quantile(x,.99,method='inverted_cdf'))
                    cr = {'boundary':boundary,'category':category,'raw_mean':float(raw.mean()),'context_mean':float(ctx.mean()),
                        'raw_q99':q(raw),'context_q99':q(ctx),'mean_ratio':float(ctx.mean()/raw.mean()),'q99_ratio':q(ctx)/q(raw)}
                    clean_records.append(cr)
                    assert cr['mean_ratio']<=1.25 and cr['q99_ratio']<=1.25,'normal safety guard failed'
                    assert equal_state(before,m.state_dict()),'score-time mutation'
                    write(args.out/f'progress_{boundary}_{category}.json', {'completed':len(records),'clean_safety':clean_records})
                assert file_sha(checkpoint)==cp['sha256']
        diagnostics['status']='DONE'
        rc = 0
    except (Exception,KeyboardInterrupt) as exc:
        diagnostics['status']='STOPPED'
        diagnostics['failure']={'type':type(exc).__name__,'message':str(exc)}
        print(json.dumps(diagnostics['failure']),flush=True)
    finally:
        signal.alarm(0)
        if args.device=='cuda':
            torch.cuda.synchronize()
            diagnostics['device']=torch.cuda.get_device_name(0)
            diagnostics['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            diagnostics['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        diagnostics.update(aggregate(records,clean_records))
        diagnostics['worker_seconds']=time.monotonic()-wall
        diagnostics['global_elapsed_seconds']=time.time()-args.window_start_epoch
        diagnostics['cache_hashes_verified']=loaded_hashes
        diagnostics['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        diagnostics['artifact_bytes_before_summary']=sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
        write(args.out/'summary.json',diagnostics)
        print(json.dumps({k:diagnostics[k] for k in ('status','verdict','clean_grids','counterfactual_grids','worker_seconds','global_elapsed_seconds')}),flush=True)
    raise SystemExit(rc)


if __name__=='__main__':
    main()
