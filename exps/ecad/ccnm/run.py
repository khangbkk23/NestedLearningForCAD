"""One bounded CCNM-01 stream. Only normal writers/witnesses and synthetic probes."""
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
from exps.ecad.r1.distance import configure, Euclidean
from exps.ecad.r1.train_normal import (
    ROOT, FEATURES, CATEGORIES, SOURCE_SHA, CACHE_SHA, CP_SHA,
    ArrivalCache, writer_plan, file_sha, tensor_sha)
from .mechanism import CCNM, VERSION
from .test_mechanism import equal_state, collision_fixture

START = datetime.datetime(2026, 10, 18, 13, 28, 1, tzinfo=datetime.timezone.utc).timestamp()
MODES = ('raw', 'context', 'shuffled', 'routing')
R1B = ROOT / 'results/hope_cad/ecad/r1b/normal_fit_8011_20261018'


class SafetyStop(RuntimeError):
    pass


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, sort_keys=True, indent=2)


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def cpu_state(m):
    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.cpu()
        return {k: cpu(v) for k, v in x.items()} if isinstance(x, dict) else x
    return cpu(m.state_dict())


def witness_plan(source, writers):
    out = []
    for c in CATEGORIES:
        rows = [r for r in source['entries'] if r['category'] == c]
        rows.sort(key=lambda r: hashlib.sha256(('ecad-r1a-split-11021:' + r['raw_png_sha256']).encode()).hexdigest())
        out.extend(dict(r) for r in rows[190:206])
    assert len(out) == 48
    assert len({r['raw_png_sha256'] for r in out + writers}) == 398
    assert len({r['relative_path'] for r in out + writers}) == 398
    return out


def metrics(scores):
    a = np.asarray(scores)
    return {'patch_mean': float(a.mean()), 'patch_q95': float(np.quantile(a, .95, method='inverted_cdf')),
            'patch_q99': float(np.quantile(a, .99, method='inverted_cdf')),
            'mean_image_q99': float(np.quantile(a, .99, axis=1, method='inverted_cdf').mean()),
            'image_max_mean': float(a.max(axis=1).mean())}


def gpu_gate(device):
    m = CCNM(capacity=23, patches=11, candidates=5, device=device)
    g = torch.Generator().manual_seed(924)
    for t in range(5):
        m.update(torch.randn(11, 768, generator=g).to(device), event=t)
    x = torch.randn(11, 768, generator=g).to(device)
    c = m.context(x)
    oracle = ((x[:, None].double()-m.model.bank[None].double()).square().sum(2)
              + m.alpha.double().square()*(c.double()-m.contexts.double()).square().sum(1)).sqrt()
    for chunk in (7, 256):
        out = m.score(x, chunk=chunk)
        torch.testing.assert_close(out['scores'], oracle.min(1).values.float(), atol=2e-6, rtol=2e-6)
        assert torch.equal(out['ids'], m.model.serials[oracle.argmin(1)])
    b = CCNM.from_state_dict(cpu_state(m), device=device)
    assert equal_state(m.state_dict(), b.state_dict())
    before = m.state_dict()
    for mode in MODES:
        assert equal_state(m.score(x, mode=mode), b.score(x, mode=mode))
    assert equal_state(before, m.state_dict())
    probe = torch.randn(11, 768, generator=g).to(device)
    for obj in (m, b):
        obj.update(probe, event=5)
    assert equal_state(m.state_dict(), b.state_dict())
    toy = collision_fixture(device)
    q = torch.tensor([[3., 0.], [3., 0.]], device=device)
    a = torch.tensor([1., 0.], device=device)
    assert toy.score(q, query_context=a)['scores'].tolist() == [3., 3.]
    assert toy.score(q, mode='shuffled', query_context=a)['scores'].tolist() == [0., 0.]
    return {'status': 'PASS', 'device': device, 'chunks': [7, 256], 'dimension': 768,
            'direct_fp64_oracle': True, 'exact_restore_and_continuation': True,
            'read_only_all_modes': True, 'known_and_shuffled_collision': True}


def synthetic(m):
    origins = m.model.serials // m.model.patches
    _, counts = torch.unique_consecutive(origins, return_counts=True)
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    contexts = m.contexts[starts]
    d = (contexts[:, None] - contexts[None]).square().sum(2)
    # Furthest context for up to16 evenly spaced live prototype anchors.
    anchors = torch.linspace(0, len(starts)-1, min(16, len(starts)), device=starts.device).long()
    records = []
    for a in anchors:
        b = d[a].argmax()
        ca = contexts[a]
        clean = m.model.bank[starts[a]:starts[a]+1]
        collision = m.model.bank[starts[b]:starts[b]+1]
        rec = {'anchor_serial': int(m.model.serials[starts[a]]),
               'collision_serial': int(m.model.serials[starts[b]]),
               'context_distance': float(d[a, b].sqrt())}
        for mode in MODES:
            normal = m.score(clean, mode=mode, query_context=ca)
            counterfactual = m.score(collision, mode=mode, query_context=ca)
            rec[mode] = {'clean': float(normal['scores'][0]),
                         'collision': float(counterfactual['scores'][0]),
                         'gap': float(counterfactual['scores'][0]-normal['scores'][0]),
                         'selected_id': int(counterfactual['ids'][0])}
        rec['wrong_context_score'] = float(m.score(collision, query_context=contexts[b])['scores'][0])
        records.append(rec)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    configure()
    torch.set_num_threads(2)
    def check():
        if time.time() >= START + 840:
            raise TimeoutError('CCNM900s global cap;60s reserved for reporting')
    def alarm(*_):
        raise TimeoutError('CCNM worker hard deadline')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(max(1, int(START + 840 - time.time())))
    elapsed_start = time.monotonic()
    reader = None
    m = None
    result = {'status': 'IN_PROGRESS', 'verdict': 'CCNM_INCONCLUSIVE', 'completed': 0,
              'boundaries': [], 'metrics': [], 'latencies': {k: [] for k in MODES},
              'synthetic': {}, 'checkpoints': [], 'update_seconds': [],
              'global_window_start_utc': '2026-10-18T13:28:01Z', 'global_cap_seconds': 900}
    try:
        check()
        write(args.out/'device_gate.json', gpu_gate(args.device))
        source = json.loads((FEATURES/'source_manifest.json').read_text())
        digest = hashlib.sha256(json.dumps({k: v for k, v in source.items() if k != 'manifest_digest'},
                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert digest == source['manifest_digest'] == SOURCE_SHA
        updates, _ = writer_plan(source)
        r1b_contract = json.loads((R1B/'fit_contract.json').read_text())
        assert updates == r1b_contract['updates']
        witnesses = witness_plan(source, updates)
        pins = json.loads((ROOT/'exps/ecad/r1/IMPLEMENTATION_PIN.json').read_text())
        for n, h in pins['files'].items():
            assert file_sha(ROOT/n) == h
        write(args.out/'contract.json', {'version': VERSION, 'writer_seed': 8011, 'updates': updates,
              'witnesses': witnesses, 'source_manifest_sha': SOURCE_SHA, 'encoder_sha': CP_SHA,
              'source_hashes': {str(f.relative_to(ROOT)): file_sha(f)
                  for f in sorted((ROOT/'exps/ecad/ccnm').glob('*.py'))},
              'core_pins': pins, 'torch': torch.__version__, 'numpy': np.__version__,
              'normal_guard_ratio': 1.25, 'routing_prototypes': 1, 'shuffle_seed': 52011,
              'calibration': 'first writer median patch-to-mean L2 / sqrt(2), frozen',
              'access': 'train/good writers and eligible held-out normals only; synthetic counterfactuals'})
        reader = ArrivalCache(source, updates)
        evaluator = ArrivalCache(source, witnesses)
        m = CCNM(device=args.device)
        with (args.out/'updates.jsonl').open('x') as log:
            for event in range(350):
                check()
                grid = reader.next(event).to(args.device)
                sync(args.device)
                t = time.perf_counter()
                stats = m.update(grid, event=event)
                sync(args.device)
                update_elapsed = time.perf_counter()-t
                result['update_seconds'].append(update_elapsed)
                result['completed'] = event+1
                log.write(json.dumps({'event': event, 'seconds': update_elapsed,
                           **{k: int(v) for k, v in stats.items()}})+'\n')
                log.flush()
                if (event+1) % 25 == 0:
                    print(json.dumps({'completed': event+1, 'elapsed_seconds': time.monotonic()-elapsed_start}), flush=True)
                if event+1 not in (100, 200, 300, 350):
                    continue
                boundary = event+1
                # Read only previously retained reference bank for exact allocator parity.
                expected = torch.load(R1B/f'contcore_{boundary:03}.pt', weights_only=True, map_location='cpu')
                assert torch.equal(expected['serials'], m.model.serials.cpu())
                assert torch.equal(expected['bank'], m.model.bank.cpu())
                del expected
                path = args.out/f'ccnm_{boundary:03}.pt'
                with path.open('xb') as f:
                    torch.save(cpu_state(m), f)
                restored = CCNM.from_state_dict(torch.load(path, weights_only=True), device=args.device)
                assert equal_state(m.state_dict(), restored.state_dict())
                result['checkpoints'].append({'boundary': boundary, 'sha256': file_sha(path),
                    'bytes': path.stat().st_size, 'bank_sha256': tensor_sha(m.model.bank),
                    'serial_sha256': tensor_sha(m.model.serials), 'ledger': m.ledger()})
                result['synthetic'][str(boundary)] = synthetic(m)
                seen = min(boundary//100, 3)
                for category in CATEGORIES[:seen]:
                    check()
                    rows = [r for r in witnesses if r['category'] == category]
                    # Reset evaluator's service cursor only; writer service remains causal.
                    # Witness scoring is permitted repeatedly; witnesses NEVER enter update.
                    witness_reader = ArrivalCache(source, rows)
                    witness_reader.checked = reader.checked.copy()
                    arrays = {mode: [] for mode in MODES}
                    ids = {mode: [] for mode in MODES}
                    before = m.state_dict()
                    for wi in range(16):
                        check()
                        query = witness_reader.next(wi).to(args.device)
                        for mode in MODES:
                            sync(args.device)
                            start = time.perf_counter()
                            values = m.score(query, mode=mode)
                            sync(args.device)
                            result['latencies'][mode].append(time.perf_counter()-start)
                            arrays[mode].append(values['scores'].cpu().numpy())
                            ids[mode].append(values['ids'].cpu().numpy())
                            if wi == 0:
                                assert equal_state(values, restored.score(query, mode=mode))
                        if wi == 0:
                            assert equal_state(m.score(query, mode='raw'), m.score(query, weight=0))
                            for chunk in (7, 256):
                                d = Euclidean(chunk).matrix(query[:19], m.model.bank[:31])
                                reference = (query[:19, None].double()-m.model.bank[None, :31].double()).square().sum(2).sqrt().float()
                                torch.testing.assert_close(d, reference, atol=2e-6, rtol=2e-6)
                    assert equal_state(before, m.state_dict())
                    arrays = {k: np.stack(v) for k, v in arrays.items()}
                    ids = {k: np.stack(v) for k, v in ids.items()}
                    payload = {f'{mode}_{kind}': v for kind, values in [('scores', arrays), ('ids', ids)] for mode, v in values.items()}
                    np.savez_compressed(args.out/f'witness_{boundary}_{category}.npz', **payload)
                    raw_metrics = metrics(arrays['raw'])
                    for mode in MODES:
                        rec = {'boundary': boundary, 'category': category, 'mode': mode, **metrics(arrays[mode]),
                            'changed_nn_fraction': float((ids[mode] != ids['raw']).mean()),
                            'origin_category_fraction': {c: float(np.mean(np.array([updates[int(s)//784]['category'] for s in ids[mode].flat]) == c)) for c in CATEGORIES}}
                        result['metrics'].append(rec)
                    context_metrics = metrics(arrays['context'])
                    result['boundaries'].append({'boundary': boundary, 'category': category,
                        'mean_ratio': context_metrics['patch_mean']/raw_metrics['patch_mean'],
                        'q99_ratio': context_metrics['patch_q99']/raw_metrics['patch_q99']})
                    write(args.out/f'progress_{boundary}_{category}.json', result)
                    if any(context_metrics[k] > 1.25*raw_metrics[k] for k in ('patch_mean', 'patch_q99')):
                        raise SafetyStop(f'normal inflation guard failed at{boundary}/{category}')
                # Continue from the restored real state, without replaying any writer.
                m = restored
        result['status'] = 'DONE'
        # Mechanism-specific advantage requires shuffle specificity AND improvement
        # over simpler routing, not merely positive additive distances.
        syn = result['synthetic']['350']
        gap = {k: float(np.mean([r[k]['gap'] for r in syn])) for k in MODES}
        result['final_synthetic_mean_gaps'] = gap
        result['verdict'] = ('CCNM_NO_MECHANISTIC_ADVANTAGE' if gap['routing'] >= gap['context']
                             else 'CCNM_INCONCLUSIVE')
    except (Exception, KeyboardInterrupt) as exc:
        result['status'] = 'STOPPED_SAFETY' if isinstance(exc, SafetyStop) else 'INTERRUPTED_OR_FAILED'
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        if m is not None:
            with (args.out/'partial_checkpoint.pt').open('xb') as f:
                torch.save(cpu_state(m), f)
        print(json.dumps(result['failure']), flush=True)
    finally:
        signal.alarm(0)
        sync(args.device)
        result['cache_hashes_verified'] = reader.checked if reader else {}
        result['worker_seconds'] = time.monotonic()-elapsed_start
        result['global_elapsed_seconds'] = time.time()-START
        result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if args.device == 'cuda':
            result['device'] = torch.cuda.get_device_name(0)
            result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['latency_summary'] = {k: {'calls': len(v), 'median_ms': float(np.median(v)*1000),
                    'p95_ms': float(np.quantile(v, .95)*1000)} for k, v in result['latencies'].items() if v}
        if m is not None:
            result['final_ledger'] = m.ledger()
            result['alpha'] = float(m.alpha)
        result['artifact_bytes_before_summary'] = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
        write(args.out/'summary.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'verdict', 'completed', 'worker_seconds', 'global_elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
