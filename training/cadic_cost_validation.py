"""Bounded CADIC R/O validation; no anomaly evaluation or benchmark adoption."""
import argparse
import io
import json
import math
import time
from pathlib import Path

import torch

from training.runtime import atomic_json, atomic_torch_save
from training.support_throughput_v5_1.contract import ROOT, Evidence, contract, digest, sha, verify_parent, safe
from training.support_throughput_v5_1.replay import bounded_replay, assert_equal, state_digest
from training.support_benchmark_v5_1.contract import session_for
from training.support_benchmark_v5_1.data import NormalFeatures
from training.support_nn_write_v5_1.candidate import activation
from training.support_nn_write_v5_1.contract import verify_source as verify_base
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
from notebook_support.support_v5_1.common import configure_runtime, runtime_identity

PASSES = ('R1', 'O1', 'O2', 'R2')
SCHEMA = 'hope_cadic_cost_validation_v1'


def verify_delivery():
    verify_base()
    m = json.loads((ROOT/'cadic_cost_manifest.json').read_text())
    if m['schema'] != 'hope_cadic_cost_source_v1':
        raise ValueError('Wrong source schema')
    for n, h in m['files'].items():
        if sha(safe(ROOT, n)) != h:
            raise ValueError('Source drift: '+n)
    return digest(m['files'])


def summarize(values, nested):
    if set(values) != set(PASSES):
        raise ValueError('Four complete passes required')
    if len({v['final_state_digest'] for v in values.values()}) != 1 or len({v['trace_digest'] for v in values.values()}) != 1:
        raise ValueError('Exact CADIC trajectory parity failed')
    times = {n: v['pipeline_seconds'] for n, v in values.items()}
    if any(not math.isfinite(v) or v <= 0 for v in times.values()):
        raise ValueError('Invalid timing')
    for v in values.values():
        if v['status'] != 'complete' or v['start_cursor'] != 179 or v['end_cursor'] != 195 or v['completed_images'] != 16 or v['finalized'] or not v['global_rng_unchanged']:
            raise ValueError('Wrong/incomplete measured window')
    r = (times['R1']+times['R2'])/2
    o = (times['O1']+times['O2'])/2
    noise = max(abs(times['R1']-times['R2']), r*abs(times['O1']-times['O2'])/o)
    pairs = [times['O1']/times['R1'], times['O2']/times['R2']]
    rp = max(values[n]['gpu']['peak_allocated_bytes'] for n in ('R1', 'R2'))
    op = max(values[n]['gpu']['peak_allocated_bytes'] for n in ('O1', 'O2'))
    return dict(status='complete_pending_review', times=times, reference_mean_seconds=r,
        optimized_mean_seconds=o, speedup=r/o, time_reduction_fraction=1-o/r,
        paired_ratios=pairs, noise_envelope_seconds=noise,
        speed_gate=all(v < 1 for v in pairs) and r-o > noise,
        memory_gate=op <= rp+64*1024**2, peak_reference_allocated_bytes=rp,
        peak_optimized_allocated_bytes=op,
        work_counters={n:v.get('work_counters',{}) for n,v in values.items()},
        historical_nested=dict(reference_mean_seconds=nested['reference_mean_seconds'],
            optimized_mean_seconds=nested['optimized_mean_seconds'],
            reference_nested_over_cadic=nested['reference_mean_seconds']/r,
            optimized_nested_over_cadic=nested['optimized_mean_seconds']/o,
            optimized_nested_extra_seconds=nested['optimized_mean_seconds']-o,
            paired_between_methods=False, same_bank_trajectory=False,
            provenance={k:v for k,v in nested.items() if k not in ('reference_mean_seconds','optimized_mean_seconds')},
            caveat='Same raw window/runtime/source distance primitive; separate sessions and different method banks. NESTED excludes synchronized diagnostic capture intervals; CADIC has no learner captures.'),
        reportable=False, anomaly_metrics_computed=False, full_stream_ETA=False,
        benchmark_adoption=False)


def worker(c):
    configure_runtime()
    source = verify_delivery()
    cfg = contract()
    work = Path(c['work']); name = c['pass']
    folder = work/'passes'/name; folder.mkdir(parents=True, exist_ok=False)
    if name not in ('PREP', *PASSES):
        raise ValueError('Unknown worker pass')
    if runtime_identity() != c['runtime'] or not torch.cuda.is_available() or runtime_identity()['device_name'] != 'Tesla T4':
        raise ValueError('Exact canonical T4 runtime required')
    with Evidence(c['parent']) as e:
        p = verify_parent(e, cfg); rows = p['rows']
        if runtime_identity() != p['suite']['identity']['runtime']:
            raise ValueError('Parent runtime changed')
        expected = {digest(r): e.json('runs/NESTED_seed0/feature_receipts/'+digest(r)+'.json') for r in rows[:195]}
    began = time.monotonic()
    with activation(name in ('O1', 'O2')):
        actor = session_for('CADIC_seed0', 'cuda')
        if name == 'PREP':
            start, end, chain = 0, 179, digest('empty event stream')
        else:
            old = torch.load(work/'predecessor179.pt', map_location='cpu', weights_only=True)
            if old['source'] != source or old['cursor'] != 179 or old['method'] != 'CADIC_seed0':
                raise ValueError('Wrong CADIC predecessor identity')
            actor.load_state_dict(old['session'])
            if actor.learner is not None or actor.pending is not None:
                raise ValueError('CADIC must not inherit NESTED learner or pending state')
            start, end, chain = 179, 195, old['trace_digest']
        extractor = CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'], checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'), 'cuda')
        try:
            provider = NormalFeatures(c['dataset'], extractor, folder/'feature_receipts', 'cuda')
            provider.job_digest = digest(dict(source=source, method='CADIC_seed0', experiment=SCHEMA))
            provider.prefix_chain = chain
            boot = time.monotonic()-began
            attempted = []
            m, state = bounded_replay(actor, provider, rows, folder, start=start, end=end,
                chain=chain, deadline=c['deadline'], attempt=attempted.append, device='cuda')
            m.update(attempted_event_ids=attempted, bootstrap_seconds=boot, method='CADIC_seed0',
                operator='optimized' if name in ('O1', 'O2') else 'reference', source_digest=source)
            atomic_json(m, folder/'measurement.json')
            if m['status'] != 'complete':
                return m['status']
            for row in rows[start:end]:
                key = digest(row)
                assert_equal(json.loads((folder/'feature_receipts'/(key+'.json')).read_text()), expected[key], 'feature_receipt')
            if state['learner'] is not None or state['pending'] is not None or state['cursor'] != end:
                raise ValueError('Invalid CADIC end state')
            if name == 'PREP':
                atomic_torch_save(dict(source=source, method='CADIC_seed0', cursor=179,
                    session=state, trace_digest=m['trace_digest']), work/'predecessor179.pt')
            else:
                ref = work/'reference_final195.pt'
                if name == 'R1':
                    atomic_torch_save(state, ref)
                    for i in range(179, 195):
                        # R1 events remain available for independent event-by-event audit.
                        assert json.loads((folder/'events'/f'{i:05d}.json').read_text())['cursor'] == i
                else:
                    assert_equal(torch.load(ref, map_location='cpu', weights_only=True), state, 'full_CADIC_final')
                    for i in range(179, 195):
                        assert_equal(json.loads((work/'passes/R1/events'/f'{i:05d}.json').read_text()),
                            json.loads((folder/'events'/f'{i:05d}.json').read_text()), 'CADIC_event')
            atomic_json(dict(status='complete', final_state_digest=state_digest(state),
                features_exact_to_reviewed_parent=True,
                full_state_exact_to_R1=name != 'PREP', source_digest=source,
                predecessor_sha256=sha(work/'predecessor179.pt'),
                measurement_sha256=sha(folder/'measurement.json')),
                folder/'complete.json')
            return 'complete'
        finally:
            extractor.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    c = json.loads(Path(p.parse_args().config).read_text())
    status = worker(c)
    print('[CADIC COST]', c['pass'], status, flush=True)


if __name__ == '__main__':
    main()
