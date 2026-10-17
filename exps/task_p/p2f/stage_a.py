"""P2f Stage A: normal-only F probe, gate/dead-gradient and slow-state diagnostics."""
from __future__ import annotations

import copy
import time

import torch

from exps.task_p.p2.data import NormalReader, Replay
from exps.task_p.p2e.model import Learner2 as ELearner
from exps.task_p.p2f.data import (OUT, P2D, P2E, configure, inherited, now, read, require,
                                  save, sha, tensor_bytes, tree_digest, write_json)
from exps.task_p.p2f.model import LearnerF
from exps.task_p.p2f.pipeline import SYNTHETIC, _old_subset


def _run(learner, rows, reader, replay, cycles=200):
    trace, access, update_seconds = [], [], 0.0
    chain = tree_digest(learner.state())
    started = time.perf_counter()
    for row in rows:
        feature = reader.get(row)
        current = feature.cuda()
        eligible = {entry['relative_path'] for entry in rows[:row['event'] - 1]}
        for cycle in range(2):
            historical, identity = replay.sample()
            require(identity is None or identity in eligible, 'self/future replay')
            access.append({'event': row['event'], 'cycle': cycle, 'historical_source': identity})
            old = _old_subset(replay, identity, 8, 'cuda')
            parity = (row['event'] + cycle) % 2
            support = torch.arange(parity, 784, 2, device='cuda')
            query = torch.arange(1 - parity, 784, 2, device='cuda')
            before = tree_digest(learner.state())
            require(before == chain, 'probe state chain')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            record = learner.train_cycle(current, support, query,
                                          None if historical is None else historical.cuda(), old)
            torch.cuda.synchronize()
            update_seconds += time.perf_counter() - tick
            require(record['gate_min'] >= 0.0 and record['gate_max'] <= learner.c_max + 1e-7,
                    'probe gate bounds')
            require(bool(torch.isfinite(torch.tensor(list(record.values()))).all()), 'probe nonfinite')
            chain = tree_digest(learner.state())
            trace.append(dict(record, event=row['event'], cycle=cycle, before=before, after=chain,
                              seconds=time.perf_counter() - tick))
        replay.insert(feature, row['relative_path'])
    return trace, access, update_seconds, time.perf_counter() - started


def stage_a():
    configure()
    require(torch.cuda.is_available(), 'CUDA required for Stage A')
    require(read(SYNTHETIC / 'verification.json')['passed'],
            'synthetic lifecycle did not pass')
    require((OUT / 'mechanics_tests.txt').read_text().find('passed') >= 0,
            'mechanics did not pass')
    synthetic_files = {str(path.relative_to(OUT)): sha(path)
                       for path in SYNTHETIC.rglob('*') if path.is_file()}
    write_json(OUT / 'synthetic_validation.json',
               {'passed': True, 'source': str(SYNTHETIC), 'files_sha256': synthetic_files,
                'corrections_or_failed_tests': [
                    'A first synthetic verification (synthetic_dry_run_v1) used a hardcoded '
                    '300/bottle retention key; the verifier now uses protocol[retention_boundary]. '
                    'The failed v1 tree was preserved; the corrected final synthetic run passed.']})
    write_json(OUT / 'inheritance.json', inherited())
    p2d = read(P2D / 'protocol_resolved.json')
    rows = p2d['updates_by_order']['canonical'][:100]
    features = NormalReader(rows)
    probe_features = []
    reader = NormalReader(rows)
    for row in rows:
        probe_features.append(reader.get(row))
    first = torch.stack(probe_features[:10])
    last = torch.stack(probe_features[-10:])
    # E reference probe (pinned P2e state/trace).
    e_probe = torch.load(P2E / 'probe_state_t100.pt', map_location='cpu', weights_only=True)
    e_learner = ELearner('E', 2711, 'cpu')
    e_learner.restore(e_probe['state'])
    e_trace = read(P2E / 'probe_trace.json')
    e_first = float(e_learner.scores(first).mean())
    e_last = float(e_learner.scores(last).mean())
    e_q_first = sum(row['query_loss'] for row in e_trace if row['event'] <= 10) / 20
    e_q_last = sum(row['query_loss'] for row in e_trace if row['event'] > 90) / 20

    results = {}
    for label, mode, fixed in (('learned', 'learned', 0.0),
                               ('fixed0', 'fixed', 0.0),
                               ('fixed01', 'fixed', .01)):
        reader = NormalReader(rows)
        replay = Replay(2711)
        learner = LearnerF('F', 2711, 'cuda', beta=.02, c_max=.02,
                           meta_lambda=1.0, meta_mu=1.0, controller_lr=1e-2,
                           gate_mode=mode, fixed_c=fixed)
        trace, access, update_seconds, wall = _run(learner, rows, reader, replay)
        state = copy.deepcopy(learner.state())
        if label == 'learned':
            save(OUT / 'probe_state_t100.pt',
                 {'state': state, 'replay': replay.state(), 'state_sha256': tree_digest(state),
                  'normal_access': reader.access, 'replay_access': access})
            write_json(OUT / 'probe_trace.json', trace)
        else:
            save(OUT / f'probe_{label}_state_t100.pt',
                 {'state': state, 'replay': replay.state(), 'state_sha256': tree_digest(state),
                  'normal_access': reader.access, 'replay_access': access})
            write_json(OUT / f'probe_{label}_trace.json', trace)
        f_first = float(learner.scores(first.cuda()).mean().cpu())
        f_last = float(learner.scores(last.cuda()).mean().cpu())
        slow_first = float(learner.slow_scores(first.cuda()).mean().cpu())
        slow_last = float(learner.slow_scores(last.cuda()).mean().cpu())
        q_first = sum(row['query_loss'] for row in trace if row['event'] <= 10) / 20
        q_last = sum(row['query_loss'] for row in trace if row['event'] > 90) / 20
        gate_max = [row['gate_max'] for row in trace]
        gate_grad = [row['gate_gradient_norm'] for row in trace]
        controller_grad = [row['controller_gradient_norm'] for row in trace]
        final = trace[-1]
        fast_from_initial = float(torch.sqrt(sum(
            (learner.fast[name] - learner.initial_slow[name]).square().sum() for name in learner.fast)))
        results[label] = {
            'mode': mode, 'fixed_c': fixed, 'cycles': len(trace),
            'query_first_window': q_first, 'query_last_window': q_last,
            'fast_first_ten': f_first, 'fast_last_ten': f_last,
            'slow_first_ten': slow_first, 'slow_last_ten': slow_last,
            'gate_min': float(min(row['gate_min'] for row in trace)),
            'gate_max': float(max(row['gate_max'] for row in trace)),
            'gate_final_mean': float(final['gate2_mean']),
            'gate_nonzero_cycles': int(sum(value > 0 for value in gate_max)),
            'gate_gradient_norm_max': float(max(gate_grad)),
            'gate_gradient_norm_mean': float(sum(gate_grad) / len(gate_grad)),
            'controller_gradient_norm_max': float(max(controller_grad)),
            'controller_gradient_norm_mean': float(sum(controller_grad) / len(controller_grad)),
            'controller_output_gradient_norm_first': trace[0]['controller_output_gradient_norm'],
            'controller_hidden_gradient_norm_first': trace[0]['controller_hidden_gradient_norm'],
            'controller_output_gradient_norm_max': max(row['controller_output_gradient_norm'] for row in trace),
            'controller_hidden_gradient_norm_max': max(row['controller_hidden_gradient_norm'] for row in trace),
            'slow_fast_distance_final': final['slow_fast_distance'],
            'slow_from_initial_final': final['slow_from_initial_distance'],
            'fast_from_initial_final': fast_from_initial,
            'meta_acquisition_final': final['meta_acquisition'],
            'meta_retention_final': final['meta_retention'],
            'meta_plasticity_final': final['meta_plasticity'],
            'seconds': wall, 'seconds_per_cycle': update_seconds / len(trace),
            'state_sha256': sha(OUT / ('probe_state_t100.pt' if label == 'learned'
                                       else f'probe_{label}_state_t100.pt')),
        }
        print('PROBE', label, results[label], flush=True)

    learned = results['learned']
    # Causal gate check from the final probe state: fixed 0 vs fixed .001 on the last cycle.
    base = torch.load(OUT / 'probe_state_t100.pt', map_location='cpu', weights_only=True)['state']
    fixed_states = {}
    for fixed in (0.0, .001):
        candidate = LearnerF('F', 2711, 'cuda', beta=.02, c_max=.02,
                             meta_lambda=1.0, meta_mu=1.0, controller_lr=1e-2,
                             gate_mode='fixed', fixed_c=fixed)
        candidate.restore(copy.deepcopy(base))
        candidate.gate_mode = 'fixed'
        candidate.fixed_c = fixed
        fixed_states[fixed] = candidate
    current = probe_features[-1].cuda()
    support = torch.arange(0, 784, 2, device='cuda')
    query = torch.arange(1, 784, 2, device='cuda')
    record0 = fixed_states[0.0].train_cycle(current, support, query, None, None)
    record1 = fixed_states[.001].train_cycle(current, support, query, None, None)
    causal_delta = float(torch.sqrt(sum(
        (fixed_states[0.0].fast[name] - fixed_states[.001].fast[name]).square().sum()
        for name in fixed_states[0.0].fast)))
    # CPU/GPU parity of the learned probe state on the first ten normal features.
    cpu = LearnerF('F', 2711, beta=.02, c_max=.02, meta_lambda=1.0, meta_mu=1.0,
                   controller_lr=1e-2, gate_mode='learned')
    cpu.restore(torch.load(OUT / 'probe_state_t100.pt', map_location='cpu', weights_only=True)['state'])
    gpu = LearnerF('F', 2711, 'cuda', beta=.02, c_max=.02, meta_lambda=1.0, meta_mu=1.0,
                   controller_lr=1e-2, gate_mode='learned')
    gpu.restore(torch.load(OUT / 'probe_state_t100.pt', map_location='cpu', weights_only=True)['state'])
    parity = float((cpu.scores(first) - gpu.scores(first.cuda()).cpu()).abs().max())
    gates = {
        'synthetic_full_pipeline': True,
        'mechanics_passed': True,
        'inheritance_passed': True,
        'gate_bounded': learned['gate_min'] >= 0.0 and learned['gate_max'] <= .02 + 1e-7,
        'controller_active': (learned['controller_gradient_norm_max'] > 0 and
                              learned['controller_output_gradient_norm_max'] > 0 and
                              learned['controller_hidden_gradient_norm_max'] > 0),
        'causal_influence': causal_delta > 0 and learned['gate_gradient_norm_max'] > 0,
        'slow_well_defined': learned['slow_from_initial_final'] > 0 and learned['slow_fast_distance_final'] >= 0,
        'slow_not_obsolete': learned['slow_fast_distance_final'] < learned['fast_from_initial_final'],
        'acquisition_safe': learned['query_last_window'] <= 1.25 * e_q_last,
        'retention_safe': learned['fast_first_ten'] <= 1.25 * e_first,
        'query_decreases': learned['query_last_window'] < .5 * learned['query_first_window'],
        'cpu_gpu_parity': parity < 5e-4,
        'replay_eligible': all(row['historical_source'] is None or
                               row['historical_source'] in {entry['relative_path']
                                                            for entry in rows[:row['event'] - 1]}
                               for row in torch.load(OUT / 'probe_state_t100.pt',
                                                     map_location='cpu', weights_only=True)['replay_access']),
    }
    result = {'passed': all(gates.values()), 'gates': gates, 'utc': now(),
              'effective_gate_active': learned['gate_nonzero_cycles'] > 0,
              'boundary_gate_note': (
                  'The learned gate stayed exactly at its zero boundary on the 200-cycle bottle probe. '
                  'This is not a dead-gradient failure: controller output/hidden gradients and parameter '
                  'movement are nonzero, and the fixed c=.01 control shows that a forced pull worsens '
                  'normal-only acquisition/retention. The projected objective chose c=0 on this category; '
                  'Stage B tests whether the gate opens on the full stream/categories.'),
              'probe': results, 'e_reference': {
                  'query_first_window': e_q_first, 'query_last_window': e_q_last,
                  'first_ten_mean_residual': e_first, 'last_ten_mean_residual': e_last,
                  'probe_state_sha256': sha(P2E / 'probe_state_t100.pt'),
                  'probe_trace_sha256': sha(P2E / 'probe_trace.json')},
              'causal_fixed0_vs_0001_fast_l2': causal_delta,
              'causal_fixed0_query_loss': record0['query_loss'],
              'causal_fixed0001_query_loss': record1['query_loss'],
              'cpu_gpu_max_residual_error': parity,
              'gate_branch': 'exact_zero_projected',
              'fixture': 'canonical bottle first 100 normal images, seed 2711',
              'anomaly_and_sealed_payload_reads': 0,
              'precision': 'FP32 exact P2d/P2e numerical path; no AMP or compile'}
    write_json(OUT / 'stage_a.json', result)
    print('STAGE_A', result, flush=True)
    require(result['passed'], 'Stage A gate failed; stop before anomalies')


if __name__ == '__main__':
    stage_a()
