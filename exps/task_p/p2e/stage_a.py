"""One fixed normal-only probe, saved states, provenance and complete gate record."""
from __future__ import annotations

import copy
import time

import torch

from exps.task_p.p2.model import Learner as AdamLearner
from exps.task_p.p2d.model import Learner2 as NestedLearner
from exps.task_p.p2e.data import (OUT, P2, P2D, NormalReader, Replay, configure,
    inherited, now, read, require, save, sha, tensor_bytes, tree_digest, write_json)
from exps.task_p.p2e.model import Learner2


def stage_a():
    configure()
    require(torch.cuda.is_available(), 'CUDA required for Stage A')
    require(read(OUT / 'synthetic_dry_run/verification.json')['passed'], 'synthetic lifecycle did not pass')
    write_json(OUT / 'inheritance.json', inherited())
    rows = read(P2D / 'protocol_resolved.json')['updates_by_order']['canonical'][:100]
    reader, replay = NormalReader(rows), Replay(2711)
    learner = Learner2('E', 2711, 'cuda')
    initial = copy.deepcopy(learner.state())
    initial_norm = float(torch.sqrt(sum(value.square().sum() for value in learner.fast.values())))
    initial_slow = tree_digest(initial['model'])
    initial_controller = tree_digest(initial['controller'])
    chain = tree_digest(initial)
    trace, replay_access, first_ten, last_ten, update_seconds = [], [], [], [], 0.0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for row in rows:
        feature = reader.get(row)
        if row['event'] <= 10:
            first_ten.append(feature.clone())
        if row['event'] > 90:
            last_ten.append(feature.clone())
        current = feature.cuda()
        eligible = {entry['relative_path'] for entry in rows[:row['event']-1]}
        for cycle in range(2):
            old, identity = replay.sample()
            require(identity is None or identity in eligible, 'self/future replay')
            replay_access.append({'event': row['event'], 'cycle': cycle, 'historical_source': identity})
            parity = (row['event']+cycle) % 2
            support, query = torch.arange(parity, 784, 2, device='cuda'), torch.arange(1-parity, 784, 2, device='cuda')
            before = tree_digest(learner.state())
            require(before == chain, 'probe state chain')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            record = learner.train_cycle(current, support, query, None if old is None else old.cuda())
            torch.cuda.synchronize()
            duration = time.perf_counter()-tick
            update_seconds += duration
            require(bool(torch.isfinite(torch.tensor(list(record.values()))).all()), 'nonfinite probe')
            chain = tree_digest(learner.state())
            trace.append(dict(record, event=row['event'], cycle=cycle, seconds=duration, before=before, after=chain))
        replay.insert(feature, row['relative_path'])
    state = copy.deepcopy(learner.state())
    save(OUT / 'probe_state_t100.pt', {'state': state, 'replay': replay.state(), 'state_sha256': chain,
                                     'normal_access': reader.access, 'replay_access': replay_access})
    write_json(OUT / 'probe_trace.json', trace)
    diagnostics = {}
    first_features, last_features = torch.stack(first_ten), torch.stack(last_ten)
    for arm in ('E', 'C2', 'D2', 'B'):
        if arm == 'E':
            cpu = Learner2('E', 2711)
            cpu.restore(state)
        else:
            root = P2 if arm == 'B' else P2D
            checkpoint = torch.load(root / 'canonical_seed2711/state_t100.pt', map_location='cpu', weights_only=True)
            cpu = AdamLearner(arm, 2711) if arm == 'B' else NestedLearner(arm, 2711)
            cpu.restore(checkpoint['states'][arm])
        diagnostics[arm] = {'first_ten_mean_residual': float(cpu.scores(first_features).mean()),
                             'last_ten_mean_residual': float(cpu.scores(last_features).mean())}
    cpu = Learner2('E', 2711)
    cpu.restore(state)
    parity = float((cpu.scores(first_features)-learner.scores(first_features.cuda()).cpu()).abs().max())
    first = sum(row['query_loss'] for row in trace if row['event'] <= 10)/20
    last = sum(row['query_loss'] for row in trace if row['event'] > 90)/20
    final_norm = float(torch.sqrt(sum(value.square().sum() for value in learner.fast.values())))
    gates = {'query_decreases': last < .5*first, 'bounded_parameter_norm': final_norm < 3*initial_norm,
             'zero_slow_gradient': all(row['outer_gradient_norm'] == 0 for row in trace),
             'controller_inactive': all(row['controller_gradient_norm'] == 0 for row in trace)
                                    and tree_digest(state['controller']) == initial_controller,
             'slow_state_unchanged': tree_digest(state['model']) == initial_slow,
             'cpu_gpu_parity': parity < 5e-4, 'synthetic_full_pipeline': True}
    result = {'passed': all(gates.values()), 'gates': gates, 'utc': now(),
              'first_query_window': first, 'last_query_window': last,
              'normal_residuals': diagnostics, 'initial_parameter_norm': initial_norm, 'final_parameter_norm': final_norm,
              'cpu_gpu_max_residual_error': parity, 'probe_state_sha256': sha(OUT / 'probe_state_t100.pt'),
              'probe_trace_sha256': sha(OUT / 'probe_trace.json'),
              'state_tensor_bytes': tensor_bytes(state), 'replay_tensor_bytes': tensor_bytes(replay.state()),
              'seconds': time.perf_counter()-started, 'seconds_per_cycle': update_seconds/200,
              'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
              'precision': 'FP32 exact P2d numerical path, no AMP or compile',
              'normal_rows': 100, 'anomaly_and_sealed_payload_reads': 0,
              'comparisons_do_not_select_hyperparameters': True}
    write_json(OUT / 'stage_a.json', result)
    print('STAGE_A', result, flush=True)
    require(result['passed'], 'Stage A gate failed; stop')


if __name__ == '__main__':
    stage_a()
