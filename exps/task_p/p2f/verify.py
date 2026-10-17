"""Independent numerical/invariant self-checks for P2f; no learner updates or anomaly rescores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from exps.task_p.p1_readout import native_map
from exps.task_p.p2f.analysis import classify_f
from exps.task_p.p2f.data import (CATEGORIES, P2E, REFERENCES, ROOT, configure, masks_at,
    model_arguments, now, protocol_at, read, reference_artifacts, require, sha,
    tensor_bytes, tree_digest, write_json)
from exps.task_p.p2f.model import LearnerF


def _learner(protocol, seed, device='cpu'):
    config = protocol['f_config']
    return LearnerF('F', seed, device, **model_arguments(protocol),
                    beta=config['beta'], c_max=config['c_max'],
                    meta_lambda=config['meta_lambda'], meta_mu=config['meta_mu'],
                    controller_lr=config['controller_lr'],
                    gate_mode=config['gate_mode'], fixed_c=0.0)


def verify(directory):
    directory = Path(directory)
    configure()
    protocol = protocol_at(directory)
    digest = sha(directory / 'protocol_resolved.json')
    points, samples, summary = (read(directory / name) for name in
        ('point_metrics.json', 'bootstrap_samples.json', 'summary.json'))
    sources = {row['relative_path']: row for row in protocol['evaluations']}
    masks = masks_at(directory, protocol)
    config = protocol['f_config']
    result = {'runs': {}, 'max_sklearn_ap_difference': 0.0,
              'max_sklearn_image_auc_difference': 0.0, 'max_image_bootstrap_difference': 0.0,
              'max_paired_summary_difference': 0.0, 'anomaly_forward_passes': 0,
              'sealed_payload_reads': 0, 'native_groups_independently_checked': 0,
              'controller_moved_runs': 0, 'slow_moved_runs': 0}
    draws = {}
    for category_index, category in enumerate(CATEGORIES):
        category_rows = [row for row in protocol['evaluations'] if row['category'] == category]
        labels = np.array([row['label'] for row in category_rows])
        generator = np.random.default_rng(102701 + category_index)
        multiplicities = []
        for _ in range(protocol['bootstrap_draws']):
            draw = np.zeros(len(category_rows), dtype=np.int64)
            for label in (0, 1):
                positions = np.where(labels == label)[0]
                for position in generator.choice(positions, len(positions), replace=True):
                    draw[position] += 1
            multiplicities.append(draw)
        draws[category] = np.stack(multiplicities)
    for order in protocol['orders']:
        rows = protocol['updates_by_order'][order]
        patches = protocol['model_dimensions']['side'] ** 2
        for seed in protocol['seeds']:
            output = directory / f'{order}_seed{seed}'
            normal, replay_log, trace = (read(output / name) for name in
                ('normal_access.json', 'replay_access.json', 'trace.json'))
            require([row['source'] for row in normal] == [row['relative_path'] for row in rows],
                    'normal chronology')
            require(len(trace) == len(rows) * protocol['cycles_per_image'], 'cycle count')
            initial = _learner(protocol, seed)
            initial_state = initial.state()
            require(tree_digest(initial_state) == trace[0]['before'], 'initial chain digest')
            require(all(first['after'] == second['before']
                        for first, second in zip(trace, trace[1:])), 'broken state chain')
            require([(row['event'], row['cycle']) for row in trace] ==
                    [(event, cycle) for event in range(1, len(rows) + 1) for cycle in range(2)],
                    'trace ordering')
            for record in trace:
                require(np.isfinite([value for value in record.values()
                                     if isinstance(value, (int, float))]).all(), 'nonfinite trace')
                require(record['gate_min'] >= 0.0 and
                        record['gate_max'] <= config['c_max'] + 1e-7, 'gate outside bounds')
                require(record['rate_value'] == .001, 'fast rate changed')
                require(record['slow_fast_distance'] >= 0.0 and
                        record['slow_from_initial_distance'] >= 0.0, 'invalid slow diagnostic')
                require(record['gate_gradient_norm'] >= 0.0 and
                        record['controller_gradient_norm'] >= 0.0, 'invalid controller gradient')
                require(record['fast_parameter_displacement'] > 0, 'fast not updated')
            generator = torch.Generator().manual_seed(seed + 9000)
            reservoir, replay_snapshots, replay_index = [], {}, 0
            for event, row in enumerate(rows, start=1):
                for cycle in range(2):
                    source = None if not reservoir else reservoir[
                        int(torch.randint(len(reservoir), (1,), generator=generator))]
                    require(replay_log[replay_index] ==
                            {'event': event, 'cycle': cycle, 'historical_source': source},
                            'Algorithm R draw mismatch')
                    require(source is None or source in {previous['relative_path']
                                                        for previous in rows[:event - 1]},
                            'self/future replay')
                    replay_index += 1
                if len(reservoir) < protocol['replay_capacity']:
                    reservoir.append(row['relative_path'])
                else:
                    slot = int(torch.randint(event, (1,), generator=generator))
                    if slot < protocol['replay_capacity']:
                        reservoir[slot] = row['relative_path']
                if event in protocol['boundaries']:
                    replay_snapshots[event] = (list(reservoir), generator.get_state().clone())
            require(len(replay_log) == replay_index, 'replay record count')
            checkpoint_hashes, distances = {}, {}
            budgets = read(output / 'budgets.json')
            controller_moved = False
            slow_moved = False
            for boundary in protocol['boundaries']:
                checkpoint = torch.load(output / f'state_t{boundary}.pt', map_location='cpu',
                                        weights_only=True)
                state = checkpoint['states']['F']
                require(checkpoint['boundary'] == boundary and
                        checkpoint['protocol_sha256'] == digest, 'checkpoint identity')
                checkpoint_hashes[boundary] = tree_digest(state)
                require(checkpoint_hashes[boundary] == checkpoint['state_sha256']['F'] ==
                        trace[2 * boundary - 1]['after'], 'boundary link')
                require(state['cycles'] == 2 * boundary, 'optimizer cycle counter')
                require(all(bool(torch.isfinite(value).all()) for value in state['model'].values()),
                        'nonfinite slow state')
                require(all(bool(torch.isfinite(value).all()) for value in state['controller'].values()),
                        'nonfinite controller')
                require(all((not value.requires_grad) and bool(torch.isfinite(value).all())
                            for value in state['fast'].values()), 'fast state invalid')
                require(all((not value.requires_grad) and bool(torch.isfinite(value).all())
                            for pair in state['moments'].values() for value in pair),
                        'fast moments invalid')
                if tree_digest(state['model']) != tree_digest(initial_state['model']):
                    slow_moved = True
                if tree_digest(state['controller']) != tree_digest(initial_state['controller']):
                    controller_moved = True
                historical = checkpoint['replay']
                expected_ids, expected_rng = replay_snapshots[boundary]
                require(historical['identities'] == expected_ids and
                        torch.equal(historical['rng'], expected_rng), 'reservoir state')
                budget = budgets[str(boundary)]['F']
                require(budget['learner_tensor_bytes'] == tensor_bytes(state), 'learner byte accounting')
                require(budget['replay_tensor_bytes'] == tensor_bytes(historical), 'replay byte accounting')
                require(budget['online_tensor_bytes'] == tensor_bytes(state) + tensor_bytes(historical),
                        'online byte accounting')
                require(budget['inference_tensor_bytes'] == tensor_bytes(state['fast']) +
                        tensor_bytes(state['model']['coordinates']), 'inference byte accounting')
                distances[str(boundary)] = float(torch.sqrt(sum(
                    (value - initial_state['fast'][name]).square().sum()
                    for name, value in state['fast'].items())))
            result['controller_moved_runs'] += int(controller_moved)
            result['slow_moved_runs'] += int(slow_moved)
            records = read(output / 'predictions.json')
            expected_records = {(boundary, row['relative_path']) for boundary in protocol['boundaries']
                                for row in protocol['evaluations']
                                if row['category'] in {entry['category'] for entry in rows[:boundary]}}
            require(len(records) == len(expected_records) and
                    {(row['boundary'], row['source']) for row in records} == expected_records,
                    'prediction budget or duplicate')
            for row in records:
                require(row['arm'] == 'F' and row['state_sha256'] == checkpoint_hashes[row['boundary']],
                        'score state linkage')
                require(row['source'] not in protocol['sealed_identity_paths'], 'sealed prediction')
                require(len(row['patch_scores']) == patches and np.isfinite(row['patch_scores']).all(),
                        'patch vector invalid')
                require(max(row['patch_scores']) == row['image_score'], 'image max readout')
            for boundary in protocol['boundaries']:
                for category in CATEGORIES:
                    group = [row for row in records if row['boundary'] == boundary and
                             sources[row['source']]['category'] == category]
                    if not group:
                        continue
                    values = np.concatenate([native_map(torch.tensor(row['patch_scores'], dtype=torch.float32),
                                                        sources[row['source']]['hw']).flatten()
                                             for row in group])
                    targets = np.concatenate([masks[row['source']].flatten() for row in group])
                    labels = np.array([sources[row['source']]['label'] for row in group])
                    scores = np.array([row['image_score'] for row in group])
                    key = f'{order}/{seed}/{boundary}/{category}/F'
                    ap, auc = average_precision_score(targets, values), roc_auc_score(labels, scores)
                    result['max_sklearn_ap_difference'] = max(
                        result['max_sklearn_ap_difference'], abs(ap - points[key]['pixel_ap']))
                    result['max_sklearn_image_auc_difference'] = max(
                        result['max_sklearn_image_auc_difference'], abs(auc - points[key]['image_auroc']))
                    positive, negative = np.where(labels == 1)[0], np.where(labels == 0)[0]
                    comparison = (scores[positive, None] > scores[negative]).astype(float) + \
                        .5 * (scores[positive, None] == scores[negative])
                    weights = draws[category]
                    independent_auc = (((weights[:, positive] @ comparison) * weights[:, negative]).sum(1) /
                                       (weights[:, positive].sum(1) * weights[:, negative].sum(1)))
                    result['max_image_bootstrap_difference'] = max(
                        result['max_image_bootstrap_difference'],
                        float(np.max(np.abs(independent_auc - np.array(samples[key]['image_auroc'])))))
                    result['native_groups_independently_checked'] += 1
            error = read(output / 'inference_resources.json')['cpu_gpu_max_patch_score_difference']
            require(error < 5e-4, 'recorded CPU/GPU parity')
            result['runs'][f'{order}/{seed}'] = {
                'cycles': len(trace), 'replay_draws_exact': replay_index,
                'checkpoints': len(checkpoint_hashes), 'predictions': len(records),
                'endpoint_L2_from_initialization': distances, 'max_score_parity_error': error,
                'max_support_gradient_norm': max(row['support_gradient_norm'] for row in trace),
                'max_query_gradient_norm': max(row['query_gradient_norm'] for row in trace),
                'controller_moved': controller_moved, 'slow_moved': slow_moved,
                'gate_max_global': max(row['gate_max'] for row in trace),
                'gate_nonzero_cycles': int(sum(row['gate_max'] > 0 for row in trace)),
                'slow_fast_distance_final': trace[-1]['slow_fast_distance'],
                'slow_from_initial_final': trace[-1]['slow_from_initial_distance']}
            print('VERIFIED', order, seed, flush=True)
    if not protocol['synthetic']:
        reference_points, reference_samples = reference_artifacts(protocol)
        require(all(points[key] == value for key, value in reference_points.items()),
                'reference point metrics changed')
        require(all(samples[key] == reference_samples[key] for key, value in reference_samples.items()),
                'reference bootstrap samples changed')
        result['preserved_files_checked'] = len(read(directory / 'inheritance.json')['preserved'])
        for name, expected in read(directory / 'inheritance.json')['preserved'].items():
            require(sha(ROOT / name) == expected, f'preserved file changed: {name}')
    all_pairs = []
    for order in protocol['orders']:
        final_boundary = protocol['boundaries'][-1]
        for reference in REFERENCES:
            point_delta = float(np.mean([
                points[f'{order}/{seed}/{final_boundary}/{category}/F']['pixel_ap'] -
                points[f'{order}/{seed}/{final_boundary}/{category}/{reference}']['pixel_ap']
                for category in CATEGORIES for seed in protocol['seeds']]))
            low = np.mean([np.array(samples[f'{order}/{seed}/{final_boundary}/{category}/F']['ap_low']) -
                           np.array(samples[f'{order}/{seed}/{final_boundary}/{category}/{reference}']['ap_high'])
                           for category in CATEGORIES for seed in protocol['seeds']], axis=0)
            high = np.mean([np.array(samples[f'{order}/{seed}/{final_boundary}/{category}/F']['ap_high']) -
                            np.array(samples[f'{order}/{seed}/{final_boundary}/{category}/{reference}']['ap_low'])
                            for category in CATEGORIES for seed in protocol['seeds']], axis=0)
            expected = summary['contrasts'][order]['F_minus_' + reference]
            discrepancy = max(abs(point_delta - expected['difference']),
                              abs(float(np.quantile(low, .025)) - expected['ci'][0]),
                              abs(float(np.quantile(high, .975)) - expected['ci'][1]))
            result['max_paired_summary_difference'] = max(result['max_paired_summary_difference'], discrepancy)
            all_pairs.append(discrepancy)
    require(result['max_sklearn_ap_difference'] < 1e-9, 'independent AP mismatch')
    require(result['max_sklearn_image_auc_difference'] < 1e-12, 'independent image AUC mismatch')
    require(result['max_image_bootstrap_difference'] < 1e-12, 'image bootstrap mismatch')
    require(max(all_pairs) < 1e-12, 'paired inference mismatch')
    times = [protocol['locked_utc']] + [read(directory / name)['utc'] for name in
             ('training_started.json', 'training_completed.json', 'evaluation_started.json',
              'evaluation_completed.json', 'analysis_started.json', 'analysis_completed.json')]
    require(times == sorted(times), 'lock/lifecycle chronology')
    # Recompute the predeclared decision from the stored arrays and summary.
    primary = summary['contrasts']['canonical']['boundary_categories'][f'{protocol["retention_boundary"]}/bottle/F_minus_E']
    decision = classify_f(primary, summary['acquisition_safety'], summary['macro_safety'],
                          protocol['decision'])
    require(decision['verdict'] == summary['verdict'], 'decision mismatch')
    result.update({'passed': True, 'utc': now(), 'protocol_sha256': digest,
                   'decision_reproduced': decision,
                   'scope': 'independent numerical implementations inside primary session, not a secondary-agent audit'})
    write_json(directory / 'verification.json', result)
    ledger = {str(path.relative_to(directory)): {'sha256': sha(path), 'bytes': path.stat().st_size}
              for path in directory.rglob('*') if path.is_file() and path.name != 'artifact_integrity.json'}
    write_json(directory / 'artifact_integrity.json', {'files': ledger, 'protocol_sha256': digest})
    print('VERIFICATION_PASS', result['native_groups_independently_checked'], 'AP groups; max differences',
          result['max_sklearn_ap_difference'], result['max_image_bootstrap_difference'],
          result['max_paired_summary_difference'], flush=True)
