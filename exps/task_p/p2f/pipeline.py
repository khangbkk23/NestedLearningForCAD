"""One exclusive P2f lifecycle, exercised first on a synthetic fixture."""
from __future__ import annotations

import argparse
import copy
import resource
import time
from pathlib import Path

import torch

from exps.task_p.p2f.data import (CATEGORIES, OUT, P2D, P2E, ROOT, Reader, Replay,
    configure, eval_features, key_references, model_arguments, now, pinned_sources,
    protocol_at, read, reference_samples, require, save, sha, tensor_bytes, tree_digest,
    write_json)
from exps.task_p.p2f.model import LearnerF

SYNTHETIC = OUT / 'synthetic_dry_run_final'

F_CONFIG = {'fast_rate': .001, 'beta': .02, 'c_max': .02, 'meta_lambda': 1.0,
            'meta_mu': 1.0, 'controller_lr': 1e-2, 'old_replay_count': 8,
            'gate_mode': 'learned', 'gate_init': 'exact_zero',
            'gate_clamp': 'straight_through_projected', 'slow_rule': 'ema'}


def _e_reference():
    points = read(P2E / 'point_metrics.json')
    seeds = read(P2D / 'protocol_resolved.json')['seeds']
    bottle = [points[f'canonical/{seed}/300/bottle/E']['pixel_ap'] for seed in seeds]
    macro = [points[f'canonical/{seed}/350/{category}/E']['pixel_ap']
             for seed in seeds for category in CATEGORIES]
    return {'bottle300': sum(bottle) / len(bottle), 'macro350': sum(macro) / len(macro)}


def prepare(directory, synthetic=False):
    directory = Path(directory)
    configure()
    require(torch.cuda.is_available(), 'CUDA required')
    if synthetic:
        require(not directory.exists(), 'synthetic output already exists')
        directory.mkdir(parents=True)
        generator = torch.Generator().manual_seed(86103)
        orders = {'canonical': list(CATEGORIES),
                  'alternate': ['bottle', 'hazelnut', 'carpet']}
        normal, evaluation, masks, rows = {}, {}, {}, []
        for category_index, category in enumerate(CATEGORIES):
            for image_index in range(3):
                name = f'{category}/train/good/{image_index}.synthetic'
                normal[name] = torch.randn(784, 8, generator=generator) + .1 * category_index
            for image_index in range(4):
                label = int(image_index >= 2)
                name = f'{category}/development/{image_index}.synthetic'
                evaluation[name] = torch.randn(784, 8, generator=generator)
                shape = [17 + category_index + image_index, 19 + image_index]
                mask = torch.zeros(shape, dtype=torch.bool)
                if label:
                    mask[2:8, 4:12] = True
                masks[name] = mask
                rows.append({'relative_path': name, 'category': category, 'label': label,
                             'hw': shape, 'role': 'synthetic_evaluation'})
        streams = {}
        for order, categories in orders.items():
            entries = []
            for category in categories:
                entries += [{'relative_path': f'{category}/train/good/{index}.synthetic',
                             'category': category, 'role': 'normal_train'} for index in range(2)]
            entries += [{'relative_path': 'bottle/train/good/2.synthetic',
                         'category': 'bottle', 'role': 'normal_train'}]
            streams[order] = [dict(row, event=index + 1) for index, row in enumerate(entries)]
        save(directory / 'synthetic_inputs.pt',
             {'normal': normal, 'evaluation': evaluation, 'masks': masks})
        base = {'orders': orders, 'updates_by_order': streams, 'seeds': [2711],
                'boundaries': [2, 4, 6, 7], 'evaluations': rows,
                'sealed_identity_paths': [], 'dataset_root': None,
                'model_dimensions': {'channels': 8, 'latent': 4, 'hidden': 8, 'side': 28},
                'bootstrap_draws': 1000, 'reference_sha256': {},
                'inheritance_sha256': '', 'stage_a_sha256': {},
                'retention_boundary': 6}
    else:
        require(directory.resolve() == OUT.resolve(), 'real output must be the P2f directory')
        stage = read(directory / 'stage_a.json')
        require(stage['passed'], 'Stage A did not pass')
        require(read(SYNTHETIC / 'verification.json')['passed'],
                'synthetic verifier failed')
        tests = (directory / 'mechanics_tests.txt').read_text()
        require('passed' in tests and 'failed' not in tests and 'skipped' not in tests,
                'mechanics not all passed')
        inherited = read(directory / 'inheritance.json')
        require(inherited['passed'], 'inheritance failed')
        historical = read(P2D / 'protocol_resolved.json')
        base = {name: historical[name] for name in
                ('updates_by_order', 'seeds', 'boundaries', 'evaluations',
                 'sealed_identity_paths', 'dataset_root')}
        base.update({'orders': {'canonical': list(CATEGORIES),
                                'alternate': ['bottle', 'hazelnut', 'carpet']},
                     'model_dimensions': {'channels': 768, 'latent': 64, 'hidden': 128, 'side': 28},
                     'bootstrap_draws': 1000,
                     'reference_sha256': key_references(),
                     'inheritance_sha256': sha(directory / 'inheritance.json'),
                     'retention_boundary': 300,
                     'stage_a_sha256': {path.name: sha(path) for path in directory.iterdir()
                                        if path.is_file() and path.suffix in ('.json', '.txt', '.pt')}})
        base['stage_a_sha256']['probe_state_t100.pt'] = sha(directory / 'probe_state_t100.pt')
    e_reference = _e_reference() if not synthetic else {'bottle300': .5, 'macro350': .5}
    protocol = dict(base, schema='p2f_conditional_consolidation_v1', locked_utc=now(),
        synthetic=synthetic, arms=['F'], cycles_per_image=2, replay_capacity=24,
        p2e_protocol_sha256='18ea5eb104b6cdccce1e89073f3b509d84a583bf37f6ac83212bd5c5fbb6fb75',
        p2d_protocol_sha256='182601095371de52627e8a8e2bae26299a3573f5bcf7a63a88f07c762e44d94b',
        p2_protocol_sha256='be3a0fa0a351f4acf0e087667090d25b3257e5b5cfccc5dac8b0daa3b1ea0fa2',
        code_sha256=pinned_sources(), f_config=dict(F_CONFIG),
        intervention='F = E direct fast path at fixed rate .001 + bounded learned per-tensor per-step EMA consolidation gate (clamp 0..0.02) + normal-only retention-aware controller objective',
        exposure='SYNTHETIC ONLY' if synthetic else 'DEVELOPMENT EVIDENCE ONLY: consumed P1 fixture',
        optimizer={'fast_rate': .001, 'controller_lr': 1e-2,
                   'betas': [.9, .999], 'sqrt_epsilon': 1e-12,
                   'denominator_epsilon': 1e-8, 'gradient_clip': 1.0,
                   'tensor_update_rms_clip': .02, 'precision': 'FP32; deterministic; no TF32'},
        metrics={'primary': 'canonical bottle native Pixel AP F-E at t300 >= +.05 with conservative paired lower > 0',
                 'acquisition_safety': 'no category acquisition AP loss > .01 versus E',
                 'macro_safety': 'final canonical macro F >= E - .01',
                 'bootstrap': '1000 stratified whole-image draws, seed102701+category; fixed seeds averaged within draw',
                 'paired_interval': 'quantile(F_low-E_high,.025), quantile(F_high-E_low,.975)',
                 'point_ap': 'exact tied-score ranking; native bilinear interpolation; no smoothing',
                 'secondary': 'final macro; Image AUROC; carpet/hazelnut retention; orders; seeds; gate/slow-state evolution; cost'},
        decision={'primary_target': .05, 'acquisition_margin': .01, 'macro_margin': .01,
                  'harm_margin': -.02, 'reference_E_bottle300': e_reference['bottle300'],
                  'reference_E_macro350': e_reference['macro350'],
                  'supported': 'F-E bottle300 >= .05 AND paired lower > 0 AND all acquisition deltas >= -.01 AND macro F >= E-.01',
                  'partial': 'retention target met but safety condition fails, or point >= .05 with lower <= 0',
                  'no_benefit': 'F-E bottle300 < .05 or lower <= 0 and F indistinguishable from E / gate near zero',
                  'harm': 'F materially worse than E (bottle/macro <= E-.02 or acquisition loss > .01)',
                  'inconclusive': 'failed uncertainty/integrity gate'},
        stop='fail closed on nonfinite, gate outside [0,.02], invalid slow state, hash drift, chronology, sealed or replay violation; no outcome tuning or rescoring',
        accounting='F state includes slow predictor, fast weights, fast moments, controller, controller optimizer and 24 replay grids; no model optimizer',
        environment={'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'threads': 4})
    require(not set(row['relative_path'] for row in protocol['evaluations']) &
            set(protocol['sealed_identity_paths']), 'sealed overlap')
    write_json(directory / 'protocol_resolved.json', protocol)
    with (directory / 'protocol.sha256').open('x') as stream:
        stream.write(sha(directory / 'protocol_resolved.json') + '\n')
    print('PREPARED', directory, sha(directory / 'protocol_resolved.json'), flush=True)


def _old_subset(replay, current_identity, count, device):
    images = [image for identity, image in zip(replay.identities, replay.images)
              if identity != current_identity]
    if not images:
        return None
    return torch.stack([image.to(device) for image in images[:count]])


def train(directory):
    directory = Path(directory)
    configure()
    require(torch.cuda.is_available(), 'CUDA required for training')
    protocol = protocol_at(directory)
    digest = sha(directory / 'protocol_resolved.json')
    config = protocol['f_config']
    write_json(directory / 'training_started.json', {'utc': now(), 'protocol_sha256': digest})
    overall = time.perf_counter()
    patches = protocol['model_dimensions']['side'] ** 2
    for order in protocol['orders']:
        for seed in protocol['seeds']:
            output = directory / f'{order}_seed{seed}'
            output.mkdir(exist_ok=False)
            write_json(output / 'started.json', {'utc': now(), 'protocol_sha256': digest})
            rows = protocol['updates_by_order'][order]
            reader, replay = Reader(directory, protocol, rows), Replay(seed, capacity=protocol['replay_capacity'])
            learner = LearnerF('F', seed, 'cuda', **model_arguments(protocol),
                               beta=config['beta'], c_max=config['c_max'],
                               meta_lambda=config['meta_lambda'], meta_mu=config['meta_mu'],
                               controller_lr=config['controller_lr'],
                               gate_mode=config['gate_mode'], fixed_c=0.0)
            initial_slow = tree_digest(learner.state()['model'])
            initial_controller = tree_digest(learner.state()['controller'])
            chain = tree_digest(learner.state())
            trace, replay_trace, budgets, update_seconds = [], [], {}, 0.0
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for row in rows:
                features = reader.get(row)
                current = features.cuda()
                eligible = {entry['relative_path'] for entry in rows[:row['event'] - 1]}
                for cycle in range(protocol['cycles_per_image']):
                    historical, identity = replay.sample()
                    require(identity is None or identity in eligible, 'self/future replay')
                    replay_trace.append({'event': row['event'], 'cycle': cycle, 'historical_source': identity})
                    historical_gpu = None if historical is None else historical.cuda()
                    old_gpu = _old_subset(replay, identity, config['old_replay_count'], 'cuda')
                    parity = (row['event'] + cycle) % 2
                    support = torch.arange(parity, patches, 2, device='cuda')
                    query = torch.arange(1 - parity, patches, 2, device='cuda')
                    before = tree_digest(learner.state())
                    require(before == chain, 'state chain reset')
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    record = learner.train_cycle(current, support, query, historical_gpu, old_gpu)
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - tick
                    update_seconds += duration
                    require(bool(torch.isfinite(torch.tensor(list(record.values()))).all()),
                            'nonfinite diagnostics')
                    require(record['gate_min'] >= 0.0 and record['gate_max'] <= config['c_max'] + 1e-7,
                            'gate outside bounds')
                    require(record['rate_value'] == .001, 'fast rate changed')
                    require(record['slow_fast_distance'] >= 0.0 and record['slow_from_initial_distance'] >= 0.0,
                            'invalid slow state')
                    after = tree_digest(learner.state())
                    require(after != before, 'no state write')
                    trace.append(dict(record, event=row['event'], cycle=cycle, arm='F',
                                      before=before, after=after, seconds=duration))
                    chain = after
                replay.insert(features, row['relative_path'])
                if row['event'] in protocol['boundaries']:
                    state, replay_state = copy.deepcopy(learner.state()), replay.state()
                    require(tree_digest(state['model']) != initial_slow or row['event'] == 0,
                            'slow state never moved')
                    require(all(bool(torch.isfinite(value).all())
                                for value in state['controller'].values()),
                            'controller state invalid')
                    boundary = row['event']
                    save(output / f'state_t{boundary}.pt',
                         {'boundary': boundary, 'protocol_sha256': digest,
                          'states': {'F': state}, 'replay': replay_state,
                          'state_sha256': {'F': chain}})
                    budgets[str(boundary)] = {'F': {
                        'learner_tensor_bytes': tensor_bytes(state),
                        'replay_tensor_bytes': tensor_bytes(replay_state),
                        'online_tensor_bytes': tensor_bytes(state) + tensor_bytes(replay_state),
                        'inference_tensor_bytes': tensor_bytes(state['fast']) +
                                                  tensor_bytes(state['model']['coordinates'])}}
                    print('TRAIN', order, seed, boundary, round(time.perf_counter() - started, 2), flush=True)
            write_json(output / 'trace.json', trace)
            write_json(output / 'replay_access.json', replay_trace)
            write_json(output / 'normal_access.json', reader.access)
            write_json(output / 'budgets.json', budgets)
            write_json(output / 'resources.json', {'seconds': time.perf_counter() - started,
                'update_seconds_by_arm': {'F': update_seconds},
                'cycles_per_arm': {'F': learner.cycles},
                'direct_steps_per_arm': {'F': 2 * learner.cycles},
                'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(),
                'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'state_hash_and_serialization_overhead': 'included in wall time, not update time'})
            write_json(output / 'completed.json', {'utc': now(), 'protocol_sha256': digest})
            del learner, state, replay, replay_state, current, historical_gpu
            torch.cuda.empty_cache()
    write_json(directory / 'training_completed.json',
               {'utc': now(), 'protocol_sha256': digest,
                'seconds': time.perf_counter() - overall})


def evaluate(directory):
    directory = Path(directory)
    configure()
    protocol = protocol_at(directory)
    require((directory / 'training_completed.json').is_file(), 'training incomplete')
    digest = sha(directory / 'protocol_resolved.json')
    config = protocol['f_config']
    write_json(directory / 'evaluation_started.json', {'utc': now(), 'protocol_sha256': digest})
    for order in protocol['orders']:
        for seed in protocol['seeds']:
            output = directory / f'{order}_seed{seed}'
            records, timings, error_max = [], [], 0.0
            for boundary in protocol['boundaries']:
                checkpoint = torch.load(output / f'state_t{boundary}.pt', map_location='cpu',
                                        weights_only=True)
                require(checkpoint['protocol_sha256'] == digest, 'wrong checkpoint lock')
                learner = LearnerF('F', seed, 'cuda', **model_arguments(protocol),
                                   beta=config['beta'], c_max=config['c_max'],
                                   meta_lambda=config['meta_lambda'], meta_mu=config['meta_mu'],
                                   controller_lr=config['controller_lr'],
                                   gate_mode=config['gate_mode'], fixed_c=0.0)
                learner.restore(checkpoint['states']['F'])
                before = tree_digest(learner.state())
                require(before == checkpoint['state_sha256']['F'], 'checkpoint digest')
                admitted = {row['category'] for row in protocol['updates_by_order'][order][:boundary]}
                first = True
                for row in protocol['evaluations']:
                    if row['category'] not in admitted:
                        continue
                    features = eval_features(directory, row, protocol)
                    current = features.cuda()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    scores = learner.scores(current)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - started)
                    scores = scores.cpu()
                    require(bool(torch.isfinite(scores).all()), 'nonfinite anomaly scores')
                    require(tree_digest(learner.state()) == before, 'scoring mutated state')
                    if first:
                        cpu = LearnerF('F', seed, **model_arguments(protocol),
                                       beta=config['beta'], c_max=config['c_max'],
                                       meta_lambda=config['meta_lambda'], meta_mu=config['meta_mu'],
                                       controller_lr=config['controller_lr'],
                                       gate_mode=config['gate_mode'], fixed_c=0.0)
                        cpu.restore(checkpoint['states']['F'])
                        error = float((cpu.scores(features) - scores).abs().max())
                        require(error < 5e-4, 'CPU/GPU score parity')
                        error_max = max(error_max, error)
                        first = False
                    records.append({'boundary': boundary, 'arm': 'F', 'source': row['relative_path'],
                                    'patch_scores': scores.tolist(), 'image_score': float(scores.max()),
                                    'state_sha256': before})
            write_json(output / 'predictions.json', records)
            write_json(output / 'inference_resources.json', {'seconds_by_arm': {'F': timings},
                'cpu_gpu_max_patch_score_difference': error_max,
                'encoder_extraction': 'none, frozen caches'})
            print('SCORED', order, seed, len(records), flush=True)
    write_json(directory / 'evaluation_completed.json', {'utc': now(), 'protocol_sha256': digest})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'train', 'evaluate', 'analyze', 'verify'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--synthetic', action='store_true')
    arguments = parser.parse_args()
    if arguments.mode == 'prepare':
        prepare(arguments.out, arguments.synthetic)
    elif arguments.mode == 'train':
        train(arguments.out)
    elif arguments.mode == 'evaluate':
        evaluate(arguments.out)
    elif arguments.mode == 'analyze':
        from exps.task_p.p2f.analysis import analyze
        analyze(arguments.out)
    else:
        from exps.task_p.p2f.verify import verify
        verify(arguments.out)


if __name__ == '__main__':
    main()
