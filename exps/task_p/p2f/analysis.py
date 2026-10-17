"""Fixed P2f native metrics, F-versus-reference summaries and paired inference."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p1_readout import native_map
from exps.task_p.p2.metrics import bootstrap_bounds, exact_ranking, image_auroc
from exps.task_p.p2f.data import (CATEGORIES, P2E, REFERENCES, configure, image_draws,
    masks_at, now, protocol_at, read, reference_artifacts, require, write_json)

ARMS = ('E', 'B', 'C2', 'D2', 'F')


def interval(lower, upper):
    return [float(np.quantile(lower, .025)), float(np.quantile(upper, .975))]


def classify_f(primary, acquisition_safety, macro_safety, decision):
    point = float(primary['difference'])
    lower = float(primary['ci'][0])
    target = float(decision['primary_target'])
    acq_margin = float(decision['acquisition_margin'])
    macro_margin = float(decision['macro_margin'])
    harm_margin = float(decision['harm_margin'])
    acq_pass = all(item['delta'] >= -acq_margin for item in acquisition_safety.values())
    macro_pass = macro_safety['delta'] >= -macro_margin
    harm = (point <= harm_margin or macro_safety['delta'] <= harm_margin or
            any(item['delta'] < -acq_margin for item in acquisition_safety.values()))
    if harm:
        verdict = 'P2F_HARM'
    elif point >= target and lower > 0 and acq_pass and macro_pass:
        verdict = 'P2F_LEARNED_CONSOLIDATION_SUPPORTED'
    elif point >= target:
        verdict = 'P2F_PARTIAL'
    else:
        verdict = 'P2F_NO_BENEFIT'
    return {'verdict': verdict, 'retention_point': point, 'retention_lower': lower,
            'retention_target': target, 'acquisition_pass': bool(acq_pass),
            'macro_pass': bool(macro_pass), 'macro_delta': float(macro_safety['delta']),
            'harm': bool(harm)}


def _gate_evolution(directory, protocol):
    runs = {}
    for order in protocol['orders']:
        for seed in protocol['seeds']:
            trace = read(Path(directory) / f'{order}_seed{seed}/trace.json')
            gate_max = np.array([row['gate_max'] for row in trace])
            gate1 = np.array([row['gate1_mean'] for row in trace])
            gate2 = np.array([row['gate2_mean'] for row in trace])
            slow_fast = np.array([row['slow_fast_distance'] for row in trace])
            slow_init = np.array([row['slow_from_initial_distance'] for row in trace])
            runs[f'{order}/{seed}'] = {
                'cycles': len(trace),
                'gate_max_final': float(gate_max[-1]),
                'gate_max_max': float(gate_max.max()),
                'gate_nonzero_cycles': int((gate_max > 0).sum()),
                'gate1_mean_mean': float(gate1.mean()),
                'gate2_mean_mean': float(gate2.mean()),
                'gate1_mean_final': float(gate1[-1]),
                'gate2_mean_final': float(gate2[-1]),
                'gate_gradient_norm_mean': float(np.mean([row['gate_gradient_norm'] for row in trace])),
                'gate_gradient_norm_max': float(np.max([row['gate_gradient_norm'] for row in trace])),
                'controller_gradient_norm_mean': float(np.mean([row['controller_gradient_norm'] for row in trace])),
                'controller_hidden_gradient_norm_mean': float(np.mean([row['controller_hidden_gradient_norm'] for row in trace])),
                'slow_fast_distance_final': float(slow_fast[-1]),
                'slow_fast_distance_max': float(slow_fast.max()),
                'slow_from_initial_final': float(slow_init[-1]),
                'meta_acquisition_final': float(trace[-1]['meta_acquisition']),
                'meta_retention_final': float(trace[-1]['meta_retention']),
                'meta_plasticity_final': float(trace[-1]['meta_plasticity']),
                'gate_min_global': float(min(row['gate_min'] for row in trace)),
                'gate_max_global': float(max(row['gate_max'] for row in trace)),
            }
    return runs


def summarize_f(protocol, points, samples, separation, directory):
    seeds, final_boundary = protocol['seeds'], protocol['boundaries'][-1]
    category_arms = ARMS

    def category_point(order, boundary, category, arm, metric='pixel_ap'):
        return float(np.mean([points[f'{order}/{seed}/{boundary}/{category}/{arm}'][metric]
                              for seed in seeds]))

    def category_draw(order, boundary, category, arm, metric):
        return np.mean([samples[f'{order}/{seed}/{boundary}/{category}/{arm}'][metric]
                        for seed in seeds], axis=0)

    def macro_draw(order, arm, metric, boundary=final_boundary):
        return np.mean([category_draw(order, boundary, category, arm, metric)
                        for category in CATEGORIES], axis=0)

    def category_pair(order, boundary, category, reference):
        return {'difference': category_point(order, boundary, category, 'F') -
                              category_point(order, boundary, category, reference),
                'ci': interval(category_draw(order, boundary, category, 'F', 'ap_low') -
                               category_draw(order, boundary, category, reference, 'ap_high'),
                               category_draw(order, boundary, category, 'F', 'ap_high') -
                               category_draw(order, boundary, category, reference, 'ap_low'))}

    final, trajectories, forgetting, contrasts, per_seed = {}, {}, {}, {}, {}
    for order, categories in protocol['orders'].items():
        acquired = dict(zip(categories, protocol['boundaries'][:3]))
        final[order], contrasts[order], per_seed[order] = {}, {}, {}
        for arm in category_arms:
            final[order][arm] = {
                'pixel_ap': float(np.mean([category_point(order, final_boundary, category, arm)
                                           for category in CATEGORIES])),
                'image_auroc': float(np.mean([category_point(order, final_boundary, category, arm, 'image_auroc')
                                              for category in CATEGORIES])),
                'pixel_ap_ci': interval(macro_draw(order, arm, 'ap_low'),
                                        macro_draw(order, arm, 'ap_high')),
                'image_auroc_ci': interval(macro_draw(order, arm, 'image_auroc'),
                                           macro_draw(order, arm, 'image_auroc')),
                'categories': {category: {metric: category_point(order, final_boundary, category, arm, metric)
                                          for metric in ('pixel_ap', 'image_auroc', 'prevalence')}
                               for category in CATEGORIES}}
            for boundary in protocol['boundaries']:
                for category in CATEGORIES:
                    if boundary < acquired[category]:
                        continue
                    key = f'{order}/{boundary}/{category}/{arm}'
                    trajectories[key] = {metric: category_point(order, boundary, category, arm, metric)
                                         for metric in ('pixel_ap', 'image_auroc')}
                    if boundary > acquired[category]:
                        forgetting[key] = {
                            'pixel_ap_loss': category_point(order, acquired[category], category, arm) -
                                             category_point(order, boundary, category, arm),
                            'ci': interval(category_draw(order, acquired[category], category, arm, 'ap_low') -
                                           category_draw(order, boundary, category, arm, 'ap_high'),
                                           category_draw(order, acquired[category], category, arm, 'ap_high') -
                                           category_draw(order, boundary, category, arm, 'ap_low'))}
        for reference in REFERENCES:
            contrasts[order]['F_minus_' + reference] = {
                'difference': final[order]['F']['pixel_ap'] - final[order][reference]['pixel_ap'],
                'ci': interval(macro_draw(order, 'F', 'ap_low') - macro_draw(order, reference, 'ap_high'),
                               macro_draw(order, 'F', 'ap_high') - macro_draw(order, reference, 'ap_low'))}
        contrasts[order]['boundary_categories'] = {
            f'{boundary}/{category}/F_minus_{reference}': category_pair(order, boundary, category, reference)
            for boundary in protocol['boundaries'] for category in CATEGORIES
            if boundary >= acquired[category] for reference in REFERENCES}
        for seed in seeds:
            per_seed[order][str(seed)] = {
                arm: {'pixel_ap': float(np.mean([points[f'{order}/{seed}/{final_boundary}/{category}/{arm}']['pixel_ap']
                                                for category in CATEGORIES])),
                      'image_auroc': float(np.mean([points[f'{order}/{seed}/{final_boundary}/{category}/{arm}']['image_auroc']
                                                    for category in CATEGORIES])),
                      'bottle_retention_ap': points[f'{order}/{seed}/{protocol["retention_boundary"]}/bottle/{arm}']['pixel_ap']}
                for arm in category_arms}
    retention = contrasts['canonical']['boundary_categories'][f'{protocol["retention_boundary"]}/bottle/F_minus_E']
    acquisition_safety = {}
    for order, categories in protocol['orders'].items():
        acquired = dict(zip(categories, protocol['boundaries'][:3]))
        for category in CATEGORIES:
            boundary = acquired[category]
            e_value = category_point(order, boundary, category, 'E')
            f_value = category_point(order, boundary, category, 'F')
            acquisition_safety[f'{order}/{category}'] = {
                'boundary': boundary, 'E': e_value, 'F': f_value, 'delta': f_value - e_value,
                'pass': bool(f_value - e_value >= -protocol['decision']['acquisition_margin'])}
    macro_safety = {'E': final['canonical']['E']['pixel_ap'],
                    'F': final['canonical']['F']['pixel_ap'],
                    'delta': final['canonical']['F']['pixel_ap'] - final['canonical']['E']['pixel_ap']}
    decision = classify_f(retention, acquisition_safety, macro_safety, protocol['decision'])
    signal = {order: {arm: sum(value['image_auroc'] >= .7 and
                               value['pixel_ap'] >= max(3 * value['prevalence'], value['prevalence'] + .05)
                               for value in final[order][arm]['categories'].values()) >= 2
                      for arm in category_arms} for order in protocol['orders']}
    evolution = _gate_evolution(directory, protocol)
    return {'verdict': decision['verdict'], 'decision': decision,
            'primary': retention, 'acquisition_safety': acquisition_safety,
            'macro_safety': macro_safety, 'final': final, 'contrasts': contrasts,
            'bottle_retention_primary': retention, 'trajectories': trajectories,
            'forgetting': forgetting, 'per_seed': per_seed,
            'useful_signal_flags': signal,
            'gate_evolution': evolution,
            'max_bootstrap_bound_width': float(max(
                np.max(np.array(row['ap_high']) - np.array(row['ap_low'])) for row in samples.values())),
            'metric_groups': len(points), 'separation_groups': len(separation)}


def analyze(directory):
    directory = Path(directory)
    configure()
    protocol = protocol_at(directory)
    require((directory / 'evaluation_completed.json').is_file(), 'evaluation incomplete')
    write_json(directory / 'analysis_started.json', {'utc': now()})
    started = time.perf_counter()
    masks, draws = masks_at(directory, protocol), image_draws(protocol)
    sources = {row['relative_path']: row for row in protocol['evaluations']}
    if protocol['synthetic']:
        points, samples, separation = {}, {}, {}
    else:
        points, samples = reference_artifacts(protocol)
        separation = {key: value for key, value in read(P2E / 'score_separation.json').items()
                      if key.split('/')[-1] in REFERENCES}
    for order in protocol['orders']:
        for seed in protocol['seeds']:
            records = read(directory / f'{order}_seed{seed}/predictions.json')
            if protocol['synthetic']:
                generator = np.random.default_rng(seed)
                synthetic_reference = []
                for row in records:
                    for arm in REFERENCES:
                        scores = (np.array(row['patch_scores']) +
                                  generator.uniform(0, .2, len(row['patch_scores']))).astype(np.float32)
                        synthetic_reference.append(dict(row, arm=arm, patch_scores=scores.tolist(),
                                                       image_score=float(scores.max())))
                records += synthetic_reference
            for boundary in protocol['boundaries']:
                for category in CATEGORIES:
                    expected = [row for row in protocol['evaluations'] if row['category'] == category]
                    for arm in (ARMS if protocol['synthetic'] else ('F',)):
                        group = [row for row in records if row['boundary'] == boundary and
                                 row['arm'] == arm and sources[row['source']]['category'] == category]
                        if not group:
                            continue
                        require([row['source'] for row in group] == [row['relative_path'] for row in expected],
                                'image pairing order')
                        maps = [torch.from_numpy(native_map(torch.tensor(row['patch_scores'], dtype=torch.float32),
                                                            sources[row['source']]['hw'])).cuda()
                                for row in group]
                        targets = [torch.from_numpy(masks[row['source']]).cuda() for row in group]
                        labels = [sources[row['source']]['label'] for row in group]
                        scores = [row['image_score'] for row in group]
                        metric = exact_ranking(torch.cat([value.flatten() for value in maps]),
                                               torch.cat([value.flatten() for value in targets]))
                        metric['image_auroc'] = float(image_auroc(labels, scores, np.ones(len(group)))[0])
                        key = f'{order}/{seed}/{boundary}/{category}/{arm}'
                        points[key] = metric
                        low, high = bootstrap_bounds(maps, targets, draws[category])
                        samples[key] = {'ap_low': low.tolist(), 'ap_high': high.tolist(),
                                        'image_auroc': image_auroc(labels, scores, draws[category]).tolist()}
                        normal = np.array(scores)[np.array(labels) == 0]
                        anomalous = np.array(scores)[np.array(labels) == 1]
                        separation[key] = {'normal_image_mean': float(normal.mean()),
                                           'normal_image_p95': float(np.quantile(normal, .95)),
                                           'anomaly_image_mean': float(anomalous.mean())}
                        del maps, targets
            print('ANALYZED', order, seed, round(time.perf_counter() - started, 2), flush=True)
    write_json(directory / 'point_metrics.json', points)
    write_json(directory / 'bootstrap_samples.json', samples)
    write_json(directory / 'score_separation.json', separation)
    summary = summarize_f(protocol, points, samples, separation, directory)
    summary['analysis_seconds'] = time.perf_counter() - started
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'gate_evolution.json', summary['gate_evolution'])
    write_json(directory / 'analysis_completed.json', {'utc': now(), 'verdict': summary['verdict']})
    print('DECISION', summary['decision'], 'F canonical', summary['final']['canonical']['F'], flush=True)
