"""Fixed native metrics and paired whole-image inference; references never rescored."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p1_readout import native_map
from exps.task_p.p2.metrics import bootstrap_bounds, exact_ranking, image_auroc
from exps.task_p.p2e.data import (CATEGORIES, P2D, REFERENCES, configure, image_draws,
    masks_at, now, protocol_at, read, reference_artifacts, require, write_json)


def interval(lower, upper):
    return [float(np.quantile(lower, .025)), float(np.quantile(upper, .975))]


def classify(macro, canonical_pair, bottle_pair, decision):
    if macro <= decision['low_threshold']:
        band = 'LOW_RECOVERY'
    elif macro >= decision['near_B_threshold']:
        band = 'NEAR_B_RECOVERY' if canonical_pair['ci'][0] > 0 else 'UPPER_BAND_UNCERTAIN'
    else:
        band = 'PARTIAL_RECOVERY'
    bottle_pass = bottle_pair['difference'] >= .05 and bottle_pair['ci'][0] > 0
    if band == 'NEAR_B_RECOVERY' and bottle_pass:
        verdict = 'P2E_SLOW_PULL_MAJOR_COST'
    elif band == 'LOW_RECOVERY' and not bottle_pass:
        verdict = 'P2E_SLOW_PULL_NOT_MAJOR'
    elif band == 'UPPER_BAND_UNCERTAIN':
        verdict = 'P2E_INCONCLUSIVE'
    else:
        verdict = 'P2E_PARTIAL_EFFECT'
    return {'macro_band': band, 'bottle_retention_pass': bool(bottle_pass), 'verdict': verdict}


def summarize(protocol, points, samples, separation):
    arms = REFERENCES + ('E',)
    seeds, final_boundary = protocol['seeds'], protocol['boundaries'][-1]

    def category_point(order, boundary, category, arm, metric='pixel_ap'):
        return float(np.mean([points[f'{order}/{seed}/{boundary}/{category}/{arm}'][metric] for seed in seeds]))

    def category_draw(order, boundary, category, arm, metric):
        return np.mean([samples[f'{order}/{seed}/{boundary}/{category}/{arm}'][metric] for seed in seeds], axis=0)

    def macro_draw(order, arm, metric):
        return np.mean([category_draw(order, final_boundary, category, arm, metric) for category in CATEGORIES], axis=0)

    def category_pair(order, boundary, category, reference):
        return {'difference': category_point(order, boundary, category, 'E')-category_point(order, boundary, category, reference),
                'ci': interval(category_draw(order, boundary, category, 'E', 'ap_low')-category_draw(order, boundary, category, reference, 'ap_high'),
                               category_draw(order, boundary, category, 'E', 'ap_high')-category_draw(order, boundary, category, reference, 'ap_low'))}

    final, trajectories, forgetting, contrasts, per_seed = {}, {}, {}, {}, {}
    for order, categories in protocol['orders'].items():
        acquired = dict(zip(categories, protocol['boundaries'][:3]))
        final[order], contrasts[order], per_seed[order] = {}, {}, {}
        for arm in arms:
            final[order][arm] = {'pixel_ap': float(np.mean([category_point(order, final_boundary, category, arm) for category in CATEGORIES])),
                'image_auroc': float(np.mean([category_point(order, final_boundary, category, arm, 'image_auroc') for category in CATEGORIES])),
                'pixel_ap_ci': interval(macro_draw(order, arm, 'ap_low'), macro_draw(order, arm, 'ap_high')),
                'image_auroc_ci': interval(macro_draw(order, arm, 'image_auroc'), macro_draw(order, arm, 'image_auroc')),
                'categories': {category: {metric: category_point(order, final_boundary, category, arm, metric)
                                          for metric in ('pixel_ap', 'image_auroc', 'prevalence')} for category in CATEGORIES}}
            for boundary in protocol['boundaries']:
                for category in CATEGORIES:
                    if boundary < acquired[category]:
                        continue
                    key = f'{order}/{boundary}/{category}/{arm}'
                    trajectories[key] = {metric: category_point(order, boundary, category, arm, metric)
                                         for metric in ('pixel_ap', 'image_auroc')}
                    if boundary > acquired[category]:
                        forgetting[key] = {'pixel_ap_loss': category_point(order, acquired[category], category, arm)-category_point(order, boundary, category, arm),
                            'ci': interval(category_draw(order, acquired[category], category, arm, 'ap_low')-category_draw(order, boundary, category, arm, 'ap_high'),
                                           category_draw(order, acquired[category], category, arm, 'ap_high')-category_draw(order, boundary, category, arm, 'ap_low'))}
        for reference in REFERENCES:
            contrasts[order]['E_minus_'+reference] = {'difference': final[order]['E']['pixel_ap']-final[order][reference]['pixel_ap'],
                'ci': interval(macro_draw(order, 'E', 'ap_low')-macro_draw(order, reference, 'ap_high'),
                               macro_draw(order, 'E', 'ap_high')-macro_draw(order, reference, 'ap_low'))}
        contrasts[order]['boundary_categories'] = {
            f'{boundary}/{category}/E_minus_{reference}': category_pair(order, boundary, category, reference)
            for boundary in protocol['boundaries'] for category in CATEGORIES if boundary >= acquired[category]
            for reference in REFERENCES}
        for seed in seeds:
            per_seed[order][str(seed)] = {arm: {'pixel_ap': float(np.mean([points[f'{order}/{seed}/{final_boundary}/{category}/{arm}']['pixel_ap'] for category in CATEGORIES])),
                'image_auroc': float(np.mean([points[f'{order}/{seed}/{final_boundary}/{category}/{arm}']['image_auroc'] for category in CATEGORIES])),
                'bottle_retention_ap': points[f'{order}/{seed}/{protocol["retention_boundary"]}/bottle/{arm}']['pixel_ap']}
                for arm in arms}
    retention = contrasts['canonical']['boundary_categories'][f'{protocol["retention_boundary"]}/bottle/E_minus_C2']
    decision = classify(final['canonical']['E']['pixel_ap'], contrasts['canonical']['E_minus_C2'], retention, protocol['decision'])
    signal = {order: {arm: sum(value['image_auroc'] >= .7 and value['pixel_ap'] >= max(3*value['prevalence'], value['prevalence']+.05)
                               for value in final[order][arm]['categories'].values()) >= 2 for arm in arms} for order in protocol['orders']}
    return {'verdict': decision['verdict'], 'exposure': protocol['exposure'], 'decision': decision,
            'final': final, 'contrasts': contrasts, 'bottle_retention_primary': retention,
            'trajectories': trajectories, 'forgetting': forgetting, 'per_seed': per_seed,
            'useful_signal_flags': signal, 'max_bootstrap_bound_width': float(max(np.max(np.array(row['ap_high'])-np.array(row['ap_low'])) for row in samples.values())),
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
        separation = {key: value for key, value in read(P2D / 'score_separation.json').items() if key.split('/')[-1] in REFERENCES}
    for order in protocol['orders']:
        for seed in protocol['seeds']:
            records = read(directory / f'{order}_seed{seed}/predictions.json')
            if protocol['synthetic']:
                generator = np.random.default_rng(seed)
                synthetic_reference = []
                for row in records:
                    for arm in REFERENCES:
                        scores = (np.array(row['patch_scores']) + generator.uniform(0, .2, len(row['patch_scores']))).astype(np.float32)
                        synthetic_reference.append(dict(row, arm=arm, patch_scores=scores.tolist(), image_score=float(scores.max())))
                records += synthetic_reference
            for boundary in protocol['boundaries']:
                for category in CATEGORIES:
                    expected = [row for row in protocol['evaluations'] if row['category'] == category]
                    for arm in REFERENCES+('E',) if protocol['synthetic'] else ('E',):
                        group = [row for row in records if row['boundary'] == boundary and row['arm'] == arm and sources[row['source']]['category'] == category]
                        if not group:
                            continue
                        require([row['source'] for row in group] == [row['relative_path'] for row in expected], 'image pairing order')
                        maps = [torch.from_numpy(native_map(torch.tensor(row['patch_scores'], dtype=torch.float32), sources[row['source']]['hw'])).cuda() for row in group]
                        targets = [torch.from_numpy(masks[row['source']]).cuda() for row in group]
                        labels = [sources[row['source']]['label'] for row in group]
                        scores = [row['image_score'] for row in group]
                        metric = exact_ranking(torch.cat([value.flatten() for value in maps]), torch.cat([value.flatten() for value in targets]))
                        metric['image_auroc'] = float(image_auroc(labels, scores, np.ones(len(group)))[0])
                        key = f'{order}/{seed}/{boundary}/{category}/{arm}'
                        points[key] = metric
                        low, high = bootstrap_bounds(maps, targets, draws[category])
                        samples[key] = {'ap_low': low.tolist(), 'ap_high': high.tolist(),
                                        'image_auroc': image_auroc(labels, scores, draws[category]).tolist()}
                        normal = np.array(scores)[np.array(labels) == 0]
                        anomalous = np.array(scores)[np.array(labels) == 1]
                        separation[key] = {'normal_image_mean': float(normal.mean()), 'normal_image_p95': float(np.quantile(normal, .95)),
                                           'anomaly_image_mean': float(anomalous.mean())}
                        del maps, targets
            print('ANALYZED', order, seed, round(time.perf_counter()-started, 2), flush=True)
    write_json(directory / 'point_metrics.json', points)
    write_json(directory / 'bootstrap_samples.json', samples)
    write_json(directory / 'score_separation.json', separation)
    summary = summarize(protocol, points, samples, separation)
    summary['analysis_seconds'] = time.perf_counter()-started
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'analysis_completed.json', {'utc': now(), 'verdict': summary['verdict']})
    print('DECISION', summary['decision'], 'E canonical', summary['final']['canonical']['E'], flush=True)
