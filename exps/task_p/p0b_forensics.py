"""P0b read-only forensics for the Task P / P0 normal-support experiment.

READ-ONLY by construction: this module never mutates P0 artifacts, never trains,
never changes a threshold, and never reads defect/test/ground-truth data. All
inputs are saved P0 artifacts (predictions, traces, states, protocol) plus the
already-authorized frozen clean caches of the three P0 categories, re-read here
only as an offline evaluator at/after the final boundary t=350.

Sections are separate CLI commands so each can be audited independently:
    integrity, readout, compression, representation, witness

Registered readout (unchanged from P0):
    U(i, t) = mean of the 7 largest of 64 probe-patch distances, divided by the
    category training-only scale for cross-category aggregation.
Distances use exps.task_p.p0_normal_support.distances (the frozen P0 scorer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.task_p.p0_normal_support import (  # noqa: E402
    distances, fingerprint, nearest_two, require, Support,
)

ROOT = Path('/mnt/d/code_for_fun/project/nestedlearningforcad/nestedlearningforcad')
P0 = ROOT / 'results/hope_cad/task_p/p0'
OUT = ROOT / 'results/hope_cad/task_p/p0b'
FEATURES = ROOT / 'results/hope_cad/frozen_features_v1'
ARMS = ('FROZEN_FIRST', 'RESERVOIR', 'STREAM_GREEDY', 'WITNESS_PROTECTED')
CATEGORIES = ('bottle', 'carpet', 'hazelnut')
TRAIN_ROWS = (3, 10, 17, 24)
PROBE_ROWS = (1, 4, 8, 11, 15, 18, 22, 26)
TRAIN_GRID = tuple(r * 28 + c for r in TRAIN_ROWS for c in TRAIN_ROWS)
PROBE_GRID = tuple(r * 28 + c for r in PROBE_ROWS for c in PROBE_ROWS)
TAIL_K = 7
CHUNK = 256

ACCESS: list[dict] = []


def log(message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
    log(f'wrote {path.relative_to(ROOT)}')


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def position_offsets(index: int) -> dict:
    """Offset of probe position `index` to the nearest training-grid position."""
    row, column = PROBE_GRID[index] // 28, PROBE_GRID[index] % 28
    dr = min(abs(row - value) for value in TRAIN_ROWS)
    dc = min(abs(column - value) for value in TRAIN_ROWS)
    return {'row': row, 'column': column, 'offset_row': dr, 'offset_col': dc,
            'chebyshev': max(dr, dc), 'manhattan': dr + dc}


def load_cache(category: str, caches: dict):
    if category not in caches:
        path = FEATURES / f'clean/{category}_train.pt'
        ACCESS.append({'role': 'p0b_frozen_clean_cache', 'path': str(path.relative_to(ROOT)),
                       'bytes_sha256': sha256_file(path), 'scope': 'read-only offline evaluator'})
        caches[category] = torch.load(path, mmap=True, weights_only=True)
    return caches[category]


def tail_mean(values: torch.Tensor) -> float:
    return float(values.topk(TAIL_K).values.mean())


# ---------------------------------------------------------------------------------------
# 1. integrity and reproducibility audit
# ---------------------------------------------------------------------------------------

def section_integrity() -> None:
    manifest = load_json(P0 / 'artifact_integrity.json')
    checked, mismatched = {}, {}
    for relative, record in manifest['files_excluding_this_manifest'].items():
        path = ROOT / relative
        actual = sha256_file(path)
        checked[relative] = {'expected': record['sha256'], 'actual': actual, 'match': actual == record['sha256']}
        if actual != record['sha256']:
            mismatched[relative] = checked[relative]
    protocol_sha = sha256_file(P0 / 'protocol_resolved.json')
    pinned = (P0 / 'protocol.sha256').read_text().strip()

    records = load_json(P0 / 'per_image_results.json')
    summary = load_json(P0 / 'summary.json')
    boundaries = load_json(P0 / 'boundaries.json')
    final = [r for r in records if r['event'] == 350]

    # internal consistency: recompute U from the saved 64 distances and compare with the saved tail
    max_tail_delta = 0.0
    for record in records:
        for arm in ARMS:
            values = torch.tensor(record['arms'][arm]['distances'], dtype=torch.float64)
            max_tail_delta = max(max_tail_delta, abs(tail_mean(values) - record['arms'][arm]['tail_raw']))

    recomputed = {category: {} for category in CATEGORIES}
    for category in CATEGORIES:
        rows = [r for r in final if r['category'] == category]
        for arm in ARMS:
            recomputed[category][arm] = float(np.mean([r['arms'][arm]['tail_normalized'] for r in rows]))
    macro = {arm: float(np.mean([recomputed[category][arm] for category in CATEGORIES])) for arm in ARMS}
    recorded_macro = {arm: summary['macro'][arm]['normalized_risk'] for arm in ARMS}

    draws = {arm: {category: np.array([r['arms'][arm]['tail_normalized'] for r in final
                                       if r['category'] == category]) for category in CATEGORIES}
             for arm in ARMS}
    generator = np.random.default_rng(7201)
    replicates = np.empty(10000)
    for replicate in range(10000):
        d_mean, c_mean = [], []
        for category in CATEGORIES:
            index = generator.integers(0, 20, size=20)
            d_mean.append(draws['WITNESS_PROTECTED'][category][index].mean())
            c_mean.append(draws['STREAM_GREEDY'][category][index].mean())
        replicates[replicate] = 1 - float(np.mean(d_mean)) / float(np.mean(c_mean))
    ci = [float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))]
    primary = 1 - macro['WITNESS_PROTECTED'] / macro['STREAM_GREEDY']

    access = load_json(P0 / 'access_log.json')
    illegal = [row for row in access if ('test' in row['path'] or 'ground_truth' in row['path']
                                        or row['path'].split('/')[0] not in CATEGORIES)]

    result = {
        'artifact_manifest_files_checked': len(checked),
        'artifact_manifest_mismatches': mismatched,
        'protocol_sha256': {'recorded': pinned, 'actual': protocol_sha, 'match': protocol_sha == pinned},
        'all_expected_shas_pinned_in_protocol': {
            name: {'pinned': pinned_sha, 'actual': sha256_file(FEATURES / name),
                   'match': sha256_file(FEATURES / name) == pinned_sha}
            for name, pinned_sha in load_json(P0 / 'protocol_resolved.json')['cache_sha256'].items()},
        'checkpoint_sha256_match': sha256_file(ROOT / 'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth')
        == load_json(P0 / 'protocol_resolved.json')['checkpoint_sha256'],
        'max_abs_delta_recomputed_tail_vs_recorded': max_tail_delta,
        'recomputed_macro_from_saved_distances': macro,
        'reported_macro': recorded_macro,
        'macro_recomputation_max_abs_delta': max(abs(macro[arm] - recorded_macro[arm]) for arm in ARMS),
        'recomputed_primary_benefit': primary,
        'reported_primary_benefit': summary['primary_relative_gain'],
        'recomputed_primary_ci95': ci,
        'reported_primary_ci95': summary['primary_gain_ci95'],
        'recorded_gates': summary['gates'],
        'access_log_entries': len(access),
        'access_log_illegal_entries': illegal,
        'boundary_events': [b['event'] for b in boundaries],
        'probe_records': len(records),
        'inputs_read_by_this_audit': ACCESS,
        'attestation': ('read-only audit; no learner update, no probe-driven selection, no defect/test/'
                        'ground-truth/sealed access, no artifact mutation'),
    }
    save('p0b_integrity_audit.json', result)


# ---------------------------------------------------------------------------------------
# 2. readout relevance / masking analysis
# ---------------------------------------------------------------------------------------

def section_readout() -> None:
    records = load_json(P0 / 'per_image_results.json')
    final = [r for r in records if r['event'] == 350]
    per_arm: dict = {}
    position_stats: dict = {}

    for event in (100, 200, 300, 350):
        subset = [r for r in records if r['event'] == event]
        counts_by_category = Counter(r['category'] for r in subset)
        for arm in ARMS:
            aggregate = defaultdict(list)
            top7_counts = defaultdict(Counter)
            for record in subset:
                category = record['category']
                values = torch.tensor(record['arms'][arm]['distances'], dtype=torch.float64)
                sorted_desc = torch.sort(values, descending=True).values
                top7, rest = sorted_desc[:TAIL_K], sorted_desc[TAIL_K:]
                aggregate[category].append({
                    'u': float(top7.mean()), 'bulk_mean': float(rest.mean()),
                    'eighth': float(sorted_desc[TAIL_K]), 'maximum': float(sorted_desc[0]),
                    'u_over_bulk': float(top7.mean() / rest.mean()),
                    'slack_to_eighth': float(top7.mean() - sorted_desc[TAIL_K]),
                    'tail_share_of_mass': float(top7.sum() / values.sum()),
                    'eighth_over_u': float(sorted_desc[TAIL_K] / top7.mean()),
                })
                for index in torch.topk(values, TAIL_K).indices.tolist():
                    top7_counts[category][index] += 1
            key = f'{arm}@{event}'
            per_arm[key] = {}
            for category, rows in aggregate.items():
                per_arm[key][category] = {
                    'images': len(rows),
                    'u_mean_raw': float(np.mean([r['u'] for r in rows])),
                    'u_mean_over_bulk_mean': float(np.mean([r['u_over_bulk'] for r in rows])),
                    'eighth_over_u_mean': float(np.mean([r['eighth_over_u'] for r in rows])),
                    'slack_to_eighth_mean': float(np.mean([r['slack_to_eighth'] for r in rows])),
                    'bulk_mean': float(np.mean([r['bulk_mean'] for r in rows])),
                    'tail_share_of_distance_mass': float(np.mean([r['tail_share_of_mass'] for r in rows])),
                    'u_min': float(min(r['u'] for r in rows)), 'u_max': float(max(r['u'] for r in rows)),
                }
            if event == 350:
                for category, counts in top7_counts.items():
                    images = counts_by_category[category]
                    ordered = sorted(counts.items(), key=lambda item: -item[1])
                    groups: dict = defaultdict(lambda: {'positions': 0, 'appearances': 0})
                    for index in range(64):
                        group = groups[f"chebyshev_{position_offsets(index)['chebyshev']}"]
                        group['positions'] += 1
                        group['appearances'] += counts.get(index, 0)
                    total = sum(counts.values())
                    position_stats.setdefault(category, {})[arm] = {
                        'images': images,
                        'distinct_positions_ever_in_top7': len(counts),
                        'positions_scored_64': 64,
                        'positions_in_top7_for_all_images': sum(1 for _, c in counts.items() if c == images),
                        'positions_in_top7_for_majority': sum(1 for _, c in counts.items() if c > images / 2),
                        'top_positions': [
                            {'position_index': index, **position_offsets(index), 'top7_appearances': count,
                             'appearance_rate': count / images} for index, count in ordered[:12]],
                        'offset_group_appearance': {
                            name: {**value, 'share_of_top7': value['appearances'] / max(1, total),
                                   'appearance_rate_per_position': value['appearances'] / max(1, value['positions'])}
                            for name, value in sorted(groups.items())},
                    }

    equality = []
    for record in [r for r in records if r['event'] == 350]:
        left = torch.tensor(record['arms']['STREAM_GREEDY']['distances'], dtype=torch.float64)
        right = torch.tensor(record['arms']['WITNESS_PROTECTED']['distances'], dtype=torch.float64)
        left_top = torch.topk(left, TAIL_K).indices.tolist()
        right_top = torch.topk(right, TAIL_K).indices.tolist()
        equality.append({
            'category': record['category'], 'path': record['path'],
            'all_64_distances_bitwise_equal': bool(torch.equal(left, right)),
            'n_distances_exactly_equal': int((left == right).sum()),
            'top7_position_sets_equal': sorted(left_top) == sorted(right_top),
            'top7_distance_multiset_equal': bool(torch.equal(torch.sort(left[left_top]).values,
                                                             torch.sort(right[right_top]).values)),
            'u_equal': tail_mean(left) == tail_mean(right),
            'u_difference_c_minus_d': tail_mean(left) - tail_mean(right),
            'nearest_serial_agreement_all64': float(np.mean([
                a == b for a, b in zip(record['arms']['STREAM_GREEDY']['nearest_serials'],
                                       record['arms']['WITNESS_PROTECTED']['nearest_serials'])])),
            'nearest_serial_agreement_top7': float(np.mean([
                record['arms']['STREAM_GREEDY']['nearest_serials'][index]
                == record['arms']['WITNESS_PROTECTED']['nearest_serials'][index]
                for index in sorted(set(left_top) | set(right_top))])),
        })
    per_category_equality = {}
    for category in CATEGORIES:
        rows = [r for r in equality if r['category'] == category]
        per_category_equality[category] = {
            'images': len(rows),
            'images_with_identical_u': sum(1 for r in rows if r['u_equal']),
            'images_with_identical_64_distances': sum(1 for r in rows if r['all_64_distances_bitwise_equal']),
            'images_with_identical_top7_position_set': sum(1 for r in rows if r['top7_position_sets_equal']),
            'images_with_identical_top7_multiset': sum(1 for r in rows if r['top7_distance_multiset_equal']),
            'mean_fraction_of_64_distances_equal': float(np.mean([r['n_distances_exactly_equal'] / 64 for r in rows])),
            'mean_u_difference_c_minus_d': float(np.mean([r['u_difference_c_minus_d'] for r in rows])),
            'mean_nearest_serial_agreement_all64': float(np.mean([r['nearest_serial_agreement_all64'] for r in rows])),
            'mean_nearest_serial_agreement_top7': float(np.mean([r['nearest_serial_agreement_top7'] for r in rows])),
        }

    # C versus D: how large are memory-induced distance changes at score-relevant vs masked positions?
    masking = []
    for record in [r for r in records if r['event'] == 350]:
        left = torch.tensor(record['arms']['STREAM_GREEDY']['distances'], dtype=torch.float64)
        right = torch.tensor(record['arms']['WITNESS_PROTECTED']['distances'], dtype=torch.float64)
        critical = set(torch.topk(left, TAIL_K).indices.tolist()) | set(torch.topk(right, TAIL_K).indices.tolist())
        delta = (left - right).abs()
        masked_positions = [i for i in range(64) if i not in critical]
        masking.append({
            'category': record['category'], 'path': record['path'],
            'critical_positions': len(critical), 'masked_positions': len(masked_positions),
            'max_abs_delta_masked_positions': float(delta[masked_positions].max()),
            'mean_abs_delta_masked_positions': float(delta[masked_positions].mean()),
            'max_abs_delta_critical_positions': float(delta[sorted(critical)].max()),
            'max_abs_delta_all_positions': float(delta.max()),
            'u_delta': abs(tail_mean(left) - tail_mean(right)),
            'n_masked_positions_changed_at_all': int((delta[masked_positions] > 0).sum()),
        })
    all_deltas = np.array([r['max_abs_delta_all_positions'] for r in masking])
    delta_distribution = {
        'images': len(masking),
        'mean_abs_change_at_masked_positions': float(np.mean([r['mean_abs_delta_masked_positions'] for r in masking])),
        'max_abs_change_per_image_max': float(all_deltas.max()),
        'max_abs_change_per_image_median': float(np.median(all_deltas)),
        'images_with_some_change_larger_than_0p5_raw': int(sum(1 for r in masking
                                                              if r['max_abs_delta_all_positions'] > 0.5)),
        'images_with_zero_change': int(sum(1 for r in masking if r['max_abs_delta_all_positions'] == 0.0)),
    }
    per_category_masking = {}
    for category in CATEGORIES:
        rows = [r for r in masking if r['category'] == category]
        per_category_masking[category] = {
            'images': len(rows),
            'max_abs_distance_change_that_could_not_affect_U': float(max(r['max_abs_delta_masked_positions'] for r in rows)),
            'mean_abs_distance_change_at_masked_positions': float(np.mean([r['mean_abs_delta_masked_positions'] for r in rows])),
            'mean_number_of_masked_positions_changed': float(np.mean([r['n_masked_positions_changed_at_all'] for r in rows])),
            'max_U_difference_observed': float(max(r['u_delta'] for r in rows)),
        }

    # distance decay across ranks 1..64 (mean over images), arm C at t=350
    decay = {}
    for category in CATEGORIES:
        rows = [r for r in records if r['event'] == 350 and r['category'] == category]
        values = torch.stack([torch.tensor(r['arms']['STREAM_GREEDY']['distances'], dtype=torch.float64)
                              for r in rows])
        sorted_desc = torch.sort(values, descending=True).values
        scale = float(np.median([r['scale'] for r in rows]))
        decay[category] = {
            'rank_means_raw': [float(sorted_desc[:, rank].mean()) for rank in range(64)],
            'rank_means_over_scale': [float(sorted_desc[:, rank].mean() / scale) for rank in range(64)],
            'rank7_over_rank64': float(sorted_desc[:, 6].mean() / sorted_desc[:, 63].mean()),
            'rank1_over_rank64': float(sorted_desc[:, 0].mean() / sorted_desc[:, 63].mean()),
        }

    # readout-definition sensitivity (diagnostic only; the registered gate is frozen and cannot change)
    alternatives = {
        'registered_top7_mean': lambda v: float(torch.sort(v, descending=True).values[:7].mean()),
        'max': lambda v: float(v.max()),
        'top3_mean': lambda v: float(torch.sort(v, descending=True).values[:3].mean()),
        'top16_mean': lambda v: float(torch.sort(v, descending=True).values[:16].mean()),
        'p90': lambda v: float(torch.quantile(v, 0.9)),
        'mean': lambda v: float(v.mean()),
        'median': lambda v: float(torch.quantile(v, 0.5)),
        'half_mean_half_max': lambda v: float(0.5 * v.mean() + 0.5 * v.max()),
    }
    sensitivity = {}
    for label, function in alternatives.items():
        per_category = {}
        for category in CATEGORIES:
            rows = [r for r in final if r['category'] == category]
            scale = float(np.median([r['scale'] for r in rows]))
            values = {arm: np.array([function(torch.tensor(r['arms'][arm]['distances'], dtype=torch.float64)) / scale
                                     for r in rows]) for arm in ARMS}
            per_category[category] = {arm: float(values[arm].mean()) for arm in ARMS}
        macro = {arm: float(np.mean([per_category[c][arm] for c in CATEGORIES])) for arm in ARMS}
        per_category_benefit = {
            category: 1 - per_category[category]['WITNESS_PROTECTED'] / per_category[category]['STREAM_GREEDY']
            for category in CATEGORIES}
        sensitivity[label] = {
            'macro': macro,
            'macro_benefit_D_vs_C': 1 - macro['WITNESS_PROTECTED'] / macro['STREAM_GREEDY'],
            'per_category_benefit_D_vs_C': per_category_benefit,
            'categories_where_D_beats_C': sum(1 for value in per_category_benefit.values() if value > 0),
        }

    result = {
        'registered_readout': 'U(i,t) = mean of the 7 largest of 64 probe-patch distances',
        'readout_surface_positions': 64,
        'score_relevant_positions_per_image': TAIL_K,
        'score_relevant_fraction': TAIL_K / 64,
        'lower_bound_property': ('a strict decrease of any distance outside the current top-7 leaves U '
                                 'unchanged; only the seven largest distances per image are actionable'),
        'per_arm_boundary_category': per_arm,
        'top7_position_statistics_final': position_stats,
        'distance_decay_by_rank_arm_C_final': decay,
        'masking_quantification_C_vs_D_final': {'per_category': per_category_masking, 'per_image': masking,
                                                'aggregate_change_distribution': delta_distribution},
        'readout_definition_sensitivity': sensitivity,
        'cross_arm_readout_equality_final': {
            'contrast': 'STREAM_GREEDY versus WITNESS_PROTECTED',
            'per_category': per_category_equality, 'per_image': equality},
    }
    save('p0b_readout_analysis.json', result)


# ---------------------------------------------------------------------------------------
# 3. compression analysis
# ---------------------------------------------------------------------------------------

def _bank_rows(state: dict) -> torch.Tensor:
    return state['bank'][:state['n']].clone().float()


def _u_over_queries(bank: torch.Tensor, query_matrix, categories_of_query, scale_of_position):
    per_category = defaultdict(list)
    for start in range(0, query_matrix.shape[0], CHUNK):
        block = query_matrix[start:start + CHUNK]
        values = distances(block, bank).min(1).values
        for index in range(block.shape[0]):
            per_category[categories_of_query[start + index]].append(float(values[index]))
    result = {}
    for category, values in per_category.items():
        tensor = torch.tensor(values, dtype=torch.float64).reshape(-1, 64)
        u = tensor.topk(TAIL_K, dim=1).values.mean(1)
        result[category] = {'u_raw': float(u.mean()),
                            'u_normalized': float((u / scale_of_position[category]).mean())}
    result['macro_normalized'] = float(np.mean([result[c]['u_normalized'] for c in CATEGORIES]))
    return result


def _probe_queries(protocol: dict, records: list, caches: dict):
    scale_by_category = {}
    for record in records:
        scale_by_category[record['category']] = record['scale']
    rows = []
    for probe in protocol['probes']:
        obj = load_cache(probe['category'], caches)
        require(obj['relative_paths'][probe['cache_index']] == probe['relative_path'],
                'cache row mismatch during audit')
        rows.append({'category': probe['category'], 'path': probe['relative_path'],
                     'cache_index': probe['cache_index'],
                     'query': obj['patches'][probe['cache_index']][list(PROBE_GRID)].clone()})
    return rows, scale_by_category


def section_compression(do_fps_full: bool) -> None:
    protocol = load_json(P0 / 'protocol_resolved.json')
    records = load_json(P0 / 'per_image_results.json')
    states = torch.load(P0 / 'states_t350.pt', map_location='cpu', weights_only=False)['models']
    caches: dict = {}
    queries, scale_by_category = _probe_queries(protocol, records, caches)
    query_matrix = torch.cat([row['query'] for row in queries]).double()
    categories_of_query = [row['category'] for row in queries for _ in range(64)]
    serial_category = {serial: protocol['updates'][serial // 16]['category']
                       for serial in range(len(protocol['updates']) * 16)}

    bank_report = {}
    for arm in ARMS:
        state = states[arm]
        rows = _bank_rows(state)
        entry = {'occupied': int(state['n']), 'capacity': int(state['capacity']),
                 'composition_by_arrival_category': dict(Counter(
                     serial_category[serial] for serial in state['ids'][:state['n']].tolist())),
                 'occupied_payload_bytes': int(rows.numel() * 4)}
        if arm != 'RESERVOIR' and 'nn_distance' in state:
            nn = state['nn_distance'][:state['n']].double()
            finite = nn[torch.isfinite(nn)]
            entry['nearest_other_distance'] = {
                'min': float(finite.min()), 'q10': float(torch.quantile(finite, 0.1)),
                'median': float(finite.median()), 'mean': float(finite.mean()),
                'q90': float(torch.quantile(finite, 0.9)), 'max': float(finite.max())}
        bank_report[arm] = entry

    functional = {}
    for arm in ('STREAM_GREEDY', 'WITNESS_PROTECTED'):
        state = states[arm]
        rows = _bank_rows(state)
        first, second, index = [], [], []
        for start in range(0, query_matrix.shape[0], CHUNK):
            matrix = distances(query_matrix[start:start + CHUNK], rows)
            values, idx = matrix.min(1)
            copy = matrix.clone()
            copy[torch.arange(copy.shape[0]), idx] = torch.inf
            first.append(values)
            second.append(copy.min(1).values)
            index.append(idx)
        first, second, index = torch.cat(first), torch.cat(second), torch.cat(index)
        gap = second - first
        critical = []
        for start in range(0, 64 * len(queries), 64):
            block = first[start:start + 64].clone()
            critical.append(torch.topk(block, TAIL_K).indices + start)
        critical = torch.cat(critical)
        nearest_serials = state['ids'][:state['n']][index]
        counts = Counter(nearest_serials[critical].tolist())
        ordered = counts.most_common()
        cumulative = np.cumsum([count for _, count in ordered]) / len(critical)
        functional[arm] = {
            'distinct_nearest_rows_all_3840_queries': int(len(set(nearest_serials.tolist()))),
            'distinct_nearest_rows_score_critical_420': int(len(counts)),
            'rows_for_50pct_of_score_critical': int(np.searchsorted(cumulative, 0.5) + 1),
            'rows_for_90pct_of_score_critical': int(np.searchsorted(cumulative, 0.9) + 1),
            'rows_for_100pct_of_score_critical': int(len(counts)),
            'swap_tolerance_d2_minus_d1': {
                'median': float(gap.median()), 'mean': float(gap.mean()),
                'q10': float(torch.quantile(gap, 0.1)), 'q90': float(torch.quantile(gap, 0.9)),
                'fraction_below_1e-6': float((gap < 1e-6).double().mean()),
                'fraction_below_0p1pct_of_d1': float((gap < 0.001 * first).double().mean())},
        }

    rows_c = _bank_rows(states['STREAM_GREEDY'])
    rows_d = _bank_rows(states['WITNESS_PROTECTED'])
    delta = []
    for start in range(0, query_matrix.shape[0], CHUNK):
        block = query_matrix[start:start + CHUNK]
        delta.append((distances(block, rows_c).min(1).values
                      - distances(block, rows_d).min(1).values).abs())
    delta = torch.cat(delta)
    equivalence = {
        'per_query_absolute_distance_difference': {
            'mean': float(delta.mean()), 'median': float(delta.median()),
            'q90': float(torch.quantile(delta, 0.9)), 'max': float(delta.max()),
            'fraction_below_1e-9': float((delta < 1e-9).double().mean())},
        'registered_macro_difference_C_minus_D': (
            _u_over_queries(rows_c.double(), query_matrix, categories_of_query, scale_by_category)['macro_normalized']
            - _u_over_queries(rows_d.double(), query_matrix, categories_of_query, scale_by_category)['macro_normalized']),
    }

    # static capacity sensitivity of the registered readout (read-only diagnostic, not an arm)
    capacity_curve: dict = {}
    sizes = [2350, 2000, 1700, 1350, 1000, 700, 400, 200, 100, 50]
    for size in sizes:
        runs = []
        for seed in range(5):
            generator = torch.Generator().manual_seed(1000 + seed)
            index = torch.randperm(rows_c.shape[0], generator=generator)[:size]
            runs.append(_u_over_queries(rows_c[index].double(), query_matrix,
                                        categories_of_query, scale_by_category)['macro_normalized'])
        capacity_curve[str(size)] = {'random_subset_mean': float(np.mean(runs)),
                                     'random_subset_min': float(np.min(runs)),
                                     'random_subset_max': float(np.max(runs)),
                                     'fraction_of_final_bank': size / rows_c.shape[0]}
    if do_fps_full:
        _fps_full_grid(protocol, caches, query_matrix, categories_of_query, scale_by_category,
                       target=int(rows_c.shape[0]))

    selected = [0]
    min_gap = distances(rows_c[0:1].double(), rows_c.double())[0].clone()
    min_gap[0] = torch.inf
    for step in range(1, rows_c.shape[0]):
        nxt = int(torch.argmax(min_gap))
        selected.append(nxt)
        min_gap[nxt] = torch.inf
        row = rows_c[nxt:nxt + 1].double()
        min_gap = torch.minimum(min_gap, distances(row, rows_c.double())[0])
        size = len(selected)
        if str(size) in capacity_curve:
            capacity_curve[str(size)]['farthest_point_subset'] = _u_over_queries(
                rows_c[selected].double(), query_matrix, categories_of_query,
                scale_by_category)['macro_normalized']
            log(f'fps bank size {size} evaluated')

    result = {
        'bank_report': bank_report,
        'functional_bank_size_final': functional,
        'c_versus_d_functional_equivalence': equivalence,
        'capacity_curve_registered_readout': {
            'full_bank_2702': _u_over_queries(rows_c.double(), query_matrix, categories_of_query,
                                             scale_by_category)['macro_normalized'],
            **capacity_curve},
        'notes': ('static read-only diagnostic on frozen final banks; subset selection never uses probe '
                  'information; not a P0 arm and cannot change the P0 gate'),
    }
    save('p0b_compression_analysis.json', result)


def _fps_full_grid(protocol, caches, query_matrix, categories_of_query, scale_by_category, target: int) -> None:
    """Audit-only diagnostic: how far does an 8-MiB bank drawn from ALL 784 positions get?"""
    parts = []
    for event in range(1, len(protocol['updates']) + 1):
        row = protocol['updates'][event - 1]
        obj = load_cache(row['category'], caches)
        parts.append(obj['patches'][row['cache_index']].clone())
    full = torch.cat(parts).float()
    log(f'full-grid bank for FPS: {tuple(full.shape)}')
    generator = torch.Generator().manual_seed(7)
    start = int(torch.randint(full.shape[0], (1,), generator=generator))
    norms = full.square().sum(1)
    min_gap = (norms + norms[start] - 2 * (full @ full[start])).clamp_min_(0).sqrt_()
    min_gap[start] = torch.inf
    selected = [start]
    started = time.perf_counter()
    while len(selected) < target:
        nxt = int(torch.argmax(min_gap))
        selected.append(nxt)
        min_gap[nxt] = torch.inf
        new = (norms + norms[nxt] - 2 * (full @ full[nxt])).clamp_min_(0).sqrt_()
        min_gap = torch.minimum(min_gap, new)
        if len(selected) % 500 == 0:
            log(f'full-grid FPS {len(selected)}/{target} ({time.perf_counter() - started:.0f}s)')
        if time.perf_counter() - started > 900:
            log('full-grid FPS time-limited')
            return
    subset = full[torch.tensor(selected)].double()
    value = _u_over_queries(subset, query_matrix, categories_of_query, scale_by_category)
    composition_image = Counter((index // 784) for index in selected)
    order = [protocol['updates'][image]['category'] for image in range(350)]
    composition_category = Counter(order[image] for image in composition_image)
    positions = Counter(index % 784 for index in selected)
    training_positions = set(TRAIN_GRID)
    inside = sum(count for position, count in positions.items() if position in training_positions)
    neighbours = distances(subset, subset)
    neighbours.fill_diagonal_(torch.inf)
    nn = neighbours.min(1).values
    save('p0b_full_grid_fps.json', {
        'selected': len(selected), 'u': value, 'seconds': time.perf_counter() - started,
        'composition_by_image_category': dict(composition_category),
        'composition_by_position_share_in_16_training_grid': inside / len(selected),
        'distinct_positions_used': len(positions),
        'top_positions': [{'position': position, 'row': position // 28, 'column': position % 28, 'count': count}
                          for position, count in positions.most_common(10)],
        'internal_nearest_other': {'min': float(nn.min()), 'median': float(nn.median())},
        'selected_indices': selected,
        'note': 'audit-only diagnostic bank; not a P0 arm'})


# ---------------------------------------------------------------------------------------
# 4. representation analysis
# ---------------------------------------------------------------------------------------

def section_representation() -> None:
    protocol = load_json(P0 / 'protocol_resolved.json')
    records = load_json(P0 / 'per_image_results.json')
    caches: dict = {}
    final = [r for r in records if r['event'] == 350]
    index_by_path = {row['relative_path']: row['cache_index'] for row in protocol['probes']}

    # audit-only floors
    bank_probe_positions = []
    for event in range(1, 351):
        row = protocol['updates'][event - 1]
        obj = load_cache(row['category'], caches)
        bank_probe_positions.append(obj['patches'][row['cache_index']][list(PROBE_GRID)])
    bank_probe_positions = torch.stack(bank_probe_positions).double()  # [350, 64, 768]

    same_position, within_image = defaultdict(list), defaultdict(list)
    for record in final:
        obj = load_cache(record['category'], caches)
        query = obj['patches'][index_by_path[record['path']]][list(PROBE_GRID)].double()  # [64, 768]
        gaps = torch.cdist(query.unsqueeze(0), bank_probe_positions.reshape(-1, 768))
        gaps = gaps.reshape(64, 350, 64)  # [query position, bank image, bank position]
        per_position = torch.stack([gaps[p, :, p].min() for p in range(64)])
        same_position[record['category']].append(per_position)
        training = obj['patches'][index_by_path[record['path']]][list(TRAIN_GRID)].double()
        within_image[record['category']].append(torch.cdist(query, training).min(1).values)

    scale_by_category = {record['category']: record['scale'] for record in records}
    floors = {}
    for category in CATEGORIES:
        scale = scale_by_category[category]

        def describe(stacked: torch.Tensor):
            tensor = stacked.reshape(-1, 64)
            u = tensor.topk(TAIL_K, dim=1).values.mean(1)
            descending = torch.sort(tensor, dim=1, descending=True).values
            return {'top7_mean_raw': float(u.mean()), 'top7_mean_normalized': float((u / scale).mean()),
                    'all64_mean_raw': float(tensor.mean()), 'all64_mean_normalized': float(tensor.mean() / scale),
                    'bulk57_mean_raw': float(descending[:, TAIL_K:].mean()),
                    'median_raw': float(tensor.median()), 'max_raw': float(tensor.max())}

        floors[category] = {
            'scale': scale,
            'same_position_cross_image_floor': describe(torch.stack(same_position[category])),
            'within_image_position_shift': describe(torch.stack(within_image[category])),
        }

    spatial = {}
    for category in CATEGORIES:
        rows = [r for r in final if r['category'] == category]
        values = torch.stack([torch.tensor(r['arms']['STREAM_GREEDY']['distances'], dtype=torch.float64)
                              for r in rows])
        scale = scale_by_category[category]
        by_offset = defaultdict(list)
        for index in range(64):
            by_offset[position_offsets(index)['chebyshev']].append(values[:, index])
        spatial[category] = {
            'scale': scale,
            'mean_distance_by_chebyshev_offset': {
                str(key): {'positions': len(group), 'mean_raw': float(torch.cat(group).mean()),
                           'mean_normalized': float(torch.cat(group).mean() / scale)}
                for key, group in sorted(by_offset.items())},
            'mean_distance_by_probe_row': {str(row): float(values[:, [i for i in range(64)
                                                                      if PROBE_GRID[i] // 28 == row]].mean())
                                           for row in PROBE_ROWS},
            'mean_distance_by_probe_column': {str(column): float(values[:, [i for i in range(64)
                                                                            if PROBE_GRID[i] % 28 == column]].mean())
                                              for column in PROBE_ROWS},
        }

    result = {'floors': floors, 'spatial_structure_final_arm_C': spatial,
              'notes': ('same-position floor: nearest descriptor at the SAME probe position among the 350 '
                        'admitted images; within-image position shift: nearest of the 16 training positions of '
                        'the probe image itself; both audit-only, no learner involvement')}
    save('p0b_representation_analysis.json', result)


# ---------------------------------------------------------------------------------------
# 5. witness guard forensics
# ---------------------------------------------------------------------------------------

def section_witness(replay_events: int) -> None:
    protocol = load_json(P0 / 'protocol_resolved.json')
    trace = [row for row in load_json(P0 / 'update_trace.json') if row['arm'] == 'WITNESS_PROTECTED']
    states = torch.load(P0 / 'states_t350.pt', map_location='cpu', weights_only=False)['models']
    state_d = states['WITNESS_PROTECTED']

    scales = [row['witness_scale'] for row in trace if row.get('witness_scale')]
    accepted_costs, accepted_tails, guard_changes, blocked = [], [], 0, 0
    decisions = 0
    for row in trace:
        for decision in row.get('decisions', []):
            decisions += 1
            guard_changes += int(decision.get('guard_changes_decision', False))
            blocked += int(decision.get('guard_blocked_positive_gain', 0))
            if decision.get('witness_cost') is not None:
                accepted_costs.append(decision['witness_cost'])
                accepted_tails.append(decision.get('tail_cost'))
    witness_scale = [row['witness_scale'] for row in trace if row.get('witness_scale') is not None]

    witnesses = state_d['witnesses'][:state_d['wn']].double()
    bank = state_d['bank'][:state_d['n']].double()
    wm = distances(witnesses, witnesses)
    wm.fill_diagonal_(torch.inf)
    wnn = wm.min(1).values
    positive = wnn[(wnn > 0) & torch.isfinite(wnn)]
    bm = distances(bank, bank)
    bm.fill_diagonal_(torch.inf)
    bnn = bm.min(1).values
    bpos = bnn[(bnn > 0) & torch.isfinite(bnn)]

    replay = _replay_guard(protocol, trace, replay_events)

    result = {
        'trace_statistics': {
            'post_fill_decisions': decisions,
            'guard_changed_decision_count': guard_changes,
            'guard_blocked_positive_gain_victims': blocked,
            'accepted_replacements': len(accepted_costs),
            'accepted_with_positive_mean_witness_cost': int(sum(1 for c in accepted_costs if c > 0)),
            'accepted_with_zero_mean_witness_cost': int(sum(1 for c in accepted_costs if c == 0)),
            'accepted_tail_cost_nonzero_count': int(sum(1 for t in accepted_tails if t and t > 0)),
            'accepted_tail_cost_max': max(accepted_tails) if accepted_tails else None,
            'accepted_mean_witness_cost_sum': float(sum(accepted_costs)),
            'witness_scale_min': float(min(witness_scale)), 'witness_scale_median': float(np.median(witness_scale)),
            'witness_scale_max': float(max(witness_scale)),
        },
        'guard_mathematics': {
            'witness_pool_size': int(state_d['wn']),
            'linear_interpolation_index': 0.9 * (int(state_d['wn']) - 1),
            'interpolated_order_statistics_used': [114, 115],
            'vacuity_condition': ('with 128 nonnegative increments and linear interpolation the 90th percentile is '
                                  'exactly zero whenever at most 12 increments are positive, so damage of any '
                                  'magnitude to at most 12 witnesses passes the tail guard'),
            'dilution_condition': ('G is a mean over the 16 current descriptors and H a mean over 128 witnesses, so '
                                   'G > H permits aggregate witness damage up to 8x the aggregate current-image gain'),
            's_old_final': float(torch.quantile(positive, 0.5)),
            'tail_threshold_0p05_s_old_final': float(0.05 * torch.quantile(positive, 0.5)),
            'witness_pool_internal_nn': {'q10': float(torch.quantile(positive, 0.1)),
                                         'median': float(torch.quantile(positive, 0.5)),
                                         'q90': float(torch.quantile(positive, 0.9))},
            'full_bank_internal_nn': {'q10': float(torch.quantile(bpos, 0.1)),
                                      'median': float(torch.quantile(bpos, 0.5)),
                                      'q90': float(torch.quantile(bpos, 0.9))},
            's_old_inflation_vs_full_bank_median': float(torch.quantile(positive, 0.5) / torch.quantile(bpos, 0.5)),
        },
        'exact_replay_post_boundary': replay,
        'notes': ('accepted gain/cost sums are overlapping within-image diagnostics, not telescoping held-out '
                  'quantities; they bound the accepted trade-off, not the end-to-end support change'),
    }
    save('p0b_witness_analysis.json', result)


def _replay_guard(protocol: dict, trace: list, replay_events: int) -> dict:
    """Replay D's exact post-boundary decisions from the saved t=300 state and measure witness damage."""
    saved = torch.load(P0 / 'states_t300.pt', map_location='cpu', weights_only=False)['models']['WITNESS_PROTECTED']
    model = Support('WITNESS_PROTECTED')
    for key, value in saved.items():
        if key in ('method', 'capacity'):
            continue
        if key == 'rng_state':
            model.rng.set_state(value)
        else:
            setattr(model, key, value.clone() if isinstance(value, torch.Tensor) else value)
    model.capacity = int(saved['capacity'])
    trace_by_event = {row['event']: row for row in trace}
    caches: dict = {}
    per_decision = []
    per_event = []
    for event in range(301, 301 + replay_events):
        row = protocol['updates'][event - 1]
        obj = load_cache(row['category'], caches)
        x = obj['patches'][row['cache_index']][list(TRAIN_GRID)].clone()
        ids = torch.arange((event - 1) * 16, event * 16, dtype=torch.int64)
        xd = x.double()
        witnesses = model.witnesses[:model.wn].double()
        bank = model.bank[:model.n].double()
        scale = model._witness_scale()
        remaining = list(range(16))
        while remaining:
            novelty = distances(xd[remaining], bank).min(1).values
            chosen = remaining[int(novelty.argmax())]
            prior = float(novelty.max())
            remaining.remove(chosen)
            if prior == 0:
                continue
            if model.n < model.capacity:
                model._put(model.n, x[chosen], int(ids[chosen]))
                bank = model.bank[:model.n].double()
                continue
            shortlist = torch.argsort(model.nn_distance[:model.n], stable=True)[:16]
            q = torch.cat((xd, witnesses))
            near, second, near_idx = nearest_two(distances(q, bank))
            removed = torch.where(near_idx[:, None] == shortlist[None], second[:, None], near[:, None])
            to_new = distances(q, xd[chosen:chosen + 1])[:, 0]
            tentative = torch.minimum(removed, to_new[:, None])
            gains = (near[:16, None] - tentative[:16]).mean(0)
            increments = (tentative[16:] - near[16:, None]).clamp_min(0)
            costs = increments.mean(0)
            tails = torch.quantile(increments, 0.9, dim=0)
            positives = (increments > 0).sum(0)
            legal = (gains > costs) & (tails <= 0.05 * scale) if scale is not None else torch.zeros_like(gains, dtype=torch.bool)
            best = int((gains - costs).masked_fill(~legal, -torch.inf).argmax()) if bool(legal.any()) else None
            if best is not None:
                slot = int(shortlist[best])
                accepted = True
                victim_positives = int(positives[best])
                victim_cost = float(costs[best])
                victim_gain = float(gains[best])
                victim_tail = float(tails[best])
                victim_max_increment = float(increments[:, best].max())
                model._put(slot, x[chosen], int(ids[chosen]))
                bank = model.bank[:model.n].double()
                witnesses = model.witnesses[:model.wn].double()
                threshold = 0.05 * scale if scale is not None else None
                per_decision.append({'event': event, 'victim_positives': victim_positives,
                                     'victim_cost': victim_cost, 'victim_gain': victim_gain,
                                     'victim_tail_cost': victim_tail, 'victim_max_increment': victim_max_increment,
                                     'accepted': accepted,
                                     'witness_scale': scale,
                                     'threshold_0p05_s_old': threshold,
                                     'violates_worst_case_intent': (threshold is not None
                                                                    and victim_max_increment > threshold),
                                     'guard_vacuous_for_this_victim': victim_tail == 0.0,
                                     'shortlist_positives_min': int(positives.min()),
                                     'shortlist_positives_max': int(positives.max())})
            else:
                per_decision.append({'event': event, 'accepted': False,
                                     'guard_vacuous_for_all_victims': bool((tails == 0).all()),
                                     'shortlist_positives_min': int(positives.min()),
                                     'shortlist_positives_max': int(positives.max())})
        model._witness_update(x, ids)
        record = trace_by_event.get(event)
        reproduced = fingerprint({'bank': model.bank, 'ids': model.ids}) == record['score_bank_sha256'] \
            if record is not None else None
        per_event.append({'event': event, 'bank_sha256_reproduced': reproduced,
                          'recorded_replaced': record['replaced'] if record else None})
        if reproduced is False:
            log(f'replay mismatch at event {event}')
    matched = [row for row in per_event if row['bank_sha256_reproduced']]
    accepted = [row for row in per_decision if row['accepted']]
    violating = [row for row in accepted if row['violates_worst_case_intent']]
    return {
        'replayed_events': len(per_event),
        'events_with_exact_bank_sha256_reproduction': len(matched),
        'per_event': per_event,
        'decisions': len(per_decision),
        'accepted_decisions': len(accepted),
        'accepted_with_vacuous_tail_guard': int(sum(1 for row in accepted if row['guard_vacuous_for_this_victim'])),
        'accepted_victim_positives_histogram': dict(Counter(row['victim_positives'] for row in accepted)),
        'accepted_victim_max_increment_max': max((row['victim_max_increment'] for row in accepted), default=None),
        'accepted_decisions_whose_largest_single_witness_increment_exceeded_0p05_s_old': len(violating),
        'accepted_decisions_with_zero_harmed_witnesses': int(sum(1 for row in accepted if row['victim_positives'] == 0)),
        'worst_single_witness_increment_over_0p05_s_old_ratio': max(
            (row['victim_max_increment'] / row['threshold_0p05_s_old'] for row in violating), default=None),
        'shortlist_positives_max_histogram': dict(Counter(row['shortlist_positives_max'] for row in per_decision)),
        'per_decision': per_decision,
    }


def section_oracles() -> None:
    """Audit-only decomposition: how much of the residual risk is budget vs observation grid."""
    protocol = load_json(P0 / 'protocol_resolved.json')
    records = load_json(P0 / 'per_image_results.json')
    states = torch.load(P0 / 'states_t350.pt', map_location='cpu', weights_only=False)['models']
    caches: dict = {}
    queries, scale_by_category = _probe_queries(protocol, records, caches)
    query_matrix = torch.cat([row['query'] for row in queries]).double()
    categories_of_patch = [row['category'] for row in queries for _ in range(64)]

    proposals = torch.cat([update_descriptor_from_cache(protocol, caches, event)
                           for event in range(1, 351)]).double()
    log(f'proposal bank: {tuple(proposals.shape)}')
    proposals_u = _u_over_queries(proposals, query_matrix, categories_of_patch, scale_by_category)
    full = torch.cat([load_cache(row['category'], caches)['patches'][row['cache_index']].clone()
                      for row in protocol['updates']]).double()
    log(f'full-grid bank: {tuple(full.shape)}')
    full_u = _u_over_queries(full, query_matrix, categories_of_patch, scale_by_category)
    c_u = _u_over_queries(_bank_rows(states['STREAM_GREEDY']).double(), query_matrix,
                          categories_of_patch, scale_by_category)
    d_u = _u_over_queries(_bank_rows(states['WITNESS_PROTECTED']).double(), query_matrix,
                          categories_of_patch, scale_by_category)
    a_u = _u_over_queries(_bank_rows(states['FROZEN_FIRST']).double(), query_matrix,
                          categories_of_patch, scale_by_category)
    b_u = _u_over_queries(_bank_rows(states['RESERVOIR']).double(), query_matrix,
                          categories_of_patch, scale_by_category)

    result = {
        'registered_arms_final': {'FROZEN_FIRST': a_u, 'RESERVOIR': b_u,
                                  'STREAM_GREEDY': c_u, 'WITNESS_PROTECTED': d_u},
        'proposal_oracle_5600': proposals_u,
        'full_grid_oracle_274400': full_u,
        'decomposition_macro_normalized': {
            'budget_gap_C_minus_proposal_oracle': c_u['macro_normalized'] - proposals_u['macro_normalized'],
            'observation_grid_gap_proposal_oracle_minus_full_grid': proposals_u['macro_normalized'] - full_u['macro_normalized'],
            'total_C_minus_full_grid': c_u['macro_normalized'] - full_u['macro_normalized'],
            'full_grid_floor': full_u['macro_normalized'],
        },
        'counterfactual_minimum_16_position_floor': (
            'the proposal oracle is bounded below by the same 16-position observation grid; the full-grid '
            'oracle is the smallest reachable U for this feature space with unlimited storage'),
        'labels': {
            'proposal_oracle_5600': 'audit-only unbounded bank of all 5600 admitted training proposals',
            'full_grid_oracle_274400': 'audit-only unbounded bank of all 784 positions of all 350 admitted images',
        },
    }
    save('p0b_oracle_decomposition.json', result)


def update_descriptor_from_cache(protocol: dict, caches: dict, event: int) -> torch.Tensor:
    row = protocol['updates'][event - 1]
    obj = load_cache(row['category'], caches)
    require(obj['relative_paths'][row['cache_index']] == row['relative_path'], 'cache row mismatch')
    return obj['patches'][row['cache_index']][list(TRAIN_GRID)].clone()


def main() -> None:
    parser = argparse.ArgumentParser(description='P0b read-only forensics')
    parser.add_argument('sections', nargs='+',
                        choices=['integrity', 'readout', 'compression', 'representation', 'witness',
                                 'oracles', 'all'])
    parser.add_argument('--fps-full', action='store_true')
    parser.add_argument('--replay-events', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    wanted = args.sections
    if 'all' in wanted:
        wanted = ['integrity', 'readout', 'compression', 'representation', 'witness']
    for section in wanted:
        log(f'=== {section} ===')
        if section == 'integrity':
            section_integrity()
        elif section == 'readout':
            section_readout()
        elif section == 'compression':
            section_compression(args.fps_full)
        elif section == 'representation':
            section_representation()
        elif section == 'witness':
            section_witness(args.replay_events)
        elif section == 'oracles':
            section_oracles()


if __name__ == '__main__':
    main()
