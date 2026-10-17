"""P0c independent analysis, offline calibration and frozen decision evaluation.

Read-only over P0c artifacts: recomputes every reported statistic from saved
per-image distances, runs the preregistered paired bootstrap, evaluates the
offline full-grid farthest-point reference and the unbounded oracles on the NEW
P0c probes, and applies the frozen decision rules.  Evaluator-only banks are
never inputs to a learner.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.task_p.p0_normal_support import BUDGET, distances  # noqa: E402

ROOT = Path('/mnt/d/code_for_fun/project/nestedlearningforcad/nestedlearningforcad')
FEATURES = ROOT / 'results/hope_cad/frozen_features_v1'
OUT = ROOT / 'results/hope_cad/task_p/p0c'
CATEGORIES = ('bottle', 'carpet', 'hazelnut')
ARMS = ('GRID16_GREEDY_REFERENCE', 'FULL784_GREEDY', 'FULL784_RESERVOIR')
REFERENCE_ARM, CANDIDATE_ARM = 'GRID16_GREEDY_REFERENCE', 'FULL784_GREEDY'
FPS_TARGET = 2702
CHUNK = 64
REPAIR_FACTOR = 64 * float(torch.finfo(torch.float64).eps)


def log(message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
    log(f'wrote {path.relative_to(ROOT)}')


def load(name):
    return json.loads((OUT / name).read_text())


# ---------------------------------------------------------------------------------------
# numerically stable FP64 distance helpers (validated against the frozen scorer)
# ---------------------------------------------------------------------------------------

def stable_row(point: torch.Tensor, bank: torch.Tensor, bank_norms: torch.Tensor) -> torch.Tensor:
    """Exact FP64 norm-dot distance from one point to every bank row, with the frozen repair rule."""
    point = point.double()
    squared = (bank_norms + point.square().sum()).sub_(2 * (bank @ point)).clamp_min_(0)
    norms = bank_norms + point.square().sum()
    repair = squared <= REPAIR_FACTOR * norms
    if bool(repair.any()):
        rows = repair.nonzero(as_tuple=True)[0]
        squared[rows] = (bank[rows] - point).square().sum(1)
    return squared.sqrt_()


def stable_min_distance(queries: torch.Tensor, bank: torch.Tensor, bank_norms: torch.Tensor) -> torch.Tensor:
    out = []
    for start in range(0, queries.shape[0], CHUNK):
        block = queries[start:start + CHUNK].double()
        squared = (bank_norms[None, :] + block.square().sum(1)[:, None]) - 2 * (block @ bank.T)
        squared.clamp_min_(0)
        norms = bank_norms[None, :] + block.square().sum(1)[:, None]
        repair = squared <= REPAIR_FACTOR * norms
        if bool(repair.any()):
            rows, cols = repair.nonzero(as_tuple=True)
            squared[rows, cols] = (block[rows] - bank[cols]).square().sum(1)
        out.append(squared.sqrt_().min(1).values)
    return torch.cat(out)


def validate_helpers(queries: torch.Tensor, bank: torch.Tensor) -> dict:
    """Bounded bitwise comparison against the frozen P0 scorer."""
    bank = bank.double()
    bank_norms = bank.square().sum(1)
    sample_bank = bank[:2000]
    sample_norms = bank_norms[:2000]
    reference = distances(queries[:64], sample_bank)
    computed = torch.stack([stable_row(row, sample_bank, sample_norms) for row in queries[:64].double()])
    difference = (reference - computed).abs().max()
    minimum = stable_min_distance(queries[:64], sample_bank, sample_norms)
    reference_min = reference.min(1).values
    return {'max_abs_difference_vs_frozen_scorer': float(difference),
            'min_distance_max_abs_difference': float((minimum - reference_min).abs().max()),
            'bitwise_equal': bool(torch.equal(reference, computed)),
            'repair_factor': REPAIR_FACTOR}


# ---------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------

def per_category_macro(records, arm, event=350):
    """Per-category means over the categories actually probed at this boundary (equal-category macro)."""
    rows = [r for r in records if r['event'] == event]
    present = [category for category in CATEGORIES if any(r['category'] == category for r in rows)]
    per_category = {category: float(np.mean([r['arms'][arm]['tail_normalized'] for r in rows
                                             if r['category'] == category])) for category in present}
    macro = float(np.mean(list(per_category.values())))
    return per_category, macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', action='store_true', help='also run the offline calibration')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    protocol = load('protocol_resolved.json')
    protocol_sha = sha(OUT / 'protocol_resolved.json')
    records = load('per_image_results.json')
    boundaries = load('boundaries.json')
    resources = load('resources.json')
    reproduction = load('p0_reproduction.json')
    assert protocol_sha == (OUT / 'protocol.sha256').read_text().strip(), 'protocol changed'

    # ---- independent recomputation of the registered statistic
    recomputed = {}
    for event in (100, 200, 300, 350):
        for arm in ARMS:
            per_category, macro = per_category_macro(records, arm, event)
            recomputed[f'{arm}@{event}'] = {'per_category': per_category, 'macro': macro}
    recorded = {}
    for boundary in boundaries:
        for arm in ARMS:
            entry = boundary['methods'][arm]
            values = boundary['categories']
            recorded[f'{arm}@{boundary["event"]}'] = {
                'per_category': {c: values[c][arm]['normalized_risk'] for c in values},
                'macro': float(np.mean([values[c][arm]['normalized_risk'] for c in values]))}
    max_delta = 0.0
    for key in recorded:
        if key in recomputed:
            shared = set(recomputed[key]['per_category']) & set(recorded[key]['per_category'])
            assert shared == set(recorded[key]['per_category']), f'category coverage mismatch at {key}'
            for category in shared:
                max_delta = max(max_delta, abs(recomputed[key]['per_category'][category]
                                               - recorded[key]['per_category'][category]))
            max_delta = max(max_delta, abs(recomputed[key]['macro'] - recorded[key]['macro']))

    # ---- preregistered paired bootstrap on the final boundary
    final = [r for r in records if r['event'] == 350]
    draws = {arm: {category: np.array([r['arms'][arm]['tail_normalized'] for r in final
                                       if r['category'] == category]) for category in CATEGORIES}
             for arm in ARMS}
    generator = np.random.default_rng(protocol['bootstrap']['seed'])
    replicates = np.empty(protocol['bootstrap']['replicates'])
    for replicate in range(protocol['bootstrap']['replicates']):
        candidate_macro, reference_macro = [], []
        for category in CATEGORIES:
            size = len(draws[CANDIDATE_ARM][category])
            index = generator.integers(0, size, size=size)
            candidate_macro.append(draws[CANDIDATE_ARM][category][index].mean())
            reference_macro.append(draws[REFERENCE_ARM][category][index].mean())
        replicates[replicate] = 1 - float(np.mean(candidate_macro)) / float(np.mean(reference_macro))
    candidate_category, candidate_macro = per_category_macro(records, CANDIDATE_ARM)
    reference_category, reference_macro = per_category_macro(records, REFERENCE_ARM)
    reservoir_category, reservoir_macro = per_category_macro(records, 'FULL784_RESERVOIR')
    primary = 1 - candidate_macro / reference_macro
    ci = [float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))]
    category_benefit = {c: 1 - candidate_category[c] / reference_category[c] for c in CATEGORIES}

    # ---- gate evaluation (frozen thresholds)
    gates = protocol['gates']
    valid_execution = bool(reproduction['passed']) and all(
        entry['ledger']['charged_persistent_bytes'] <= BUDGET
        for boundary in boundaries for entry in boundary['methods'].values())
    gate_results = {
        'valid_execution': valid_execution,
        'final_gain_at_least_8pct': bool(primary >= gates['final_gain']),
        'gain_ci_lower_above_zero': bool(ci[0] > gates['ci_lower']),
        'per_category_within_10pct': bool(all(value >= -gates['per_category_final_deterioration']
                                              for value in category_benefit.values())),
    }

    # ---- offline calibration (evaluator-only)
    calibration = {'performed': False}
    if args.calibration:
        calibration = run_calibration(protocol, records, reference_category, reference_macro)

    # ---- decision
    meaningful_offline = calibration.get('fps_reference_benefit_vs_reference_arm', 0.0) >= gates['final_gain']
    if not valid_execution:
        decision = 'BLOCKED'
    elif (gate_results['final_gain_at_least_8pct'] and gate_results['gain_ci_lower_above_zero']
          and gate_results['per_category_within_10pct']):
        decision = 'P0C_GRID_CONFIRMED'
    elif (gate_results['final_gain_at_least_8pct'] and gate_results['gain_ci_lower_above_zero']
          and not gate_results['per_category_within_10pct']):
        decision = 'P0C_ALLOCATION_TRADEOFF'
    elif meaningful_offline:
        decision = 'P0C_STREAMING_SELECTOR_BOTTLENECK'
    else:
        decision = 'P0C_GRID_NOT_CONFIRMED'

    summary = {
        'schema': 'task_p_p0c_summary', 'protocol_sha256': protocol_sha, 'decision': decision,
        'primary': {'reference_arm': REFERENCE_ARM, 'candidate_arm': CANDIDATE_ARM,
                    'reference_macro': reference_macro, 'candidate_macro': candidate_macro,
                    'benefit': primary, 'ci95': ci,
                    'reference_per_category': reference_category, 'candidate_per_category': candidate_category,
                    'per_category_benefit': category_benefit},
        'descriptive_arms': {'FULL784_RESERVOIR': {'macro': reservoir_macro,
                                                   'per_category': reservoir_category}},
        'gates': gate_results,
        'gate_thresholds': gates,
        'recomputation': {'max_abs_delta_vs_recorded': max_delta,
                          'recomputed': recomputed, 'recorded': recorded},
        'calibration': calibration,
        'resources': resources,
        'boundary_trajectories': {
            arm: {str(boundary['event']): boundary['categories'] for boundary in boundaries}
            for arm in ARMS},
        'bank_composition_final': {
            arm: _bank_composition(protocol, arm) for arm in ARMS},
        'limitations': protocol['limitations'],
    }
    save('summary.json', summary)
    save('bootstrap.json', {'replicates': protocol['bootstrap']['replicates'],
                            'seed': protocol['bootstrap']['seed'],
                            'benefit_samples': replicates.tolist(),
                            'ci95': ci, 'benefit': primary})
    save('independent_reanalysis.json', {'max_abs_delta_vs_recorded': max_delta,
                                         'primary_recomputed': primary, 'ci95_recomputed': ci,
                                         'valid_execution': valid_execution,
                                         'decision': decision})
    log(f'decision {decision} | benefit {primary:+.4%} CI [{ci[0]:+.4%}, {ci[1]:+.4%}]')
    print(json.dumps({'decision': decision, 'benefit': primary, 'ci': ci,
                      'gates': gate_results}, indent=1))


def _bank_composition(protocol, arm):
    state = torch.load(OUT / 'states_t350.pt', map_location='cpu', weights_only=False)['models'][arm]
    stride = 16 if arm == 'GRID16_GREEDY_REFERENCE' else 784
    counts = Counter()
    positions = Counter()
    for serial in state['ids'][:state['n']].tolist():
        counts[protocol['updates'][serial // stride]['category']] += 1
        positions[serial % stride] += 1
    training = set(protocol['train_grid'])
    inside = sum(value for position, value in positions.items() if position in training)
    return {'occupied': int(state['n']), 'capacity': int(state['capacity']),
            'by_category': dict(counts), 'share_at_train16_positions': (inside / max(1, state['n'])
                                                                        if stride == 784 else 1.0),
            'distinct_positions_used': len(positions),
            'top_positions': [{'position': position, 'count': count}
                              for position, count in positions.most_common(5)]}


def run_calibration(protocol, records, reference_category, reference_macro):
    """Offline evaluator-only banks on the NEW P0c probes."""
    started = time.perf_counter()
    # probe queries in the registered order
    queries, categories = [], []
    caches = {category: torch.load(FEATURES / f'clean/{category}_train.pt', mmap=True, weights_only=True)
              for category in CATEGORIES}
    for probe in protocol['probes']:
        row = caches[probe['category']]['patches'][probe['cache_index']]
        queries.append(row[list(protocol['probe_grid'])].clone())
        categories.append(probe['category'])
    query_matrix = torch.cat(queries)
    scale_by_category = {category: next(record['scale'] for record in records
                                        if record['category'] == category) for category in CATEGORIES}
    categories_of_patch = [category for category in categories for _ in range(64)]

    def macro_for(bank, norms, label):
        minimum = stable_min_distance(query_matrix, bank, norms)
        per_category = defaultdict(list)
        for index in range(query_matrix.shape[0]):
            per_category[categories_of_patch[index]].append(float(minimum[index]))
        out = {}
        for category, values in per_category.items():
            tensor = torch.tensor(values, dtype=torch.float64).reshape(-1, 64)
            u = tensor.topk(7, dim=1).values.mean(1)
            out[category] = {'u_raw': float(u.mean()), 'u_normalized': float((u / scale_by_category[category]).mean())}
        out['macro_normalized'] = float(np.mean([out[c]['u_normalized'] for c in CATEGORIES]))
        out['label'] = label
        log(f'calibration {label}: macro {out["macro_normalized"]:.6f}')
        return out

    # validation of the stable helpers against the frozen scorer
    helper_validation = validate_helpers(query_matrix, torch.randn(4000, 768, generator=torch.Generator().manual_seed(2)))

    # full-grid bank of the 350 authorised update images
    parts = []
    for row in protocol['updates']:
        parts.append(caches[row['category']]['patches'][row['cache_index']].clone())
    full = torch.cat(parts)
    del parts
    gc.collect()
    full_norms = full.double().square().sum(1)

    # unbounded oracles (evaluator-only)
    proposals = torch.cat([caches[row['category']]['patches'][row['cache_index']]
                           [list(protocol['train_grid'])].clone() for row in protocol['updates']])
    proposal_norms = proposals.double().square().sum(1)
    proposal_u = macro_for(proposals.double(), proposal_norms, 'proposal_oracle_5600')
    full_u = macro_for(full.double(), full_norms, 'full_grid_oracle_274400')

    # offline farthest-point bank at the same 8-MiB budget, built only from the 350 admitted images
    bank64 = full.double()
    norms = full_norms
    generator = torch.Generator().manual_seed(17)
    start = int(torch.randint(bank64.shape[0], (1,), generator=generator))
    min_gap = stable_row(bank64[start], bank64, norms)
    min_gap[start] = torch.inf
    selected = [start]
    while len(selected) < FPS_TARGET:
        nxt = int(torch.argmax(min_gap))
        selected.append(nxt)
        min_gap[nxt] = torch.inf
        min_gap = torch.minimum(min_gap, stable_row(bank64[nxt], bank64, norms))
        if len(selected) % 500 == 0:
            log(f'fps {len(selected)}/{FPS_TARGET} ({time.perf_counter() - started:.0f}s)')
    index = torch.tensor(selected)
    fps_u = macro_for(bank64[index], norms[index], f'offline_fps_{FPS_TARGET}')
    del bank64, norms, full
    gc.collect()

    state = torch.load(OUT / 'states_t350.pt', map_location='cpu', weights_only=False)['models']
    causal = {}
    for arm in ARMS:
        bank = state[arm]['bank'][:state[arm]['n']].clone()
        causal[arm] = macro_for(bank.double(), bank.double().square().sum(1), f'causal_{arm}')

    result = {
        'performed': True,
        'helper_validation': helper_validation,
        'proposal_oracle_5600': proposal_u,
        'full_grid_oracle_274400': full_u,
        f'offline_fps_{FPS_TARGET}': fps_u,
        'causal_banks': causal,
        'reference_arm_macro_from_summary': reference_macro,
        'fps_reference_benefit_vs_reference_arm': 1 - fps_u['macro_normalized'] / reference_macro,
        'candidate_benefit_vs_fps_reference': 1 - causal['FULL784_GREEDY']['macro_normalized'] / fps_u['macro_normalized'],
        'candidate_within_15pct_of_fps_reference': bool(
            causal['FULL784_GREEDY']['macro_normalized'] <= 1.15 * fps_u['macro_normalized']),
        'seconds': time.perf_counter() - started,
        'notes': ('all calibration banks are evaluator-only and were never read by a learner; the offline '
                  'farthest-point bank uses the identical scorer and the 350 authorised update images'),
    }
    save('calibration.json', result)
    return result


if __name__ == '__main__':
    main()
