"""Artifact-only audit and descriptive tables; never opens a feature tensor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from exps.ecad.r1.train_normal import ROOT, FEATURES, CATEGORIES, file_sha
from .mechanism import CCNM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    summary = json.loads((args.run/'summary.json').read_text())
    contract = json.loads((args.run/'contract.json').read_text())
    updates = [json.loads(line) for line in (args.run/'updates.jsonl').read_text().splitlines()]
    assert [r['event'] for r in updates] == list(range(summary['completed']))
    assert all(r['observed'] == 784 and r['candidates'] == 78 for r in updates)
    assert len({r['raw_png_sha256'] for r in contract['updates'] + contract['witnesses']}) == 398
    assert all(Path(r['relative_path']).parts[1:3] == ('train', 'good')
               for r in contract['updates'] + contract['witnesses'])
    for name, digest in contract['source_hashes'].items():
        assert file_sha(ROOT/name) == digest
    snapshots = {}
    checks = []
    for record in summary['checkpoints']:
        boundary = record['boundary']
        path = args.run/f'ccnm_{boundary:03}.pt'
        assert file_sha(path) == record['sha256']
        m = CCNM.from_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        ids = m.model.serials.numpy()
        ref = torch.load(ROOT/f'results/hope_cad/ecad/r1b/normal_fit_8011_20261018/contcore_{boundary:03}.pt', weights_only=True)
        assert torch.equal(m.model.bank, ref['bank']) and torch.equal(m.model.serials, ref['serials'])
        assert m.model.next_event == boundary
        for origin in torch.unique(m.model.serials//784):
            ctx = m.contexts[m.model.serials//784 == origin]
            assert torch.equal(ctx, ctx[:1].expand_as(ctx))
        torch.testing.assert_close(m.contexts.norm(dim=1), torch.ones(len(ids)), rtol=2e-6, atol=2e-6)
        ledger = m.ledger()
        assert ledger['tensor_total_bytes'] == 8118308
        assert path.stat().st_size == ledger['serialized_bytes'] == record['bytes']
        snapshots[boundary] = set(ids.tolist())
        checks.append({'boundary': boundary, 'exact_r1b_bank_and_ids': True,
                       'source_image_context_consistency': True, 'finite_unit_contexts': True,
                       'complete_tensor_ledger': True, 'checkpoint_hash_verified': True})
    witness_files = []
    for path in sorted(args.run.glob('witness_*.npz')):
        boundary = int(path.stem.split('_')[1])
        with np.load(path) as values:
            assert len(values.files) == 8
            for mode in ('raw', 'context', 'shuffled', 'routing'):
                scores, ids = values[f'{mode}_scores'], values[f'{mode}_ids']
                assert scores.shape == ids.shape == (16, 784)
                assert np.isfinite(scores).all() and (scores >= 0).all()
                assert set(ids.flat) <= snapshots[boundary]
            assert (values['context_scores'] + 2e-6 >= values['raw_scores']).all()
            assert (values['shuffled_scores'] + 2e-6 >= values['raw_scores']).all()
        witness_files.append({'path': str(path), 'sha256': file_sha(path), 'bytes': path.stat().st_size})
    acquisition = {}
    changes = []
    for row in summary['metrics']:
        key = (row['category'], row['mode'])
        if key not in acquisition:
            acquisition[key] = row
        base = acquisition[key]
        changes.append({k: row[k] for k in ('boundary', 'category', 'mode')} | {
            'delta_mean_from_acquisition': row['patch_mean']-base['patch_mean'],
            'delta_q99_from_acquisition': row['patch_q99']-base['patch_q99'],
            'acquisition_boundary': base['boundary']})
    synthetic = {}
    for boundary, rows in summary['synthetic'].items():
        synthetic[boundary] = {}
        for mode in ('raw', 'context', 'shuffled', 'routing'):
            values = {k: [r[mode][k] for r in rows] for k in ('clean', 'collision', 'gap')}
            synthetic[boundary][mode] = {f'mean_{k}': float(np.mean(v)) for k, v in values.items()}
            synthetic[boundary][mode]['positive_gap_count'] = sum(r[mode]['gap'] > 1e-6 for r in rows)
            synthetic[boundary][mode]['changed_from_colliding_id_count'] = sum(r[mode]['selected_id'] != r['collision_serial'] for r in rows)
        synthetic[boundary]['pairs'] = len(rows)
        synthetic[boundary]['wrong_context_max_score'] = max(r['wrong_context_score'] for r in rows)
    cache = {c: (FEATURES/f'clean/{c}_train.pt').stat().st_size for c in CATEGORIES}
    out = {'status': 'PASS', 'checkpoint_audit': checks, 'witness_artifact_audit': witness_files,
           'audit_source_sha256': file_sha(__file__), 'contiguous_writer_events': len(updates),
           'unique_writer_and_witness_identities': 398, 'normal_only_paths_verified': True,
           'acquisition_relative_changes': changes, 'synthetic_summary': synthetic,
           'cache_infrastructure_bytes': cache, 'cache_total_bytes': sum(cache.values()),
           'persistent_learner_tensor_bytes': summary['final_ledger']['tensor_total_bytes'],
           'additional_context_bytes': 418308, 'extra_fraction_of_raw_tensor_payload': 418308/7700000,
           'raw_tensor_payload_bytes': 7700000,
           'checkpoint_total_bytes': sum(r['bytes'] for r in summary['checkpoints']),
           'latency_context_over_raw_median': summary['latency_summary']['context']['median_ms']/summary['latency_summary']['raw']['median_ms'],
           'update_median_ms': float(np.median(summary['update_seconds'])*1000),
           'update_total_seconds': sum(summary['update_seconds']),
           'verdict': summary['verdict']}
    with (args.out or args.run/'analysis.json').open('x') as f:
        json.dump(out, f, sort_keys=True, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
