"""NANM-EVAL-01 Stage 1: source / custody / availability audit -> results/hope_cad/nanm_eval_01/SOURCE_AUDIT.json.

No anomaly label, mask or score is decoded here. Test files are hashed as bytes; images are not opened (MVTec counts
are file counts). Real-IAD is not on disk and is NOT downloaded.

    python -B -m exps.nanm_eval_01.audit
"""
from __future__ import annotations

import subprocess
import sys
import time

import torch

from exps.nanm_eval_01 import common as U

ORACLE = ('nanm/g3.py', 'exps/bench_g3/run.py', 'exps/bench_g3/metrics_split.py', 'exps/bench_g3/test_run.py',
          'exps/nanm_r0/audit.py', 'exps/nanm_r1/run.py', 'exps/nanm_r2/run.py', 'exps/nanm_r2b/run.py',
          'exps/nar_lite_01/run.py', 'exps/nar_b14_01/run.py', 'exps/calm02/native.py', 'exps/calm02/data.py',
          'exps/cadfinal/metrics.py', 'exps/bench_aud_00/run.py', 'exps/tcs_00/run.py', 'exps/vbench_00/run.py',
          'exps/nar_pilot_00/core.py', 'exps/dinomaly/encoder.py', 'exps/nanm_repro_01/run.py',
          'agents/tasks/task_bench_g3.md', 'agents/tasks/task_bench_g3_amendment_a1.md',
          'agents/tasks/task_bench_g3_amendment_a2.md', 'agents/tasks/task_nanm_repro_01.md')
HISTORY = {
    'NANM-REPRO-01': ('agents/reports/nanm_repro_01_2026_11_22.md', 'results/hope_cad/nanm_repro_01/COMPARE.json'),
    'BENCH-G3 (historical MPDD)': ('agents/reports/bench_g3.md', 'results/hope_cad/bench_g3/RESULT_mpdd.json'),
    'BENCH-G3 repro MPDD': (None, 'results/hope_cad/nanm_repro_01/bench_g3/RESULT_mpdd.json'),
    'BENCH-G3 repro VisA': (None, 'results/hope_cad/nanm_repro_01/bench_g3/RESULT_visa.json'),
    'NANM-NL-01D': ('agents/reports/nanm_nl_01_d.md', None),
    'NANM-NL-02 v1 lock': (None, 'results/hope_cad/nanm_nl_02/PREREG_LOCK.json'),
    'NANM-NL-02 v2 results': ('agents/reports/nanm_nl_02_stage0_v2.md', 'results/hope_cad/nanm_nl_02_v2/DESIGN_V2_RESULTS.json'),
    'NANM-NL-02 v2 lock': (None, 'results/hope_cad/nanm_nl_02_v2/PREREG_V2_LOCK.json'),
    'NANM-NL-02 closure': ('agents/reports/nanm_nl_02_closure.md', None),
    'NANM-GEOM-01': ('agents/reports/nanm_geom_01.md', 'results/hope_cad/nanm_geom_01/GEOM01_RESULTS.json'),
}
PAPERS = {'CADIC': 'docs/papers/CADIC_Continual_Anomaly_Detection_Based_on_Incremental_Coreset.pdf',
          'ReplayCAD': 'docs/papers/ReplayCAD_Generative_Diffusion_Replay_for_Continual_Anomaly_Detection.pdf',
          'ContCore': 'docs/papers/related_cad/arxiv_2608.15277.pdf',
          'Nested Learning': 'docs/papers/Nested_Learning_The_Illusion_of_Deep_Learning_Architectures.pdf'}


def fhash(rel):
    p = U.ROOT / rel
    return U.sha(p) if p.exists() else None


def cache_inventory():
    """Every token cache / normal state / score grid on K: re-hashed against its recorded sha256."""
    out = {'features': {}, 'normal': {}, 'scores': {}, 'problems': []}
    for d in (U.REPRO_CACHE / 'bench_g3/features', U.REPRO_CACHE / 'vbench_00/features'):
        for p in sorted(d.glob('*.pt')):
            meta = U.rj(p.with_suffix('.json'))
            h = U.sha(p)
            out['features'][str(p)] = {'bytes': p.stat().st_size, 'sha256': h, 'recorded': meta.get('sha256'),
                                       'equal': h == meta.get('sha256'), 'encoder_sha256': meta.get('encoder_sha256'),
                                       'n': meta.get('n'), 'json_sha256': U.sha(p.with_suffix('.json'))}
            if h != meta.get('sha256'):
                out['problems'].append(f'feature hash mismatch {p.name}')
            U.log('hash', p.name, h == meta.get('sha256'))
    norm = U.rj(U.REPRO / 'bench_g3/NORMAL.json')
    for ds, rec in norm.items():
        for seed, srec in rec['seeds'].items():
            for name, h0 in srec['files'].items():
                p = U.REPRO_CACHE / 'bench_g3/normal' / name
                h = U.sha(p)
                out['normal'][name] = {'bytes': p.stat().st_size, 'sha256': h, 'equal': h == h0}
                if h != h0:
                    out['problems'].append(f'normal state hash mismatch {name}')
    sc = U.rj(U.REPRO / 'bench_g3/SCORES_COMPLETE.json')
    for name, h0 in sc['files'].items():
        p = U.REPRO_CACHE / 'bench_g3/scores' / name
        h = U.sha(p)
        out['scores'][name] = {'bytes': p.stat().st_size, 'sha256': h, 'equal': h == h0}
        if h != h0:
            out['problems'].append(f'score grid hash mismatch {name}')
    out['footprint_bytes'] = {k: sum(v['bytes'] for v in out[k].values()) for k in ('features', 'normal', 'scores')}
    return out


def mvtec_custody():
    root = U.ROOT / 'data/mvtec'
    from exps.calm02 import data as CD
    cats, files = {}, {}
    for c in CD.MVTEC_ORDER:
        tr, te = CD._mvtec(root, c)
        masks = sorted((root / c / 'ground_truth').rglob('*.png'))
        cats[c] = {'train_good': len(tr), 'test': len(te), 'test_good': sum(r['label'] == 0 for r in te),
                   'test_anomalous': sum(r['label'] for r in te), 'masks': len(masks),
                   'masks_match_anomalous': len(masks) == sum(r['label'] for r in te)}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            files[p.relative_to(root).as_posix()] = U.sha(p)
    tr_h = {h for k, h in files.items() if '/train/' in k}
    dups = sorted(k for k, h in files.items() if '/test/' in k and h in tr_h)
    tot = {k: sum(v[k] for v in cats.values()) for k in ('train_good', 'test', 'test_good', 'test_anomalous', 'masks')}
    tar = U.ROOT / 'data/mvtec_anomaly_detection.tar.xz'
    return {'root': str(root), 'order': list(CD.MVTEC_ORDER), 'categories': cats, 'totals': tot,
            'official_totals': {'train_good': 3629, 'test': 1725}, 'complete_vs_official': tot['train_good'] == 3629 and tot['test'] == 1725,
            'n_files': len(files), 'manifest_sha256': U.ssha('\n'.join(f'{k} {v}' for k, v in sorted(files.items()))),
            'train_test_exact_duplicates': dups, 'license': 'CC BY-NC-SA 4.0 (data/mvtec/license.txt, readme.txt)',
            'license_txt_sha256': files.get('license.txt'), 'tarball': {'path': str(tar), 'bytes': tar.stat().st_size if tar.exists() else None,
                                                                        'note': 'smaller than the official 4.9 GB archive; NOT used; the extracted tree is complete by file count'},
            'g3_token_cache': None, 'g3_cache_status': 'MISSING (no DINOv2-reg B/14 392 token cache for MVTec on K: or E:)',
            'prior_exposure': 'consumed development: G3 selected among 8 arms on MVTec (NANM-R2); R2b robustness',
            'extraction_estimate': {'images': tot['train_good'] + tot['test'], 'gpu_seconds_est': round((tot['train_good'] + tot['test']) / 12.3),
                                    'throughput_basis': 'REPRO-01 extract_visa: 900 images / 73.1 s on the same RTX 4050 (12.3 img/s)',
                                    'cache_bytes_fp32_oracle_format': (tot['train_good'] + tot['test']) * 784 * 768 * 4},
            'status': 'OWNER_GO_REQUIRED'}


def realiad_audit():
    found = [str(p) for p in (U.ROOT / 'data').glob('*') if 'real' in p.name.lower()]
    n_img, n_norm, n_anom = 151_050, 99_721, 51_329
    return {'on_disk': bool(found), 'paths_found': found, 'status': 'OWNER_GO_REQUIRED',
            'official_facts': {'source': 'Wang et al., Real-IAD, CVPR 2024 (arXiv 2403.12580), Table statistics',
                               'categories': 30, 'views_per_specimen': 5, 'images': n_img, 'normal_images': n_norm,
                               'anomalous_images': n_anom, 'resolution': 'up to 2000-5000 px (raw); 1024 px release per the task',
                               'download_realiad_1024_gb_per_task': 53, 'raw_release_gb_per_task': 507},
            'unresolved': ['official split JSON (train/test counts per class) not inspected: not on disk',
                           'license text not inspected (not on disk; not stated in the search results)',
                           'specimen/sample-ID field names for grouping the 5 views (must be read from the official JSON)',
                           'exact realiad_1024 archive size and checksum'],
            'storage_estimate_bytes': {'fp32_tokens_all_images_ceiling': n_img * 784 * 768 * 4,
                                       'fp16_unit_tokens_all_images': n_img * 784 * 768 * 2,
                                       'download_realiad_1024': 53 * 10 ** 9},
            'note_fp16': ('The canonical scorer consumes A.unit(fp32).half() tokens (bench_g3.load_test / fit_tokens). A cache '
                          'of unit fp16 tokens is therefore bit-identical at the scorer input; this must be PROVEN on a '
                          'pilot (parity test) before use and is not a silent precision change. Not done without GO.'),
            'compute_estimate': {'extract_gpu_hours_at_12.3_img_s': round(n_img / 12.3 / 3600, 2),
                                 'note': 'excludes 1024 px decoding overhead and stage-wise scoring of 30 tasks'},
            'plan_requirements': ['owner GO for 53 GB download to a non-D drive', 'streaming extraction + scoring with bounded temp',
                                  'pilot parity on a small class', 'grouping of the 5 views by specimen for bootstrap']}


def main():
    t0 = time.time()
    U.OUT.mkdir(parents=True, exist_ok=True)
    assert not (U.OUT / 'SOURCE_AUDIT.json').exists(), 'SOURCE_AUDIT already written'
    st0 = U.storage()
    G = U.bind()
    rec = {'storage_before': st0, 'time': time.strftime('%FT%T'),
           'git': {'branch': subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=U.ROOT, capture_output=True, text=True).stdout.strip(),
                   'head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=U.ROOT, capture_output=True, text=True).stdout.strip(),
                   'status_short': subprocess.run(['git', 'status', '--short'], cwd=U.ROOT, capture_output=True, text=True).stdout.splitlines()},
           'agents_md': {'sha256': U.sha(U.ROOT / 'AGENTS.md'), 'pin': U.AGENTS_PIN},
           'task_sha256': U.sha(U.TASKFILE),
           'env': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'cuda_available': torch.cuda.is_available(),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'cudnn': torch.backends.cudnn.version()}}
    rec['agents_md']['equal'] = rec['agents_md']['sha256'] == U.AGENTS_PIN
    try:
        rec['env']['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,memory.total', '--format=csv,noheader'],
                                              capture_output=True, text=True).stdout.strip()
    except OSError:
        rec['env']['driver'] = None
    # canonical sources vs the historical BENCH-G3 / REPRO locks
    hl = U.rj(U.ROOT / 'results/hope_cad/bench_g3/LOCK.json')
    rl = U.rj(U.REPRO / 'bench_g3/LOCK.json')
    rec['canonical_sources'] = {p: fhash(p) for p in ORACLE}
    rec['vs_historical_bench_g3_lock'] = {p: {'now': fhash(p), 'locked': h, 'equal': fhash(p) == h} for p, h in hl['sources'].items()}
    rec['vs_repro_lock'] = {p: {'now': fhash(p), 'locked': h, 'equal': fhash(p) == h} for p, h in rl['sources'].items()}
    b14 = G.B14()
    ck = U.sha(b14.ENC_PATH)
    rec['encoder'] = {'path': str(b14.ENC_PATH), 'sha256': ck, 'locked': b14.ENC_SHA, 'equal': ck == b14.ENC_SHA,
                      'substrate': 'DINOv2-reg ViT-B/14, 392 px resize, 28x28x768 tokens, mean of blocks 6-9, l2-normalised (BENCH-G3 §1)',
                      'not_this_substrate': 'the 15-task ViT-B/8 Dinomaly/Replay-24 pipeline is a different substrate'}
    rec['caches'] = cache_inventory()
    rec['datasets'] = {
        'mpdd': {'status': 'CACHED_VERIFIED', 'tasks': list(G.MPDD_ORDER), 'n_tasks': 6, 'custody': 'REPRO-01 mpdd_custody 1628/1628 files equal to MPDD_MANIFEST',
                 'test_counts': rl['test_counts']['mpdd'], 'exposure': 'BENCH-G3 clean confirmation (one-shot) consumed 2026-11-20; now exposed DEV'},
        'visa': {'status': 'CACHED_VERIFIED', 'tasks': list(G.order('visa')), 'n_tasks': 12, 'test_counts': rl['test_counts']['visa'],
                 'exposure': 'DEV (240) consumed 3x; FINAL accessed by VBENCH-00 and BENCH-G3; exposed'},
        'mvtec': mvtec_custody(),
        'realiad': realiad_audit()}
    rec['historical_results'] = {k: {'report': r, 'report_sha256': fhash(r) if r else None, 'artifact': a,
                                     'artifact_sha256': fhash(a) if a else None} for k, (r, a) in HISTORY.items()}
    rm, rv = U.rj(U.REPRO / 'bench_g3/RESULT_mpdd.json'), U.rj(U.REPRO / 'bench_g3/RESULT_visa.json')
    hm = U.rj(U.ROOT / 'results/hope_cad/bench_g3/RESULT_mpdd.json')
    rec['anchors'] = {'mpdd': {k: rm['macro'][f'mpdd|{k}|s2711'] for k in ('G3', 'A0b', 'G1b')},
                      'visa': {k: rv['macro'][f'visa|{k}|s2711'] for k in ('G3', 'A0b', 'G1b')},
                      'mpdd_repro_equals_historical': rm['macro'] == hm['macro'],
                      'task_quoted': {'mpdd_G3_ap': 0.5004, 'mpdd_G3_auroc': 0.9396, 'visa_G3_ap': 0.4866, 'visa_G3_auroc': 0.9720, 'mpdd_A0b_auroc': 0.961}}
    rec['metric_provenance'] = {
        'pixel_ap_native_exact': 'exps.nanm_r0.audit.exact_ap -> exps.calm02.native.native_ap with NL.map_native (bilinear 392 -> native, align_corners False); all pixels of all test images of a category; macro over categories',
        'pixel_ap_native_binned': 'exps.nanm_r0.audit.image_hists + binned_ap (8192 bins, [0, max grid + 1e-12]); BENCH-G3 stage matrices; validated vs exact within 1e-3',
        'pixel_ap_224': 'bench_g3.maps224 + bench_aud_00.mask224 + cadfinal.metrics.exact_ap (paper-comparable readout)',
        'image_auroc': 'sklearn roc_auc_score on A.image_scores = mean of top 1 % of the A1 392 map (R2.maps_of: bilinear + Gaussian sigma 4)',
        'image_auroc_mapmax': 'max of the A1 392 map (descriptive)',
        'fm': 'bench_g3.chaudhry: f_j = max_{l in j..T-2} a_{l,j} - a_{T-1,j}; FM = mean_{j<T-1}; also acquisition - final',
        'replaycad_eq8': 'ReplayCAD Eq. 8 (max over gamma <= N-1 of a_gamma,j minus final) has the same intent as Chaudhry FM; source PDF local; exact evaluator not available -> reported as Chaudhry FM, labelled',
        'bootstrap': 'paired category-stratified image bootstrap, B=2000, seed 20261120 (BENCH-G3 §6)'}
    rec['baselines'] = {
        'A0b': {'status': 'SOURCE_VERIFIED', 'source': 'bench_g3.a0b_sizes / euclid; stored REPRO grids', 'matching': 'same encoder/FIT/seed/k-centre; bytes = G3 + <=1 key'},
        'G1b': {'status': 'SOURCE_VERIFIED', 'source': 'bench_g3._normal_ds (Sx, s1) + R2.shrink 0.5; stored REPRO grids', 'matching': 'G3 1 % bank + centred feature covariance; bytes = G3 + 3,072'},
        'CADIC': {'status': 'UNMATCHED_PUBLISHED', 'paper_sha256': fhash(PAPERS['CADIC']),
                  'mismatches': ['ViT-B/8 IN-21k layer 9 vs DINOv2-reg B/14 blocks 6-9', '224 vs 392 input', 'no smoothing vs Gaussian sigma 4',
                                 'Eq. 9 image score vs top-1 % mean', 'all train normals vs FIT (~60 %)', 'local paper-faithful code lacks exact checkpoint/preprocessing parity (agents/reports/01_cadic_parity_audit.md)'],
                  'published': {'mvtec_1x15': {'i_auroc': 0.972, 'p_aupr': 0.584}, 'visa_1x12': {'i_auroc': 0.891, 'p_aupr': 0.438}}},
        'ReplayCAD': {'status': 'UNMATCHED_PUBLISHED', 'paper_sha256': fhash(PAPERS['ReplayCAD']),
                      'mismatches': ['diffusion generator (~1B params, not counted) + InvAD detector', 'detector input UNRESOLVED', 'no local source code'],
                      'published': {'mvtec_1x15': {'i_auroc': 0.948, 'p_aupr': 0.537}, 'visa_1x12': {'i_auroc': 0.903, 'p_aupr': 0.415}}},
        'ContCore': {'status': 'UNMATCHED_PUBLISHED', 'paper_sha256': fhash(PAPERS['ContCore']),
                     'mismatches': ['WRN50 backbone', 'reports pixel AUROC, not P-AUPR (not comparable to our primary metric)', 'no local source code'],
                     'published': {'mvtec_1x15': {'i_auroc': 0.988, 'p_auroc': 0.974}, 'visa_1x12': {'i_auroc': 0.937, 'p_auroc': 0.972},
                                   'realiad_1x30': {'i_auroc': 0.902, 'p_auroc': 0.975}},
                     'provenance': 'agents/reports/lit_01_cad_benchmark_accuracy_memory_time.md (owner-verified ledger)'},
        'Dinomaly/Replay-24 ViT-B/8 pipeline': {'status': 'DIFFERENT_SUBSTRATE', 'note': 'never presented as a G3 reproduction'}}
    rec['nl_history'] = {
        'NANM-NL-01D': 'NO_ACTIONABLE_HEADROOM within the registered guarded quota-prefix lattice only',
        'NANM-NL-02 v2': 'registered NL02_V2_PROTOCOL_BLOCKED (M12, external write); conditional DESIGN assessment NL02_V2_ADAPTATION_NOT_SUPPORTED',
        'NANM-GEOM-01': 'GEOM01_WHITENING_DRIFT_DOMINANT (normal-only NLL); actionability passed only via a level shift'}
    disc = []
    if not rec['agents_md']['equal']:
        disc.append('AGENTS.md sha differs from the AGENTS.local.md pin')
    disc += [f'canonical source changed vs historical lock: {p}' for p, v in rec['vs_historical_bench_g3_lock'].items() if not v['equal']]
    disc += [f'canonical source changed vs repro lock: {p}' for p, v in rec['vs_repro_lock'].items() if not v['equal']]
    disc += rec['caches']['problems']
    if not rec['encoder']['equal']:
        disc.append('encoder checkpoint hash differs')
    if not rec['datasets']['mvtec']['complete_vs_official']:
        disc.append('MVTec local tree incomplete vs official counts')
    rec['blocking_discrepancies'] = disc
    rec['non_blocking_notes'] = [
        'MVTec AD images complete locally but no G3 token cache: extraction is a new resource cost -> OWNER_GO_REQUIRED (task §8.6)',
        'Real-IAD absent -> OWNER_GO_REQUIRED; official split JSON, license and specimen IDs UNRESOLVED',
        'BENCH-G3 uses FIT (~60 % of train normals) for G3 and all arms; published CAD methods use all train normals',
        f"VHDX {st0['VHDX']} B at audit vs 179,733,266,432 B at NL-02 v2 end (2026-11-22 20:07): +{st0['VHDX'] - 179733266432} B grown outside this task (no EVAL-01 process ran before this audit)"]
    rec['pass'] = not disc
    rec['storage_after'] = U.storage()
    rec['seconds'] = time.time() - t0
    U.wj(U.OUT / 'SOURCE_AUDIT.json', rec, exclusive=True)
    U.log('SOURCE_AUDIT', 'PASS' if rec['pass'] else 'FAIL', disc[:6])


if __name__ == '__main__':
    main()
