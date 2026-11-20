"""B326: memory reduction for detector F (card replication/experiments/B326/card.md, sections 2 and 6).

Reuses replication/scripts/hope_replicate.py (B325 driver) for gate, train-feature extraction, training, fidelity,
test extraction, scoring and metrics; Khang's functions are imported unmodified (pinned, see khang_source.json). New:
    locate         find the pinned B325 checkpoints / BASE files under the input directories and verify SHA-256
    quantize       M2: int8 archive (per rank component, fp16 scales) + dequantised fp32 checkpoint for scoring
    int8-gate      M2: normal-only min-grid error, int8 vs bf16 archive (gate <= 0.01) + Khang's bf16 fidelity floor
    metrics-plain  A1 + raw metrics for a scored run without training records
    cpu-base       M1: BASE from a seeded CPU torch.Generator (run with ATEN_CPU_CAPABILITY=default)
    label          M2 / M1 labels from RESULT.json files only
    job            orchestration, one process per GPU lane
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hope_replicate as D  # noqa: E402

REPO = D.REPO
CONFIG = D.REPLICATION / 'configs' / 'b326.json'
SELF = Path(__file__).resolve()
BASE_DRIVER = Path(D.__file__).resolve()
read, write_json, require, sha, log = D.read, D.write_json, D.require, D.sha, D.log


def config():
    return read(CONFIG)


# ---------------------------------------------------------------- pure rules (no torch)
def label_m2(per_seed, rules, seeds):
    """Card s2 M2 + s6.6. per_seed: {seed: {'d_ap', 'd_auc', 'gate_passed', 'gate_rel'}} for exactly the planned
    `seeds` (None for a missing seed). The gate is re-checked from the measured scalar, not trusted from the flag."""
    if not seeds or set(per_seed) != set(seeds) or any(v is None for v in per_seed.values()):
        return 'NO_LABEL_INCOMPLETE'
    if any(bool(v['gate_passed']) != (v['gate_rel'] <= rules['m2_fidelity_max']) for v in per_seed.values()):
        return 'NO_LABEL_INCONSISTENT'
    gate = all(v['gate_rel'] <= rules['m2_fidelity_max'] for v in per_seed.values())
    if gate and all(abs(v['d_ap']) <= rules['m2_lossless_abs'] and abs(v['d_auc']) <= rules['m2_lossless_abs']
                    for v in per_seed.values()):
        return 'INT8_LOSSLESS'
    if gate and all(v['d_ap'] >= -rules['m2_within'] and v['d_auc'] >= -rules['m2_within'] for v in per_seed.values()):
        return 'INT8_WITHIN_0005'
    return 'INT8_LOSSY'


def label_m1a(canonical, seeds):
    """Card s6.2, in-job part only: {seed: [canonical sha, ...]} from separate processes on this host, for exactly
    the planned `seeds`. BASE_PORTABLE needs a second machine type and is decided outside this job."""
    if not seeds or set(canonical) != set(seeds) or any(len(v) < 2 for v in canonical.values()):
        return 'NO_LABEL_INCOMPLETE'
    return 'BASE_SAME_MACHINE_ONLY' if all(len(set(v)) == 1 for v in canonical.values()) else 'BASE_NOT_REPRODUCIBLE'


def label_m1b(deltas, rules):
    """Card s2 M1b + s6.7: deltas = F(CPU BASE) - F(T4 BASE), same job, final Pixel AP, one per seed."""
    if len(deltas) != 2 or any(d is None for d in deltas):
        return 'NO_LABEL_INCOMPLETE'
    mean = sum(deltas) / 2
    if abs(mean) < rules['m1b_null_mean_abs'] and all(abs(d) < rules['m1b_null_seed_abs'] for d in deltas):
        return 'BASE_FROM_SEED_OK'
    if all(d <= rules['m1b_worse'] for d in deltas):
        return 'BASE_FROM_SEED_WORSE'
    return 'BASE_FROM_SEED_MIXED'


def lanes_round_robin(items, n):
    lanes = [[] for _ in range(n)]
    for i, x in enumerate(items):
        lanes[i % n].append(x)
    return [l for l in lanes if l]


# ---------------------------------------------------------------- torch helpers
def quantize_archive(archive, torch):
    """Symmetric int8 per rank component: A [r,in] per row, B [out,r] per column; fp16 scales (card s6.5)."""
    out = []
    for st in archive:
        q = {}
        for name, (A, B) in st.items():
            A, B = A.float(), B.float()
            sa = (A.abs().amax(dim=1) / 127).to(torch.float16)
            sb = (B.abs().amax(dim=0) / 127).to(torch.float16)
            sa = torch.where(sa == 0, torch.ones_like(sa), sa)       # zero component (or fp16 underflow) -> q = 0
            sb = torch.where(sb == 0, torch.ones_like(sb), sb)
            qa = torch.clamp(torch.round(A / sa.float()[:, None]), -127, 127).to(torch.int8)   # round half to even
            qb = torch.clamp(torch.round(B / sb.float()[None, :]), -127, 127).to(torch.int8)
            q[name] = (qa, sa, qb, sb)
        out.append(q)
    return out


def dequantize_archive(qarchive):
    return [{n: (qa.float() * sa.float()[:, None], qb.float() * sb.float()[None, :])
             for n, (qa, sa, qb, sb) in st.items()} for st in qarchive]


def stored_bytes(qarchive):
    return sum(x.numel() * x.element_size() for st in qarchive for quad in st.values() for x in quad)


def canonical_sha(state):
    """SHA-256 over key names and fp32 tensor bytes in state_dict order (card s6.2), not torch.save bytes."""
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode())
        h.update(v.detach().float().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def base_from_seed(seed, torch, nn):
    """Same module order and distributions as dinomaly.official.init_trainable, randomness from a CPU generator."""
    from exps.vitb8_q2.common import SingleStreamDecoder
    model = SingleStreamDecoder(checkpoint=True)
    require(len(list(model.buffers())) == 0, 'decoder has buffers; seed-only BASE would miss them')
    g = torch.Generator(device='cpu').manual_seed(seed)
    done = set()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03, generator=g)
                done.add(id(m.weight))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    done.add(id(m.bias))
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                done.update((id(m.bias), id(m.weight)))
    require(done == {id(p) for p in model.parameters()}, 'some decoder parameters were not seed-initialised')
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------- phases
def locate(a):
    cfg = config()
    root = Path(a.inputs)
    found = {}
    files = [p for p in root.rglob('*.pt')]
    for seed, spec in cfg['checkpoints_F'].items():
        rec = {}
        for key in ('stage15', 'base'):
            hits = [p for p in files if p.as_posix().endswith('/' + spec[key])]
            require(len(hits) == 1, f'{seed} {key}: expected one mounted {spec[key]}, found {len(hits)}')
            require(sha(hits[0]) == spec[key + '_sha256'], f'{seed} {key}: SHA-256 differs from the B325 run')
            rec[key] = str(hits[0])
        rec['hist'] = str(Path(rec['base']).parent)
        require(Path(rec['hist'], 'BASE.pt').exists(), 'BASE.pt name')
        found[seed] = rec
    write_json(Path(a.out) / 'INPUTS.json', found)
    log('LOCATE', json.dumps(found))


def quantize(a):
    import torch
    src = Path(a.stage15)
    require(sha(src) == a.sha, 'stage15 hash')
    st = torch.load(src, weights_only=True)
    bf16, int8 = Path(a.out_bf16), Path(a.out_int8)
    for d in (bf16, int8):
        require(not d.exists(), f'preserve existing run dir {d}')
        d.mkdir(parents=True)
    shutil.copyfile(src, bf16 / 'stage15.pt')
    q = quantize_archive(st['archive'], torch)
    torch.save({'archive_int8': q, 'scheme': 'symmetric int8 per rank component; fp16 scales; A rows, B columns'},
               int8 / 'archive_int8.pt')
    deq = dequantize_archive(q)
    torch.save({**{k: v for k, v in st.items() if k != 'archive'}, 'archive': deq, 'variant': 'int8_dequantised_fp32'},
               int8 / 'stage15.pt')
    bf16_bytes = sum(x.numel() * 2 for s in st['archive'] for pair in s.values() for x in pair)
    amax = [float(x) for s in st['archive'] for A, B in s.values()
            for x in (*A.float().abs().amax(dim=1), *B.float().abs().amax(dim=0))]
    guard_hits = sum(1 for x in amax if x > 0 and torch.tensor(x / 127).to(torch.float16).item() == 0)
    err = max(float((x - y).abs().max()) for s0, s1 in zip(st['archive'], deq) for n in s0 for x, y in zip(s0[n], s1[n]))
    write_json(int8 / 'QUANT.json', {'stored_bytes_int8_plus_scales': stored_bytes(q), 'bf16_bytes': bf16_bytes,
                                     'experts': len(q), 'sites': len(q[0]), 'max_abs_dequant_error_fp32': err,
                                     'min_component_amax': min(amax), 'zero_components': sum(x == 0 for x in amax),
                                     'scale_underflow_guard_hits': guard_hits,
                                     'archive_int8_sha256': sha(int8 / 'archive_int8.pt')})
    for d, variant in ((bf16, 'bf16'), (int8, 'int8')):
        write_json(d / 'TRAIN_COMPLETE.json', {'arm': 'F', 'seed': a.seed, 'variant': variant, 'source_stage15': str(src),
                                               'source_stage15_sha256': a.sha, 'stage15_sha256': sha(d / 'stage15.pt'),
                                               'records': {}})
    log('QUANTIZE', a.seed, stored_bytes(q), 'bytes vs', bf16_bytes, 'max dequant err', err)


def int8_gate(a):
    cfg = config()
    m = D.khang(a.data, a.weights, hist=a.hist)
    m.R.configure()
    t, R, L, K, E = m.torch, m.R, m.L, m.K, m.E
    budget = D.Deadline(a.seconds)                   # one deadline for the whole phase
    base = t.load(Path(a.hist) / 'BASE.pt', weights_only=True)
    raw = {v: t.load(Path(d) / 'stage15.pt', weights_only=True)['archive'] for v, d in (('fp32', a.bf16), ('int8', a.int8))}

    def error(dtype):
        """Relative mean abs error of the min-over-experts grid, dequantised int8 vs original archive, both run in
        `dtype` (model, encoder, archive), on the same 4 EVAL train/good images per category as nar_lite_01.fidelity."""
        enc = L.lite_encoder(dtype)
        model, loras, _ = R.model_new(base)
        model.eval().to(dtype)
        arch = {v: [{n: (x.to(dtype), y.to(dtype)) for n, (x, y) in s.items()} for s in raw[v]] for v in raw}
        num = den = 0.
        with t.no_grad():
            for c in R.ORDER:
                budget.check()
                rows = R.rows_for(c, 'train')
                idx = K.normal_partition(rows)['EVAL'][:4]
                x = t.stack([E.preprocess(R.DATA / rows[i]['relative_path']) for i in idx]).cuda().to(dtype)
                f = L.tokens(enc, x)
                g0 = K.task_agnostic_grid(list(L.expert_grids(model, loras, arch['fp32'], f, dtype)))[0]
                g8 = K.task_agnostic_grid(list(L.expert_grids(model, loras, arch['int8'], f, dtype)))[0]
                num += float((g8 - g0).abs().sum())
                den += float(g0.abs().sum())
        del enc, model, loras, arch
        t.cuda.empty_cache()
        return num / den

    rel32 = error(t.float32)                         # gate (card s8.1): isolates quantisation from bf16 arithmetic
    rel16 = error(t.bfloat16)                        # descriptive: same comparison inside the bf16 deployment
    thr = cfg['rules']['m2_fidelity_max']
    gate = {'relative_mean_abs_error_int8_vs_fp32_in_fp32': rel32, 'threshold': thr, 'passed': rel32 <= thr,
            'relative_mean_abs_error_int8_vs_original_in_bf16': rel16, 'images': 4 * len(R.ORDER)}
    L.OUT = L.REF = Path(a.bf16)                     # noise floor: Khang's whole-pipeline bf16-vs-fp32 fidelity
    floor = L.fidelity(t.bfloat16, budget)
    write_json(Path(a.bf16) / 'FIDELITY.json', floor)
    write_json(Path(a.int8) / 'INT8_GATE.json', {**gate, 'bf16_pipeline_vs_fp32_floor': floor.get('relative_mean_abs_error')})
    log('INT8_GATE', a.seed, 'fp32', round(rel32, 5), gate['passed'], 'bf16', round(rel16, 5),
        'floor', floor.get('relative_mean_abs_error'))


def metrics_plain(a):
    tc = D.verify_sealed(a.run)
    m = D.khang(a.data, a.weights, run=a.run)
    budget = D.Deadline(a.seconds)
    a1, raw = m.L.metrics('smooth', budget), m.L.metrics('raw', budget)
    run = Path(a.run)
    write_json(run / 'RESULT.json', {'arm': tc['arm'], 'seed': tc['seed'], 'variant': tc['variant'], 'a1': a1, 'raw': raw,
                                     'dtype': read(run / 'SCORES_COMPLETE.json')['dtype'],
                                     'exposure': 'historically exposed MVTec; exploratory only'})
    log('RESULT', tc['variant'], tc['seed'], round(a1['final_pixel_ap'], 4), round(a1['final_image_auroc'], 4))


def cpu_base(a):
    import torch
    import torch.nn as nn
    D.khang(a.data, a.weights)                       # puts Khang's code (repository root) on sys.path
    import inspect
    require('generator' in inspect.signature(nn.init.trunc_normal_).parameters, 'trunc_normal_ has no generator')
    base = base_from_seed(a.seed, torch, nn)
    out = Path(a.out)
    require(not (out / 'BASE.pt').exists(), 'preserve existing BASE')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(base, out / 'BASE.pt')
    rec = {'seed': a.seed, 'canonical_sha256': canonical_sha(base), 'file_sha256': sha(out / 'BASE.pt'),
           'generator': 'torch.Generator(cpu).manual_seed (mt19937)', 'torch': torch.__version__,
           'aten_cpu_capability_env': os.environ.get('ATEN_CPU_CAPABILITY'),
           'cpu_capability': torch.backends.cpu.get_cpu_capability(), 'os': platform.platform(),
           'machine': platform.machine(), 'processor': platform.processor(),
           'code_sha256': sha(SELF), 'khang_code_commit': D.khang_code_intact()}
    write_json(out / 'BASE_SEED.json', rec)
    log('CPU_BASE', a.seed, rec['canonical_sha256'][:16], rec['cpu_capability'])


def label(a):
    m = D.khang(a.data, a.weights)
    work = Path(a.work)
    out = assemble_labels(a.job, work, config(), m.L.ci)
    write_json(work / 'LABELS.json', out)
    log('LABELS', json.dumps({k: v for k, v in out.items() if k in ('job', 'M2', 'M1a', 'M1b')}))


def assemble_labels(job, work, cfg, ci):
    """Read RESULT / INT8_GATE / QUANT / BASE_SEED files under `work` and compute labels; `ci` is
    nar_lite_01.ci (paired category bootstrap), injected so the assembly is testable without torch."""
    out, rules = {'job': job}, cfg['rules']
    if job == 'M2':
        per, detail = {}, {}
        for s in cfg['jobs']['M2']['seeds']:
            r16, r8 = work / f's{s}' / 'bf16' / 'RESULT.json', work / f's{s}' / 'int8' / 'RESULT.json'
            if not (r16.exists() and r8.exists()):
                per[s] = None
                continue
            j16, j8 = read(r16), read(r8)
            require(j16.get('seed') == s and j16.get('variant') == 'bf16', f'{r16}: seed/variant identity')
            require(j8.get('seed') == s and j8.get('variant') == 'int8', f'{r8}: seed/variant identity')
            x16, x8 = j16['a1'], j8['a1']
            g = read(work / f's{s}' / 'int8' / 'INT8_GATE.json')
            cat = {c: x8['per_category'][c]['final']['pixel_ap'] - x16['per_category'][c]['final']['pixel_ap']
                   for c in x16['per_category']}
            per[s] = {'d_ap': x8['final_pixel_ap'] - x16['final_pixel_ap'],
                      'd_auc': x8['final_image_auroc'] - x16['final_image_auroc'], 'gate_passed': g['passed'],
                      'gate_rel': g['relative_mean_abs_error_int8_vs_fp32_in_fp32']}
            detail[s] = {**per[s], 'int8_gate': g, 'quant': read(work / f's{s}' / 'int8' / 'QUANT.json'),
                         'per_category_max_abs_d_ap': max(abs(v) for v in cat.values()),
                         'bf16': {k: x16[k] for k in ('final_pixel_ap', 'final_image_auroc')},
                         'int8': {k: x8[k] for k in ('final_pixel_ap', 'final_image_auroc')},
                         'ci_int8_minus_bf16': ci(x8, x16)}
        out.update(M2=label_m2(per, rules, cfg['jobs']['M2']['seeds']), seeds=detail)
    else:
        seeds = cfg['jobs']['M1']['seeds']
        canon = {s: [read(p)['canonical_sha256'] for p in sorted((work / f's{s}').glob('cpu_base*/BASE_SEED.json'))]
                 for s in seeds}
        deltas, detail = [], {}
        for s in seeds:
            pc, pt = work / f's{s}' / 'cpu' / 'RESULT.json', work / f's{s}' / 't4' / 'RESULT.json'
            if not (pc.exists() and pt.exists()):
                deltas.append(None)
                continue
            jc, jt = read(pc), read(pt)
            require(jc.get('seed') == s and jt.get('seed') == s, f's{s}: RESULT seed identity')
            cpu_file = read(work / f's{s}' / 'cpu_base' / 'BASE_SEED.json')['file_sha256']
            for arm, want in (('cpu', cpu_file), ('t4', cfg['checkpoints_F'][str(s)]['base_sha256'])):
                got = read(work / f's{s}' / arm / 'TRAIN_COMPLETE.json')['base_sha256']
                require(got == want, f's{s}/{arm}: trained on BASE {got[:12]}, expected {want[:12]}')
            xc, xt = jc['a1'], jt['a1']
            deltas.append(xc['final_pixel_ap'] - xt['final_pixel_ap'])
            ref = cfg['job_R_F_a1'][str(s)]
            detail[s] = {'cpu_minus_t4': deltas[-1], 'ci_cpu_minus_t4': ci(xc, xt),
                         'cpu': {k: xc[k] for k in ('final_pixel_ap', 'final_image_auroc')},
                         't4': {k: xt[k] for k in ('final_pixel_ap', 'final_image_auroc')},
                         'rerun_t4_minus_job_R': xt['final_pixel_ap'] - ref['final_pixel_ap']}
        out.update(M1a=label_m1a(canon, seeds), M1b=label_m1b(deltas, rules), canonical_sha256=canon, seeds=detail)
    return out


# ---------------------------------------------------------------- orchestration
def run_parallel(commands, gpus=None, env_extra=None, poll=2.0):
    """Each command is a full argv after the interpreter: [script, phase, args...]."""
    procs = []
    for i, cmd in enumerate(commands):
        env = os.environ.copy()
        env.update(env_extra or {})
        if gpus is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpus[i])
        cmd = [str(x) for x in cmd]
        log('START', ' '.join(cmd[1:])[:300], 'gpu', None if gpus is None else gpus[i])
        procs.append(subprocess.Popen([sys.executable, '-B', *cmd], env=env))
    while True:                                      # fail fast: stop siblings as soon as one child fails
        codes = [p.poll() for p in procs]
        if any(c not in (None, 0) for c in codes):
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            require(False, f'child failure {codes}')
        if all(c == 0 for c in codes):
            return
        time.sleep(poll)


def by_lanes(lanes, gpus):
    """One process per lane, lane i on gpus[i]; with fewer GPUs than lanes the lanes run one after another."""
    cmds = [[SELF, 'lane', '--steps', json.dumps([[str(x) for x in s] for s in lane])] for lane in lanes]
    if len(gpus) >= len(cmds):
        run_parallel(cmds, gpus=gpus[:len(cmds)])
    else:
        for i, cmd in enumerate(cmds):
            run_parallel([cmd], gpus=[gpus[i % len(gpus)]])


def lane(a):
    for step in json.loads(a.steps):
        run_parallel([step])


def job(a):
    cfg = config()
    spec = cfg['jobs'][a.job]
    work, cache = Path(a.work), Path(a.cache)
    work.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    io = ['--data', a.data, '--weights', a.weights]
    caps = spec['phase_seconds']                     # per-phase time limits (seconds)
    gpus = [int(g) for g in a.gpus.split(',')] if getattr(a, 'gpus', None) else spec['gpus']
    run_parallel([[BASE_DRIVER, 'gate', *io, '--manifests', a.khang, '--cache', cache, '--out', work,
                   *(['--base', Path(a.khang) / 'BASE.pt'] if a.job == 'M2' else [])]])
    run_parallel([[SELF, 'locate', '--inputs', a.inputs, '--out', work]])
    inputs = read(work / 'INPUTS.json')
    if a.job == 'M2':
        seeds = spec['seeds']
        runs = []
        for s in seeds:
            ck = cfg['checkpoints_F'][str(s)]
            run_parallel([[SELF, 'quantize', '--seed', s, '--stage15', inputs[str(s)]['stage15'], '--sha',
                           ck['stage15_sha256'], '--out-bf16', work / f's{s}' / 'bf16', '--out-int8', work / f's{s}' / 'int8']])
            runs += [(s, 'bf16'), (s, 'int8')]
        hist = {s: inputs[str(s)]['hist'] for s in seeds}
        by_lanes(lanes_round_robin([[SELF, 'int8-gate', *io, '--seed', s, '--hist', hist[s], '--bf16', work / f's{s}' / 'bf16',
                                     '--int8', work / f's{s}' / 'int8', '--seconds', caps['fidelity_seed']] for s in seeds],
                                   len(gpus)), gpus)
        sealed = [x for s, v in runs for x in ('--sealed', work / f's{s}' / v)]
        run_parallel([[BASE_DRIVER, 'extract-test', *io, '--dtype', 'bfloat16', '--out', cache / 'test_bfloat16', *sealed]],
                     gpus=[gpus[0]])
        by_lanes(lanes_round_robin([[BASE_DRIVER, 'score', *io, '--hist', hist[s], '--run', work / f's{s}' / v,
                                     '--dtype', 'bfloat16', '--test', cache / 'test_bfloat16'] for s, v in runs],
                                   len(gpus)), gpus)
        for i in range(0, len(runs), spec['metric_workers']):
            run_parallel([[SELF, 'metrics-plain', *io, '--run', work / f's{s}' / v, '--seconds', caps['metrics_arm']]
                          for s, v in runs[i:i + spec['metric_workers']]])
    else:
        seeds = spec['seeds']
        feat = cache / 'train_fp32'
        run_parallel([[BASE_DRIVER, 'extract-train', *io, '--feat', feat, '--manifests', a.khang]], gpus=[gpus[0]])
        cpu_env = {'ATEN_CPU_CAPABILITY': 'default', 'CUDA_VISIBLE_DEVICES': ''}
        for rep in ('cpu_base', 'cpu_base_repeat'):    # two separate processes per seed (card s6.2)
            run_parallel([[SELF, 'cpu-base', *io, '--seed', s, '--out', work / f's{s}' / rep] for s in seeds],
                         env_extra=cpu_env)
        hist = {(s, 'cpu'): work / f's{s}' / 'cpu_base' for s in seeds}
        hist.update({(s, 't4'): inputs[str(s)]['hist'] for s in seeds})
        bsha = {(s, 'cpu'): sha(work / f's{s}' / 'cpu_base' / 'BASE.pt') for s in seeds}
        bsha.update({(s, 't4'): cfg['checkpoints_F'][str(s)]['base_sha256'] for s in seeds})
        lanes = [[(s, arm) for arm in spec['arms']] for s in seeds]          # one GPU per seed: CPU BASE, then T4 BASE
        runs = [x for l in lanes for x in l]
        rdir = lambda s, arm: work / f's{s}' / arm
        by_lanes([[[BASE_DRIVER, 'train', *io, '--arm', 'F', '--seed', s, '--hist', hist[s, arm], '--base-sha', bsha[s, arm],
                    '--feat', feat, '--out', rdir(s, arm), '--seconds', caps['train_arm']] for s, arm in l] for l in lanes],
                 gpus)
        by_lanes([[[BASE_DRIVER, 'fidelity', *io, '--hist', hist[s, arm], '--run', rdir(s, arm),
                    '--seconds', caps['fidelity_arm']] for s, arm in l] for l in lanes], gpus)
        dtype = {s: D.common_dtype([read(rdir(s, arm) / 'FIDELITY.json') for arm in spec['arms']]) for s in seeds}
        write_json(work / 'PRECISION.json', {str(s): d for s, d in dtype.items()})
        sealed = [x for s, arm in runs for x in ('--sealed', rdir(s, arm))]
        for d in sorted(set(dtype.values())):
            run_parallel([[BASE_DRIVER, 'extract-test', *io, '--dtype', d, '--out', cache / f'test_{d}', *sealed]],
                         gpus=[gpus[0]])
        by_lanes([[[BASE_DRIVER, 'score', *io, '--hist', hist[s, arm], '--run', rdir(s, arm), '--dtype', dtype[s],
                    '--test', cache / f'test_{dtype[s]}'] for s, arm in l] for l in lanes], gpus)
        for i in range(0, len(runs), spec['metric_workers']):
            run_parallel([[BASE_DRIVER, 'metrics', *io, '--run', rdir(s, arm), '--seconds', caps['metrics_arm']]
                          for s, arm in runs[i:i + spec['metric_workers']]])
    run_parallel([[SELF, 'label', *io, '--job', a.job, '--work', work]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='phase', required=True)

    def cmd(name, fn, io, *extra):
        q = sub.add_parser(name)
        if io:
            q.add_argument('--data', required=True)
            q.add_argument('--weights', required=True)
        for args, kw in extra:
            q.add_argument(*args, **kw)
        q.set_defaults(fn=fn)

    req = lambda *n, **kw: (n, {'required': True, **kw})
    cmd('locate', locate, False, req('--inputs'), req('--out'))
    cmd('quantize', quantize, False, req('--seed', type=int), req('--stage15'), req('--sha'), req('--out-bf16'),
        req('--out-int8'))
    cmd('int8-gate', int8_gate, True, req('--seed', type=int), req('--hist'), req('--bf16'), req('--int8'),
        req('--seconds', type=float))
    cmd('metrics-plain', metrics_plain, True, req('--run'), req('--seconds', type=float))
    cmd('cpu-base', cpu_base, True, req('--seed', type=int), req('--out'))
    cmd('label', label, True, req('--job', choices=('M2', 'M1')), req('--work'))
    cmd('job', job, True, req('--job', choices=('M2', 'M1')), req('--inputs'), req('--khang'), req('--work'),
        req('--cache'), (('--gpus',), {'help': 'comma-separated GPU ids overriding the config, e.g. 0'}))
    cmd('lane', lane, False, req('--steps'))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
