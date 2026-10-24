"""Standalone K2500/K5000 CADIC timing/QA, isolated from study acceptance."""
import gc
import json
import math
import os
import platform
import time
import uuid
from pathlib import Path

import torch

from hope_cad.contract import CONTRACT, digest, file_sha
from hope_cad.baselines.cadic.bank import CADICPatchCoresetConfig, CADICPatchCoresetV1
from hope_cad.baselines.cadic.cache import CachedPairCoreset
from hope_cad.benchmark.engine import Pause, boundary, cell, save_or_verify
from hope_cad.benchmark.stream import _state_equal
from hope_cad.data.fulltrain import build_manifest
from hope_cad.models.primitives.state import finite_tensors
from hope_cad.paths import session_export_directory
from hope_cad.runtime.backend import activation
from hope_cad.runtime.backend import ORIGINAL_NEAREST
from hope_cad.runtime.evidence import export_archive
from hope_cad.runtime.execution import exclusive_work, verify_checkpoint
from hope_cad.runtime.identity import configure_runtime, check_remote_runtime, output_directory, source_digest
from hope_cad.runtime.system import atomic_json, atomic_torch_save, resource_usage, lock_json, capture_rng

SCHEMA = 'cadic_fulltrain_diagnostic_v1'
PARTS = ('input_seconds', 'features_seconds', 'update_seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def host_information():
    cpu = Path('/proc/cpuinfo')
    model = platform.processor()
    if cpu.is_file():
        model = next((line.split(':', 1)[1].strip() for line in cpu.read_text().splitlines()
                      if line.startswith('model name')), model)
    quota = Path('/sys/fs/cgroup/cpu.max')
    return dict(instance_id=os.environ.get('CONTAINER_ID'), hostname=platform.node(), cpu_model=model,
                cpu_affinity_count=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                cpu_max=quota.read_text().strip() if quota.is_file() else None,
                platform=platform.platform())


class Inputs:
    def __init__(self, root, extractor, device):
        self.root, self.extractor, self.device = root, extractor, device

    def get(self, row):
        from hope_cad.data.prefix import HashedNormalDataset
        if row['split'] != 'train' or '/train/good/' not in row['relative_path']:
            raise ValueError('Diagnostic training accepts train/good only')
        sync(self.device)
        start = time.perf_counter()
        g = CONTRACT['geometry']
        image = HashedNormalDataset(self.root, [row], 224, g['mean'], g['std'], True)[0]['images']
        input_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with torch.no_grad():
            raw = self.extractor.extract_patch_features(image[None].to(self.device)).detach().contiguous()
        if raw.shape != (1, 784, 768) or raw.dtype != torch.float32:
            raise ValueError('Canonical raw feature geometry changed')
        finite_tensors((raw,), 'diagnostic features')
        sync(self.device)
        return raw, dict(input_seconds=input_seconds, features_seconds=time.perf_counter()-start)


def load_active(folder, bank, identity, rows):
    pointer = read(folder/'active.json')
    if pointer.get('slot') not in (0, 1):
        raise ValueError('Invalid diagnostic checkpoint slot')
    path = folder/f"state_{pointer['slot']}.pt"
    if file_sha(path) != pointer['sha256']:
        raise ValueError('Diagnostic checkpoint hash mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('schema') != SCHEMA or value.get('identity') != identity:
        raise ValueError('Diagnostic state/source/runtime identity changed')
    cursor, records = value['cursor'], value['records']
    if (type(cursor) is not int or not 0 <= cursor <= len(rows) or len(records) != cursor
            or [r['identity'] for r in records] != [r['relative_path'] for r in rows[:cursor]]
            or pointer['cursor'] != cursor):
        raise ValueError('Diagnostic checkpoint prefix mismatch')
    bank.load_state_dict(value['bank'])
    if bank.update_batches != cursor:
        raise ValueError('Bank/image clock mismatch')
    finite_tensors((bank.features,), 'restored bank')
    return value


def train_case(folder, rows, provider, bank, identity, *, deadline=None, checkpoint_every=64):
    """Direct bank train; fixture-callable on CPU, production passes full real rows."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lock_json(folder/'job_lock.json', identity)
    records = []
    checkpoint_seconds = 0.
    if (folder/'active.json').exists():
        value = load_active(folder, bank, identity, rows)
        records = value['records']
        checkpoint_seconds = read(folder/'active.json').get('checkpoint_seconds', 0.)
    elif (folder/'train_complete.json').exists():
        raise ValueError('Completion without committed diagnostic state')
    if (folder/'train_complete.json').exists():
        done = read(folder/'train_complete.json')
        if (len(records) != len(rows) or done['identity_digest'] != digest(identity)
                or done['active_receipt_sha256'] != file_sha(folder/'active.json')):
            raise ValueError('Invalid diagnostic training completion')
        return done

    def save():
        nonlocal checkpoint_seconds
        checkpoint_start = time.perf_counter()
        slot = 1-read(folder/'active.json')['slot'] if (folder/'active.json').exists() else 0
        state = dict(schema=SCHEMA, identity=identity, cursor=len(records), records=records,
                     bank=bank.state_dict())
        path = folder/f'state_{slot}.pt'
        atomic_torch_save(state, path)
        hashed = file_sha(path)
        checkpoint_seconds += time.perf_counter()-checkpoint_start
        atomic_json(dict(slot=slot, cursor=len(records), sha256=hashed,
                         checkpoint_seconds=checkpoint_seconds), folder/'active.json')

    if not (folder/'active.json').exists():
        save()
    for i in range(len(records), len(rows)):
        try:
            boundary(deadline)
        except Pause:
            save()
            raise
        raw, timings = provider.get(rows[i])
        finite_tensors((raw,), 'diagnostic input')
        sync(bank.device)
        started = time.perf_counter()
        change = bank.update(raw)
        sync(bank.device)
        timings = dict(timings, update_seconds=time.perf_counter()-started)
        if (bank.count > bank.config.budget or bank.update_batches != i+1
                or change['incoming'] != change['accepted']+change['replaced']+change['rejected']
                or any(not math.isfinite(timings[k]) or timings[k] < 0 for k in PARTS)):
            raise ValueError('Invalid diagnostic bank/counter/timing state')
        records.append(dict(identity=rows[i]['relative_path'], task=rows[i]['task_name'],
                            cursor=i+1, **timings, **change))
        task_end = i+1 == len(rows) or rows[i]['task_name'] != rows[i+1]['task_name']
        if (i+1) % checkpoint_every == 0 or task_end:
            save()
        if (i+1) % 16 == 0 or task_end:
            print('[CADIC DIAGNOSTIC]', bank.config.budget, i+1, '/', len(rows),
                  'update_s', round(sum(r['update_seconds'] for r in records), 3), flush=True)
    finite_tensors((bank.features,), 'final diagnostic bank')
    # State has just been committed at the terminal image; no additional update.
    if bank.count != min(bank.config.budget, bank.seen_features):
        raise ValueError('Final diagnostic bank occupancy mismatch')
    totals = {k: sum(r[k] for r in records) for k in PARTS}
    by_task = {}
    for r in records:
        task = by_task.setdefault(r['task'], dict(images=0, replaced=0, **{k: 0. for k in PARTS}))
        task['images'] += 1
        task['replaced'] += r['replaced']
        for k in PARTS:
            task[k] += r[k]
    done = dict(status='train_complete', identity_digest=digest(identity), images=len(rows),
                active_receipt_sha256=file_sha(folder/'active.json'), stats=bank.stats(),
                checkpoint_tensor_and_hash_seconds=checkpoint_seconds,
                pipeline_seconds=sum(totals.values()), **totals, per_category=by_task,
                scope='Committed image input/features/update spans only; excludes checkpoint/QA/export, not author timing parity')
    atomic_json(done, folder/'train_complete.json')
    return done


class Scores:
    def __init__(self, bank):
        self.bank = bank

    def state_dict(self):
        return self.bank.state_dict()

    def score(self, raw):
        return self.bank.pixel_scores(raw, b=9)


def operator_parity(bank, raw):
    """Independent original NN on the reference update as well as scoring."""
    before, rng = bank.state_dict(), capture_rng()
    ref = CADICPatchCoresetV1(bank.config, device=bank.device)
    fast = CachedPairCoreset(bank.config, device=bank.device)
    ref.load_state_dict(before)
    fast.load_state_dict(before)
    ref._nearest = ORIGINAL_NEAREST.__get__(ref, type(ref))
    rv, ri = ref._closest_pair()
    fv, fi = fast._closest_pair()
    if not torch.equal(rv, fv) or ri != fi:
        raise ValueError('Diagnostic reference pair differs')
    candidates, query = raw[0, :4], raw[:, :7]
    if (ref.update(candidates) != fast.update(candidates)
            or not _state_equal(ref.state_dict(), fast.state_dict())
            or not _state_equal(ref.pixel_scores(query), fast.pixel_scores(query))):
        raise ValueError('Diagnostic reference update/score differs')
    if not _state_equal(before, bank.state_dict()) or not _state_equal(rng, capture_rng()):
        raise ValueError('Diagnostic QA changed main state/RNG')
    return dict(passed=True, original_NN_for_update_and_score=True, pair_exact=True,
                bank_count=bank.count, candidate_vectors=4,
                scope='Discarded four-patch operator fixture, not universal author parity')


def final_qa(bank, raw, folder, identity, rows):
    """Read-only normal diagnostics, explicitly outside training timing."""
    before = bank.state_dict()
    other = CachedPairCoreset(bank.config, device=bank.device)
    load_active(Path(folder), other, identity, rows)
    if not _state_equal(before, other.state_dict()):
        raise ValueError('Bank roundtrip mismatch')
    # The clone uses the original NN writes even when the outer train loop has
    # activated fixed-shape writes. This checks NN/Eq9, not just pair caching.
    other._nearest = ORIGINAL_NEAREST.__get__(other, type(other))
    with torch.no_grad():
        scored = bank.pixel_scores(raw, b=9)
        restored = other.pixel_scores(raw, b=9)
    finite_tensors(scored, 'diagnostic normal scores')
    if not _state_equal(scored, restored) or not _state_equal(before, bank.state_dict()):
        raise ValueError('Scoring/roundtrip/read-only failure')
    parity = operator_parity(bank, raw)
    return dict(passed=True, reference=parity, bank_roundtrip=True, score_read_only=True,
                original_NN_Eq9_exact=True, serialized_checkpoint_verified=True,
                interpretation='finite normal/operator fixture, not anomaly quality')


def evaluate_case(folder, bank, rows, tasks, provider, identity, deadline):
    from hope_cad.benchmark.metrics import macro_task_metrics
    before = bank.state_dict()
    per_task = {}
    for i, task in enumerate(tasks):
        selected = [r for r in rows if r['task_id'] == i]
        per_task[task] = cell(folder, len(tasks)-1, i, Scores(bank), selected, provider,
                              identity, deadline=deadline)
    if not _state_equal(before, bank.state_dict()):
        raise ValueError('Evaluation changed trained bank')
    macro = macro_task_metrics(per_task)
    macro['macro_final_p_ap'] = macro['p_aupr']
    save_or_verify(folder/'final_per_task.json', per_task)
    save_or_verify(folder/'final_macro.json', macro)
    return macro


def run(c):
    """No study preflight files are read, accepted, generated, or rewritten."""
    hours = c['max_hours']
    if not math.isfinite(hours) or not 0 < hours <= 4:
        raise ValueError('max-hours must be in (0,4]')
    budgets = c['budgets']
    if not budgets or budgets != sorted(set(budgets)) or any(k not in (2500, 5000) for k in budgets):
        raise ValueError('Use distinct ascending diagnostic budgets2500/5000')
    if c['phase'] not in ('train', 'evaluate'):
        raise ValueError('Invalid diagnostic phase')
    configure_runtime()
    work = output_directory(c['run_dir'], data_root=c['data_root'], checkpoint=c['checkpoint'])
    started = time.monotonic()
    deadline = started+max(0., hours*3600-30.)
    attempt = uuid.uuid4().hex
    extractor = None
    failure = None
    status = 'failed_preserve_evidence'
    phases = []
    with exclusive_work(work):
        # Never overwrite research handoff/locks even when --run-dir is mistyped.
        existing = {p.name for p in work.iterdir()}- {'.writer.lock'}
        foreign = existing & {'suite_lock.json','preflight.json','pilot.json','preflight_review.json','pilot_review.json'}
        diagnostic_owned = ((work/'diagnostic_lock.json').is_file()
                            or ((work/'handoff.json').is_file() and read(work/'handoff.json').get('schema') == SCHEMA))
        if foreign or (existing and not diagnostic_owned):
            raise ValueError('Diagnostic requires a fresh or diagnostic-owned run directory')
        receipt = work/'attempts'/(attempt+'.json')
        atomic_json(dict(status='running', phase=c['phase'], started_unix=time.time()), receipt)
        try:
            actual = torch.cuda.get_device_name(c['device']) if torch.cuda.is_available() else ''
            rt = check_remote_runtime(c['device'], c.get('expected_gpu') or actual)
            torch.manual_seed(0)
            # Kernel smoke: neither a scientific preflight nor a performance warmup loop.
            x = torch.ones((4, 4), device=c['device'])
            if not torch.equal(x@x, torch.full_like(x, 4.)):
                raise ValueError('CUDA smoke failed')
            sync(c['device'])
            manifest, rd = build_manifest(c['data_root'], source_digest())
            checkpoint = verify_checkpoint(c['checkpoint'])
            identity = dict(schema=SCHEMA, source_digest=source_digest(), runtime=rt,
                            normal_rows_digest=rd, backbone_sha256=checkpoint, budgets=budgets,
                            producer='ViT_block8_raw784x768_ImageNet_direct224bilinear_FP32',
                            backend='cached_pair_fixed_shape_NN_imagewise_Eq9b9', threads=2,
                            host=host_information())
            lock_json(work/'diagnostic_lock.json', identity)
            save_or_verify(work/'manifest_train.json', manifest)
            if c['phase'] == 'evaluate' and not all((work/f'K{k}/qa.json').is_file() for k in budgets):
                raise ValueError('Finish training and QA for all selected budgets before test access')
            from hope_cad.models.features import CADICViTConfig, CADICViTFeatureExtractor
            extractor = CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],
                checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'), c['device'])
            provider = Inputs(c['data_root'], extractor, c['device'])
            if c['phase'] == 'evaluate':
                from hope_cad.benchmark.data import EvaluationFeatures, test_manifest
                tm = test_manifest(c['data_root'], manifest['task_order'])
                if len(tm['rows']) != 1725:
                    raise ValueError('Expected1725 official test images')
                save_or_verify(work/'test_manifest.json', tm)
            with activation(True):
                for k in budgets:
                    boundary(deadline)
                    folder = work/f'K{k}'
                    bank = CachedPairCoreset(CADICPatchCoresetConfig(budget=k, chunk_size=256,
                        query_chunk_size=256, pair_chunk_size=256), device=c['device'])
                    ji = dict(suite_digest=digest(identity), budget=k, bank_config=bank.config.__dict__)
                    phase_start = time.monotonic()
                    sync(c['device'])
                    torch.cuda.reset_peak_memory_stats(c['device'])
                    try:
                        if c['phase'] == 'train':
                            train_start = time.monotonic()
                            done = train_case(folder, manifest['online'], provider, bank, ji, deadline=deadline)
                            train_wall = time.monotonic()-train_start
                            qa_start = time.monotonic()
                            if not (folder/'qa.json').exists():
                                boundary(deadline)
                                raw, _ = provider.get(manifest['online'][0])
                                qa = final_qa(bank, raw, folder, ji, manifest['online'])
                                qa['active_receipt_sha256'] = file_sha(folder/'active.json')
                                atomic_json(qa, folder/'qa.json')
                            elif (read(folder/'qa.json').get('passed') is not True
                                  or read(folder/'qa.json')['active_receipt_sha256'] != file_sha(folder/'active.json')):
                                raise ValueError('QA receipt/checkpoint changed')
                            phases.append(dict(budget=k, phase='train_and_QA', status='complete',
                                invocation_seconds=time.monotonic()-phase_start, pipeline_seconds=done['pipeline_seconds'],
                                train_invocation_wall_seconds=train_wall,
                                qa_invocation_wall_seconds=time.monotonic()-qa_start,
                                resources=resource_usage(c['device'])))
                        else:
                            done = read(folder/'train_complete.json')
                            load_active(folder, bank, ji, manifest['online'])
                            if (done['active_receipt_sha256'] != file_sha(folder/'active.json')
                                    or read(folder/'qa.json')['active_receipt_sha256'] != file_sha(folder/'active.json')
                                    or done['identity_digest'] != digest(ji)):
                                raise ValueError('Evaluation training/QA binding changed')
                            ei = dict(diagnostic_digest=digest(identity), test_manifest_digest=tm['digest'],
                                      trained_active_sha256=file_sha(folder/'active.json'), budget=k)
                            evaluate_case(folder, bank, tm['rows'], manifest['task_order'],
                                EvaluationFeatures(c['data_root'], extractor, c['device']), ei, deadline)
                            phases.append(dict(budget=k, phase='final_evaluation', status='complete',
                                invocation_seconds=time.monotonic()-phase_start, resources=resource_usage(c['device'])))
                    except BaseException:
                        phases.append(dict(budget=k, phase=c['phase'], status='incomplete',
                            invocation_seconds=time.monotonic()-phase_start, resources=resource_usage(c['device'])))
                        raise
                    del bank
                    gc.collect()
                    torch.cuda.empty_cache()
            status = 'diagnostic_train_complete' if c['phase'] == 'train' else 'diagnostic_final_evaluation_complete'
        except Pause:
            status = 'paused_at_committed_boundary'
        except BaseException as exc:
            failure = exc
        finally:
            if extractor is not None:
                try:
                    extractor.close()
                except BaseException as exc:
                    failure = failure or exc
            if failure:
                status = 'failed_preserve_evidence'
            cases = {}
            for k in budgets:
                folder = work/f'K{k}'
                case = {}
                for field, name in (('train','train_complete.json'), ('QA','qa.json'), ('metrics','final_macro.json')):
                    if (folder/name).is_file():
                        case[field] = read(folder/name)
                if case:
                    cases[str(k)] = case
            result = dict(schema=SCHEMA, status=status, phase=c['phase'], attempt=attempt,
                error=repr(failure) if failure else None, phases=phases,
                invocation_seconds_before_export=time.monotonic()-started,
                scientific_acceptance=False, exact_paper_parity_claim=False,
                scope='Diagnostic only; not research preflight/pilot acceptance; no automatic rental control')
            atomic_json(result, receipt)
            atomic_json(result, work/'handoff.json')
            atomic_json(dict(schema=SCHEMA, status=status, cases=cases,
                paper_reference_seconds={'2500':63, '5000':379},
                comparison_scope='Author timing inclusions/numerical parity not verified; do not infer speedup',
                lock_digest=file_sha(work/'diagnostic_lock.json') if (work/'diagnostic_lock.json').is_file() else None),
                work/'summary.json')
        slot = session_export_directory(work)
        export = export_archive(work, slot/'evidence.zip', schema=SCHEMA)
        publication = dict(archive=export, attempt=attempt, status=status,
            invocation_seconds_including_export=time.monotonic()-started, not_an_invoice=True)
        atomic_json(publication, slot/'timing.json')
    if failure:
        raise RuntimeError('Diagnostic failed; evidence preserved at '+export['path']) from failure
    return dict(status=status, cases=cases, evidence=export, timing=publication)
