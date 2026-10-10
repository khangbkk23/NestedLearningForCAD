"""Prepare/execute the explicit W1 matched CADIC route; no modification of legacy runners."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from training.benchmark_config_v1 import load_configs
from training.benchmark_config_v2 import validate_configs
from training.benchmark_artifacts_v1 import BenchmarkArtifacts
from training.benchmark_engine_v1 import BenchmarkEngineV1
from dataset.benchmark_protocol_v2 import build_prefix_manifest, validate_prefix_manifest, calibration_roles, PrefixContinualProtocol
from dataset.benchmark_manifest_v1 import canonical_digest
from models.benchmark_registry_v2 import build_adapter
from scripts.hope_cad_v2.artifacts import provenance, write_json


def baseline_identity(protocol, method, resolved, manifest_digest, *, seed, smoke, max_tasks, max_train_images):
    protocol, method = copy.deepcopy(protocol), copy.deepcopy(method)
    # Machine mount paths are transport details. Content hashes and relative
    # identities remain strict so Kaggle session changes are resumable.
    protocol['dataset'].pop('root',None)
    method['extractor'].pop('checkpoint_path',None)
    return dict(protocol=protocol,method=method,effective_route=resolved,manifest_digest=manifest_digest,
                seed=seed,smoke=smoke,max_tasks=max_tasks,max_train_images=max_train_images)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', default='conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml')
    p.add_argument('--method', default='conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--phase', choices=('prepare','train','eval','fm'), default='prepare')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--max-train-images', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if (args.max_tasks is not None or args.max_train_images is not None) and not args.smoke:
        p.error('caps are smoke-only')
    protocol, method = load_configs(args.protocol, args.method)
    resolved = validate_configs(protocol, method, smoke=args.smoke)
    manifest_path = args.run_dir/'manifest_train.json'
    evidence = provenance()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        validate_prefix_manifest(manifest, protocol)
    else:
        manifest = build_prefix_manifest(protocol, seed=args.seed, source_commit=evidence['source_commit'])
    identity = baseline_identity(protocol,method,resolved,manifest['digest'],seed=args.seed,smoke=args.smoke,
                                 max_tasks=args.max_tasks,max_train_images=args.max_train_images)
    lock = dict(identity=identity, digest=canonical_digest(identity))
    lock_path = args.run_dir/'baseline_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding='utf-8')) != lock:
            raise ValueError('baseline lock differs; refuse route/config/data/source drift')
    elif args.phase != 'prepare':
        raise ValueError('prepare and lock baseline before executing')
    art = BenchmarkArtifacts(args.run_dir)
    if args.phase == 'prepare':
        if not lock_path.exists():
            art.initialize(protocol, method, manifest, evidence, args.smoke, resolved['reportable'])
            write_json(lock_path, lock)
            write_json(args.run_dir/'calibration_roles.json', calibration_roles(manifest))
            write_json(args.run_dir/'provenance.json', evidence)
            write_json(args.run_dir/'run.json', dict(seed=args.seed, smoke=args.smoke, reportable=False,
                                                    protocol_eligible=resolved['reportable'], status='prepared_not_run'))
        print(f'Baseline locked: {lock_path}')
        return
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = PrefixContinualProtocol(protocol, manifest, method, args.seed)
    adapter = build_adapter(method, args.device)
    engine = BenchmarkEngineV1(protocol, adapter, art, args.device, fail_on_state_mutation=True)
    if args.phase == 'train':
        if any(art.states.iterdir()) and not args.resume:
            raise ValueError('existing states require explicit --resume')
        engine.train(dataset, args.max_tasks, args.max_train_images, args.resume)
    else:
        if protocol['dataset']['name'] == 'VisA':
            from dataset.visa_mask_qa_v2 import mask_qa
            write_json(args.run_dir/'visa_mask_qa.json', mask_qa(dataset))
        if args.phase == 'eval':
            adapter.load_state_dict(torch.load(art.states/'final.pt', map_location=args.device, weights_only=False))
            engine.evaluate_final(dataset, args.max_tasks)
        else:
            engine.evaluate_forgetting(dataset, args.max_tasks)
    write_json(args.run_dir/f'{args.phase}_evidence.json', dict(provenance=evidence,
               effective_method=adapter.method_metadata(), runtime=engine.times, smoke=args.smoke,
               status='phase_complete', reportable=False))


if __name__ == '__main__': main()
