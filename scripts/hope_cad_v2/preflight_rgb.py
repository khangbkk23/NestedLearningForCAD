"""Real frozen-extractor cache parity and exact RGB event-resume smoke on prefix normals."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from dataset.hope_mask_views import mask_normalized_rgb
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.teacher_head import frozen_teacher
from models.hope_cad_v2.benchmark_adapter import HOPEBenchmarkAdapterV3
from training.hope_outer_learning_v2 import export_initialization,capture_rng,restore_rng
from training.hope_feature_contract_v3 import load_context,validate_bundle
from training.hope_runtime_v3 import configure_runtime,source_identity,runtime_identity,atomic_json,resource_usage,SPECS
from training.hope_benchmark_engine_v3 import HOPEBenchmarkEngineV3,AtomicBenchmarkArtifacts
from training.benchmark_engine_v1 import _state_equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',default='conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml')
    p.add_argument('--baseline-method',default='conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    for key in ('lock-dir','bundle','output-dir'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    configure_runtime()
    torch.manual_seed(103)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(status='failed',passed=False,reportable=False,random_unselected_initialization=True,
                official_test_accessed=False,source_sha256=source_identity(),runtime=runtime_identity())
    extractor=None
    started=time.perf_counter()
    try:
        context=load_context(args.protocol,args.baseline_method,args.lock_dir)
        bundle=validate_bundle(torch.load(args.bundle,map_location='cpu',weights_only=False),context)
        method=context['method']; pre=method['preprocessing']
        rows=context['roles']['roles']['meta_support'][:9]
        dataset=HashedNormalDataset(context['protocol']['dataset']['root'],rows,pre['image_size'],pre['mean'],pre['std'],True)
        config=CADICViTConfig(checkpoint_path=method['extractor']['checkpoint_path'],
                             checkpoint_identity=method['extractor']['checkpoint_identity'])
        extractor=CADICViTFeatureExtractor(config,args.device)
        if str(args.device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(args.device)
        parity=[]
        for i in range(2):
            image=dataset[i]['images'].unsqueeze(0).to(args.device)
            cached=bundle['images'][rows[i]['relative_path']]['tensors']
            for view in range(-1,4):
                actual=(frozen_teacher(extractor.extract_patch_features(image)) if view==-1 else
                        extractor.extract_patch_features(mask_normalized_rgb(image,view)))
                expected=cached['teacher' if view==-1 else f'masked_{view}'].to(args.device)
                torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
                parity.append(dict(identity=rows[i]['relative_path'],view=view,
                                   max_absolute_error=(actual-expected).abs().max().item()))
        hashes=source_identity()
        initial=export_initialization(HOPEFeaturePredictor(),dict(roles_digest=context['roles']['digest'],
            schedule_digest='preflight-only',source_sha256=hashes,spec_sha256={k:hashes[k] for k in SPECS},
            feature_identity=bundle['feature_identity']))
        def make_adapter():
            return HOPEBenchmarkAdapterV3(initial,checkpoint_path=config.checkpoint_path,
                checkpoint_sha256=method['extractor']['checkpoint_sha256'],checkpoint_identity=config.checkpoint_identity,
                device=args.device,extractor=extractor,allow_test_components=True,
                protocol_identity={'scope':'random prefix-only preflight','manifest_digest':context['manifest']['digest']})
        class PrefixStream:
            task_names=['prefix_part_1','prefix_part_2']
            manifest=context['manifest']
            def build_train_loader(self,i,cap=None):
                from torch.utils.data import Subset
                indices=range(0,6) if i==0 else range(6,9)
                return DataLoader(Subset(dataset,indices),batch_size=1,shuffle=False,
                                  generator=torch.Generator().manual_seed(0))
        full,part=make_adapter(),make_adapter()
        rng=capture_rng()
        full_engine=HOPEBenchmarkEngineV3({},full,AtomicBenchmarkArtifacts(args.output_dir/'uninterrupted'),args.device)
        full_engine.train(PrefixStream(),checkpoint_every=1)
        expected_state=full.session.checkpoint(); expected_rng=capture_rng()
        batch=next(iter(PrefixStream().build_train_loader(0)))
        expected_scores=full.score_batch(batch)
        restore_rng(rng)
        part_engine=HOPEBenchmarkEngineV3({},part,AtomicBenchmarkArtifacts(args.output_dir/'resumed'),args.device)
        part_engine.train(PrefixStream(),checkpoint_every=1,stop_after_events=5)
        resumed=make_adapter()
        resumed_engine=HOPEBenchmarkEngineV3({},resumed,AtomicBenchmarkArtifacts(args.output_dir/'resumed'),args.device)
        resumed_engine.train(PrefixStream(),checkpoint_every=1,resume=True)
        if not _state_equal(expected_rng,capture_rng()) or not _state_equal(expected_state,resumed.session.checkpoint()):
            raise RuntimeError('RGB resume weights/accumulators/RNG differ from uninterrupted run')
        before=resumed.session.checkpoint()
        actual=resumed.score_batch(batch)
        if not _state_equal(before,resumed.session.checkpoint()) or not _state_equal(actual,expected_scores):
            raise RuntimeError('RGB scoring changed state or resumed scores differ')
        report.update(status='passed',passed=True,cache_parity=parity,
            resume_events=9,interruption_after=5,commits=list(resumed.session.state.commits),
            boundary_scope='simulated task boundary after event 6; all images from meta_support',
            exact_tensor_rng_score_resume=True,resources=resource_usage(args.device),
            manifest_digest=context['manifest']['digest'],baseline_lock_digest=context['lock']['digest'])
    except (ValueError,RuntimeError,AssertionError,FloatingPointError) as error:
        report['error']=str(error)
    finally:
        if extractor: extractor.close()
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_json(report,args.output_dir/'preflight.json')
    print(report['status'],flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
