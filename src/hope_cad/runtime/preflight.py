"""Discarded numerical/normal acceptance on the actual device; never a new model fit."""
import json,time,gc
from pathlib import Path

def run(c):
    import torch
    from hope_cad.runtime.identity import configure_runtime,runtime_identity,check_remote_runtime,source_digest,output_directory
    from hope_cad.runtime.execution import read,normal_manifest,verify_checkpoint,exclusive_work
    from hope_cad.runtime.numerical import canonical_meta_gate
    from hope_cad.runtime.backend import activation
    from hope_cad.runtime.evidence import export_run
    from hope_cad.contract import design_digest,digest,CONTRACT
    from hope_cad.runtime.system import atomic_json,resource_usage
    from hope_cad.benchmark.stream import _state_equal
    from hope_cad.models.session import SupportSession
    from hope_cad.baselines.cadic.bank import CADICPatchCoresetConfig,CADICPatchCoresetV1
    from hope_cad.baselines.cadic.cache import reference_parity
    work=output_directory(c['run_dir'],data_root=c.get('data_root'),checkpoint=c.get('checkpoint'))
    work.mkdir(parents=True,exist_ok=True)
    if (work/'preflight.json').exists() or (work/'suite_lock.json').exists():
        raise ValueError('Use a fresh preflight directory; no automatic overwrite/retry of a completed attempt')
    configure_runtime();extractor=None;started=time.monotonic();failure=None
    result=dict(status='failed_preserve_evidence',remote_acceptance=False,reportable=False,official_test_accessed=False)
    with exclusive_work(work):
        try:
            if c.get('local_smoke'):
                if c['device']!='cpu':raise ValueError('local-smoke is CPU only and cannot authorize remote train')
                rt=runtime_identity('cpu');binding=dict(source_digest=source_digest(),design_digest=design_digest(),runtime=rt)
            else:
                if not c.get('data_root') or not c.get('checkpoint'):raise ValueError('Remote preflight requires explicit data-root and checkpoint')
                rt=check_remote_runtime(c['device'],c['expected_gpu'])
                checkpoint_sha=verify_checkpoint(c['checkpoint'])
                manifest,rows_digest=normal_manifest(c['data_root'])
                binding=dict(source_digest=source_digest(),design_digest=design_digest(),runtime=rt,
                    normal_rows_digest=rows_digest,backbone_sha256=checkpoint_sha)
                atomic_json(manifest,work/'manifest_train.json')
            result.update(binding=binding,binding_digest=digest(binding))
            print('[PREFLIGHT] runtime/source checked; exact meta numerical gate',flush=True)
            result['numerical']=canonical_meta_gate(c['device'])
            # Compare the relocated fixed-shape operator against its original on this device.
            cfg=CADICPatchCoresetConfig(budget=17,dim=8,chunk_size=5,query_chunk_size=7,pair_chunk_size=5)
            gen=torch.Generator().manual_seed(12)
            bank=torch.randn(17,8,generator=gen).to(c['device']);query=torch.randn(19,8,generator=gen).to(c['device'])
            original=CADICPatchCoresetV1(cfg,device=c['device']);original.features=bank
            ref=original._nearest(query,bank)
            with activation(True):candidate=original._nearest(query,bank)
            if not _state_equal(ref,candidate):raise RuntimeError('Fixed-shape nearest parity failed')
            result['nearest_parity']=True
            if c.get('local_smoke'):
                result['status']='local_synthetic_smoke_passed'
            else:
                from hope_cad.models.features import CADICViTConfig,CADICViTFeatureExtractor
                from hope_cad.benchmark.pilot import NormalFeatureCache
                extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),c['device'])
                provider=NormalFeatureCache(c['data_root'],extractor,work/'discarded_cache',work/'feature_receipts',c['device'])
                rows=manifest['online'][:24];measurements={}
                with activation(True):
                    for mode in ('NESTED','CADIC'):
                        actor=SupportSession(mode,device=c['device'],seed=0);middle=None;start=time.monotonic()
                        for i,row in enumerate(rows):
                            actor.observe(provider.get(row),row['relative_path'])
                            if i==19:middle=actor.state_dict()
                            if (i+1)%4==0:print('[PREFLIGHT]',mode,i+1,'/24 discarded normals',flush=True)
                        clone=SupportSession(mode,device=c['device'],seed=0);clone.load_state_dict(middle)
                        for row in rows[20:24]:clone.observe(provider.get(row),row['relative_path'])
                        if not _state_equal(actor.state_dict(),clone.state_dict()):raise RuntimeError('Pending/partial-window resume parity failed')
                        raw=provider.get(rows[-1]);before=actor.state_dict()
                        with torch.no_grad():scores=actor.score(raw)
                        if not _state_equal(before,actor.state_dict()) or any(not bool(torch.isfinite(v).all()) for v in scores):
                            raise RuntimeError('Scoring nonfinite or not read-only')
                        if mode=='NESTED' and (not actor.bank.is_full or actor.learner.slow_updates<1):
                            raise RuntimeError('Real preflight missed full bank or slow update')
                        measurements[mode]=dict(images=24,extra_resume_images=4,elapsed_seconds=time.monotonic()-start,
                            state_resume_exact=True,score_read_only=True,counters=actor.counters,memory=actor.memory_profile(),
                            slow_updates=actor.learner.slow_updates if actor.learner else 0)
                        if mode=='NESTED':
                            result['slow_gradient_trace']=actor.learner.last_slow
                        del clone,middle,before;gc.collect()
                        if mode!='CADIC':del actor
                        gc.collect();torch.cuda.empty_cache()
                # The outer optimized activation is now closed: test the real pair-cache
                # against the original operator, then compare full-budget Eq9 with optimized NN.
                result['full_budget_pair_parity']=reference_parity(actor.bank,raw[0,:4],raw[:,:7])
                before=actor.state_dict()
                with torch.no_grad():reference_score=actor.score(raw)
                with activation(True),torch.no_grad():optimized_score=actor.score(raw)
                if not _state_equal(reference_score,optimized_score) or not _state_equal(before,actor.state_dict()):
                    raise RuntimeError('Full-budget raw Eq9/optimized NN parity failed')
                result['full_budget_NN_Eq9_exact']=True
                del actor,before;gc.collect();torch.cuda.empty_cache()
                result.update(status='remote_preflight_passed_pending_review',remote_acceptance=True,
                    discarded_normals=measurements,learned_weights_reused=False,
                    runtime_transfer='new execution; not cross-device/T4 checkpoint equivalence')
        except BaseException as exc:
            failure=exc;result.update(status='failed_preserve_evidence',remote_acceptance=False,error=repr(exc))
        finally:
            if extractor is not None:
                try:extractor.close()
                except BaseException as exc:failure=failure or exc;result.update(status='failed_preserve_evidence',remote_acceptance=False,error=repr(exc))
            result['elapsed_seconds']=time.monotonic()-started
            try:result['resources']=resource_usage(c['device'])
            except BaseException as exc:
                result['resource_error']=repr(exc);failure=failure or exc
                result.update(status='failed_preserve_evidence',remote_acceptance=False)
            atomic_json(result,work/'preflight.json')
            atomic_json(dict(status=result['status'],reportable=False),work/'handoff.json')
    receipt=export_run(work,work.parent/(work.name+'.zip'))
    if failure is not None:raise RuntimeError('Preflight failed; exported '+receipt['path']) from failure
    return dict(**result,evidence=receipt)
