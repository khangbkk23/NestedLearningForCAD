"""Canonical GPU bridge worker. Failure never publishes a candidate or shadow continuation."""
import argparse,io,json,time
from pathlib import Path
import torch
from training.runtime import atomic_json,resource_usage
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from training.support_throughput_v5_1.contract import Evidence,verify_parent,contract as profile_contract,digest
from training.support_throughput_v5_1.replay import assert_equal,state_digest
from training.support_benchmark_v5_1.contract import session_for,source_digest,study_digest
from training.support_benchmark_v5_1.data import NormalFeatures
from models.hope_cad_v5_1.contract import design_digest
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from training.support_nn_write_v5_1.candidate import activation
from .contract import contract,verify_source,make_identity
from .data import ArrivalProvider
from .bridge import shadow,Incomplete
from .lineage import publish,Execution

def load_owner(factory,state,initialization,optimized):
    with activation(optimized):
        owner=factory()
        # This API hashes CURRENT parameters/RNG; check it before restoring learned state.
        assert_equal(owner.initialization_hashes(),initialization,'fresh_initialization')
        owner.load_state_dict(state)
        assert_equal(owner.state_dict(),state,'active_strict_roundtrip')
        if state_digest(owner.state_dict())!=state_digest(state):raise ValueError('Loaded scientific bits differ')
        return owner

def run(c):
    configure_runtime();cfg=contract();pc=profile_contract();source=verify_source();runtime=runtime_identity();work=Path(c['work']);extractor=provider=None
    if source!=c['source'] or runtime!=c['runtime'] or not torch.cuda.is_available():raise ValueError('Worker source/runtime differs')
    if source_digest()!=pc['frozen_reference']['source_digest'] or study_digest()!=pc['study_digest'] or design_digest()!=pc['method_digest']:raise ValueError('Scientific contract changed')
    try:
        with Evidence(c['parent']) as e:
            info=verify_parent(e,pc)
            if runtime!=info['suite']['identity']['runtime']:raise ValueError('T4 parent runtime mismatch')
            parent=torch.load(io.BytesIO(e.read(pc['parent']['active_entry'])),map_location='cpu',weights_only=True)
            keys={'schema','job_digest','cursor','session','trace_digest','peak_persistent_tensor_bytes','elapsed_seconds'}
            if set(parent)!=keys or parent['schema']!='support_benchmark_active_v1' or parent['cursor']!=376 or parent['job_digest']!=info['job']['digest']:raise ValueError('Wrong active376 checkpoint')
            if e.json('runs/NESTED_seed0/active.json')!={'cursor':376,'sha256':pc['parent']['active_sha256']}:raise ValueError('Parent active receipt differs')
            chain=digest('empty event stream')
            for i in range(376):
                event=e.json(f'runs/NESTED_seed0/events/{i:05d}.json')
                if event['cursor']!=i or event['identity']!=info['rows'][i]['relative_path']:raise ValueError('Parent event prefix changed')
                chain=digest([chain,event])
            if chain!=parent['trace_digest']:raise ValueError('Parent trace digest differs')
            if any(r['task_id']!=info['rows'][376]['task_id'] for r in info['rows'][376:384]):raise ValueError('Unregistered bridge crossing task boundary')
            owners=[]
            for optimized in [False,True]:
                owners.append(load_owner(lambda:session_for('NESTED_seed0','cuda'),parent['session'],e.json('runs/NESTED_seed0/initialization.json')['identity'],optimized))
            extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),'cuda')
            raw=NormalFeatures(c['dataset'],extractor,work/'feature_receipts','cuda')
            provider=ArrivalProvider(info['rows'][376:384],raw,archive=c['dataset_archive'],assets=c['dataset'])
            atomic_json(dict(schema='nn_write_bridge_dispatch_v1',max_unique_arrivals=8,max_owner_transitions=16,source=source,deadline=c['deadline']),work/'dispatch.json')
            result=shadow(*owners,provider,info['rows'],work/'shadow',start=376,end=384,chain=chain,deadline=c['deadline'])
            if time.monotonic()>=c['deadline']:raise Incomplete('No publication after deadline')
            # Validate the export state before any accepted publication is created.
            with activation(True):
                check=session_for('NESTED_seed0','cuda');check.load_state_dict(result['state']);assert_equal(check.state_dict(),result['state'],'published384_reload')
            identities=make_identity(info,source,runtime,cfg)
            policy=dict(cfg,parent_archive_sha256=pc['parent']['archive_sha256'],parent_active_sha256=pc['parent']['active_sha256'])
            root=Path(__file__).resolve().parents[2]
            receipt=publish(work,e,info,identities,result,source,policy,parent,
                source_map=json.loads((root/'bridge_manifest.json').read_text()),base_source_map=json.loads((root/'bundle_manifest.json').read_text()),pins=pc['parent'])
            execution=Execution(work/'benchmark',source,runtime,cfg,pc['parent'])
            atomic_json(dict(status='bridge_complete_pending_review',new_execution_digest=receipt['execution_digest'],
                bridge_state_digest=state_digest(result['state']),reload384_exact=True,gpu=resource_usage('cuda'),full_benchmark_started=False),work/'bridge_result.json')
            return 'bridge_complete_pending_review'
    finally:
        if provider:provider.close()
        if extractor:extractor.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);c=json.loads(Path(p.parse_args().config).read_text());work=Path(c['work'])
    try:status=run(c)
    except Incomplete as exc:
        status='bridge_incomplete_preserve_parent';atomic_json(dict(status=status,error=repr(exc)),work/'failure.json')
    except BaseException as exc:
        atomic_json(dict(status='bridge_failed_preserve_parent',error=repr(exc)),work/'failure.json');raise
    atomic_json(dict(status=status),work/'worker_status.json');print('[BRIDGE STATUS]',status,flush=True)
if __name__=='__main__':main()
