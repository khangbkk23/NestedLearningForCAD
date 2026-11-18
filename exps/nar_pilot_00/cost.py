"""Read-only end-to-end NAR latency and persistent-memory measurement."""
from __future__ import annotations
import hashlib
import time
import numpy as np
import torch
from PIL import Image

from exps.nar_pilot_00 import core as K
from exps.nar_pilot_00.detector import Detector
from exps.nar_pilot_00 import full as R
from dataset.benchmark_protocol_v1 import _ImageDataset


def quantiles(values):
    return {'p50_ms':float(np.percentile(values,50)*1000),
            'p95_ms':float(np.percentile(values,95)*1000),
            'mean_ms':float(np.mean(values)*1000),'n':len(values)}


@torch.no_grad()
def main():
    R.configure(); R.verify_training()
    K.require(not (R.OUT/'COST.json').exists(),'preserve measured cost')
    data=torch.load(R.OUT/'features/bottle_train.pt',weights_only=True,mmap=True)
    rows=data['rows'][:20]
    ds=_ImageDataset(R.DATA,rows,224,(.5,)*3,(.5,)*3,training=True)
    encoder=R.make_encoder()
    encoder_bytes=sum(t.numel()*t.element_size() for t in encoder.state_dict().values())
    base=torch.load(R.OUT/'BASE.pt',weights_only=True)
    base_bytes=sum(t.numel()*t.element_size() for t in base.values())
    results={}
    for arm in R.ARMS:
        state=torch.load(R.OUT/arm/'stage15.pt',weights_only=True)
        model,loras,_=R.model_new(base)
        detector=Detector(model,loras,state['archive'])
        pre,enc,det=[],[],[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(23):
            j=max(0,i-3)
            start=time.perf_counter(); x=ds[j]['images'][None]
            with Image.open(R.DATA/rows[j]['relative_path']) as image: hw=(image.height,image.width)
            preprocessing=time.perf_counter()-start
            torch.cuda.synchronize(); start=time.perf_counter()
            feature=encoder.extract_patch_features(x.cuda()).float()
            torch.cuda.synchronize(); enc_time=time.perf_counter()-start
            start=time.perf_counter(); detector.predict(feature,hw)
            torch.cuda.synchronize(); det_time=time.perf_counter()-start
            if i>=3:
                pre.append(preprocessing); enc.append(enc_time); det.append(det_time)
        adapter_bytes=sum(K.tensor_state_bytes(a) for a in state['archive'])
        policy_bytes=K.slow_bytes(state['lam'],state['phi'])
        results[arm]={'preprocess':quantiles(pre),'encoder':quantiles(enc),'detector':quantiles(det),
            'end_to_end':quantiles(np.array(pre)+enc+det),'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
            'shared_encoder_tensor_bytes':encoder_bytes,'shared_decoder_tensor_bytes':base_bytes,
            'archive_tensor_bytes':adapter_bytes,'policy_tensor_bytes':policy_bytes,
            'continual_tensor_bytes':base_bytes+adapter_bytes+policy_bytes,
            'complete_model_tensor_bytes':encoder_bytes+base_bytes+adapter_bytes+policy_bytes,
            'final_state_file_bytes':(R.OUT/arm/'stage15.pt').stat().st_size,
            'base_file_bytes':(R.OUT/'BASE.pt').stat().st_size,
            'encoder_file_bytes':R.CHECKPOINT.stat().st_size}
        del model,loras,detector
        torch.cuda.empty_cache()
    R.write_json(R.OUT/'COST.json',{'protocol_sha256':R.sha(R.ROOT/'agents/plans/nar_full_15_cost_protocol.txt'),
        'arms':results,'query_source':'first20 bottle/train/good; no task ID supplied to detector',
        'device':torch.cuda.get_device_name(0),'batch':1})
    R.log('COST',results)


if __name__=='__main__': main()
