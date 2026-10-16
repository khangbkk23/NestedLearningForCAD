"""Compact CADIC checkpoint transport; coreset math/scoring remain unchanged."""
from copy import deepcopy


class CompactCADIC:
    def __init__(self, adapter, backbone_sha256):
        self.adapter,self.backbone_sha256 = adapter,backbone_sha256
        self.train_transport_batch_size=int(adapter.config['runtime']['batch_size'])

    def __getattr__(self,name):
        return getattr(self.adapter,name)

    def state_dict(self):
        return dict(schema='hope_w7_cadic_compact_v1',backbone_sha256=self.backbone_sha256,
                    coreset=deepcopy(self.adapter.coreset.state_dict()),
                    extractor_versions=[(n,int(p._version)) for n,p in self.adapter.extractor.named_parameters()])

    def load_state_dict(self,payload):
        if payload.get('schema')!='hope_w7_cadic_compact_v1' or payload['backbone_sha256']!=self.backbone_sha256:
            raise ValueError('CADIC compact checkpoint backbone mismatch')
        self.adapter.coreset.load_state_dict(payload['coreset'])


def baseline_config(context,job):
    from training.benchmark_config_v2 import validate_configs
    # The frozen validator remains strict for the 10k primary baseline. This
    # registered secondary changes only ID/budget; no scoring/update fallback.
    validate_configs(context['protocol'],context['method'])
    method = deepcopy(context['method'])
    if job['method']!='CADIC' or job['budget'] not in (2500,10000):
        raise ValueError('unregistered matched CADIC budget')
    if job['budget']==2500:
        method['id']='cadic_paperfaithful_matched_2500_w7_v4'
        method['memory']['budget']=2500
    return method
