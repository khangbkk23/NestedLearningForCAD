"""Full-train/H32 session revision. Reuses bank/credit transactions and score path."""
from copy import deepcopy
import random
import torch
from hope_cad.models.session import SupportSession
from hope_cad.models.consolidation import HistoricalUtilityLearner,NEURAL_VARIANTS
from hope_cad.models.validation import keys,integer,scalar,tree_finite
from hope_cad.models.primitives.state import tensor_digest
from hope_cad.contract import canonical,digest
from hope_cad.research import VARIANTS,method_identity


class ScalarUtilityControl:
    """No neural model, projected contexts, optimizer, or historical records."""
    def __init__(self,variant,seed):
        self.variant=variant;self.seed=seed;self.labels=self.fast_updates=self.slow_updates=0
        self.bias=0.;self.rng=random.Random(seed+95537);self.draws=0;self.last_draw=None
        self.records=[];self.block_initial=self.saved_meta_episode=self.optimizer=None

    def predict(self,tokens,stats,c,gamma):return gamma+self.bias,1.

    def observe(self,record):
        scalar(record['target'],'scalar target');scalar(record['gamma'],'scalar gamma')
        if record['target']!=(record['Jminus']-record['Jplus'])/record['c']:raise ValueError('Scalar target arithmetic')
        p=record['gamma']+self.bias
        if self.variant=='BIAS_VETO':self.bias-=.05*(p-record['target']);self.fast_updates+=1
        self.labels+=1
        return dict(labels=self.labels,fast_updates=self.fast_updates,slow_updates=0,bias=self.bias,
            prediction_before=p,fast_loss=.5*(p-record['target'])**2,completed_label_blocks=self.labels//8)

    def choose(self,warmup,pred):
        if self.variant=='BIAS_VETO':return warmup or pred>=0
        if warmup:self.last_draw=None;return True
        self.last_draw=self.rng.random();self.draws+=1
        return self.last_draw<66/211

    def state_dict(self):
        return dict(schema='scalar_utility_control_v1',variant=self.variant,seed=self.seed,labels=self.labels,
            fast_updates=self.fast_updates,slow_updates=0,bias=self.bias,rng=self.rng.getstate(),draws=self.draws,last_draw=self.last_draw)

    def load_state_dict(self,state):
        keys(state,self.state_dict(),'scalar control');tree_finite(state)
        if any(state[k]!=self.state_dict()[k] for k in ('schema','variant','seed')):raise ValueError('Scalar identity changed')
        for k in ('labels','fast_updates','slow_updates','draws'):integer(state[k],k)
        if state['fast_updates']!=(state['labels'] if self.variant=='BIAS_VETO' else 0) or state['slow_updates']!=0:raise ValueError('Fabricated scalar optimizer')
        if state['draws']>state['labels']+1:raise ValueError('Random decisions exceed completed/pending labels')
        scalar(state['bias'],'bias')
        if self.variant=='PILOT_RATE_RANDOM_VETO' and state['bias']!=0.:raise ValueError('Random control cannot train bias')
        expected=0 if self.variant=='BIAS_VETO' else None
        if expected==0 and state['draws']!=0:raise ValueError('Bias control drew random admission')
        rng=random.Random(self.seed+95537);last=None
        for _ in range(state['draws']):last=rng.random()
        if rng.getstate()!=state['rng'] or (state['last_draw'] is not None and state['last_draw']!=last):raise ValueError('Scalar RNG draw provenance')
        self.labels,self.fast_updates,self.bias,self.draws,self.last_draw=(state[k] for k in ('labels','fast_updates','bias','draws','last_draw'));self.rng=rng


class ResearchSession(SupportSession):
    state_schema='hope_hq_session_v1'

    def __init__(self,variant,*,device='cpu',seed=0,bank_config=None,model_config=None):
        if variant not in VARIANTS:raise ValueError('Unknown research variant')
        self.variant=variant;self.method=method_identity(variant)
        scalar_control=variant in ('BIAS_VETO','PILOT_RATE_RANDOM_VETO')
        mode='CADIC' if variant=='CADIC' or scalar_control else ('FIRST_ORDER' if variant.endswith('_FO') else ('FAST_ONLY' if variant=='FAST_ONLY' else 'NESTED'))
        super().__init__(mode,device=device,seed=seed,bank_config=bank_config,model_config=model_config)
        if variant in NEURAL_VARIANTS:
            self.learner=HistoricalUtilityLearner(self.learner.model,variant,seed,digest(self.method))
        if scalar_control:
            self.mode=variant;self.learned=True;self.learner=ScalarUtilityControl(variant,seed)

    def pending_token_shape(self):
        return (1,0,self.model_config.get('context_dim',32)) if isinstance(self.learner,ScalarUtilityControl) else super().pending_token_shape()

    def context_tokens(self,features):
        if isinstance(self.learner,ScalarUtilityControl):return torch.empty(self.pending_token_shape(),device=self.device)
        return super().context_tokens(features)

    def learner_hash(self):
        if isinstance(self.learner,ScalarUtilityControl):return digest(self.learner.state_dict())
        return super().learner_hash()

    def initialization_hashes(self):
        if isinstance(self.learner,ScalarUtilityControl):return dict(theta=None,phi=None,projection=None,anchor_rng=digest(self.anchors.rng.getstate()))
        return super().initialization_hashes()

    def admission(self,prediction,warmup):
        if isinstance(self.learner,ScalarUtilityControl):return self.learner.choose(warmup,prediction)
        return super().admission(prediction,warmup)

    def valid_admission(self,pending,prediction):
        if self.variant=='PILOT_RATE_RANDOM_VETO':
            return pending['accepted']==(True if pending['warmup'] else self.learner.last_draw is not None and self.learner.last_draw<66/211)
        return super().valid_admission(pending,prediction)

    def replay_records(self):
        return self.learner.validation_records() if isinstance(self.learner,HistoricalUtilityLearner) else []

    def identity(self):return canonical(dict(base=super().identity(),method=self.method))

    def load_state_dict(self,payload):
        other=ResearchSession(self.variant,device=self.device,seed=self.seed,bank_config=self.bank_config,model_config=self.model_config)
        other._load_validated(payload);self.__dict__.update(other.__dict__)

    def memory_profile(self):
        if isinstance(self.learner,ScalarUtilityControl):
            # No dummy neural allocation: parent counter sees bank/anchors/pending only.
            owner=self.learner;self.learner=None
            try:return super().memory_profile()
            finally:self.learner=owner
        value=super().memory_profile()
        if self.learner:
            # History payloads are disjoint from current replay; selected queries own IDs only.
            storages={}
            for item in self.learner.history.buffer:
                for v in item['record'].values():
                    if torch.is_tensor(v):storages[(str(v.device),v.untyped_storage().data_ptr())]=v.untyped_storage().nbytes()
            if self.learner.debug_queries:
                for r in self.learner.debug_queries['queries']:
                    for v in r.values():
                        if torch.is_tensor(v):storages[(str(v.device),v.untyped_storage().data_ptr())]=v.untyped_storage().nbytes()
            extra=sum(storages.values());value['history_and_debug_query_tensor_bytes']=extra
            value['persistent_tensor_bytes']+=extra;value['tensor_plus_pair_cache_bytes']+=extra
        return value
