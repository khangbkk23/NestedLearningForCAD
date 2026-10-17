"""Persistent fast CMS and eight-label exact nested slow learning of utility."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from torch import nn
from torch.func import functional_call
from models.hope_cad_v2.cms import CMSLevel
from models.hope_cad_v2.visual_block import HOPECADVisualBlock,VisualBlockConfig
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.state import finite_tensors
from .contract import CONTRACT,canonical,design_digest

class UtilityModel(nn.Module):
    def __init__(self,*,dim=32,context_dim=32,stats_dim=16,height=28,width=28,memory_dim=32,head_dim=8,self_modifying=True):
        super().__init__();self.visual=HOPECADVisualBlock(VisualBlockConfig(context_dim,dim,memory_dim,head_dim,height,width),row_retention=.9)
        self.geometry=nn.Linear(stats_dim,dim);self.fast=CMSLevel(dim,2*dim);self.slow=CMSLevel(dim,2*dim);self.head=nn.Linear(dim,1)
        self.self_modifying=self_modifying;nn.init.xavier_uniform_(self.geometry.weight);nn.init.zeros_(self.geometry.bias)
        nn.init.xavier_uniform_(self.head.weight);self.head.weight.data.mul_(.01);nn.init.constant_(self.head.bias,.05)
        self.fast.requires_grad_(False)
        if not self_modifying:
            for name in ('initial_key','initial_value','initial_eta','initial_alpha'):getattr(self.visual.memory,name).requires_grad_(False)
    def encode(self,tokens,stats):
        if tokens.shape[0]!=1 or stats.shape!=(self.geometry.in_features,):raise ValueError('one causal context with declared statistics')
        h=self.visual(tokens) if self.self_modifying else HOPEFeaturePredictor.encode(SimpleNamespace(visual=self.visual,dynamics=SimpleNamespace(visual_updates=False)),tokens)
        return h.mean(dim=1,keepdim=True)+self.geometry(stats).reshape(1,1,-1)
    def from_encoded(self,encoded,fast):return torch.tanh(self.head(self.slow(functional_call(self.fast,fast,(encoded,),strict=True)))).reshape(())
    def predict(self,tokens,stats,fast):return self.from_encoded(self.encode(tokens,stats),fast)
    def initial_fast(self):return {k:v.detach().clone().requires_grad_(True) for k,v in self.fast.named_parameters()}
    def slow_parameters(self):return [(k,v) for k,v in self.named_parameters() if v.requires_grad]

def fast_step(model,phi,record,lr,*,exact):
    pred=model.predict(record['tokens'],record['stats'],phi);loss=.5*(pred-record['target'])**2
    grads=torch.autograd.grad(loss,tuple(phi.values()),create_graph=exact)
    finite_tensors((loss,*grads),'utility fast gradients')
    denom=torch.sqrt(1+sum(g.square().sum() for g in grads))
    # FIRST_ORDER keeps precisely the same forward values, removes only inner gradients' dependency.
    updates=[g/denom for g in grads]
    if not exact:updates=[g.detach() for g in updates]
    return {k:v-lr*g for (k,v),g in zip(phi.items(),updates)},loss

def meta_loss(model,initial,records,*,exact=True,fast_lr=.05,support=4):
    if len(records)!=2*support:raise ValueError('complete chronological support/query block required')
    phi={k:v.detach().clone().requires_grad_(True) for k,v in initial.items()}
    for record in records[:support]:phi,_=fast_step(model,phi,record,fast_lr,exact=exact)
    return torch.stack([.5*(model.predict(r['tokens'],r['stats'],phi)-r['target'])**2 for r in records[support:]]).mean()

class NestedUtilityLearner:
    def __init__(self,model,mode='NESTED'):
        if mode not in ('NESTED','FIRST_ORDER','FAST_ONLY','NO_FAST_ADAPTATION'):raise ValueError('invalid learning policy')
        self.model,self.mode=model,mode;self.phi=model.initial_fast();self.labels=0;self.slow_updates=0;self.records=[];self.block_initial=None;self.last_slow=None
        self.optimizer=torch.optim.Adam([p for _,p in model.slow_parameters()],lr=CONTRACT['learning']['slow_lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    @torch.no_grad()
    def predict(self,tokens,stats):return float(self.model.predict(tokens,stats,self.phi))
    def observe(self,tokens,stats,target,decision_id):
        if not -1.000001<=target<=1.000001:raise ValueError('utility label outside declared bound')
        record=dict(tokens=tokens.detach(),stats=stats.detach(),target=float(target),decision_id=decision_id)
        if not self.records:self.block_initial={k:v.detach().clone() for k,v in self.phi.items()}
        if self.mode!='NO_FAST_ADAPTATION':
            new,fast_loss=fast_step(self.model,self.phi,record,CONTRACT['learning']['fast_lr'],exact=False)
            self.phi={k:v.detach().requires_grad_(True) for k,v in new.items()}
        else:fast_loss=.5*(self.model.predict(tokens,stats,self.phi)-target)**2
        self.labels+=1;self.records.append(record);details=dict(labels=self.labels,fast_loss=float(fast_loss.detach()),slow_updates=self.slow_updates)
        if len(self.records)==8:
            if self.mode!='FAST_ONLY':
                self.optimizer.zero_grad(set_to_none=True)
                if self.mode=='NO_FAST_ADAPTATION':loss=torch.stack([.5*(self.model.predict(r['tokens'],r['stats'],self.block_initial)-r['target'])**2 for r in self.records[4:]]).mean()
                else:loss=meta_loss(self.model,self.block_initial,self.records,exact=self.mode=='NESTED')
                loss.backward();params=self.model.slow_parameters()
                if any(p.grad is None for _,p in params):raise RuntimeError('disconnected slow utility group')
                finite_tensors((p.grad for _,p in params),'utility slow gradients')
                norms={name:float(p.grad.detach().double().norm()) for name,p in params}
                norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)
                self.optimizer.step();finite_tensors((p for _,p in params),'slow utility proposal')
                finite_tensors((v for state in self.optimizer.state.values() for v in state.values() if torch.is_tensor(v)),'slow Adam')
                self.slow_updates+=1;self.last_slow=dict(update=self.slow_updates,loss=float(loss.detach()),gradient_norm=float(norm),group_norms=norms,support_ids=[r['decision_id'] for r in self.records[:4]],query_ids=[r['decision_id'] for r in self.records[4:]])
                details['slow']=self.last_slow
            self.records=[];self.block_initial=None
        details['slow_updates']=self.slow_updates;return details
    def state_dict(self):
        cpu=lambda v:v.detach().cpu().clone()
        return dict(schema='hope_support_learner_v5',design_digest=design_digest(),mode=self.mode,model=deepcopy(self.model.state_dict()),phi={k:cpu(v) for k,v in self.phi.items()},optimizer=deepcopy(self.optimizer.state_dict()),labels=self.labels,slow_updates=self.slow_updates,records=[dict(r,tokens=cpu(r['tokens']),stats=cpu(r['stats'])) for r in self.records],block_initial={k:cpu(v) for k,v in self.block_initial.items()} if self.block_initial else None,last_slow=deepcopy(self.last_slow))
    def load_state_dict(self,state):
        if state['schema']!='hope_support_learner_v5' or state['mode']!=self.mode or state['design_digest']!=design_digest():raise ValueError('learner identity changed')
        # Validate every floating payload before mutating the current owner.
        tensors=[]
        def collect(value):
            if torch.is_tensor(value):tensors.append(value)
            elif isinstance(value,dict):
                for v in value.values():collect(v)
            elif isinstance(value,(list,tuple)):
                for v in value:collect(v)
        collect(state);finite_tensors(tensors,'learner checkpoint')
        if type(state['labels']) is not int or state['labels']<0:raise ValueError('invalid completed-label clock')
        for r in state['records']:
            if not isinstance(r['target'],(int,float)) or not -1<=r['target']<=1:raise ValueError('invalid replay target')
        device=next(self.model.parameters()).device;self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(deepcopy(state['optimizer']))
        self.phi={k:v.to(device).detach().clone().requires_grad_(True) for k,v in state['phi'].items()};self.labels=state['labels'];self.slow_updates=state['slow_updates']
        self.records=[dict(r,tokens=r['tokens'].to(device).detach().clone(),stats=r['stats'].to(device).detach().clone()) for r in state['records']]
        self.block_initial={k:v.to(device).detach().clone() for k,v in state['block_initial'].items()} if state['block_initial'] else None;self.last_slow=deepcopy(state['last_slow'])
        if len(self.records)!=self.labels%8 or (bool(self.records)!=(self.block_initial is not None)):raise ValueError('corrupt utility window clock')
        if self.slow_updates!=(0 if self.mode=='FAST_ONLY' else self.labels//8):raise ValueError('corrupt slow update count')
        template=dict(self.model.fast.named_parameters())
        if self.phi.keys()!=template.keys() or any(self.phi[k].shape!=v.shape for k,v in template.items()):raise ValueError('corrupt fast weights')
        finite_tensors(self.phi.values(),'restored fast state')
