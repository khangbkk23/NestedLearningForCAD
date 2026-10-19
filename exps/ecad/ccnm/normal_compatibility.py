"""Normal-only, bounded context compatibility on a fixed global allocator."""
import io
import math
import sys
import torch
from .mechanism import CCNM
from .precise_distance import PreciseEuclidean,isolated_radii

VERSION='ccnm_normal_context_boxes_isolated_support_v1'


def merge_bounds(lower,upper,targets,source_lower,source_upper):
    """Commutative exact min/max union; no floating-point sum or RNG."""
    if len(targets):
        index=targets[:,None].expand(-1,lower.shape[1])
        lower.scatter_reduce_(0,index,source_lower,reduce='amin',include_self=True)
        upper.scatter_reduce_(0,index,source_upper,reduce='amax',include_self=True)


class NormalCompatibility:
    def __init__(self,**kwargs):
        self.base=CCNM(**kwargs)
        self.lower=self.base.contexts.clone();self.upper=self.base.contexts.clone()

    @torch.no_grad()
    def update(self,features,*,event):
        b=self.base
        if event!=b.model.next_event:raise ValueError('noncausal or replayed image')
        context=b.context(features)
        kernel=PreciseEuclidean(b.model.chunk)
        old_ids,old_bank=b.model.serials,b.model.bank
        old_lower,old_upper=self.lower.clone(),self.upper.clone()
        radii=isolated_radii(old_bank,kernel)
        learned=0
        if len(old_bank):
            distance,nearest=kernel.nearest(features,old_bank)
            valid=distance<=radii[nearest]
            targets=torch.unique(nearest[valid],sorted=True)
            learned=len(targets)
            merge_bounds(old_lower,old_upper,targets,context.expand(len(targets),-1),context.expand(len(targets),-1))
        stats=b.update(features,event=event) # pinned allocator, never replaced
        ids=b.model.serials;historical=ids<event*b.model.patches
        lower=context.expand(len(ids),-1).clone();upper=lower.clone()
        if bool(historical.any()):
            old_row=torch.searchsorted(old_ids,ids[historical])
            lower[historical],upper[historical]=old_lower[old_row],old_upper[old_row]
        transported=0
        if len(old_ids):
            evicted=~torch.isin(old_ids,ids)
            rows=evicted.nonzero(as_tuple=True)[0]
            if len(rows):
                distance,target=kernel.nearest(old_bank[rows],b.model.bank)
                valid=distance<=radii[rows]
                transported=int(valid.sum())
                merge_bounds(lower,upper,target[valid],old_lower[rows[valid]],old_upper[rows[valid]])
        self.lower,self.upper=lower,upper
        if not bool(((lower<=b.contexts)&(b.contexts<=upper)).all()):
            raise ValueError('source context association lost')
        return {**stats,'normal_prototypes_observed':learned,'evicted_profiles_transported':transported}

    @torch.no_grad()
    def score(self,features,*,mode='boxes',weight=1.,chunk=None,query_context=None):
        b=self.base
        if mode not in ('raw','legacy','boxes','shuffled') or weight<0 or not math.isfinite(weight):
            raise ValueError('invalid readout')
        if not len(b.model.bank):raise ValueError('empty support')
        if (features.ndim!=2 or features.shape[1]!=b.model.dim or features.dtype!=torch.float32
                or features.device!=b.model.bank.device or not bool(torch.isfinite(features).all())):
            raise ValueError('invalid patch query')
        context=b.context(features) if query_context is None else query_context
        if context.shape!=(b.width,) or context.device!=features.device or not bool(torch.isfinite(context).all()):
            raise ValueError('invalid query context')
        if mode=='legacy':
            delta=b.contexts.double()-context.double()
        else:
            lower,upper=self.lower,self.upper
            if mode=='shuffled':
                p=torch.randperm(len(lower),generator=torch.Generator().manual_seed(52011)).to(lower.device)
                lower,upper=lower[p],upper[p]
            delta=(lower.double()-context.double()).clamp_min(0)+(context.double()-upper.double()).clamp_min(0)
        penalty=delta.square().sum(1)*(b.alpha.double()*weight).square()
        kernel=PreciseEuclidean(chunk or b.model.chunk)
        best=torch.full((len(features),),torch.inf,device=features.device)
        selected=torch.full((len(features),),-1,device=features.device,dtype=torch.int64)
        for i,j,d in kernel.tiles(features,b.model.bank):
            if mode!='raw' and weight!=0:
                d=(d.double().square()+penalty[j:j+d.shape[1]]).sqrt().float()
            value,ix=d.min(1);sl=slice(i,i+len(value));better=value<best[sl]
            selected[sl]=torch.where(better,j+ix,selected[sl]);best[sl]=torch.minimum(best[sl],value)
        return {'scores':best,'ids':b.model.serials[selected]}

    def state_dict(self):
        return {'schema':VERSION,'base':self.base.state_dict(),'lower':self.lower.clone(),'upper':self.upper.clone()}

    @classmethod
    def from_state_dict(cls,s,device='cpu'):
        if s['schema']!=VERSION:raise ValueError('unknown context compatibility schema')
        obj=cls.__new__(cls);obj.base=CCNM.from_state_dict(s['base'],device=device)
        lo,hi=s['lower'],s['upper']
        if (lo.dtype!=torch.float32 or hi.dtype!=torch.float32 or lo.shape!=obj.base.contexts.shape
                or hi.shape!=lo.shape or not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all())
                or not bool(((lo<=s['base']['contexts'])&(s['base']['contexts']<=hi)).all())):
            raise ValueError('invalid compatibility bounds')
        obj.lower,obj.upper=lo.to(device).clone(),hi.to(device).clone()
        return obj

    def ledger(self):
        base=self.base.ledger()
        extra={'lower':self.lower.untyped_storage().nbytes(),'upper':self.upper.untyped_storage().nbytes()}
        seen=set()
        def size(x):
            if id(x) in seen:return 0
            seen.add(id(x));n=sys.getsizeof(x)
            if isinstance(x,torch.Tensor):return n
            if isinstance(x,dict):n+=sum(size(k)+size(v) for k,v in x.items())
            elif hasattr(x,'__dict__'):n+=size(vars(x))
            return n
        headers=size(self);total=base['tensor_total_bytes']+sum(extra.values())
        f=io.BytesIO();torch.save(self.state_dict(),f)
        return {'base':base,'extra_tensor_bytes':extra,
            'tensor_total_bytes':total,'python_graph_bytes':headers,'online_total_bytes':total+headers,
            'additional_context_bytes_over_raw':total-len(self.base.model.bank)*(self.base.model.dim*4+8),
            'checkpoint_serialized_bytes':len(f.getvalue()),
            'note':'includes source anchors for legacy control, both box arrays, projection and alpha; workspace transient'}
