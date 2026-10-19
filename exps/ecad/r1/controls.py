"""Existing controls under common selection distance/global NN readout.

No canonical baseline is edited. CADIC/S4/R4 retain the AD fast implementation's
live quadratic caches, slot ties and prefix fill. P0 is a device/numeric port.
"""
import io
import sys
import copy
import dataclasses
import torch
from exps.ad01.hope_cad_ad01_fast_coreset import FastCADICPatchCoresetV1
from exps.ad02.hope_cad_ad02_partition import build_partition, per_group_quota
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig
from .p0_matched import MatchedP0
from .contcore import CausalContCore
from .distance import Euclidean, global_nn

ARMS = ("contcore", "reservoir", "cadic", "p0", "s4", "r4")


def charged_bytes(name,k,dim=768):
    """Fully occupied, warm-cache charge for this exact implementation."""
    if name=='contcore':return (4*dim+8)*k+16384
    if name=='reservoir':return (4*dim+8)*k+5056+16384
    if name=='p0':return (4*dim+24)*k+128*(4*dim+8)+5056+16384
    if name=='cadic':return 4*dim*k+4*k*k+16384
    if name in ('s4','r4'):
        return 4*dim*k+4*sum(q*q for q in per_group_quota(k,16))+784*8+65536
    raise ValueError('unknown arm')


def byte_capacities(k=2500):
    ceiling=charged_bytes('contcore',k)
    return {name:max(n for n in range(16,k+1) if charged_bytes(name,n)<=ceiling) for name in ARMS}


class MatchedCADIC(FastCADICPatchCoresetV1):
    def __init__(self, capacity, dim, device, chunk):
        super().__init__(CADICPatchCoresetConfig(budget=capacity,dim=dim),device=device)
        self.kernel = Euclidean(chunk)

    def _distance_matrix(self):
        if self._pair_matrix is None or self._pair_dirty or len(self._pair_matrix)!=self.count:
            self._pair_matrix = self.kernel.matrix(self.features,self.features)
            self._pair_dirty = False
        return self._pair_matrix

    def _refresh_pair_row(self,index):
        row = self.kernel.matrix(self.features[index:index+1],self.features)[0]
        self._pair_matrix[index,:] = row
        self._pair_matrix[:,index] = row

    def _nearest(self,query,bank):
        # Preserve winning row/slot semantics; no historical image weighting used.
        matrix = self.kernel.matrix(query,bank)
        return matrix.min(1)


class MemoryArm:
    def __init__(self, name, capacity=2500, dim=768, device="cpu", chunk=256, seed=7202):
        if name not in ARMS or capacity<16:
            raise ValueError("unknown arm or capacity below 16")
        self.name, self.capacity, self.dim = name, capacity, dim
        self.device = torch.device(device)
        self.chunk, self.event = chunk, 0
        self.assignment = torch.empty(0,dtype=torch.long,device=device)
        if name == "contcore":
            self.model = CausalContCore(capacity=capacity,dim=dim,device=device,chunk=chunk)
        elif name == "p0":
            self.model = MatchedP0(capacity,dim,device,chunk)
        elif name == "reservoir":
            self.rng = torch.Generator().manual_seed(seed)
            self.rows = torch.empty((capacity,dim),dtype=torch.float32,device=device)
            self.serials = torch.full((capacity,),-1,dtype=torch.long,device=device)
            self.n = 0
        else:
            groups = 16 if name in ("s4","r4") else 1
            if groups==16:
                assignment,_,_ = build_partition("spatial" if name=="s4" else "random",grid=4,seed=17011)
                self.assignment = assignment.to(device)
            self.banks = [MatchedCADIC(k,dim,device,chunk) for k in per_group_quota(capacity,groups)]

    @property
    def bank(self):
        if self.name=="contcore": return self.model.bank
        if self.name=="p0": return self.model.bank[:self.model.n]
        if self.name=="reservoir": return self.rows[:self.n]
        return torch.cat([b.features for b in self.banks])

    @torch.no_grad()
    def update(self,features,*,event,work=None):
        if event!=self.event or features.shape!=(784,self.dim) or features.dtype!=torch.float32:
            raise ValueError("full-grid, causal image arrival required")
        if not bool(torch.isfinite(features).all()) or features.device!=self.device:
            raise ValueError("invalid feature values/device")
        ids = torch.arange(event*784,(event+1)*784,device=features.device)
        if self.name=="contcore":
            result = self.model.update(features,event=event,work=work)
        elif self.name=="p0":
            result = self.model.update_matched(features,ids,work)
        elif self.name=="reservoir":
            replaced=0
            # Algorithm R over ALL patches; independent CPU RNG. Count total seen
            # derives from the contiguous event counter, no category identifiers.
            take=min(self.capacity-self.n,len(features))
            self.rows[self.n:self.n+take]=features[:take]
            self.serials[self.n:self.n+take]=ids[:take];self.n+=take
            # RNG opportunities are exactly sequential Algorithm R. Within an
            # image only the last accepted write per slot survives; batch those
            # distinct writes without changing state or the RNG stream.
            last={}
            for j in range(take,len(features)):
                slot=int(torch.randint(event*784+j+1,(1,),generator=self.rng))
                if slot<self.capacity:last[slot]=j;replaced+=1
            if last:
                slots=torch.tensor(list(last),device=features.device)
                incoming=torch.tensor(list(last.values()),device=features.device)
                self.rows[slots]=features[incoming];self.serials[slots]=ids[incoming]
            result={"replaced":replaced,"occupied":self.n}
        else:
            result={"accepted":0,"replaced":0,"rejected":0}
            for g,b in enumerate(self.banks):
                b.kernel=Euclidean(self.chunk,work=work)
                try:
                    part=features if len(self.banks)==1 else features[self.assignment==g]
                    stats=b.update(part)
                    for key in result: result[key]+=stats[key]
                finally:
                    b.kernel=Euclidean(self.chunk)
        self.event+=1
        return result

    def score(self,features):
        return global_nn(features,self.bank,Euclidean(self.chunk))

    def snapshot(self):
        """CPU copy includes caches/RNG; restoring preserves online cache lifecycle."""
        def cp(x):
            if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
            if isinstance(x,torch.Generator):return {"generator_state":x.get_state().clone()}
            if isinstance(x,(list,tuple)):return [cp(v) for v in x]
            if isinstance(x,dict):return {k:cp(v) for k,v in x.items()}
            if hasattr(x,"__dict__"):return {"class":type(x).__name__,"attrs":cp(vars(x))}
            return copy.deepcopy(x)
        return cp(self)

    def ledger(self):
        """Count actual retained storage once, plus observed Python shallow headers.

        No Python allocator/RSS precision is implied by the header subtotal.
        Checkpoints and process RSS are separate measurements.
        """
        seen, storages, tensors = set(), set(), []
        def visit(x,path):
            if id(x) in seen:return 0
            seen.add(id(x)); overhead=sys.getsizeof(x)
            if isinstance(x,torch.Tensor):
                st=x.untyped_storage(); key=(str(x.device),st.data_ptr())
                if key not in storages:
                    storages.add(key);tensors.append({"path":path,"bytes":st.nbytes(),"shape":list(x.shape),"device":str(x.device)})
            elif isinstance(x,torch.Generator):
                tensors.append({"path":path+".rng_state","bytes":x.get_state().numel(),"device":"cpu"})
            elif isinstance(x,dict):
                overhead+=sum(visit(k,path+".key")+visit(v,path+"."+str(k)) for k,v in x.items())
            elif isinstance(x,(list,tuple)):
                overhead+=sum(visit(v,path+f"[{i}]") for i,v in enumerate(x))
            elif hasattr(x,"__dict__"):
                overhead+=visit(vars(x),path)
            return overhead
        headers=visit(self,"arm")
        stream=io.BytesIO();torch.save(self.snapshot(),stream)
        infer=io.BytesIO();torch.save({"bank":self.bank.detach().cpu()},infer)
        payload=sum(t["bytes"] for t in tensors)
        envelope=max(65536 if self.name in ('s4','r4') else 16384,headers)
        return {"name":self.name,"capacity":self.capacity,"occupied":len(self.bank),
                "storages":tensors,"persistent_tensor_and_rng_bytes":payload,
                "python_headers_estimate":headers,"metadata_envelope":envelope,
                "charged_online_bytes":payload+envelope,
                "checkpoint_bytes":stream.tell(),"inference_export_bytes":infer.tell(),
                "inference_descriptor_bytes":self.bank.numel()*self.bank.element_size()}

    @classmethod
    def restore(cls,state,device="cpu"):
        a=state["attrs"]
        obj=cls(a["name"],a["capacity"],a["dim"],device,a["chunk"])
        def restore_into(target,source):
            if isinstance(source,dict) and "generator_state" in source:
                target.set_state(source["generator_state"]);return target
            if isinstance(source,torch.Tensor):return source.to(device).clone()
            if isinstance(source,dict) and "class" in source:
                if dataclasses.is_dataclass(target) and target.__dataclass_params__.frozen:
                    return type(target)(**source['attrs'])
                for k,v in source["attrs"].items():setattr(target,k,restore_into(getattr(target,k,None),v))
                return target
            if isinstance(source,list):return [restore_into(t,s) for t,s in zip(target,source)]
            if isinstance(source,dict):return {k:restore_into(target.get(k),v) for k,v in source.items()}
            if isinstance(source,torch.device):return torch.device(device)
            return copy.deepcopy(source)
        return restore_into(obj,state)
