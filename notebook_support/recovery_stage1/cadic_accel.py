"""Exact tile-minimum cache for frozen CADIC; no new vectors or changed distance kernel.

A replacement changes one bank row. Only pair tiles containing that row can
change. Cached minima of every other tile remain exact, including FP32 ties.
Recomputed tiles call the original GEMM on the original slices/shapes.
"""
import sys
import torch
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1,euclidean_distance_mm

BACKEND='cadic_exact_pair_tile_cache_v2'

class CachedPairCoreset(CADICPatchCoresetV1):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._clear_pair_cache()
        self._cache_peak_bytes=0
        self._cache_counts=dict(tile_evaluations=0,closest_pair_calls=0,full_rebuilds=0)

    def _clear_pair_cache(self):
        self._cache_signature=None
        self._tiles=[];self._adjacent=[];self._dirty=set()
        self._tile_values=None;self._tile_pairs=None;self._triangle=None

    def _signature(self):
        return (id(self.features),self.features.data_ptr(),self.features._version,self.count)

    def _ensure_pair_cache(self):
        signature=self._signature()
        if signature==self._cache_signature:return
        chunk=self.config.pair_chunk_size;n=self.count
        blocks=(n+chunk-1)//chunk
        self._tiles=[];self._adjacent=[[] for _ in range(blocks)]
        for a in range(blocks):
            for b in range(a,blocks):
                index=len(self._tiles);self._tiles.append((a*chunk,b*chunk))
                self._adjacent[a].append(index)
                if a!=b:self._adjacent[b].append(index)
        self._tile_values=torch.full((len(self._tiles),),float('inf'),device=self.device)
        self._tile_pairs=torch.full((len(self._tiles),),n*n,dtype=torch.long,device=self.device)
        arange=torch.arange(min(chunk,n),device=self.device)
        self._triangle=arange[None,:]<=arange[:,None]
        self._dirty=set(range(len(self._tiles)));self._cache_signature=signature
        self._cache_counts['full_rebuilds']+=1
        self.cache_profile()

    def _changed(self,row):
        signature=self._signature()
        old=self._cache_signature
        if old is None or old[:2]!=signature[:2] or old[3]!=signature[3] or old[2]+1!=signature[2]:
            self._clear_pair_cache()
            return
        self._dirty.update(self._adjacent[row//self.config.pair_chunk_size])
        self._cache_signature=signature

    @torch.no_grad()
    def _closest_pair(self):
        if self.count<2:return super()._closest_pair()
        self._ensure_pair_cache();chunk=self.config.pair_chunk_size;n=self.count
        self._cache_counts['closest_pair_calls']+=1
        for index in sorted(self._dirty):
            row_start,col_start=self._tiles[index]
            rows=self.features[row_start:row_start+chunk]
            cols=self.features[col_start:col_start+chunk]
            distances=euclidean_distance_mm(rows,cols)
            self._record_distance_block('pair',distances)
            if row_start==col_start:distances.masked_fill_(self._triangle[:len(rows),:len(cols)],float('inf'))
            value,flat=torch.min(distances.reshape(-1),dim=0)
            row=row_start+torch.div(flat,len(cols),rounding_mode='floor')
            col=col_start+flat.remainder(len(cols))
            self._tile_values[index]=value;self._tile_pairs[index]=row*n+col
        self._cache_counts['tile_evaluations']+=len(self._dirty);self._dirty.clear()
        # NaN tile minima are ignored by the reference's strict comparisons.
        values=torch.where(torch.isnan(self._tile_values),float('inf'),self._tile_values)
        best=values.min()
        keys=torch.where(self._tile_values==best,self._tile_pairs,n*n)
        row=int(torch.div(keys.min(),n,rounding_mode='floor').item())
        return best,row

    @torch.no_grad()
    def update(self,patch_features):
        # Identical selection/update/counter operations to the frozen reference;
        # the sole additional operation is cache invalidation after a replacement.
        x=patch_features.detach().to(self.device,dtype=torch.float32)
        if x.ndim==3:x=x.reshape(-1,x.shape[-1])
        if x.ndim!=2 or x.shape[1]!=self.config.dim:
            raise ValueError(f'expected [N,{self.config.dim}] or [B,N,{self.config.dim}], got {tuple(patch_features.shape)}')
        if x.shape[0]==0:return dict(incoming=0,accepted=0,replaced=0,rejected=0)
        self.update_batches+=1;self.seen_features+=int(x.shape[0])
        accepted=replaced=rejected=0
        if not self.is_full:
            take=min(self.config.budget-self.count,x.shape[0])
            if take:self.features=torch.cat((self.features,x[:take]),dim=0);accepted+=take
            x=x[take:]
        while x.numel() and self.is_full:
            nearest,_=self._nearest(x,self.features)
            farthest_value,farthest_row=torch.max(nearest,dim=0)
            closest_value,closest_row=self._closest_pair()
            if not bool(farthest_value>closest_value):rejected+=int(x.shape[0]);break
            self.features[closest_row]=x[farthest_row]
            self._changed(closest_row)
            replaced+=1
            keep=torch.ones(x.shape[0],dtype=torch.bool,device=x.device)
            keep[farthest_row]=False;x=x[keep]
        self.accepted_features+=accepted;self.replaced_features+=replaced;self.rejected_features+=rejected
        return dict(incoming=int(patch_features.reshape(-1,patch_features.shape[-1]).shape[0]),
            accepted=accepted,replaced=replaced,rejected=rejected)

    def load_state_dict(self,state):
        super().load_state_dict(state);self._clear_pair_cache()

    def cache_profile(self):
        tensors=[self._tile_values,self._tile_pairs,self._triangle]
        tensor_bytes=sum(t.numel()*t.element_size() for t in tensors if t is not None)
        visited=set()
        def size(value):
            if id(value) in visited:return 0
            visited.add(id(value));total=sys.getsizeof(value)
            if isinstance(value,(list,tuple,set)):total+=sum(size(x) for x in value)
            return total
        python_bytes=sum(size(v) for v in (self._tiles,self._adjacent,self._dirty))
        self._cache_peak_bytes=max(self._cache_peak_bytes,tensor_bytes+python_bytes)
        return dict(backend=BACKEND,**self._cache_counts,tile_count=len(self._tiles),dirty_tiles=len(self._dirty),
            runtime_cache_tensor_bytes=tensor_bytes,runtime_cache_python_bytes=python_bytes,
            runtime_cache_bytes=tensor_bytes+python_bytes,runtime_cache_peak_bytes=self._cache_peak_bytes,cache_in_checkpoint=False,
            unchanged_reference_kernel=True,unchanged_pair_tile_shape=True,
            accounting='tensor payload + unique Python layout objects; allocator and general profiler overhead excluded')

    def distance_profile(self):
        return dict(super().distance_profile(),execution_optimization=self.cache_profile())


class CachedCADICAdapter:
    def __init__(self,adapter):
        from models.benchmark_registry_v2 import RegisteredAdapter
        self.adapter=adapter
        self._closed=False
        # __getattr__ forwards reads only. Assign on the concrete owner, NEVER
        # on RegisteredAdapter, whose fit/score methods are bound to the owner.
        self._registered=isinstance(adapter,RegisteredAdapter)
        self._owner=adapter.adapter if self._registered else adapter
        if self._registered and 'coreset' in vars(adapter):raise RuntimeError('invalid registry coreset shadow binding')
        original=self._owner.coreset
        if type(original) is not CADICPatchCoresetV1:raise ValueError('requires frozen paperfaithful coreset')
        self._cached_core=CachedPairCoreset(original.config,device=original.device)
        self._cached_core.load_state_dict(original.state_dict())
        self._owner.coreset=self._cached_core
        self._check_binding()
    def __getattr__(self,name):return getattr(self.adapter,name)
    def binding_profile(self):
        shared=(self.adapter.coreset is self._owner.coreset is self._cached_core)
        bound=True
        if self._registered:
            bound=all(getattr(getattr(self.adapter,name),'__self__',None) is self._owner
                for name in ('fit_task','score_batch','state_dict','load_state_dict'))
        return dict(registry_wrapper=self._registered,concrete_owner_class=type(self._owner).__name__,
            fit_task_and_snapshot_shared=shared and bound,cache_backend=BACKEND)
    def _check_binding(self):
        profile=self.binding_profile()
        if not profile['fit_task_and_snapshot_shared']:raise RuntimeError('CADIC train/score/snapshot cache binding changed')
        return profile
    def fit_task(self,*args,**kwargs):
        self._check_binding();result=self.adapter.fit_task(*args,**kwargs);self._check_binding()
        if result.get('coreset')!=self._cached_core.stats():raise RuntimeError('CADIC fit-return/snapshot bank counters disagree')
        return result
    def score_batch(self,*args,**kwargs):
        self._check_binding();result=self.adapter.score_batch(*args,**kwargs);self._check_binding();return result
    def close(self):
        # The concrete CADIC adapter has no close() contract; the extractor
        # owns the ViT forward hook. Do not rely on __getattr__ forwarding.
        if self._closed:return
        cleanup=getattr(self._owner,'close',None)
        if not callable(cleanup):cleanup=getattr(getattr(self._owner,'extractor',None),'close',None)
        if callable(cleanup):cleanup()
        self._closed=True
    def method_metadata(self):
        self._check_binding()
        return dict(self.adapter.method_metadata(),execution_optimization=BACKEND,
            equivalence='same scientific update/scoring; separate execution identity; requires parity gate')
    def memory_stats(self):
        self._check_binding()
        values=self.adapter.memory_stats();profile=self._cached_core.cache_profile();cache=profile['runtime_cache_bytes']
        values['runtime_cache_peak_bytes']=profile['runtime_cache_peak_bytes']
        values['runtime_cache_bytes']=cache
        for key in ('current_state_bytes','total_deployment_bytes'):values[key]+=cache
        values['accounting_basis']+='; includes exact pair tile-min cache and Python cache layout'
        return values


def accelerate(adapter):
    return CachedCADICAdapter(adapter)


def reference_parity(coreset,candidates,query):
    """Cloned operator audit on actual normal features; never updates the main bank."""
    import time
    from training.benchmark_engine_v1 import _state_equal
    from training.hope_outer_learning_v4 import capture_rng
    if not coreset.is_full or candidates.shape[0]!=4:raise ValueError('parity requires a full bank and exactly four candidates')
    before=coreset.state_dict();rng=capture_rng()
    ref=CADICPatchCoresetV1(coreset.config,device=coreset.device)
    fast=CachedPairCoreset(coreset.config,device=coreset.device)
    ref.load_state_dict(before);fast.load_state_dict(before)
    start=time.perf_counter()
    rv,ri=ref._closest_pair();fv,fi=fast._closest_pair()
    if not torch.equal(rv,fv) or ri!=fi:raise RuntimeError('full-budget pair parity failed')
    ra=ref.update(candidates);fa=fast.update(candidates)
    if ra!=fa or not _state_equal(ref.state_dict(),fast.state_dict()):raise RuntimeError('normal candidate decision/state parity failed')
    if not _state_equal(ref.pixel_scores(query),fast.pixel_scores(query)):raise RuntimeError('reference Eq9 score parity failed')
    if not _state_equal(before,coreset.state_dict()) or not _state_equal(rng,capture_rng()):
        raise RuntimeError('reference parity altered main state/RNG')
    return dict(passed=True,backend=BACKEND,pair_exact=True,decisions_and_counters_exact=True,
        features_exact=True,pixel_and_image_scores_exact=True,main_state_rng_unchanged=True,
        candidate_vectors=int(candidates.shape[0]),bank_count=coreset.count,seconds=time.perf_counter()-start,
        scope='cloned full-budget operator fixture using four real normal patches; not an exhaustive dual 16-image stream',
        qualification='exact tile-invalidation argument plus CPU/CUDA stream regressions; no unknown-input universal testing claim')
