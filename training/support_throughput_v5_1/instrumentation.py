"""Reversible wrappers: coarse CUDA spans, aggregate nested host regions, no math changes."""
from contextlib import contextmanager, nullcontext
from collections import defaultdict
from functools import wraps
import time
import torch

def logical_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(logical_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)): return sum(logical_bytes(v) for v in value)
    return 0

class Profiler:
    def __init__(self, device='cpu', *, clock=time.perf_counter):
        self.device, self.clock = str(device), clock
        self.cursor = None; self.stack=[]; self.regions={}; self.cuda=[]; self.patches=[]
        self.counters=defaultdict(int); self.max_cuda_spans=20000
        self.enabled=True

    @contextmanager
    def region(self, name, *, cuda=False):
        if not self.enabled:
            yield
            return
        path='/'.join([f['name'] for f in self.stack]+[name])
        key=(self.cursor,path); start=self.clock(); frame=dict(name=name,children=0.)
        pair=None
        if cuda and self.device.startswith('cuda'):
            if len(self.cuda)>=self.max_cuda_spans: raise RuntimeError('Profiler CUDA span cap; preserve evidence')
            pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            pair[0].record()
        self.stack.append(frame)
        try: yield
        finally:
            if pair is not None: pair[1].record(); self.cuda.append((key,pair))
            elapsed=self.clock()-start
            self.stack.pop()
            if self.stack: self.stack[-1]['children']+=elapsed
            v=self.regions.setdefault(key,dict(calls=0,inclusive_host_seconds=0.,exclusive_host_seconds=0.,cuda_stream_span_seconds=0.))
            v['calls']+=1;v['inclusive_host_seconds']+=elapsed
            v['exclusive_host_seconds']+=max(0.,elapsed-frame['children'])

    def patch(self, owner, name, label, *, cuda=False, counter=None):
        old=getattr(owner,name)
        # Preserve staticmethod descriptors when restoring class attributes.
        own=vars(owner).get(name); had=name in vars(owner)
        @wraps(old)
        def wrapped(*args,**kwargs):
            if not self.enabled:return old(*args,**kwargs)
            if counter is not None: counter(args,kwargs)
            with self.region(label,cuda=cuda): return old(*args,**kwargs)
        setattr(owner,name,wrapped)
        self.patches.append((owner,name,own,had))

    def _hash_counter(self,args,kwargs):
        self.counters['bank_hash_calls']+=1
        self.counters['bank_hash_logical_tensor_bytes']+=logical_bytes(args[0])

    def _distance_counter(self,args,kwargs):
        kind=args[1]; block=args[2]
        self.counters['distance_blocks_'+str(kind)]+=1
        self.counters['distance_block_elements_'+str(kind)]+=block.numel()

    def _cache_counter(self,args,kwargs):
        bank=args[0]
        key='pair_cache_signature_hits' if bank._signature()==bank._cache_signature else 'pair_cache_full_rebuild_requests'
        self.counters[key]+=1

    def _refresh_counter(self,args,kwargs):
        bank=args[0]
        if bank._signature()==bank._cache_signature:
            self.counters['pair_tiles_requiring_refresh']+=len(bank._dirty)
        else:
            blocks=(bank.count+bank.config.pair_chunk_size-1)//bank.config.pair_chunk_size
            self.counters['pair_tiles_requiring_refresh']+=blocks*(blocks+1)//2

    def __enter__(self):
        try:return self.install()
        except BaseException:
            self.__exit__()
            raise

    def install(self):
        from models.hope_cad_v5_1 import learner as L, session as S, geometry as G
        from models.hope_cad_v5 import geometry as OLD
        from models.cadic_paperfaithful.cache import CachedPairCoreset
        from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1
        from training.support_benchmark_v5_1.data import NormalFeatures
        from models.feature_extractors.cadic_vit_v1 import CADICViTFeatureExtractor
        from dataset.benchmark_protocol_v2 import HashedNormalDataset
        # Module aliases are separate call sites and are all restored on exit.
        for owner,name,label,cuda in [
            (NormalFeatures,'get','provider',False),
            (HashedNormalDataset,'__getitem__','decode_preprocess_input_hash',False),
            (CADICViTFeatureExtractor,'extract_patch_features','ViT',True),
            (S.SupportSession,'observe','observe_transaction',True),
            (S.SupportSession,'_resolve','delayed_credit',True),
            (S.SupportSession,'learner_hash','learner_hash',False),
            (L.NestedUtilityLearner,'state_dict','learner_backup_or_save',False),
            (OLD.AnchorReservoir,'state_dict','anchor_backup_or_save',False),
            (L.NestedUtilityLearner,'observe','actual_and_slow_learning',True),
            (L.NestedUtilityLearner,'predict','admission_predict',True),
            (L,'prediction','pre_step_prediction',True),
            (L,'fast_step','fast_step',True),
            (L,'meta_loss','meta_4support_4query',True),
            (L.UtilityModel,'encode','visual_geometry_encode',True),
            (L.UtilityModel,'evaluate','utility_forward',True),
            (L,'norm','diagnostic_norm_transfer',True),
            (torch.autograd,'grad','autograd_grad',True),
            (torch.Tensor,'backward','meta_backward',True),
            (torch.nn.utils,'clip_grad_norm_','gradient_clip',True),
            (torch.optim.Adam,'step','Adam_step',True),
            (OLD.ForkableCoreset,'fork','bank_fork',True),
            (CachedPairCoreset,'update','full_proposal',True),
            (CachedPairCoreset,'_changed','pair_cache_invalidation',False),
            (CADICPatchCoresetV1,'_nearest','nearest_neighbor',False),
            (OLD.AnchorReservoir,'update','reservoir_update',True),
            (G,'context_statistics','context_stats',True),
            (S,'context_statistics','context_stats',True),
            (S,'risk','session_risk',True),
            (OLD,'risk','anchor_single_risk',True),
            (S,'anchor_risk','anchor_risks',True),
            (S.SupportSession,'memory_profile','memory_accounting',False),
        ]:
            self.patch(owner,name,label,cuda=cuda)
        self.patch(S,'bank_digest','bank_hash',counter=self._hash_counter)
        self.patch(CachedPairCoreset,'_closest_pair','closest_pair',counter=self._refresh_counter)
        self.patch(CachedPairCoreset,'_ensure_pair_cache','ensure_pair_cache',counter=self._cache_counter)
        self.patch(CADICPatchCoresetV1,'_record_distance_block','distance_block_counter',counter=self._distance_counter)
        return self

    def __exit__(self,*args):
        for owner,name,old,had in reversed(self.patches):
            if had: setattr(owner,name,old)
            else: delattr(owner,name)
        self.patches.clear()

    def report(self, primary_seconds):
        if self.stack: raise RuntimeError('Unclosed profiler regions')
        if self.cuda:
            torch.cuda.synchronize(self.device)
            for key,(a,b) in self.cuda:
                self.regions[key]['cuda_stream_span_seconds']+=a.elapsed_time(b)/1000.
        rows=[dict(cursor=k[0],path=k[1],**v) for k,v in sorted(self.regions.items(),key=lambda x:str(x[0]))]
        exclusive=sum(v['exclusive_host_seconds'] for v in self.regions.values())
        return dict(regions=rows,counters=dict(self.counters),primary_seconds=primary_seconds,
            attributed_exclusive_host_seconds=exclusive,unattributed_host_seconds=max(0.,primary_seconds-exclusive),
            cuda_semantics='stream-ordered elapsed spans include launch gaps; not kernel-busy time; never add to host total',
            byte_semantics='logical tensor payload counts; actual PCIe bytes not measured',
            cuda_spans=len(self.cuda),nested_times_are_not_additive=True)

def scope(profiler,name,*,cuda=False):
    return profiler.region(name,cuda=cuda) if profiler else nullcontext()
