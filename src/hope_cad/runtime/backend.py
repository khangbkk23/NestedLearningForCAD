"""Portable alias for the reviewed fixed-shape NN implementation, not a new distance rule."""
from contextlib import contextmanager
import torch
from hope_cad.baselines.cadic.bank import CADICPatchCoresetV1,euclidean_distance_mm

ORIGINAL_NEAREST=CADICPatchCoresetV1._nearest

@torch.no_grad()
def nearest_fixed_shape_writes(self,query,bank):
    if query.ndim!=2 or bank.ndim!=2 or query.shape[1]!=bank.shape[1] or not bank.shape[0]:
        raise ValueError('nearest requires [Q,D] and a nonempty [M,D] bank')
    best=torch.full((query.shape[0],),float('inf'),dtype=torch.float32,device=query.device)
    indices=torch.zeros(query.shape[0],dtype=torch.long,device=query.device)
    for query_start in range(0,query.shape[0],self.config.query_chunk_size):
        query_part=query[query_start:query_start+self.config.query_chunk_size]
        best_part=best[query_start:query_start+len(query_part)]
        index_part=indices[query_start:query_start+len(query_part)]
        for start in range(0,bank.shape[0],self.config.chunk_size):
            part=bank[start:start+self.config.chunk_size]
            distances=euclidean_distance_mm(query_part,part)
            self._record_distance_block('nearest',distances)
            values,local=torch.min(distances,dim=1)
            mask=values<best_part
            best_part.copy_(torch.where(mask,values,best_part))
            index_part.copy_(torch.where(mask,start+local,index_part))
    return best,indices

@contextmanager
def activation(optimized=True):
    if CADICPatchCoresetV1._nearest is not ORIGINAL_NEAREST:
        raise RuntimeError('Unknown NN operator already active; fresh process required')
    if optimized:CADICPatchCoresetV1._nearest=nearest_fixed_shape_writes
    try:yield
    finally:CADICPatchCoresetV1._nearest=ORIGINAL_NEAREST
