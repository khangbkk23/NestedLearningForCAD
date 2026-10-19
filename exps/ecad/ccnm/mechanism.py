"""Frozen CCNM-01 rule. Contexts affect readout, never ContCore admissions.

Context = unit-normalized 32-D DCT-II projection of the image's patch mean.
Distance = sqrt(patch_distance**2 + alpha**2 * context_distance**2).
alpha = median patch-to-image-mean distance / sqrt(2), frozen on writer zero.
This is Euclidean NN in an augmented feature space, not a new NN algorithm.
"""
import io
import math
import sys
import torch
from exps.ecad.r1.contcore import CausalContCore
from exps.ecad.r1.distance import Euclidean

VERSION = 'ccnm_dct32_mean_unit_first_writer_v1'
SHUFFLE_SEED = 52011


def projection(dim, width):
    if not 1 <= width <= dim:
        raise ValueError('invalid context width')
    pos = torch.arange(dim, dtype=torch.float64)[:, None] + .5
    freq = torch.arange(width, dtype=torch.float64)[None]
    p = torch.cos(math.pi * pos * freq / dim) * math.sqrt(2 / dim)
    p[:, 0] /= math.sqrt(2)
    return p.float()


class CCNM:
    def __init__(self, *, width=32, device='cpu', **allocator):
        self.model = CausalContCore(device=device, **allocator)
        self.width = width
        self.projection = projection(self.model.dim, width).to(device)
        self.contexts = torch.empty(0, width, device=device)
        self.alpha = torch.zeros((), dtype=torch.float32, device=device)

    @torch.no_grad()
    def context(self, features):
        if (features.shape != (self.model.patches, self.model.dim)
                or features.dtype != torch.float32 or features.device != self.model.bank.device
                or not bool(torch.isfinite(features).all())):
            raise ValueError('invalid full-grid query')
        c = features.mean(0) @ self.projection
        return c / c.norm().clamp_min(1e-12)

    @torch.no_grad()
    def update(self, features, *, event):
        if event != self.model.next_event:
            raise ValueError('noncausal arrival')
        c = self.context(features)
        if event == 0:
            alpha = (features - features.mean(0)).norm(dim=1).median() / math.sqrt(2)
        else:
            alpha = self.alpha
        old_ids, old_contexts = self.model.serials, self.contexts
        stats = self.model.update(features, event=event)
        ids = self.model.serials
        historical = ids < event * self.model.patches
        contexts = c.expand(len(ids), -1).clone()
        contexts[historical] = old_contexts[torch.searchsorted(old_ids, ids[historical])]
        self.contexts, self.alpha = contexts, alpha.clone()
        return stats

    def shuffled_contexts(self):
        # Permute IMAGE contexts, preserving multiplicities within each image.
        events, inverse, counts = torch.unique_consecutive(
            self.model.serials // self.model.patches, return_inverse=True, return_counts=True)
        starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        perm = torch.randperm(len(events), generator=torch.Generator().manual_seed(SHUFFLE_SEED))
        return self.contexts[starts][perm.to(starts.device)][inverse]

    @torch.no_grad()
    def score(self, features, *, mode='context', chunk=None, weight=1., query_context=None):
        if mode not in ('raw', 'context', 'shuffled', 'routing') or weight < 0 or not math.isfinite(weight):
            raise ValueError('invalid readout')
        if not len(self.model.bank):
            raise ValueError('empty bank')
        if (features.ndim != 2 or features.shape[1] != self.model.dim
                or features.dtype != torch.float32 or features.device != self.model.bank.device
                or not bool(torch.isfinite(features).all())):
            raise ValueError('invalid patch query')
        rows = torch.arange(len(self.model.bank), device=features.device)
        if mode != 'raw' and weight != 0:
            c = self.context(features) if query_context is None else query_context
            if c.shape != (self.width,) or c.device != features.device or not bool(torch.isfinite(c).all()):
                raise ValueError('invalid query context')
            ctx = self.shuffled_contexts() if mode == 'shuffled' else self.contexts
            penalty = ((ctx - c).square().sum(1) * (self.alpha * weight).square())
        else:
            penalty = torch.zeros(len(rows), device=features.device)
        if mode == 'routing':
            # Training-free nearest LIVE source-image prototype, then patch NN.
            # All prototypes derive from current associated contexts; no archive.
            origin = self.model.serials // self.model.patches
            chosen = origin[penalty.argmin()]
            rows = rows[origin == chosen]
        bank = self.model.bank[rows] if mode == 'routing' else self.model.bank
        kernel = Euclidean(chunk or self.model.chunk)
        best = torch.full((len(features),), torch.inf, device=features.device)
        indices = torch.full((len(features),), -1, device=features.device, dtype=torch.int64)
        for i, j, distances in kernel.tiles(features, bank):
            if mode in ('context', 'shuffled') and weight != 0:
                distances = (distances.square() + penalty[j:j+distances.shape[1]]).sqrt()
            values, local = distances.min(1)
            target = slice(i, i + len(values))
            better = values < best[target]
            indices[target] = torch.where(better, rows[j + local], indices[target])
            best[target] = torch.minimum(values, best[target])
        return {'scores': best, 'ids': self.model.serials[indices]}

    def state_dict(self):
        return {'schema': VERSION, 'width': self.width, 'allocator': self.model.state_dict(),
                'projection': self.projection.clone(), 'contexts': self.contexts.clone(),
                'alpha': self.alpha.clone()}

    @classmethod
    def from_state_dict(cls, s, device='cpu'):
        if s['schema'] != VERSION:
            raise ValueError('unknown context version')
        model = CausalContCore.from_state_dict(s['allocator'], device=device)
        p, c, a = (s[k] for k in ('projection', 'contexts', 'alpha'))
        if (p.dtype != torch.float32 or p.shape != (model.dim, s['width'])
                or not torch.equal(p.cpu(), projection(model.dim, s['width']))
                or c.dtype != torch.float32 or c.shape != (len(model.bank), s['width'])
                or a.dtype != torch.float32 or a.shape != ()
                or not all(bool(torch.isfinite(t).all()) for t in (p, c, a)) or a < 0):
            raise ValueError('invalid context checkpoint')
        obj = cls.__new__(cls)
        obj.model, obj.width = model, s['width']
        obj.projection, obj.contexts, obj.alpha = [t.to(device).clone() for t in (p, c, a)]
        return obj

    def ledger(self):
        tensors = {'bank': self.model.bank, 'serials': self.model.serials,
                   'projection': self.projection, 'contexts': self.contexts, 'alpha': self.alpha}
        payload = {k: t.untyped_storage().nbytes() for k, t in tensors.items()}
        # Full Python object graph including tensor headers; tensor payload separate.
        seen = set()
        def size(x):
            if id(x) in seen:
                return 0
            seen.add(id(x))
            if isinstance(x, torch.Tensor):
                return sys.getsizeof(x)
            n = sys.getsizeof(x)
            if isinstance(x, dict):
                n += sum(size(k) + size(v) for k, v in x.items())
            elif hasattr(x, '__dict__'):
                n += size(vars(x))
            return n
        headers = size(self)
        f = io.BytesIO()
        torch.save(self.state_dict(), f)
        return {'tensor_bytes': payload, 'tensor_total_bytes': sum(payload.values()),
                'python_graph_bytes': headers, 'online_total_bytes': sum(payload.values()) + headers,
                'serialized_bytes': len(f.getvalue()),
                'routing_extra_persistent_bytes': 0, 'shuffle_extra_persistent_bytes': 0,
                'encoder_shared_parameter_bytes': 343231488,
                'score_workspace': 'chunk matrices, context permutation and routing indices are transient; no cached index'}
