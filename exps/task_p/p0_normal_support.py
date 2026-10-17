"""Isolated P0 support selectors. Learners receive descriptors and opaque serial IDs only."""
from __future__ import annotations

import hashlib
import io
import sys
import torch

BUDGET = 8 * 1024**2
METADATA_RESERVE = 16 * 1024
DIM = 768
ARMS = ("FROZEN_FIRST", "RESERVOIR", "STREAM_GREEDY", "WITNESS_PROTECTED")
RNG_BUFFER_BYTES = 5056
RESERVOIR_CAPACITY = (BUDGET - METADATA_RESERVE - RNG_BUFFER_BYTES) // (DIM * 4 + 8)
GREEDY_CAPACITY = (BUDGET - METADATA_RESERVE - RNG_BUFFER_BYTES) // (DIM * 4 + 8 + 8 + 8)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def distances(x, y):
    """FP64 norm/dot Euclidean formula; directly recompute cancellation-sized entries."""
    require(x.ndim == y.ndim == 2 and x.shape[1] == y.shape[1], "bad distance shapes")
    a, b = x.double(), y.double()
    norms = a.square().sum(1)[:, None] + b.square().sum(1)[None, :]
    squared = (norms - 2 * (a @ b.T)).clamp_min_(0)
    near = squared <= 64 * torch.finfo(torch.float64).eps * norms
    rows, cols = near.nonzero(as_tuple=True)
    if len(rows):
        squared[rows, cols] = (a[rows] - b[cols]).square().sum(1)
    result = squared.sqrt_()
    require(bool(torch.isfinite(result).all()), "nonfinite distance")
    return result


def nearest_two(matrix):
    require(matrix.shape[1] >= 2, "two neighbors require two bank vectors")
    first, index = matrix.min(1)
    other = matrix.clone()
    other[torch.arange(len(other)), index] = torch.inf
    second = other.min(1).values
    return first, second, index


def fingerprint(state):
    h = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            q = value.detach().contiguous()
            h.update(str((q.shape, q.dtype)).encode()); h.update(q.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode()); add(value[key])
        else:
            h.update(repr(value).encode())
    add(state)
    return h.hexdigest()


def tensor_bytes(state):
    return sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))


def serialized_bytes(state):
    stream = io.BytesIO(); torch.save(state, stream); return stream.tell()


class Support:
    def __init__(self, method, capacity=None, dim=DIM, witness_capacity=128):
        require(method in ARMS, "unknown method")
        self.method = method
        self.capacity = capacity if capacity is not None else (
            2350 if method == "WITNESS_PROTECTED" else
            RESERVOIR_CAPACITY if method == "RESERVOIR" else GREEDY_CAPACITY)
        self.bank = torch.zeros(self.capacity, dim, dtype=torch.float32)
        self.ids = torch.full((self.capacity,), -1, dtype=torch.int64)
        self.n = 0; self.seen = 0; self.events = 0
        self.rng = torch.Generator().manual_seed(7202 if method == "RESERVOIR" else 7203)
        if method != "RESERVOIR":
            self.nn_distance = torch.full((self.capacity,), torch.inf, dtype=torch.float64)
            self.nn_index = torch.full((self.capacity,), -1, dtype=torch.int64)
        if method == "WITNESS_PROTECTED":
            self.witnesses = torch.zeros(witness_capacity, dim, dtype=torch.float32)
            self.witness_ids = torch.full((witness_capacity,), -1, dtype=torch.int64)
            self.wn = 0; self.wseen = 0

    def state(self):
        out = {k: v for k, v in vars(self).items() if k != "rng"}
        out["rng_state"] = self.rng.get_state()
        return out

    def ledger(self):
        state = self.state()
        payload = tensor_bytes(state)
        # Conservative declared envelope includes all Python metadata, container/tensor
        # headers and serialized overhead; RNG byte state is already counted above.
        python_bytes = sys.getsizeof(self) + sys.getsizeof(vars(self)) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) for k, v in state.items())
        require(python_bytes < METADATA_RESERVE, "metadata exceeds declared envelope")
        total = payload + METADATA_RESERVE
        require(total <= BUDGET, "persistent state exceeds 8 MiB")
        return {"tensor_bytes": payload, "metadata_envelope_bytes": METADATA_RESERVE,
                "python_object_bytes_estimate": python_bytes, "charged_persistent_bytes": total,
                "descriptor_slots": self.capacity, "occupied": self.n,
                "witness_slots": len(self.witnesses) if hasattr(self, "witnesses") else 0,
                "witness_occupied": self.wn if hasattr(self, "wn") else 0,
                "read_only_descriptor_bytes": self.n * self.bank.shape[1] * 4}

    def score(self, queries):
        require(self.n > 0, "empty support bank")
        matrix = distances(queries, self.bank[:self.n])
        values, idx = matrix.min(1)
        return values, self.ids[idx]

    def _put(self, slot, vector, serial):
        """Exact incremental nearest-distinct graph; no persistent quadratic matrix."""
        adding = slot == self.n
        old_n = self.n
        affected = [] if adding else torch.where(self.nn_index[:old_n] == slot)[0].tolist()
        self.bank[slot] = vector; self.ids[slot] = serial
        if adding: self.n += 1
        if self.n == 1: return
        d = distances(vector[None], self.bank[:self.n])[0]; d[slot] = torch.inf
        changed = (d < self.nn_distance[:self.n]) | (
            (d == self.nn_distance[:self.n]) & (slot < self.nn_index[:self.n]))
        self.nn_distance[:self.n][changed] = d[changed]
        self.nn_index[:self.n][changed] = slot
        affected = sorted(set(affected + [slot]))
        q = distances(self.bank[affected], self.bank[:self.n])
        q[torch.arange(len(affected)), torch.tensor(affected)] = torch.inf
        v, ix = q.min(1)
        self.nn_distance[affected] = v; self.nn_index[affected] = ix

    def _witness_update(self, x, ids):
        for row, serial in zip(x, ids):
            self.wseen += 1
            slot = self.wn if self.wn < len(self.witnesses) else int(
                torch.randint(self.wseen, (1,), generator=self.rng))
            if slot < len(self.witnesses):
                self.witnesses[slot] = row; self.witness_ids[slot] = serial
                self.wn = min(self.wn + 1, len(self.witnesses))

    def _witness_scale(self):
        if self.wn < 2: return None
        d = distances(self.witnesses[:self.wn], self.witnesses[:self.wn])
        d.fill_diagonal_(torch.inf)
        positive = d.min(1).values
        positive = positive[(positive > 0) & torch.isfinite(positive)]
        if not len(positive): return None
        return float(torch.quantile(positive, 0.5))

    def update(self, x, ids, write=True, guard_enabled=True):
        require(x.ndim == 2 and x.shape[1] == self.bank.shape[1] and x.dtype == torch.float32,
                "wrong proposal representation")
        require(bool(torch.isfinite(x).all()) and len(ids) == len(x), "invalid proposals")
        stats = {"inserted": 0, "replaced": 0, "rejected": 0, "evaluated_replacements": 0,
                 "nonzero_witness_cost_proposals": 0, "guard_effect_attempts": 0,
                 "guard_rejected_proposals": 0, "acquisition_gain_sum": 0.,
                 "witness_cost_sum": 0., "accepted_readout_changes": 0,
                 "rejected_bank_unchanged": True, "decisions": []}
        if not write:
            stats["no_write"] = True; return stats
        self.events += 1; self.seen += len(x)
        if self.method == "FROZEN_FIRST" and self.events > 100:
            stats["rejected"] = len(x); return stats
        if self.method == "RESERVOIR":
            for j, (vector, serial) in enumerate(zip(x, ids)):
                arrived = self.seen - len(x) + j + 1
                slot = self.n if self.n < self.capacity else int(
                    torch.randint(arrived, (1,), generator=self.rng))
                if slot < self.capacity:
                    adding = self.n < self.capacity
                    self.bank[slot] = vector; self.ids[slot] = serial
                    self.n += int(adding)
                    stats["inserted" if adding else "replaced"] += 1
                else: stats["rejected"] += 1
            return stats
        historical_w = self.witnesses[:self.wn] if self.method == "WITNESS_PROTECTED" else None
        scale = self._witness_scale() if historical_w is not None else None
        stats["witness_scale"] = scale
        remaining = list(range(len(x)))
        # Current-image distances are mutable transient workspace, never retained.
        q = x if historical_w is None else torch.cat((x, historical_w))
        matrix = distances(q, self.bank[:self.n]) if self.n else None
        while remaining:
            if self.n:
                novelty = matrix[remaining].min(1).values
                chosen = remaining[int(novelty.argmax())]
                prior = float(novelty.max())
            else: chosen, prior = remaining[0], float("inf")
            remaining.remove(chosen)
            if prior == 0:
                stats["rejected"] += 1; continue
            if self.n < self.capacity:
                slot = self.n; self._put(slot, x[chosen], int(ids[chosen]))
                col = distances(q, x[chosen:chosen + 1])
                require(float(col[chosen, 0]) == 0 and prior > 0, "accepted insertion lacks causal readout change")
                matrix = col if matrix is None else torch.cat((matrix, col), 1)
                stats["inserted"] += 1; stats["accepted_readout_changes"] += 1
                continue
            shortlist = torch.argsort(self.nn_distance[:self.n], stable=True)[:min(16, self.n)]
            if self.method != "WITNESS_PROTECTED":
                slot = int(shortlist[0])
                if prior <= float(self.nn_distance[slot]):
                    stats["rejected"] += 1 + len(remaining); break
                self._put(slot, x[chosen], int(ids[chosen]))
                matrix[:, slot] = distances(q, x[chosen:chosen + 1])[:, 0]
                stats["replaced"] += 1; stats["accepted_readout_changes"] += 1
                continue
            bank_before = fingerprint({'bank': self.bank, 'ids': self.ids})
            near, second, near_idx = nearest_two(matrix)
            removed = torch.where(near_idx[:, None] == shortlist[None], second[:, None], near[:, None])
            to_new = distances(q, x[chosen:chosen + 1])[:, 0]
            tentative = torch.minimum(removed, to_new[:, None])
            gains = (near[:len(x), None] - tentative[:len(x)]).mean(0)
            increments = (tentative[len(x):] - near[len(x):, None]).clamp_min(0)
            costs = increments.mean(0) if self.wn else torch.zeros(len(shortlist), dtype=torch.float64)
            tails = torch.quantile(increments, 0.9, dim=0) if self.wn else torch.full_like(costs, torch.inf)
            legal = (gains > costs) & (tails <= .05 * scale) if scale is not None else torch.zeros_like(gains, dtype=torch.bool)
            unguarded = int(gains.argmax()) if bool((gains > 0).any()) else None
            merit = (gains - costs).masked_fill(~legal, -torch.inf)
            best = int(merit.argmax()) if bool(legal.any()) else None
            if not guard_enabled: best = unguarded  # Synthetic intervention only.
            changed = unguarded is not None and (best is None or best != unguarded)
            stats["guard_effect_attempts"] += int(changed)
            stats["evaluated_replacements"] += len(shortlist)
            stats["nonzero_witness_cost_proposals"] += int((costs > 0).sum())
            stats["guard_rejected_proposals"] += int(((gains > 0) & ~legal).sum())
            decision = {"candidate_row": chosen, "unguarded_slot": int(shortlist[unguarded]) if unguarded is not None else None,
                        "selected_slot": int(shortlist[best]) if best is not None else None,
                        "guard_changes_decision": changed, "proposals": len(shortlist),
                        "positive_witness_costs": int((costs > 0).sum()),
                        "max_witness_cost": float(costs.max()), "max_tail_cost": float(tails.max()) if self.wn else None,
                        "guard_blocked_positive_gain": int(((gains > 0) & ~legal).sum())}
            if best is None:
                require(bank_before == fingerprint({'bank': self.bank, 'ids': self.ids}), "rejected decision changed score bank")
                stats["rejected"] += 1; stats["decisions"].append(decision); continue
            slot = int(shortlist[best])
            require(float(to_new[chosen]) == 0 and prior > 0, "accepted replacement lacks causal readout change")
            decision.update(gain=float(gains[best]), witness_cost=float(costs[best]), tail_cost=float(tails[best]),
                            before_candidate_distance=prior, after_candidate_distance=float(to_new[chosen]))
            self._put(slot, x[chosen], int(ids[chosen]))
            matrix[:, slot] = to_new
            stats["replaced"] += 1; stats["accepted_readout_changes"] += 1
            stats["acquisition_gain_sum"] += float(gains[best]); stats["witness_cost_sum"] += float(costs[best])
            stats["decisions"].append(decision)
        if self.method == "WITNESS_PROTECTED": self._witness_update(x, ids)
        return stats
