"""Common FP32 Euclidean kernel; no fitted index or persistent distance cache.

Inputs and outputs are FP32. Cancellation-sized norm/dot results are recomputed
by direct FP64 subtraction, then square-rooted before casting. Chunking bounds
GEMM and repair workspace. Call configure() before using CUDA.
"""
from dataclasses import dataclass
import os
import torch


def configure():
    # Set before creating a CUDA context; required by deterministic CUDA GEMM.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


@dataclass
class Work:
    pairs: int = 0
    blocks: int = 0
    repair_pairs: int = 0
    dynamic_compactions: int = 0
    host_scalar_reads: int = 0
    explicit_cuda_syncs: int = 0
    max_block_bytes: int = 0


class Euclidean:
    def __init__(self, chunk=256, repair_chunk=1024, work=None):
        if chunk < 1 or repair_chunk < 1:
            raise ValueError("positive chunks required")
        self.chunk, self.repair_chunk, self.work = chunk, repair_chunk, work

    @torch.no_grad()
    def block(self, x, y):
        if x.dtype != torch.float32 or y.dtype != torch.float32:
            raise ValueError("FP32 descriptors required")
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.device != y.device:
            raise ValueError("matching [N,D] arrays on one device required")
        with torch.autocast(device_type=x.device.type, enabled=False):
            norms = x.square().sum(1)[:, None] + y.square().sum(1)[None, :]
            squared = norms - 2 * (x @ y.T)
            # nonzero compacts dynamic output and can synchronize CUDA. Charge it.
            near = squared <= 64 * torch.finfo(torch.float32).eps * norms
            rows, cols = near.nonzero(as_tuple=True)
            if self.work is not None:
                self.work.pairs += len(x) * len(y)
                self.work.blocks += 1
                self.work.repair_pairs += len(rows)
                self.work.dynamic_compactions += int(x.is_cuda)
                self.work.max_block_bytes = max(self.work.max_block_bytes, squared.numel() * 4)
            result = squared.clamp_min_(0).sqrt_()
            for start in range(0, len(rows), self.repair_chunk):
                rr, cc = rows[start:start+self.repair_chunk], cols[start:start+self.repair_chunk]
                result[rr, cc] = (x[rr].double() - y[cc].double()).square().sum(1).sqrt().float()
            return result

    def tiles(self, x, y):
        for i in range(0, len(x), self.chunk):
            for j in range(0, len(y), self.chunk):
                yield i, j, self.block(x[i:i+self.chunk], y[j:j+self.chunk])

    def matrix(self, x, y):
        out = torch.empty((len(x), len(y)), device=x.device, dtype=torch.float32)
        for i, j, d in self.tiles(x, y):
            out[i:i+len(d), j:j+d.shape[1]] = d
        return out

    def nearest(self, x, bank):
        best = torch.full((len(x),), torch.inf, device=x.device, dtype=torch.float32)
        for i, _, d in self.tiles(x, bank):
            best[i:i+len(d)] = torch.minimum(best[i:i+len(d)], d.min(1).values)
        return best


@torch.no_grad()
def global_nn(features, bank, kernel=None):
    """Identical task-free scorer for every memory arm; never writes learner state."""
    if not len(bank):
        raise ValueError("cannot score an empty bank")
    return (kernel or Euclidean()).nearest(features, bank)
