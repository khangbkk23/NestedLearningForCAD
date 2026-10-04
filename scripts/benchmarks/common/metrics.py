# scripts/benchmarks/common/metrics.py
import math
import numpy as np
from training.benchmark_metrics_v1 import forgetting_matrix as _forgetting

def unavailable(name, reason, source="reproduced"):
    return {"metric": name, "value": None, "availability": False, "reason": reason, "source": source}

def forgetting_matrix(matrix, formula="mean_prior_max_minus_final"):
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]: raise ValueError("continual matrix must be square")
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[1]):
            if np.isfinite(a[i, j]): raise ValueError("future continual cells must be null")
    return _forgetting(a, formula)

def validate_continual_matrix(matrix):
    a = np.asarray([[float("nan") if v is None else v for v in row] for row in matrix])
    forgetting_matrix(a)
    return True
