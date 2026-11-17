"""SCRC core for NL-CAD-PILOT-00 P0 (spec: agents/plans/nl_cad_design_00_spec.md, sections 2-5).

Levels implemented here (Level 0, the C2-DOM DOM-PQ bank, is used read-only and never modified):

* Level 1 (fast): K_t = U diag(lam) U^T, U in R^{d x r} orthonormal, lam >= 0, r = 32. One update per task boundary,
  the exact minimiser of 0.5||K - C_hat||_F^2 + 0.5 rho ||K - K_prev||_F^2 over PSD rank <= r:
  K_t = Pi_r((C_hat + rho K_prev) / (1 + rho)), computed by the FP64 thin SVD of
  B = [sqrt(rho/(1+rho)) U_prev diag(lam_prev)^{1/2}, ((1+rho) m)^{-1/2} R^T].
* Level 2 (slow) interface: theta = (theta_rho[4], theta_tau, theta_gamma); rho_t = 64 sigmoid(theta_rho . h_t),
  h_t = [1, log(1+t), log(1+nu_t), log a_t]; tau_t = exp(theta_tau) tr(K_t)/r; gamma = 2 sigmoid(theta_gamma).
  Fixed policies of the preregistered controls (cumulative rho = t-1, latest rho = 0, constant) share the interface.
* Score: s(x) = min_y ||M^{1/2}(x - y)||, M^{1/2} = (I + K/tau)^{-gamma/2} = I - U diag(c) U^T,
  c_i = 1 - (1 + lam_i/tau)^{-gamma/2}.

Persistence and precision contract (P0 decision, recorded in the P0 report):
* Persistent fast state = U FP32 [d, r] + lam FP32 [r] (98,432 B). Slow state = 6 FP32 + int64 counter (32 B).
* Working precision is FP64. The canonical FP64 reconstruction of a persisted state is
  U64 = Q of the QR factorisation of U32 (columns with lam > 0; sign fixed so diag(R) > 0), lam64 = lam32.
* Every update consumes the reconstruction of the persisted previous state and immediately persists its own
  result, so an interrupted-and-reloaded stream is bitwise identical to an uninterrupted one.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import torch

D_FEAT = 768
RANK = 32
M_RES = 4096
RHO_MAX = 64.0


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


# ------------------------------------------------------------------ fast state

@dataclass
class FastState:
    U: torch.Tensor                 # float32 [d, r]; columns with lam == 0 are zero
    lam: torch.Tensor               # float32 [r], non-increasing, >= 0
    t: int = 0                      # number of boundaries absorbed

    @staticmethod
    def empty(d=D_FEAT, r=RANK):
        return FastState(torch.zeros(d, r), torch.zeros(r), 0)

    def tensors(self):
        return {"U": self.U, "lam": self.lam, "t": torch.tensor([self.t], dtype=torch.int64)}

    def nbytes(self):
        return int(self.U.numel() * self.U.element_size() + self.lam.numel() * self.lam.element_size())

    def to_bytes(self):
        buf = io.BytesIO()
        torch.save(self.tensors(), buf)
        return buf.getvalue()

    @staticmethod
    def from_bytes(b):
        o = torch.load(io.BytesIO(b), weights_only=True)
        return FastState(o["U"], o["lam"], int(o["t"][0]))


def factors64(state):
    """Canonical FP64 reconstruction (U64 [d, k] exactly orthonormal up to FP64 rounding, lam64 [k]), k = #lam > 0."""
    lam = state.lam.double()
    keep = lam > 0
    k = int(keep.sum())
    d = state.U.shape[0]
    if k == 0:
        return torch.zeros(d, 0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
    require(bool(keep[:k].all()), "lam must be sorted non-increasing with zeros last")
    Q, Rr = torch.linalg.qr(state.U[:, :k].double())
    s = torch.sign(torch.diagonal(Rr))
    s[s == 0] = 1.0
    return Q * s[None], lam[:k].clone()


def _fix_signs(U):
    """Deterministic sign: the entry of largest |value| in each column is positive (lowest index on ties)."""
    idx = U.abs().argmax(0)
    s = torch.sign(U[idx, torch.arange(U.shape[1])])
    s[s == 0] = 1.0
    return U * s[None]


# ------------------------------------------------------------------ level 1: proximal update

def prox_update(state, R, rho, r=RANK, return_dense_check=False):
    """One fast update. R: residual rows [m, d] (float32 or float64). Returns (new FastState, diagnostics)."""
    require(rho >= 0 and math.isfinite(rho), f"bad rho {rho}")
    R64 = R.double()
    m, d = R64.shape
    require(m > 0, "empty residual set")
    U, lam = factors64(state)
    blocks = []
    if rho > 0 and len(lam):
        blocks.append(U * (lam.sqrt() * math.sqrt(rho / (1.0 + rho)))[None])
    blocks.append(R64.T / math.sqrt((1.0 + rho) * m))
    B = torch.cat(blocks, 1)                                         # d x (k + m)
    Us, S, _ = torch.linalg.svd(B, full_matrices=False)
    ev = S.square()                                                  # eigenvalues of K_tilde, non-increasing
    k = min(r, int((ev > 0).sum()))
    Un = torch.zeros(d, r, dtype=torch.float64)
    ln = torch.zeros(r, dtype=torch.float64)
    Un[:, :k] = _fix_signs(Us[:, :k])
    ln[:k] = ev[:k]
    new = FastState(Un.float().contiguous(), ln.float().contiguous(), state.t + 1)
    diag = {"rho": float(rho), "lam_r": float(ev[r - 1]) if len(ev) >= r else 0.0,
            "lam_r1": float(ev[r]) if len(ev) > r else 0.0, "lam_1": float(ev[0]),
            "trace_tilde": float(ev.sum()), "trace_kept": float(ev[:r].sum()),
            "rel_gap_r": float((ev[r - 1] - ev[r]) / ev[r - 1]) if len(ev) > r and ev[r - 1] > 0 else float("nan")}
    if return_dense_check:
        diag["_ev"] = ev
    return new, diag


def prox_update_gram(state, Cdense, rho, r=RANK):
    """Same minimiser as ``prox_update`` from a precomputed dense C_hat (FP64 d x d, theta-independent, cacheable
    across ES evaluations): eigh of K_tilde = (C_hat + rho K_prev)/(1+rho). Offline-only speed path; equivalence to
    the SVD path is tested (P0)."""
    U, lam = factors64(state)
    Kt = (Cdense + rho * ((U * lam[None]) @ U.T)) / (1.0 + rho) if rho > 0 and len(lam) else Cdense / (1.0 + rho)
    Kt = 0.5 * (Kt + Kt.T)
    ev, V = torch.linalg.eigh(Kt)
    ev, V = ev.flip(0).clamp_min(0), V.flip(1)
    d = Cdense.shape[0]
    k = min(r, int((ev > 0).sum()))
    Un = torch.zeros(d, r, dtype=torch.float64)
    ln = torch.zeros(r, dtype=torch.float64)
    Un[:, :k] = _fix_signs(V[:, :k])
    ln[:k] = ev[:k]
    return FastState(Un.float().contiguous(), ln.float().contiguous(), state.t + 1), {"lam_r1": float(ev[r])}


def k_tilde_dense(state, R, rho):
    """Dense FP64 K_tilde = (C_hat + rho K_prev)/(1+rho) (tests only)."""
    U, lam = factors64(state)
    R64 = R.double()
    C = R64.T @ R64 / len(R64)
    Kp = (U * lam[None]) @ U.T
    return (C + rho * Kp) / (1.0 + rho)


def dense_K(state):
    U, lam = factors64(state)
    return (U * lam[None]) @ U.T


# ------------------------------------------------------------------ slow -> fast: features and stiffness

def novelty_features(state, R, t):
    """h_t = [1, log(1+t), log(1+nu_t), log a_t] from low-rank factors and a transient d x d Gram (never stored).
    nu_t = ||C_hat - K_prev||_F / ||K_prev||_F, a_t = tr C_hat / tr K_prev. t is 1-based; defined for t >= 2."""
    U, lam = factors64(state)
    require(len(lam) > 0 and float(lam.sum()) > 0, "novelty undefined for an empty fast state")
    R64 = R.double()
    m = len(R64)
    trC = float(R64.square().sum()) / m
    G = R64.T @ R64 / m
    C_F2 = float(G.square().sum())
    P = R64 @ U
    CK = float((P.square().sum(0) * lam).sum()) / m
    K_F2 = float(lam.square().sum())
    num2 = max(C_F2 - 2.0 * CK + K_F2, 0.0)
    nu = math.sqrt(num2) / math.sqrt(K_F2)
    a = trC / float(lam.sum())
    return {"nu": nu, "a": a, "h": [1.0, math.log1p(t), math.log1p(nu), math.log(a)]}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


@dataclass
class Theta:
    rho: tuple = (0.0, 0.0, 0.0, 0.0)
    tau: float = 0.0
    gamma_raw: float = 0.0

    def vector(self):
        return list(self.rho) + [self.tau, self.gamma_raw]

    @staticmethod
    def from_vector(v):
        v = [float(x) for x in v]
        return Theta(tuple(v[:4]), v[4], v[5])

    @property
    def gamma(self):
        return 2.0 * sigmoid(self.gamma_raw)

    def nbytes(self):
        return 6 * 4 + 8                 # 6 FP32 + int64 boundary counter


class Policy:
    """rho_t for t >= 2 (rho_1 := 0 for every policy)."""
    name = "base"

    def rho(self, t, h):
        raise NotImplementedError

    def __call__(self, t, h):
        return 0.0 if t <= 1 else float(self.rho(t, h))


class Learned(Policy):
    name = "learned"

    def __init__(self, theta_rho, rho_max=RHO_MAX):
        self.w, self.rho_max = [float(x) for x in theta_rho], rho_max

    def rho(self, t, h):
        return self.rho_max * sigmoid(sum(a * b for a, b in zip(self.w, h)))


class Cumulative(Policy):
    name = "cumulative"

    def rho(self, t, h):
        return t - 1.0


class Latest(Policy):
    name = "latest"

    def rho(self, t, h):
        return 0.0


class Constant(Policy):
    name = "constant"

    def __init__(self, c):
        self.c = float(c)

    def rho(self, t, h):
        return self.c


class CausalPermuted(Learned):
    """A6 control (P0 amendment proposal): learned map applied to h_{pi(t)} with pi(t) drawn by a fixed hash from the
    already-observed boundaries {2..t}. The preregistered 'h_t permuted across boundaries' would read future h values
    in an online stream; this causal form needs the h history (4 FP32 per boundary) as extra persistent state."""
    name = "causal_permuted"

    def __init__(self, theta_rho, key=0, rho_max=RHO_MAX):
        super().__init__(theta_rho, rho_max)
        self.key, self.hist = key, {}

    def rho(self, t, h):
        import hashlib
        self.hist[t] = list(h)
        j = 2 + int(hashlib.sha256(f"NL-CAD-PILOT-00/A6/{self.key}/{t}".encode()).hexdigest(), 16) % (t - 1)
        return super().rho(t, self.hist[j])


# ------------------------------------------------------------------ metric

@dataclass
class Metric:
    U: torch.Tensor                  # FP64 [d, k]
    c: torch.Tensor                  # FP64 [k], M^{1/2} = I - U diag(c) U^T, c in [0, 1)
    gamma: float
    tau: float
    mu_t: float                      # smallest eigenvalue of M (actual)
    mu_theta: float                  # theta-only lower bound (1 + r e^{-theta_tau})^{-gamma}
    extra: dict = field(default_factory=dict)

    @property
    def is_identity(self):
        return bool((self.c == 0).all())

    @property
    def kappa_t(self):
        return self.mu_t ** -0.5

    @property
    def kappa_theta(self):
        return self.mu_theta ** -0.5


def make_metric(state, theta_tau, gamma, r=RANK):
    U, lam = factors64(state)
    trK = float(lam.sum())
    mu_theta = (1.0 + r * math.exp(-theta_tau)) ** (-gamma)
    if trK == 0.0 or len(lam) == 0:
        return Metric(U, torch.zeros(len(lam), dtype=torch.float64), gamma, float("nan"), 1.0, mu_theta)
    tau = math.exp(theta_tau) * trK / r
    f = (1.0 + lam / tau)
    c = 1.0 - f.pow(-gamma / 2.0)
    mu_t = float(f.max()) ** (-gamma)
    return Metric(U, c, gamma, tau, mu_t, mu_theta)


def apply_sqrt(x, metric, chunk=65536):
    """z = M^{1/2} x computed in FP64 (rows of x), returned FP64."""
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    U = metric.U.to(x.device)
    c = metric.c.to(x.device)
    for s in range(0, len(x), chunk):
        X = x[s:s + chunk].double()
        out[s:s + chunk] = X - ((X @ U) * c[None]) @ U.T
    return out


def metric_dist_pairs(q, y, metric, chunk=256):
    """FP64 reference D_M(q_j, y_j) = ||w||, w = diff - sum_i c_i (u_i . diff) u_i, diff = q_j - y_j, for aligned
    pairs: the definition every exactness claim is measured against. Elementwise products with fixed-length
    reductions only (batch-invariant on CPU). For the identity metric w == diff bitwise, so the value equals the
    canonical exact.reference bitwise."""
    dev = q.device
    Ut = metric.U.T.contiguous().to(dev)                        # [k, d]
    c = metric.c.to(dev)
    out = torch.empty(len(q), dtype=torch.float64, device=dev)
    for s in range(0, len(q), chunk):
        diff = q[s:s + chunk].double() - y[s:s + chunk].double()                    # [n, d]
        p = (diff[:, None, :] * Ut[None]).sum(2)                                     # [n, k]
        w = diff - ((p * c[None])[:, :, None] * Ut[None]).sum(1)                     # [n, d]
        out[s:s + chunk] = w.square().sum(1).sqrt()
    return out
