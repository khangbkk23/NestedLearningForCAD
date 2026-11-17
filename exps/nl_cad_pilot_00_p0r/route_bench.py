"""NL-CAD-PILOT-00 P0-R — routed task-agnostic SCRC scoring: latency and agreement with the exact union oracle.

    python -m exps.nl_cad_pilot_00_p0r.route_bench

Addendum to the completed P0 (agents/reports/nl_cad_pilot_00_p0.md, verdict P0_VALID_BUT_COST_BLOCKED), which was
blocked on inference latency (exact union search over 93,680 rows). This measures, without changing any scorer:
  * a nearest-training-centroid router (15 x 768 FP32, from NL-DRIFT-00 STATS.pt; R15 training normals only),
  * exact certified metric NN restricted to the routed category's rows, and
  * its agreement with the exact certified union NN (the oracle), on normal training probes only.
Custody: importing exps.nl_cad_pilot_00.p0_real installs the P0 audit hook (no test features, test directories,
masks, ground truth, evaluation cells or dataset manifests can be opened). Reuses P0 code unmodified. Writes only to
results/hope_cad/nl_cad_pilot_00_p0r/.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from exps.nl_cad_pilot_00 import p0_real as P          # installs the P0 custody hook first
from exps.nl_cad_pilot_00 import nnx
from exps.nl_cad_pilot_00 import scrc as S
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a_c2dom import common as D

OUT = P.ROOT / "results/hope_cad/nl_cad_pilot_00_p0r"
STATS = P.ROOT / "results/hope_cad/nl_drift_00/STATS.pt"
STATS_LOCK = P.ROOT / "results/hope_cad/nl_drift_00/STATS_LOCK.json"
REPLAY24_E2E_MS = 52.0                                   # P0 report, same GPU (cost context only)


def centroids():
    S.require(C.sha(STATS) == json.loads(STATS_LOCK.read_text())["stats_sha256"], "STATS.pt changed")
    P.LOADS.append(str(STATS))
    st = torch.load(STATS, weights_only=False, map_location="cpu")["centroid"]
    return torch.stack([torch.as_tensor(np.asarray(st[c]), dtype=torch.float64) for c in D.ORDER])


def route(q_img, cent):
    """q_img [784, d] FP32 -> category index (nearest centroid of the image's mean token, L2)."""
    m = q_img.double().mean(0)
    return int(((cent.to(m.device) - m) ** 2).sum(1).argmin())


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    S.require(P.CUDA, "GPU required for the latency comparison with P0")
    t_all = time.time()
    fe = P.Feat()
    bank, sup, brec = P.canonical_bank(14)
    st = P.load_state(15)
    qs = P.probe_queries(fe)                                         # 6 EVAL probes per category, never written
    cent = centroids()
    cats = list(D.ORDER)
    res = {"schema": "nl_cad_p0r_route_v1", "bank_rows": int(len(bank.ids)), "bank_bytes": brec["persistent_bytes"],
           "router_bytes_fp32": int(cent.numel() * 4), "probe_images_per_category": 6, "gammas": {}}
    # routing on probes (category metadata only; probes are R15 identities, so routing is mildly optimistic)
    routes = {c: [route(qs[c][i * 784:(i + 1) * 784], cent) for i in range(6)] for c in cats}
    res["routing_accuracy_probes"] = float(np.mean([r == k for k, c in enumerate(cats) for r in routes[c]]))
    res["routing_errors"] = {c: [cats[r] for r in routes[c] if r != k] for k, c in enumerate(cats) if any(r != k for r in routes[c])}
    for gamma in (0.0, 1.0):
        met = S.make_metric(st, 0.0, gamma)
        g = {}
        U = nnx.Rows.transformed(bank.x, bank.ids, met, device="cuda")
        Rc = {c: nnx.Rows.transformed(sup[c].x, sup[c].ids, met, device="cuda") for c in cats}
        g["rows_per_category"] = {c: int(len(sup[c].ids)) for c in cats}
        excess, equal, own, img_u, img_r, rel_img = [], 0, [], [], [], []
        for k, c in enumerate(cats):
            ids_c = set(sup[c].ids.tolist())
            for i in range(6):
                q = qs[c][i * 784:(i + 1) * 784]
                du, iu = nnx.certified_nn(q, U, met)
                dr, ir = nnx.certified_nn(q, Rc[cats[routes[c][i]]], met)
                du, dr, iu = du.cpu(), dr.cpu(), iu.cpu()
                S.require(bool((dr >= du).all()), "routed distance below the union oracle (impossible)")
                equal += int((dr == du).sum())
                excess.append(((dr - du) / du.clamp_min(1e-300)).numpy())
                own.append(float(np.mean([int(j) in ids_c for j in iu.tolist()])))
                img_u.append(float(du.max()))
                img_r.append(float(dr.max()))
                rel_img.append((img_r[-1] - img_u[-1]) / img_u[-1])
        ex = np.concatenate(excess)
        g["patch_distance_equal_to_oracle_fraction"] = equal / ex.size
        g["patch_relative_excess_quantiles"] = {q: float(np.quantile(ex, q)) for q in (0.5, 0.9, 0.99, 1.0)}
        g["union_nn_in_own_category_fraction_mean"] = float(np.mean(own))
        g["union_nn_in_own_category_by_category"] = {c: float(np.mean(own[k * 6:(k + 1) * 6])) for k, c in enumerate(cats)}
        g["image_score_rel_diff_quantiles"] = {q: float(np.quantile(rel_img, q)) for q in (0.5, 0.9, 1.0)}
        g["image_score_spearman_routed_vs_union"] = spearman(np.array(img_u), np.array(img_r))
        # latency, GPU batch 1 (one probe image per category)
        q1 = {c: qs[c][:784].contiguous() for c in cats}
        lat_u, lat_r = [], []
        for c in cats:
            t, _ = P.timed(lambda: nnx.certified_nn(q1[c], U, met), reps=10, warmup=2, cuda=True)
            lat_u.append(t["median_s"])
            t, _ = P.timed(lambda: nnx.certified_nn(q1[c], Rc[c], met), reps=20, warmup=3, cuda=True)
            lat_r.append(t["median_s"])
        cg = cent.cuda()
        qg = q1[cats[0]].cuda()
        t_route, _ = P.timed(lambda: route(qg, cg), reps=50, warmup=5, cuda=True)
        g["gpu_union_search_ms_median_over_categories"] = 1e3 * float(np.median(lat_u))
        g["gpu_routed_search_ms_median_over_categories"] = 1e3 * float(np.median(lat_r))
        g["gpu_routed_search_ms_max_over_categories"] = 1e3 * float(np.max(lat_r))
        g["gpu_router_ms"] = 1e3 * t_route["median_s"]
        # CPU routed search, one category
        Rcpu = nnx.Rows.transformed(sup[cats[0]].x, sup[cats[0]].ids, met)
        t, _ = P.timed(lambda: nnx.certified_nn(q1[cats[0]], Rcpu, met), reps=5)
        g["cpu_routed_search_ms_bottle"] = 1e3 * t["median_s"]
        g["gpu_peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        res["gammas"][str(gamma)] = g
        del U, Rc
        torch.cuda.empty_cache()
        P.log("gamma", gamma, json.dumps({k: v for k, v in g.items() if not isinstance(v, dict)})[:900])
    res["backbone"] = P.bench_backbone()
    res["readout"] = P.bench_readout()
    trunk = 1e3 * res["backbone"]["gpu_trunk_1img_fp32"]["median_s"]
    rd = 1e3 * res["readout"]["gpu_1024"]["median_s"]
    e2e = {}
    for gk, g in res["gammas"].items():
        e2e[gk] = {"routed_ms": trunk + g["gpu_router_ms"] + g["gpu_routed_search_ms_max_over_categories"] + rd,
                   "union_ms": trunk + g["gpu_union_search_ms_median_over_categories"] + rd}
        e2e[gk]["routed_vs_replay24_ratio"] = e2e[gk]["routed_ms"] / REPLAY24_E2E_MS
    res["end_to_end_gpu_batch1_ms"] = e2e
    res["replay24_end_to_end_ms_reference"] = REPLAY24_E2E_MS
    res["access"] = {"explicit_loads": sorted(set(P.LOADS)), "audited_project_opens": sorted(P.ACCESS)}
    res["env"] = P.env()
    res["wall_s"] = time.time() - t_all
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ROUTE_BENCH.json").open("x") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    P.log("e2e", json.dumps(e2e), "routing", res["routing_accuracy_probes"], "wall", round(res["wall_s"]))


if __name__ == "__main__":
    main()
