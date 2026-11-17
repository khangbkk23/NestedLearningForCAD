"""SCRC-P1-LITE-00 Phase S — label-free scoring of the 14 arms (prereg §6, §11, §12).

    python -m exps.scrc_p1_lite_00.score --owner-authorized-A4      # real run (owner checklist A4 required)

Real mode refuses to start unless ENGINEERING_LOCK.json verifies, the lock/prereg/plan/inputs/old locks verify, the
C2-DOM lock verifies, CUDA is available, and no previous Phase S output exists. Custody phase S: the 15 MVTec
``*_test.pt`` files are opened through ``custody.guarded_features`` (``features`` key only); manifests, masks, labels,
VisA and historical evaluation outputs are refused by the audit hook. Phase L is never started from here (A6 = no).

``run(..., mode="smoke")`` is the engineering smoke (test U10): identical code path on CPU with MVTec *train* eval-probe
images (normal, not bank members) as queries, custody phase PRE, outputs in a scratch directory.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from exps.scrc_p1_lite_00 import custody as CU
from exps.scrc_p1_lite_00 import hybrid as H
from exps.scrc_p1_lite_00 import protocol as P
from exps.scrc_p1_lite_00 import support as SU


class CapExceeded(RuntimeError):
    pass


class ProvenanceFailure(RuntimeError):
    pass


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def write_json_x(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2, default=_default)
        f.write("\n")


def save_x(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        torch.save(obj, f)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def dir_gb(d):
    d = Path(d)
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9 if d.exists() else 0.0


def env_record(device):
    rec = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "threads": torch.get_num_threads(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
           "device": str(device), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if str(device).startswith("cuda"):
        rec.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    return rec


# ------------------------------------------------------------------ query providers

def real_queries(c):
    p = P.ROOT / f"results/hope_cad/calm02/features/mvtec/{c}_test.pt"
    ff = CU.guarded_features(p)
    f = ff.features
    P.require(f.dtype == torch.float32 and tuple(f.shape[1:]) == (P.PATCHES, P.DIM), f"bad test features {c}")
    q = f.reshape(-1, P.DIM).clone()
    return q, {"file": str(p.relative_to(P.ROOT)), "sha256": P.sha_file(p), "n_images": int(f.shape[0])}


def smoke_queries_factory(per=2):
    from exps.nl_sgm_01a import common as C
    from exps.nl_sgm_01a_c2dom import common as D
    co, _ = D.setup()
    F = C.Features()

    def provider(c):
        paths = co[c]["eval"][:per]
        q = F.images(paths).reshape(-1, P.DIM).clone()
        F.drop(c)
        return q, {"file": f"calm02/features/mvtec/{c}_train.pt (eval probes {paths})", "sha256": None,
                   "n_images": per}
    return provider


# ------------------------------------------------------------------ run

class Clock:
    def __init__(self, cap_h, out_dir, device):
        self.t0 = time.time()
        self.cap_s = cap_h * 3600
        self.out_dir, self.device = out_dir, str(device)

    def elapsed(self):
        return time.time() - self.t0

    def check(self, where):
        rec = {"where": where, "elapsed_s": self.elapsed(), "peak_rss_gb": peak_rss_gb(),
               "disk_gb": dir_gb(self.out_dir)}
        if self.device.startswith("cuda"):
            rec["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if rec["elapsed_s"] > self.cap_s:
            raise CapExceeded(f"Phase S wall {rec['elapsed_s']:.0f} s > cap {self.cap_s:.0f} s at {where}")
        if rec["peak_rss_gb"] > P.CAPS["rss_gb"]:
            raise CapExceeded(f"peak RSS {rec['peak_rss_gb']:.2f} GB > {P.CAPS['rss_gb']} at {where}")
        if rec["disk_gb"] > P.CAPS["disk_gb"]:
            raise CapExceeded(f"disk {rec['disk_gb']:.3f} GB > {P.CAPS['disk_gb']} at {where}")
        if rec.get("vram_peak_gb", 0.0) > P.CAPS["vram_gb"]:
            raise CapExceeded(f"VRAM {rec['vram_peak_gb']:.2f} GB > {P.CAPS['vram_gb']} at {where}")
        return rec


def _quant(x, qs=(0.01, 0.5, 0.99)):
    x = np.asarray(x, dtype=np.float64)
    return {f"p{int(round(100 * q))}": float(np.quantile(x, q)) for q in qs} | {"min": float(x.min()),
                                                                                 "max": float(x.max())}


def run(mode, out_dir, device, provider, n_probe_patches=P.N_PROBE_PATCHES, pv5_est_s_per_query=None,
        n_total_images=P.N_TEST_IMAGES, log=print):
    """Phase S. Returns the SCORES_LOCK dict. Raises CapExceeded / ProvenanceFailure after writing a stop record."""
    out_dir = Path(out_dir)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(P.CAPS["threads"])
    clock = Clock(P.CAPS["phase_S_gpu_h"], out_dir, device)
    if str(device).startswith("cuda"):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        clock.t0 = time.time()                           # wall clock from first CUDA initialisation
    P.require(not (out_dir / "SCORES_LOCK.json").exists() and not (out_dir / "scores").exists(),
              "Phase S output already exists (create-exclusive)")
    rec = {"schema": "scrc_p1_lite_00_scores_lock_v1", "mode": mode, "env": env_record(device), "pv": {},
           "categories": {}, "arms": list(P.ARMS), "timing": {}, "cap_checks": []}
    try:
        # ---------------------------------------------------------- PV1 inputs
        pv1 = {"documents": P.verify_documents(), "constants": P.check_constants(),
               "c2dom_lock": P.verify_c2dom_code_sources(),            # PREREG_AMENDMENT_01 (E1)
               "prereg_amendment_01_sha256": P.sha_file(P.PREREG_AMENDMENT_01)}
        if mode == "real":
            el = P.verify_engineering_lock()
            pv1["engineering_lock_sha256"] = P.sha_file(P.ENGINEERING_LOCK)
            pv5_est_s_per_query = el["measured"]["pv5_brute_metric_gpu_s_per_query"]
        rec["pv"]["PV1"] = {"pass": True, **pv1}
        # ---------------------------------------------------------- supports, PV2, state, banks
        ta = time.time()
        ck, brec = SU.load_ckpt(P.BOUNDARY)
        S = SU.full_support(ck)
        Sp = SU.byte_matched_support(ck)
        del ck
        pv2 = SU.checks(S, Sp)
        rec["pv"]["PV2"] = pv2
        if not pv2["pass"]:
            raise ProvenanceFailure(f"PV2 failed: {pv2}")
        st = SU.load_state()
        mets = SU.metrics(st)
        rec["metrics"] = {k: {"gamma": m.gamma, "tau": m.tau, "mu_t": m.mu_t, "kappa_t": m.kappa_t,
                              "kappa_theta": m.kappa_theta, "c_min": float(m.c.min()), "c_max": float(m.c.max())}
                          for k, m in mets.items() if k in P.CONFIGS}
        banks = {"R0": H.ScreenRows(S.x, S.ids), "C0": H.ScreenRows(Sp.x, Sp.ids),
                 P.PV3_ARM: H.ScreenRows(Sp.x, Sp.ids, mets[P.PV3_ARM])}
        for k in P.METRIC_ARMS:
            banks[k] = H.ScreenRows(Sp.x, Sp.ids, mets[k])
        P.require(torch.equal(banks[P.PV3_ARM].cpu.z, Sp.x), "gamma = 0 transformed rows differ from raw rows")
        rec["timing"]["prep_s"] = time.time() - ta
        log("prep", round(rec["timing"]["prep_s"], 1), "s")
        # ---------------------------------------------------------- scoring
        order = P.ARMS + (P.PV3_ARM,)
        probes = {}
        pv3 = {}
        diag = {k: {} for k in P.ARMS}
        arm_seconds = {k: 0.0 for k in order}
        score_s_total = 0.0
        n_seen = 0
        for ci, c in enumerate(P.ORDER):
            tc = time.time()
            q, frec = provider(c)
            n_c = len(q) // P.PATCHES
            P.require(len(q) == n_c * P.PATCHES and n_c > 0, f"query count {c}")
            crec = {"feature": frec, "n_images": n_c, "score_sha256": {}, "id_sha256": {}}
            res = {}
            for arm in order:
                tb = time.time()
                sr = banks[arm].to(device)
                stt = {}
                d, i = H.hybrid_certified_nn(q, sr, stats=stt, per_query_counts=True)
                sr.release()
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                P.require(bool(torch.isfinite(d).all()) and bool((i >= 0).all()), f"non-finite score {arm} {c}")
                dt = time.time() - tb
                arm_seconds[arm] += dt
                cnt = stt.pop("per_query_counts").numpy()
                diag_arm = {"seconds": dt, "ms_per_image": 1e3 * dt / n_c,
                            "s_per_query_row": dt / (len(q) * len(sr.cpu.ids)),
                            "seconds_screen": stt["seconds_screen"], "seconds_final_cpu": stt["seconds_final_cpu"],
                            "seconds_prep": stt["seconds_prep"],
                            "candidates_mean": float(cnt.mean()), "candidates_p99": float(np.quantile(cnt, 0.99)),
                            "candidates_max": int(cnt.max())}
                if arm == P.PV3_ARM:
                    pv3[c] = {"dist_bitwise": bool(torch.equal(d, res["C0"][0])),
                              "ids_equal": bool(torch.equal(i, res["C0"][1])), **diag_arm}
                else:
                    res[arm] = (d, i)
                    dd = d.reshape(n_c, P.PATCHES).contiguous()
                    ii = i.reshape(n_c, P.PATCHES).contiguous()
                    save_x(out_dir / "scores" / arm / f"{c}.pt", {"d": dd, "nn_id": ii})
                    crec["score_sha256"][arm] = P.sha_file(out_dir / "scores" / arm / f"{c}.pt")
                    crec["id_sha256"][arm] = P.sha_tensor(ii)
                    diag[arm][c] = diag_arm
                rec["cap_checks"].append(clock.check(f"{c}/{arm}"))
            # label-free diagnostics vs C0 (same support identities)
            dC, iC = res["C0"]
            for arm in P.METRIC_ARMS:
                dM, iM = res[arm]
                ratio = (dM / dC).numpy()
                m = mets[arm]
                lo = math.sqrt(m.mu_t) * (1 - 1e-12)
                diag[arm][c].update(nn_change_fraction_vs_C0=float((iM != iC).float().mean()),
                                    dist_ratio_vs_C0=_quant(ratio),
                                    sandwich_violations=int(((ratio < lo) | (ratio > 1 + 1e-12)).sum()))
            diag["C0"][c]["nn_change_fraction_vs_R0"] = float((iC != res["R0"][1]).float().mean())
            # probes (label-free index rules)
            j = P.probe_image(c, n_c)
            pp = P.probe_patches(c, n_probe_patches)
            rows_img = slice(j * P.PATCHES, (j + 1) * P.PATCHES)
            probes[c] = {"image": j, "patches": pp, "q_image": q[rows_img].clone(),
                         "R0": (res["R0"][0][rows_img].clone(), res["R0"][1][rows_img].clone()),
                         "q_patch": q[rows_img][pp].clone(),
                         "hyb": {a: (res[a][0][rows_img][pp].clone(), res[a][1][rows_img][pp].clone())
                                 for a in ("C0",) + P.PV5_ARMS}}
            # PV6 determinism on the first category
            if ci == 0:
                td = time.time()
                d6, i6 = H.hybrid_certified_nn(q, banks["R0"].to(device))
                banks["R0"].release()
                rec["pv"]["PV6"] = {"category": c, "pass": bool(P.sha_tensor(d6) == P.sha_tensor(res["R0"][0])
                                                                and P.sha_tensor(i6) == P.sha_tensor(res["R0"][1])),
                                    "seconds": time.time() - td}
                if not rec["pv"]["PV6"]["pass"]:
                    raise ProvenanceFailure("PV6 determinism failed")
            n_seen += n_c
            crec["seconds"] = time.time() - tc
            score_s_total += crec["seconds"]
            rec["categories"][c] = crec
            log("scored", c, n_c, "images", round(crec["seconds"], 1), "s", "elapsed", round(clock.elapsed(), 1))
            if not all(v["dist_bitwise"] and v["ids_equal"] for v in pv3.values()):
                raise ProvenanceFailure(f"PV3 failed at {c}")
            if ci == 0:                                   # prereg §12 projection after bottle, all arms
                tail = (pv5_est_s_per_query or 0.0) * len(P.PV5_ARMS) * n_probe_patches * len(P.ORDER) + 120.0
                proj = rec["timing"]["prep_s"] + crec["seconds"] * n_total_images / n_c + tail
                rec["timing"]["projection_after_first_category"] = {
                    "first_category_s": crec["seconds"], "n_first": n_c, "n_total_images": n_total_images,
                    "pv5_and_tail_estimate_s": tail, "projected_total_s": proj, "cap_s": clock.cap_s}
                log("projection", round(proj, 0), "s of cap", clock.cap_s)
                if proj > clock.cap_s:
                    raise CapExceeded(f"projected Phase S {proj:.0f} s > cap {clock.cap_s:.0f} s")
            del q, res
        rec["pv"]["PV3"] = {"pass": all(v["dist_bitwise"] and v["ids_equal"] for v in pv3.values()),
                            "per_category": pv3}
        # ---------------------------------------------------------- PV4 canonical equivalence (CPU exact.nn)
        from exps.nl_sgm_01a import exact as X
        from exps.nl_sgm_01a_c2dom import stream as CS
        td = time.time()
        sup = CS.supports_at(P.ARM_SOURCE, P.BOUNDARY)
        pv4 = {}
        for c in P.ORDER:
            qi = probes[c]["q_image"]
            dm = im = None
            for k, rs in sup.items():
                dk, ik = X.nn(qi, rs)
                dm, im = (dk, ik) if dm is None else X.merge(dm, im, dk, ik)
            d0, i0 = probes[c]["R0"]
            pv4[c] = {"dist_bitwise": bool(torch.equal(dm, d0)), "ids_equal": bool(torch.equal(im, i0))}
        del sup
        rec["pv"]["PV4"] = {"pass": all(v["dist_bitwise"] and v["ids_equal"] for v in pv4.values()),
                            "per_category": pv4, "seconds": time.time() - td}
        rec["cap_checks"].append(clock.check("PV4"))
        if not rec["pv"]["PV4"]["pass"]:
            raise ProvenanceFailure("PV4 canonical equivalence failed")
        # ---------------------------------------------------------- PV5 metric exactness vs nnx.brute_metric
        from exps.nl_cad_pilot_00 import nnx
        td = time.time()
        qp = torch.cat([probes[c]["q_patch"] for c in P.ORDER])
        xs, ids = Sp.x.to(device), Sp.ids.to(device)
        pv5 = {}
        for a in P.PV5_ARMS:
            ta = time.time()
            bd, bi = nnx.brute_metric(qp.to(device), xs, ids, mets[a])
            hd = torch.cat([probes[c]["hyb"][a][0] for c in P.ORDER])
            hi = torch.cat([probes[c]["hyb"][a][1] for c in P.ORDER])
            rel = float(((bd.cpu() - hd).abs() / hd).max())
            pv5[a] = {"queries": int(len(qp)), "id_mismatches": int((bi.cpu() != hi).sum()), "max_rel_diff": rel,
                      "seconds": time.time() - ta, "s_per_query": (time.time() - ta) / len(qp)}
            rec["cap_checks"].append(clock.check(f"PV5/{a}"))
        del xs, ids
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        rec["pv"]["PV5"] = {"pass": all(v["id_mismatches"] == 0 and v["max_rel_diff"] <= P.PV5_REL_TOL
                                        for v in pv5.values()), "per_arm": pv5, "seconds": time.time() - td}
        if not rec["pv"]["PV5"]["pass"]:
            raise ProvenanceFailure(f"PV5 metric exactness failed: {pv5}")
        # ---------------------------------------------------------- P5 fallback diagnostic (no gating role)
        td = time.time()
        d1 = torch.cat([probes[c]["hyb"]["C0"][0] for c in P.ORDER])
        d64 = X.kth(qp, Sp, P.FALLBACK_K)
        r64 = (d64 / d1).numpy()
        fb = {"queries": int(len(qp)), "K": P.FALLBACK_K, "ratio_64nn_to_1nn": _quant(r64, (0.5, 0.9)), "per_arm": {}}
        for a in P.METRIC_ARMS:
            cert = d64 > d1 / math.sqrt(mets[a].mu_t) * (1 + 1e-9)
            fb["per_arm"][a] = {"fallback_rate": float((~cert).float().mean()), "mu_t": mets[a].mu_t}
        fb["seconds"] = time.time() - td
        rec["diagnostics"] = {"per_arm_category": diag, "p5_fallback_diagnostic": fb,
                              "arm_seconds_total": arm_seconds,
                              "max_score_fp32": {a: max(float(torch.load(out_dir / "scores" / a / f"{c}.pt",
                                                                         weights_only=True)["d"].float().max())
                                                        for c in P.ORDER) for a in P.ARMS}}
        rec["probes"] = {c: {"image": probes[c]["image"], "patches_sha": P.sha_text(str(probes[c]["patches"]))}
                         for c in P.ORDER}
        rec["n_images_total"] = n_seen
        rec["timing"]["scoring_s"] = score_s_total
        rec["timing"]["phase_S_wall_s"] = clock.elapsed()
        rec["cap_checks"].append(clock.check("end"))
        rec["peak"] = {"rss_gb": peak_rss_gb(), "disk_gb": dir_gb(out_dir)}
        if str(device).startswith("cuda"):
            rec["peak"]["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec["pv_all_pass"] = all(rec["pv"][k]["pass"] for k in ("PV1", "PV2", "PV3", "PV4", "PV5", "PV6"))
        rec["access_log"] = CU.access_log()
        rec["labels_opened"] = False
        P.require(rec["pv_all_pass"], "provenance")
        write_json_x(out_dir / "SCORES_LOCK.json", rec)
        log("SCORES_LOCK written", out_dir / "SCORES_LOCK.json", "wall", round(clock.elapsed(), 1), "s")
        return rec
    except (CapExceeded, ProvenanceFailure) as e:
        kind = "SCREEN_STOPPED_COST" if isinstance(e, CapExceeded) else "SCREEN_INVALID"
        rec.update(stopped=kind, reason=str(e), trace=traceback.format_exc(), access_log=CU.access_log(),
                   labels_opened=False, wall_s=clock.elapsed())
        write_json_x(out_dir / f"PHASE_S_{kind}.json", rec)
        raise


def verify_scores_lock(out_dir):
    """Every score file hash recorded in SCORES_LOCK.json must still match (Phase L precondition)."""
    out_dir = Path(out_dir)
    sl = json.loads((out_dir / "SCORES_LOCK.json").read_text())
    P.require(sl.get("pv_all_pass") is True and sl.get("labels_opened") is False, "SCORES_LOCK not clean")
    bad = [f"{a}/{c}" for c, cr in sl["categories"].items() for a, h in cr["score_sha256"].items()
           if P.sha_file(out_dir / "scores" / a / f"{c}.pt") != h]
    P.require(not bad, f"score files changed after SCORES_LOCK: {bad}")
    P.require(set(sl["categories"]) == set(P.ORDER) and all(set(cr["score_sha256"]) == set(P.ARMS)
                                                             for cr in sl["categories"].values()), "incomplete")
    return sl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner-authorized-A4", action="store_true", dest="a4")
    args = ap.parse_args()
    P.require(args.a4, "Phase S requires owner checklist item A4 (--owner-authorized-A4)")
    P.require(torch.cuda.is_available(), "CUDA required for Phase S")
    CU.install("S")
    P.verify_engineering_lock()
    (P.OUT / "logs").mkdir(parents=True, exist_ok=True)
    lf = (P.OUT / "logs" / "phase_S.log").open("a")

    def log(*a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        lf.write(s + "\n"); lf.flush()
    run("real", P.OUT, "cuda", real_queries, log=log)


if __name__ == "__main__":
    main()
