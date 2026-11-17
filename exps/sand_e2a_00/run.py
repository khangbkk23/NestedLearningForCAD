"""SAND-E2A-00 runner (CPU only; CUDA is never initialised).

    python -m exps.sand_e2a_00.run calib      # phase C: label-free witness calibrators + diagnostics -> CALIB.json
    python -m exps.sand_e2a_00.run evaluate   # phase L: parity, gate, conditional forgetting pass -> RESULT.json

Phase C reads only decoder grids, support distances, acquisition calibrators and stage / boundary records of
SAND-FINAL (no labels, no masks, no published metrics). Phase L verifies LOCK and CALIB_LOCK before the label
allow-list opens. Prereg: agents/plans/sand_e2a_00_prereg.md.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from exps.sand_e2a_00 import common as E

W = [f"E2A1_w{j}" for j in range(E.N_SEL)]
WT = [f"E2A1T_w{j}" for j in range(E.N_SEL)]


def _torch():
    import torch
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


def _inputs(torch, t, c, bnd):
    G = torch.load(E.GRIDS / f"stage{t + 1:02d}_{c}.pt", map_location="cpu", weights_only=False)
    with np.load(E.dist_path(c, t, bnd)) as z:
        Z = {k: z[k] for k in z.files}
    return G, Z


def _caps(t_start, extra_s=0.0):
    wall = time.time() - t_start + extra_s
    E.require(wall <= E.CAPS["wall_h_total"] * 3600, f"CAP: wall {wall:.0f}s")
    E.require(E.peak_rss_gb() <= E.CAPS["rss_gb"], f"CAP: RSS {E.peak_rss_gb():.2f}GB")
    return wall


class Log:
    def __init__(self, name):
        (E.OUT / "logs").mkdir(parents=True, exist_ok=True)
        self.f = (E.OUT / "logs" / f"{name}.log").open("a")

    def __call__(self, *a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


# ------------------------------------------------------------------ phase C (label-free)

def calib():
    t_start = time.time()
    E.install_custody("C")
    lk = E.verify_lock()
    E.require(not E.CALIB.exists(), "CALIB.json exists (create-exclusive)")
    torch = _torch()
    from exps.sand import common as S                       # canonical calibrate / zscores (hash-locked)
    E.require(tuple(S.ORDER) == E.ORDER, "category order differs from SAND")
    log = Log("calib")
    n_in = E.verify_inputs(lk)
    log("PV1 lock + inputs ok", n_in)
    bnd = [E.read_json(E.MEM / f"boundary_t{t:02d}.json") for t in range(E.NT)]
    stage_sha = {t: E.read_json(E.RUN / f"STAGE{t + 1:02d}.json")["checkpoint_sha256"] for t in range(E.NT)}
    cal, diag = {}, {}
    checks = {"pv2_grid_checkpoint": 0, "pv4_acquisition_calibrators_exact": 0, "pv5_e2a6_vs_d1_rel_max": 0.0,
              "e2a_equals_frozen_at_acquisition": 0, "witness_identities": {}}
    for k, c in enumerate(E.ORDER):
        kal = E.read_json(E.CALDIR / f"{c}.json")
        E.require(kal["category"] == c and kal["stage"] == k + 1, f"acquisition calibrators {c}")
        E.require(len(kal["probe_paths"]) == E.N_SEL and kal["probe_paths"] == sorted(kal["probe_paths"]),
                  f"SEL probe list {c}")
        checks["witness_identities"][c] = kal["probe_paths"]
        cal_a = {x: float(kal[x]) for x in E.CAL_KEYS}
        st, d1, ev = {}, {}, {}
        for t in range(k, E.NT):
            G, Z = _inputs(torch, t, c, bnd[t])
            E.require(G["checkpoint_sha256"] == stage_sha[t], f"PV2: grid checkpoint stage {t + 1} {c}")
            checks["pv2_grid_checkpoint"] += 1
            a_sel, a_ev = (G[x].numpy().reshape(-1, E.TOKENS) for x in ("sel", "eval"))
            d_sel, d_ev = (Z[x].reshape(-1, E.TOKENS) for x in ("sel", "eval"))
            E.require(a_sel.shape == d_sel.shape == a_ev.shape == d_ev.shape == (E.N_SEL, E.TOKENS),
                      f"probe shapes {t + 1} {c}")
            d1[t] = {x: float(v) for x, v in S.calibrate(a_sel, d_sel).items() if x in E.CAL_KEYS}
            if t == k:
                E.require(kal["gids_sha"] == bnd[t]["per_category_gids_sha"][c], f"acquisition gids {c}")
                E.require(kal["checkpoint_sha256"] == stage_sha[t], f"acquisition checkpoint {c}")
                E.require(all(d1[t][x] == cal_a[x] for x in E.CAL_KEYS), f"PV4: acquisition calibrators {c}")
                checks["pv4_acquisition_calibrators_exact"] += 1
            st[t] = [E.channel_stats(a_sel[j], d_sel[j]) for j in range(E.N_SEL)]
            ev[t] = (a_ev.copy(), d_ev.copy())
            del G, Z
        pooled = {t: {ch: (d1[t][f"mu_{ch}"], d1[t][f"sd_{ch}"]) for ch in E.CH} for t in d1}
        for t in range(k, E.NT):
            arms = {"frozen": cal_a, "D1": d1[t], "E2A6": E.e2a(cal_a, pooled[k], pooled[t])}
            for j in range(E.N_SEL):
                arms[W[j]] = E.e2a(cal_a, st[k][j], st[t][j])
                arms[WT[j]] = E.e2a(cal_a, st[k][j], st[t][j], channels=("theta",))
            r = max(abs(arms["E2A6"][x] - d1[t][x]) / abs(d1[t][x]) for x in E.CAL_KEYS)
            checks["pv5_e2a6_vs_d1_rel_max"] = max(checks["pv5_e2a6_vs_d1_rel_max"], r)
            E.require(r <= E.TOL["d1_calib_rel"], f"PV5: E2A with all six probes != D1 ({t + 1}, {c}): {r}")
            if t == k:
                E.require(all(arms[n] == cal_a for n in arms if n.startswith("E2A")), f"E2A != frozen at acq {c}")
                checks["e2a_equals_frozen_at_acquisition"] += 1
            cal[f"{t}|{c}"] = arms
            if t > k:                                         # label-free diagnostics (non-gating)
                a_ev, d_ev = ev[t]
                z = {}
                for n in ("frozen", "D1", *W):
                    zt, zm = S.zscores(a_ev, d_ev, arms[n])
                    z[n] = {"theta_mean": float(zt.mean()), "theta_std": float(zt.std()), "m_mean": float(zm.mean()),
                            "m_std": float(zm.std()), "fused_mean": float(np.minimum(zt, zm).mean())}
                diag[f"{t}|{c}"] = {"tracking_vs_D1": {n: E.tracking_error(arms[n], d1[t]) for n in ("frozen", *W)},
                                    "eval_probe_z": z}
        wall = _caps(t_start)
        log(c, "boundaries", E.NT - k, f"wall {wall:.0f}s", f"rss {E.peak_rss_gb():.2f}GB")
    summ = {}
    for n in ("frozen", "E2A1"):
        names = ["frozen"] if n == "frozen" else W
        for ch in E.CH:
            em = [abs(d["tracking_vs_D1"][m][ch][0]) for d in diag.values() for m in names]
            el = [abs(d["tracking_vs_D1"][m][ch][1]) for d in diag.values() for m in names]
            summ[f"{n}_{ch}"] = {"mean_abs_mu_err_sd": float(np.mean(em)), "median_abs_mu_err_sd": float(np.median(em)),
                                 "p95_abs_mu_err_sd": float(np.quantile(em, 0.95)),
                                 "mean_abs_log_sd_err": float(np.mean(el)), "n": len(em)}
    fin = {c: diag[f"{E.NT - 1}|{c}"]["eval_probe_z"] for c in E.ORDER[:-1]}
    summ["final_eval_probe_theta_z_mean"] = {
        "frozen": float(np.mean([fin[c]["frozen"]["theta_mean"] for c in fin])),
        "E2A1": float(np.mean([np.mean([fin[c][w]["theta_mean"] for w in W]) for c in fin])),
        "D1": float(np.mean([fin[c]["D1"]["theta_mean"] for c in fin]))}
    wall = _caps(t_start)
    out = {"schema": "sand_e2a_00_calib_v1", "lock_sha256": E.sha_file(E.LOCK), "cal": cal, "diag": diag,
           "diag_summary": summ, "checks": checks, "wall_s": wall, "rss_peak_gb": E.peak_rss_gb(),
           "cuda_initialized": bool(torch.cuda.is_initialized()), "access_log": E.access_log(), "utc": E.now()}
    E.require(not out["cuda_initialized"], "CUDA initialised")
    h = E.write_json_x(E.CALIB, out)
    hl = E.write_json_x(E.CALIB_LOCK, {"calib_sha256": h, "lock_sha256": E.sha_file(E.LOCK), "wall_s": wall,
                                        "n_access_log": len(out["access_log"]), "utc": E.now()})
    log("CALIB DONE", f"wall {wall:.0f}s", "CALIB", h[:16], "CALIB_LOCK", hl[:16])
    log("tracking summary", {k: (round(v["mean_abs_mu_err_sd"], 3), round(v["mean_abs_log_sd_err"], 3))
                             for k, v in summ.items() if "mean_abs_mu_err_sd" in v})


# ------------------------------------------------------------------ phase L (labels)

def _cell(S, torch, t, c, arms, labels, masks, items, hw, bnd, lock_sha):
    """Fused metrics of the named calibrator sets at one (t, c) cell; create-exclusive, resumable (deterministic)."""
    p = E.CELL_OUT / f"stage{t + 1:02d}_{c}.json"
    if p.exists():
        r = E.read_json(p)
        E.require(r["lock_sha256"] == lock_sha and set(r["arms"]) == set(arms), f"stale cell {p}")
        return r, 0
    t0 = time.time()
    G, Z = _inputs(torch, t, c, bnd)
    E.require(G["paths"] == [x["relative_path"] for x in items], f"test identity order {t + 1} {c}")
    a, d = G["grid"].numpy().reshape(-1, E.TOKENS), Z["test"]
    E.require(a.shape == d.shape == (len(labels), E.TOKENS), f"test map shapes {t + 1} {c}")
    out, memo = {}, {}
    for name, cl in arms.items():
        key = tuple(cl[x] for x in E.CAL_KEYS)
        if key not in memo:
            g, frac = S.fuse(a, d, cl)
            maps, img = S.readout_native(g, hw)
            memo[key] = {**S.metrics(maps, img, labels, masks), "theta_is_min_fraction": frac}
            del maps, g
        out[name] = memo[key]
    r = {"stage": t + 1, "category": c, "arms": out, "n_metric_computations": len(memo),
         "seconds": time.time() - t0, "peak_rss_gb": E.peak_rss_gb(), "lock_sha256": lock_sha, "utc": E.now()}
    E.write_json_x(p, r)
    return r, len(memo)


def evaluate():
    t_start = time.time()
    E.install_custody("L0")
    lk = E.verify_lock()
    lock_sha = E.sha_file(E.LOCK)
    cl = E.read_json(E.CALIB_LOCK)
    E.require(cl["lock_sha256"] == lock_sha and E.sha_file(E.CALIB) == cl["calib_sha256"], "CALIB_LOCK mismatch")
    E.mark("calib_lock_verified")
    E.set_phase("L")                                          # label allow-list opens only here
    E.require(not E.RESULT.exists(), "RESULT.json exists (one-shot)")
    torch = _torch()
    from exps.nl_sgm_01a import am03_eval as A3
    from exps.sand import common as S
    log = Log("evaluate")
    S.verify_lock()                                           # SAND-FINAL sources / protocol unchanged
    E.verify_inputs(lk)
    for n, h in E.SAND_PUBLISHED.items():
        E.require(E.sha_file(E.SAND / n) == h, f"SAND published artifact changed: {n}")
    C = E.read_json(E.CALIB)
    cal, phase_c_wall = C["cal"], C["wall_s"]
    sand = E.read_json(E.SAND / "PHASE0_RESULTS.json")
    pub = {"A1": sand["summary"]["A1"], "A2": sand["summary"]["A2"], "A3": sand["summary"]["A3"],
           "D1": sand["variants_non_gating"]["A3_D1_refreshed_calibrators"],
           "TH_P": sand["variants_non_gating"]["TH_theta_channel_under_P"]}
    for a, (ap, au) in E.PUBLISHED.items():
        E.require(abs(pub[a]["final_pixel_ap"] - ap) <= E.TOL["published"]
                  and abs(pub[a]["final_image_auroc"] - au) <= E.TOL["published"], f"published value changed {a}")
    man = E.read_json(E.MANIFEST)
    bnd = [E.read_json(E.MEM / f"boundary_t{t:02d}.json") for t in range(E.NT)]
    a1_final = E.read_json(E.A1_OUT / f"EVAL_STAGE{E.NT:02d}.json")["per_category"]
    sand_final = {c: E.read_json(E.CELLS / f"stage{E.NT:02d}_{c}.json") for c in E.ORDER}
    sand_acq = {c: E.read_json(E.CELLS / f"stage{k + 1:02d}_{c}.json") for k, c in enumerate(E.ORDER)}
    final_arms = ["frozen", "E2A6", *W, *WT]
    n_metric, parity = 0, {"a3_pixel_ap": 0.0, "a3_image_auroc": 0.0, "d1_pixel_ap": 0.0, "d1_image_auroc": 0.0,
                           "frozen_calib_exact": 0, "d1_calib_rel_max": 0.0}
    fin = {}
    log("phase L start; pass 1 (final cells)")
    for k, c in enumerate(E.ORDER):
        labels, masks, items = A3.load_masks(c)
        hw = tuple(man["categories"][c]["images"][0]["hw"])
        E.require(all(tuple(x["hw"]) == hw for x in items), f"mixed native sizes {c}")
        t = E.NT - 1
        arms = {n: cal[f"{t}|{c}"][n] for n in final_arms}
        sc = sand_final[c]
        E.require(all(arms["frozen"][x] == sc["calibration"]["frozen"][x] for x in E.CAL_KEYS), f"frozen cal {c}")
        parity["frozen_calib_exact"] += 1
        rd = max(abs(cal[f"{t}|{c}"]["D1"][x] - sc["calibration"]["refreshed"][x]) / abs(sc["calibration"]["refreshed"][x])
                 for x in E.CAL_KEYS)
        parity["d1_calib_rel_max"] = max(parity["d1_calib_rel_max"], rd)
        E.require(rd <= E.TOL["d1_calib_rel"], f"D1 calibrators differ from SAND cell {c}: {rd}")
        r, m = _cell(S, torch, t, c, arms, labels, masks, items, hw, bnd[t], lock_sha)
        n_metric += m
        for key, arm, ref, tol in (("a3", "frozen", sc["fusion_P"], E.TOL["a3_parity"]),
                                   ("d1", "E2A6", sc["fusion_D1"], E.TOL["d1_parity"])):
            for mt in ("pixel_ap", "image_auroc"):
                dv = abs(r["arms"][arm][mt] - ref[mt])
                parity[f"{key}_{mt}"] = max(parity[f"{key}_{mt}"], dv)
                E.require(dv <= tol, f"PARITY FAIL {key} {mt} {c}: {dv}")
        fin[c] = r["arms"]
        del masks
        wall = _caps(t_start, phase_c_wall)
        e2 = np.mean([r["arms"][w]["pixel_ap"] for w in W])
        log(c, "final", "A3", round(r["arms"]["frozen"]["pixel_ap"], 4), "D1", round(r["arms"]["E2A6"]["pixel_ap"], 4),
            "E2A1", round(float(e2), 4), f"{r['seconds']:.0f}s", f"wall {wall:.0f}s", f"rss {E.peak_rss_gb():.2f}GB")
    t_pass1 = time.time() - t_start
    # primary arm and single detectors (per category, final boundary)
    e2a_ap = [float(np.mean([fin[c][w]["pixel_ap"] for w in W])) for c in E.ORDER]
    e2a_au = [float(np.mean([fin[c][w]["image_auroc"] for w in W])) for c in E.ORDER]
    singles_ap = {"A1": [a1_final[c]["pixel_ap"] for c in E.ORDER],
                  "A2": [sand_final[c]["support_canonical"]["pixel_ap"] for c in E.ORDER],
                  "TH_P": [sand_final[c]["theta_P"]["pixel_ap"] for c in E.ORDER]}
    singles_au = {"A1": [a1_final[c]["image_auroc"] for c in E.ORDER],
                  "A2": [sand_final[c]["support_canonical"]["image_auroc"] for c in E.ORDER],
                  "TH_P": [sand_final[c]["theta_P"]["image_auroc"] for c in E.ORDER]}
    for a in E.SINGLES:
        E.require(abs(np.mean(singles_ap[a]) - E.PUBLISHED[a][0]) <= E.TOL["published"]
                  and abs(np.mean(singles_au[a]) - E.PUBLISHED[a][1]) <= E.TOL["published"], f"single macro {a}")
    boot = lambda x, y: A3.bootstrap_delta(x, y, seed=E.GATE["boot_seed"], B=E.GATE["boot_B"])  # noqa: E731
    g = E.gate(e2a_ap, singles_ap, float(np.mean(e2a_au)), {a: float(np.mean(v)) for a, v in singles_au.items()}, boot)
    g.update({"parity": parity, "calib_sha256": cl["calib_sha256"], "lock_sha256": lock_sha})
    if E.GATE_FILE.exists():                                   # resumed run: the gate is deterministic, never rewritten
        old = E.read_json(E.GATE_FILE)
        E.require(old["verdict"] == g["verdict"] and old["final_pixel_ap_gain"] == g["final_pixel_ap_gain"]
                  and old["bootstrap"] == g["bootstrap"], "resumed gate differs from GATE.json")
        gh = E.sha_file(E.GATE_FILE)
    else:
        gh = E.write_json_x(E.GATE_FILE, {**g, "utc": E.now()})
    log("GATE", g["verdict"], "gain", round(g["final_pixel_ap_gain"], 4), "CI", round(g["bootstrap"]["lo"], 4),
        round(g["bootstrap"]["hi"], 4), "AUROC", round(g["final_image_auroc"], 4), "GATE.json", gh[:16])
    # pass 2: intermediate boundaries for the primary arm, only if the gate passed (prereg section 4)
    series = None
    if g["pass"]:
        n_imgs = {c: sand_final[c]["n_images"] for c in E.ORDER}
        done_units = sum(n_imgs[c] * len({tuple(cal[f"{E.NT - 1}|{c}"][n][x] for x in E.CAL_KEYS) for n in final_arms})
                         for c in E.ORDER)
        todo_units = sum(n_imgs[c] * E.N_SEL * max(0, E.NT - 2 - k) for k, c in enumerate(E.ORDER))
        proj = phase_c_wall + t_pass1 + t_pass1 / max(done_units, 1) * todo_units
        log("pass 2 projection", f"{proj:.0f}s", "cap", E.CAPS["wall_h_total"] * 3600)
        E.require(proj <= E.CAPS["wall_h_total"] * 3600, f"CAP: projected wall {proj:.0f}s")
        inter = {}
        for k, c in enumerate(E.ORDER):
            if k >= E.NT - 2:
                continue
            labels, masks, items = A3.load_masks(c)
            hw = tuple(man["categories"][c]["images"][0]["hw"])
            for t in range(k + 1, E.NT - 1):
                r, m = _cell(S, torch, t, c, {w: cal[f"{t}|{c}"][w] for w in W}, labels, masks, items, hw, bnd[t],
                             lock_sha)
                n_metric += m
                inter[(t, c)] = r["arms"]
                wall = _caps(t_start, phase_c_wall)
            del masks
            log(c, "pass 2 done", f"wall {wall:.0f}s", f"rss {E.peak_rss_gb():.2f}GB")
        series = build_series(fin, sand_acq, inter)
    wall = _caps(t_start, phase_c_wall)
    res = analyse(E, g, fin, sand, sand_final, sand_acq, a1_final, singles_ap, singles_au, e2a_ap, e2a_au, series, boot)
    res.update({"schema": "sand_e2a_00_result_v1", "lock_sha256": lock_sha, "calib_sha256": cl["calib_sha256"],
                "gate_sha256": gh, "parity": parity, "calib_checks": C["checks"], "calib_diag_summary": C["diag_summary"],
                "resources": {"phase_c_wall_s": phase_c_wall, "phase_l_wall_s": time.time() - t_start,
                              "pass1_wall_s": t_pass1, "total_wall_s": wall, "n_metric_computations": n_metric,
                              "rss_peak_gb": E.peak_rss_gb(), "cuda_initialized": bool(torch.cuda.is_initialized())},
                "custody": custody_summary(), "exposure": "MVTec AD official test, historically exposed; "
                "owner-approved exploratory development cohort (task_sand_e2a_00.md)", "utc": E.now()})
    E.require(not res["resources"]["cuda_initialized"], "CUDA initialised")
    h = E.write_json_x(E.RESULT, res)
    log("RESULT", res["verdict"], "sha", h[:16], f"total wall {wall:.0f}s")


def build_series(fin, sand_acq, inter):
    """series[(t, c, w)] = {'pixel_ap', 'image_auroc'} of witness choice w over every KEYS cell. At acquisition E2A
    equals the frozen calibrators exactly (checked in phase C), so the published A3 acquisition cell is used; final
    cells come from pass 1, intermediate cells from pass 2."""
    series = {}
    for k, c in enumerate(E.ORDER):
        for w in W:
            series[(k, c, w)] = {m: sand_acq[c]["fusion_P"][m] for m in ("pixel_ap", "image_auroc")}
            for t in range(k + 1, E.NT - 1):
                series[(t, c, w)] = {m: inter[(t, c)][w][m] for m in ("pixel_ap", "image_auroc")}
            if k < E.NT - 1:
                series[(E.NT - 1, c, w)] = {m: fin[c][w][m] for m in ("pixel_ap", "image_auroc")}
    E.require(len(series) == len(E.KEYS) * E.N_SEL, "series incomplete")
    return series


def custody_summary():
    logl = E.access_log()
    idx_lock = next(i for i, x in enumerate(logl) if x.get("mark") == "calib_lock_verified")
    lab = [i for i, x in enumerate(logl) if "path" in x and ("/data/mvtec/" in x["path"]
                                                             or x["path"].endswith("DATASET_MANIFEST.json"))]
    return {"n_entries": len(logl), "n_denied": sum("denied" in x for x in logl), "calib_lock_verified_index": idx_lock,
            "first_label_access_index": lab[0] if lab else None,
            "labels_after_calib_lock": bool(not lab or lab[0] > idx_lock), "n_label_file_opens": len(lab),
            "governed_paths_opened": sorted({x["path"] for x in logl if "path" in x and "/data/mvtec/" not in x["path"]})}


def analyse(E, g, fin, sand, sand_final, sand_acq, a1_final, singles_ap, singles_au, e2a_ap, e2a_au, series, boot):
    O = E.ORDER
    per = {}
    for i, c in enumerate(O):
        sc = sand_final[c]
        per[c] = {"A1": a1_final[c]["pixel_ap"], "A2": sc["support_canonical"]["pixel_ap"], "TH_P": sc["theta_P"]["pixel_ap"],
                  "A3": fin[c]["frozen"]["pixel_ap"], "D1": fin[c]["E2A6"]["pixel_ap"], "E2A1_mean": e2a_ap[i],
                  "E2A1_min": min(fin[c][w]["pixel_ap"] for w in W), "E2A1_max": max(fin[c][w]["pixel_ap"] for w in W),
                  "E2A1T_mean": float(np.mean([fin[c][w]["pixel_ap"] for w in WT])),
                  "E2A1_image_auroc_mean": e2a_au[i], "A3_image_auroc": fin[c]["frozen"]["image_auroc"],
                  "D1_image_auroc": fin[c]["E2A6"]["image_auroc"],
                  "theta_is_min_A3": fin[c]["frozen"]["theta_is_min_fraction"],
                  "theta_is_min_E2A1": float(np.mean([fin[c][w]["theta_is_min_fraction"] for w in W])),
                  "theta_is_min_D1": fin[c]["E2A6"]["theta_is_min_fraction"]}
    col = lambda k: [per[c][k] for c in O]  # noqa: E731
    macro = {k: float(np.mean(col(k))) for k in per[O[0]]}
    macro["E2A1_per_choice"] = [float(np.mean([fin[c][w]["pixel_ap"] for c in O])) for w in W]
    macro["E2A1T_image_auroc"] = float(np.mean([np.mean([fin[c][w]["image_auroc"] for w in WT]) for c in O]))
    comps = {f"E2A1 - {a}": boot(col("E2A1_mean"), col(a)) for a in ("A1", "A2", "TH_P", "A3")}
    comps["D1 - E2A1"] = boot(col("D1"), col("E2A1_mean"))
    comps["E2A1T - E2A1 (diagnostic)"] = boot(col("E2A1T_mean"), col("E2A1_mean"))
    comps["E2A1 - A1 (image AUROC)"] = boot(e2a_au, singles_au["A1"])
    comps["E2A1 - A3 (image AUROC)"] = boot(e2a_au, col("A3_image_auroc"))
    for v in comps.values():
        v["n_categories_positive"] = None
    for name, (x, y) in {"E2A1 - TH_P": (col("E2A1_mean"), col("TH_P")), "E2A1 - A3": (col("E2A1_mean"), col("A3")),
                         "E2A1 - A2": (col("E2A1_mean"), col("A2")), "E2A1 - A1": (col("E2A1_mean"), col("A1"))}.items():
        comps[name]["n_categories_positive"] = int(sum(a > b for a, b in zip(x, y)))
    gap = macro["D1"] - macro["A3"]
    out = {"verdict": g["verdict"], "gate": g, "per_category_final": per, "macro_final": macro, "comparisons": comps,
           "d1_gain_recovered_fraction": (macro["E2A1_mean"] - macro["A3"]) / gap if gap > 0 else None,
           "bytes": {"A3_total": E.BYTES["a3_total"], "witness_fp32_15": 15 * E.BYTES["fp32_grid"],
                     "witness_anchor_stats_15": 15 * E.BYTES["witness_anchor_per_category"],
                     "E2A1_total": E.BYTES["a3_total"] + 15 * (E.BYTES["fp32_grid"] + E.BYTES["witness_anchor_per_category"]),
                     "D1_extra_not_deployable": E.BYTES["d1_extra"], "A2_total": 15359984},
           "forgetting": None}
    if series is not None:
        fm_p, fm_i, acq_p, stage = [], [], [], {}
        for w in W:
            sp = {(t, c): series[(t, c, w)]["pixel_ap"] for t, c in E.KEYS}
            si = {(t, c): series[(t, c, w)]["image_auroc"] for t, c in E.KEYS}
            fm_p.append(E.forgetting(sp))
            fm_i.append(E.forgetting(si))
        fm_cat = {c: float(np.mean([f[1][c] for f in fm_p])) for c in O[:-1]}
        a3s = sand["summary"]["A3"]
        out["forgetting"] = {
            "E2A1_pixel_fm": float(np.mean([f[0] for f in fm_p])), "E2A1_image_fm": float(np.mean([f[0] for f in fm_i])),
            "E2A1_pixel_fm_per_choice": [f[0] for f in fm_p], "E2A1_pixel_fm_per_category": fm_cat,
            "A3_pixel_fm": a3s["pixel_fm"], "A1_pixel_fm": sand["summary"]["A1"]["pixel_fm"],
            "A2_pixel_fm": sand["summary"]["A2"]["pixel_fm"], "A3_image_fm": a3s["image_fm"],
            "E2A1_acq_pixel_ap": float(np.mean([sand_acq[c]["fusion_P"]["pixel_ap"] for c in O])),
            "A3_acq_pixel_ap": a3s["acq_pixel_ap"],
            "stage_macro_pixel_ap": {t + 1: float(np.mean([np.mean([series[(t, c, w)]["pixel_ap"] for w in W])
                                                           for c in O[:t + 1]])) for t in range(E.NT)}}
        a3_fm_cat = {c: sand["summary"]["A3"]["per_category"][c]["pixel_fm"] for c in O[:-1]}
        out["forgetting"]["E2A1_minus_A3_pixel_fm"] = boot([fm_cat[c] for c in O[:-1]], [a3_fm_cat[c] for c in O[:-1]])
    return out


if __name__ == "__main__":
    {"calib": calib, "evaluate": evaluate}[sys.argv[1]]()
