"""SAND-FINAL scoring of one (decoder run, support-memory stream) pair at all 120 (stage, category) cells.

    python -m exps.sand.score grids TAG           # GPU: decoder residual grids for test images and normal probes
    python -m exps.sand.score cells TAG W N       # CPU worker W of N: support distances, calibrators, fused metrics

Opens MVTec test features and masks; refuses to run without the LOCK. Cells are create-exclusive and resumable.
Protocol sections 2 and 4 (agents/plans/sand_final_protocol.md).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from exps.cadfinal import metrics as MX
from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_c2dom import common as D2
from exps.sand import common as S
from exps.task_p.p0_normal_support import require

HELD = ("bottle", "carpet", "hazelnut")            # categories with PTAG held-out normals (in-sample diagnostic)
SPOT = ((1, "bottle"), (5, "grid"), (15, "zipper"))


def arms():
    """TAG -> decoder run, canonical decoder reference (or None: compute it), memory stream, canonical memory cells."""
    out = {"A3_s2711": {"decoder_run": S.A1_RUN, "decoder_ref": S.A1_OUT, "memory": S.MEM_DIR, "memory_ref": S.MEM_EVAL,
                        "memory_name": S.MEM_ARM}}
    reg = S.OUT / "ARMS.json"                         # later phases register their runs here (create-exclusive records)
    if reg.exists():
        for tag, a in S.read_json(reg).items():
            out[tag] = {k: (None if v is None else (S.ROOT / v if k != "memory_name" else v)) for k, v in a.items()}
    return out


def supports(mem_dir, t, only=None):
    """Decoded category supports of a PQ memory stream at boundary t (exps.nl_sgm_01a_c2dom.stream.supports_at for
    any stream directory)."""
    ck = torch.load(mem_dir / f"ckpt_t{t:02d}.pt", weights_only=False)
    n_rows = int(ck["header"][1:].sum())
    ids, own = ck["ids"][:n_rows].long(), ck["owner"][:n_rows].long()
    out = {}
    for k in range(t + 1):
        if only is not None and S.ORDER[k] != only:
            continue
        m = own == k
        x = K.decode(ck["codes"][:n_rows][m], ck["codebooks"][k])
        g = ids[m]
        o = torch.argsort(g)
        out[S.ORDER[k]] = X.RowSet(x[o].contiguous(), g[o])
    return out


# ------------------------------------------------------------------ GPU: decoder grids

@torch.no_grad()
def _decode(dec, f, batch=16):
    from exps.vitb8_q2 import common as V
    grid, s256, img = [], [], []
    for xb in f.split(batch):
        en, de = dec(xb.cuda())
        grid.append(V.grid(en, de).cpu())
        s, i = V.score(en, de)
        s256.append(s.cpu()); img.append(i.cpu())
    return torch.cat(grid), torch.cat(s256), torch.cat(img)


def do_grids(tag):
    from exps.dinomaly import run as DR
    from exps.vitb8_q2 import common as V
    from exps.vitb8_q2 import evaluate as QE
    S.verify_lock()
    arm = arms()[tag]
    DR.configure()
    log = S.Log(f"grids_{tag}")
    gdir = S.OUT / tag / "grids"
    gdir.mkdir(parents=True, exist_ok=True)
    co = C.cohorts()
    F_ = C.Features()
    probes = {c: (F_.images(co[c]["sel"]).clone(), F_.images(co[c]["eval"]).clone()) for c in S.ORDER}
    F_.drop()
    held = {c: V.heldout_features(c).clone() for c in HELD}
    ids = V.identities()
    spot = []
    t_all = time.time()
    for s in range(1, 16):
        rec = S.read_json(arm["decoder_run"] / f"STAGE{s:02d}.json")
        todo = [c for c in S.ORDER[:s] if not (gdir / f"stage{s:02d}_{c}.pt").exists()]
        checked = (S.OUT / tag / "DECODER_CHECK.json").exists()
        need_spot = arm["decoder_ref"] is not None and any((s, c) in SPOT for c in S.ORDER[:s]) and not checked
        canon = S.OUT / tag / "decoder_canonical" / f"EVAL_STAGE{s:02d}.json"
        if not todo and not need_spot and (arm["decoder_ref"] is not None or canon.exists()):
            continue
        ck = S.ROOT / rec["checkpoint"]
        require(S.sha(ck) == rec["checkpoint_sha256"], f"checkpoint changed: {ck}")
        dec = V.SingleStreamDecoder(checkpoint=False).cuda()
        dec.load_state_dict(torch.load(ck, weights_only=True)["state"])
        dec.eval()
        ref = None if arm["decoder_ref"] is None else S.read_json(arm["decoder_ref"] / f"EVAL_STAGE{s:02d}.json")
        per = {}
        for c in S.ORDER[:s]:
            p = gdir / f"stage{s:02d}_{c}.pt"
            do_spot = ref is not None and (s, c) in SPOT and not checked
            if p.exists() and ref is not None and not do_spot:
                continue
            f, paths = V.test_features(c)
            grid, s256, img = _decode(dec, f)
            require(bool(torch.isfinite(grid).all()) and bool(torch.isfinite(img).all()), f"non-finite scores {c}")
            rows = ids["test"][c]
            require(paths == [r["relative_path"] for r in rows], f"identity order {c}")
            labels = np.array([r["label"] for r in rows])
            out = {"grid": grid, "image": img, "paths": paths, "checkpoint_sha256": rec["checkpoint_sha256"],
                   "sel": _decode(dec, probes[c][0])[0], "eval": _decode(dec, probes[c][1])[0]}
            if c in held:
                out["held"] = _decode(dec, held[c])[0]
            if ref is not None:
                r = np.asarray(ref["per_category"][c]["image_scores"], dtype=np.float64)
                out["image_score_max_abs_diff"] = float(np.abs(img.numpy().astype(np.float64) - r).max())
                require(out["image_score_max_abs_diff"] <= S.TOL["a1_image_score"],
                        f"decoder image scores do not reproduce the reference ({s}, {c}): {out['image_score_max_abs_diff']}")
            if ref is None or do_spot:
                hws, masks = QE.masks_for(c, rows)
                maps = [MX.to_native("dinomaly", x, hw) for x, hw in zip(s256.numpy(), hws)]
                ap = float(MX.exact_ap(maps, masks)["ap"])
                del maps, masks
                per[c] = {"pixel_ap": ap, "image_auroc": float(MX.auroc(labels, img.numpy())), "n": len(rows),
                          "n_anomalous": int(labels.sum()), "image_scores": img.numpy().tolist()}
                if ref is not None:
                    d = abs(ap - ref["per_category"][c]["pixel_ap"])
                    spot.append({"stage": s, "category": c, "pixel_ap": ap, "pixel_ap_ref": ref["per_category"][c]["pixel_ap"],
                                 "abs_diff": d, "ok": bool(d <= S.TOL["a1_pixel_ap"])})
                    require(spot[-1]["ok"], f"decoder Pixel AP does not reproduce the reference: {spot[-1]}")
            if not p.exists():
                part = p.with_suffix(".part")
                torch.save(out, part)
                part.replace(p)
            log(tag, "grid", s, c, "n", len(paths), "diff", out.get("image_score_max_abs_diff"),
                per.get(c, {}).get("pixel_ap"))
        if ref is None and not canon.exists():
            require(len(per) == s, "canonical decoder evaluation incomplete")
            S.write_json(canon, {"stage": s, "checkpoint_sha256": rec["checkpoint_sha256"], "per_category": per,
                                 "macro_pixel_ap": float(np.mean([v["pixel_ap"] for v in per.values()])),
                                 "macro_image_auroc": float(np.mean([v["image_auroc"] for v in per.values()])),
                                 "scorer": "exps.vitb8_q2.common.score + exps.cadfinal.metrics (as VITB8-NL-00)",
                                 "utc": S.now()})
        del dec
        torch.cuda.empty_cache()
    if arm["decoder_ref"] is not None and not (S.OUT / tag / "DECODER_CHECK.json").exists():
        require(len(spot) == len(SPOT), "decoder spot checks incomplete")
        S.write_json(S.OUT / tag / "DECODER_CHECK.json", {"spot": spot, "all_ok": all(x["ok"] for x in spot)})
    if not (S.OUT / tag / "GRIDS_DONE.json").exists():
        S.write_json(S.OUT / tag / "GRIDS_DONE.json", {"tag": tag, "seconds_this_process": time.time() - t_all,
                                                      "vram_peak_bytes": int(torch.cuda.max_memory_allocated()),
                                                      "utc": S.now()})
    log("GRIDS DONE", tag)


# ------------------------------------------------------------------ CPU: distances, calibration, fusion

def _dist(arm, c, t, gsha, q):
    """Exact 1-NN distances of the test / SEL-probe / EVAL-probe patches to c's support at boundary t. Cached by the
    support's gid digest (a support with the same ids has the same rows, as in the C2-DOM evaluation)."""
    ddir = S.OUT / "dist" / arm["memory_name"]
    ddir.mkdir(parents=True, exist_ok=True)
    p = ddir / f"{c}_{gsha[:24]}.npz"
    if p.exists():
        z = np.load(p)
        return z["test"], z["sel"], z["eval"], 0.0
    rows = supports(arm["memory"], t, only=c)[c]
    require(D2.gid_sha(rows.ids) == gsha, "support ids differ from the stream ledger")
    t0 = time.time()
    out = [X.nn(x, rows)[0].reshape(-1, S.TOKENS).numpy() for x in q]
    sec = time.time() - t0
    part = ddir / f"{c}_{gsha[:24]}.part.npz"
    np.savez(part, test=out[0], sel=out[1], eval=out[2])
    os.replace(part, p)
    return out[0], out[1], out[2], sec


def do_cells(tag, worker, n_workers):
    from exps.vitb8_q2 import common as V
    S.verify_lock()
    arm = arms()[tag]
    require((S.OUT / tag / "GRIDS_DONE.json").exists(), "decoder grids missing")
    C.set_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    log = S.Log(f"cells_{tag}_w{worker}")
    man = S.read_json(A3.OUT / "DATASET_MANIFEST.json")
    cdir, kdir, gdir = S.OUT / tag / "cells", S.OUT / tag / "calib", S.OUT / tag / "grids"
    cdir.mkdir(parents=True, exist_ok=True); kdir.mkdir(parents=True, exist_ok=True)
    co = C.cohorts()
    F_ = C.Features()
    bnd = [S.read_json(arm["memory"] / f"boundary_t{t:02d}.json") for t in range(15)]
    t_all = time.time()
    for k, c in enumerate(S.ORDER):
        if k % n_workers != worker or all((cdir / f"stage{t + 1:02d}_{c}.json").exists() for t in range(k, 15)):
            continue
        feats, paths = V.test_features(c)
        labels, masks, items = A3.load_masks(c)
        require([x["relative_path"] for x in items] == paths, f"test identity order {c}")
        hw = tuple(man["categories"][c]["images"][0]["hw"])
        require(all(tuple(x["hw"]) == hw for x in items), f"mixed native sizes {c}")
        q = [feats.reshape(-1, S.DIM).contiguous(), F_.images(co[c]["sel"]).reshape(-1, S.DIM).clone(),
             F_.images(co[c]["eval"]).reshape(-1, S.DIM).clone()]
        F_.drop(c)
        for t in range(k, 15):
            s = t + 1
            cp = cdir / f"stage{s:02d}_{c}.json"
            if cp.exists():
                continue
            t0 = time.time()
            gsha = bnd[t]["per_category_gids_sha"][c]
            d, d_sel, d_ev, nn_s = _dist(arm, c, t, gsha, q)
            G = torch.load(gdir / f"stage{s:02d}_{c}.pt", weights_only=False)
            require(G["paths"] == paths, "grid identity order")
            a = G["grid"].numpy().reshape(-1, S.TOKENS)
            a_sel, a_ev = G["sel"].numpy().reshape(-1, S.TOKENS), G["eval"].numpy().reshape(-1, S.TOKENS)
            cell = {"tag": tag, "stage": s, "category": c, "gids_sha": gsha, "rows": bnd[t]["per_category_rows"][c],
                    "n_images": int(len(labels)), "n_anomalous": int(labels.sum()), "nn_seconds": nn_s}
            # canonical support channel: identical to the locked C2-DOM cell
            if arm["memory_ref"] is not None:
                ref = S.read_json(arm["memory_ref"] / f"stage{s:02d}_{c}.json")
                m2 = A3.native_maps(torch.from_numpy(d), hw)
                diff = float(np.abs(m2.reshape(len(m2), -1).max(1).astype(np.float64) - np.asarray(ref["image_scores"])).max())
                del m2
                require(diff <= S.TOL["a2_image_score"], f"support image scores do not reproduce C2-DOM ({s}, {c}): {diff}")
                cell["support_canonical"] = {"pixel_ap": ref["pixel_ap"], "image_auroc": ref["image_auroc"],
                                             "image_score_max_abs_diff": diff, "source": "locked C2-DOM cell"}
            else:
                m2 = A3.native_maps(torch.from_numpy(d), hw)
                cell["support_canonical"] = S.metrics(m2, m2.reshape(len(m2), -1).max(1), labels, masks)
                del m2
            cell["decoder_image_auroc"] = float(roc_auc_score(labels, G["image"].numpy().astype(np.float64)))
            # calibrators: frozen at acquisition (primary); recomputed now (D1, diagnostic only)
            kp = kdir / f"{c}.json"
            if t == k:
                require(not kp.exists(), f"calibrators exist before acquisition cell {c}")
                S.write_json(kp, {**S.calibrate(a_sel, d_sel), "category": c, "stage": s, "gids_sha": gsha,
                                  "checkpoint_sha256": G["checkpoint_sha256"], "probe_paths": co[c]["sel"]})
            cal = S.read_json(kp)
            require(cal["stage"] == k + 1, "calibrators are not the acquisition calibrators")
            cal_now = S.calibrate(a_sel, d_sel)
            g, frac = S.fuse(a, d, cal)
            maps, img = S.readout_native(g, hw)
            cell["fusion_P"] = {**S.metrics(maps, img, labels, masks), "theta_is_min_fraction": frac,
                                "image_scores": [float(v) for v in img]}
            del maps
            if t == k or t == 14:
                maps, img = S.readout_dinomaly(g, hw)
                cell["fusion_S1"] = S.metrics(maps, img, labels, masks)
                del maps
                maps, img = S.readout_native(np.maximum(a, 0).astype(np.float32), hw)
                cell["theta_P"] = S.metrics(maps, img, labels, masks)
                del maps
                if t == k:
                    cell["fusion_D1"] = {k2: cell["fusion_P"][k2] for k2 in ("pixel_ap", "image_auroc")}
                else:
                    g1, _ = S.fuse(a, d, cal_now)
                    maps, img = S.readout_native(g1, hw)
                    cell["fusion_D1"] = S.metrics(maps, img, labels, masks)
                    del maps, g1
            # normal-only calibration validation (EVAL probes are disjoint from the SEL probes and from M)
            zt, zm = S.zscores(a_ev, d_ev, cal)
            zts, zms = S.zscores(a_sel, d_sel, cal)
            cell["calibration"] = {"frozen": {k2: cal[k2] for k2 in ("mu_theta", "sd_theta", "mu_m", "sd_m")},
                                   "refreshed": {k2: cal_now[k2] for k2 in ("mu_theta", "sd_theta", "mu_m", "sd_m")},
                                   "eval_probe_z_theta": S.zstats(zt), "eval_probe_z_m": S.zstats(zm),
                                   "eval_probe_fused": S.zstats(np.minimum(zt, zm)),
                                   "sel_probe_z_theta": S.zstats(zts), "sel_probe_z_m": S.zstats(zms)}
            if "held" in G:                              # PTAG held-out normals: never in the decoder's training set
                ah = np.maximum(G["held"].numpy().astype(np.float64), 0).ravel()
                cell["calibration"]["heldout_theta"] = {"mean": float(ah.mean()), "std": float(ah.std()),
                                                         "z_mean_under_frozen": float((ah.mean() - cal["mu_theta"]) / cal["sd_theta"])}
            cell["seconds"] = time.time() - t0
            cell["peak_rss_gb"] = C.peak_rss_gb()
            require(cell["peak_rss_gb"] <= S.CAPS["rss_gb"], "CAP: RSS")
            S.write_json(cp, cell)
            log(tag, "stage", s, c, "P", round(cell["fusion_P"]["pixel_ap"], 4), round(cell["fusion_P"]["image_auroc"], 4),
                "M", round(cell["support_canonical"]["pixel_ap"], 4), "theta<M", round(frac, 3), f"{cell['seconds']:.0f}s")
        del feats, masks, q
    log("CELLS DONE", tag, "worker", worker, round(time.time() - t_all))


if __name__ == "__main__":
    if sys.argv[1] == "grids":
        do_grids(sys.argv[2])
    elif sys.argv[1] == "cells":
        do_cells(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(sys.argv[1])
