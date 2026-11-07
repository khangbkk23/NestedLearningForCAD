"""VITB8-NL-00 Phase 2: the matched Q2-OFF control (task agents/tasks/task_vitb8_nl_00.md).

Identical to exps.vitb8_q2.train.train except aux=None (rho = 0, no Q2 forward/backward). Everything else is imported
unchanged from the locked VITB8-Q2 sources (features, decoder + init, stage seeds, Replay-24, optimizer, scheduler,
clip, numerics, diagnostics, scorer). Q2-ON's auxiliary pass runs inside fork_rng with dropout disabled and its own
host/negative generators, so the OFF arm consumes the identical data/dropout stream.

  python -m exps.vitb8_nl00.run lock       # comparison contract (sources, VITB8-Q2 lock + results hashes)
  python -m exps.vitb8_nl00.run train      # resumable 15 stages
  python -m exps.vitb8_nl00.run follow     # evaluate each stage as it appears (frozen scorer, per-image scores)
  python -m exps.vitb8_nl00.run summary    # FINAL.json + paired ON vs OFF comparison
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from exps.cadfinal import metrics as M
from exps.dinomaly import common as D
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nl_align_01 import g1 as G
from exps.vitb8_q2 import common as V
from exps.vitb8_q2 import evaluate as QE
from exps.vitb8_q2 import train as T

OUT = V.ROOT / "results/hope_cad/vitb8_nl00"
RUN = OUT / "runs" / f"q2off_seed{V.SEED}"
LOCK = OUT / "LOCK.json"
ON = V.OUT
SOURCES = ["exps/vitb8_nl00/run.py", "agents/tasks/task_vitb8_nl_00.md"] + T.SOURCES
CONTRACT = {
    "arm": "Q2_OFF: rho = 0, auxiliary Q2 pass removed (aux=None); all else = VITB8-Q2 PROTOCOL.json",
    "paired_with": "VITB8-Q2 Q2_ON seed 2711 (results/hope_cad/vitb8_q2), not retrained",
    "primary": "final macro native Pixel AP over 15 categories; ON - OFF",
    "secondary": ["final macro Image AUROC", "signed / positive-part acquisition->final forgetting (AP and AUROC)",
                  "per-category acquisition and final AP/AUROC", "stage macro AP trajectory",
                  "test/good image-score drift (mean, q99) per category across stages", "time, storage"],
    "inference": "one training seed: paired per-category differences are descriptive; no significance claim",
    "scorer": "exps.vitb8_q2.common.score + exps.cadfinal.metrics (identical to VITB8-Q2)",
}


def lock():
    require(not LOCK.exists(), "already locked")
    T.verify_lock()
    on = {f"EVAL_STAGE{s:02d}.json": sha(ON / f"EVAL_STAGE{s:02d}.json") for s in range(1, 16)}
    on["FINAL.json"] = sha(ON / "FINAL.json")
    D.write_json(LOCK, {"locked_utc": DR.now(), "contract": CONTRACT,
                        "sources": {s: sha(V.ROOT / s) for s in SOURCES},
                        "vitb8_q2_lock_sha256": sha(T.LOCK), "vitb8_q2_results": on, "seed": V.SEED})
    print("locked", sha(LOCK), flush=True)


def verify_lock():
    lk = D.read(LOCK)
    bad = [s for s, d in lk["sources"].items() if sha(V.ROOT / s) != d]
    require(not bad and lk["contract"] == CONTRACT, f"post-lock change: {bad}")
    T.verify_lock()
    return lk


def run_stage(dec, s, ids, items, role, held):
    c, current, replay, replay_keys, idx = T.stage_data(ids, s)
    ck, js = RUN / f"stage{s:02d}_{c}.pt", RUN / f"STAGE{s:02d}.json"
    dseed, gseed = T.RB.stage_seeds("R15", V.SEED, s)
    torch.manual_seed(dseed)
    gen = torch.Generator().manual_seed(gseed)
    theta0 = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses, skips, log = T.train_stage(dec, current, V.ITERS, gen, replay, aux=None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
    require(all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), "non-finite parameters")
    a = np.asarray(log["norm_pre_clip"], dtype=np.float64)
    rec = {"index": s, "name": c, "iters": V.ITERS, "seconds": seconds, "train_images": len(current), "arm": "Q2_OFF",
           "stage_seeds": [dseed, gseed], "nonfinite_skips": skips,
           "loss_curve_every50": [float(np.nanmean(losses[j:j + 50])) for j in range(0, len(losses), 50)],
           "loss_last50": float(np.nanmean(losses[-50:])), "clip_saturation": float(np.mean(a > 0.1)),
           "heldout_normal": {h: T.normal_stats(dec, held[h]) for h in held if h in V.STREAM[:s]},
           "current_normal_first64": T.normal_stats(dec, current[:64]),
           "replay_normal": None if replay is None else T.normal_stats(dec, replay),
           "synthetic_q2_diag_ap": T.synthetic_diag(dec, items, role, s),
           "replay": None if replay is None else {"allocation": {d: len(idx[d]) for d in idx}, "slots": len(replay_keys),
                                                  "identities": [ids["train"][d][i]["relative_path"] for d, i in replay_keys]},
           "displacement_from_stage_start": T.displacement(state, theta0),
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    torch.save(log, RUN / f"LOG{s:02d}.pt")
    rec["checkpoint"] = str(ck.relative_to(V.ROOT))
    rec["checkpoint_sha256"] = T.save_ckpt(dec, ck, {"seed": V.SEED, "stage": s, "name": c, "arm": "Q2_OFF"})
    rec["utc"] = DR.now()
    D.write_json(js, rec)
    print("stage", s, c, f"{seconds:.0f}s", "skips", len(skips), "loss", round(rec["loss_last50"], 4),
          "synAP", round(rec["synthetic_q2_diag_ap"]["macro"], 4),
          {h: round(v["global_cosine"], 4) for h, v in rec["heldout_normal"].items()}, flush=True)


def train():
    T.env()
    verify_lock()
    ids = V.identities()
    items, hosts = V.load_bank()                      # label-free synthetic diagnostic only; never a gradient here
    role = G.roles(hosts)
    held = {h: V.heldout_features(h).clone() for h in T.HELD}
    RUN.mkdir(parents=True, exist_ok=True)
    require(not (RUN / "TRAIN.json").exists(), "run already completed")
    dec, last = V.new_decoder(V.SEED), 0
    for s in range(1, 16):
        js = RUN / f"STAGE{s:02d}.json"
        if not js.exists():
            break
        rec = D.read(js)
        require(sha(V.ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
        last = s
    if last:
        dec.load_state_dict(torch.load(V.ROOT / D.read(RUN / f"STAGE{last:02d}.json")["checkpoint"],
                                       weights_only=True)["state"])
        print("resume after stage", last, flush=True)
    for s in range(last + 1, 16):
        c = V.STREAM[s - 1]
        for orphan in (RUN / f"stage{s:02d}_{c}.pt", RUN / f"stage{s:02d}_{c}.part"):
            if orphan.exists():
                orphan.unlink()
        run_stage(dec, s, ids, items, role, held)
    recs = [D.read(RUN / f"STAGE{s:02d}.json") for s in range(1, 16)]
    D.write_json(RUN / "TRAIN.json", {
        "arm": "Q2_OFF", "seed": V.SEED, "finished_utc": DR.now(), "train_seconds_total": sum(r["seconds"] for r in recs),
        "peak_allocated_mib_max": max(r["peak_allocated_mib"] for r in recs),
        "persistent_decoder_bytes": V.state_bytes(dec),
        "nonfinite_skips_total": sum(len(r["nonfinite_skips"]) for r in recs)})
    print("train complete", flush=True)


# --- evaluation (VITB8-Q2 eval_stage with this run's paths, plus per-image image scores) ---------------------------


def eval_path(s):
    return OUT / f"EVAL_STAGE{s:02d}.json"


def eval_stage(s, ids):
    target = eval_path(s)
    require(not target.exists(), f"one-shot: {target}")
    rec = D.read(RUN / f"STAGE{s:02d}.json")
    ck = V.ROOT / rec["checkpoint"]
    require(sha(ck) == rec["checkpoint_sha256"], f"checkpoint changed: {ck}")
    dec = V.SingleStreamDecoder(checkpoint=False).cuda()
    dec.load_state_dict(torch.load(ck, weights_only=True)["state"])
    dec.eval()
    out, t0 = {}, time.perf_counter()
    for c in V.STREAM[:s]:
        rows = ids["test"][c]
        s256, img, paths = QE.predict(dec, c)
        require(paths == [r["relative_path"] for r in rows], f"identity order {c}")
        require(bool(torch.isfinite(s256).all()) and bool(torch.isfinite(img).all()), f"non-finite scores {c}")
        labels = np.array([r["label"] for r in rows])
        hws, masks = QE.masks_for(c, rows)
        maps = [M.to_native("dinomaly", x, hw) for x, hw in zip(s256.numpy(), hws)]
        ap = M.exact_ap(maps, masks)["ap"]
        del maps, masks
        im = img.numpy()
        good, bad = im[labels == 0], im[labels == 1]
        out[c] = {"pixel_ap": ap, "image_auroc": M.auroc(labels, im), "n": len(rows), "n_anomalous": int(labels.sum()),
                  "good_image_score_mean": float(good.mean()), "good_image_score_q99": float(np.quantile(good, 0.99)),
                  "anomalous_image_score_mean": float(bad.mean()), "map_min": float(s256.min()),
                  "map_max": float(s256.max()), "map_std": float(s256.std()), "image_scores": im.tolist()}
        print("eval stage", s, c, "AP", round(ap, 4), "AUROC", round(out[c]["image_auroc"], 4), flush=True)
    res = {"stage": s, "arm": "Q2_OFF", "checkpoint_sha256": rec["checkpoint_sha256"], "per_category": out,
           "macro_pixel_ap": float(np.mean([v["pixel_ap"] for v in out.values()])),
           "macro_image_auroc": float(np.mean([v["image_auroc"] for v in out.values()])),
           "seconds": time.perf_counter() - t0, "lock_sha256": sha(LOCK), "utc": DR.now()}
    D.write_json(target, res)
    print("stage", s, "macro AP", round(res["macro_pixel_ap"], 4), "macro AUROC", round(res["macro_image_auroc"], 4),
          flush=True)
    del dec
    torch.cuda.empty_cache()


def follow(poll=60):
    DR.configure()
    T.cap("2.0")
    verify_lock()
    ids = V.identities()
    for s in range(1, 16):
        if eval_path(s).exists():
            continue
        while not (RUN / f"STAGE{s:02d}.json").exists():
            time.sleep(poll)
        eval_stage(s, ids)
    print("evaluation complete", flush=True)


def summary():
    verify_lock()
    target = OUT / "FINAL.json"
    require(not target.exists(), "exists")
    arms = {"ON": {s: D.read(ON / f"EVAL_STAGE{s:02d}.json") for s in range(1, 16)},
            "OFF": {s: D.read(eval_path(s)) for s in range(1, 16)}}
    res = {}
    for a, ev in arms.items():
        fin = ev[15]["per_category"]
        acq = {c: ev[j + 1]["per_category"][c] for j, c in enumerate(V.STREAM)}
        f_ap = {c: acq[c]["pixel_ap"] - fin[c]["pixel_ap"] for c in V.STREAM[:-1]}
        f_au = {c: acq[c]["image_auroc"] - fin[c]["image_auroc"] for c in V.STREAM[:-1]}
        res[a] = {"final_macro_pixel_ap": ev[15]["macro_pixel_ap"], "final_macro_image_auroc": ev[15]["macro_image_auroc"],
                  "acq_macro_pixel_ap": float(np.mean([acq[c]["pixel_ap"] for c in V.STREAM])),
                  "forgetting_ap_signed": float(np.mean(list(f_ap.values()))),
                  "forgetting_ap_positive": float(np.mean(np.maximum(list(f_ap.values()), 0))),
                  "forgetting_auroc_signed": float(np.mean(list(f_au.values()))),
                  "forgetting_auroc_positive": float(np.mean(np.maximum(list(f_au.values()), 0))),
                  "per_category": {c: {"acq_ap": acq[c]["pixel_ap"], "final_ap": fin[c]["pixel_ap"],
                                       "acq_auroc": acq[c]["image_auroc"], "final_auroc": fin[c]["image_auroc"]}
                                   for c in V.STREAM},
                  "trajectory_ap": [ev[s]["macro_pixel_ap"] for s in range(1, 16)],
                  "trajectory_auroc": [ev[s]["macro_image_auroc"] for s in range(1, 16)],
                  "good_drift": {c: [[ev[s]["per_category"][c]["good_image_score_mean"],
                                      ev[s]["per_category"][c]["good_image_score_q99"]]
                                     for s in range(j + 1, 16)] for j, c in enumerate(V.STREAM)}}
    d = {c: res["ON"]["per_category"][c]["final_ap"] - res["OFF"]["per_category"][c]["final_ap"] for c in V.STREAM}
    da = {c: res["ON"]["per_category"][c]["acq_ap"] - res["OFF"]["per_category"][c]["acq_ap"] for c in V.STREAM}
    res["paired_ON_minus_OFF"] = {
        "final_macro_ap": res["ON"]["final_macro_pixel_ap"] - res["OFF"]["final_macro_pixel_ap"],
        "final_macro_auroc": res["ON"]["final_macro_image_auroc"] - res["OFF"]["final_macro_image_auroc"],
        "acq_macro_ap": res["ON"]["acq_macro_pixel_ap"] - res["OFF"]["acq_macro_pixel_ap"],
        "forgetting_ap_signed": res["ON"]["forgetting_ap_signed"] - res["OFF"]["forgetting_ap_signed"],
        "forgetting_ap_positive": res["ON"]["forgetting_ap_positive"] - res["OFF"]["forgetting_ap_positive"],
        "per_category_final_ap": d, "per_category_acq_ap": da,
        "categories_ON_better_final": sum(v > 0 for v in d.values()),
        "stage_ap": [a - b for a, b in zip(res["ON"]["trajectory_ap"], res["OFF"]["trajectory_ap"])],
        "note": "one training seed; descriptive paired differences only"}
    tr = D.read(RUN / "TRAIN.json")
    res["cost_OFF"] = {"train_seconds": tr["train_seconds_total"], "peak_allocated_mib": tr["peak_allocated_mib_max"],
                       "decoder_bytes": tr["persistent_decoder_bytes"], "nonfinite_skips": tr["nonfinite_skips_total"]}
    res["cost_ON"] = {k: D.read(ON / "FINAL.json")["training"][k] for k in ("seconds_total", "peak_allocated_mib")}
    res["lock_sha256"] = sha(LOCK)
    D.write_json(target, res)
    p = res["paired_ON_minus_OFF"]
    print("ON-OFF final AP", round(p["final_macro_ap"], 4), "AUROC", round(p["final_macro_auroc"], 4),
          "acqAP", round(p["acq_macro_ap"], 4), "F+", round(p["forgetting_ap_positive"], 4),
          "ON better in", p["categories_ON_better_final"], "/15", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("lock", "train", "follow", "summary"))
    a = ap.parse_args()
    {"lock": lock, "train": train, "follow": follow, "summary": summary}[a.stage]()


if __name__ == "__main__":
    main()
