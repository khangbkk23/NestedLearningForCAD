# exps/nlcad/stage_d.py
"""NLCAD-01 Stage D: native Pixel AP of a finished NL segment on the authorized development cohort (consumed P1
cohort: bottle/carpet/hazelnut, 60 images), paired with the RH-01b Replay segment from the same task-6 checkpoint.

Order (enforced): lock (configuration, sources, every checkpoint SHA, the comparison definitions) -> score -> metrics
-> compare. Single seed => descriptive only: image-bootstrap intervals are NOT training-seed variability.

  python -m exps.nlcad.stage_d lock    --run N_per_task_seed2711_from6
  python -m exps.nlcad.stage_d score   --run N_per_task_seed2711_from6 [--identity-too]
  python -m exps.nlcad.stage_d metrics --run N_per_task_seed2711_from6
  python -m exps.nlcad.stage_d compare --run N_per_task_seed2711_from6
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import analysis as DA
from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB
from exps.rh01b.analysis import ACQ, FINAL
from exps.task_p.p1_readout import native_map
from exps.task_p.p2.metrics import exact_ranking

OUT = D.ROOT / "results/hope_cad/nlcad"
SOURCES = ["exps/nlcad/controller.py", "exps/nlcad/fast.py", "exps/nlcad/objectives.py", "exps/nlcad/bank.py",
           "exps/nlcad/run.py", "exps/nlcad/stage_d.py", "exps/rh01b/run.py", "exps/rh01b/analysis.py",
           "exps/dinomaly/analysis.py", "exps/task_p/p2/metrics.py", "exps/task_p/p1_readout.py"]


def d(run):
    return OUT / "stage_d" / run


def checkpoints(run, identity=False):
    cfg = D.read(OUT / f"runs/{run}/CONFIG.json")
    out = []
    first = 1 if identity else cfg["start"] + 1                  # identity: also the shared acquisition stages 1-6
    for s in range(first, len(C.STREAM) + 1):
        if identity:
            rec = D.read(C.OUT / f"runs/R15_seed{cfg['seed']}/STAGE{s}.json")
        else:
            rec = D.read(OUT / f"runs/{run}/STAGE{s}.json")
        path = D.ROOT / rec["checkpoint"]
        require(sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
        admitted = [c for c in C.EVAL if C.STREAM.index(c) < s]
        out.append(("ID" if identity else "NL", cfg["seed"], s, admitted, path, rec["checkpoint_sha256"]))
    return cfg, out


def lock(run):
    target = d(run) / "LOCK.json"
    require(not target.exists(), "already locked")
    cfg, nl = checkpoints(run)
    _, idn = checkpoints(run, identity=True)
    require(len(nl) == len(C.STREAM) - cfg["start"], "NL segment incomplete")
    D.write_json(target, {
        "locked_utc": DR.now(), "run": run, "config": cfg, "sources": {s: sha(D.ROOT / s) for s in SOURCES},
        "nl_checkpoints": {str(s): h for _, _, s, _, _, h in nl}, "identity_checkpoints": {str(s): h for _, _, s, _, _, h in idn},
        "evaluation": "consumed P1 development cohort (bottle/carpet/hazelnut, 10 good + 10 defective each); "
                      "dev features from the frozen DRF-01 cache; R_D primary, R_P1 secondary",
        "comparison": {"primary": "final (stage 15) macro native Pixel AP over the three categories, NL - identity",
                       "secondary": ["pooled AP", "per-category final AP", "forgetting F_c = AP(acq) - AP(final) with "
                                     "the shared acquisition checkpoints (stages 1/4/6, identical for both arms)",
                                     "per-stage AP trajectory 7-15", "Image AUROC"],
                       "uncertainty": "conservative paired whole-image bootstrap on the identical P1 draws; one seed, "
                                      "so no training-seed variability: descriptive only"},
        "identity_reference": "RH-01b R15 same seed: same task-6 start, same seeds, bitwise-equivalent machinery"})
    print("locked", sha(target), flush=True)


def verify(run):
    lk = D.read(d(run) / "LOCK.json")
    for s, h in lk["sources"].items():
        require(sha(D.ROOT / s) == h, f"post-lock change: {s}")
    return lk


@torch.no_grad()
def score(run, identity_too):
    DR.configure()
    lk = verify(run)
    proto = RB.verify_lock()
    key = "results/hope_cad/dinomaly/features/dev.pt"
    require(sha(D.ROOT / key) == R1.frozen()[key]["sha256"], "dev features not frozen")
    dev = torch.load(D.ROOT / key, weights_only=True)
    index = {p: i for i, p in enumerate(dev["paths"])}
    arms = [False, True] if identity_too else [False]
    for identity in arms:
        tag = "identity" if identity else "nl"
        target = d(run) / f"SCORES_{tag}.json"
        require(not target.exists(), "scoring is one-shot")
        _, cks = checkpoints(run, identity)
        record = {}
        for arm, seed, s, admitted, path, h in cks:
            require(lk[f"{'identity' if identity else 'nl'}_checkpoints"][str(s)] == h, "checkpoint not locked")
            dec = O.Decoder(768, 12).cuda().eval()
            snap = torch.load(path, weights_only=False)
            dec.load_state_dict(snap["state"])
            rows = [r for c in admitted for r in proto["evaluation"][c]]
            x = dev["features"][[index[r["relative_path"]] for r in rows]]
            s256, img, grids = [], [], []
            for xb in x.split(10):
                en, de = dec(xb[:, 0].cuda(), xb[:, 1].cuda())
                sm, i = R.dinomaly(en, de)
                s256.append(sm.cpu()); img.append(i.cpu()); grids.append(R.grid(en, de).cpu())
            out = d(run) / f"maps/{tag}_stage{s}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            require(not out.exists(), f"exists: {out}")
            torch.save({"paths": [r["relative_path"] for r in rows], "s256": torch.cat(s256),
                        "image_RD": torch.cat(img), "grid": torch.cat(grids)}, out)
            record[f"{arm}/{seed}/{s}"] = {"maps": str(out.relative_to(D.ROOT)), "sha256": sha(out), "admitted": admitted}
            del dec
        D.write_json(target, record)
        print("scored", tag, len(record), flush=True)


def metrics(run):
    DR.configure()
    verify(run)
    proto = RB.verify_lock()
    w = DA.draws(proto)
    check = DA.verify_draws(proto, w)
    rows = {r["relative_path"]: r for c in C.EVAL for r in proto["evaluation"][c]}
    masks = {p: torch.from_numpy(DA.native_mask(r)) for p, r in rows.items()}
    for tag in ("nl", "identity"):
        src = d(run) / f"SCORES_{tag}.json"
        if not src.exists():
            continue
        target = d(run) / f"METRICS_{tag}.json"
        require(not target.exists(), "one-shot")
        points, reps = {}, {}
        for key, rec in D.read(src).items():
            path = D.ROOT / rec["maps"]
            require(sha(path) == rec["sha256"], "maps changed")
            mp = torch.load(path, weights_only=True)
            pos = {p: i for i, p in enumerate(mp["paths"])}
            pooled = {ro: ([], []) for ro in DA.READOUTS}
            for c in rec["admitted"]:
                group = proto["evaluation"][c]
                labels = [r["label"] for r in group]
                for ro in DA.READOUTS:
                    maps, image = [], []
                    for r in group:
                        i = pos[r["relative_path"]]
                        if ro == "RD":
                            maps.append(R.native(mp["s256"][i].cuda(), r["hw"]))
                            image.append(float(mp["image_RD"][i]))
                        else:
                            maps.append(torch.from_numpy(native_map(mp["grid"][i].flatten(), r["hw"])).cuda())
                            image.append(float(mp["grid"][i].max()))
                    targets = [masks[r["relative_path"]].cuda() for r in group]
                    m, b = DA.category_metrics(maps, targets, labels, image, w[c])
                    points[f"{key}/{c}/{ro}"], reps[f"{key}/{c}/{ro}"] = m, b
                    pooled[ro][0].extend(x.cpu() for x in maps)
                    pooled[ro][1].extend(t.cpu() for t in targets)
                    del maps, targets
            for ro in DA.READOUTS:
                points[f"{key}/pooled/{ro}"] = exact_ranking(torch.cat([x.flatten() for x in pooled[ro][0]]),
                                                             torch.cat([t.flatten() for t in pooled[ro][1]]))
            print("metrics", tag, key, {c: round(points[f"{key}/{c}/RD"]["pixel_ap"], 4) for c in rec["admitted"]},
                  flush=True)
        D.write_json(target, {"points": points, "draw_check_max_abs": check})
        D.write_json(d(run) / f"REPLICATES_{tag}.json", reps)


def compare(run):
    lk = verify(run)
    seed = lk["config"]["seed"]
    P, B = {}, {}
    for tag in ("nl", "identity"):
        P[tag] = D.read(d(run) / f"METRICS_{tag}.json")["points"]
        B[tag] = {k: {m: np.array(v) for m, v in x.items()} for k, x in D.read(d(run) / f"REPLICATES_{tag}.json").items()}
    k = lambda tag, s, c, ro="RD": f"{'NL' if tag == 'nl' else 'ID'}/{seed}/{s}/{c}/{ro}"
    res = {"categories": {}, "trajectory": {}}
    lo = {t: [] for t in ("nl", "identity")}
    hi = {t: [] for t in ("nl", "identity")}
    for c in C.EVAL:
        acq = P["identity"][k("identity", ACQ[c], c)]["pixel_ap"]
        fin = {t: P[t][k(t, FINAL, c)]["pixel_ap"] for t in ("nl", "identity")}
        for t in ("nl", "identity"):
            lo[t].append(B[t][k(t, FINAL, c)]["ap_low"])
            hi[t].append(B[t][k(t, FINAL, c)]["ap_high"])
        res["categories"][c] = {
            "acq_shared": acq, "final_nl": fin["nl"], "final_identity": fin["identity"],
            "nl_minus_identity": fin["nl"] - fin["identity"],
            "nl_minus_identity_ci95_images": DA.diff_ci(lo["nl"][-1], hi["nl"][-1], lo["identity"][-1], hi["identity"][-1]),
            "F_nl": acq - fin["nl"], "F_identity": acq - fin["identity"],
            "image_auroc_final": {t: P[t][k(t, FINAL, c)]["image_auroc"] for t in ("nl", "identity")},
            "RP1_final": {t: P[t][k(t, FINAL, c, "RP1")]["pixel_ap"] for t in ("nl", "identity")}}
        res["trajectory"][c] = {t: {s: P[t][k(t, s, c)]["pixel_ap"] for s in range(max(ACQ[c], lk["config"]["start"] + 1), FINAL + 1)
                                    if k(t, s, c) in P[t]} for t in ("nl", "identity")}
    mean = lambda xs: np.mean(xs, axis=0)
    res["macro"] = {"final_nl": float(np.mean([res["categories"][c]["final_nl"] for c in C.EVAL])),
                    "final_identity": float(np.mean([res["categories"][c]["final_identity"] for c in C.EVAL])),
                    "nl_minus_identity_ci95_images": DA.diff_ci(mean(lo["nl"]), mean(hi["nl"]), mean(lo["identity"]), mean(hi["identity"])),
                    "Fbar_nl": float(np.mean([res["categories"][c]["F_nl"] for c in C.EVAL])),
                    "Fbar_identity": float(np.mean([res["categories"][c]["F_identity"] for c in C.EVAL]))}
    res["macro"]["nl_minus_identity"] = res["macro"]["final_nl"] - res["macro"]["final_identity"]
    res["pooled"] = {t: P[t][f"{'NL' if t == 'nl' else 'ID'}/{seed}/{FINAL}/pooled/RD"]["pixel_ap"] for t in ("nl", "identity")}
    res["note"] = "One seed: descriptive only; image-bootstrap intervals are not training-seed variability."
    D.write_json(d(run) / "COMPARE.json", res)
    print("macro", {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res["macro"].items()}, "pooled", res["pooled"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("lock", "score", "metrics", "compare"))
    ap.add_argument("--run", default="N_per_task_seed2711_from6")
    ap.add_argument("--identity-too", action="store_true")
    a = ap.parse_args()
    {"lock": lambda: lock(a.run), "score": lambda: score(a.run, a.identity_too),
     "metrics": lambda: metrics(a.run), "compare": lambda: compare(a.run)}[a.stage]()


if __name__ == "__main__":
    main()
