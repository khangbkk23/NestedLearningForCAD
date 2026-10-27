"""Offline NLCAD-05A ranking audit on the existing frozen synthetic maps.

This is deliberately separate from the recorder.  It reads only the existing
label-free ID maps and BANK_seed2711 masks, and computes the preregistered
within-image (A) and legal historical cross-image (B) summaries without
running a model or touching any checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from exps.dinomaly import common as D
from exps.nlcad import bank as BK
from exps.nlcad import forensic_syn as FS
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
TARGET = OUT / "RETENTION_AUDIT_EXISTING.json"
SEED, START, FINAL = 2711, 6, 9


def metrics(rows, maps, items):
    pos, neg, per, margins = [], [], [], []
    for j, it in enumerate(items):
        m = maps[j].float().numpy()
        p, n = m[it["coverage"].numpy() >= .5], m[it["coverage"].numpy() == 0]
        if len(p) and len(n):
            pos.append(p); neg.append(n)
            per.append(float(average_precision_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])))
            margins.append(float(p.mean() - n.mean()))
    if not pos:
        return {"pooled_ap": None, "per_image_ap": None, "background_p99": None, "ranking_margin": None, "items": 0}
    p, n = np.concatenate(pos), np.concatenate(neg)
    return {"pooled_ap": float(average_precision_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])),
            "per_image_ap": float(np.mean(per)), "background_p99": float(np.quantile(n, .99)),
            "ranking_margin": float(np.mean(margins)), "items": len(per)}


def audit_cell(bank, hosts, n, stage, gen, src, maps):
    items = FS.items_at(bank, hosts, n, stage)[gen][src]
    within = metrics([], maps, items)
    cats = sorted({it["host"][0] for it in items})
    cross = {"per_category": {}, "eligible_categories": [], "ineligible_categories": []}
    for cat in cats:
        cat_items = [it for it in items if it["host"][0] == cat]
        indices = [j for j, it in enumerate(items) if it["host"][0] == cat]
        hs = sorted({tuple(it["host"]) for it in cat_items})
        if len(hs) < 2:
            cross["ineligible_categories"].append(cat)
            cross["per_category"][cat] = {"distinct_hosts": len(hs), "eligible": False}
            continue
        chosen = hs[:8]
        take = [j for j, it in zip(indices, cat_items) if tuple(it["host"]) in chosen]
        vals = []
        for j in take:
            it = items[j]; m = maps[j].float().numpy()
            p, b = m[it["coverage"].numpy() >= .5], m[it["coverage"].numpy() == 0]
            if len(p) and len(b): vals.append((p, b))
        pos = np.concatenate([x[0] for x in vals]); bg = np.concatenate([x[1] for x in vals])
        hard = np.sort(bg)[-min(64, len(bg)):]
        per = [float(average_precision_score(np.r_[np.ones(len(p)), np.zeros(len(b))], np.r_[p, b])) for p, b in vals]
        cross["eligible_categories"].append(cat)
        cross["per_category"][cat] = {"distinct_hosts": len(chosen), "hosts": [list(x) for x in chosen],
            "positive_patches": len(pos), "hard_negative_patches": len(hard),
            "pooled_ap": float(average_precision_score(np.r_[np.ones(len(pos)), np.zeros(len(hard))], np.r_[pos, hard])),
            "per_image_ap": float(np.mean(per)), "background_p99": float(np.quantile(bg, .99)),
            "ranking_margin": float(pos.mean() - hard.mean())}
    return {"within_image_A": within, "cross_image_B": cross,
            "eligibility": {c: {"distinct_hosts": len({tuple(x["host"]) for x in items if x["host"][0] == c}),
                                 "source": "legal_allowed_hosts"} for c in cats}}


def run():
    proto = RB.verify_lock(); n = R1.n_train(proto); bank, hosts = BK.load_bank(SEED)
    result = {"source": "existing FORENSIC_SYN ID maps; no new forward pass", "stages": {}}
    for s in range(START + 1, FINAL + 1):
        maps = torch.load(FS.CELLS / f"MAPS_ID_{s}.pt", weights_only=True)["end"]
        result["stages"][str(s)] = {}
        for gen in BK.GENERATORS:
            for src in ("current", "replay"):
                result["stages"][str(s)][f"{gen}/{src}"] = audit_cell(bank, hosts, n, s, gen, src, maps[f"{gen}/{src}"])
    D.write_json(TARGET, result)
    print("wrote", TARGET, flush=True)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); run()
