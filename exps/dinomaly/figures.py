# exps/dinomaly/figures.py
"""Descriptive failure heatmaps for DRF-01 reports (post-verdict; enters no decision).

  python -m exps.dinomaly.figures gate0
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from exps.dinomaly import analysis as A
from exps.dinomaly import common as C
from exps.dinomaly import readout as R
from exps.dinomaly.common import require

SEEDS = (2711, 2712, 2713)


def worst_defects(points, key, rows, k):
    """Defect images with the lowest seed-mean per-image native AP."""
    per = np.array([[np.nan if v is None else v for v in points[key.format(s=s)]["separation"]["per_image_ap_defects"]]
                    for s in SEEDS]).mean(0)
    order = [i for i in np.argsort(np.nan_to_num(per, nan=np.inf)) if rows[i]["label"] == 1][:k]
    return [(i, float(per[i])) for i in order]


def gate0(k=2):
    proto = C.identities()
    points = C.read(C.OUT / "gate0" / "METRICS.json")["points"]
    maps = torch.load(C.OUT / "gate0" / "maps" / f"C_seed{SEEDS[0]}_stage1.pt", weights_only=False)
    index = {p: i for i, p in enumerate(maps["paths"])}
    picks = [(c, i, ap) for c in C.CATS
             for i, ap in worst_defects(points, "C/{s}/1/" + c + "/RD", proto["evaluation"][c], k)]
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    for row, (c, i, ap) in zip(axes, picks):
        r = proto["evaluation"][c][i]
        with Image.open(C.source(r)) as im:
            img = np.array(im.convert("RGB"))
        mask = A.native_mask(r)
        s = R.native(maps["s256"][index[r["relative_path"]]], r["hw"]).numpy()
        row[0].imshow(img); row[1].imshow(mask, cmap="gray")
        row[2].imshow(s, cmap="jet", vmin=0, vmax=max(float(s.max()), 1e-6))
        row[0].set_title(f"{r['relative_path']}", fontsize=8)
        row[1].set_title("native mask", fontsize=8)
        row[2].set_title(f"R_D seed {SEEDS[0]} | seed-mean image AP {ap:.3f}", fontsize=8)
        for a in row:
            a.axis("off")
    fig.tight_layout()
    out = C.OUT / "gate0" / "figures"
    out.mkdir(exist_ok=True)
    path = out / "failure_heatmaps.png"
    require(not path.exists(), "figure exists (create-exclusive)")
    fig.savefig(path, dpi=110)
    C.write_json(out / "FIGURES.json", {"failure_heatmaps": {"path": str(path.relative_to(C.ROOT)), "sha256": C.sha(path),
                                                             "picks": [{"category": c, "relative_path": proto["evaluation"][c][i]["relative_path"],
                                                                        "seed_mean_image_ap": ap} for c, i, ap in picks]}})
    print(json.dumps([(c, proto["evaluation"][c][i]["relative_path"], round(ap, 3)) for c, i, ap in picks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gate0",))
    ap.parse_args()
    gate0()


if __name__ == "__main__":
    main()
