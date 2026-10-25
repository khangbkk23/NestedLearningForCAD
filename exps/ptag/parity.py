# exps/ptag/parity.py
"""PTAG-01 scorer parity: re-score development NORMAL images and compare with stored frozen-detector scores.

Only label-0 identities are loaded; no defect image, defect feature or mask is opened.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from exps.calm.decompose import build_cells
from exps.calm.load import CAT_INDEX, load_state
from exps.calm02.score import Scorer
from exps.ptag import common as C
from exps.ptag import scoring as S

TOL = {"nn64": 1e-9, "calm": 1e-9, "ntm": 1e-9, "p2": 1e-4}


def p1_normals():
    """P1 evaluation feature cache, label-0 rows only (identity checked through the sidecar)."""
    out = {}
    for side in sorted((C.P1 / "evaluation_features").glob("*.json")):
        meta = json.loads(side.read_text())
        src = meta["source"]
        if int(src["label"]) != 0:
            continue
        x = torch.load(side.with_suffix(".pt"), map_location="cpu", weights_only=True)
        out[src["relative_path"]] = x
    return out


def cohort():
    rows = C.read(C.P1 / "protocol_resolved.json")["evaluations"]
    return {c: [r for r in rows if r["category"] == c] for c in C.CATS}


def calm_normals():
    out = {}
    for c in C.CATS:
        obj = torch.load(C.CALM / f"features/mvtec/{c}_test.pt", map_location="cpu", weights_only=False, mmap=True)
        idx = {r["relative_path"]: i for i, r in enumerate(obj["rows"])}
        out[c] = {r["relative_path"]: obj["features"][idx[r["relative_path"]]].clone()
                  for r in cohort()[c] if r["label"] == 0}
    return out


def run_parity(man, device):
    err = {k: 0.0 for k in TOL}
    count = {k: 0 for k in TOL}
    feats = p1_normals()
    for t, cats in C.P1_SEEN.items():
        recs = C.read(C.P1 / f"predictions_t{t}.json")
        for a in C.P1_ARMS:
            nn = S.NN64(S.nn64_bank(["p1", a, t]), device)
            for r in recs:
                rel = r["source"]["relative_path"]
                if r["arm"] == a and rel in feats:
                    d = nn(feats[rel][None])[0].numpy()
                    err["nn64"] = max(err["nn64"], float(np.abs(d - np.asarray(r["patch_distances"])).max()))
                    count["nn64"] += 1
    groups = C.read(C.C00 / "LOCK.json")["constants"]["groups"]
    for a in C.C00_ARMS:
        for g in groups:
            c, acq, T = g["cat"], g["acq"], g["T"]
            z = np.load(C.C00 / "scores" / f"{a}__{c}@{T}.npz")
            rels = sorted(r["relative_path"] for r in cohort()[c])          # CALM-00A row order (load_eval)
            cells = build_cells(load_state(acq, a), load_state(T, a), CAT_INDEX[c])
            for name, cell in cells.items():
                nn = S.NN64(cell["bank"], device)
                for i, rel in enumerate(rels):
                    if rel in feats:
                        d = nn(feats[rel][None])[0].numpy()
                        err["nn64"] = max(err["nn64"], float(np.abs(d - z[f"{name}_d1"][i]).max()))
                        count["nn64"] += 1
    cal = calm_normals()
    for a in C.CALM_ARMS:
        z = np.load(C.CALM / f"gate_mvtec_b1/{a}_scores.npz")
        for c in C.CATS:
            for k in (C.CALM_ACQ[c], C.CALM_FINAL):
                rows, _, _ = S.calm_bank(a, k)
                sc = Scorer(rows, device)
                for i, r in enumerate(cohort()[c]):
                    if r["label"] == 0:
                        d, _ = sc.patch(cal[c][r["relative_path"]])
                        err["calm"] = max(err["calm"], float(np.abs(d.numpy() - z[f"{c}_{k:02d}"][i]).max()))
                        count["calm"] += 1
    prep = np.load(C.NTM / "prepared.npz")
    for c in C.CATS:
        rels = [r["relative_path"] for r in cohort()[c]]
        normal = [i for i, r in enumerate(cohort()[c]) if r["label"] == 0]
        dx = S.context(torch.stack([cal[c][rels[i]] for i in normal]), device).numpy()
        err["ntm"] = max(err["ntm"], float(np.abs(dx - prep[f"dX_{c}"][normal]).max()))
        count["ntm"] += len(normal)
    for x in man["cells"]:
        sc = x["scorer"]
        if sc["kind"] != "p2" or sc["seed"] != 2711 or sc["order"] != "canonical" or sc["boundary"] not in (100, 350):
            continue
        if x["category"] != "bottle":                                     # one evaluation per state
            continue
        recs = C.read(C.ROOT / f"results/hope_cad/task_p/{sc['dir']}/canonical_seed2711/predictions.json")
        learner = S.p2_learner(sc, device)
        for r in recs:
            if r["arm"] == sc["arm"] and r["boundary"] == sc["boundary"] and r["source"] in feats \
                    and r["state_sha256"] == sc["state_sha256"]:
                s = S.p2_scores(learner, feats[r["source"]][None], device)[0].numpy()
                err["p2"] = max(err["p2"], float(np.abs(s - np.asarray(r["patch_scores"])).max()))
                count["p2"] += 1
    ok = all(count[k] > 0 and err[k] <= TOL[k] for k in TOL)
    return {"max_abs": err, "comparisons": count, "tolerance": TOL, "pass": ok,
            "scope": "development NORMAL images only; stored frozen-detector patch scores"}
