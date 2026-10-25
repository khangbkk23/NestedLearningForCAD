# exps/ptag/common.py
"""PTAG-01 constants, identities and the detector-cell manifest. Opens no mask and no defect image."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p0_normal_support import require

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/ptag"
DATA = ROOT / "data/mvtec"
TASK = ROOT / "agents/tasks/task_ptag_01_alignment_gate.md"
P = ROOT / "results/hope_cad/task_p"
P1 = P / "p1"
C00 = ROOT / "results/hope_cad/calm/calm00a_20261023"
CALM = ROOT / "results/hope_cad/calm02"
NTM = ROOT / "results/hope_cad/ntm/stage0"

CATS = ("bottle", "carpet", "hazelnut")
SEED = 20261023
HOSTS, VARIANTS, MAX_ATTEMPTS = 20, 5, 20
FAMILIES = ("Q1", "Q2", "Q3")                 # synthetic families, in image-index order after the clean hosts
PROXIES = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
REPS, BINS = 1000, 65536
P1_ARMS = ("FULL784_GREEDY", "FULL784_RESERVOIR", "GRID16_GREEDY")
P1_SEEN = {100: ("bottle",), 200: ("bottle", "carpet"), 300: CATS, 350: CATS}
C00_ARMS = ("FULL784_GREEDY", "FULL784_RESERVOIR")
CALM_ARMS = ("CALM_L1", "CALM_L2")
CALM_ACQ, CALM_FINAL = {"bottle": 0, "carpet": 3, "hazelnut": 5}, 14
NTM_ARMS = ("BANK", "CTX", "NTM_FLOOR", "NTM_MIN")
P2_DIRS = ("p2", "p2d", "p2e", "p2f")
MVTEC_ORDER = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
               "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper")


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=True)


def sample_seed(*parts) -> int:
    return int(np.random.SeedSequence([SEED, *[int(p) for p in parts]]).generate_state(1)[0])


# ---------------------------------------------------------------- hosts

def train_paths(cat):
    return sorted(p.relative_to(DATA).as_posix() for p in (DATA / cat / "train" / "good").glob("*.png"))


def p_family_training_identities():
    """Every training image any P1 / CALM-00A / P2-family detector could have learned from."""
    used = {r["relative_path"] for r in read(P / "p0c/protocol_resolved.json")["updates"]}
    used |= {r["relative_path"] for r in read(P1 / "protocol_resolved.json")["updates"]}
    for d in P2_DIRS:
        proto = P / d / "protocol_resolved.json"
        if proto.exists():
            for rows in read(proto).get("updates_by_order", {}).values():
                used |= {r["relative_path"] for r in rows}
        for name in ("normal_access.json", "replay_access.json"):
            for f in (P / d).glob(f"*_seed*/{name}"):
                text = f.read_text()
                for c in CATS:
                    used |= set(re.findall(rf"{c}/train/good/\d+\.png", text))
    return used


def select_hosts():
    used = p_family_training_identities()
    sealed = set(read(P1 / "protocol_resolved.json")["sealed_identity_paths"])
    out = {}
    for i, c in enumerate(CATS):
        pool = [p for p in train_paths(c) if p not in used]
        require(len(pool) >= HOSTS, f"host pool too small for {c}: {len(pool)}")
        pick = sorted(np.random.default_rng(SEED + i).permutation(pool)[:HOSTS].tolist())
        require(all("/train/good/" in p and p not in sealed for p in pick), "host provenance violation")
        out[c] = {"pool_size": len(pool), "hosts": pick, "sha256": [sha(DATA / p) for p in pick]}
    return out


def calm_global_index():
    """Global training-image index used by CALM-02 row serials (serial // 784), as in exps/ntm/stage0."""
    off, index = 0, {}
    for t in MVTEC_ORDER:
        for j, p in enumerate(train_paths(t)):
            index[p] = off + j
        off += len(train_paths(t))
    return index


# ---------------------------------------------------------------- cells

def _p1_cells():
    summ, boot = read(P1 / "summary.json")["per_boundary_category"], P1 / "bootstrap.json"
    cells = []
    for t, cats in P1_SEEN.items():
        for c in cats:
            for a in P1_ARMS:
                key = f"{t}/{c}/{a}"
                cells.append({"id": f"P1/{a}/t{t}/{c}", "family": "P1", "category": c, "split": "selection",
                              "real_ap": summ[key]["pixel_ap"],
                              "real_source": [str((P1 / "summary.json").relative_to(ROOT)), key],
                              "rep_source": [str(boot.relative_to(ROOT)), key, "ap_lower", "ap_upper"],
                              "scorer": {"kind": "nn64", "bank": ["p1", a, t]}})
    return cells


def _c00_cells():
    lock, audit = read(C00 / "LOCK.json"), read(C00 / "audit.json")["groups"]
    cells, seen = {}, {}
    for a in C00_ARMS:
        for g in lock["constants"]["groups"]:
            c, acq, T = g["cat"], g["acq"], g["T"]
            ap = audit[f"{a}__{c}@{T}"]["ap"]
            ids = {"00": f"00/a{acq}", "11": f"11/T{T}", "10": f"10/a{acq}T{T}", "01": f"01/a{acq}T{T}",
                   "oracle": f"oracle/T{T}"}
            for name, tag in ids.items():
                cid = f"C00/{a}/{tag}/{c}"
                if cid in cells:
                    require(abs(cells[cid]["real_ap"] - ap[name]) < 1e-12, f"inconsistent duplicate {cid}")
                    continue
                cells[cid] = {"id": cid, "family": "C00", "category": c, "split": "selection", "real_ap": ap[name],
                              "real_source": [str((C00 / "audit.json").relative_to(ROOT)), f"{a}__{c}@{T}", name],
                              "rep_source": None,
                              "scorer": {"kind": "nn64", "bank": ["c00", a, acq, T, name, c]}}
    return list(cells.values())


def _calm_cells():
    ev = read(CALM / "gate_mvtec_b1/GATE_EVAL.json")["arms"]
    cells = []
    for a in CALM_ARMS:
        log = read(CALM / f"runs/mvtec/{a}_B1/train_log.json")["tasks"]
        for c in CATS:
            for k in range(CALM_ACQ[c], CALM_FINAL + 1):
                cells.append({"id": f"CALM/{a}/k{k:02d}/{c}", "family": "CALM", "category": c, "split": "selection",
                              "real_ap": ev[a]["ap_trajectory"][c][str(k)],
                              "real_source": ["results/hope_cad/calm02/gate_mvtec_b1/GATE_EVAL.json", a, c, str(k)],
                              "rep_source": None, "deficit": float(log[k]["deficit"][c]),
                              "scorer": {"kind": "calm", "arm": a, "k": k}})
    return cells


def _ntm_cells():
    ev = read(NTM / "EVAL.json")["arms"]
    cells = []
    for c in CATS:
        for a in NTM_ARMS:
            ks = (CALM_ACQ[c],) if a == "CTX" else (CALM_ACQ[c], CALM_FINAL)
            if a == "CTX":
                require(abs(ev[a]["acquisition_ap"][c] - ev[a]["final_ap"][c]) < 1e-12, "CTX must be bank-free")
            for k in ks:
                field = "acquisition_ap" if k == CALM_ACQ[c] else "final_ap"
                cid = f"NTM/CTX/{c}" if a == "CTX" else f"NTM/{a}/k{k:02d}/{c}"
                cells.append({"id": cid, "family": "NTM", "category": c, "split": "selection",
                              "real_ap": ev[a][field][c],
                              "real_source": ["results/hope_cad/ntm/stage0/EVAL.json", a, field, c],
                              "rep_source": None, "scorer": {"kind": "ntm", "arm": a, "k": k}})
    return cells


def _p2_cells():
    owner, cells = {}, {}
    for d in P2_DIRS:
        pm = read(P / d / "point_metrics.json")
        for f in sorted((P / d).glob("*_seed*/state_t*.pt"),        # canonical first: shared t100 states keep it
                        key=lambda f: (f.parent.name.startswith("alternate"), str(f))):
            st = torch.load(f, map_location="cpu", weights_only=False)
            order, seed = f.parent.name.split("_seed")
            b = int(st["boundary"])
            for arm, h in st["state_sha256"].items():
                if h in owner:
                    continue
                owner[h] = (d, order, int(seed), b, arm)
                for c in CATS:
                    key = f"{order}/{seed}/{b}/{c}/{arm}"
                    if key not in pm:
                        continue
                    cid = f"P2/{arm}/{order}/{seed}/t{b}/{c}"
                    cells[cid] = {"id": cid, "family": "P2", "category": c,
                                  "split": "selection" if (order == "canonical" or b == 100) else "heldout",
                                  "real_ap": pm[key]["pixel_ap"],
                                  "real_source": [f"results/hope_cad/task_p/{d}/point_metrics.json", key],
                                  "rep_source": [f"results/hope_cad/task_p/{d}/bootstrap_samples.json", key,
                                                 "ap_low", "ap_high"],
                                  "scorer": {"kind": "p2", "dir": d, "order": order, "seed": int(seed), "boundary": b,
                                             "arm": arm, "state_file": str(f.relative_to(ROOT)),
                                             "state_sha256": h}}
    # aliases: the same state reached under the other order (t100) keeps its first id
    alias = {}
    for d in P2_DIRS:
        pm = read(P / d / "point_metrics.json")
        for key, v in pm.items():
            order, seed, b, c, arm = key.split("/")
            cid = f"P2/{arm}/{order}/{seed}/t{b}/{c}"
            if cid in cells:
                require(abs(cells[cid]["real_ap"] - v["pixel_ap"]) < 1e-12, f"P2 real AP disagrees across dirs: {key}")
            else:
                twin = f"P2/{arm}/canonical/{seed}/t{b}/{c}"
                require(int(b) == 100 and twin in cells, f"P2 metric without a state: {key}")
                require(abs(cells[twin]["real_ap"] - v["pixel_ap"]) < 1e-12, f"t100 twin disagrees: {key}")
                alias[cid] = twin
    return list(cells.values()), alias


def build_manifest():
    p2, alias = _p2_cells()
    cells = _p1_cells() + _c00_cells() + _calm_cells() + _ntm_cells() + p2
    ids = [c["id"] for c in cells]
    require(len(ids) == len(set(ids)), "duplicate cell ids")
    return {"cells": cells, "p2_alias": alias}


# ---------------------------------------------------------------- contrasts (cluster level)

def contrasts(manifest, split):
    """Pre-registered contrasts as lists of (reference, new) cell-id pairs, averaged within the cluster."""
    have = {c["id"]: c for c in manifest["cells"]}
    alias = manifest["p2_alias"]

    def cid(x):
        x = alias.get(x, x)
        return x if x in have and have[x]["split"] == split else None

    out = []
    if split == "selection":
        for c in CATS:
            out.append(("P1", "RESERVOIR-GREEDY", c, [(f"P1/FULL784_GREEDY/t{t}/{c}", f"P1/FULL784_RESERVOIR/t{t}/{c}")
                                                      for t, cats in P1_SEEN.items() if c in cats]))
        groups = read(C00 / "LOCK.json")["constants"]["groups"]
        for a in C00_ARMS:
            for c in CATS:
                gs = [g for g in groups if g["cat"] == c]
                for new in ("10", "01"):
                    out.append(("C00", f"{new}-00 {a}", c,
                                [(f"C00/{a}/00/a{g['acq']}/{c}", f"C00/{a}/{new}/a{g['acq']}T{g['T']}/{c}") for g in gs]))
        for c in CATS:
            out.append(("CALM", "L2-L1", c, [(f"CALM/CALM_L1/k{k:02d}/{c}", f"CALM/CALM_L2/k{k:02d}/{c}")
                                             for k in range(CALM_ACQ[c], CALM_FINAL + 1)]))
        for c in CATS:
            for a in ("CTX", "NTM_FLOOR", "NTM_MIN"):
                pairs = []
                for k in (CALM_ACQ[c], CALM_FINAL):
                    new = f"NTM/CTX/{c}" if a == "CTX" else f"NTM/{a}/k{k:02d}/{c}"
                    pairs.append((f"NTM/BANK/k{k:02d}/{c}", new))
                out.append(("NTM", f"{a}-BANK", c, pairs))
    p2_pairs = (("B-D", "D", "B"), ("D2-C2", "C2", "D2"), ("E-C2", "C2", "E"), ("F-E", "E", "F"))
    seeds = sorted({c["scorer"]["seed"] for c in manifest["cells"] if c["family"] == "P2"})
    orders = ("canonical",) if split == "selection" else ("alternate",)
    for name, ref, new in p2_pairs:
        for c in CATS:
            pairs = []
            for o in orders:
                for s in seeds:
                    for b in (100, 200, 300, 350):
                        if split == "heldout" and b == 100:
                            continue
                        r, n = cid(f"P2/{ref}/{o}/{s}/t{b}/{c}"), cid(f"P2/{new}/{o}/{s}/t{b}/{c}")
                        if r and n:
                            pairs.append((r, n))
            out.append(("P2", name, c, pairs))
    for fam, name, c, pairs in out:
        require(pairs and all(a in have and b in have for a, b in pairs), f"bad contrast {fam} {name} {c}")
    return out
