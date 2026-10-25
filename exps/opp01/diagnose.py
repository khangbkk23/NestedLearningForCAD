# exps/opp01/diagnose.py
"""OPP-01 diagnostics on the frozen DRF-01 artifacts: normal features only, forward passes only, no labels.

  python -m exps.opp01.diagnose o1   # replay coverage vs over-fit
  python -m exps.opp01.diagnose o2   # forgetting anatomy in parameter space (CPU)
  python -m exps.opp01.diagnose o3   # oracle compressibility of the replay correction
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from exps.dinomaly import common as C
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly import run
from exps.dinomaly.common import require

SRC = C.OUT
OUT = C.ROOT / "results/hope_cad/opp01"
SEEDS = run.SEEDS
RANKS = (1, 4, 16, 64)
MODULES = ["bottleneck.0"] + [f"decoder.{i}" for i in range(8)]


def frozen():
    """Every artifact read here must match the DRF-01 freeze manifest."""
    return C.read(SRC / "FROZEN_BASELINE.json")["results_files"]


def state(arm, seed, stage, man):
    rec = C.read(SRC / f"runs/{arm}_seed{seed}/TRAIN.json")
    st = next(s for s in rec["stages"] if s["index"] == stage)
    path = C.ROOT / st["checkpoint"]
    require(C.sha(path) == st["checkpoint_sha256"] == man[st["checkpoint"]]["sha256"], f"checkpoint not frozen: {path}")
    return torch.load(path, weights_only=True)["state"]


def features(man):
    out = {}
    for split in ("train", "heldout"):
        for c in C.CATS:
            key = f"results/hope_cad/dinomaly/features/{split}_{c}.pt"
            require(C.sha(C.ROOT / key) == man[key]["sha256"], f"feature cache not frozen: {key}")
            out[split, c] = run.load_split(split, c)
    return out


def buffer_index(proto, seed, done):
    """Exact re-derivation of run.replay_buffer indices (checked against TRAIN.json identities by the caller)."""
    gen = torch.Generator().manual_seed(seed + 9000)
    perms = {c: torch.randperm(len(proto["train"][c]), generator=gen).tolist() for c in C.ORDER}
    k = run.REPLAY_CAPACITY // len(done)
    return {c: perms[c][:k] for c in done}


def buffers(proto, seed):
    """B's buffer at stage 2 (carpet) and stage 3 (hazelnut), verified against the recorded identities."""
    rec = C.read(SRC / f"runs/B_seed{seed}/TRAIN.json")
    out = {}
    for st in rec["stages"]:
        done = list(C.ORDER[:C.ORDER.index(st["name"])])
        idx = buffer_index(proto, seed, done)
        ids = [proto["train"][c][i]["relative_path"] for c in done for i in idx[c]]
        require(ids == st["replay"]["identities"], f"buffer re-derivation mismatch seed {seed} stage {st['index']}")
        out[st["index"]] = idx
    return out


def decoder(sd):
    dec = O.Decoder(768, 12).cuda().eval()
    dec.load_state_dict(sd)
    return dec


@torch.no_grad()
def per_image(dec, f):
    """Per-image mean patch 1-cos, official global cosine (per image) and R_D normal image score."""
    m, g, s = [], [], []
    for xb in f.split(10):
        xb = xb.cuda()
        en, de = dec(xb[:, 0], xb[:, 1])
        m += R.grid(en, de).mean(dim=(1, 2)).tolist()
        g += torch.stack([1 - F.cosine_similarity(e.flatten(1), d.flatten(1)) for e, d in zip(en, de)], 1).mean(1).tolist()
        s += R.dinomaly(en, de)[1].tolist()
    return np.array(m), np.array(g), np.array(s)


def summary(x):
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "n": int(len(x))} if len(x) else None


# --- O1 ----------------------------------------------------------------------------------------------


def o1():
    run.configure()
    proto = run.verify_lock()
    man = frozen()
    feats = features(man)
    res = {}
    for seed in SEEDS:
        buf = buffers(proto, seed)
        cks = [("C", 1), ("A", 1), ("A", 2), ("A", 3), ("B", 2), ("B", 3)]
        for arm, stage in cks:
            dec = decoder(state(arm, seed, stage, man))
            for c in C.CATS:
                tr = feats["train", c]
                m_tr, g_tr, s_tr = per_image(dec, tr)
                m_ho, g_ho, s_ho = per_image(dec, feats["heldout", c])
                entry = {"heldout": {"mcos": summary(m_ho), "gcos": summary(g_ho), "score_RD": summary(s_ho)},
                         "train_all": {"mcos": summary(m_tr), "gcos": summary(g_tr)}}
                # Buffer split uses B's buffers of the same seed for every arm, so A/C give the "not special" reference.
                for bstage, idx in buf.items():
                    inb = np.zeros(len(tr), bool)
                    inb[idx.get(c, [])] = True
                    if inb.any():
                        entry[f"bufB{bstage}"] = {"in": {"mcos": summary(m_tr[inb]), "gcos": summary(g_tr[inb])},
                                                  "out": {"mcos": summary(m_tr[~inb]), "gcos": summary(g_tr[~inb])}}
                res[f"{arm}/{seed}/{stage}/{c}"] = entry
            del dec
            torch.cuda.empty_cache()
            print("o1", arm, seed, stage, {c: round(res[f'{arm}/{seed}/{stage}/{c}']['heldout']['gcos']['mean'], 4)
                                          for c in C.CATS}, flush=True)
    C.write_json(OUT / "O1_coverage.json", res)


# --- O2 ----------------------------------------------------------------------------------------------


def module_of(name):
    return next(m for m in MODULES if name.startswith(m + "."))


def svd_stats(delta, prev):
    s = torch.linalg.svdvals(delta.double())
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    rank_at = {f"rank{int(q * 100)}": int((e < q).sum()) + 1 for q in (.5, .9, .99)}
    return {"rel_drift": float(delta.norm() / prev.norm()), "fro2": float((s ** 2).sum()),
            "energy_at": {r: float(e[min(r, len(e)) - 1]) for r in RANKS}, **rank_at,
            "sq_sv": (s ** 2).tolist()[:64]}


def o2():
    man = frozen()
    res = {}
    for seed in SEEDS:
        sd = {(a, s): state(a, seed, s, man) for a, s in (("A", 1), ("A", 2), ("A", 3), ("B", 2), ("B", 3))}
        deltas = {"A2-A1": (sd["A", 2], sd["A", 1]), "A3-A2": (sd["A", 3], sd["A", 2]),
                  "B2-A1": (sd["B", 2], sd["A", 1]), "B3-B2": (sd["B", 3], sd["B", 2]), "B3-A3": (sd["B", 3], sd["A", 3])}
        vec = {k: torch.cat([(x[n] - y[n]).flatten().double() for n in x]) for k, (x, y) in deltas.items()}
        cos = lambda a, b: float(vec[a] @ vec[b] / (vec[a].norm() * vec[b].norm()))
        out = {"cosine": {"A2-A1|A3-A2": cos("A2-A1", "A3-A2"), "B2-A1|B3-B2": cos("B2-A1", "B3-B2"),
                          "A2-A1|B2-A1": cos("A2-A1", "B2-A1"), "A3-A2|B3-B2": cos("A3-A2", "B3-B2"),
                          "B3-A3|A3-A2": cos("B3-A3", "A3-A2"), "B3-A3|A2-A1": cos("B3-A3", "A2-A1")},
               "global_rel_drift": {k: float(vec[k].norm() / torch.cat([y[n].flatten().double() for n in y]).norm())
                                    for k, (x, y) in deltas.items()}}
        for k, (x, y) in deltas.items():
            mats, vec1 = {}, 0.
            for n in x:
                d = (x[n] - y[n]).float()
                if d.dim() == 2:
                    mats[n] = svd_stats(d, y[n].float())
                else:
                    vec1 += float(d.double().norm() ** 2)
            tot = sum(v["fro2"] for v in mats.values()) + vec1
            captured = {r: (sum(v["fro2"] * v["energy_at"][r] for v in mats.values()) + vec1) / tot for r in RANKS}
            per_module = {m: sum(v["fro2"] for n, v in mats.items() if module_of(n) == m) / tot for m in MODULES}
            out[k] = {"matrices": mats, "energy_captured_at_rank": captured, "vector_param_energy_share": vec1 / tot,
                      "module_energy_share": per_module}
            print("o2", seed, k, "drift", round(out["global_rel_drift"][k], 3),
                  {r: round(captured[r], 3) for r in RANKS}, flush=True)
        res[str(seed)] = out
    C.write_json(OUT / "O2_anatomy.json", res)


# --- O3 ----------------------------------------------------------------------------------------------


def factor_bytes(sd, r):
    """FP32 bytes of rank-r factors for every weight matrix plus all 1-D parameters taken in full."""
    return sum(r * sum(v.shape) * 4 if v.dim() == 2 else v.numel() * 4 for v in sd.values())


def o3():
    run.configure()
    proto = run.verify_lock()
    man = frozen()
    feats = features(man)
    res = {}
    for seed in SEEDS:
        buf3 = buffers(proto, seed)[3]
        evalsets = {}
        for c in C.CATS:
            keep = np.ones(len(feats["train", c]), bool)
            keep[buf3.get(c, [])] = False
            evalsets[c] = {"heldout": feats["heldout", c], "train_nonbuffer": feats["train", c][torch.from_numpy(keep)]}
        a3, b3 = state("A", seed, 3, man), state("B", seed, 3, man)
        svd = {n: torch.linalg.svd((b3[n] - a3[n]).double(), full_matrices=False) for n in a3 if a3[n].dim() == 2}

        def lowrank(r):
            sd = {n: (b3[n] if v.dim() == 1 else v) for n, v in a3.items()}
            for n, (U, S, Vh) in svd.items():
                k = S.numel() if r is None else r
                sd[n] = (a3[n].double() + (U[:, :k] * S[:k]) @ Vh[:k]).float()
            return sd

        variants = {"A3": a3, "B3": b3, "A3+1d": lowrank(0)}
        variants.update({f"A3+r{r}": lowrank(r) for r in RANKS})
        variants["A3+full"] = lowrank(None)
        for m in MODULES:
            variants[f"A3<-B3[{m}]"] = {n: (b3[n] if n.startswith(m + ".") else v) for n, v in a3.items()}
        variants["C"] = state("C", seed, 1, man)
        out = {}
        for name, sd in variants.items():
            dec = decoder(sd)
            out[name] = {c: {k: dict(zip(("mcos", "gcos", "score_RD"), [summary(x) for x in per_image(dec, f)]))
                             for k, f in evalsets[c].items()} for c in C.CATS}
            del dec
            torch.cuda.empty_cache()
            print("o3", seed, name, {c: round(out[name][c]["heldout"]["gcos"]["mean"], 4) for c in C.CATS}, flush=True)
        out["_bytes"] = {**{f"A3+r{r}": factor_bytes(a3, r) for r in RANKS},
                         "A3+1d": factor_bytes(a3, 0), "full_decoder": sum(v.numel() * 4 for v in a3.values())}
        res[str(seed)] = out
    C.write_json(OUT / "O3_oracle.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("o1", "o2", "o3"))
    {"o1": o1, "o2": o2, "o3": o3}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
