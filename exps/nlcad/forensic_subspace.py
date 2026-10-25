# exps/nlcad/forensic_subspace.py
"""Feasibility of an NL "memory of the old gradient subspace" (Nested Learning, sec. 4.3) for the Dinomaly decoder
(CPU only; normal training features; no labels, no development data, no GPU).

For every Linear layer of the decoder at the RH-01b R15 checkpoint after the last pre-segment task, the input
activations of the completed tasks' normal training images define an old subspace U (energy threshold tau, as in
gradient projection memory). Weight gradients of a Linear layer are outer products with its inputs, so the fraction of
a later task's input energy inside U bounds how much of that task's update a projection would remove:
  ~1 -> projection freezes the layer for the new task (no plasticity);  ~0 -> nothing to protect.
Also reports the rank r(tau) per layer and the bytes of the stored bases (the persistent cost of such a memory).

  python -m exps.nlcad.forensic_subspace --seed 2711 --stage 6 --images 32
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.nlcad.forensic_blocks import checkpoint
from exps.rh01b import common as C

OUT = D.ROOT / "results/hope_cad/nlcad"
TAUS = (0.9, 0.97, 0.99)


@torch.no_grad()
def covariances(dec, x, micro=4):
    names = [n for n, m in dec.named_modules() if isinstance(m, torch.nn.Linear)]
    cov, count, hooks = {}, {}, []

    def hook(name):
        def fn(_, inp):
            a = inp[0].reshape(-1, inp[0].shape[-1]).float()
            cov[name] = cov.get(name, 0) + (a.T @ a).double()
            count[name] = count.get(name, 0) + len(a)
        return fn

    for n, m in dec.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_pre_hook(hook(n)))
    for xb in x.split(micro):
        dec(xb[:, 0].float(), xb[:, 1].float())
    for h in hooks:
        h.remove()
    return {n: cov[n] / count[n] for n in names}


def basis(cov, tau):
    w, v = torch.linalg.eigh(cov)
    w, v = w.flip(0).clamp(min=0), v.flip(1)
    r = int(torch.searchsorted(torch.cumsum(w, 0) / w.sum(), torch.tensor(tau, dtype=w.dtype)).item()) + 1
    return v[:, :r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--stage", type=int, default=6)
    ap.add_argument("--images", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sd, h = checkpoint(a.seed, a.stage)
    dec = O.Decoder(768, 12).eval()
    dec.load_state_dict(sd)
    old_cats, new_cats = C.STREAM[:a.stage], C.STREAM[a.stage:]
    old = None
    for c in old_cats:
        cv = covariances(dec, C.load_features(c)[:a.images])
        old = cv if old is None else {k: old[k] + cv[k] for k in cv}
    old = {k: v / len(old_cats) for k, v in old.items()}
    bases = {tau: {k: basis(v, tau) for k, v in old.items()} for tau in TAUS}
    res = {"seed": a.seed, "stage": a.stage, "checkpoint": h, "images_per_task": a.images, "old_tasks": list(old_cats),
           "rank": {str(t): {k: int(b.shape[1]) for k, b in bases[t].items()} for t in TAUS},
           "bytes_fp16": {str(t): int(sum(b.numel() * 2 for b in bases[t].values())) for t in TAUS},
           "new_task_energy_in_old_subspace": {}}
    for c in new_cats:
        cv = covariances(dec, C.load_features(c)[:a.images])
        res["new_task_energy_in_old_subspace"][c] = {
            str(t): {k: float(torch.trace(b.T @ cv[k] @ b) / torch.trace(cv[k])) for k, b in bases[t].items()} for t in TAUS}
        e = res["new_task_energy_in_old_subspace"][c]["0.97"]
        print(c, "energy in old subspace (tau .97): median", round(float(np.median(list(e.values()))), 3),
              "min", round(min(e.values()), 3), "max", round(max(e.values()), 3), flush=True)
    for t in TAUS:
        r = res["rank"][str(t)]
        print("tau", t, "rank median", int(np.median(list(r.values()))), "max", max(r.values()),
              "bytes fp16", res["bytes_fp16"][str(t)], flush=True)
    D.write_json(OUT / f"FORENSIC_SUBSPACE_R15_{a.seed}_stage{a.stage}.json", res)


if __name__ == "__main__":
    main()
