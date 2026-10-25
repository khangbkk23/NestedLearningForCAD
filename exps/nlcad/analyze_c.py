# exps/nlcad/analyze_c.py
"""NLCAD-01 Stage C analysis from logs and checkpoints only (no development data, no labels):
controller learning, mechanism diagnostics, training integrity. Identity reference = RH-01b R15 (same seed, same
task-6 start, bitwise-equivalent machinery).

  python -m exps.nlcad.analyze_c --run N_per_task_seed2711_from6 [--coverage]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad.controller import BLOCKS
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"


def stage_records(run, start):
    cfg = D.read(OUT / f"runs/{run}/CONFIG.json")
    rows = {}
    for s in range(start + 1, len(C.STREAM) + 1):
        js = OUT / f"runs/{run}/STAGE{s}.json"
        if js.exists():
            rec = D.read(js)
            require(sha(D.ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], f"checkpoint changed: stage {s}")
            rows[s] = {"stage": rec, "events": D.read(OUT / f"runs/{run}/EVENTS{s}.json")}
    return cfg, rows


def controller_learning(rows):
    out = {}
    for s, r in rows.items():
        ev = r["events"]
        dja = np.array([[e["dJ_da"][b] for b in BLOCKS] for e in ev])
        dev = dja - dja.mean(1, keepdims=True)
        hg = np.array([e["hypergrad"] for e in ev])
        out[s] = {"events": len(ev), "a_end": dict(zip(BLOCKS, ev[-1]["a"])), "phi_end": dict(zip(BLOCKS, ev[-1]["phi"])),
                  "max_abs_a_minus_1": float(np.max(np.abs(np.array([e["a"] for e in ev]) - 1))),
                  "saturation_max": max(e["saturation"] for e in ev),
                  "hypergrad_sign_consistency": dict(zip(BLOCKS, np.mean(np.sign(hg) == np.sign(hg.mean(0)), 0).tolist())),
                  "global_dJ_dlogscale_positive_frac": float(np.mean(dja.sum(1) > 0)),
                  "relative_t_stat": dict(zip(BLOCKS, (dev.mean(0) / (dev.std(0) / np.sqrt(len(ev)) + 1e-30)).tolist())),
                  "outer_first_last": [ev[0]["outer"], ev[-1]["outer"]],
                  "outer_current_mean": float(np.mean([e["loss_current"] for e in ev])),
                  "outer_replay_mean": float(np.mean([e["loss_replay"] for e in ev])),
                  "syn_Q3_mean": float(np.nanmean([e.get("syn_Q3", np.nan) for e in ev])),
                  "syn_Q2_mean": float(np.nanmean([e.get("syn_Q2", np.nan) for e in ev]))}
    return out


def heldout_vs_identity(rows, seed):
    out = {}
    for s, r in rows.items():
        ref = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")
        out[s] = {c: {"N": r["stage"]["heldout_normal"][c]["global_cosine"],
                      "identity": ref["heldout_normal"][c]["global_cosine"]} for c in C.EVAL}
    return out


def integrity(rows, seed):
    out = {}
    for s, r in rows.items():
        ref = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")
        st = r["stage"]
        out[s] = {"checkpoint_sha256": st["checkpoint_sha256"], "nonfinite_skips": len(st["nonfinite_skips"]),
                  "slow_events": st["slow_events"], "seconds": st["seconds"], "identity_seconds": ref["seconds"],
                  "peak_allocated_mib": st["peak_allocated_mib"],
                  "gradient_calls": {"fast_backward_microbatches": R1.ITERS * 2,
                                     "outer_backward_microbatches": st["slow_events"] * 2,
                                     "diagnostic_forward_batches": st["slow_events"] * 4}}
    return out


@torch.no_grad()
def coverage(run, s, seed, device="cuda"):
    """Buffer-member vs non-member vs held-out normal loss at stage s, for the NL run and the identity run."""
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    idx = C.buffer_index(seed, n, list(C.STREAM[:s - 1]))
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    res = {}
    for name, path in (("N", D.ROOT / D.read(OUT / f"runs/{run}/STAGE{s}.json")["checkpoint"]),
                       ("identity", D.ROOT / D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")["checkpoint"])):
        snap = torch.load(path, weights_only=False)
        dec = O.Decoder(768, 12).to(device).eval()
        dec.load_state_dict(snap["state"])
        res[name] = {}
        for c in C.EVAL:
            f = C.load_features(c)
            loss = []
            for i in range(0, len(f), 8):
                xb = f[i:i + 8].to(device)
                en, de = dec(xb[:, 0], xb[:, 1])
                loss += torch.stack([1 - F.cosine_similarity(e.flatten(1), d.flatten(1)) for e, d in zip(en, de)], 1).mean(1).tolist()
            loss = np.array(loss)
            m = np.zeros(len(loss), bool)
            m[idx.get(c, [])] = True
            xb = held[c].to(device)
            en, de = dec(xb[:, 0], xb[:, 1])
            res[name][c] = {"members_n": int(m.sum()), "members": float(loss[m].mean()) if m.any() else None,
                            "nonmembers": float(loss[~m].mean()),
                            "heldout": float(torch.stack([1 - F.cosine_similarity(e.flatten(1), d.flatten(1))
                                                          for e, d in zip(en, de)], 1).mean())}
        del dec
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="N_per_task_seed2711_from6")
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    cfg = D.read(OUT / f"runs/{a.run}/CONFIG.json")
    cfg_, rows = stage_records(a.run, cfg["start"])
    res = {"config": cfg, "stages_done": sorted(rows), "controller": controller_learning(rows),
           "heldout_vs_identity": heldout_vs_identity(rows, cfg["seed"]), "integrity": integrity(rows, cfg["seed"])}
    if a.coverage:
        DR.configure()
        res["coverage_final"] = coverage(a.run, max(rows), cfg["seed"])
    D.write_json(OUT / f"ANALYSIS_C_{a.run}.json", res)
    for s in sorted(rows):
        c = res["controller"][s]
        print(s, "max|a-1|", round(c["max_abs_a_minus_1"], 4), "global+", c["global_dJ_dlogscale_positive_frac"],
              "t", {b[-1] if b != "bottleneck.0" else "bn": round(v, 1) for b, v in c["relative_t_stat"].items()},
              "held", {k: (round(v["N"], 4), round(v["identity"], 4)) for k, v in res["heldout_vs_identity"][s].items()})


if __name__ == "__main__":
    main()
