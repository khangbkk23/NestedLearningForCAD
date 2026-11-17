"""Amendment 02/03 validation (not part of any registered arm): canonical-CPU shadow of DOM_PQ_B1 from its canonical
boundary t=1 through boundary T_END, compared with the registered GPU-path checkpoints (per-category gid sets).

    python -m exps.nl_sgm_01a_c2dom.validate_fps [T_END]
"""
from __future__ import annotations

import sys
import time

import torch

from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a_c2dom import common as D
from exps.nl_sgm_01a_c2dom import stream as S


def main(t_end=4):
    D.verify_lock()
    co, n = D.setup()
    part = D.partition(co, n)
    arm = "DOM_PQ_B1"
    s = S.Stream(arm, allow_gpu=False)
    reg = D.OUT / "streams" / arm
    out = D.OUT / "streams_validation" / arm
    out.mkdir(parents=True, exist_ok=True)
    s.restore(torch.load(reg / "ckpt_t01.pt", weights_only=False), 1)
    res = {"arm": arm, "start_from": "registered canonical-CPU boundary t=1", "boundaries": {}}
    for t in range(2, t_end + 1):
        c = D.ORDER[t]
        obj, _ = S.load_cb(t)
        s.codebooks.append(obj["codebook"])
        s.kept[t] = (S.K.decode(obj["sel_codes"], obj["codebook"]), D.pool_gids(n, c, part[c]["sel"]), obj["sel_codes"])
        t0 = time.time()
        s.requota(D.capacity("pq_dom", D.BUDGETS["B1"], t), t)
        regb = D.read_json(reg / f"boundary_t{t:02d}.json")
        ck = torch.load(reg / f"ckpt_t{t:02d}.pt", weights_only=False)
        nr = int(ck["header"][1:].sum())
        ids, own = ck["ids"][:nr].long(), ck["owner"][:nr].long()
        per = {}
        for k in range(t + 1):
            a, b = set(s.kept[k][1].tolist()), set(ids[own == k].tolist())
            per[D.ORDER[k]] = {"canonical_rows": len(a), "gpu_rows": len(b), "jaccard": len(a & b) / max(1, len(a | b)),
                               "identical": a == b,
                               "same_order": bool(torch.equal(s.kept[k][1], ids[own == k]))}
        res["boundaries"][t] = {"cpu_seconds": time.time() - t0, "per_category": per,
                                "gpu_ambiguous_steps": sum(f.get("ambiguous_steps", 0) for f in regb["fps_calls"]),
                                "gpu_ambiguous_distinct": sum(f.get("ambiguous_distinct", 0) for f in regb["fps_calls"])}
        print(t, c, {k: (v["identical"], round(v["jaccard"], 6)) for k, v in per.items()}, flush=True)
    D.write_json(out / f"VALIDATION_t02_t{t_end:02d}.json", res)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
