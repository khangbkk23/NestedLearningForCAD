# exps/calm01/smoke_p1_stream.py
"""Label-free CPU smoke run on the P1 normal stream (cached frozen train features only).

Reads no test image, label or mask and computes no anomaly metric. Reports bank composition per
category at the P1 boundaries, CALM tracked deficits/prices and update time, for feasibility only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from exps.calm01 import arms

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/hope_cad/frozen_features_v1/clean"
STREAM = ROOT / "results/hope_cad/task_p/p0c/protocol_resolved.json"
CATS = {"bottle": 0, "carpet": 1, "hazelnut": 2}
BOUNDARIES = (100, 200, 300, 350)
P1_GREEDY_BYTES = 2702 * 768 * 4  # P1 FULL784_GREEDY descriptor bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=list(arms.ARMS))
    ap.add_argument("--events", type=int, default=350)
    args = ap.parse_args()
    torch.set_num_threads(8)
    updates = json.loads(STREAM.read_text())["updates"][:args.events]
    feats = {c: torch.load(CACHE / f"{c}_train.pt", map_location="cpu", weights_only=False, mmap=True)["patches"]
             for c in CATS}
    report = {"budget_bytes": P1_GREEDY_BYTES, "arms": {}}
    for name in args.arms:
        m = arms.build(name, P1_GREEDY_BYTES, len(CATS), n_ref=100)
        rec = {"capacity": m.capacity, "ledger": arms.ledger(name, m.capacity, len(CATS)),
               "boundaries": {}, "update_seconds": []}
        prev = None
        for u in updates:
            c = CATS[u["category"]]
            if c != prev:
                if prev is not None:
                    m.end_category()
                m.begin_category(c); prev = c
            x = feats[u["category"]][u["cache_index"]].clone()
            serials = list(range((u["event"] - 1) * 784, u["event"] * 784))
            t = time.perf_counter(); m.update(x, serials)
            rec["update_seconds"].append(time.perf_counter() - t)
            if u["event"] in BOUNDARIES:
                row = {k: m.count(v) for k, v in CATS.items()}
                if hasattr(m, "price"):
                    row["price"] = {k: m.price.get(v) for k, v in CATS.items()}
                    row["deficit"] = {k: m.deficits().get(v) for k, v in CATS.items()}
                    row["scale"] = {k: m.scales.s.get(v) for k, v in CATS.items()}
                rec["boundaries"][u["event"]] = row
                print(name, u["event"], row, f"{sum(rec['update_seconds']):.0f}s", flush=True)
        s = rec["update_seconds"]
        rec["update_summary"] = {"total": sum(s), "mean": sum(s) / len(s), "max": max(s)}
        report["arms"][name] = rec
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
