"""Create-exclusive PROTOCOL.json + LOCK.json for NSGP-F0 (after A0, before any FITQ / probe read).

    python -m exps.nsgp_f0.lock
"""
from __future__ import annotations

from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a_c2dom import common as D2
from exps.nsgp_f0 import common as S
from exps.task_p.p0_normal_support import require

SOURCES = [
    "AGENTS.md", S.PLAN,
    "exps/nsgp_f0/__init__.py", "exps/nsgp_f0/common.py", "exps/nsgp_f0/run.py", "exps/nsgp_f0/analyze.py",
    "exps/nsgp_f0/lock.py", "exps/nsgp_f0/test_nsgp_f0.py",
    "exps/nl_sgm_01a/common.py", "exps/nl_sgm_01a/codecs.py", "exps/nl_sgm_01a/exact.py", "exps/nl_sgm_01a/am02_run.py",
    "exps/nl_sgm_01a/am03_eval.py", "exps/nl_sgm_01a_c2dom/common.py", "exps/nl_sgm_01a_c2dom/stream.py",
    "exps/cadfinal/metrics.py", "exps/cadfinal/common.py", "exps/calm02/native.py", "exps/vitb8_q2/common.py",
    "results/hope_cad/calm02/features/mvtec/MANIFEST.json", "results/hope_cad/vitb8_q2/CACHE.json",
    "results/hope_cad/nl_sgm_01a_c2_dom_sweep/LOCK.json", "results/hope_cad/nl_sgm_01a_c2_dom_sweep/PROTOCOL.json",
    "results/hope_cad/nl_sgm_01a_c2_dom_sweep/NORMAL_DIAG.json",
    "results/hope_cad/nsgp_f0/A0_AUDIT.json", "results/hope_cad/nsgp_f0/SMOKE_PRELOCK.json",
]


def main():
    require(not S.LOCK.exists() and not S.PROTOCOL.exists(), "already locked")
    require(not any((S.OUT / d).exists() for d in ("fit", "eval")) and not (S.OUT / "SELECTION.json").exists(),
            "fit / selection / evaluation artifacts exist before the lock")
    require(S.read_json(S.OUT / "A0_AUDIT.json")["verdict_A0"] == "PASS", "A0 did not pass")
    co, n = S.setup()
    part = D2.partition(co, n)
    bank = {t: {"ckpt_sha256": S.read_json(S.MEM_DIR / f"boundary_t{t:02d}.json")["ckpt_sha256"],
                "boundary_record_sha256": S.sha(S.MEM_DIR / f"boundary_t{t:02d}.json")} for t in S.BOUNDARIES}
    cbs = {c: S.sha(D2.OUT / "codebooks" / f"{i:02d}_{c}.pt") for i, c in enumerate(S.ORDER)}
    cohorts = {c: {"eval": co[c]["eval"], "sel": co[c]["sel"], "fitq": S.fitq(co, part, c),
                   "heldout": S.heldout_paths(c) if c in ("bottle", "carpet", "hazelnut") else []} for c in S.ORDER}
    protocol = {
        "schema": "nsgp_f0_protocol_v1", "plan": S.PLAN, "plan_sha256": S.sha(S.ROOT / S.PLAN),
        "task_sha256_at_lock": S.sha(S.ROOT / S.TASK), "boundaries_0based": list(S.BOUNDARIES), "final": S.FINAL,
        "bank": {"arm": S.MEM_ARM, "checkpoints": bank, "codebooks": cbs, "final_bytes": S.B0_FINAL_BYTES},
        "arms": {"C0": S.K_C0, "N0": S.K_N0, "N0_hidden": S.HID, "N0_train": S.TRAIN,
                 "P0": {"seed": S.PQ_SEED, "iters": S.PQ_ITERS}},
        "bytes_final": {"C0_C": S.bytes_c0(93680, 15, S.K_C0["C"]), "C0_Q": S.bytes_c0(93680, 15, S.K_C0["Q"]),
                        "N0_C": S.bytes_n0(93680, 15, S.K_N0["C"]), "N0_Q": S.bytes_n0(93680, 15, S.K_N0["Q"]),
                        "P0": S.bytes_pq(93680, 15), "B0": S.B0_FINAL_BYTES},
        "selection_rule": "N0* = N0_C if FITQ macro P95(N0_C) <= 1.03 x FITQ macro P95(B0) at the final boundary, "
                          "else N0_Q",
        "gate": S.GATE, "absolute_budget": S.ABS_BUDGET, "bootstrap": S.BOOT, "caps": S.CAPS,
        "cohorts": cohorts, "fitq_key": S.FITQ_KEY,
        "inputs": {"calm02_manifest_sha256": S.sha(C.FEATURES / "MANIFEST.json"), "r15_digest": C.r15_digest(),
                   "c2dom_lock_sha256": S.sha(D2.LOCK)},
        "verdicts": ["F0_FEASIBLE", "F0_NOT_SUPPORTED", "F0_INCONCLUSIVE_VALIDATION", "F0_BLOCKED"],
        "exposure": "EVAL probes were AM02's normal-only Stage-B cohort and SAND diagnostics; never a fit or selection "
                    "input here; MVTec test data are not opened",
    }
    S.write_json(S.PROTOCOL, protocol)
    S.write_json(S.LOCK, {"schema": "nsgp_f0_lock_v1", "protocol_sha256": S.sha(S.PROTOCOL),
                          "sources": {s: S.sha(S.ROOT / s) for s in SOURCES}, "utc": S.now()})
    print("LOCK", S.sha(S.LOCK), "PROTOCOL", S.sha(S.PROTOCOL))


if __name__ == "__main__":
    main()
