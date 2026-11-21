"""nanm — NANM-RECOVERY-01 reconstruction (2026-11-21).

PROVENANCE: this is NOT the nanm/ package that existed on the (now offline) school server; that source was
never pushed and is lost. This package is rebuilt from the surviving, committed G3 oracle
(exps/nanm_r0/audit.py, exps/nanm_r1/run.py, exps/nanm_r2/run.py, exps/nanm_r2b/run.py, exps/bench_g3/run.py).

- nanm.g3          G3 core. Every numerical step delegates to the oracle functions (no re-implementation);
                   this module only composes them without the oracle's CUDA / cache-path assumptions.
- nanm.checkpoint  Checkpoint publisher (new code; the server publisher is lost, parity with it is UNVERIFIED).

Not reconstructed: DYN-00 / DYN-01 interfaces and policies (C1..C6, F/P/B tiers). Their specification exists
only in an owner-pasted chat summary; see agents/reports/nanm_dyn_01_archive_2026_11_21.md.
"""
