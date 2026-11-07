"""NL-RET-01A shared definitions: stream, eligibility, Replay-24 membership, cohorts, checkpoint lineage.

Custody pools
  ONLINE_LEGAL      at the end of stage s: current-task (task s) train normals; the Replay-24 members chosen for stage s+1
                    (buffer_index(done = tasks 1..s), a deterministic function of the seed and completed tasks only);
                    Q2 synthetic items whose host is such a member; the stage-s decoder and its archived logs.
  EVAL_ONLY_NORMAL  train/good identities of an old category that are NEVER Replay-24 members at any stage (fixed cohort),
                    and Q2 items whose host is such a nonmember. Targets / oracle diagnostics only.
  EXPOSED_DEFECT_EVAL  MVTec test anomalies + masks: only the sealed NL-RET-00B stage AP trajectories (Y4), post hoc.
"""
from __future__ import annotations

from pathlib import Path

from exps.dinomaly import common as D
from exps.dinomaly.common import require, sha
from exps.rh01 import common as RC

ROOT = D.ROOT
OUT = ROOT / "results/hope_cad/nl_ret_01a"
CK = Path.home() / "hope_cad_cache/nl_ret_01a/ckpt"           # ext4 copies of decoder states (SHA-verified)
STREAM = list(RC.STREAM)
SEEDS = (2711, 2712, 2713)
ARMS = ("ID", "FS_B")
CAPACITY = RC.CAPACITY


def a_of(c):
    return STREAM.index(c) + 1


def states(arm):
    """Decoder states that exist for `arm` (FS_B stages <= 6 ARE the R15 states: not separate replicates)."""
    return list(range(1, 16)) if arm == "ID" else list(range(7, 16))


def owner(arm, s):
    """Which run owns the state (arm, s)."""
    return "ID" if (arm == "ID" or s <= 6) else "FS_B"


def eligible(arm):
    """Transition rows (c, s): observation at end of stage s, outcome over s -> s+1, s = a(c)..14.
    FS_B rows with s <= 5 are identical to ID rows (shared states and shared outcome) and are flagged `shared`."""
    rows = []
    for c in STREAM:
        for s in range(a_of(c), 15):
            rows.append({"category": c, "s": s, "shared_with_ID": arm == "FS_B" and s <= 5})
    return rows


def proto_and_n():
    from exps.rh01 import run as R1
    from exps.rh01b import run as RB
    proto = RB.verify_lock()
    return proto, R1.n_train(proto)


def members(seed, n, s):
    """Replay-24 members held at the END of stage s (= the buffer replayed during stage s+1): {c: [train index]}."""
    require(1 <= s <= 15, "stage")
    return RC.buffer_index(seed, n, STREAM[:s])


def ever_members(seed, n, c):
    """Union over all stages of c's member indices (prefix of c's permutation, largest at s = a(c))."""
    perms, _ = RC.buffer_plan(seed, n)
    q = max(len(members(seed, n, s).get(c, [])) for s in range(a_of(c), 16))
    return perms[c][:q]


def nonmembers(seed, n, c):
    ev = set(ever_members(seed, n, c))
    return [i for i in range(n[c]) if i not in ev]


def checkpoint(arm, seed, s):
    """(path, sha256, provenance) of the decoder state after stage s for `arm`, verified against its run record."""
    if owner(arm, s) == "ID":
        rec = D.read(ROOT / f"results/hope_cad/rh01b/runs/R15_seed{seed}/STAGE{s}.json")
        prov = "RH-01b R15"
    elif s <= 9:
        rec = D.read(ROOT / f"results/hope_cad/nl_align_01/g1/runs/FS_B_seed{seed}/STAGE{s}.json")
        prov = "NL-ALIGN-01 G1 FS_B"
    else:
        rec = D.read(ROOT / f"results/hope_cad/nl_align_02_mvtec/m3/runs/FS_B_seed{seed}/STAGE{s}.json")
        prov = "NL-ALIGN-02 M3 FS_B"
    return ROOT / rec["checkpoint"], rec["checkpoint_sha256"], prov, rec


def local_ckpt(arm, seed, s):
    return CK / f"{owner(arm, s)}_seed{seed}_s{s}.pt"
