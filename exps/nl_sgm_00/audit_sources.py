"""NL-SGM-00 mechanical source-lineage audit (read-only; CPU; no model, no mask, no score file).

Hashes the report / source / artifact paths that the NL-SGM-00 audit cites, and records
SOURCE_NOT_AVAILABLE for anything absent. Writes results/hope_cad/nl_sgm_00/SOURCE_LINEAGE.json.

    python -m exps.nl_sgm_00.audit_sources
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/nl_sgm_00/SOURCE_LINEAGE.json"
AGENTS_PIN = "b88393815586295209c2704a6e16262730f2aba4000a6904b7c5c6d2b65ed622"

# family -> list of repository-relative paths (reports, sources, artifacts) actually cited.
FAMILIES = {
    "META_NATH_NEOVIT": ["models/meta_nath_core.py", "models/titans_memory.py", "docs/runs.md",
                         "agents/reports/02_leakage_and_eval_audit.md"],
    "TITANS_FAST_MEMORY": ["models/titans_memory.py"],
    "HOPE_CMS": ["models/hope_cad/continuum_memory.py", "models/hope_cad/self_modifying_titans.py",
                 "exps/hope_image_synchronous_memory.py",
                 "agents/reports/r0_recovered_evidence/A_P1_SMT/p1_memory_learning_gate.md",
                 "agents/reports/r0_recovered_evidence/D_CMS_NM01/cms_nm01_final_failure_audit.md"],
    "P0_P0C": ["exps/task_p/p0_normal_support.py", "agents/reports/task_p_p0_normal_support_falsification.md",
               "agents/reports/task_p_p0c_full_grid_causal_verification.md"],
    "P1": ["exps/task_p/p1_readout.py", "exps/task_p/p1_data.py", "agents/reports/task_p_p1_anomaly_signal_validation.md",
           "results/hope_cad/task_p/p1/protocol_resolved.json"],
    "P2_FAMILY": ["exps/task_p/p2/model.py", "agents/reports/task_p_p2_independent_scientific_audit.md",
                  "agents/reports/task_p_p2e_fast_path_decomposition.md",
                  "agents/reports/task_p_p2f_conditional_consolidation.md",
                  "agents/reports/task_p_p2h_mechanism_implementation.md"],
    "AD01_AD02": ["exps/ad01/hope_cad_ad01_normal_support.py", "exps/ad02/hope_cad_ad02_partition.py",
                  "exps/ad02/hope_cad_ad02_run.py",
                  "agents/reports/r0_recovered_evidence/C_AD01_AD02/ad01_independent_review.md",
                  "agents/reports/r0_recovered_evidence/C_AD01_AD02/ad02_capacity_allocation_validation.md",
                  "results/hope_cad/ad02/protocol_resolved.json",
                  "results/hope_cad/ad01_phase0/arms_gpu_v5/arms_results.json"],
    "ECAD_CCNM": ["agents/reports/task_ecad_r0_evidence_architecture_decision.md",
                  "agents/reports/task_ecad_r1b_normal_fit.md", "agents/reports/task_ccnm_01_mechanism_and_results.md",
                  "agents/reports/task_ccnm_02_inferred_context_results.md"],
    "NSG01": ["agents/reports/r0_recovered_evidence/E_NSG_NS00/nsg01_g0_normal_support_geometry.md"],
    "CSG00": ["exps/csg00/fit_modes.py", "agents/reports/task_csg00_compact_geometry.md"],
    "CSG01": ["exps/csg01/select.py", "exps/csg01/quantize.py", "exps/csg01/scoring.py",
              "agents/reports/task_csg01_distance_preserving_compression.md"],
    "PQ_CODEBOOK": ["agents/reports/task_ncsl_01_nested_compressible_support_research.md"],
    "CS_LRNSM01": ["agents/reports/task_cs_lrnsm_01_compact_support.md"],
    "CALM": ["exps/calm01/calm.py", "exps/calm01/arms.py", "exps/calm01/storage.py", "exps/calm01/closest_pair.py",
             "exps/calm02/score.py", "exps/calm02/extract.py", "exps/calm02/data.py",
             "agents/reports/task_calm_01_design_repair.md", "agents/reports/task_calm_02a_early_gate.md",
             "agents/reports/task_calm_02b_failure_analysis.md", "results/hope_cad/calm02/FREEZE_mvtec.json",
             "results/hope_cad/calm02/gate_mvtec_b1/GATE_LOCK.json",
             "results/hope_cad/calm02/features/mvtec/MANIFEST.json",
             "results/hope_cad/calm02/runs/mvtec/CALM_L1_B1/train_log.json"],
    "NCSL": ["exps/ncsl/core.py", "agents/reports/task_ncsl_01_stage0.md",
             "agents/reports/task_ncsl_01_nested_compressible_support_research.md"],
    "NTM": ["exps/ntm/stage0.py", "agents/reports/task_ntm_00r_review.md"],
    "NPIM": ["models/hope_cad/npim_cad.py", "agents/reports/task_npim_01_real_anomaly_dev_r1.md",
             "agents/reports/task_npim_02_forensic.md", "agents/reports/task_npim_03_uncertainty.md"],
    "NLCAD": ["agents/reports/task_nlcad_01_build_nested_learning.md", "agents/reports/task_nlcad_05b_retention_slow_objective.md",
              "agents/reports/task_nlcad_06c_equilibrium_anchor.md", "agents/reports/task_nlcad_next_architecture_decision.md"],
    "R15_CADFINAL": ["exps/dinomaly/readout.py", "exps/dinomaly/official.py", "exps/cadfinal/metrics.py",
                     "exps/cadfinal/memory.py", "agents/reports/task_rh_01b_replay_headroom_fp32core.md",
                     "agents/reports/cad_reset_final_01.md", "agents/reports/cad_research_evidence_ledger.md",
                     "results/hope_cad/cad_final_v1/M1_EVAL.json", "results/hope_cad/cad_final_v1/PROTOCOL_LOCK_v1.json",
                     "results/hope_cad/rh01b/runs/R15_seed2711/STAGE15.json"],
    "NL_ALIGN": ["agents/reports/task_nl_align_01_decision.md", "agents/reports/task_nl_align_02_mvtec.md",
                 "results/hope_cad/nl_align_02_mvtec/FINAL_VERDICT.json",
                 "results/hope_cad/nl_align_02_mvtec/TABLE_15CLASS_FINAL.csv"],
    "NL_RET": ["agents/reports/task_nl_ret_00_forensics.md", "agents/reports/task_nl_ret_00b_native_decomposition.md",
               "agents/reports/task_nl_ret_01a_full_signal_survey.md", "agents/reports/task_nl_ret_02_incoming_task_risk.md",
               "results/hope_cad/nl_ret_01a/FINAL_VERDICT.json", "results/hope_cad/nl_ret_02/FINAL_VERDICT.json"],
    "INDEPENDENT_ARCH_CONSULTATION": [],   # cited only in the owner brief; no repository file found
    "CONTRACT": ["AGENTS.md", "AGENTS.local.md"],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "branch": git("branch", "--show-current"),
              "head": git("rev-parse", "HEAD"), "agents_md_pin_expected": AGENTS_PIN, "families": {}}
    record["agents_md_pin_match"] = sha(ROOT / "AGENTS.md") == AGENTS_PIN
    for fam, paths in FAMILIES.items():
        rows = []
        for rel in paths:
            p = ROOT / rel
            if p.is_file():
                rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha(p)})
            else:
                rows.append({"path": rel, "status": "SOURCE_NOT_AVAILABLE"})
        if not paths:
            rows.append({"path": None, "status": "SOURCE_NOT_AVAILABLE",
                         "note": "named in the owner brief only; no report/task file in agents/ or results/"})
        record["families"][fam] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1))
    missing = [r["path"] for v in record["families"].values() for r in v if r.get("status")]
    print("pin_match", record["agents_md_pin_match"], "missing", missing)


if __name__ == "__main__":
    main()
