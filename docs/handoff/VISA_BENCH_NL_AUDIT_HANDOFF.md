# HOPE-CAD handoff: TCS-00, VBENCH-00 (full VisA), NL architecture audit, CMS-B plan

Branch `dikhang_space`, 2026-11-18. The full reports live in the git-ignored `agents/` directory:
- `agents/reports/tcs_00.md`
- `agents/reports/vbench_00.md`
- `agents/reports/nl_architecture_audit_2026_11_18.md`
- `agents/tasks/task_cms_bounded_00.md`

Machine-readable results are in `results/hope_cad/{tcs_00,vbench_00}/RESULT.json`.

## 1. F, the detector under test

**Fast level (every step):**
- rank-8 LoRA at 34 sites on a frozen BASE decoder;
- surprise-weighted patch cosine with fixed γ = 2.1;
- 300 steps per task.

**Slow level (once per task):**
- the endpoint is written to the expert archive;
- the next task warm-starts from it.

**Frozen:** DINOv2-reg B/14.

**Inference:**
- task-free min over the experts;
- bf16 lite deployment;
- A1 readout.

Code path: `exps/vbench_00/run.py` → `exps.hope_s_01.run.acquire(learn=False)`.

## 2. TCS-00 E0: per-task surprise headroom (VisA-DEV) → `NO_TASK_CONDITIONAL_HEADROOM`

**Design:** γ ∈ {1, 2, 3} × seeds {2711, 2712}, 240 train normals per class, scored on VisA-DEV (240 images).

**Guards:**

| Guard | Value | Threshold | Result |
|---|---|---|---|
| Cross-fitted headroom (per-task γ vs best constant) | −0.0020, CI [−0.0121, +0.0039] | ≥ +0.010 | fail |
| Argmax agreement across seeds | 6/12 | ≥ 8 | fail |
| γ effect vs seed noise | 0.019 vs 2 × 0.011 | effect > 2 × noise | fail |

**Conclusion:** a task-conditional γ controller is closed. The hard-patch effect itself (γ ≥ 2 beats γ = 1 by
≈ 0.014 AP) does replicate on VisA.

## 3. VBENCH-00: one-shot full VisA benchmark of F (paper protocol, 12 tasks, seed 2711)

The configuration was locked (LOCK `ae5ad29c…`) before any VisA test access.

| Method | Image AUROC | Pixel AUPR |
|---|---:|---:|
| UCAD (ReplayCAD Tab. 8) | 0.874 | 0.300 |
| CADIC (CADIC Tab. V) | 0.891 | **0.438** |
| ReplayCAD (Tab. 8) | 0.903 | 0.415 |
| **F, full test** | **0.932** | 0.417 |
| F, VisA-FINAL only (DEV excluded) | 0.935 | 0.420 |

**Forgetting:** Pixel FM 0.0035, Image FM 0.0010.

**Cost:**
- persistent state 288 MB: encoder 145 MB, BASE 123 MB, 12 experts 20 MB;
- 0.33 s per image;
- 404 s update time.

**Caveats:**
- one seed;
- Pixel AP is native resolution, whereas the papers resize;
- the published numbers were not reproduced in our pipeline.

**Weak classes:** macaroni1/2 (≈ 0.09–0.11), pcb2/4.

## 4. NL architecture audit (paper re-read in full, code checked)

**F is a valid nested system (NL Def. 2–3/6).** Its levels:
- per-image linear-attention memory;
- per-step LoRA plus Adam moments;
- per-task non-parametric archive;
- frozen level.

Transfer between levels uses two mechanisms: initialization (Eq. 28, copy form) and non-parametric retrieval (Eq. 27).

**F is not CMS and not HOPE.** Its slow level stores snapshots and does not learn.

**Gaps a reviewer would raise:**
- **D3:** optimizer moments are reset at every task (`exps/hope_s_01/run.py:84`), contrary to §4.5.
- **D4:** the BASE decoder is random and never trained (§6, §7.3).
- **D6:** the archive's contribution has not been ablated. Near-zero forgetting is structural, and memory and inference
  grow as O(T).

## 5. Next: CMS-B plan (bounded memory, multi-frequency, gain-gated; nothing run yet)

Constant-in-T budget:
- ≤ 3 LoRA sets;
- ≤ 16 MB slow-level context;
- ≤ 2 decoder passes.

Every step uses 2 seeds and must pass a gain gate on dev data (synthetic proxy on held-out train normals, plus VisA-DEV
as a guard).

| Step | What it does | Must beat |
|---|---|---|
| S0 | References F-archive vs F-last (no training), plus proxy check | — |
| S1 | CMS-2: fast LoRA per step, slow LoRA per task with persistent moments | F-last |
| S2 | The slow level gets its own replay context | S1, **and** a single-level control with the same replay |
| S3 | M3-style slow momentum | S2 |
| S4 | Frequency spectrum | S3 |
| S5 | Lock, then one-shot VisA and MVTec, 3 seeds | — |

Owner decisions pending:
- whether replay is allowed (≤ 16 MB);
- the dev-gate protocol.
