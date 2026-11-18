# HOPE-CAD handoff — HOPE-S: nested learning on top of the strongest NAR model

Branch `dikhang_space`, 2026-11-18.

Setting:
- sequential, normal-only continual anomaly detection on MVTec AD;
- 15 tasks in alphabetical order, seed 2711;
- no replay, no task ID at inference.

All numbers are exploratory: one seed, one order, and the MVTec test set was already exposed in earlier work.

## 0. Status update after HOPE-S-02 — read this first

The fixed-γ control **matches HOPE-S**:

| arm | Pixel AP | Image AUROC | Pixel FM |
|---|---:|---:|---:|
| F: γ = 2.1 fixed, λ = 0, no slow-level learning | 0.6013 | 0.9783 | 0.0031 |
| HOPE-S | 0.6006 | 0.9761 | 0.0052 |

HOPE-S − F = **−0.0007, CI [−0.0100, +0.0083]**.

What this means:
- The gain over C comes from the **surprise-weighted (hard-patch) objective**, not from nested learning of γ and λ.
- **The nested-learning contribution is therefore NOT supported by the current evidence.** This is a diagnostic
  result: γ = 2.1 was chosen post hoc, and only one seed was run.
- Engineering default from here: the fixed hard-patch objective, which must not be described as NL.
- Report: `agents/reports/hope_s_02.md`.
- The next NL candidate is in §4.

Sections 1–3 describe HOPE-S as it was evaluated against C. They remain accurate as statements about C.

## 1. Finding against C (superseded as an NL claim by §0)

The HOPE-S slow level adds a measurable gain over the exact ablation that switches it off:

> **HOPE-S vs C (identical system with the slow level switched off): Pixel AP +0.0108, 95% category-bootstrap CI
> [+0.0017, +0.0198], Image AUROC +0.011, lower forgetting. Preregistered `NL_GATE` (Δ ≥ +0.010, CI lower > 0,
> AUROC / acquisition / FM tolerance) passed.**
>
> HOPE-S-02 then showed that a fixed γ achieves the same result, so this gain is not attributable to the nested
> learning.

**Why C is the exact ablation.** HOPE-S with the slow-level learning rates set to 0 reproduces C exactly. Before the
first slow update, the first 10 training losses of task 1 are bitwise equal to C (`TASK1_PRE_SLOW_PARITY_PASS`).

**Same between the two runs:** data, initialisation, optimizer, step budget, memory, scorer and readout. The only
difference is that the nested slow level is learning.

**What this does not establish yet:**
1. **Learned versus fixed weighting.** The slow level learns a surprise exponent γ (0 → 3.44) and per-site rates λ. A
   conventional fixed hard-patch objective with a similar γ might capture part of the gain. In that case the evidence
   would point to "hard-patch weighting", not specifically to *nested learning*.
   - **HOPE-S-02** tests this: fixed γ = 2.1, λ = 0, matched compute, with γ chosen post hoc from the HOPE-S
     trajectory. It was **running at the time of this handoff**.
   - Its verdict will be added to §4. Until then, the NL contribution is established only against C.
2. **Interaction between γ and λ.** Even if HOPE-S beats fixed γ, the learned-γ and learned-λ effects stay
   confounded.
3. **Robustness.** One seed and one order on exposed MVTec. The point estimate is only just above the +0.010 margin.

## 2. System (HOPE-S; the fixed-γ variant is identical except for the slow level)

```
image ──► DINOv2-reg ViT-B/14, 392 px full-view resize, blocks 0-9 only (bf16)
          └─ mean of blocks 6-9, 784 patch tokens x 768
      ──► frozen random-init BASE decoder (Dinomaly single-stream: bMLP + 8 linear-attention blocks, bf16)
          + per-task rank-8 LoRA expert on 34 Linear sites (archived after each task, bf16)
      ──► per expert: patch map 1 − cos(feature, reconstruction)
      ──► task-free minimum over all arrived experts (no task ID)
      ──► A1 readout: bilinear to 392², Gaussian σ = 4, Pixel AP at native resolution;
          image score = mean of the top 1% of the map
```

### Training: two nested levels per task

**Fast level** (every step; 300 steps per task; FIT normals):
- the LoRA starts from the previous task's endpoint (warm start);
- loss = mean_p w_p (1 − cos_p), with surprise weights w ∝ e^γ normalised to mean 1, where e is the detached patch
  error;
- optimizer: StableAdamW with per-site learning rate lr · exp(λ_site).

**Slow level** (every 10 fast steps; 30 per task; carried across all tasks — this is the NL memory):
- **Objective:** the held-out normal tail, i.e. the mean of the top-1% patch errors on a CREDIT normal image. It is a
  normal-only proxy for false-positive peaks and uses no anomaly labels.
- **λ update:** λ ← clip(λ − 0.2 · cos(Δθ_site, ∇_site L_credit)).
- **γ update:** γ ← clip(γ + 0.05 · cos(∇L_credit, ∂γ∇L_fit), 0, 4). This is a one-step hypergradient direction.
  The derivative of the fit gradient with respect to γ is computed exactly, as the gradient of mean(s·e) with
  s = w(ln e − mean(w ln e)).

**Learned trajectory:**
- γ rose steadily: 0.36 → 1.16 → 2.05 → 3.44 (tasks 1, 4, 8, 15);
- λ mean fell from 0 to −1.13 (about 3× lower learning rates), with std 0.63 across sites.

## 3. Lineage and attribution (final native Pixel AP, macro over 15)

| step | Pixel AP | Δ | Image AUROC | NL? |
|---|---:|---:|---:|:-:|
| fixed: a new LoRA per task from a fixed init (B/8) | 0.5123 | — | 0.929 | — |
| nested v1 (λ + Reptile, B/8) | 0.5270 | +0.0146 vs fixed | 0.938 | yes, but only +0.0036 over warm |
| warm start (B/8) | 0.5234 | +0.0111 vs fixed | 0.938 | no |
| + mean patch-cosine loss (B/8) | 0.5301 | +0.0067 | 0.942 | no |
| + DINOv2 B/14 front end | 0.5665 | +0.0364 | 0.960 | no |
| + bf16 lite deployment | 0.5668 | +0.0003 | 0.961 | no |
| + A1 readout = **C** | 0.5898 | +0.0230 | 0.965 | no |
| **+ HOPE-S slow level** | **0.6006** | **+0.0108** | **0.976** | yes, but matched by fixed γ (§0) |
| + fixed hard-patch γ = 2.1 (HOPE-S-02, post hoc) | 0.6013 | +0.0115 vs C | 0.978 | no |

### Per category, C → HOPE-S (final AP)

| category | C | HOPE-S | Δ AP | note |
|---|---:|---:|---:|---|
| screw | .334 | .378 | **+.044** | AUROC .737 → .820 |
| metal_nut | .642 | .680 | **+.038** | |
| transistor | .469 | .499 | **+.030** | |
| toothbrush | .638 | .665 | **+.027** | |
| capsule | .544 | .567 | **+.023** | AUROC .834 → .913 |
| cable | .535 | .548 | +.013 | |
| pill | .540 | .553 | +.013 | |
| bottle | .845 | .852 | +.007 | |
| wood | .678 | .681 | +.003 | |
| grid | .464 | .466 | +.002 | |
| hazelnut | .768 | .769 | +.001 | |
| zipper | .554 | .555 | +.001 | |
| tile | .693 | .689 | −.004 | |
| carpet | .785 | .776 | −.009 | |
| leather | .357 | .332 | **−.025** | |

- HOPE-S wins 12/15 categories, mostly the object categories with small defects, as the tail hypothesis predicted.
- It loses mainly on textures, with leather the clearest.
- Forgetting is low: Pixel FM 0.0052 (C 0.0057); Image FM 0.0008 (C 0.0020).

### Cost: HOPE-S vs C

| | C | HOPE-S |
|---|---:|---:|
| persistent state | 296.5 MB | 293.1 MB (the NL state λ + γ is 272 B) |
| training (4,500 fast + 450 slow steps) | 559 s | 513 s (session noise; HOPE-S adds one backward per slow step) |
| latency mean / p95 (batch 1, 15 experts) | 383 / 408 ms | 391 / 418 ms (same inference compute) |
| inference CUDA peak | 360 MiB | 392 MiB |

The baseline B/14 fp32 deployment used 646 MB; the lite deployment halved this with ΔAP +0.0002.

## 4. HOPE-S-02 (fixed-γ attribution control) — done

- **Arm:** γ = 2.1 fixed, λ = 0, identical code path. The CREDIT and γ-direction gradients are computed and
  discarded, so compute is matched (513 s vs 513 s of updates).
- **γ = 2.1** is a post-hoc, trajectory-informed diagnostic comparator; it is not preregistered.

| | C | F: fixed γ | HOPE-S |
|---|---:|---:|---:|
| final Pixel AP | 0.5898 | **0.6013** | 0.6006 |
| final Image AUROC | 0.9649 | **0.9783** | 0.9761 |
| acquisition Pixel AP | 0.5814 | **0.5934** | 0.5928 |
| Pixel / Image FM | 0.0057 / 0.0020 | **0.0031 / 0.0004** | 0.0052 / 0.0008 |

- **HOPE-S − F:** −0.0007, CI [−0.0100, +0.0083]. The diagnostic threshold is **not passed**, so
  `LEARNED_SLOW_LEVEL_ADDED_VALUE` = not supported.
- **F − C:** +0.0115, CI [−0.0018, +0.0261].

Per category:
- HOPE-S is better on late and texture tasks: toothbrush +.029, zipper +.019, tile +.017, wood +.011.
- F is better on small-defect objects: screw +.047, leather +.027, capsule / pill / transistor about +.008.
- The differences mostly follow the learned schedule (γ below 2.1 early, above it late), not consistent adaptivity.

**Next NL candidate.** A slow level must do something no single constant can. The data hint at category-dependent γ
preferences. That hint was generated from the exposed test set, so it must be tested on a fresh seed/order or a dev
split. The candidate is a **task-conditional γ** set from normal-only signals at task start, compared against the best
constant chosen without this test set. The minimal control on the way is learned γ with λ = 0, plus a second seed of
F.

## 5. Reproduce

Environment: `~/miniconda3/envs/nested_learning/bin/python` (torch + CUDA).

Large inputs, not in git:
- `checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth`;
- `results/hope_cad/nar_full_15/BASE.pt`;
- the train-feature cache `~/hope_cad_cache/nar_b14_01/features` (12.9 GB, built by `nar_b14_01 extract-train`).

```bash
python -B -m exps.nar_b14_01.run extract-train && python -B -m exps.nar_b14_01.run lock
python -B -m exps.nar_b14_01.run execute --owner-authorized        # C training (B/14, warm + patch)
python -B -m exps.nar_lite_01.run lock && python -B -m exps.nar_lite_01.run execute --owner-authorized   # C deploy
python -B -m exps.hope_s_01.run lock   && python -B -m exps.hope_s_01.run execute --owner-authorized    # HOPE-S
python -B -m exps.hope_s_02.run lock   && python -B -m exps.hope_s_02.run execute --owner-authorized    # control
pytest exps/nar_b14_01 exps/nar_lite_01 exps/hope_s_01 exps/hope_s_02
```

How each runner protects the result:
- Each runner writes a `LOCK.json` (source, input and reference hashes) before training.
- It refuses to restart if `STARTED.json` exists.
- It seals all 15 checkpoints (`TRAIN_COMPLETE.json`) before any test access.
- It ends with `FINAL_SEAL.json`.

Resources per HOPE-S run: about 35 min wall and 0.28 GPU-h. Peak CUDA is 1.35 GB, during the fidelity check; peak
RSS is 5 GB.

## 6. Code map

| path | role |
|---|---|
| `exps/nar_pilot_00/{core,full,detector}.py` | canonical NAR: LoRA, acquire loop, task-free scorer, Eq.8 FM (unchanged) |
| `exps/nar_b14_01/run.py` | B/14 front end: feature extraction, training of C |
| `exps/nar_lite_01/run.py` | bf16 / truncated deployment, A1 readout, native metrics, cost |
| `exps/hope_s_01/run.py` | **HOPE-S**: surprise loss, tail objective, nested `acquire` (λ + γ slow level) |
| `exps/hope_s_02/run.py` | fixed-γ attribution control (`acquire(learn=False)`) |

Reports, which are local and git-ignored: `agents/reports/{nar_b14_01,nar_lite_01,hope_s_01}.md`.

## 7. Protocol rules for whoever continues

- **No test-label selection.** HOPE-S hyperparameters (η_γ 0.05, γ₀ 0, top-8 tail, η_λ 0.2) were fixed before the
  run. The fixed-γ control is explicitly post hoc.
- **Immutable results.** Do not modify sealed results:
  - HOPE-S-01 `RESULT.json` sha `e8f03922…`;
  - C (NAR-LITE-01) `RESULT.json` sha `bd66bb75…`.
- **One seed.** Run a second seed before any confirmatory claim.
- **Out of scope:** no VisA access yet, and the full benchmark is not part of this work.
