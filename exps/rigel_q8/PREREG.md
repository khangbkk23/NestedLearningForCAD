# RIGEL-Q8-01: pre-registration (written 2026-11-26, before any candidate outcome)

- **Owner request (2026-11-26).** Items 1–3 of `agents/reports/rigel_cad_head_plan_01.md`:
  1. measure the int8 key error;
  2. image score and map smoothing on existing grids;
  3. more keys in the same 8 MiB (triangular S + int8 keys).
  "If item 3 passes DEV, continue to the CMS level". That CMS level (plan D1) will get its own pre-registration later.
- **Frozen.** RIGEL-CAD v1 (`rigel_cad/` at bc34f76) is unchanged. All candidate code lives in `exps/rigel_q8/`.
- **Integrity.** This file's sha256 is written into every stage output, and every stage refuses to run if it changed. Stages run in the order S1 → S2 → S3. No gate, candidate set or threshold is changed after an outcome is seen.

---

## S1. int8 key error (label-free diagnostic; GO/NO-GO for S3)

**Input:**
- the V1 final DEV state from GG-01: `/mnt/k/NLCAD/cache/rigel_gg01/dev/states/V1.pt`, which holds 3,925 fp16 keys, S and n;
- 4,096 training tokens per AD 2 scenario, drawn seeded with `torch.Generator().manual_seed(20261126 + l)` from the AD 2 train tokens.
No test image and no label is used.

**Quantiser (Q):**
- symmetric per-key absmax: `scale = fp16(max|x| / 127)`, `q = clamp(round(x / scale), −127, 127)` as int8;
- dequantised `x̂ = q · scale` (fp32);
- 770 B per key (768 int8 values plus an fp16 scale).

**Measured:**
- key cosine(x, x̂) and relative L2 error;
- whitened distance error ‖(x − x̂)W‖ relative to the median nearest-key whitened distance;
- for every sample token, the energy e(q) under fp16 keys vs dequantised keys (same W), giving relative |Δe| / e (median, p99) and the fraction of tokens whose nearest key changes;
- idempotence: Q(dequant(Q(x))) = Q(x).

**GO rule (frozen).** GO if all three hold:
- median relative |Δe| ≤ 0.01;
- p99 relative |Δe| ≤ 0.05;
- Q is idempotent on all keys.

Otherwise S3 is not run (`Q8_NO_GO`).

---

## S2. Read-out: map smoothing σ and image score (DEV selection, then one-shot FULL confirmation)

**DEV grids:**
- the GG-01 V1 final-stage DEV grids `/mnt/k/NLCAD/cache/rigel_gg01/dev/V1/<scenario>.pt` (MVTec AD 2 test_public, 1,084 images, corrected masks: only `test_public/bad/` images carry a mask);
- these grids are bitwise equal to the locked RIGEL-CAD AD 2 arm E.

**Candidates (frozen):**
- Gaussian σ ∈ {2, 3, **4** (v1), 6, 8}, with kernel size 2·int(4σ + 0.5) + 1. The read-out is otherwise v1: bilinear upsampling to 392, zero padding, batch shift to ≥ 0.
- Image score ∈ {top 0.1 %, top 0.5 %, **top 1 %** (v1), top 2 %, top 5 %, max} of the smoothed 392 map.

**Metrics:**
- Pixel AP: native binned AP (8,192 bins), exactly as `rigel_cad.metrics`.
- Image AUROC per scenario.
- Both are macro means over the 8 scenarios.
- Paired bootstrap: `boot_weights(n_s, position_s)`, 2,000 replicates.

**Selection (frozen):**
1. **Step a.** σ\* = argmax DEV macro pixel AP. Adopt σ\* only if Δ(σ\* − 4) ≥ +0.003 **and** the paired CI95 lower bound > 0. Otherwise σ\* = 4.
2. **Step b.** With σ\*, s\* = argmax DEV macro image AUROC. Adopt s\* only if Δ(s\* − top 1 %) ≥ +0.005 **and** the CI95 lower bound > 0. Otherwise s\* = top 1 %.
3. **Verdict.** `RO_DEV_PASS` if σ\* or s\* was adopted, otherwise `RO_NO_CHANGE`.

**FULL confirmation** (CPU, one shot, only after `RO_DEV_PASS`):
- Data: the locked E@8 final-stage grids of order a, seed 2711 (`/mnt/k/NLCAD/cache/nl03_multirate_01/bench/<ds>_a2711/E/<cat>.pt`) on MVTec AD, VisA and MPDD, read-only.
- Compared: v1 read-out vs (σ\*, s\*), paired bootstrap per dataset.
- **Confirmed** if all of these hold:
  - the adopted metric's Δ ≥ 0 on all 3 datasets, and its CI lower bound > 0 on at least 2 of 3;
  - the other metric drops by no more than 0.005 on any dataset.
- **Verdict:** `RO_CONFIRMED` or `RO_NOT_CONFIRMED`.

---

## S3. Q8T: triangular S + int8 keys in the same 8 MiB (DEV gate)

**Stream:**
- MVTec AD 2, 8 scenarios in alphabetical order, train/ only, seed 2711.
- Tokens: `/mnt/k/NLCAD/cache/rigel_cad/dinov2_vitb14_reg4/ad2/*.pt`, sha-verified.
- Per-task fits (C_t, S_t, n_t): the GG-01 cache `/mnt/k/NLCAD/cache/rigel_gg01/fit/ad2/*_s2711.pt`, i.e. the v1 computation.
- Evaluation: test_public, all stages, v1 read-out (σ 4, top 1 %), corrected masks.

This is the GG-01 DEV set. Its V1, U and F numbers are already known: 0.2379, 0.2555 and 0.2025 pixel AP. The candidate's outcome is unknown.

**Byte accounting (8 MiB = 8,388,608 B):**
- Key: fp16 = 1,536 B; int8 + fp16 scale = 770 B.
- S full (v1 convention): 2,359,304 B (768² fp32 + 8).
- S triangular: 1,181,192 B (768·769/2 fp32 + 8).

**Arms.** Shared per-task fit; consolidation is the v1 rule (seeded projected k-centre over candidates, survivors keep insertion order).

| Arm | Keys | Cap | S stored as | Scoring |
|---|---|---|---|---|
| V1 | fp16 | 3,925 | v1 (fp64 in memory, charged fp32 full) | whitened |
| T16 | fp16 | 4,692 | fp32 triangle | whitened |
| Q8 | int8 | 7,830 | v1 | whitened |
| **Q8T (candidate)** | int8 | **9,360** | fp32 triangle | whitened |
| F | fp16 | 5,461 | none | Euclidean |
| F8 | int8 | 10,894 | none | Euclidean |
| U | fp16 | unbounded | v1 | whitened |

**Rules for the int8 and triangular arms:**
- int8 arms quantise C_t with Q *before* consolidation. Consolidation, storage and scoring all see the dequantised stored vectors.
- Triangular-S arms keep the stored statistic as an fp32 upper triangle, updated as S ← fp32(fp64(S) + S_t). W is built from that stored S, so a reloaded state scores exactly like the in-memory one.
- Q8T payload = 9,360 × 770 + 1,181,184 + 8 = 8,388,392 B ≤ 8,388,608. Asserted at every boundary.

**Primary metric:** macro final-stage native binned pixel AP, as in GG-01. The paired CI95 uses the same bootstrap.

**DEV gate (conjunctive; frozen):**

| Gate | Condition |
|---|---|
| G1 | Δ macro pixel AP (Q8T − V1) ≥ +0.005 |
| G2 | Paired CI95 lower bound of Δ > 0 |
| G3 | No scenario loses more than 0.03 final pixel AP |
| G4 | Forgetting AP(Q8T) − forgetting AP(V1) ≤ 0.005 |
| G5 | Payload bytes ≤ 8,388,608 at every boundary (all bounded arms) |
| G6 | Correctness: V1 grids bitwise equal to the locked AD 2 arm E (all stages); Q idempotent; serialized Q8T state reloads to identical keys, S and W; finite W at every stage; unit tests pass |
| G7 | Nested-Learning sanity: Δ macro pixel AP (Q8T − F8) > 0 (the slow level still beats spending its bytes on int8 keys) |

**Verdicts:**
- `Q8T_REJECTED_DEV` if G1 or any of G3–G7 fails.
- `Q8T_INCONCLUSIVE_DEV` if only G2 fails.
- `Q8T_DEV_PASS` if all gates pass.

**After `Q8T_DEV_PASS`:**
- The next step is the CMS medium-level study (plan D1), with its own pre-registration, built on Q8T storage.
- The FULL confirmation of Q8T is frozen below and runs one-shot when the owner asks for it, or together with D1's FULL.

**After the other verdicts:** v1 stays; D1 is not started on this basis.

**FULL (frozen, deferred):**
- Cells: MVTec AD, VisA and MPDD; order a, seed 2711; all stages.
- Arms: V1 (= locked E@8 grids), Q8T and F8.
- **Confirmed** if both hold:
  - Q8T − V1 macro pixel AP > 0 with paired CI95 lower bound > 0 on at least 2 of 3 datasets, and no dataset < −0.005;
  - Q8T − F8 > 0 on all 3.

**Diagnostics (not gating):** keys retained per task, update time, latency, serialized file size, and the S1 error measures at every stage.

**Compute.** S3 is about 0.5 GPU-h (cached fits; 7 arms; 36 stage-grids per arm), plus CPU metrics of about 30 min. Stop if any drive falls below 30 GiB free. One GPU job at a time.

**Forbidden after outcomes:** another quantiser, cap, scale format, seed, order, metric or threshold.
