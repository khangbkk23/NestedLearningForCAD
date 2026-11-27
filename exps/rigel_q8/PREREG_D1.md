# RIGEL-D1-01: CMS medium level for evicted keys (pre-registration, 2026-11-26, before any D1 outcome)

**Trigger.** The owner's instruction (2026-11-26) was: "sau khi pass dev sẽ tiếp tục đến phần này: thêm một tầng bộ nhớ thứ ba (kiểu CMS) giữ key bị đẩy ra ở dạng nén". S3 gave `Q8T_DEV_PASS` on 2026-11-26.

**Disclosure from S3**, which is relevant to how D1 is read:
- On DEV, key count and pixel AP do not rise together: Q8 (7,830 keys) 0.2577, T16 (4,692) 0.2516, Q8T (9,360) 0.2443, U 0.2555.
- The W of triangular S equals the full-S W to 2·10⁻⁸ relative. So the arm differences come from which keys are kept, i.e. selection variance. Per scenario that variance reaches ±0.03–0.05.

## Hypothesis

**NL §7.1 (CMS).** Knowledge forgotten by a high-frequency level can persist in a lower-frequency level.

**Today.** Keys evicted at a boundary are deleted.

**D1.** D1 adds a **medium level**, so the memory has three update frequencies:
- **Fast level:** int8 keys, written at every task.
- **Medium level:** an int4 archive of evicted keys. It changes only when the fast level evicts, and it consolidates with the same k-centre rule when it is full.
- **Slow level:** S (fp32 triangle), giving W.

Scoring is the whitened nearest-key energy over fast ∪ medium (dequantised).

## Storage (8 MiB = 8,388,608 B)

**Formats:**
- S triangle: 1,181,192 B.
- int8 key: 770 B (768 + an fp16 scale). int4 key: 386 B (384 B of packed nibbles + an fp16 scale).
- int4 quantiser: symmetric per key, `scale = fp16(max|x| / 7)`, `q = clamp(round(x / scale), −7, 7)`. The int4 arms store keys at this precision and score on the dequantised values.

**Byte split of D1 (fixed a priori, not tuned): 50 / 50 of the key bytes.**
- Fast: (8,388,608 − 1,181,192) / 2 / 770 → 4,680 int8 keys.
- Medium: the remaining bytes / 386 → 9,336 int4 keys.
- Total payload 8,388,488 B.

## Arms

All arms use the shared v1 per-task fits, the v1 consolidation rule (seeded projected k-centre, survivors keep insertion order), and the S3 stream and DEV data.

| Arm | Content | Role |
|---|---|---|
| V1 | v1 | parity with locked AD 2 grids |
| Q8T | int8, 9,360 keys, S triangle | S3 candidate (the baseline D1 must beat) |
| Q4T | int4, single level, 18,672 keys, S triangle | uniform-compression control: is it the hierarchy or just more, cheaper keys? |
| **D1** | fast int8 4,680 + medium int4 9,336, S triangle | candidate |
| F8 | int8, 10,894 keys, Euclidean | NL sanity |
| U | unbounded fp16 | reference |

**D1 write rule at boundary l:**
1. Fast candidates = fast keys ∪ Q8(C_t). If over cap, k-centre to 4,680.
2. The evicted int8 keys (dequantised) go through Q4 into the medium candidates = medium ∪ Q4(evicted).
3. If over 9,336, k-centre the medium candidates.

## DEV gate (MVTec AD 2 test_public, as S3; frozen)

| Gate | Condition |
|---|---|
| G1 | Δ macro pixel AP (D1 − Q8T) ≥ +0.005 |
| G2 | Paired CI95 lower bound of (D1 − Q8T) > 0 |
| G3 | No scenario loses > 0.03 final pixel AP vs Q8T |
| G4 | Forgetting AP(D1) − forgetting AP(Q8T) ≤ 0.005 |
| G5 | Payload ≤ 8,388,608 B at every boundary (all bounded arms) |
| G6 | Correctness: V1 grids bitwise equal to the locked AD 2 arm E; Q8T grids bitwise equal to the S3 Q8T grids; int4 and int8 quantisers idempotent; serialized D1 state reloads to identical keys and W; finite W |
| G7 | Hierarchy matters: Δ macro pixel AP (D1 − Q4T) > 0 |
| G8 | NL sanity: Δ (D1 − F8) > 0 |

**Verdicts:**
- `D1_REJECTED_DEV` if G1 or any of G3–G8 fails.
- `D1_INCONCLUSIVE_DEV` if only G2 fails.
- `D1_DEV_PASS` if all gates pass.

**Reading rule fixed now.**
- If Q4T beats both D1 and Q8T, the result says "uniform int4 is the better storage", not "CMS helps". That outcome would be a separate v1.1 candidate, needing its own registration.
- Because of the S3 disclosure, any Δ below about 0.01 is reported as within selection variance, whatever its bootstrap CI.

**Diagnostics (not gating):**
- int4 error, measured as in S1;
- the medium hit rate: the share of test tokens whose nearest key is in the medium level;
- keys retained per task and level;
- FM and acquisition AP;
- update time and latency.

**FULL (frozen, deferred to an owner request):**
- Cells: MVTec AD, VisA and MPDD; order a, seed 2711.
- Arms: V1, Q8T, Q4T, D1 and F8.
- **Confirmed** if both hold:
  - D1 − Q8T > 0 with CI lower bound > 0 on at least 2 of 3 datasets, and none < −0.005;
  - D1 > Q4T and D1 > F8 on all 3.

**Compute:** about 0.1 GPU-h stream plus about 30 min CPU metrics. One GPU job at a time. Drives ≥ 30 GiB free.

**Forbidden after outcomes:** another split, bit width, cap, quantiser, seed, order, metric or threshold.
