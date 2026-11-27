# RIGEL-CC-01: Compression Confirmation & Final Promotion Gate (evaluation specification)

**Owner task (2026-11-26):** "RIGEL-CAD — Compression Confirmation & Final Promotion Gate".

**When this was written.** Before any MVTec AD / VisA / MPDD result of T16, Q8, Q8T, Q4T, F8 or F4 existed. The sha256 of this file goes into `results/hope_cad/rigel_cc01/SPEC_LEDGER.txt` before any run. Every stage refuses to run if the file has changed.

**Repository state at registration:**
- branch `dikhang_space`, HEAD `bc34f76e73ce9c0ac6218054ced89ccf785ab6e7`;
- untracked: `exps/rigel_q8/` and 4 files in `docs/handoff/`;
- RIGEL-CAD v1 (`rigel_cad/`) is frozen and is not modified.

---

## 1. Candidates

These are frozen as implemented in `exps/rigel_q8/q8.py` and `d1.py`. Source hashes are recorded at run time. Nothing else is added.

| Arm | Keys (cap at 8 MiB) | Slow statistic | Scoring | Status |
|---|---|---|---|---|
| V1 | fp16, 3,925 | S (fp64 sum; charged fp32 full 2,359,304 B) | whitened | reference (RIGEL-CAD v1) |
| T16 | fp16, 4,692 | S, fp32 upper triangle (1,181,192 B) | whitened | **exploratory**: chosen after Q8 DEV results were seen |
| Q8 | int8 + fp16 scale, 7,830 | S as V1 | whitened | **exploratory** |
| Q8T | int8 + fp16 scale, 9,360 | S triangle | whitened | **registered**: passed its DEV gate (`exps/rigel_q8/PREREG.md`, `Q8T_DEV_PASS`) |
| Q4T | int4 (packed) + fp16 scale, 18,672 | S triangle | whitened | **exploratory**: found in the D1 study after DEV results were seen |
| F8 | int8, 10,894 | none | Euclidean | control (fast-only); not eligible |
| F4 | int4, 21,732 | none | Euclidean | control (fast-only); not eligible |

**Notes on the arms:**
- F4 is the int4 counterpart of F8: q8 quantiser rules at 4 bits (`d1.quantize4`), cap ⌊8,388,608 / 386⌋.
- **Eligible for selection:** T16, Q8, Q8T, Q4T. F8 and F4 are the matched fast-only controls. They are reported for the Nested-Learning sanity check (candidate − its control), are not eligible, and do not gate.
- **Shared by every arm:**
  - the per-task fit from v1: 1 % greedy k-centre coreset C_t, residual moment S_t and count n_t (`exps.rigel_gg01.gg.fit_task`);
  - the v1 consolidation rule (seeded projected greedy k-centre; survivors keep insertion order);
  - the v1 read-out (σ 4, top-1 %);
  - no hyperparameter is changed.

---

## 2. Data and protocol (canonical)

**Streams.**
- MVTec AD (15 tasks), VisA (12) and MPDD (6).
- Primary cell: **order a, seed 2711**. Orders come from `exps.nl03_multirate_01.final.bench_order`.

**Tokens.**
- Frozen DINOv2-reg ViT-B/14 at 392, read from the canonical hash-verified K: caches through `exps.nl03_pressure_00.core.bind()`. The caches are read-only.
- Train tokens: `exps.nl03_multirate_01.run.fit_tokens(G, t, order=train_rows_sorted)`, i.e. 100 % of train/good.
- Test tokens: `G.load_test`.
- No feature is regenerated.

**Per-task fits.**
- MVTec and VisA: the cached v1 fits `/mnt/k/NLCAD/cache/rigel_gg01/fit/{mvtec,visa}/*_s2711.pt` (computed by `gg.fit_task`).
- MPDD: computed with the same function and cached under `/mnt/k/NLCAD/cache/rigel_cc01/fit/`.

**Metrics.**
- `rigel_cad.metrics.evaluate_category` on `rigel_cad.metrics.Category` built from `rigel_cad.data.category_split`, every stage ≥ acquisition.
- Measured: native binned pixel AP (8,192 bins), image AUROC (top-1 %), acquisition AP, forgetting AP, prequential AP (`rigel_cad.metrics.stream_summary`), and per-class final AP and AUROC.
- **Bootstrap:** the registered paired protocol: 2,000 replicates, `boot_weights(n, position)`, final stage, dataset macro.

**V1 parity (correctness):**
- the V1 grids must equal the locked benchmark grids `/mnt/k/NLCAD/cache/nl03_multirate_01/bench/{ds}_a2711/E/*.pt` bitwise, at every stage;
- the V1 metrics must equal the locked `results/hope_cad/nl03_multirate_01/bench/METRICS_{ds}_a2711.json` (arm E) at every stage.

---

## 3. Memory accounting (stated separately)

| Convention | Definition |
|---|---|
| **L, logical (registered)** | Bytes charged by the registered rule: keys × {1,536 fp16; 770 int8; 386 int4} + {2,359,304 full S; 1,181,192 triangular S; 0}. Asserted ≤ 8,388,608 at every task boundary |
| **P, persisted payload** | Sum of tensor `nbytes` in the checkpoint that restores the arm. Keys at stored precision (int4 packed 2 per byte), scales, S at the precision needed to restore, count. Task-id provenance tensors are excluded (diagnostic only) |
| **F, serialized file** | `os.path.getsize` of the `torch.save` checkpoint, including container overhead |
| **R, runtime** | GPU working set during scoring: dequantised fp32 keys, W, whitened keys, measured peak allocation for scoring one test category |

**Persisted S per arm:**
- V1 persists S in fp64 (its v1 package behaviour; known limitation L1).
- Q8 persists S as **fp32 full** (2,359,296 B), the precision its registered charge assumes. Restoring Q8 is therefore a numerical approximation, not bitwise; see §4.
- The triangular arms persist the fp32 triangle, which is their exact state.

**Accounting validity of a candidate:** L ≤ 8,388,608 at every boundary **and** P ≤ 8,388,608. V1 is the reference and is exempt; its P is 10,747,400 B, which is reported.

---

## 4. Correctness audit (must pass before performance is read)

1. **AD 2 mask lookup** (`rigel_cad/data.py`):
   - good images return None; bad images return `ground_truth/bad/<stem>_mask.png`;
   - on the real dataset, all 379 good / 705 bad are checked;
   - the `test_unit.py` same-stem test is run.
   If the lookup is still defective, patch it (not expected: fixed in bc34f76).
2. **Key representations.**
   - fp16: unit tokens stored losslessly as given.
   - int8 and int4 quantisers: range, idempotence, pack/unpack round trip (int4).
   - Dequantised values are exactly what is scored.
3. **Triangular S:** pack/unpack round trip, and W from the triangle vs W from the full fp32 matrix, which must be **bitwise equal**.
4. **Save/load and continual update after restoration** (MPDD order a, seed 2711):
   - run tasks 0–2, persist each arm (convention P), restore into fresh objects, continue tasks 3–5;
   - compare against the uninterrupted run.
   - **Mathematical equivalence class:** V1, T16, Q8T, Q4T, F8, F4, where keys, S and W must be bitwise equal.
   - **Numerical-approximation class:** Q8, where the keys must be bitwise equal and W may differ through the fp64→fp32 rounding of S. The scoring tolerance is max |Δenergy| / energy ≤ 1e-4 on the final-stage test tokens.
5. **Equivalence classes documented in the report:**
   - mathematical equivalence: triangle vs full;
   - numerical approximation: fp32 vs fp64 S (scoring tolerance stated);
   - representational approximation: int8 and int4 (quantisation distortion measured; it is part of the method, not a tolerance).
6. **Accounting:** L, P, F and R are reported for every arm, plus the validity rule of §3.

Any failure of items 1–4 or 6 blocks the performance reading (`COMPRESSION_BLOCKED`), unless a patch is possible within this task without touching frozen code.

---

## 5. Measurements (primary cell)

- Final pixel AP, image AUROC, acquisition AP, forgetting AP and prequential AP, per dataset, plus per-class final AP and AUROC.
- Persisted payload P, serialized file size F, runtime R and logical L.
- Update time: the shared fit time from the cache record, plus each arm's write/consolidation time (CUDA-synchronised). Inference latency: head energy per image, p50.
- **Quantisation distortion (label-free).** On each dataset's V1 final keys:
  - int8 and int4 relative energy change (median, p99) on 4,096 seeded train tokens per task;
  - the nearest-key assignment change rate;
  - cosine and relative L2 of the keys.
- Paired bootstrap CI95 of candidate − V1 (pixel AP and image AUROC) per dataset, and of candidate − matched fast-only control.

---

## 6. Provisional selection gate (frozen; candidate vs V1, primary cell)

A candidate is eligible for robustness only if **all** of these hold:

1. Mean over the 3 datasets of Δ final pixel AP ≥ **+0.005**.
2. Δ pixel AP > 0 on at least **2** datasets.
3. No dataset has Δ pixel AP < **−0.01**.
4. Δ image AUROC ≥ **−0.005** on every dataset.
5. Δ forgetting AP ≤ **+0.005** on every dataset.
6. No category (any of the 33) has Δ final pixel AP < **−0.03**.
7. Accounting is valid (§3) and every correctness check (§4) passes.

Gates are evaluated on point estimates; CIs are reported.

**Ranking if several candidates pass** (deterministic):
1. higher mean Δ pixel AP;
2. then higher worst-dataset Δ pixel AP;
3. then lower inference latency (p50, mean over the 3 datasets).

Exactly one candidate is selected. If none passes: `COMPRESSION_REJECTED`; keep v1 and stop.

---

## 7. Conditional robustness (selected candidate only)

**Cells:** orders {a, b, c} × seeds {2711, 2712, 2713} × {MVTec, VisA, MPDD}, i.e. 27 cells. The candidate is computed with per-seed v1 fits; seeds other than 2711 are computed and cached under `/mnt/k/NLCAD/cache/rigel_cc01/fit/`.

**V1 values** for the same 27 cells are read from the locked, hash-recorded METRICS files:
- a2711, b2711, c2711: `results/hope_cad/nl03_multirate_01/bench/METRICS_*` (MPDD c2711 from `results/hope_cad/g3e_paper_01/cells/METRICS_A1_dinov2_mpdd_c2711.json`);
- a2712, a2713: `cells/METRICS_A3_*`;
- b/c × 2712/2713: `cells/METRICS_A2_*`.
In the candidate runs, the V1 arm is recomputed alongside as a parity check against those files (final pixel AP equal).

**Reported:** per-cell Δ, mean ± std per dataset, sign counts, the bootstrap CI for the order-a cells, and negative per-class results.

**Outcome classification (frozen):**

| Outcome | Rule |
|---|---|
| `COMPRESSION_BENCHMARK_SUPPORTED` | On every dataset the mean Δ pixel AP over 9 cells is > 0. At least 2 datasets have Δ > 0 in ≥ 7 of 9 cells. On every dataset the mean Δ image AUROC is ≥ −0.005 and the mean Δ forgetting is ≤ +0.005 |
| `COMPRESSION_INCONCLUSIVE` | The provisional gate passed, but the robustness rule fails |
| `COMPRESSION_REJECTED` | No candidate passes the provisional gate |
| `COMPRESSION_BLOCKED` | A correctness or accounting check fails and cannot be fixed within scope |

**Caveat.** The official test sets of MVTec AD, VisA and MPDD were already exposed during earlier work. A positive outcome is "benchmark-supported", not an unbiased held-out validation. No fresh dataset is downloaded or evaluated here.

**Not allowed:**
- changing any candidate, cap, quantiser, seed, order, metric, threshold or gate after results are seen;
- adding candidates;
- overwriting v1, existing artifacts, or caches;
- pushing to the remote.
