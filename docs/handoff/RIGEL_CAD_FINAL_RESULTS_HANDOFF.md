# RIGEL-CAD final results handoff

> **For readers outside the project.**
> - This is a copy of the internal report `agents/reports/g3e_paper_01.md` (G3E-PAPER-01, 2026-11-25). That report is not in git.
> - **G3-E is the research name of RIGEL-CAD**, the proposed method:
>   - arm `E@8` = RIGEL-CAD (8 MiB);
>   - `U` = RIGEL-CAD (unbounded);
>   - `F` = the ablation without the slow level.
> - **Code:**
>   - official package: `rigel_cad/` (see `rigel_cad/README.md`);
>   - frozen research code: `exps/g3e_paper_01/`, `exps/nl03_multirate_01/`.
> - **Machine-readable results:**
>   - `results/hope_cad/g3e_paper_01/RESULTS.json`;
>   - the same tables under the RIGEL-CAD names: `results/rigel_cad/paper/RIGEL_CAD_TABLES.md`.
> - **File locations.** Paths under `/mnt/k/...` are local caches and are not in the repository. `agents/...` files are internal and also not in git.

---

# G3E-PAPER-01: paper numbers for G3-E (proposed method, frozen)

**Status:** DONE. All 27 planned cells ran, and none is NOT_RUN. The job used 5.19 GPU-h of the 10 GPU-h ceiling, plus 2.37 CPU-h of metrics.

**Task:** `agents/tasks/task_ad2_dthr_g3e_paper_01.md`, PART 2. PART 1 (AD 2 D-THR) is reported in `agents/reports/nl03_multirate_01.md` §10.5–§10.7.

**Results:** `results/hope_cad/g3e_paper_01/RESULTS.json` (sha256 f450d20b9cd59e5e…). Every number below is read from that file.

**NL contribution: CONCLUDED (owner decision 2026-11-25).**
- The evidence is the byte-matched slow-level ablation E − F, i.e. with versus without the slow whitening level at 8 MiB. Its paired CI95 is above 0 on MVTec, VisA and MPDD with DINOv2 (§4).
- It is also above 0 on MVTec and VisA with the ViT-B/8 backbone (§6).
- NL is used here under the NL paper's definition (Def. 2–3, CMS Eq. 70–71), adopted by the owner. G3-E has two levels with different update frequencies:
  - fast per-task coreset keys;
  - a slow pooled whitening statistic that shapes how memory is read.
- The slow level is closed-form, not gradient-descent, so the mapping is PAPER_INFERRED.

---

## 0. Protocol (unchanged)

The protocol is NL03-MULTIRATE-01 §17, exactly. The method is frozen.

**Encoder and readout.**
- Frozen DINOv2-reg ViT-B/14 at 392 px, full-field square resize, 28×28 unit fp16 tokens.
- `g3.fit_task`: 1 % greedy coreset, plus a residual S that excludes the coreset members.
- `B.whiten(S, n)` with α = 0.5. Energy readout as in G3.
- A1 readout: `R2.maps_of` (smoothed 392 px), image score = top 1 %, native binned pixel AP, image AUROC.
- Final model, task-agnostic. Full train/good per category.

**G3-E.**
- At each task boundary, greedy k-centre (`PC.kc_select`) is run over the current keys ∪ the new task's coreset.
- The cap is (B − S_B)/1536 keys, with S_B = 2,359,304 bytes: 3925 keys at 8 MiB, 1194 at 4, 9386 at 16 and 20309 at 32.
- S is pooled over all tasks.
- Encoder, α, ratio, readout, eviction rule and byte accounting are unchanged.

**Seeds, orders, metrics.**
- Seeds are 2711, 2712 and 2713.
- Orders a (alphabetical), b and c are the registered MULTIRATE-01 orders.
- FM is forgetting on pixel AP: the mean over tasks of the max over stages minus the final value.
- acq_ap is the AP at the acquisition stage. prequential_ap is the mean over stages of the AP over the tasks seen so far.

**CIs.** CI95 are paired bootstrap intervals with the registered weights `M.boot_weights(n, cat_index)`, seeded by [BOOT_SEED, position]. They are paired across cells of the same order, and computed for order a only, as registered.

**Reuse.** The seed-2711 U and E@8 rows are the MULTIRATE-01 benchmark cells, reused and not recomputed. Their hashes are in `REUSE.json` (3ec3cce03d78cb38…), and each one's bootstrap sha256 was verified.

---

## 1. Tests before running (`results/hope_cad/g3e_paper_01/TESTS.json`, pass = true)

| Test | Result |
|---|---|
| `pytest exps/g3e_paper_01/test_g3e.py` (3 tests: caps/bytes, overflow fixture, small-stream learner) | 3 passed, rc 0 |
| Overflow fixture: `assert_budget(arm, cap+1)` raises BudgetViolation for every bounded arm; U and E@∞ never raise | pass |
| Byte assertion: `bytes_of(arm, cap) ≤ budget` for every bounded arm (keys × 1536 + slow bytes) | pass |
| T1, MPDD a\|2711: E@∞ ≡ U (keys, W, grids) | **bitwise**, max \|Δ\| = 0 (6/6 categories) |
| T1: the new driver's U and E@8 equal the MULTIRATE-01 U and E grids | **bitwise**, max \|Δ\| = 0 |

The budget was also asserted at every task boundary of every cell (`learner.assert_budget`), and no violation occurred.

---

## 2. Main table: order a, seed 2711 (final model)

| Dataset | Arm | Pixel AP [CI95] | Image AUROC [CI95] | FM | acq AP | Bytes | Update s | ms/img |
|---|---|---|---|---|---|---|---|---|
| MVTec | U | 0.6348 [0.6164, 0.6540] | 0.9953 [0.9917, 0.9979] | 0.0034 | 0.6293 | 46.07 MB | 98.3 | 9.93 |
| MVTec | **E@8** | **0.5927** [0.5747, 0.6123] | **0.9826** [0.9767, 0.9877] | 0.0333 | 0.6129 | 8,388,104 | 107.4 | 2.28 |
| VisA | U | 0.4868 [0.4573, 0.5129] | 0.9746 [0.9680, 0.9800] | 0.0030 | 0.4784 | 106.64 MB | 669.0 | 20.60 |
| VisA | **E@8** | **0.4603** [0.4298, 0.4853] | **0.9349** [0.9242, 0.9448] | 0.0119 | 0.4638 | 8,388,104 | 677.4 | 2.15 |
| MPDD | U | 0.5175 [0.5000, 0.5379] | 0.9627 [0.9449, 0.9785] | 0.0012 | 0.5178 | 13.06 MB | 17.7 | 2.66 |
| MPDD | **E@8** | **0.5082** [0.4895, 0.5296] | **0.9399** [0.9164, 0.9606] | 0.0128 | 0.5163 | 8,388,104 | 20.3 | 1.89 |

- **U − E pixel AP (paired):** MVTec +0.0421 [0.0367, 0.0468], VisA +0.0265 [0.0201, 0.0338], MPDD +0.0092 [0.0015, 0.0186].
- 8,388,104 B = 8 MiB − 504 B.
- Update s is the total stream update wall time on the GPU. ms/img is the p50 inference latency.

---

## 3. A1: MPDD order c, seed 2711 (missing in MULTIRATE-01; now run)

| Arm | Pixel AP | Image AUROC | FM | acq AP | Prequential AP | Bytes |
|---|---|---|---|---|---|---|
| U | 0.5175 | 0.9627 | 0.0036 | 0.5180 | 0.6687 | 13,057,544 |
| E@8 | 0.5020 | 0.9381 | 0.0115 | 0.5103 | 0.6623 | 8,388,104 |

There are no CIs here, because the bootstrap is registered for order a only. U's final state is order-invariant by construction (union of per-task coresets plus pooled S), so its final AP equals the order-a value.

---

## 4. A4: ablations at 8 MiB (order a, seed 2711), paired E − X

**The arms.**
- **Q:** hindsight k-centre over the union of *all* seen coresets, at E's cap. Q has to keep every past coreset to make its choice, so it is a hindsight selection reference, not a streaming-feasible method within 8 MiB.
- **E4:** E's keys, with W built from the current task's S only.
- **F:** Euclidean (W = I), all 8 MiB spent on keys (5461 keys).
- **G:** a generic covariance W (canonical G1b), at (B − G1_B)/1536 = 3923 keys.
- **FIFO:** evicts the oldest keys first.
- **B1:** E's keys without whitening (6.03 MB).

| Dataset | X | Pixel AP | Image AUROC | FM | E − X AP [CI95] | E − X AUROC [CI95] |
|---|---|---|---|---|---|---|
| MVTec | (E) | 0.5927 | 0.9826 | 0.0333 | | |
| MVTec | Q | 0.6092 | 0.9865 | 0.0211 | **−0.0166** [−0.0210, −0.0123] | −0.0039 [−0.0096, 0.0015] |
| MVTec | E4 | 0.5885 | 0.9793 | 0.0550 | +0.0042 [0.0003, 0.0087] | +0.0033 [−0.0007, 0.0069] |
| MVTec | F | 0.5681 | 0.9740 | 0.0368 | **+0.0245** [0.0177, 0.0335] | +0.0086 [0.0010, 0.0155] |
| MVTec | G | 0.5970 | 0.9786 | 0.0216 | −0.0043 [−0.0103, 0.0018] | +0.0040 [−0.0018, 0.0100] |
| MVTec | FIFO | 0.3945 | 0.8520 | 0.2614 | +0.1981 [0.1824, 0.2131] | +0.1306 [0.1125, 0.1493] |
| MVTec | B1 | 0.5425 | 0.9702 | 0.0488 | +0.0501 [0.0438, 0.0581] | +0.0124 [0.0066, 0.0183] |
| VisA | (E) | 0.4603 | 0.9349 | 0.0119 | | |
| VisA | Q | 0.4518 | 0.9516 | 0.0179 | +0.0084 [0.0019, 0.0140] | **−0.0167** [−0.0255, −0.0078] |
| VisA | E4 | 0.4492 | 0.9189 | 0.0337 | +0.0110 [0.0078, 0.0145] | +0.0160 [0.0107, 0.0213] |
| VisA | F | 0.3816 | 0.8760 | 0.0406 | **+0.0786** [0.0693, 0.0878] | +0.0589 [0.0454, 0.0727] |
| VisA | G | 0.4359 | 0.9415 | 0.0214 | +0.0244 [0.0145, 0.0322] | −0.0066 [−0.0159, 0.0020] |
| VisA | FIFO | 0.1668 | 0.6764 | 0.3366 | +0.2934 [0.2713, 0.3131] | +0.2586 [0.2362, 0.2812] |
| VisA | B1 | 0.3725 | 0.8478 | 0.0424 | +0.0878 [0.0802, 0.0963] | +0.0871 [0.0739, 0.1000] |
| MPDD | (E) | 0.5082 | 0.9399 | 0.0128 | | |
| MPDD | Q | 0.5047 | 0.9338 | 0.0169 | +0.0035 [−0.0035, 0.0106] | +0.0062 [−0.0068, 0.0187] |
| MPDD | E4 | 0.5036 | 0.9388 | 0.0221 | +0.0046 [0.0001, 0.0104] | +0.0011 [−0.0064, 0.0083] |
| MPDD | F | 0.4733 | 0.9550 | 0.0211 | **+0.0349** [0.0234, 0.0460] | −0.0151 [−0.0324, 0.0015] |
| MPDD | G | 0.4911 | 0.9349 | 0.0333 | +0.0172 [0.0101, 0.0240] | +0.0050 [−0.0082, 0.0183] |
| MPDD | FIFO | 0.4657 | 0.9078 | 0.0638 | +0.0426 [0.0302, 0.0543] | +0.0321 [0.0011, 0.0607] |
| MPDD | B1 | 0.4550 | 0.9466 | 0.0280 | +0.0532 [0.0437, 0.0644] | −0.0067 [−0.0261, 0.0131] |

**Reading (facts only).**
- **Pixel AP.** E's CI is above 0 against:
  - F, B1 and FIFO on 3/3 datasets;
  - E4 on 3/3, with the MVTec and MPDD lower bounds close to 0;
  - G on VisA and MPDD (MVTec is a tie);
  - Q on VisA.
- **Q > E on MVTec pixel AP** (−0.0166) and on VisA image AUROC (−0.0167). Q is a hindsight reference that needs all past coresets.
- **Image AUROC.** F is higher than E on MPDD (−0.0151), but the CI crosses 0.

---

## 5. A3: budget curve (order a, 3 seeds, mean ± std)

| Dataset | Arm | Bytes | Pixel AP | Image AUROC | FM | ms/img |
|---|---|---|---|---|---|---|
| MVTec | E@4 | 4,193,288 | 0.5583 ± 0.0151 | 0.9661 ± 0.0070 | 0.0445 | 1.09 |
| MVTec | E@8 | 8,388,104 | 0.5946 ± 0.0063 | 0.9848 ± 0.0020 | 0.0286 | 2.13 |
| MVTec | E@16 | 16,776,200 | 0.6201 ± 0.0016 | 0.9898 ± 0.0023 | 0.0177 | 4.13 |
| MVTec | E@32 | 33,553,928 | 0.6326 ± 0.0028 | 0.9946 ± 0.0005 | 0.0056 | 8.56 |
| MVTec | U | 46,070,792 | 0.6342 ± 0.0004 | 0.9954 ± 0.0004 | 0.0030 | 9.75 |
| VisA | E@4 | 4,193,288 | 0.3882 ± 0.0199 | 0.8996 ± 0.0071 | 0.0634 | 1.07 |
| VisA | E@8 | 8,388,104 | 0.4540 ± 0.0056 | 0.9419 ± 0.0084 | 0.0153 | 2.03 |
| VisA | E@16 | 16,776,200 | 0.4651 ± 0.0045 | 0.9627 ± 0.0013 | 0.0103 | 4.11 |
| VisA | E@32 | 33,553,928 | 0.4784 ± 0.0021 | 0.9696 ± 0.0017 | 0.0051 | 8.55 |
| VisA | U | 106,638,344 | 0.4881 ± 0.0012 | 0.9738 ± 0.0013 | 0.0033 | 20.20 |
| MPDD | E@4 | 4,193,288 | 0.4618 ± 0.0097 | 0.8986 ± 0.0182 | 0.0348 | 1.16 |
| MPDD | E@8 | 8,388,104 | 0.5099 ± 0.0054 | 0.9455 ± 0.0091 | 0.0114 | 2.01 |
| MPDD | E@16 / E@32 / U | 13,057,544 | 0.5175 ± 0.0053 | 0.9616 ± 0.0015 | 0.0010 | 2.6–2.7 |

- On MPDD, U needs only 13.06 MB. At 16 and 32 MiB the budget never binds, so E@16 = E@32 = U exactly. This is a consistency check that holds.
- The curve is monotone on every dataset.
- E@32 reaches 99.7 % (MVTec) and 98.0 % (VisA) of U's pixel AP, at 73 % and 31 % of U's bytes.
- Latency scales with the number of keys: E@8 is 4.6× (MVTec) and 9.9× (VisA) faster than U.

---

## 6. B: matched backbone, ViT-B/8 IN-21k at 224 (MATCHED_BACKBONE_ABLATION; order a, seed 2711)

**Contract.** Documented in `exps/g3e_paper_01/vitb8.py` before running.
- Model: timm `vit_base_patch8_224` augreg in21k, local checkpoint `checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth` (sha256 755e9f5f…a415). It was already on disk, so nothing was downloaded.
- Features: output of block 9, before the final norm, with CLS dropped: 784 = 28×28 tokens × 768.
- Input: PIL RGB, bilinear direct resize to 224 (the same full-field geometry, so the A1 readout is unchanged).
- Normalisation: ImageNet mean/std. This is the repository's CADIC contract. timm's augreg default would be 0.5/0.5; this is a recorded ambiguity and was not tuned.
- Tokens: unit-L2 fp16.
- The cache is new, under `/mnt/k/NLCAD/cache/g3e_paper_01/vitb8/` (110 files, with per-file sha in a json). The canonical caches were not touched.
- Test row order was checked against the DINOv2 benchmark rows.

| Dataset | Arm | Pixel AP [CI95] | Image AUROC | FM | acq AP | Bytes | ms/img |
|---|---|---|---|---|---|---|---|
| MVTec | U | 0.5748 [0.5532, 0.5936] | 0.9453 | 0.0050 | 0.5755 | 46,070,792 | 9.86 |
| MVTec | **E@8** | **0.5018** [0.4789, 0.5207] | **0.8869** | 0.0630 | 0.5490 | 8,388,104 | 2.17 |
| MVTec | F | 0.4232 [0.3987, 0.4419] | 0.9018 | 0.1246 | 0.5349 | 8,388,096 | 2.40 |
| VisA | U | 0.2910 [0.2679, 0.3125] | 0.8783 | 0.0023 | 0.2741 | 106,638,344 | 20.34 |
| VisA | **E@8** | **0.2356** [0.2105, 0.2541] | **0.7644** | 0.0341 | 0.2420 | 8,388,104 | 2.12 |
| VisA | F | 0.2176 [0.1936, 0.2361] | 0.7863 | 0.0330 | 0.2378 | 8,388,096 | 2.30 |

- **E − F pixel AP (paired):** MVTec +0.0786 [0.0708, 0.0888]; VisA +0.0179 [0.0098, 0.0249].
- **U − E pixel AP (paired):** MVTec +0.0731 [0.0655, 0.0817]; VisA +0.0554 [0.0471, 0.0674].

**For reference only:** CADIC (ViT-B/8) as reported by its authors. This is not a matched run, and the protocol may differ.
- 10k patches (≈ 30.7 MB): MVTec 0.972 I-AUROC / 0.584 P-AUPR; VisA 0.891 / 0.438.
- 2.5k patches (≈ 7.7 MB): MVTec 0.955 / 0.559; VisA 0.828 / 0.415.
- Source: `agents/reports/lit_01_cad_benchmark_accuracy_memory_time.md`.

**Reading (facts only).**
- With this backbone contract, U and E@8 are below CADIC's reported numbers on both datasets.
- The advantage over CADIC in the main comparison table therefore depends on the DINOv2 at 392 front end, not on the memory rule alone.
- The slow-level effect (E > F) holds on both backbones.
- The normalisation ambiguity above is one untested possible contributor to the low ViT-B/8 numbers.

---

## 7. A2: robustness over orders × seeds (3 × 3 = 9 cells per dataset, mean ± std)

| Dataset | Arm | n | Pixel AP | Image AUROC | FM | acq AP |
|---|---|---|---|---|---|---|
| MVTec | U | 9 | 0.6342 ± 0.0004 | 0.9954 ± 0.0003 | 0.0047 ± 0.0047 | 0.6315 ± 0.0032 |
| MVTec | **E@8** | 9 | **0.5944 ± 0.0063** | **0.9832 ± 0.0029** | 0.0314 ± 0.0075 | 0.6157 ± 0.0050 |
| VisA | U | 9 | 0.4881 ± 0.0010 | 0.9738 ± 0.0011 | 0.0036 ± 0.0010 | 0.4806 ± 0.0043 |
| VisA | **E@8** | 9 | **0.4505 ± 0.0052** | **0.9457 ± 0.0056** | 0.0225 ± 0.0091 | 0.4635 ± 0.0068 |
| MPDD | U | 9 | 0.5175 ± 0.0046 | 0.9616 ± 0.0013 | 0.0015 ± 0.0017 | 0.5169 ± 0.0048 |
| MPDD | **E@8** | 9 | **0.5102 ± 0.0078** | **0.9473 ± 0.0070** | 0.0104 ± 0.0025 | 0.5165 ± 0.0059 |

**E@8 pixel AP per cell** (order|seed: a|2711, a|2712, a|2713, b|2711, b|2712, b|2713, c|2711, c|2712, c|2713):

| Dataset | Values | Range |
|---|---|---|
| MVTec | .5927 .6016 .5895 .5930 .5925 .5879 .5974 .6065 .5889 | 0.588–0.607 |
| VisA | .4603 .4522 .4494 .4489 .4497 .4409 .4475 .4520 .4535 | 0.441–0.460 |
| MPDD | .5082 .5160 .5056 .5030 .5164 .5028 .5020 .5244 .5131 | 0.502–0.524 |

**Notes.**
- U's final state is order-invariant, so its std reflects the coreset seed only.
- Prequential AP depends on the order's composition. MPDD's std is 0.168, so it is not averaged across orders in the paper; it is in RESULTS.json.
- Order-a seed 2711 (§2) lies inside the 9-cell range on every dataset.
- The seed-2711 cells of orders b and c are the reused MULTIRATE-01 cells.

---

## 8. A5: per-class pixel AP (final model, order a, seed 2711)

Columns: U final AP, E@8 final AP, U AUROC, E@8 AUROC, E@8 acq AP, E@8 FM. The last task in the order has no FM (—).

### MVTec

| Class | U AP | E AP | U AUROC | E AUROC | E acq AP | E FM |
|---|---|---|---|---|---|---|
| bottle | 0.855 | 0.833 | 1.000 | 1.000 | 0.842 | 0.019 |
| cable | 0.497 | 0.485 | 0.991 | 0.978 | 0.487 | 0.061 |
| capsule | 0.590 | 0.472 | 0.978 | 0.897 | 0.592 | 0.120 |
| carpet | 0.727 | 0.703 | 1.000 | 1.000 | 0.702 | 0.032 |
| grid | 0.494 | 0.485 | 1.000 | 1.000 | 0.509 | 0.024 |
| hazelnut | 0.808 | 0.756 | 1.000 | 1.000 | 0.764 | 0.011 |
| leather | 0.468 | 0.427 | 1.000 | 1.000 | 0.449 | 0.054 |
| metal_nut | 0.723 | 0.637 | 1.000 | 1.000 | 0.680 | 0.043 |
| pill | 0.575 | 0.510 | 0.994 | 0.989 | 0.541 | 0.038 |
| screw | 0.600 | 0.491 | 0.983 | 0.887 | 0.544 | 0.053 |
| tile | 0.727 | 0.725 | 1.000 | 1.000 | 0.707 | −0.001 |
| toothbrush | 0.669 | 0.661 | 1.000 | 1.000 | 0.674 | 0.013 |
| transistor | 0.509 | 0.449 | 0.991 | 0.996 | 0.451 | 0.002 |
| wood | 0.679 | 0.669 | 0.992 | 0.994 | 0.667 | −0.003 |
| zipper | 0.601 | 0.586 | 1.000 | 0.998 | 0.586 | — |

### VisA

| Class | U AP | E AP | U AUROC | E AUROC | E acq AP | E FM |
|---|---|---|---|---|---|---|
| candle | 0.444 | 0.422 | 0.977 | 0.906 | 0.426 | 0.008 |
| capsules | 0.628 | 0.606 | 0.967 | 0.937 | 0.576 | −0.002 |
| cashew | 0.692 | 0.652 | 0.968 | 0.938 | 0.671 | 0.030 |
| chewinggum | 0.702 | 0.671 | 0.990 | 0.994 | 0.682 | 0.018 |
| fryum | 0.467 | 0.411 | 0.958 | 0.905 | 0.431 | 0.031 |
| macaroni1 | 0.157 | 0.182 | 0.933 | 0.867 | 0.165 | −0.000 |
| macaroni2 | 0.137 | 0.094 | 0.946 | 0.839 | 0.110 | 0.020 |
| pcb1 | 0.798 | 0.784 | 0.996 | 0.955 | 0.779 | 0.004 |
| pcb2 | 0.427 | 0.417 | 0.982 | 0.943 | 0.421 | 0.004 |
| pcb3 | 0.444 | 0.391 | 0.993 | 0.971 | 0.404 | 0.013 |
| pcb4 | 0.419 | 0.375 | 0.997 | 0.992 | 0.381 | 0.006 |
| pipe_fryum | 0.527 | 0.520 | 0.989 | 0.974 | 0.520 | — |

### MPDD

| Class | U AP | E AP | U AUROC | E AUROC | E acq AP | E FM |
|---|---|---|---|---|---|---|
| bracket_black | 0.197 | 0.182 | 0.872 | 0.842 | 0.202 | 0.020 |
| bracket_brown | 0.407 | 0.383 | 0.975 | 0.922 | 0.400 | 0.032 |
| bracket_white | 0.081 | 0.061 | 0.982 | 0.932 | 0.082 | 0.022 |
| connector | 0.758 | 0.767 | 1.000 | 1.000 | 0.756 | −0.011 |
| metal_plate | 0.885 | 0.884 | 1.000 | 1.000 | 0.885 | 0.001 |
| tubes | 0.777 | 0.772 | 0.947 | 0.942 | 0.772 | — |

**Largest E losses versus U:** MVTec capsule (−0.118 AP, FM 0.120), screw (−0.109) and metal_nut (−0.086). On these classes the image AUROC also drops: capsule 0.897, screw 0.887.

---

## 9. MVTec AD 2 held-out result (from PART 1; owner-reported server numbers)

- Arm E (G3-E, 8 MiB, DINOv2 at 392) was submitted once by the owner as **RIGEL-CAD**, with the D-THR threshold.
- AU-PRO 0.05 (test_private / test_private_mixed) is **38.2 / 34.0**. SegF1 is 20.5 / 15.3; ClassF1 is 75.4 / 76.6.
- That AU-PRO is above the dataset authors' 256 px per-category baselines (the best is EfficientAD at 30.8 / 19.2).
- It is far below high-resolution PatchCore (62.3 / 52.6) and the leaderboard top (about 83–89).
- Details are in `agents/reports/nl03_multirate_01.md` §10.5.
- U was not submitted (owner's choice, possibly added later).

---

## 10. Publication claim audit

| Claim | Status |
|---|---|
| NL contribution: a two-level, two-frequency memory (fast keys plus slow pooled whitening), whose slow level beats spending the same bytes on keys | **Can claim** (owner decision 2026-11-25). E − F > 0 with CI on 3/3 datasets (DINOv2) and 2/2 (ViT-B/8). Mapping to NL Def. 2–3 is PAPER_INFERRED (closed-form slow level) |
| G3-E at 8 MiB keeps 93–98 % of unbounded pixel AP (MVTec 0.593/0.635, VisA 0.460/0.487, MPDD 0.508/0.518) at 7.9–64 % of U's bytes | **Can claim** (measured, with CIs). The U − E gap is significant on all 3 datasets |
| Robust over orders and seeds | **Can claim.** E std ≤ 0.008 AP over 9 cells |
| Smooth budget curve; E@32 ≈ U | **Can claim** (A3) |
| Higher raw numbers than published CAD methods (CADIC, UCAD, ReplayCAD) | **Only with the backbone stated.** Unmatched backbone (DINOv2 at 392). With matched ViT-B/8, G3-E is below CADIC's reported numbers (§6) |
| AD 2: above the dataset authors' 256 px baselines | **Can claim with qualifiers:** 392 px DINOv2 vs 256 px per-category models; single continual model; one submission. Far below the leaderboard top. Do not call it SOTA |
| Task-boundary k-centre is the best selection rule | **Cannot claim.** Hindsight Q is better on MVTec pixel AP and VisA image AUROC |
| Image AUROC parity with U | **Cannot claim.** For example, VisA 0.935 vs 0.975 |

---

## 11. Deviations, incidents and skipped items

| ID | Item | Effect |
|---|---|---|
| D-TAB | `exps/g3e_paper_01/tables.py` crashed while aggregating A2. The reused MULTIRATE-01 order b/c METRICS files carry a `boot_file` key that points to an empty npz, because the bootstrap is order-a only. Fixed before the final `tables` step by using the bootstrap only when the arm's keys are present. | Aggregation only. No number changed |
| Part 1 | `ad2_aupro` needed `tifffile` from the existing local `pytorch_env` (no download). Recorded in nl03_multirate_01 §10.7 | None |
| Incident | `pkill -f pipeline_part2.sh` killed the invoking shell (exit 144). The old pipeline had already exited, and it was relaunched as `logs/pipeline_cont.sh` | None; cells are resumable |
| Ambiguity | ViT-B/8 normalisation: ImageNet mean/std (repository contract) instead of timm's augreg 0.5/0.5. Documented before running, not tuned | May lower the B numbers |
| NOT_RUN | None. 27/27 cells are DONE (`PLAN_STATUS.json`) | — |

- No method change.
- Nothing was downloaded in PART 2. Nothing was uploaded.
- At most one GPU job ran at a time, and the CPU metrics pool was 2.
- **Disk at the end:** D: 275 GB, E: 127 GB and K: 92 GB free (≥ 30 GiB each). The new cache under `/mnt/k/NLCAD/cache/g3e_paper_01` is 21 GB. No cache or log was deleted.

---

## 12. Budget (`results/hope_cad/g3e_paper_01/BUDGET_LEDGER.jsonl`)

| Phase | Hours |
|---|---|
| GPU cells, including ViT-B/8 extraction inside the B cells | 5.19 (ceiling 10; projected 7.79) |
| CPU metrics | 2.37 |
| Wall time | 07:36 → 15:11 on 2026-11-25 |

---

## 13. Reproducibility

Commands, run from the repo root with `~/miniconda3/envs/nested_learning/bin/python`:

```
python -B -m pytest -q exps/g3e_paper_01/test_g3e.py
python -B -m exps.g3e_paper_01.run reuse      # REUSE.json (hash-verified MULTIRATE-01 cells)
python -B -m exps.g3e_paper_01.run tests      # TESTS.json (pytest + T1 bitwise checks)
python -B -m exps.g3e_paper_01.run plan       # all cells, priority A1 > A4 > A3 > B > A2, resumable, auto-stop at 10 GPU-h
python -B -m exps.g3e_paper_01.tables         # RESULTS.json
```

- Pipeline: `results/hope_cad/g3e_paper_01/logs/pipeline_cont.sh`.
- Logs: `logs/pipeline_cont.log`, `logs/run.log` and `logs/part2_*.log`.
- Per-cell artifacts: `results/hope_cad/g3e_paper_01/cells/SCORE_*.json` and `METRICS_*.json`, plus bootstrap npz files. Grids are under `/mnt/k/NLCAD/cache/g3e_paper_01/cells/`.

| File | sha256 (16) |
|---|---|
| exps/g3e_paper_01/learner.py | 94e44284e2df2d6b |
| exps/g3e_paper_01/run.py | ff70a3c8025affeb |
| exps/g3e_paper_01/tables.py (after D-TAB) | f7327ebc201f8fbc |
| exps/g3e_paper_01/test_g3e.py | 83f2bac259473494 |
| exps/g3e_paper_01/vitb8.py | 7b06e2e280debff6 |
| RESULTS.json | f450d20b9cd59e5e |
| TESTS.json | 262c2f23235aa5ca |
| REUSE.json | 3ec3cce03d78cb38 |
| PLAN_STATUS.json | 24d42d57c75f36a2 |
| PROJECTION.json | 601af40c2a9f8440 |

---

## Handoff

**Task:** G3E-PAPER-01 (owner task `agents/tasks/task_ad2_dthr_g3e_paper_01.md`, PART 2; PART 1 is DONE and reported in `nl03_multirate_01.md`).

**Status:** DONE.

**Last agent:** Codex (role held by Claude Code Opus).

**Next agent:** User.

**Branch / HEAD:** dikhang_space / ca91e00 (no commit, no push by Claude).

**Completed work:**
- Tests: bitwise E@∞ ≡ U, bitwise reproduction of MULTIRATE-01, byte assertion and overflow fixture.
- A1, A2 (9 cells × 3 datasets), A3 (3 seeds × 4 budgets × 3 datasets), A4 (6 ablations × 3 datasets, paired CIs), A5 (per-class), and B (ViT-B/8, MVTec and VisA).
- RESULTS.json and this report.

**Changed files:**
- New: `exps/g3e_paper_01/*` (6 files), `results/hope_cad/g3e_paper_01/**`, `/mnt/k/NLCAD/cache/g3e_paper_01/**` (new cache, 21 GB) and this report.
- PART 1 files are listed in `nl03_multirate_01.md`.
- Updated: §10.5 and §13 of `agents/reports/nl03_multirate_01.md`, with the owner-reported server numbers.
- `git status --short` shows only `M exps/g3e_paper_01/tables.py` relative to HEAD, which is D-TAB. The rest of `exps/g3e_paper_01/` was already in HEAD ca91e00, committed by the owner.

**Verification / Tests:**
- `pytest exps/g3e_paper_01/test_g3e.py`: 3 passed.
- T1 is bitwise for E@∞ ≡ U and for MR U/E reproduction (TESTS.json `pass: true`).
- Budget asserted at every boundary, with no violation.
- PLAN_STATUS: 27/27 DONE.

**Unresolved issues:**
- The ViT-B/8 normalisation ambiguity is untested.
- Hindsight Q beats E on MVTec pixel AP.
- The U AD 2 submission is optional (owner).

**Exact next action:** the owner uses §2–§8 for the paper, with the claim audit in §10 and the CADIC comparison qualified by backbone. The next agent may proceed only on a new owner task.
