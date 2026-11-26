# RIGEL-CAD code and architecture report

> **Erratum, 2026-11-26.** This report's local MVTec AD 2 test_public AU-PRO / SegF1 numbers (for example, a mean AU-PRO 0.05 of 0.2633) were computed with a defective mask lookup: all 379 good images received the defect mask of the bad image with the same file name. The corrected values are 0.3594 / 0.5921 / 0.1680 (AU-PRO 0.05 / 0.30 / SegF1). Everything else is unaffected: server numbers (38.2 / 34.0), thresholds, submission maps, and MVTec AD / VisA / MPDD. The release report that supersedes this one is `docs/handoff/RIGEL_CAD_RELEASE_REPORT.md`.

> **For readers outside the project.**
> - This is a copy of the internal review report `agents/reports/rigel_cad_01.md` (2026-11-26). That report is not in git.
> - It describes the official package `rigel_cad/`:
>   - method as implemented;
>   - architecture and data flow;
>   - module table with sha256s;
>   - public API and the research → official renaming map;
>   - design decisions;
>   - tests and end-to-end checks (MPDD and MVTec AD 2, bitwise equal to the locked research results);
>   - known limitations and a review checklist.
> - Companion results handoff: `docs/handoff/RIGEL_CAD_FINAL_RESULTS_HANDOFF.md`. Usage: `rigel_cad/README.md`.
> - Paths under `/mnt/k/...` are local caches, and `agents/...` files are internal. Neither is in the repository.

---

# RIGEL-CAD-01: official source package `rigel_cad/` (architecture, implementation, verification)

**Purpose of this report.** Independent code review of the official RIGEL-CAD package before paper release.

**Background.** RIGEL-CAD is the proposed method, called **G3-E** in the research code. The owner asked (2026-11-26) for two things:
- keep `exps/` unchanged;
- rename the official components, API and artifacts to RIGEL-CAD, and make the code complete.

**Pushed:**
- `dikhang_space` @ `3cd24b2`: package + results;
- `dikhang_space` @ `8b5e04f`: two small fixes, see §8;
- `dikhang_space` @ `c561dc9`: docs handoff copy of G3E-PAPER-01;
- `dikhang_space` @ `12514e2`: the MVTec AD 2 end-to-end result (§9).

---

## 1. Summary

**What the package is.**
- `rigel_cad/` is a self-contained package: 13 modules plus 2 test files, about 1,860 lines including the README.
- It does **not import anything from `exps/`**. Only `tests/test_equivalence.py` imports `exps/`, to compare against it.
- It covers:
  - the method: fast level, slow level, scoring;
  - the frozen encoders;
  - the A1 read-out;
  - the full evaluation protocol: pixel AP, AUROC, bootstrap, forgetting, AU-PRO, SegF1;
  - the MVTec AD 2 pipeline: threshold plus submission files;
  - a CLI, paper-table export under the new names, and a README.

**What was verified:**
- 9 CPU unit tests.
- 6 equivalence tests against the frozen research code and its locked artifacts, all **bitwise** on this machine.
- An end-to-end CLI run from raw images on MPDD reproduces the locked paper number:

| | CLI run | Locked |
|---|---|---|
| Pixel AP | 0.50822631 | 0.50822630 |
| Image AUROC | 0.93990785 | 0.93990785 (identical) |
| Bytes | 8,388,104 | 8,388,104 |

- An end-to-end CLI run from raw images on **MVTec AD 2** (all 8 scenarios, every stage) reproduces the locked arm-E results **bitwise**:
  - θ for all 8 scenarios;
  - 80 per-stage test_public values of AU-PRO 0.05 / 0.30 and SegF1;
  - mean AU-PRO 0.05 = 0.26331970, forgetting 0.00772843;
  - 300 of 300 sampled submission files (TIFF and PNG) are byte-identical to the submitted folder. See §9.

**What was not done:**
- No `pyproject.toml` or pip packaging. The package runs from the repo root.

---

## 2. Method as implemented

**Notation.**
- A task t has normal training images, in sorted path order.
- Each image gives 28×28 = 784 tokens x ∈ ℝ⁷⁶⁸. They are unit-L2 and stored in fp16.
- N_t is the number of tokens of task t.

**Fast level (`FastKeyMemory`), written once per task:**
1. k_t = ⌈0.01·N_t⌉.
2. C_t = greedy k-centre(x, k_t, seed), run on a seeded Gaussian projection to 128 dimensions. It returns indices in selection order, so every prefix is itself a valid smaller selection.
3. Consolidation:
   - If B = ∞, every C_t is kept (the union of all coresets).
   - Otherwise K ← C ∪ C_t. If |K| > cap, K is reduced by greedy k-centre(K, cap, seed), and the survivors keep their insertion order.
   - The ablation `selection='fifo'` evicts the oldest keys instead.

**Slow level (`SlowResidualWhitener`), accumulated over the stream:**
- For every token of task t that is not a key: r = x − (nearest key in C_t). Then S_t = Σ r rᵀ in fp64, and n_t is the number of such tokens.
- The pooled statistic is S = Σ_t S_t and n = Σ_t n_t.
- Read-out metric: W = (½·S/n + ½·(tr(S/n)/768)·I)^(-1/2), computed by an fp64 eigendecomposition. It raises an error if λ_min ≤ 1e-12.

**Scoring.**
- Patch energy: e(q) = ½·min_m ‖qW − mW‖², in fp32, over chunks of 512 query rows.
- The ablation `slow_level=False` uses W = I and spends the whole budget on keys.

**Read-out (A1):**
1. 28×28 → 392×392 bilinear (align_corners = False).
2. Separable Gaussian, σ = 4, 33 taps, zero padding.
3. Batch shift so that the minimum is ≥ 0.
4. Image score = mean of the top 1 % of the map (1,536 pixels).

**Persistent bytes:**
- bytes = |K|·1,536 + 2,359,304 (S in fp32 plus an fp64 count).
- Caps: 3925 keys at 8 MiB, 1194 at 4, 9386 at 16 and 20309 at 32. The fast-only cap is 5461.
- `BudgetViolation` is raised if the state would exceed the budget.
- The encoder is not counted (144,806,400 B, shared and frozen).

**NL framing.** There are two levels with different update frequencies. This uses the NL paper's definition (Def. 2–3, CMS Eq. 70–71), adopted by the owner. The slow level is closed-form, so the mapping is PAPER_INFERRED.

---

## 3. Architecture and data flow

```
images (sorted paths) ─► encoders.RigelEncoder.encode_paths ─► tokens [n,784,768] fp16 unit
                                    │  (TokenStore: optional cache <cache>/<backbone>/<key>.pt + sha256 json)
                                    ▼
             RigelCAD.fit_task(tokens[N,768]) ── ops.coreset_size / ops.greedy_kcenter (C_t)
                    │                         └─ ops.residual_moment (S_t, n_t)
                    ├─► FastKeyMemory.write(C_t)           (k-centre or FIFO consolidation to cap)
                    ├─► SlowResidualWhitener.write(S_t,n_t)
                    └─► state() -> RigelState(keys, W, n_tasks, bytes)  [+ history if keep_history]
                                    ▼
 RigelCAD.patch_energy_grid(test tokens, state_at(l)) ─► ops.energy_whitened / energy_euclidean ─► [n,28,28]
                                    ▼
 readout.anomaly_maps ─► [n,392,392] ─► readout.image_scores (image) / to_native_torch (pixel AP)
                                    │                                 / to_native_cv2 (AD 2 maps)
                                    ▼
 metrics.evaluate_category (per stage) ─► metrics.stream_summary ─► RESULTS_<tag>.json (+ BOOT npz, .rigel state)
 ad2.validation_threshold / metrics.ProAccumulator / metrics.SegF1 / ad2.write_maps ─► AD 2 results + submission
```

### Modules

| File | Lines | sha256 (16) | Content |
|---|---|---|---|
| `config.py` | 99 | 8005f6d686a562d8 | frozen constants; `RigelConfig` (budget_mib, slow_level, selection, seed, backbone, device), `key_capacity`, `budget_bytes` |
| `ops.py` | 113 | 020875417fc19cb7 | `unit`, `coreset_size`, `greedy_kcenter`, `sqdist`, `residual_moment`, `shrink`, `inv_sqrt_psd`, `whitener`, `energy_whitened`, `energy_euclidean` |
| `model.py` | 222 | fd714142e2d61e7d | `FastKeyMemory`, `SlowResidualWhitener`, `RigelState`, `RigelCAD` (`fit_task`, `state`, `state_at`, `patch_energy(_grid)`, `save`/`load` with format `rigel-cad-state/1`), `BudgetViolation` |
| `encoders.py` | 242 | b7d9fbe4b8241408 | `RigelDinoEncoder` (own DINOv2-reg ViT-B/14 forward, strict load, mean of blocks 6–9 (0-based), prefix removed); `RigelViTB8Encoder` (timm, block 9 (1-based) hook, CLS removed); `preprocess`; `build_encoder`; checkpoint sha256 checks |
| `readout.py` | 43 | 345ed210ac173462 | `anomaly_maps`, `image_scores`, `to_native_torch`, `to_native_cv2` |
| `data.py` | 106 | cf12dca50410117e | category lists; `stream_orders` (a / b / c); MVTec/MPDD folder and VisA csv splits; AD 2 file listing and masks |
| `metrics.py` | 203 | d15c47b58ed8062e | `binned_ap`, `image_hists`, `boot_weights`, `Category`, `evaluate_category`, `stream_summary`, `ci95`; `ProAccumulator` (AU-PRO), `trapezoid`, `SegF1` |
| `ad2.py` | 54 | 545d6f9b8bc878cf | `validation_threshold` (mean + 3σ), `write_maps` (fp16 zlib TIFF + binary PNG) |
| `pipeline.py` | 184 | 203b4fa90f399df6 | `TokenStore`, `run_stream`, `run_ad2` |
| `cli.py` / `__main__.py` | 49 / 3 | c25d4d1ab668d1cc | `python -m rigel_cad stream` and `python -m rigel_cad ad2` |
| `paper_tables.py` | 96 | 3028b8ab470ca858 | renames the locked G3E-PAPER-01 tables (RIGEL-CAD (8 MiB), Ablation: …) and writes `results/rigel_cad/paper/RIGEL_CAD_{RESULTS.json,TABLES.md}` |
| `README.md` | 175 | 5c7c9777ae889b47 | usage, layouts, protocol, provenance map, tests, limitations |
| `tests/test_unit.py` | 115 | 9db0b4cad28c9b36 | 9 CPU tests |
| `tests/test_equivalence.py` | 154 | 09269f37b283e51d | 6 equivalence tests (CUDA plus locked caches; skipped otherwise) |

### Public API

```python
from rigel_cad import RigelCAD, RigelConfig
m = RigelCAD(RigelConfig(budget_mib=8), keep_history=True)
m.fit_task(tokens16)                   # [N,768] fp16 unit tokens of one task -> RigelState
m.patch_energy_grid(test_tokens, m.state_at(l))  # [n,28,28]
m.persistent_bytes(); m.save('x.rigel'); RigelCAD.load('x.rigel')
```

### Renaming map (research → official)

| Research | Official |
|---|---|
| G3-E | RIGEL-CAD |
| arm E@B | `RigelConfig(budget_mib=B)` |
| U = E@∞ | `budget_mib=None` ("RIGEL-CAD (unbounded)") |
| F | `slow_level=False` |
| FIFO | `selection='fifo'` |
| fast keys | `FastKeyMemory` |
| pooled S / W | `SlowResidualWhitener` |
| `g3.score` | `ops.energy_whitened` |
| `R2.maps_of` | `readout.anomaly_maps` |
| `A.greedy_coreset` | `ops.greedy_kcenter` |

**Artifacts:**
- state file `*.rigel` (format `rigel-cad-state/1`);
- `results/rigel_cad/RESULTS_<dataset>_<order><seed>_<backbone>_B<budget>[_fastonly][_fifo].json`;
- `results/rigel_cad/paper/`.

---

## 4. Design decisions a reviewer should check

1. **Bitwise fidelity over refactoring freedom.**
   - Every numerical primitive keeps the research code's operation order, dtype and chunking: 512 query rows, 2048 residual rows, batch 8 for DINOv2 and 32 for ViT-B/8, and fp64 accumulation of S.
   - Consequences:
     - batch sizes are part of the contract (GPU kernels can differ by batch);
     - the histogram range for pixel AP is [0, max over *all stages* of that category's grid + 1e-12]. This is inherited and documented.
2. **Row order is part of the method.** Greedy k-centre starts from a seeded random token index, so training images are always sorted by relative path. Test order is the canonical one: defect folders sorted, then files sorted; VisA uses the csv sorted by image path.
3. **Slow level only reads its own task's keys.** The residual is taken against C_t (the task's own coreset) before consolidation, not against the consolidated memory. This is the frozen definition.
4. **Byte accounting is one function** (`RigelCAD.persistent_bytes`). It is asserted on every `state()` call.
5. **`keep_history`** stores per-stage read-out states, so forgetting can be measured. It is not part of the persistent state. The unbounded mode stores prefix lengths only (fix in `8b5e04f`).
6. **Two native-resize paths:**
   - torch bilinear for the benchmark pixel AP (as the research metric);
   - OpenCV INTER_LINEAR for the AD 2 maps (as the submitted files).
   - They differ slightly, by design, to match each locked protocol.
7. **AU-PRO** uses an exact histogram over float16 bit patterns, not the official `compute_pro`, because of memory (> 14 GB per AD 2 scenario). It was validated earlier against the official code to |Δ| ≤ 4.3e-14, and here bitwise against the locked values. `trapezoid` re-implements the official semantics, with the MVTec copyright credited.

---

## 5. Tests and results (exact commands; `nested_learning` env, RTX 4050, torch 2.10.0+cu126)

```
python -m pytest -q rigel_cad/tests/test_unit.py         -> 9 passed (5.9 s)
python -m pytest -q rigel_cad/tests/test_equivalence.py  -> 6 passed (run in parts: 3+1 in 218 s / 25 s, 2 in 85 s;
                                                             grids re-run after 8b5e04f: 1 passed, 127 s)
```

**Unit tests** (`test_unit.py`):
- capacities and byte bounds for 4 / 8 / 16 / 32 MiB and fast-only;
- k-centre determinism, the prefix property and seed sensitivity;
- W·C·W = I;
- k-centre / FIFO / unbounded memory rules;
- unbounded ≡ non-binding budget (keys and energies);
- BudgetViolation;
- fast-only state;
- save/load round trip, bounded and unbounded;
- orders;
- binned AP / AU-PRO / SegF1 / trapezoid on hand-computed cases.

**Equivalence tests** (`test_equivalence.py`):

| Test | Reference | Result |
|---|---|---|
| orders a / b / c, 3 datasets | `final.bench_order` | equal |
| grids: 8 MiB, unbounded, fast-only, FIFO, every stage, MPDD a\|2711 | locked G3E-PAPER-01 T1 / A4 grids | **bitwise** |
| per-stage AP / AUROC plus 2,000 bootstrap replicates, U and E | locked MULTIRATE-01 METRICS + boot npz | **bitwise** |
| DINOv2 tokens (16 images) | research encoder run now | **bitwise**. Against the older cache: max 6.1e-5 (1 fp16 ulp) on 0.0045 % of entries |
| AD 2 θ (8 scenarios) and test_public AU-PRO 0.05 / 0.30, SegF1, tp/fp/fn (`can`) | locked AD2_SCORE / AD2_AUPRO | **bitwise** |
| ViT-B/8 tokens (32 images) | locked ViT-B/8 cache | **bitwise** |

**End-to-end CLI run.** Command:
```
python -m rigel_cad stream --dataset mpdd --data-root data/mpdd --order a --seed 2711 --budget-mib 8 --cache-dir /mnt/k/NLCAD/cache/rigel_cad --out results/rigel_cad
```
- Encoding plus the run took about 2.5 min.

| Metric | CLI run | Locked |
|---|---|---|
| Pixel AP | 0.50822631 | 0.50822630 |
| Image AUROC | 0.93990785 | 0.93990785 (identical) |
| FM | 0.0128069 | 0.0128069 |
| CI95 | [0.48946, 0.52956] | [0.48946, 0.52956] |

- The 1e-8 differences come from re-encoding (the 1-ulp token differences).
- Output: `results/rigel_cad/RESULTS_mpdd_a2711_dinov2_vitb14_reg4_B8.json` (pushed).

---

## 6. Known limitations and open review questions

**Untested or partly tested:**
1. **`run_ad2` end-to-end from raw images.** RESOLVED 2026-11-26: it ran end-to-end and reproduces the locked values bitwise (§9). One reviewer point remains: in `run_ad2` the scenario list comes from `data.ad2_scenarios(root)`, i.e. directories with `train/`. The research code used the manifest's scenario list. They are equal for the official layout (8 scenarios), but this is not asserted.
2. **Bitwise reproducibility is per platform.** It holds for the same GPU and software stack. Across environments, expect tiny differences, as seen with the encoder cache. CI cannot run `test_equivalence.py`, because it needs CUDA and the private caches on K:.

**Environment and packaging:**
3. **Dependencies:**
   - `tifffile` is needed only by `ad2.write_maps`, and is missing in the `nested_learning` env (`pytorch_env` has it);
   - `timm` is needed only for ViT-B/8;
   - no `pyproject.toml` or requirements file for the package yet.
4. **Checkpoint paths** are resolved relative to the repo root (`encoders.CHECKPOINTS`). A `checkpoint=` argument overrides them, but the CLI has no flag for it yet.

**Resource use:**
5. **Memory.**
   - `keep_history` stores W per stage (2.4 MB each) and, when bounded, keys per stage (≤ 6 MB at 8 MiB). This is fine for 15 tasks.
   - The token cache for MVTec is about 6.4 GB on disk if `--cache-dir` is used.
6. **`state()` cost.** It recomputes the fp64 `eigh` each call. `fit_task` calls it once per task, and `patch_energy` without a state calls it again. Caching W per task is a possible optimisation; it would not change numbers.

**Reviewer questions:**
7. **`FastKeyMemory.write` with FIFO and capacity None** behaves like unbounded, which is fine. Note that FIFO keeps `C[:cap]`, the k-centre prefix, when one task alone exceeds the cap, exactly as in the research code.
8. **Naming and anonymity (owner decision; recorded, not re-argued).**
   - The repository is public, and the AD 2 leaderboard entry is "RIGEL-CAD".
   - For a double-blind submission, the release code would normally go to an anonymised mirror.
9. **API surface to review:**
   - `RigelState.keys` may be `None` only inside an unbounded history; `state_at` re-assembles the keys.
   - `RigelCAD.load` now uses `weights_only=True`.
10. **On-disk state size vs the budget.** `states/ad2.rigel` is 10,749,509 B, above 8 MiB (8,388,608 B), although `persistent_bytes` = 8,388,104 B.
    - The cause: `save` writes S in its fp64 accumulator precision (4.7 MB), plus pickle overhead.
    - The budget charges S as fp32 (2,359,296 B + 8 B count), exactly as the research protocol did.
    - Saving S as fp32 would make the file fit the budget. W rebuilt from an fp32 S could differ slightly from the in-memory run, so this was not changed silently. Reviewer and owner decision: either save fp32 S, or state "S is accumulated in fp64; the stored statistic is counted at fp32".

---

## 7. Review checklist (suggested)

- [ ] `ops.py` vs the research primitives (provenance table in the README): operation order, dtypes, chunk sizes.
- [ ] `model.py`:
  - consolidation (`sel.sort()`, insertion order);
  - S accumulation order;
  - budget assertion;
  - save/load completeness: keys, blocks, S, n, config, task names.
- [ ] `encoders.py`:
  - DINOv2 forward (pos-embed interpolation offset 0.1, bicubic without antialias);
  - block indices 6–9 (0-based);
  - prefix of 5 (CLS + 4 registers);
  - ViT-B/8 hook on block index 8 (0-based).
- [ ] `readout.py` / `metrics.py`:
  - Gaussian kernel and padding;
  - top-1 % k;
  - histogram range rule;
  - bootstrap seeding [20261120, position];
  - AUROC sample weights;
  - stream summaries (FM excludes the last task).
- [ ] `ad2.py`: θ = mean + 3σ over native float32 maps (fp64 sums); TIFF float16 zlib; PNG threshold on float32.
- [ ] `pipeline.py`:
  - grids score each test category at stages ≥ its acquisition;
  - bootstrap only for order a by default;
  - outputs and tags.
- [ ] Tests: re-run `test_unit.py` on CPU, and `test_equivalence.py` on the same machine.

---

## 8. Changes after the first push (`8b5e04f`)

1. `RigelCAD.load`: `torch.load(..., weights_only=True)`. Before, it was `False`, which allowed arbitrary unpickling of a state file.
2. `keep_history` with an unbounded memory stores `RigelState(keys=None, …)` and re-assembles the prefix in `state_at`. This avoids a full key copy per stage (VisA unbounded: about 106 MB × 12).
3. Re-tested: unit tests 9/9; grid equivalence bitwise (1 passed, 127 s).

---

## 9. MVTec AD 2 end-to-end run from raw images (2026-11-26)

**Command** (`results/rigel_cad/ad2/run_ad2.sh`; `tifffile` taken from `pytorch_env`, see §6 item 3):
```
python -m rigel_cad ad2 --data-root /mnt/k/mvtec_ad_2 --out results/rigel_cad/ad2 \
       --cache-dir /mnt/k/NLCAD/cache/rigel_cad --budget-mib 8 --seed 2711 --per-stage
```
- 08:10:05 → 09:20:23, rc = 0. Wall time 4,216 s, which includes encoding 40 splits from images and per-stage AU-PRO on CPU.
- Outputs:
  - `results/rigel_cad/ad2/RESULTS_ad2.json` and `run_ad2.log` (pushed);
  - `submission/` (9.3 GB): byte-identical to the locked `AD2_SUBMISSION/E` (full `diff -rq`), so it was deleted on 2026-11-26 as a duplicate. Regenerate it with the same command.
  - `states/ad2.rigel`: kept locally, not pushed.
- Persistent bytes 8,388,104.

**Comparison with the locked arm E** (`results/hope_cad/nl03_multirate_01/ad2/AD2_SCORE.json`, `AD2_AUPRO.json`, `AD2_SUBMISSION/E`):

| Quantity | End-to-end | Locked | Equal |
|---|---|---|---|
| θ, 8 scenarios | 569.0595 … 554.6602 | `theta_dthr.E` | 8/8 bitwise |
| test_public per stage: AU-PRO 0.05, AU-PRO 0.30, SegF1 (where locked) | 80 values | `per_stage.*.E` | 80/80 bitwise |
| mean AU-PRO 0.05 / 0.30 (final) | 0.26331970 / 0.47497739 | same | bitwise |
| mean SegF1 (final) | 0.14934623 | same | bitwise |
| forgetting AU-PRO 0.05 (mean over the 7 scenarios with a later stage) | 0.00772843 | 0.00772843 | bitwise |
| submission files | 8,180 files, same file list | 8,180 + 4 marker files | all 8,180 byte-identical (`diff -rq`, no output) |

**Per scenario (final stage, test_public):**

| Scenario | AU-PRO 0.05 | AU-PRO 0.30 | SegF1 |
|---|---|---|---|
| can | 0.0643 | 0.2224 | 0.0008 |
| fabric | 0.1602 | 0.3782 | 0.0745 |
| fruit_jelly | 0.3560 | 0.5904 | 0.1687 |
| rice | 0.3682 | 0.5479 | 0.0563 |
| sheet_metal | 0.1416 | 0.3332 | 0.1133 |
| vial | 0.5323 | 0.8006 | 0.3194 |
| wallplugs | 0.1561 | 0.3818 | 0.0767 |
| walnuts | 0.3280 | 0.5453 | 0.3852 |

**Notes:**
- These are local test_public numbers. The server numbers for test_private / test_private_mixed (38.2 / 34.0 AU-PRO 0.05) came from the locked submission, and this run regenerates the same files.
- Here the fresh encoding from images was bitwise equal to the locked AD 2 path, unlike the 1-ulp encoder difference on the older MPDD cache (§5).
- Nothing was uploaded.

---

## Handoff

**Task:** RIGEL-CAD-01, the official RIGEL-CAD source package (owner request 2026-11-26: keep `exps/`, rename official components/API/artifacts, complete code), plus a push of the results.

**Status:** REVIEW_READY.

**Last agent:** Codex (role held by Claude Code Opus).

**Next agent:** OpenCode (independent code review), or the reviewer chosen by the owner.

**Branch / HEAD:** dikhang_space. Commits on top of the owner's dc1e919: 3cd24b2 (package + results), 8b5e04f (fixes), c561dc9 (docs handoff), and 12514e2 (AD 2 end-to-end, §9), all pushed to origin.

**Completed work:**
- the `rigel_cad/` package (§3);
- unit and equivalence tests;
- the end-to-end MPDD CLI check;
- the end-to-end MVTec AD 2 run from raw images, bitwise equal to the locked arm E (§9);
- paper tables under the RIGEL-CAD names;
- push of the package, `exps/g3e_paper_01/tables.py` (D-TAB) and 116 result JSONs;
- this report.

**Changed files:**
- New: `rigel_cad/**` (16 files); `results/rigel_cad/paper/RIGEL_CAD_{RESULTS.json,TABLES.md}`; `results/rigel_cad/RESULTS_mpdd_a2711_dinov2_vitb14_reg4_B8.json`.
- Force-added, because `results/` is git-ignored: `results/hope_cad/{g3e_paper_01,nl03_multirate_01}/**/*.json(l)` and the AD 2 `UPLOAD_MANIFEST.json` / `README_UPLOAD.md`.
- Modified: `exps/g3e_paper_01/tables.py` (D-TAB).
- `exps/` is otherwise unchanged. `.gitignore` is unchanged.
- New local cache: `/mnt/k/NLCAD/cache/rigel_cad/` (MPDD tokens).
- Not pushed: `results/rigel_cad/states/*.rigel` and `BOOT_*.npz`.

**Verification / Tests:**
- `python -m pytest -q rigel_cad/tests/test_unit.py`: 9 passed.
- `python -m pytest -q rigel_cad/tests/test_equivalence.py`: 6 passed. The first encoder test failed against the older cache (1-ulp difference). It was diagnosed as an environment difference: the research encoder run now equals the RIGEL-CAD encoder bitwise. The test was then rewritten to check bitwise equality against the research code plus an ulp bound against the cache.
- CLI on MPDD: matches the locked numbers (§5).

**Unresolved issues:** §6 items 1–9. The main ones are:
- the on-disk `.rigel` size versus the fp32 accounting of S (§6 item 10);
- no packaging metadata;
- `tifffile` is missing in one env;
- the naming/anonymity point is the owner's decision.

**Exact next action:** the reviewer re-reads this report, the diff `git diff dc1e919..8b5e04f -- rigel_cad exps/g3e_paper_01/tables.py`, and the README; goes through the checklist in §7; and reports findings in a new report (for example `agents/reports/rigel_cad_01_review.md`). Do not change `exps/`.
