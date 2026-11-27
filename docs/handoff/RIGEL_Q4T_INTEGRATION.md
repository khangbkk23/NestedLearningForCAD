# RIGEL-Q4T-INT: official Q4T integration and release (RIGEL-CAD 1.1)

| Field | Value |
|---|---|
| Task | Owner task "RIGEL-CAD — Official Q4T Integration and Release" (2026-11-27): integrate the benchmark-supported Q4T storage into `rigel_cad/` as the optional backend `storage="int4_triangular"`, with fp16 kept as the default |
| Status | DONE |
| Last agent | Codex (role held by Claude Code, Opus) |
| Next agent | User (review the local commit; push only on owner approval) |
| Branch / HEAD | `dikhang_space`. Base `bc34f76e73ce9c0ac6218054ced89ccf785ab6e7`; the local integration commit is on top (see §10). **Not pushed** |
| Verdict | **`INTEGRATION_PASS`** |

---

## 0. Summary

- **New option.** `RigelCAD(RigelConfig(budget_mib=8, storage="int4_triangular"))` and
  `python -m rigel_cad stream … --storage int4_triangular` now run the Q4T algorithm inside the official package. The
  code is a verbatim port of `exps/rigel_q8/d1.py::ArmQ4`, `quantize4`, `pack4` and `unpack4`, and of
  `exps/rigel_q8/q8.py::dequantize`, `tri_pack` and `tri_unpack`. The package does not import `exps/`.
- **Default unchanged.** fp16 stays the default. Its computation, outputs, file names and checkpoint layout
  (`rigel-cad-state/1`) are unchanged.
  - fp16 grids are bitwise equal to the locked v1 grids at every stage on MVTec AD, VisA and MPDD (order a, seed 2711).
  - fp16 metrics are equal to the locked v1 METRICS at every stage.
  - A real RIGEL-CAD 1.0 checkpoint still loads.
- **Bitwise equal to the research Q4T at every stage** (MVTec AD 15 stages, VisA 12, MPDD 6). This holds for:
  - the per-task fit: coreset keys, S_t and n_t, against the cached v1 fits;
  - the int4 codes, fp16 scales and dequantised keys;
  - the fp32 triangle, the count and the whitener W;
  - every grid, against the cached RIGEL-CC-01 Q4T grids;
  - every metric cell, against the locked RIGEL-CC-01 METRICS;
  - the final checkpoint tensors (`q4`, `scale`, `S_tri`, `n`), against the research persisted state.
- **Save → load → continue** is bitwise equal to an uninterrupted run:
  - MVTec AD: restored after task 7, with consolidation active in the 8 tasks after restoration;
  - MPDD: restored after task 3.
- **Reproduced Q4T metrics** (order a, seed 2711), equal to the locked references at full precision:

  | Dataset | Pixel AP | Image AUROC |
  |---|---|---|
  | MVTec AD | 0.6304 | 0.9926 |
  | VisA | 0.4749 | 0.9707 |
  | MPDD | 0.5170 | 0.9598 |

- **CLI smoke test from raw MPDD images:** pixel AP 0.51704891 (cached tokens: 0.51704892), image AUROC identical.
- **Tests: 34 passed, 0 skipped, 0 failed.** That is 20 unit tests, 7 equivalence tests (including the new int4
  research-equivalence test) and 7 research `test_q8` tests.
- **Bug found and fixed.** The smoke test exposed a bug in `run_stream`'s log line: `len(model.history[-1].keys)` fails
  when the history holds no key copies. It is fixed; the change affects logging only.
- **Not done.** MVTec AD 2 was not run or submitted, no experiment was launched, no hyperparameter was changed, and no
  frozen research file was edited.

---

## 1. Source fidelity (inspected before implementation)

**Starting state.**
- HEAD `bc34f76`.
- Working tree before the task: untracked `exps/rigel_q8/`, `exps/rigel_cc01/` and 5 `docs/handoff/*.md`. No tracked
  modification.
- All of these were preserved; none was edited.

**Research sources and their sha256.**

| Source | sha256 | Matches the RIGEL-CC-01 ledger |
|---|---|---|
| `exps/rigel_q8/q8.py` | `3d862fa284ba50675f6eb6c2f78ae4fc91981b7c1e4abcfe535e6d7fce68e5c6` | yes |
| `exps/rigel_q8/d1.py` | `b044b4357f97d4eec23d7f5382ce8f0cc3cf8f5bbd46ff7da664a410a4d41102` | yes |
| `exps/rigel_cc01/cc.py` | `314d5faefaf9e20c0c1c4705ebd8f318cdccb0b254e3cb8d7e7fe5adb44b4c3f` | yes |
| `exps/rigel_gg01/gg.py` | `4d91a437d16c71a8d6cd2d33a05d2a8481ebbdb633dc8d7e3405a01f2714504d` | yes |

**Semantics verified in the research code, and how they were ported.**

| Aspect | Research Q4T (`d1.ArmQ4`) | Package (`int4_triangular`) |
|---|---|---|
| Per-task fit | `gg.fit_task`: identical to `RigelCAD.fit_task` (1 % greedy k-centre coreset, residual moment) | unchanged `RigelCAD.fit_task` |
| Quantiser | scale = fp16(max\|x\| / 7) clamped ≥ 1e-12; q = clamp(round(x / scale), −7, 7) as int8 | `storage.quantize4` (same operations) |
| Write order | quantise the new coreset → concatenate with the stored codes and scales → dequantise the candidates → if above the cap, `ops.greedy_kcenter(dequantised, cap, seed)` → keep sorted indices (insertion order) | `FastKeyMemory._write_int4` (same order) |
| Stored keys | int8 codes in [−7, 7] (unpacked in memory) + fp16 scales; scored as q.float() · scale.float() | same; packed (2 per byte, offset 8, even column in the low nibble) for checkpoints and history |
| Slow statistic | `S_tri = tri_pack(tri_unpack(S_tri) + S_t)`: fp64 sum re-rounded to the fp32 upper triangle each task | `SlowResidualWhitener(triangular=True).write` (same; computed on CPU, the research ran on CUDA; bitwise equal, see §3) |
| Whitener | `ops.whitener(tri_unpack(S_tri), n)` in fp64, cast to fp32 | `SlowResidualWhitener.W()` |
| Scoring | `ops.energy_whitened(q16, keys_fp32, W)` | `RigelCAD.patch_energy` (unchanged path) |
| Cap / budget | (8,388,608 − 1,181,192) // 386 = 18,672; payload asserted ≤ budget | `RigelConfig.key_capacity`; `state()` raises `BudgetViolation` if exceeded |

`BLOCKED_SOURCE` did not apply: every source file and reference artifact was present.

**Reference artifacts used.**
- RIGEL-CC-01 grids: `/mnt/k/NLCAD/cache/rigel_cc01/grids/{ds}_a2711/Q4T/`.
- RIGEL-CC-01 persisted states: `…/states/{ds}_a2711/Q4T.pt`.
- Cached v1 fits: `/mnt/k/NLCAD/cache/rigel_gg01/fit` and `/mnt/k/NLCAD/cache/rigel_cc01/fit`.
- Locked metrics: `results/hope_cad/rigel_cc01/METRICS_{ds}_a2711.json` (Q4T) and
  `results/hope_cad/nl03_multirate_01/bench/METRICS_{ds}_a2711.json` (v1 arm E).
- Locked v1 grids: `/mnt/k/NLCAD/cache/nl03_multirate_01/bench/{ds}_a2711/E/`.

All of these were read only.

---

## 2. Source files changed

| File | Change | sha256 (16) |
|---|---|---|
| `rigel_cad/storage.py` (new) | int4 quantiser, dequantiser, pack/unpack, triangular pack/unpack (verbatim port) | `c7369a3c1952283f` |
| `rigel_cad/config.py` | `RigelConfig.storage` (`'fp16'` default / `'int4_triangular'`); `KEY4_BYTES` 386, `SLOW_TRI_BYTES` 1,181,192, `STORAGES`; `key_bytes` and `slow_bytes` per format; capacity per format; validation (int4 requires a finite budget, `slow_level`, `kcenter`); `to_dict` omits `storage` for fp16, so the v1 config layout is unchanged | `83f84933a6356f46` |
| `rigel_cad/model.py` | `FastKeyMemory(storage=…)` with `_write_int4`, `packed`, `load_packed`; `SlowResidualWhitener(triangular=…)` with `matrix()`; `RigelState` gains `storage`, `q4`, `scale`; compact int4 history; `persistent_bytes` per format; checkpoint `rigel-cad-state/2` for int4; `load` accepts both formats and refuses a format/storage mismatch. The fp16 code paths are unchanged | `69c6b9fd94a6ab65` |
| `rigel_cad/pipeline.py` | `storage_suffix()`: `_int4tri` suffix on stream tags and on AD 2 `RESULTS`/`states`/`submission` names; `storage` field in RESULTS; log line uses `len(model.fast)` (bug fix, logging only) | `cdbc01be127ca310` |
| `rigel_cad/cli.py` | `--storage {fp16,int4_triangular}` (default fp16) for `stream` and `ad2` | `6f1a3dfff49c28c2` |
| `rigel_cad/__init__.py` | version 1.1.0 | `faaa4a75abceb628` |
| `rigel_cad/README.md` | rewritten as a standalone release README (§8) | `b7b50b7b0bd1909b` |
| `rigel_cad/tests/test_unit.py` | +9 tests (§6) | `a9c3ffde393d0f6e` |
| `rigel_cad/tests/test_equivalence.py` | + `test_int4_triangular_bitwise_research` | `ebf25a42117e0905` |
| `exps/rigel_q4t_int/verify.py` (new directory) | integration verification driver (§3) | `ed5b236df8042f21` |

**Unchanged against HEAD (`git diff --quiet`):**
- `rigel_cad/ops.py` (`020875417fc19cb7`), `encoders.py`, `metrics.py`, `readout.py`, `data.py`, `ad2.py`;
- `bench.py` and `paper_tables.py`;
- every file under `exps/` that existed before.

---

## 3. Numerical equivalence evidence

**Driver:** `results/rigel_cad/q4t_integration/run_verify.sh`, which runs
`python -B -m exps.rigel_q4t_int.verify stream {mpdd,mvtec,visa}` and then `… summary`. It ran 06:54–07:49 on
2026-11-27, on one GPU (RTX 4050 Laptop), torch as in `VERIFY_*.json`.

**Inputs:** the canonical K: DINOv2 token caches, through `exps.nl03_pressure_00.core.bind()`. Train tokens come from
`fit_tokens(…, train_rows_sorted)`; test tokens from `G.load_test`.

**Per task, for both backends,** the package ran its own `fit_task`. The research `cc.Stream(['Q4T'])` was fed the
cached v1 fits.

| Check (all stages) | MVTec AD (15) | VisA (12) | MPDD (6) |
|---|---|---|---|
| Package fit (coreset keys, S_t, n_t) == cached v1 fit, both backends | ✔ | ✔ | ✔ |
| int4 codes, fp16 scales, fp32 dequantised keys == research | ✔ | ✔ | ✔ |
| fp32 triangle, count, whitener W == research | ✔ | ✔ | ✔ |
| int4 grids == RIGEL-CC-01 Q4T grids (bitwise, every stage) | 15/15 categories | 12/12 | 6/6 |
| fp16 grids == locked v1 arm E grids (bitwise, every stage) | 15/15 | 12/12 | 6/6 |
| int4 metric cells == locked RIGEL-CC-01 Q4T METRICS (every stage) | ✔ | ✔ | ✔ |
| fp16 metric cells == locked v1 METRICS arm E (every stage) | ✔ | ✔ | ✔ |
| Final checkpoint `q4`, `scale`, `S_tri`, `n` == research persisted Q4T state | ✔ | ✔ | ✔ |
| `load()` → patch energies equal to the live model | ✔ | ✔ | ✔ |
| Save → load → continue == uninterrupted (codes, scales, triangle, count at every later task) | ✔ (after task 7; 8 tasks compared) | — | ✔ (after task 3; 3 tasks compared) |
| Reproduced pixel AP / image AUROC == owner references (4 d.p.) | ✔ | ✔ | ✔ |

**Key-state equivalence is checked at every stage, not only on final metrics.**
- Consolidation binds on MVTec AD (from task index 9 of 15; the run restored after task 7 then consolidated 6 times) and VisA (from task index 4 of 12), so the k-centre selection on dequantised int4 candidates is
  exercised.
- On MPDD the cap is never reached: 6,965 keys.

**Stream summaries of the integrated backends** (`VERIFY_SUMMARY.json`):

| Dataset | Storage | Pixel AP | Image AUROC | Forgetting | Acquisition | Prequential |
|---|---|---|---|---|---|---|
| MVTec AD | int4_triangular | 0.630364 | 0.992609 | 0.007416 | 0.626855 | 0.655861 |
| | fp16 | 0.592652 | 0.982616 | 0.033255 | 0.612928 | 0.639517 |
| VisA | int4_triangular | 0.474854 | 0.970658 | 0.005270 | 0.467979 | 0.497389 |
| | fp16 | 0.460251 | 0.934911 | 0.011920 | 0.463783 | 0.493411 |
| MPDD | int4_triangular | 0.517049 | 0.959816 | 0.001225 | 0.517483 | 0.347492 |
| | fp16 | 0.508226 | 0.939908 | 0.012807 | 0.516260 | 0.343312 |

Every value is identical to the locked RIGEL-CC-01 (int4) and NL03-MULTIRATE-01 / G3E (fp16) values. The bootstrap CIs
follow from bitwise-identical grids and were not recomputed in the verification. The CLI run below recomputed them for
MPDD: pixel AP CI95 [0.4990, 0.5386].

**End-to-end CLI smoke test.** The command is the owner's:
`python -m rigel_cad stream --dataset mpdd --data-root data/mpdd --storage int4_triangular --budget-mib 8`, with script
`results/rigel_cad/q4t_integration/run_cli_mpdd.sh` and log `run_cli_mpdd.log`. It runs from raw images with no token
cache, in 188 s.

| | CLI from images | Canonical cached tokens |
|---|---|---|
| Pixel AP | 0.517048909 | 0.517048918 (Δ −8.7e-9) |
| Image AUROC | 0.959816246 | 0.959816246 (identical) |
| Forgetting / acquisition / prequential | Δ ≤ 5e-8 | |
| Registered bytes (max) | 3,869,682 | 3,869,682 |

The tiny difference comes from the encoder: it reproduces the old token caches to within one fp16 ulp on 0.0045 % of
entries. The same thing happens in the v1 end-to-end check (0.50822631 vs 0.50822630).

**First smoke attempt.** It failed with `TypeError: object of type 'NoneType' has no len()` in
`pipeline.run_stream`'s log line (`len(model.history[-1].keys)`; the log is kept as
`run_cli_mpdd_attempt1_logbug.log`).
- The compact history, like the unbounded fp16 history, stores no key copies. So the same line also failed for
  `--budget-mib none` in 1.0, which was a pre-existing bug.
- Fixed to `len(model.fast)`; logging only.
- `test_stream_pipeline_int4_unbounded_and_v1_file_names` now covers `run_stream` for int4, fp16 and unbounded.
- The numerical verification above does not use `pipeline.py`. The full test suite was re-run after the fix (§6).

---

## 4. Legacy v1 compatibility

**Default and computation.**
- The default is `storage='fp16'`.
- Old configs and checkpoints have no `storage` key, and `from_dict` falls back to fp16.
- The fp16 code paths in `FastKeyMemory`, `SlowResidualWhitener`, `fit_task`, `state`, `state_at`, `patch_energy`,
  `save` and `load` compute exactly what 1.0 computed:
  - grids are bitwise equal to the locked v1 on all 3 datasets and every stage (§3);
  - `test_stream_grids_bitwise` passes for 8 MiB, unbounded, fast-only and FIFO.

**Checkpoint layout.**
- fp16 checkpoints keep `format='rigel-cad-state/1'` with the same key set and no `storage` key (unit test), so
  RIGEL-CAD 1.0 can still read them.
- The real 1.0 release state `results/rigel_cad/states/mpdd_a2711_dinov2_vitb14_reg4_B8.rigel` loads under 1.1
  (`test_loads_rigel_cad_1_0_checkpoint`).
- That file's sha256 `c85afd97…670b` is unchanged.

**Result files.**
- fp16 file names and tags are unchanged.
- RESULTS JSON gains one field, `"storage": "fp16"`; no value changes.
- v1 result files in `results/rigel_cad/` (`RESULTS_mpdd_…_B8.json`, `BOOT_…npz`, `states/…B8.rigel`) are unchanged
  against HEAD.

**Overwrite protection for int4 runs.**
- Stream: `…_int4tri` tag.
- AD 2: `RESULTS_ad2_int4tri.json`, `states/ad2_int4tri.rigel`, `submission_int4tri/`.
- Both are tested.
- The CLI smoke test wrote `RESULTS_mpdd_a2711_dinov2_vitb14_reg4_B8_int4tri.json`, `BOOT_…_int4tri.npz` and
  `states/…_int4tri.rigel` next to the v1 files.

---

## 5. INT4 storage and checkpoint correctness

**Unit tests (CPU).**
- 386 B per key; 384 packed bytes.
- Codes in [−7, 7]; the absmax element maps to ±7.
- Idempotent: quantising the dequantised keys returns the same codes and scales.
- pack/unpack is a bitwise round trip, and an all-zero row is handled.
- The error is bounded by scale / 2.
- Triangle: `tri_unpack(tri_pack(S)) == fp32(S)` exactly, and W from the triangle is bitwise equal to W from the full
  fp32 matrix.
- Consolidation keeps exactly the capacity (199 keys at 1.2 MiB). History states reproduce the live state, and the
  stored `keys` equal the dequantised packed codes.

**Checkpoint `rigel-cad-state/2`** contains:
- `config` (with `storage`), `n_tasks`, `task_names`;
- `q4` (uint8 M × 384), `scale` (fp16 M), `S_tri` (fp32 295,296), `n` (int);
- `persistent_bytes`.

Its tensor payload equals the registered bytes exactly (unit test and §7). A file whose format and storage disagree is
rejected.

**Resume.** Unit test (8 synthetic tasks, consolidation binding) and real data (MVTec AD after 7, MPDD after 3): after
restoration and continued learning, codes, scales, triangle, count and energies are bitwise equal to the uninterrupted
run.

**Continual operation.** Q4T runs throughout the stream: codes are written and consolidated at every task boundary, not
only quantised from a final fp16 state.

---

## 6. Tests and commands

**Final run, after all changes** (`results/rigel_cad/q4t_integration/pytest_final.out`):

```
python -m pytest -p no:cacheprovider rigel_cad/tests/test_unit.py rigel_cad/tests/test_equivalence.py exps/rigel_q8/test_q8.py -v
34 passed in 352.50s   (0 skipped, 0 failed)
```

**New tests:**
- `test_int4_capacity_bytes_and_config`
- `test_int4_quantiser_pack_roundtrip`
- `test_triangular_statistic_exact`
- `test_int4_memory_consolidates_dequantised_codes`
- `test_int4_save_load_resume`
- `test_v1_checkpoint_format_unchanged`
- `test_loads_rigel_cad_1_0_checkpoint`
- `test_ad2_pipeline_accepts_int4_storage_and_keeps_v1_files`
- `test_stream_pipeline_int4_unbounded_and_v1_file_names`
- `test_int4_triangular_bitwise_research` (equivalence)

The AD 2 and stream pipeline tests use synthetic folder layouts and random tokens. They check the API and file naming
only and make no evaluation claim.

**Earlier run.** Before the log fix and the stream-pipeline test, the suite gave 26 passed
(`pytest_before_logfix.out`, unit + equivalence).

**Other commands:**

```bash
results/rigel_cad/q4t_integration/run_verify.sh       # GPU integration verification (55 min; MPDD, MVTec AD, VisA)
results/rigel_cad/q4t_integration/run_cli_mpdd.sh     # end-to-end CLI smoke test from raw images (3 min)
```

The Python interpreter is `~/miniconda3/envs/nested_learning/bin/python`, the environment of the earlier RIGEL-CAD runs.
Only one GPU job ran at a time.

---

## 7. Memory and runtime (measured; order a, seed 2711)

All figures come from `results/rigel_cad/q4t_integration/VERIFY_{ds}_a2711.json`.

| Quantity | fp16 | int4_triangular |
|---|---|---|
| **Registered logical bytes**, final (MVTec AD / VisA / MPDD) | 8,388,104 / 8,388,104 / 8,388,104 | 8,388,584 / 8,388,584 / 3,869,682 |
| Logical bytes, maximum over tasks | = final | = final; ≤ 8,388,608 asserted at every task |
| **Persistent tensor payload** | 10,747,400 (S fp64, limitation L1) | **8,388,584 / 8,388,584 / 3,869,682** (= logical) |
| **Serialized checkpoint** (`torch.save`, config and task names included) | 10,750,061 / 10,749,989 / 10,749,925 | **8,391,437 / 8,391,364** / 3,872,324 |
| **Inference GPU resident memory**, live model | 6.0 MB | 71.7 MB (unpacked int8 codes + fp32 view; 18,672 keys) |
| Scoring working set (fp32 keys, keys·W, W) | 26.5 MB | 117.1 MB |
| **Peak GPU working memory** above base, scoring the final category | 53.1 / 52.2 / 52.2 MB | 233.7 / 233.4 / 89.5 MB |
| **Update time**, whole stream (`fit_task`, CUDA-synchronised) | 112.9 / 669.4 / 21.2 s | 139.6 / 709.5 / 19.5 s |
| Update time per task, median | 7.23 / 79.15 / 2.49 s | 7.83 / 81.76 / 1.96 s |
| **Scoring latency** per image, p50 (head only) | 2.17 / 2.08 / 2.51 ms | 7.95 / 7.96 / 3.43 ms |

**Notes on the table.**
- **The serialized int4 file exceeds 8 MiB** by 2,829 B on MVTec AD and 2,756 B on VisA, because of the `torch.save`
  container, config and task names. The tensor payload is within the budget. The file is **not** claimed to be below
  8 MiB.
- **Update time.** Inside each task the int4 model was fitted first and the fp16 model second, on the same
  GPU-resident tokens, so small ordering effects are possible. The per-task fit dominates. int4 consolidation on up to
  about 21k dequantised fp32 candidates adds about 0.6 s per task on MVTec AD and about 2.6 s per task on VisA.
- **Inference cost.** int4 adds about 5.9 ms per image at full capacity, about 3.7× the head latency, and about 4.4× the
  scoring working memory. The frozen encoder costs about 50 ms per image of compute on this GPU (release benchmark),
  so end-to-end inference slows by about 10 %.
- **No runtime optimisation was attempted.** For example, keeping the codes packed in GPU memory or a fused int4 kernel
  would be possible, but none was attempted, so no numerical computation changed.

---

## 8. Documentation

`rigel_cad/README.md` was rewritten as a standalone research-software README. It covers:
- the architecture: Exemplar Memory (fast) and Residual Geometry Memory (slow), and the read-out;
- the storage formats, with the exact int4 and triangle definitions and the restrictions;
- install;
- the API and CLI, including `--storage`;
- outputs and naming;
- memory accounting, with L, P, F, live GPU memory, working set and peak reported separately;
- the evaluation protocol;
- verified results: the main cell, 9-cell robustness, the categories that get worse, and quantisation fidelity;
- MVTec AD 2, labelled as fp16 / v1 only;
- runtime trade-offs;
- reproducibility, provenance, the research-code mapping, checkpoint formats and compatibility;
- limitations and release notes.

**Framing.**
- Q4T is presented as an optional compact-memory storage format of RIGEL-CAD. The README states explicitly that the two
  memories, their update schedule, the consolidation rule and the read-out are the same, so no new update frequency is
  implied.
- There is no state-of-the-art or universal-superiority claim.
- The limitations state that `int4_triangular` was chosen among compact-storage variants on development data, that the
  benchmark test sets had been used earlier, that the result is benchmark evidence and not an independent held-out
  confirmation, and that there is no AD 2 evaluation.
- No exploratory decision history appears in the API sections.

**Preserved:** the existing experiment reports, result manifests, RIGEL-CC-01 selection provenance
(`exps/rigel_cc01/SPEC.md`, the ledger, `GATE.json`, `ROBUST.json`) and `docs/handoff/*` were not altered.

**MVTec AD 2.** Nothing was rerun or resubmitted, as decided by the owner. The v1 results are preserved and attributed
to fp16 / v1 only:
- official 38.2 % (test_private) and 34.0 % (test_private_mixed) AU-PRO 0.05;
- local test_public 0.3594 / 0.5921 / SegF1 0.1680.

The int4 backend is API-compatible with `run_ad2`: the synthetic-layout test and separate file names show this. No
AD 2 evaluation of int4 was launched.

---

## 9. SHA256 of artifacts

| Artifact | sha256 (16) |
|---|---|
| `results/rigel_cad/q4t_integration/VERIFY_mvtec_a2711.json` | `ea146739b756f1be` |
| `results/rigel_cad/q4t_integration/VERIFY_visa_a2711.json` | `741ea627e16866b3` |
| `results/rigel_cad/q4t_integration/VERIFY_mpdd_a2711.json` | `0172657efbe00435` |
| `results/rigel_cad/q4t_integration/VERIFY_SUMMARY.json` | `42cf5ffb2844294e` |
| `results/rigel_cad/RESULTS_mpdd_a2711_dinov2_vitb14_reg4_B8_int4tri.json` (CLI) | `a52dd806acb5b020` |
| `results/rigel_cad/states/mpdd_a2711_dinov2_vitb14_reg4_B8_int4tri.rigel` (CLI; not committed) | `f27a0204a37e9d01` |
| `results/rigel_cad/q4t_integration/states/mvtec_a2711_int4tri.rigel` (not committed) | `7ec9f2c8f6875b6d` |
| `results/rigel_cad/q4t_integration/states/visa_a2711_int4tri.rigel` (not committed) | `62496774b4ed7d2c` |
| `results/rigel_cad/q4t_integration/states/mpdd_a2711_int4tri.rigel` (not committed) | `38fe059d04f48b5e` |
| v1 state `results/rigel_cad/states/mpdd_a2711_dinov2_vitb14_reg4_B8.rigel` (unchanged) | `c85afd97fc28cf60` |

**Hash timing.** The VERIFY JSONs record the package hashes at verification time:

| File | sha256 (16) |
|---|---|
| `config.py` | `83f84933…` |
| `model.py` | `69c6b9fd…` |
| `storage.py` | `c7369a3c…` |
| `ops.py` | `02087541…` |

These equal the committed files. `pipeline.py` changed after verification (the log line only); it is not used by the
verification.

---

## 10. Commit (local only)

The commit holds one minimal local integration change set on `dikhang_space`. **It is not pushed.**

**Included:**
- the package: `rigel_cad/` (7 modified files + `storage.py`);
- the verification driver: `exps/rigel_q4t_int/`;
- the Q4T research sources that the equivalence test and the driver import: `exps/rigel_q8/`, `exps/rigel_cc01/`.
  They were untracked before; they are committed unchanged and their hashes match the ledger;
- the locked RIGEL-CC-01 reference results: `results/hope_cad/rigel_cc01/` JSONs, ledger and log;
- the verification outputs: `results/rigel_cad/q4t_integration/` JSONs, logs, scripts and test outputs;
- the CLI result JSON;
- `docs/handoff/RIGEL_Q4T_INTEGRATION.md`, a copy of this report.

`results/` is git-ignored, so result files were force-added, as in earlier RIGEL-CAD commits.

**Not committed** (left untracked and unchanged):
- binary states (`*.rigel`) and BOOT npz;
- `results/hope_cad/rigel_q8/`;
- the 5 earlier `docs/handoff` reports.

**Repository state.**
- No unrelated tracked modifications existed before or after.
- `agents/` is git-ignored; this report lives there and is copied to `docs/handoff/`.

---

## 11. Remaining limitations

1. **Evidence level.** Q4T's accuracy is supported by benchmark evidence on test sets that had been used before. The
   variant was chosen after earlier development results had been seen. It is not an independent held-out confirmation,
   and it has no AD 2 evaluation.
2. **Category regressions.** Over the 9 robustness cells, some categories get worse:
   - MVTec AD zipper: −0.073 in one cell;
   - MVTec AD tile: lower in 6 of 9 cells;
   - VisA macaroni1: down to −0.031;
   - MPDD connector: 7 of 9 cells;
   - MPDD metal_plate: 9 of 9 cells, small;
   - MPDD bracket_white: 4 of 9 cells.
3. **Inference cost.** About 3.7× the head latency, about 4.4× the scoring working memory and about 12× the live-model
   GPU memory (unpacked codes plus an fp32 view). This could be reduced, without numerical change, by packing in GPU
   memory; that is not done.
4. **Serialized file size.** The serialized int4 checkpoint is about 2.8 kB over 8 MiB, although the tensor payload is
   within the budget.
5. **Restricted configurations.** `int4_triangular` is restricted to a finite budget, the slow level and k-centre
   selection. Unbounded, fast-only and FIFO combinations were never verified and raise `ValueError`.
6. **fp16 checkpoint size (pre-existing, unchanged).** fp16 checkpoints persist S in fp64 (10.7 MB) while the budget
   charges fp32 (limitation L1).
7. **Hardware.** The equivalence was demonstrated on this machine (RTX 4050 Laptop, the torch build in `VERIFY_*.json`).
   On other hardware, bitwise equality of the k-centre and scoring paths is expected but was not tested.

**Exact next action.** The owner reviews the local commit (`git show --stat HEAD`) and this report. If approved, the
owner authorises a push of `dikhang_space`. The next agent may proceed only after that review; no further experiment
is pending.
