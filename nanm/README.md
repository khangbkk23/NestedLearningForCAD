# NANM-G3 — canonical system (frozen)

**Nested Associative Normality Memory, configuration G3.** It is a task-boundary continual anomaly detector built
from:
- a frozen DINOv2 patch encoder;
- a bounded exemplar memory;
- a residual normality geometry;
- task-free whitened nearest-exemplar retrieval.

Status: **frozen canonical implementation** (NANM-SYS-FINAL, 2026-11-20).
- Bitwise equal to the sealed BENCH-G3 oracle on the same GPU (M1, M2, P5, P6).
- Contract sha256 (default seed 2711): `8f7f669d77cb06edcabd565eda6bc9cdb403b63662af67150f5182945708af7e`.
- Any change to `nanm/` invalidates the parity evidence and requires re-running the Tier-A suite (§13).

## 1. Claim boundary

- G3 demonstrates the value of **residual normality geometry under bounded memory**. On MPDD (clean cohort), G3
  beats byte-matched Euclidean memory by +0.0164 native Pixel AP and feature whitening by +0.0109
  (`agents/reports/bench_g3.md`).
- It does **not** establish a causal advantage of nested update dynamics. A per-prefix conventional estimator with
  the same sufficient statistic reproduces G3 exactly (roadmap §3.3).
- "Nested" refers only to the computational dependency of the geometry on the bank written at the same task boundary.
- It is not a HOPE or CMS implementation. There are no per-image updates and no learned parameters. Both levels change
  once per task boundary.

## 2. Mathematical specification

Notation: d = 768, 28×28 = 784 patch tokens per image, α = 0.5.

| Step | Definition |
|---|---|
| Tokens | z = normalize(mean_{b∈6..9} block_b(prep(x))[5:]) ∈ ℝ^{784×d}; fp16 after L2 normalisation |
| Write (task t) | X_t = all FIT tokens of task t (image order × row-major). P ∈ ℝ^{d×128} ~ N(0, 1/128) and the first centre ~ U{0..|X_t|−1}, both from `Generator(seed)` re-seeded at every write. Greedy farthest-first selection on zP picks k_t = ⌈0.01·|X_t|⌉ indices. Keys M_t = X_t[idx] (fp16) are appended to the union bank. |
| Residual statistic | r = x − NN^E_{M_t}(x) for x ∈ X_t ∖ M_t (own-task bank, Euclidean, uncentred). S ← S + Σ r rᵀ (per-chunk fp32 products accumulated in fp64), n ← n + |X_t ∖ M_t| |
| Geometry | C = S/n, T = (1−α)C + α·(tr C/d)·I, W = T^{−1/2} (fp64 `eigh`, λ_min > 1e-12 required; stored as fp32 for reading) |
| Read (task-free) | e(z) = ½·min_{m ∈ ∪M} ‖zW − mW‖² (expanded form, clamped ≥ 0, fp32) |
| Readout | A = shift(G_{σ=4, 33 taps, zero padding} ∗ bilinear_{28→392}(e)); image score = mean of the top ⌊0.01·392²⌋ = 1,536 values of A; native map = bilinear(A → H×W, no antialias); resized-224 map = bilinear(A → 224²) |

Update frequency: once per task boundary for both the bank (one batched commit of k_t keys) and the geometry. There
is no slow-to-fast write influence, no eviction and no mid-task refresh. The final state does not depend on task order
(up to fp summation order); acquisition-stage states do.

## 3. Components and source mapping

| Component | Module | Sealed oracle |
|---|---|---|
| Frozen Patch Encoder | `encoder.py` (`FrozenPatchEncoder`; body verbatim from `exps/dinomaly/encoder.py`) | `E.load`, `E.preprocess`, `nar_b14_01.tokens`, `bench_g3._extract` |
| Boundary Exemplar Writer | `memory.py` | `nanm_r0/audit.py::unit, greedy_coreset`; `nanm_r1::n_of`; `bench_g3::fit_tokens` |
| Residual Normality Geometry Estimator | `geometry.py` | `nanm_r2b::residual_stats, whiten`; `nanm_r2::shrink, inv_sqrt_psd` |
| Whitened Exemplar Reader | `retrieval.py` | `nanm_r2::sqdist`; `bench_g3::whitened` |
| Fixed Anomaly Readout | `readout.py` | `nanm_r2::maps_of`; `nanm_r0::image_scores`; `nar_lite_01::map_native`; `bench_g3::maps224` |
| Continual State | `state.py` (state, ledger, checkpoint) | `bench_g3::byte_table` (plus corrected accounting) |
| NANMSystem | `system.py` | `bench_g3::_normal_ds, score` |
| Contract | `config.py` (`G3Config`, frozen) | BENCH-G3 `LOCK.json` `ff998675…` |

`nanm/` imports only the standard library, torch, numpy, PIL and safetensors. This is enforced by
`tests/nanm/test_m1.py` and was verified at runtime by the M2 call-graph guard.

## 4. Precision and device semantics (part of the contract)

| Item | Value |
|---|---|
| Encoder | fp32, batch 8 in registered image order. Run under deterministic algorithms with TF32 off (`R.configure()` equivalent: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, TF32 off, 2 threads). |
| FIT normalisation | on the bank device (CUDA), in chunks of 64 images, then fp16 |
| Query normalisation | on CPU, then fp16 |
| k-centre, residuals, reading | on the bank device (CUDA canonical), fp32. Residual chunk 2048 rows; query chunk 512 rows of the flattened batch, so per-image energies depend on batch composition at the ≈ 1e-6 level, as in the sealed scorer. |
| Geometry | CPU, fp64 |
| Readout | CPU, fp32 |
| Normal and score phases | TF32 off; threads pinned (2) |

Bitwise reproducibility is guaranteed **only on the same machine and settings**. Different GPUs can change k-centre
selections: see the cross-machine limitation (§12).

## 5. Public API

```python
from nanm import FrozenPatchEncoder, G3Config, NANMSystem

enc = FrozenPatchEncoder.load('checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth',
                              sha256='73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71', device='cuda')
tok = enc.tokens(image_paths, batch=8)                 # [N, 784, 768] fp32, order preserved

sys = NANMSystem.create(G3Config(seed=2711), device='cuda')
sys.fit_task(tok_fit_task0, task_label='bracket_black', provenance={...})   # one boundary event, atomic
sys.fit_task(tok_fit_task1, task_label='bracket_brown')
pred = sys.predict(tok_query)        # task-free: {'energy': [N,28,28], 'map392': [N,392,392], 'image_score': [N]}
native = nanm.readout.to_native(pred['map392'][i], (H, W)); m224 = nanm.readout.to_224(pred['map392'])
gen = sys.save_state('ckpt_root')                      # publishes an immutable generation
sys2 = NANMSystem.load_state('ckpt_root', cfg=G3Config(seed=2711), device='cuda')   # verifies, rebuilds W and caches
sys.state.ledger()                                      # memory accounting (§9)
```

Rules:
- `fit_task` takes the whole normal FIT pool of one task. The image order is part of the contract. `k=` is accepted
  only for registered explicit-quota arms (G3@C) and is recorded as `quota_rule='explicit'`.
- There is no `partial_fit`.
- `predict` never changes state.
- Dataset roles (FIT/EVAL partition), task order and evaluation belong to the caller or harness, not to the system.

## 6. Checkpoint schema and recovery (`nanm.ckpt/1`)

```
<root>/gen_<completed:04d>_<uuid12>/state.safetensors   keys fp16 [K,768] (task-major) · offsets int64 [T+1]
                                                         · S fp64 [768,768] · n int64 [1] · selected_<t> int64
<root>/gen_.../manifest.json   schema, generation, parent, completed_tasks, contract {sha256, fields}, seed,
                               encoder identity, task_order, tasks[{label, n_fit_tokens, k, quota_rule, n_res,
                               cond_T, provenance}], ledger, provenance {nanm source sha256s, torch, CUDA, device,
                               threads, TF32}, files {state.safetensors: {sha256, bytes}}
<root>/LATEST                  name of the newest committed generation
```

**Publication.**
1. Write all files into `.tmp_<gen>/` and `fsync` each.
2. Write the manifest last.
3. `fsync` the directory.
4. `rename` it to `gen_…`. This is the commit point: atomic on one local POSIX filesystem (validated on ext4).
5. `fsync` the root.
6. Replace `LATEST` atomically.

Generations are immutable and are never overwritten.

**Loading.** Only through `LATEST` or an explicit `gen_…` name. The loader rejects:
- a missing or unreadable manifest, an unknown schema, or a generation-name mismatch;
- an unexpected file set, or a size or sha256 mismatch;
- a contract or encoder mismatch;
- invalid or non-finite tensors.

W and all inference caches are rebuilt; they are never stored.

**Recovery.**
- A crash before the rename leaves only `.tmp_*`, which is ignored; `LATEST` stays valid.
- A crash after the rename but before the `LATEST` update leaves a valid committed generation; `LATEST` still points to
  the previous one.
- A failed `fit_task` leaves the in-memory state unchanged (copy-then-commit).

## 7. Memory and runtime (measured; RTX A6000 shared at ≈ 100 % utilisation, so latencies are upper bounds)

| Bytes | MPDD G3 | VisA G3 | VisA G3@C |
|---|---:|---:|---:|
| Historical registered ledger (keys + S as fp32 + n) | 8,842,760 | 64,997,384 | 30,718,472 |
| Actual persistent tensors (keys + **S fp64** + n) | 11,202,056 | 67,356,680 | 33,077,768 |
| Serialized checkpoint (payload + manifest; includes offsets and selected indices) | 11,240,930 | 67,690,439 | 33,232,912 |
| Inference-only state (keys + W fp32; not resumable) | 8,842,752 | 64,997,376 | 30,718,464 |
| Runtime cache (keys fp32 + mW fp32 + norms) | 25,950,708 | 250,715,440 | 113,510,524 |
| Shared frozen encoder (fp32 parameters) | 346,334,208 | (same) | (same) |

The encoder is reported separately. The historical 144,806,400 B figure was a different accounting basis.

| Time | Value |
|---|---|
| Batch-1 latency, p50 (preprocess / fp32 encoder / retrieval / readout / end-to-end), MPDD bank 4,221 keys | 15.8 / 71.5 / 12.4 / 50.2 / 142 ms |
| Same, VisA bank 40,780 keys | 15.9 / 65.3 / 27.2 / 50.8 / 158 ms |
| Encoder throughput | 24.0 img/s (batch 8), 31.6 img/s (batch 32) |
| Boundary update, largest VisA task (pcb1, 426,496 tokens, k = 4,265) | prepare 0.18 s, k-centre 25.3 s, residual 1.30 s, geometry 0.11 s |
| Full `fit_task` stream | MPDD 26.1 s (6 tasks), VisA 291 s (12 tasks) |
| Checkpoint | save ≤ 0.47 s; load + rebuild 0.13 s (MPDD) / 0.40 s (VisA) |
| Peak memory (profiling run) | 3.43 GB CUDA allocated, 4.05 GB host RSS |

## 8. Evaluation protocol (BENCH-G3, sealed)

- FIT role per task: images after the first 2·⌊n/5⌋ in the salted-SHA order (`NAR-PILOT-00/roles/v1|path`). EVAL and
  CREDIT roles are not used for writing.
- Task order: alphabetical.
- Metrics: native exact Pixel AP; resized-224 Pixel AP; I-AUROC (top-1 %, map-max); 8,192-bin native AP per stage;
  Chaudhry FM; category-stratified bootstrap. The functions are in `exps/bench_g3/run.py`. They are the oracle and are
  not re-implemented in `nanm`.

## 9. Reproduction

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2      # physical GPU 2; never GPU 0
PY=/media/RAID5Array/backup_home/tindd4/anaconda3/envs/nlcad/bin/python
$PY -B -m pytest -q -p no:cacheprovider tests/nanm/           # CPU: contract, import boundary, kernels, checkpoints
$PY -B -m exps.nanm_sys.regen preflight                         # oracle custody and label-free identities
bash exps/nanm_sys/run_final_p5.sh                              # continuation parity (A, B restart chains)
bash exps/nanm_sys/run_final_p6.sh                              # encoder, score and metric parity
$PY -B -m exps.nanm_sys.final profile                           # P7 measurements
```

Inputs:
- the R1 token caches in `…/hope_cad_cache/nanm_sys_regen/` (regenerate with `exps.nanm_sys.regen extract_train`);
- the oracle reference state `normal_runA`.

## 10. Limitations

1. **Cross-machine.** The sealed RTX 4050 state is not reproduced bitwise on the RTX A6000. k-centre selection is
   sensitive to floating-point kernel differences (forensics: classification C). Metric-level differences relative to
   the sealed results are |Δ macro AP| ≤ 6.1e-4 (VisA) and ≤ 1.4e-4 (MPDD). The sealed BENCH-G3 numbers remain
   authoritative.
2. **Cohorts.** VisA and MVTec are exposed. MPDD was a one-shot clean cohort and has now been re-read for parity, so
   there is no further confirmation value from it.
3. **Memory growth.** The bank grows by 1 % per task. There is no fixed capacity or eviction.
4. **Image-level trade-off.** On MPDD, G3's I-AUROC is below Euclidean memory (0.9396 vs 0.9614).
5. **Unit accounting.** The ledger uses fp32 S historically. The actual state uses fp64 S (+2,359,296 B).
