# HOPE-CAD v1 architecture decision (revised Task 1)

**Decision: PASS.** The accepted direction remains `frozen ViT-B/8 → Self-Modifying Titans → CMS → HOPE representation → anomaly head`. No implementation is included here.

## Complete forward path

`image → frozen ViT-B/8 ImageNet-21k-compatible, 224×224, layer 9/block 8 → 784 row-major patches ×768 → Self-Modifying Titans (fast patch clock) → sequential CMS (long continual clock) → y → anomaly head → 28×28 map → 224×224 map → image score.`

HOPE output is the scoring representation. Raw backbone patches are never used as the actual prediction while HOPE merely gates updates.

## State persistence and official evaluation

All persistent `M_k,M_v,M_q,M_eta,M_alpha,M_memory`, Titans momentum/optimizer state, CMS parameters/optimizer state, CMS scheduler counters, and the Head-A reference state continue across all 15 MVTec tasks with **no task-boundary reset**. Only a new independent run/seed/replicate resets to initial state. For test image `i`, deep-clone the entire persistent state into `S_work_i`; primary Head B may run the 784 patches causally with Titans updates in the clone; keep CMS frozen for that image; score; discard the clone. Persistent state, reference state, counters, and relevant RNG remain unchanged, so test ordering cannot matter. Persistent state, reference bank, counters, and relevant RNG remain unchanged, so test ordering cannot matter.

CMS is frozen during evaluation because its v1 clock is completed normal-image events. This is a deliberate CAD adaptation that prevents test images from creating continual knowledge and preserves strict official-test isolation while still allowing the paper-faithful fast memory to produce a per-image working trajectory.

## Clocks

Titans uses patch chunks. The implementation contract must expose `memory_chunk_size` and `auxiliary_memory_chunk_size` separately. Auxiliary means applicable `M_k,M_v,M_q,M_eta,M_alpha`; v1 may set both to 16 as an explicit experimental hypothesis. CMS uses image events: level 1 updates after every 1 normal training image; level 2 after every 8. Thus updates/image are approximately 1 and 0.125; for a task with `N` good images they are `N` and `floor(N/8)`. These are **EXPERIMENTAL-HYPOTHESIS**, intentionally separated from the approximately 49 Titans chunk transitions per image.

## CMS objective boundary

Titans has the paper-grounded associative L2 objective and surprise/retention update. CMS Eq. 71 allows an objective of choice. `models/hope_cad/continuum_memory_v1.py` must expose a generic objective/update interface and cannot bake in anomaly loss, NN distance, or the Titans L2 loss. The CAD self-supervised CMS objective remains open for a later task if the papers do not justify one.

## Primary and secondary anomaly heads

**Primary Head B: HOPE-native residual/surprise.** Use the Titans memory prediction residual and/or gradient surprise per patch, then aggregate by max. This is an experimental anomaly interpretation, not a paper-defined anomaly detector. It is selected because it does not require a moving HOPE reference bank.

**Secondary / deferred Head A: HOPE-transformed patch NN.** A 2,500-patch bank may contain patches from many parent images. If HOPE depends on image context, retaining four parent sequences is insufficient, while retaining arbitrary parent contexts creates hidden replay/storage. Head A is therefore **DEFERRED / BLOCKED**, and no stale-vector solution is claimed. A future bounded stable-reference design must be specified before implementation.

Canonical `C0-CADIC` remains the existing CADIC-compatible patch memory with its configured support-neighborhood image scoring. Do not redefine it as max-patch scoring. A future matched control, `C0-NN-MAX`, uses the same frozen ViT and patch-reference principle with max patch distance; only then may `HOPE + NN-MAX` be compared to `C0-NN-MAX`. The final HOPE method may separately be compared with canonical `C0-CADIC`.

## Paper ambiguity and spatial operator

Eq. 79/83–88 show adaptive `M_q`, but nearby prose calls `q_t=x_tW_q` the only non-adaptive projection. The source is internally inconsistent. v1 chooses adaptive `M_q` as a **PAPER-INFERRED** interpretation of the explicit equations, with a fixed `W_q` base/initial projection and an implementation flag for the alternative. The paper's local convolution window 4 is used as its **PAPER-DEFINED** 1D raster operator. The row-boundary adjacency `(0,27)→(1,0)` is documented; a 2D equivalent is deferred as a labeled CAD adaptation/ablation.

## Versioned implementation contract

Future modules are named exactly:

- `models/hope_cad/self_modifying_titans_v1.py`
- `models/hope_cad/continuum_memory_v1.py`
- `models/hope_cad/hope_block_v1.py`
- `models/hope_cad/state_v1.py`

The adapter will implement `fit_task`, `score_batch`, `state_dict`, `load_state_dict`, `memory_stats`, and `method_metadata` without changing the benchmark protocol.

## Historical paths rejected

`meta_nath_core.py` scores raw backbone patches; `titans_memory.py` is a single CLS matrix with `k=v`, no self-modifying MLP memories, no paper chunked deep memory, and no HOPE composition; `acc_gating.py` is a heuristic gate rather than a learning level; `cadic_coreset.py` is an image-entry/N2B-NC historical memory rather than the isolated patch control; historical trainers/consolidation/NSP2/CBP/Phase 3 are not HOPE/CMS.

## Falsification matrix

Later, under identical seeds and protocol, compare CADIC control, full HOPE, HOPE−Self-Modifying Titans, HOPE−CMS, one CMS level, the fixed two-level schedule, Head A, and Head B. Report image AUROC, all-pixel P-AUPR, maps, memory/runtime, forgetting, and state-mutation checks. No official test metric may choose a head or frequency.
