# HOPE-CAD v1 architecture decision (revised Task 1)

**Decision: PASS.** The accepted direction remains `frozen ViT-B/8 → Self-Modifying Titans → CMS → HOPE representation → anomaly head`. The standalone SMT, generic CMS, and SMT-to-CMS HOPE wrapper are implemented; anomaly scoring remains deferred.

## Complete forward path

`image → frozen ViT-B/8 ImageNet-21k-compatible, 224×224, layer 9/block 8 → 784 row-major patches ×768 → Self-Modifying Titans (fast patch clock) → sequential CMS (long continual clock) → y → anomaly head → 28×28 map → 224×224 map → image score.`

HOPE output is the scoring representation. Raw backbone patches are never used as the actual prediction while HOPE merely gates updates.

## State persistence and official evaluation

All persistent linear `M_k,M_v,M_q,M_eta,M_alpha,M_memory` states, CMS current level buffers, detached gradient accumulators, CMS counters, and the Head-A reference state continue across all 15 MVTec tasks with **no task-boundary reset**. The canonical Eq. 90–93 SMT path has no extra momentum buffer; CMS v1 uses no optimizer object or momentum slots. Only a new independent run/seed/replicate resets to static initialization. For test image `i`, deep-clone the entire persistent state into `S_work_i`; primary Head B may run the 784 patches causally with Titans updates in the clone; keep CMS frozen for that image; score; discard the clone. Persistent state, reference state, counters, and relevant RNG remain unchanged, so test ordering cannot matter.

CMS is frozen during evaluation because its v1 clock is completed normal-image events. This is a deliberate CAD adaptation that prevents test images from creating continual knowledge and preserves strict official-test isolation while still allowing the paper-faithful fast memory to produce a per-image working trajectory.

## Clocks

Titans uses patch chunks. The implementation contract exposes `memory_chunk_size` and `auxiliary_memory_chunk_size` separately. Auxiliary means `M_k,M_v,M_eta,M_alpha` and optional `M_q`; each control is a scalar output. Each stream contains every token exactly once and flushes its shorter remainder. At coincident boundaries, all candidates are prepared from pre-boundary states before committing updates. This independent pending-stream behavior is a **PROJECT-MAPPING** for unequal clocks. CMS uses generic `update_periods`, with one completed normal image as one project-defined event. Period `p` commits after events `p,2p,...`; no event-0 update and no partial-period flush. The initial `[1,8]` mapping therefore gives approximately 1 and 0.125 updates/image.

## CMS objective boundary

Titans has the paper-grounded associative L2 objective and surprise/retention update. CMS Eq. 71 allows an objective of choice. `ContinuumMemorySystem.commit_image` accepts one objective callable per level with `(level_index, level_input, level_output, metadata)` and performs local detached gradient accumulation. It cannot bake in anomaly loss, NN distance, or the Titans L2 loss. The CAD self-supervised CMS objective remains open for a later task if the papers do not justify one.

## Primary and secondary anomaly heads

**Primary Head B: HOPE-native residual/surprise.** Use the Titans memory prediction residual and/or gradient surprise per patch, then aggregate by max. This is an experimental anomaly interpretation, not a paper-defined anomaly detector. It is selected because it does not require a moving HOPE reference bank.

**Secondary / deferred Head A: HOPE-transformed patch NN.** A 2,500-patch bank may contain patches from many parent images. If HOPE depends on image context, retaining four parent sequences is insufficient, while retaining arbitrary parent contexts creates hidden replay/storage. Head A is therefore **DEFERRED / BLOCKED**, and no stale-vector solution is claimed. A future bounded stable-reference design must be specified before implementation.

Canonical `C0-CADIC` remains the existing CADIC-compatible patch memory with its configured support-neighborhood image scoring. Do not redefine it as max-patch scoring. A future matched control, `C0-NN-MAX`, uses the same frozen ViT and patch-reference principle with max patch distance; only then may `HOPE + NN-MAX` be compared to `C0-NN-MAX`. The final HOPE method may separately be compared with canonical `C0-CADIC`.

## Paper ambiguity and spatial operator

The canonical self-modifying path uses fixed `q=W_q(x)` with `adaptive_q=False`. Eq. 79–82 show a preceding fully adaptive variant containing `M_q`; Eq. 83 writes `q_t=x_tW_q` as the only non-adaptive projection, while Eq. 85 still includes `q` in the optimized-memory set. `adaptive_q=True` remains an explicit **PAPER-INFERRED** alternative caused by that ambiguity. The paper's local convolution window 4 is used as its **PAPER-DEFINED** 1D raster operator. The row-boundary adjacency `(0,27)→(1,0)` is documented; a 2D equivalent is deferred as a labeled CAD adaptation/ablation.

Eq. 88 presents `alpha_t I-eta_t k_t k_t^T` while allowing arbitrary memory architectures and then instantiating the residual MLP in Eq. 89; Eq. 93 derives the rank-one form specifically for a linear memory. The canonical implementation therefore uses the fully derived linear path and defers residual MLP fast memories. Scalar `eta`/`alpha` outputs, sigmoid control postprocessors, same-length zero padding, Xavier initialization, and asynchronous unequal-clock handling are explicit **PROJECT-MAPPINGS** where the source is silent.

## Versioned implementation contract

Future modules are named exactly:

- `models/hope_cad/self_modifying_titans.py`
- `models/hope_cad/continuum_memory.py`
- `models/hope_cad/hope_block.py`
- `models/hope_cad/state.py`

The adapter will implement `fit_task`, `score_batch`, `state_dict`, `load_state_dict`, `memory_stats`, and `method_metadata` without changing the benchmark protocol.

## CMS core state contract

The current CMS implementation is generic over `K`, `dim`, `hidden_dim`, `update_periods`, and `learning_rates`. Every level owns static initialization parameters, current mutable buffers, same-shaped gradient/error accumulators, `pending_count`, and `update_count`; the module owns `completed_events`. `forward` is strictly read-only. `commit_image` requires `B=1`, computes all level inputs/outputs from pre-event states, validates every objective and gradient before any persistent mutation, then commits due levels in ascending index order. `commit_batch` is transport over ordered singleton events. `reset_state` copies static initialization into current buffers, clears accumulators/counters, and does not draw random values. Incomplete periods remain serialized. CMS test clones are isolated and reject mutation. These exact level geometry, optimizer, image-clock, local-detachment, no-flush, and test-freeze choices are **PROJECT-MAPPING**; no Eq.72/73/74 behavior is implemented.

## Historical paths rejected

`meta_nath_core.py` scores raw backbone patches; `titans_memory.py` is a single CLS matrix with `k=v`, no self-modifying MLP memories, no paper chunked deep memory, and no HOPE composition; `acc_gating.py` is a heuristic gate rather than a learning level; `cadic_coreset.py` is an image-entry/N2B-NC historical memory rather than the isolated patch control; historical trainers/consolidation/NSP2/CBP/Phase 3 are not HOPE/CMS.

## HOPE composition contract

`models/hope_cad/hope_block.py` is a thin composition layer over the locked child systems. Its read-only path is exactly `smt.forward(x, update=False) → memory_prediction → cms.forward`, with no raw-token, projection, residual, surprise, or anomaly bypass. A normal-image `commit_image` performs one SMT update pass and one CMS image event, passes the exact causal SMT output to CMS, and atomically restores both children from mutable-state snapshots if any operation fails. `commit_batch` processes images in order. `evaluate_image` deep-clones the full wrapper, allows SMT to evolve in the private clone, keeps CMS read-only, returns the clone's CMS output, and discards the clone. `reset_state` delegates once to each child; native nested `state_dict` continuation includes both children and wrapper schema metadata. The wrapper adds no online tensor state. These transaction and evaluation-isolation choices are **PROJECT-MAPPING** decisions made for CAD test isolation; the SMT→CMS ordering and final CMS output are the paper-grounded HOPE path.

## Falsification matrix

Later, under identical seeds and protocol, compare CADIC control, full HOPE, HOPE−Self-Modifying Titans, HOPE−CMS, one CMS level, the fixed two-level schedule, Head A, and Head B. Report image AUROC, all-pixel P-AUPR, maps, memory/runtime, forgetting, and state-mutation checks. No official test metric may choose a head or frequency.
